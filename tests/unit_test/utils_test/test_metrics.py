"""
Unit tests for structured metrics logging subsystem (Requirement R2).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Generator
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from src.utils.logging_config import JSONFormatter, request_context
from src.utils.metrics import (
    MetricsCollector,
    gauge,
    get_metrics,
    histogram,
    increment,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def _type_check_fixtures(
    _c: CaptureFixture[str] | None = None,
    _r: FixtureRequest | None = None,
    _l: LogCaptureFixture | None = None,
    _m: MonkeyPatch | None = None,
    _mock: MockerFixture | None = None,
) -> None:
    """Type checking helper to validate fixture typing."""


@pytest.fixture(autouse=True)
def setup_metrics_for_test() -> Generator[None, None, None]:
    """Ensure clean metrics singleton enabled for tests."""
    from src.utils import metrics
    from src.utils.logging_config import reset_logging_handlers

    reset_logging_handlers()
    metrics._metrics_instance = MetricsCollector(
        config={"metrics_enabled": True, "log_metrics": True}
    )
    yield
    metrics._metrics_instance = None
    reset_logging_handlers()


def test_increment_emits_structured_log_event() -> None:
    """Verify increment emits structured metric event with event_type='metric'."""
    collector = MetricsCollector(config={"metrics_enabled": True, "log_metrics": True})

    with patch.object(collector.logger.logger, "info") as mock_info:
        collector.increment("queries_total", 2, tags={"source": "cli"})
        mock_info.assert_called_once()
        _, kwargs = mock_info.call_args
        extra: dict[str, Any] = kwargs["extra"]
        assert extra["event_type"] == "metric"
        assert extra["metric_name"] == "queries_total"
        assert extra["value"] == 2
        assert extra["metric_value"] == 2
        assert extra["metric_type"] == "counter"
        assert extra["source"] == "cli"


def test_gauge_emits_structured_log_event() -> None:
    """Verify gauge emits structured metric event with event_type='metric'."""
    collector = MetricsCollector(config={"metrics_enabled": True, "log_metrics": True})

    with patch.object(collector.logger.logger, "info") as mock_info:
        collector.gauge("query_duration_ms", 123.45, tags={"model": "bge-small"})
        mock_info.assert_called_once()
        _, kwargs = mock_info.call_args
        extra: dict[str, Any] = kwargs["extra"]
        assert extra["event_type"] == "metric"
        assert extra["metric_name"] == "query_duration_ms"
        assert extra["value"] == 123.45
        assert extra["metric_type"] == "gauge"
        assert extra["model"] == "bge-small"


def test_histogram_emits_structured_log_event() -> None:
    """Verify histogram emits structured metric event with event_type='metric'."""
    collector = MetricsCollector(config={"metrics_enabled": True, "log_metrics": True})

    with patch.object(collector.logger.logger, "info") as mock_info:
        collector.histogram("results_count", 5.0, tags={"search_type": "hybrid"})
        mock_info.assert_called_once()
        _, kwargs = mock_info.call_args
        extra: dict[str, Any] = kwargs["extra"]
        assert extra["event_type"] == "metric"
        assert extra["metric_name"] == "results_count"
        assert extra["value"] == 5.0
        assert extra["metric_type"] == "histogram"
        assert extra["search_type"] == "hybrid"


def test_module_level_increment_gauge_histogram() -> None:
    """Verify module-level functions forward calls to global MetricsCollector."""
    collector = get_metrics()

    with patch.object(collector.logger.logger, "info") as mock_info:
        increment("module_counter", 1)
        assert mock_info.call_count == 1
        _, kwargs1 = mock_info.call_args
        assert kwargs1["extra"]["metric_name"] == "module_counter"

        gauge("module_gauge", 42.0)
        assert mock_info.call_count == 2
        _, kwargs2 = mock_info.call_args
        assert kwargs2["extra"]["metric_name"] == "module_gauge"

        histogram("module_histogram", 99.0)
        assert mock_info.call_count == 3
        _, kwargs3 = mock_info.call_args
        assert kwargs3["extra"]["metric_name"] == "module_histogram"


def test_metrics_disabled_emits_nothing() -> None:
    """Verify that disabled metrics do not emit any log calls."""
    collector_disabled = MetricsCollector(
        config={"metrics_enabled": False, "log_metrics": True}
    )
    with patch.object(collector_disabled.logger.logger, "info") as mock_info:
        collector_disabled.increment("queries_total", 1)
        collector_disabled.gauge("duration", 10.0)
        collector_disabled.histogram("count", 2.0)
        mock_info.assert_not_called()

    collector_log_disabled = MetricsCollector(
        config={"metrics_enabled": True, "log_metrics": False}
    )
    with patch.object(collector_log_disabled.logger.logger, "info") as mock_info:
        collector_log_disabled.increment("queries_total", 1)
        collector_log_disabled.gauge("duration", 10.0)
        collector_log_disabled.histogram("count", 2.0)
        mock_info.assert_not_called()


def test_no_in_memory_metrics_accumulation() -> None:
    """Verify MetricsCollector does not accumulate state in memory."""
    collector = MetricsCollector(config={"metrics_enabled": True, "log_metrics": True})

    # Assert no in-memory storage attribute
    assert not hasattr(collector, "metrics")

    # get_stats returns empty dict
    assert collector.get_stats() == {}

    # log_all_metrics is a safe no-op
    collector.log_all_metrics()


def test_no_background_thread_running() -> None:
    """Verify MetricsCollector does not spawn any background threads."""
    collector = MetricsCollector(config={"metrics_enabled": True, "log_metrics": True})
    assert not hasattr(collector, "thread")
    assert not hasattr(collector, "_start_metrics_thread")


def test_end_to_end_jsonl_formatting_with_request_id() -> None:
    """Verify that metric logs formatted by JSONFormatter include request_id when inside request_context."""
    formatter = JSONFormatter()
    test_logger = logging.getLogger("test_e2e_metric_logger")

    with request_context("trace-uuid-456"):
        # Create a LogRecord mimicking StructuredLogger.log_metric output
        record = test_logger.makeRecord(
            name="metrics",
            level=logging.INFO,
            fn="metrics.py",
            lno=42,
            msg="Metric: queries_total=1",
            args=(),
            exc_info=None,
            extra={
                "event_type": "metric",
                "metric_name": "queries_total",
                "value": 1,
                "metric_type": "counter",
            },
        )
        formatted = formatter.format(record)
        entry = json.loads(formatted)

        assert entry["event_type"] == "metric"
        assert entry["metric_name"] == "queries_total"
        assert entry["value"] == 1
        assert entry["metric_type"] == "counter"
        assert entry["request_id"] == "trace-uuid-456"
        assert "timestamp" in entry


def test_backward_compatibility_arguments() -> None:
    """Verify backward compatibility with positional 'name' and **kwargs tags."""
    collector = MetricsCollector(config={"metrics_enabled": True, "log_metrics": True})

    with patch.object(collector.logger.logger, "info") as mock_info:
        # Using name= keyword argument
        collector.increment(name="legacy_counter", value=3, environment="test")
        mock_info.assert_called_once()
        _, kwargs = mock_info.call_args
        extra = kwargs["extra"]
        assert extra["metric_name"] == "legacy_counter"
        assert extra["value"] == 3
        assert extra["environment"] == "test"

