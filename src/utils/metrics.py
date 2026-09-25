"""
Metrics collection and structured logging for VX-RAG project.

Provides zero-overhead structured metric emission directly to local JSON logs
without in-memory accumulation or external Prometheus scrapers (Requirement R2).
"""

from __future__ import annotations

import threading
from typing import Any

from .logging_config import StructuredLogger, get_logger


class MetricsCollector:
    """Lightweight metrics emitter for VX-RAG emitting structured JSON log lines."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the metrics collector.

        Args:
            config: Optional configuration dictionary containing metrics settings.
        """
        self.config: dict[str, Any] = config or {}
        if config is not None:
            self.logger: StructuredLogger = StructuredLogger("metrics", self.config)
        else:
            self.logger = get_logger("metrics")
        self._enabled: bool = bool(
            self.config.get("metrics_enabled", True)
            and self.config.get("log_metrics", True)
        )

    def increment(
        self,
        metric_name: str = "",
        value: int = 1,
        tags: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Emit a counter metric structured log event.

        Args:
            metric_name: Name of the counter metric.
            value: Integer value to increment by (default 1).
            tags: Optional key-value metadata tags.
            name: Alternative alias for metric_name for backward compatibility.
            **kwargs: Additional key-value tags.
        """
        if not self._enabled:
            return
        actual_name = name if name is not None else metric_name
        self.logger.log_metric(
            metric_name=actual_name,
            value=value,
            metric_type="counter",
            tags=tags,
            **kwargs,
        )

    def gauge(
        self,
        metric_name: str = "",
        value: float = 0.0,
        tags: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Emit a gauge metric structured log event.

        Args:
            metric_name: Name of the gauge metric.
            value: Current float value of the gauge.
            tags: Optional key-value metadata tags.
            name: Alternative alias for metric_name for backward compatibility.
            **kwargs: Additional key-value tags.
        """
        if not self._enabled:
            return
        actual_name = name if name is not None else metric_name
        self.logger.log_metric(
            metric_name=actual_name,
            value=value,
            metric_type="gauge",
            tags=tags,
            **kwargs,
        )

    def histogram(
        self,
        metric_name: str = "",
        value: float = 0.0,
        tags: dict[str, Any] | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Emit a histogram metric structured log event.

        Args:
            metric_name: Name of the histogram metric.
            value: Sampled float observation value.
            tags: Optional key-value metadata tags.
            name: Alternative alias for metric_name for backward compatibility.
            **kwargs: Additional key-value tags.
        """
        if not self._enabled:
            return
        actual_name = name if name is not None else metric_name
        self.logger.log_metric(
            metric_name=actual_name,
            value=value,
            metric_type="histogram",
            tags=tags,
            **kwargs,
        )

    def log_all_metrics(self) -> None:
        """No-op for backward compatibility: metrics are emitted immediately as structured logs."""

    def get_stats(self) -> dict[str, Any]:
        """
        Return metrics summary dictionary.

        Returns an empty dictionary as metrics are directly streamed to structured logs
        rather than accumulated in memory.

        Returns:
            Empty dictionary.
        """
        return {}


# Global instance and thread-safe creation lock (B-19)
_metrics_instance: MetricsCollector | None = None
_metrics_creation_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """
    Return the global MetricsCollector singleton.

    Uses double-checked locking so concurrent callers from different threads
    never construct more than one instance.

    Returns:
        The global MetricsCollector instance.
    """
    global _metrics_instance
    if _metrics_instance is None:
        with _metrics_creation_lock:
            if _metrics_instance is None:
                from .logging_config import load_logging_config

                config = load_logging_config()
                _metrics_instance = MetricsCollector(config)
    return _metrics_instance


def increment(
    metric_name: str = "",
    value: int = 1,
    tags: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Emit a counter metric structured log event via global MetricsCollector.

    Args:
        metric_name: Name of the counter metric.
        value: Integer value to increment by (default 1).
        tags: Optional key-value metadata tags.
        name: Alternative alias for metric_name.
        **kwargs: Additional key-value tags.
    """
    get_metrics().increment(
        metric_name=metric_name,
        value=value,
        tags=tags,
        name=name,
        **kwargs,
    )


def gauge(
    metric_name: str = "",
    value: float = 0.0,
    tags: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Emit a gauge metric structured log event via global MetricsCollector.

    Args:
        metric_name: Name of the gauge metric.
        value: Current float value of the gauge.
        tags: Optional key-value metadata tags.
        name: Alternative alias for metric_name.
        **kwargs: Additional key-value tags.
    """
    get_metrics().gauge(
        metric_name=metric_name,
        value=value,
        tags=tags,
        name=name,
        **kwargs,
    )


def histogram(
    metric_name: str = "",
    value: float = 0.0,
    tags: dict[str, Any] | None = None,
    name: str | None = None,
    **kwargs: Any,
) -> None:
    """
    Emit a histogram metric structured log event via global MetricsCollector.

    Args:
        metric_name: Name of the histogram metric.
        value: Sampled float observation value.
        tags: Optional key-value metadata tags.
        name: Alternative alias for metric_name.
        **kwargs: Additional key-value tags.
    """
    get_metrics().histogram(
        metric_name=metric_name,
        value=value,
        tags=tags,
        name=name,
        **kwargs,
    )
