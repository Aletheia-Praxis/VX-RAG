"""
Test logging configuration, RotatingFileHandler, JSON Lines formatting, and request tracing.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from src.utils.logging_config import (
    DEFAULT_BACKUP_COUNT,
    DEFAULT_MAX_BYTES,
    JSONFormatter,
    SafeRotatingFileHandler,
    StructuredLogger,
    _shared_file_handlers,
    clear_request_id,
    get_logger,
    get_request_id,
    load_logging_config,
    request_context,
    reset_logging_handlers,
    set_request_id,
    setup_logging,
    with_request_id,
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


def test_load_logging_config() -> None:
    """Test loading logging config from settings.yaml."""
    config = load_logging_config()
    assert isinstance(config, dict)
    assert "log_level" in config


def test_structured_logger_creation() -> None:
    """Test StructuredLogger creation."""
    config: dict[str, Any] = {"log_level": "INFO", "log_format": "json"}
    logger = StructuredLogger("test_logger", config)
    assert logger.name == "test_logger"
    assert logger.config == config


def test_logger_rotating_file_handler(tmp_path: Path) -> None:
    """Test that file handler creates RotatingFileHandler with single jsonl file."""
    log_dir = tmp_path / "logs"
    log_file = log_dir / "vx_rag.jsonl"
    config: dict[str, Any] = {
        "log_level": "INFO",
        "log_format": "json",
        "log_file": str(log_file),
        "max_bytes": DEFAULT_MAX_BYTES,
        "backup_count": DEFAULT_BACKUP_COUNT,
    }

    logger = StructuredLogger("test_rotating_logger", config)

    # Trigger logging to create file
    logger.info("Test message for rotating handler")

    # Verify directory and exact file name
    assert log_dir.exists()
    assert log_file.exists()

    # Find the RotatingFileHandler among handlers
    file_handlers = [
        h for h in logger.logger.handlers if isinstance(h, RotatingFileHandler)
    ]
    assert len(file_handlers) == 1
    handler = file_handlers[0]
    assert handler.maxBytes == DEFAULT_MAX_BYTES
    assert handler.backupCount == DEFAULT_BACKUP_COUNT

    # Check file content is valid JSON Lines
    content = log_file.read_text(encoding="utf-8").strip()
    data = json.loads(content)
    assert data["message"] == "Test message for rotating handler"
    assert "timestamp" in data
    assert data["timestamp"].endswith("Z")


def test_setup_logging_function(tmp_path: Path) -> None:
    """Test that setup_logging initializes central rotating file handler."""
    log_file = tmp_path / "setup_test.jsonl"
    setup_logging(
        log_file=str(log_file),
        max_bytes=1024,
        backup_count=3,
        log_level="DEBUG",
        log_format="json",
    )

    logger = get_logger("test_setup_module")
    logger.info("Message through setup_logging")

    assert log_file.exists()
    lines = [
        line.strip()
        for line in log_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) >= 1
    parsed = json.loads(lines[-1])
    assert parsed["message"] == "Message through setup_logging"


def test_log_rotation_rollover(tmp_path: Path) -> None:
    """Test that log file rotates into backup files when size exceeds max_bytes."""
    log_file = tmp_path / "rollover.jsonl"
    small_max_bytes = 150  # Very small threshold to easily trigger rotation

    config: dict[str, Any] = {
        "log_level": "INFO",
        "log_format": "json",
        "log_file": str(log_file),
        "max_bytes": small_max_bytes,
        "backup_count": 3,
    }

    logger = StructuredLogger("test_rollover_logger", config)

    # Write multiple lines exceeding 150 bytes
    for i in range(10):
        logger.info(f"Rollover message iteration payload #{i}")

    # Check that backup file was created
    backup_file = tmp_path / "rollover.jsonl.1"
    assert log_file.exists()
    assert backup_file.exists()

    # Validate that all lines in primary and backup files are valid JSON
    for fpath in (log_file, backup_file):
        for line in fpath.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                assert "timestamp" in record
                assert "message" in record


def test_shared_handler_cache_avoids_duplicate_file_descriptors(tmp_path: Path) -> None:
    """Test that multiple loggers referencing the same file share the handler."""
    log_file = str(tmp_path / "shared.jsonl")
    config: dict[str, Any] = {
        "log_level": "INFO",
        "log_format": "json",
        "log_file": log_file,
    }

    logger1 = StructuredLogger("module_a", config)
    logger2 = StructuredLogger("module_b", config)

    handler1 = next(h for h in logger1.logger.handlers if isinstance(h, RotatingFileHandler))
    handler2 = next(h for h in logger2.logger.handlers if isinstance(h, RotatingFileHandler))

    assert handler1 is handler2


def test_get_logger_singleton() -> None:
    """Test that get_logger returns cached instance for the same name."""
    logger1 = get_logger("singleton_test")
    logger2 = get_logger("singleton_test")
    assert logger1 is logger2


def test_log_event() -> None:
    """Test logging structured events."""
    config: dict[str, Any] = {"log_level": "INFO", "log_structured_events": True}
    logger = StructuredLogger("test_event_logger", config)

    with patch.object(logger.logger, "info") as mock_info:
        logger.log_event("test_event", key="value")
        mock_info.assert_called_once()
        _, kwargs = mock_info.call_args
        assert "event_type" in kwargs["extra"]
        assert kwargs["extra"]["event_type"] == "test_event"
        assert kwargs["extra"]["key"] == "value"


def test_log_metric() -> None:
    """Test logging metrics with structured event fields."""
    config: dict[str, Any] = {"log_level": "INFO", "log_metrics": True}
    logger = StructuredLogger("test_metric_logger", config)

    with patch.object(logger.logger, "info") as mock_info:
        logger.log_metric("test_metric", 42, metric_type="counter", unit="ms")
        mock_info.assert_called_once()
        _, kwargs = mock_info.call_args
        extra = kwargs["extra"]
        assert extra["event_type"] == "metric"
        assert extra["metric_name"] == "test_metric"
        assert extra["value"] == 42
        assert extra["metric_value"] == 42
        assert extra["metric_type"] == "counter"
        assert extra["unit"] == "ms"


def test_request_context_injection() -> None:
    """Test that request_context sets and restores request_id and injects into log entry."""
    formatter = JSONFormatter()
    logger = logging.getLogger("test_context_inject")

    record = logger.makeRecord(
        name="test_context_inject",
        level=logging.INFO,
        fn="",
        lno=0,
        msg="Inside context",
        args=(),
        exc_info=None,
    )

    with request_context("req-abc-123") as rid:
        assert rid == "req-abc-123"
        assert get_request_id() == "req-abc-123"
        formatted_json = formatter.format(record)
        data = json.loads(formatted_json)
        assert data["request_id"] == "req-abc-123"

    # Outside context, request_id should be None
    assert get_request_id() is None
    record_outside = logger.makeRecord(
        name="test_context_inject",
        level=logging.INFO,
        fn="",
        lno=0,
        msg="Outside context",
        args=(),
        exc_info=None,
    )
    formatted_outside = formatter.format(record_outside)
    data_outside = json.loads(formatted_outside)
    assert "request_id" not in data_outside


def test_with_request_id_decorator_sync_and_async() -> None:
    """Test that with_request_id decorator activates request_id for sync and async callables."""
    @with_request_id
    def sync_op() -> str | None:
        return get_request_id()

    @with_request_id
    async def async_op() -> str | None:
        await asyncio.sleep(0.001)
        return get_request_id()

    sync_rid = sync_op()
    assert sync_rid is not None
    assert len(sync_rid) > 0

    async_rid = asyncio.run(async_op())
    assert async_rid is not None
    assert len(async_rid) > 0
    assert sync_rid != async_rid


def test_set_and_clear_request_id() -> None:
    """Test manual set_request_id and clear_request_id."""
    rid = set_request_id("custom-req-id")
    assert rid == "custom-req-id"
    assert get_request_id() == "custom-req-id"

    clear_request_id()
    assert get_request_id() is None


def test_explicit_record_request_id() -> None:
    """Test that an explicitly passed request_id on the record is preserved."""
    formatter = JSONFormatter()
    logger = logging.getLogger("test_explicit_req")

    record = logger.makeRecord(
        name="test_explicit_req",
        level=logging.INFO,
        fn="",
        lno=0,
        msg="Explicit request ID",
        args=(),
        exc_info=None,
        extra={"request_id": "explicit-id-789"},
    )

    formatted = formatter.format(record)
    data = json.loads(formatted)
    assert data["request_id"] == "explicit-id-789"


def test_reset_logging_handlers() -> None:
    """Test that reset_logging_handlers cleans up shared handlers and context."""
    set_request_id("teardown-test")
    reset_logging_handlers()
    assert get_request_id() is None


def test_sensitive_threat_intel_data_preserved() -> None:
    """Verify cybersecurity indicators (IPs, hashes, emails) are preserved verbatim (Requirement R4)."""
    formatter = JSONFormatter()
    logger = logging.getLogger("test_threat_intel")

    threat_message = "Detected suspicious traffic to 198.51.100.42 from user admin@threat-actor.org"
    extra_payload: dict[str, Any] = {
        "ip_address": "198.51.100.42",
        "file_hash_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "email": "admin@threat-actor.org",
        "domain": "malicious-c2.test",
    }

    record = logger.makeRecord(
        name="test_threat_intel",
        level=logging.INFO,
        fn="",
        lno=0,
        msg=threat_message,
        args=(),
        exc_info=None,
        extra=extra_payload,
    )

    formatted = formatter.format(record)
    entry = json.loads(formatted)

    assert entry["message"] == threat_message
    assert entry["ip_address"] == "198.51.100.42"
    assert (
        entry["file_hash_sha256"]
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )
    assert entry["email"] == "admin@threat-actor.org"
    assert entry["domain"] == "malicious-c2.test"


def test_concurrent_get_logger_race_safety(tmp_path: Path) -> None:
    """Verify concurrent get_logger calls from multiple threads safely share a single handler."""
    log_file = tmp_path / "concurrent_race.jsonl"
    reset_logging_handlers()
    setup_logging(
        log_file=str(log_file),
        max_bytes=1024,
        backup_count=3,
        log_level="DEBUG",
        log_format="json",
    )
    num_threads = 10
    messages_per_thread = 50

    def worker(tid: int) -> int:
        logger = get_logger(f"race_worker_{tid}")
        for idx in range(messages_per_thread):
            logger.debug("concurrent race test message", tid=tid, idx=idx)
        return messages_per_thread

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, tid) for tid in range(num_threads)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert sum(results) == num_threads * messages_per_thread
    resolved_key = os.path.normcase(os.path.abspath(str(log_file)))
    assert resolved_key in _shared_file_handlers
    reset_logging_handlers()


def test_structured_logger_extra_and_kwargs_flattening(tmp_path: Path) -> None:
    """Verify StructuredLogger flattens both extra={...} and **kwargs to top-level JSON fields."""
    log_file = tmp_path / "flattening_test.jsonl"
    setup_logging(log_file=str(log_file), log_format="json", log_level="DEBUG")
    logger = get_logger("test_flattening")

    # 1. Standard kwargs style
    logger.info("kwargs call", user="alice", action="login")

    # 2. Standard library extra dict style
    logger.info("extra dict call", extra={"user": "bob", "action": "logout"})

    # 3. Hybrid style (both extra and kwargs)
    logger.warning("hybrid call", extra={"user": "carol"}, role="admin")

    # 4. Debug with extra dict
    logger.debug("debug extra call", extra={"debug_code": 99})

    # 5. Error with exc_info and extra dict
    try:
        raise ValueError("synthetic failure")
    except ValueError as exc:
        logger.error(
            "error with exc_info", exc_info=exc, extra={"error_category": "val_err"}
        )

    # 6. Non-dict extra fallback
    logger.info("non-dict extra call", extra="literal_tag")

    # 7. None extra
    logger.info("none extra call", extra=None)

    # 8. Event logging with extra dict
    logger.log_event("custom_event", extra={"event_tag": "evt_1"}, extra_key="extra_val")

    reset_logging_handlers()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 8

    # Line 1: kwargs
    r1 = json.loads(lines[0])
    assert r1["user"] == "alice" and r1["action"] == "login" and "extra" not in r1

    # Line 2: extra dict
    r2 = json.loads(lines[1])
    assert r2["user"] == "bob" and r2["action"] == "logout" and "extra" not in r2

    # Line 3: hybrid
    r3 = json.loads(lines[2])
    assert r3["user"] == "carol" and r3["role"] == "admin" and "extra" not in r3

    # Line 4: debug
    r4 = json.loads(lines[3])
    assert r4["debug_code"] == 99 and "extra" not in r4

    # Line 5: error with exc_info
    r5 = json.loads(lines[4])
    assert r5["error_category"] == "val_err" and "extra" not in r5
    assert "synthetic failure" in r5["exception"]

    # Line 6: non-dict extra preserved under "extra"
    r6 = json.loads(lines[5])
    assert r6["extra"] == "literal_tag"

    # Line 7: None extra omitted
    r7 = json.loads(lines[6])
    assert "extra" not in r7

    # Line 8: log_event with extra dict and kwargs
    r8 = json.loads(lines[7])
    assert r8["event_type"] == "custom_event"
    assert r8["event_tag"] == "evt_1"
    assert r8["extra_key"] == "extra_val"
    assert "extra" not in r8


def test_safe_rotating_file_handler_stream_recovery(tmp_path: Path) -> None:
    """
    Test that SafeRotatingFileHandler recovers its stream when rollover encounters an error.

    Asserts that handler.stream is never left as None after doRollover(), allowing subsequent
    records to be emitted without crashing.
    """
    log_file = tmp_path / "stream_recovery.jsonl"
    handler = SafeRotatingFileHandler(
        log_file,
        maxBytes=200,
        backupCount=3,
        encoding="utf-8",
    )
    formatter = JSONFormatter()
    handler.setFormatter(formatter)
    logger = logging.getLogger("test_stream_recovery")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # Write initial record to open the stream
    record = logger.makeRecord(
        "test_stream_recovery", logging.INFO, "", 0, "Initial message", (), None
    )
    handler.emit(record)
    assert handler.stream is not None
    assert not handler.stream.closed

    # Simulate an error during rotate by temporarily patching rotator to raise OSError
    def failing_rotator(source: str, dest: str) -> None:
        raise OSError(
            None,
            "The process cannot access the file because it is being used by another process",
            None,
            32,
        )

    handler.rotator = failing_rotator

    # Attempt doRollover - stream must be restored and not None in finally block
    handler.doRollover()
    assert handler.stream is not None
    assert not handler.stream.closed

    # Emit another record to confirm subsequent writes succeed without AttributeError
    record2 = logger.makeRecord(
        "test_stream_recovery", logging.INFO, "", 0, "Post-recovery message", (), None
    )
    handler.emit(record2)
    assert handler.stream is not None

    handler.close()
    logger.removeHandler(handler)


def test_safe_rotating_file_handler_safe_remove_retry(tmp_path: Path) -> None:
    """
    Test that SafeRotatingFileHandler handles safe file removal of backup files.

    Asserts that doRollover removes destination backup files cleanly with retry resilience.
    """
    log_file = tmp_path / "safe_remove.jsonl"
    handler = SafeRotatingFileHandler(
        log_file,
        maxBytes=100,
        backupCount=2,
        encoding="utf-8",
    )
    logger = logging.getLogger("test_safe_remove")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    # Pre-create a destination backup file (.1)
    backup_1 = tmp_path / "safe_remove.jsonl.1"
    backup_1.write_text("pre-existing backup content", encoding="utf-8")
    assert backup_1.exists()

    # Write enough logs to trigger rollover
    record1 = logger.makeRecord(
        "test_safe_remove", logging.INFO, "", 0, "A" * 150, (), None
    )
    handler.emit(record1)
    record2 = logger.makeRecord(
        "test_safe_remove", logging.INFO, "", 0, "B" * 150, (), None
    )
    handler.emit(record2)

    assert handler.stream is not None
    assert not handler.stream.closed
    assert log_file.exists()
    assert backup_1.exists()

    handler.close()
    logger.removeHandler(handler)


def test_multi_trial_concurrent_rollover_zero_winerror(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """
    Test multi-trial concurrency with handleError interception asserting 0 WinError 32 or 5.

    Validates that across 10 trials with 8 concurrent threads dynamically requesting loggers
    with cold cache during active rollover, zero Windows file locking exceptions occur.
    """
    import sys
    import traceback

    captured_errors: list[tuple[logging.LogRecord, BaseException, str]] = []
    old_handle_error = logging.Handler.handleError

    def intercept_handle_error(
        self: logging.Handler, record: logging.LogRecord
    ) -> None:
        _, exc_val, _ = sys.exc_info()
        if exc_val is not None:
            winerror = getattr(exc_val, "winerror", None)
            err_msg = str(exc_val)
            if (
                winerror in (32, 5)
                or "used by another process" in err_msg
                or "Access is denied" in err_msg
            ):
                captured_errors.append((record, exc_val, traceback.format_exc()))
        old_handle_error(self, record)

    monkeypatch.setattr(logging.Handler, "handleError", intercept_handle_error)

    num_trials = 10
    num_threads = 8
    messages_per_thread = 50

    try:
        for trial in range(num_trials):
            trial_dir = tmp_path / f"trial_{trial}"
            trial_dir.mkdir(parents=True, exist_ok=True)
            log_file = trial_dir / "concurrent_rollover.jsonl"

            reset_logging_handlers()
            setup_logging(
                log_file=str(log_file),
                max_bytes=800,
                backup_count=3,
                log_level="DEBUG",
                log_format="json",
            )

            def worker(tid: int, current_trial: int = trial) -> None:
                worker_logger = get_logger(f"worker_{tid}_{current_trial}")
                for idx in range(messages_per_thread):
                    worker_logger.debug(
                        "concurrent multi-trial stress message",
                        tid=tid,
                        idx=idx,
                        trial=current_trial,
                        extra={"worker_tag": f"tag_{tid}"},
                    )

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
                list(pool.map(worker, range(num_threads)))

            reset_logging_handlers()

            # Verify that log files were created and contain valid JSON Lines
            jsonl_files = list(trial_dir.glob("*.jsonl*"))
            assert len(jsonl_files) > 0
            for jf in jsonl_files:
                for line in jf.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        parsed = json.loads(line)
                        assert "timestamp" in parsed
                        assert "message" in parsed

        first_traceback = captured_errors[0][2] if captured_errors else ""
        assert (
            len(captured_errors) == 0
        ), f"Captured {len(captured_errors)} WinError exceptions: {first_traceback}"
    finally:
        reset_logging_handlers()


