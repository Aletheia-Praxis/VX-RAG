"""
Logging configuration for VX-RAG project.

Provides JSON-formatted logging with RotatingFileHandler, structured events,
and request tracing via contextvars.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import os
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, TypeVar, cast

import yaml

# Configuration Constants
DEFAULT_LOG_FILE: str = "logs/vx_rag.jsonl"
DEFAULT_MAX_BYTES: int = 10 * 1024 * 1024  # 10 MB (10485760 bytes)
DEFAULT_BACKUP_COUNT: int = 5
DEFAULT_LOG_LEVEL: str = "INFO"
DEFAULT_LOG_FORMAT: str = "json"
DEFAULT_ENCODING: str = "utf-8"

# Context Variable for Request / Trace ID (R3 Integration)
current_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_request_id", default=None
)

F = TypeVar("F", bound=Callable[..., Any])


def get_request_id() -> str | None:
    """
    Retrieve the current request ID from contextvars.

    Returns:
        The active request ID string, or None if not set.
    """
    return current_request_id.get()


def set_request_id(request_id: str | None = None) -> str:
    """
    Set the request ID in the current execution context.

    If no request_id is provided, a new UUID4 string is generated.

    Args:
        request_id: Optional request ID string. If None, generates a UUID4.

    Returns:
        The assigned request ID string.
    """
    rid = request_id if request_id is not None else str(uuid.uuid4())
    current_request_id.set(rid)
    return rid


def clear_request_id() -> None:
    """Clear the active request ID from the current execution context."""
    current_request_id.set(None)


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """
    Context manager setting request_id for a block and restoring prior state on exit.

    If no request_id is provided, reuses the existing active request ID or generates
    a new UUID4 if none is currently active.

    Args:
        request_id: Optional request ID string.

    Yields:
        The assigned request ID string.
    """
    existing = get_request_id()
    rid = request_id or existing or str(uuid.uuid4())
    token = current_request_id.set(rid)
    try:
        yield rid
    finally:
        current_request_id.reset(token)


def with_request_id(fn: F) -> F:
    """
    Decorator ensuring a unique request_id is active for function execution.

    Supports both synchronous functions and coroutines while preserving typing contracts.

    Args:
        fn: Synchronous or asynchronous callable.

    Returns:
        Wrapped callable executing inside a request_context.
    """
    if inspect.iscoroutinefunction(fn):

        @functools.wraps(fn)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            with request_context():
                return await fn(*args, **kwargs)

        return cast(F, async_wrapper)

    @functools.wraps(fn)
    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        with request_context():
            return fn(*args, **kwargs)

    return cast(F, sync_wrapper)


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as a single JSON Lines string.

        Args:
            record: The logging record to format.

        Returns:
            Formatted JSON string representing the log record.
        """
        created_dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        iso_timestamp = created_dt.isoformat()
        if iso_timestamp.endswith("+00:00"):
            iso_timestamp = iso_timestamp[:-6] + "Z"

        # Create base log entry
        log_entry: dict[str, Any] = {
            "timestamp": iso_timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Resolve request_id: check record extra attribute first, fallback to contextvars
        request_id = getattr(record, "request_id", None) or get_request_id()
        if request_id is not None:
            log_entry["request_id"] = request_id

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record — only string-keyed extras not already captured
        _STANDARD_RECORD_KEYS = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "taskName",  # added in Python 3.12
            "request_id",
        }
        if hasattr(record, "__dict__"):
            for key, value in record.__dict__.items():
                if key not in _STANDARD_RECORD_KEYS and key not in log_entry:
                    # Guard: some extra values may not be JSON-serialisable (B-17)
                    try:
                        json.dumps(value)
                        log_entry[key] = value
                    except (TypeError, ValueError):
                        log_entry[key] = str(value)

        try:
            return json.dumps(log_entry, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            # Last-resort fallback: emit plain text so the log record is never lost
            log_entry_safe = {k: str(v) for k, v in log_entry.items()}
            log_entry_safe["_serialization_error"] = str(exc)
            return json.dumps(log_entry_safe, ensure_ascii=False)


# Shared Handler and Logger Caches
_logging_lock: threading.RLock = threading.RLock()
_shared_file_handlers: dict[str, SafeRotatingFileHandler] = {}
_logger_cache: dict[str, StructuredLogger] = {}
_logger_instance: StructuredLogger | None = None
_global_logging_overrides: dict[str, Any] = {}


def _safe_windows_rotator(source: str, dest: str) -> None:
    """
    Safely rotate log files on Windows using atomic replace and retry resilience.

    Args:
        source: Source file path to rotate from.
        dest: Destination rotated backup file path.
    """
    if not os.path.exists(source):
        return
    max_retries = 20
    retry_delay = 0.01  # 10 ms
    for attempt in range(max_retries):
        try:
            os.replace(source, dest)
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if (
                winerror in (32, 5) or isinstance(exc, PermissionError)
            ) and attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            if attempt == max_retries - 1:
                raise
            time.sleep(retry_delay)


def _safe_remove(
    target_path: str, max_retries: int = 20, retry_delay: float = 0.01
) -> None:
    """
    Safely remove a file with retry resilience for transient Windows sharing violations.

    Args:
        target_path: Path to the file to remove.
        max_retries: Maximum number of removal attempts.
        retry_delay: Delay in seconds between attempts.
    """
    if not os.path.exists(target_path):
        return
    for attempt in range(max_retries):
        try:
            os.remove(target_path)
            return
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if (
                winerror in (32, 5) or isinstance(exc, PermissionError)
            ) and attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            if not os.path.exists(target_path):
                return
            if attempt == max_retries - 1 and (
                winerror in (32, 5) or isinstance(exc, PermissionError)
            ):
                return
            raise


class SafeRotatingFileHandler(RotatingFileHandler):
    """
    RotatingFileHandler subclass resilient to Windows file locking and sharing violations.

    Overrides doRollover to safely remove backup files with retries, use safe atomic rotation,
    and guarantee that self.stream is restored even if an exception occurs during rotation.
    """

    def __init__(
        self,
        filename: str | os.PathLike[str],
        mode: str = "a",
        maxBytes: int = 0,
        backupCount: int = 0,
        encoding: str | None = None,
        delay: bool = False,
        errors: str | None = None,
    ) -> None:
        """
        Initialize SafeRotatingFileHandler with default Windows-safe rotator.

        Args:
            filename: Log file path.
            mode: File open mode.
            maxBytes: Maximum bytes before rollover.
            backupCount: Number of backup log files to retain.
            encoding: Text encoding.
            delay: Whether to delay opening the file until first log write.
            errors: Text encoding error handling scheme.
        """
        super().__init__(
            filename,
            mode=mode,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
            errors=errors,
        )
        self.rotator = _safe_windows_rotator

    def doRollover(self) -> None:
        """
        Perform log rollover safely on Windows.

        Ensures safe removal of destination backup files with retries, uses safe rotation,
        and guarantees self.stream is never left as None even if transient file locking occurs.
        """
        if self.stream:
            self.stream.close()
            self.stream = None

        try:
            if self.backupCount > 0:
                for i in range(self.backupCount - 1, 0, -1):
                    sfn = self.rotation_filename(f"{self.baseFilename}.{i}")
                    dfn = self.rotation_filename(f"{self.baseFilename}.{i + 1}")
                    if os.path.exists(sfn):
                        if os.path.exists(dfn):
                            _safe_remove(dfn)
                        self.rotate(sfn, dfn)
                dfn = self.rotation_filename(f"{self.baseFilename}.1")
                if os.path.exists(dfn):
                    _safe_remove(dfn)
                self.rotate(self.baseFilename, dfn)
        except OSError as exc:
            winerror = getattr(exc, "winerror", None)
            if winerror not in (32, 5) and not isinstance(exc, PermissionError):
                raise
        finally:
            if self.stream is None:
                self.stream = self._open()


def _get_or_create_rotating_handler(
    log_file: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    formatter: logging.Formatter | None = None,
) -> SafeRotatingFileHandler:
    """
    Retrieve or create a cached SafeRotatingFileHandler for the specified path in a thread-safe manner.

    Args:
        log_file: Path to the log file.
        max_bytes: Maximum log file size before rotation.
        backup_count: Number of backup rotated log files to maintain.
        formatter: Optional formatter to apply to newly created handlers.

    Returns:
        Shared SafeRotatingFileHandler instance for the canonical resolved path.
    """
    with _logging_lock:
        log_path = Path(log_file)
        if log_path.parent not in (Path(""), Path(".")):
            log_path.parent.mkdir(parents=True, exist_ok=True)

        resolved_key = os.path.normcase(os.path.abspath(str(log_path)))
        if resolved_key not in _shared_file_handlers:
            handler = SafeRotatingFileHandler(
                log_path,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding=DEFAULT_ENCODING,
            )
            if formatter is not None:
                handler.setFormatter(formatter)
            _shared_file_handlers[resolved_key] = handler
        elif formatter is not None:
            _shared_file_handlers[resolved_key].setFormatter(formatter)
        return _shared_file_handlers[resolved_key]


def reset_logging_handlers() -> None:
    """Close and clear all cached handlers and loggers (for testing teardown)."""
    global _logger_instance
    with _logging_lock:
        for logger in list(_logger_cache.values()):
            for handler in logger.logger.handlers[:]:
                logger.logger.removeHandler(handler)
        for handler in list(_shared_file_handlers.values()):
            try:
                handler.close()
            except OSError:
                pass
        _shared_file_handlers.clear()
        _logger_cache.clear()
        _logger_instance = None
        _global_logging_overrides.clear()
        clear_request_id()


class StructuredLogger:
    """Structured logger with event logging capabilities."""

    def __init__(self, name: str, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the structured logger.

        Args:
            name: Logger name.
            config: Optional configuration dictionary.
        """
        self.name = name
        self.config: dict[str, Any] = config or {}
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Setup logger with appropriate formatters and handlers."""
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Set level
        level_str = self.config.get("log_level", DEFAULT_LOG_LEVEL)
        level = getattr(logging, str(level_str).upper(), logging.INFO)
        self.logger.setLevel(level)

        # Create formatters
        formatter: logging.Formatter
        if self.config.get("log_format", DEFAULT_LOG_FORMAT) == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        log_file = self.config.get("log_file")
        if log_file:
            max_bytes = int(self.config.get("max_bytes", DEFAULT_MAX_BYTES))
            backup_count = int(self.config.get("backup_count", DEFAULT_BACKUP_COUNT))
            file_handler = _get_or_create_rotating_handler(
                log_file=log_file,
                max_bytes=max_bytes,
                backup_count=backup_count,
                formatter=formatter,
            )
            if file_handler not in self.logger.handlers:
                self.logger.addHandler(file_handler)

        # Prevent duplicate logs
        self.logger.propagate = False

    def _prepare_extra(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        """
        Extract and merge explicit 'extra' dictionary with additional keyword arguments.

        Ensures that both standard library extra={...} and structured keyword arguments
        flatten into top-level JSON fields without nesting under an 'extra' key.

        Args:
            kwargs: Dictionary of keyword arguments passed to the logging method.

        Returns:
            Merged extra payload dictionary, or None if no extra fields were provided.
        """
        extra_payload: dict[str, Any] = {}
        if "extra" in kwargs:
            passed_extra = kwargs.pop("extra")
            if isinstance(passed_extra, dict):
                extra_payload.update(passed_extra)
            elif passed_extra is not None:
                extra_payload["extra"] = passed_extra
        extra_payload.update(kwargs)
        return extra_payload if extra_payload else None

    def log_event(self, event_type: str, **kwargs: Any) -> None:
        """
        Log a structured event.

        Args:
            event_type: Identifier of the event type.
            **kwargs: Extra attributes for the event payload.
        """
        if not self.config.get("log_structured_events", True):
            return

        combined_kwargs = dict(kwargs)
        extra = self._prepare_extra(combined_kwargs) or {}
        extra["event_type"] = event_type
        self.logger.info(
            f"Event: {event_type}", extra=extra
        )

    def log_metric(
        self,
        metric_name: str,
        value: Any,
        metric_type: str = "metric",
        tags: dict[str, Any] | None = None,
        **extra_tags: Any,
    ) -> None:
        """
        Log a structured metric event.

        Emits event_type="metric", metric_name, value, metric_type, and any tags.
        Preserves metric_value for backward compatibility.

        Args:
            metric_name: Identifier for the metric.
            value: Value of the metric.
            metric_type: Type classification of the metric (counter, gauge, histogram).
            tags: Optional dictionary of metadata tags.
            **extra_tags: Additional metadata tags passed as keyword arguments.
        """
        if not self.config.get("log_metrics", True):
            return

        combined_tags: dict[str, Any] = {}
        if tags:
            combined_tags.update(tags)
        if extra_tags:
            combined_tags.update(extra_tags)

        extra_payload: dict[str, Any] = {
            "event_type": "metric",
            "metric_name": metric_name,
            "value": value,
            "metric_value": value,
            "metric_type": metric_type,
            **combined_tags,
        }

        self.logger.info(
            f"Metric: {metric_name}={value}",
            extra=extra_payload,
        )

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log info message.

        Args:
            message: Message text.
            **kwargs: Additional parameters passed to logger.
        """
        exc_info = kwargs.pop("exc_info", None)
        extra = self._prepare_extra(kwargs)
        self.logger.info(message, exc_info=exc_info, extra=extra)

    def error(self, message: str, **kwargs: Any) -> None:
        """
        Log error message.

        Args:
            message: Message text.
            **kwargs: Additional parameters passed to logger.
        """
        exc_info = kwargs.pop("exc_info", None)
        extra = self._prepare_extra(kwargs)
        self.logger.error(message, exc_info=exc_info, extra=extra)

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log warning message.

        Args:
            message: Message text.
            **kwargs: Additional parameters passed to logger.
        """
        exc_info = kwargs.pop("exc_info", None)
        extra = self._prepare_extra(kwargs)
        self.logger.warning(message, exc_info=exc_info, extra=extra)

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log debug message.

        Args:
            message: Message text.
            **kwargs: Additional parameters passed to logger.
        """
        exc_info = kwargs.pop("exc_info", None)
        extra = self._prepare_extra(kwargs)
        self.logger.debug(message, exc_info=exc_info, extra=extra)


def setup_logging(
    log_file: str = DEFAULT_LOG_FILE,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    log_level: str = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """
    Configure central logging with RotatingFileHandler and pure JSON Lines formatting.

    Args:
        log_file: Path to the log file.
        max_bytes: Maximum size of the log file before rotation.
        backup_count: Number of rotated backup files to keep.
        log_level: Default logging level string.
        log_format: Format string, typically 'json'.
    """
    with _logging_lock:
        _global_logging_overrides.clear()
        _global_logging_overrides.update(
            {
                "log_file": log_file,
                "max_bytes": max_bytes,
                "backup_count": backup_count,
                "log_level": log_level,
                "log_format": log_format,
            }
        )
        _logger_cache.clear()

        formatter: logging.Formatter
        if log_format == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )

        _get_or_create_rotating_handler(
            log_file=log_file,
            max_bytes=max_bytes,
            backup_count=backup_count,
            formatter=formatter,
        )


def get_logger(name: str = "vx_rag") -> StructuredLogger:
    """
    Return a ``StructuredLogger`` for the given name.

    A separate instance is cached per name so that log records from different
    modules carry distinct ``logger`` fields and can be filtered independently.

    Args:
        name: Logger name (typically ``__name__`` of the calling module).

    Returns:
        A configured ``StructuredLogger`` instance.
    """
    with _logging_lock:
        if name not in _logger_cache:
            config = load_logging_config()
            _logger_cache[name] = StructuredLogger(name, config)
        return _logger_cache[name]


def load_logging_config() -> dict[str, Any]:
    """
    Load logging configuration from settings.yaml merged with any setup_logging overrides.

    Returns:
        A dictionary containing logging configuration.
    """
    config: dict[str, Any] = {}
    try:
        # Check if test mode is enabled via environment variable
        config_path_str = os.environ.get("VX_RAG_CONFIG_PATH")
        config_path = (
            Path(config_path_str) if config_path_str else Path("config/settings.yaml")
        )

        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = yaml.safe_load(f)
                if isinstance(full_config, dict):
                    config = dict(full_config.get("logging", {}))
    except (OSError, yaml.YAMLError) as e:
        logging.getLogger("vx_rag.logging_config").warning(
            "Failed to load logging config: %s", e
        )

    config.update(_global_logging_overrides)
    return config


# Convenience functions
def log_query_event(
    query: str, top_k: int, results_count: int, duration: float
) -> None:
    """
    Log a query event.

    Args:
        query: Query string.
        top_k: Top K results requested.
        results_count: Number of results returned.
        duration: Elapsed duration in seconds.
    """
    logger = get_logger()
    from .metrics import get_metrics

    metrics = get_metrics()

    logger.log_event(
        "query_processed",
        query_length=len(query),
        top_k=top_k,
        results_count=results_count,
        duration_ms=duration * 1000,
    )

    metrics.increment("queries_total")
    metrics.gauge("query_duration_ms", duration * 1000)
    metrics.histogram("query_results_count", float(results_count))


def log_index_event(
    event_type: str, documents_count: int = 0, duration: float = 0.0
) -> None:
    """
    Log an indexing event.

    Args:
        event_type: Type of index event.
        documents_count: Number of documents indexed.
        duration: Elapsed duration in seconds.
    """
    logger = get_logger()
    from .metrics import get_metrics

    metrics = get_metrics()

    logger.log_event(
        f"index_{event_type}",
        documents_count=documents_count,
        duration_ms=duration * 1000,
    )

    metrics.increment(f"index_{event_type}_total")
    if duration > 0:
        metrics.histogram(f"index_{event_type}_duration_ms", duration * 1000)


def log_service_health(service_name: str, status: str, **kwargs: Any) -> None:
    """
    Log service health status.

    Args:
        service_name: Name of the service.
        status: Health status string.
        **kwargs: Extra metadata attributes.
    """
    logger = get_logger()
    logger.log_event("service_health", service=service_name, status=status, **kwargs)
