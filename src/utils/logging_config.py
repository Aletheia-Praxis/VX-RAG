"""
Logging configuration for VX-RAG project.

Provides JSON-formatted logging with structured events.
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Create base log entry
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                              'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                              'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                              'thread', 'threadName', 'processName', 'process', 'message']:
                    log_entry[key] = value

        return json.dumps(log_entry, ensure_ascii=False)


class StructuredLogger:
    """Structured logger with event logging capabilities."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(name)
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Setup logger with appropriate formatters and handlers."""
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Set level
        level_str = self.config.get('log_level', 'INFO')
        level = getattr(logging, level_str.upper(), logging.INFO)
        self.logger.setLevel(level)

        # Create formatters
        formatter: Union[JSONFormatter, logging.Formatter]
        if self.config.get('log_format', 'json') == 'json':
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        log_file = self.config.get('log_file')
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

        # Prevent duplicate logs
        self.logger.propagate = False

    def log_event(self, event_type: str, **kwargs: Any) -> None:
        """Log a structured event."""
        if not self.config.get('log_structured_events', True):
            return

        self.logger.info(f"Event: {event_type}", extra={'event_type': event_type, **kwargs})

    def log_metric(self, metric_name: str, value: Any, **tags: Any) -> None:
        """Log a metric."""
        if not self.config.get('log_metrics', True):
            return

        self.logger.info(f"Metric: {metric_name}={value}", extra={
            'metric_name': metric_name,
            'metric_value': value,
            **tags
        })

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(message, extra=kwargs if kwargs else None)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(message, extra=kwargs if kwargs else None)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(message, extra=kwargs if kwargs else None)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(message, extra=kwargs if kwargs else None)


# Global instance
_logger_instance: Optional[StructuredLogger] = None


def get_logger(name: str = "vx_rag") -> StructuredLogger:
    """Get or create the global structured logger."""
    global _logger_instance
    if _logger_instance is None:
        # Load config
        config = load_logging_config()
        _logger_instance = StructuredLogger(name, config)
    return _logger_instance


def load_logging_config() -> Dict[str, Any]:
    """Load logging configuration from settings.yaml."""
    try:
        config_path = Path("config/settings.yaml")
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = yaml.safe_load(f)
                return full_config.get('logging', {}) if isinstance(full_config, dict) else {}
    except Exception as e:
        logging.warning(f"Failed to load logging config: {e}")
    return {}


# Convenience functions
def log_query_event(query: str, top_k: int, results_count: int, duration: float) -> None:
    """Log a query event."""
    logger = get_logger()
    from .metrics import get_metrics
    metrics = get_metrics()

    logger.log_event(
        "query_processed",
        query_length=len(query),
        top_k=top_k,
        results_count=results_count,
        duration_ms=duration * 1000
    )

    metrics.increment("queries_total")
    metrics.gauge("query_duration_ms", duration * 1000)
    metrics.histogram("query_results_count", results_count)


def log_index_event(event_type: str, documents_count: int = 0, duration: float = 0.0) -> None:
    """Log an indexing event."""
    logger = get_logger()
    from .metrics import get_metrics
    metrics = get_metrics()

    logger.log_event(
        f"index_{event_type}",
        documents_count=documents_count,
        duration_ms=duration * 1000
    )

    metrics.increment(f"index_{event_type}_total")
    if duration > 0:
        metrics.histogram(f"index_{event_type}_duration_ms", duration * 1000)


def log_service_health(service_name: str, status: str, **kwargs: Any) -> None:
    """Log service health status."""
    logger = get_logger()
    logger.log_event(
        "service_health",
        service=service_name,
        status=status,
        **kwargs
    )
