"""
Metrics collection for VX-RAG project.

Provides simple metrics collection and logging capabilities.
"""

import threading
import time
from typing import Dict, Any, Optional
from .logging_config import get_logger, StructuredLogger


class MetricsCollector:
    """Simple metrics collector for VX-RAG."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics: Dict[str, Any] = {}
        # RLock (re-entrant) allows log_all_metrics() to be called from the same
        # thread that already holds the lock (e.g. the periodic logging thread).
        self.lock = threading.RLock()
        self.logger: StructuredLogger = get_logger("metrics")

        # Start metrics logging thread if enabled
        if self.config.get('metrics_enabled', True):
            self._start_metrics_thread()

    def _start_metrics_thread(self) -> None:
        """Start background thread for periodic metrics logging."""
        interval = self.config.get('metrics_log_interval', 300)
        thread = threading.Thread(target=self._log_metrics_periodically, args=(interval,), daemon=True)
        thread.start()

    def _log_metrics_periodically(self, interval: int) -> None:
        """Log metrics periodically."""
        while True:
            time.sleep(interval)
            self.log_all_metrics()

    def increment(self, name: str, value: int = 1, **tags: Dict[str, Any]) -> None:
        """Increment a counter metric."""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = {'type': 'counter', 'value': 0, 'tags': tags}
            self.metrics[name]['value'] += value

    def gauge(self, name: str, value: float, **tags: Dict[str, Any]) -> None:
        """Set a gauge metric."""
        with self.lock:
            self.metrics[name] = {'type': 'gauge', 'value': value, 'tags': tags}

    def histogram(self, name: str, value: float, **tags: Dict[str, Any]) -> None:
        """Record a histogram value."""
        with self.lock:
            if name not in self.metrics:
                self.metrics[name] = {'type': 'histogram', 'values': [], 'tags': tags}
            self.metrics[name]['values'].append(value)
            # Keep only last 100 values
            self.metrics[name]['values'] = self.metrics[name]['values'][-100:]

    def log_all_metrics(self) -> None:
        """Log all current metrics."""
        with self.lock:
            for name, metric in self.metrics.items():
                if metric['type'] == 'histogram' and metric['values']:
                    # Log average for histograms
                    avg_value = sum(metric['values']) / len(metric['values'])
                    self.logger.log_metric(name, avg_value, **metric.get('tags', {}))
                else:
                    self.logger.log_metric(name, metric['value'], **metric.get('tags', {}))
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current metrics as a dictionary.
        
        Returns:
            Dictionary of metric names and their current values
        """
        with self.lock:
            stats = {}
            for name, metric in self.metrics.items():
                if metric['type'] == 'histogram' and metric.get('values'):
                    # Calculate statistics for histograms
                    values = metric['values']
                    stats[name] = {
                        'type': 'histogram',
                        'count': len(values),
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
                else:
                    stats[name] = metric.get('value', 0)
            return stats


# Global instance and its creation lock (B-19)
_metrics_instance: Optional[MetricsCollector] = None
_metrics_creation_lock = threading.Lock()


def get_metrics() -> MetricsCollector:
    """
    Return the global ``MetricsCollector`` singleton.

    Uses double-checked locking so concurrent callers from different threads
    never construct more than one instance.

    Returns:
        The global ``MetricsCollector`` instance.
    """
    global _metrics_instance
    if _metrics_instance is None:
        with _metrics_creation_lock:
            if _metrics_instance is None:
                from .logging_config import load_logging_config
                config = load_logging_config()
                _metrics_instance = MetricsCollector(config)
    return _metrics_instance