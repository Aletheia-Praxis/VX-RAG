"""Test logging configuration."""

from pathlib import Path
from unittest.mock import patch
from src.utils.logging_config import get_logger, StructuredLogger, load_logging_config


def test_load_logging_config():
    """Test loading logging config from settings.yaml."""
    config = load_logging_config()
    assert isinstance(config, dict)
    assert 'log_level' in config


def test_structured_logger_creation():
    """Test StructuredLogger creation."""
    config = {'log_level': 'INFO', 'log_format': 'json'}
    logger = StructuredLogger('test_logger', config)
    assert logger.name == 'test_logger'
    assert logger.config == config


def test_logger_file_handler_with_timestamp(tmp_path):
    """Test that file handler creates timestamped log files."""
    log_dir = tmp_path / "logs"
    log_file = "test.log"
    config = {
        'log_level': 'INFO',
        'log_format': 'json',
        'log_file': str(log_dir / log_file)
    }

    logger = StructuredLogger('test_logger', config)

    # Trigger logging to create file
    logger.info("Test message")

    # Check if directory was created
    assert log_dir.exists()

    # Check if a timestamped file was created
    log_files = list(log_dir.glob(f"{Path(log_file).stem}_*{Path(log_file).suffix}"))
    assert len(log_files) == 1

    # Check file content
    log_content = log_files[0].read_text(encoding='utf-8')
    assert "Test message" in log_content
    assert "timestamp" in log_content


def test_get_logger_singleton():
    """Test that get_logger returns singleton instance."""
    logger1 = get_logger()
    logger2 = get_logger()
    assert logger1 is logger2


def test_log_event():
    """Test logging structured events."""
    config = {'log_level': 'INFO', 'log_structured_events': True}
    logger = StructuredLogger('test_logger', config)

    with patch.object(logger.logger, 'info') as mock_info:
        logger.log_event('test_event', key='value')
        mock_info.assert_called_once()
        args, kwargs = mock_info.call_args
        assert 'event_type' in kwargs['extra']
        assert kwargs['extra']['event_type'] == 'test_event'
        assert kwargs['extra']['key'] == 'value'


def test_log_metric():
    """Test logging metrics."""
    config = {'log_level': 'INFO', 'log_metrics': True}
    logger = StructuredLogger('test_logger', config)

    with patch.object(logger.logger, 'info') as mock_info:
        logger.log_metric('test_metric', 42, unit='ms')
        mock_info.assert_called_once()
        args, kwargs = mock_info.call_args
        assert 'metric_name' in kwargs['extra']
        assert kwargs['extra']['metric_name'] == 'test_metric'
        assert kwargs['extra']['metric_value'] == 42
        assert kwargs['extra']['unit'] == 'ms'