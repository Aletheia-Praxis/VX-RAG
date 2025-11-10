"""
Test-specific logging configuration.

This module provides logging setup for tests to prevent pollution of production logs.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime


def configure_test_logging(log_dir: Path) -> None:
    """
    Configure test-specific logging.
    
    This ensures that:
    - Tests use separate log files
    - Log level is appropriate for tests (DEBUG)
    - Production logs are not affected
    
    Args:
        log_dir: Directory for test log files
    """
    # Remove all existing handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set test log level
    root_logger.setLevel(logging.DEBUG)
    
    # Console handler - only show warnings and above during tests
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler - capture all DEBUG messages
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = log_dir / f"test_run_{timestamp}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Disable propagation for specific loggers to prevent duplicate logs
    for logger_name in ['llama_index', 'sentence_transformers', 'transformers']:
        logger = logging.getLogger(logger_name)
        logger.propagate = False
        logger.setLevel(logging.WARNING)


def get_test_config_path() -> Path:
    """
    Get path to test configuration file.
    
    Returns:
        Path to settings.yaml in test config directory
    """
    return Path(__file__).parent / "settings.yaml"
