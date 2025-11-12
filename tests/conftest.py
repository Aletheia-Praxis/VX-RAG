"""
Pytest configuration file for VX-RAG tests.

This file imports test-specific configuration from tests/config_test/ directory
to prevent pollution of production logs and data.
"""

import os
import sys
from pathlib import Path
from typing import Generator, Any

import pytest

# Add src and tests directory to path for imports
project_root = Path(__file__).parent.parent
tests_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(tests_root))

# Import test configuration utilities after path is set
from config_test.test_logging import configure_test_logging, get_test_config_path

# Set environment variable to use test configuration
os.environ["VX_RAG_TEST_MODE"] = "true"
os.environ["VX_RAG_CONFIG_PATH"] = str(get_test_config_path())


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """
    Setup test environment before all tests.
    
    This fixture:
    - Configures test-specific logging
    - Creates test directories
    - Prevents production log pollution
    """
    # Create test directories
    test_logs_dir = project_root / "tests" / "test_logs"
    test_logs_dir.mkdir(parents=True, exist_ok=True)
    
    test_data_dir = project_root / "tests" / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger for tests using config_test module
    configure_test_logging(test_logs_dir)
    
    yield
    
    # Cleanup after all tests (optional)
    # You can add cleanup logic here if needed


@pytest.fixture(autouse=True)
def reset_singleton_logger() -> Generator[None, None, None]:
    """
    Reset singleton logger instance before each test.
    
    This ensures that each test starts with a clean logger state
    and uses the test configuration instead of production configuration.
    """
    # Import here to avoid circular imports
    from src.utils import logging_config
    
    # Reset the singleton instance
    logging_config._logger_instance = None
    
    yield
    
    # Reset again after test
    logging_config._logger_instance = None


@pytest.fixture
def temp_test_dir(tmp_path: Path) -> Path:
    """
    Provide a temporary directory for test files.
    
    This fixture is imported from config_test.test_fixtures.
    
    Args:
        tmp_path: Pytest's built-in temporary path fixture
        
    Returns:
        Path to temporary test directory
    """
    from config_test.test_fixtures import get_temp_test_dir
    return get_temp_test_dir(tmp_path)


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """
    Provide a mock configuration for tests.
    
    This fixture is imported from config_test.test_fixtures.
    
    Returns:
        Dictionary with test configuration
    """
    from config_test.test_fixtures import get_mock_config
    return get_mock_config()


# Configure pytest to capture logs
def pytest_configure(config: Any) -> None:
    """
    Pytest hook to configure test environment.
    
    Args:
        config: Pytest configuration object
    """
    # Set pytest log level
    config.option.log_cli = False
    config.option.log_cli_level = "WARNING"
    config.option.log_file_level = "DEBUG"


# Suppress warnings from third-party libraries
@pytest.fixture(autouse=True)
def suppress_warnings() -> None:
    """Suppress common warnings from third-party libraries."""
    import warnings
    
    # Suppress specific warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="llama_index")
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
