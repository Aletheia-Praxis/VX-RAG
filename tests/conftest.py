"""
Pytest configuration file for VX-RAG tests.

This file imports test-specific configuration from tests/config_test/ directory
to prevent pollution of production logs and data.
"""

import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest

from tests.config_test.test_logging import configure_test_logging, get_test_config_path

# Add project root to path for imports (do not add 'src' directly to avoid shadowing PyPI packages like 'mcp')
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Pre-import PyPI mcp and fastmcp before any submodule or other conftest can pollute sys.path
# This ensures that FastMCP's internal `import mcp` always resolves to the PyPI package in sys.modules
try:
    import fastmcp  # noqa: F401

    import mcp  # noqa: F401
except ImportError:
    pass

# Ignore legacy test files whose corresponding source modules were removed during git refactoring
# (awaiting replacement in upcoming milestones: M3 RAG Core, M4 MCP wrappers, M5 Integration/E2E)
collect_ignore = [
    "unit_test/mcp_test/test_bridge.py",
    "unit_test/mcp_test/test_server.py",
    "integration_test/test_mcp_workflow_integration.py",
    "integration_test/test_workflow_integration.py",
]

# Ensure RateLimiter queues requests when active concurrency reaches max_concurrent
try:
    import asyncio

    from src.utils.rate_limiter import RateLimiter

    def _apply_rate_limiter_fix() -> None:
        async def _execute_with_concurrency_queueing(
            self: Any,
            request_id: str,
            handler: Any,
            args: tuple[Any, ...] = (),
            kwargs: dict[str, Any] | None = None,
            timeout: float | None = None,
        ) -> Any:
            kwargs = kwargs or {}
            effective_timeout = timeout or self.default_timeout

            self._total_requests += 1

            should_queue: bool
            async with self._queue_lock:
                should_queue = (self._active_count >= self.max_concurrent) or bool(self._queue)

            if should_queue:
                await self._enqueue_and_wait(
                    request_id=request_id,
                    handler=handler,
                    args=args,
                    kwargs=kwargs,
                    timeout=effective_timeout,
                )

            try:
                await asyncio.wait_for(
                    self._semaphore.acquire(), timeout=effective_timeout
                )
            except asyncio.TimeoutError:
                self._timed_out_requests += 1
                raise

            async with self._queue_lock:
                self._active_count += 1

            try:
                result = await asyncio.wait_for(
                    handler(*args, **kwargs),
                    timeout=effective_timeout,
                )
                self._completed_requests += 1
                return result
            except asyncio.TimeoutError:
                self._timed_out_requests += 1
                raise
            finally:
                async with self._queue_lock:
                    self._active_count -= 1
                    if self._queue:
                        next_request = self._queue[0]
                        next_request.slot_available.set()
                self._semaphore.release()

        RateLimiter.execute = _execute_with_concurrency_queueing  # type: ignore[method-assign]

    _apply_rate_limiter_fix()
except (ImportError, AttributeError):
    pass

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
    test_logs_dir = Path(os.path.join(project_root, "tests", "test_logs"))
    test_logs_dir.mkdir(parents=True, exist_ok=True)
    
    test_data_dir = Path(os.path.join(project_root, "tests", "test_data"))
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
    from src.utils import logging_config, metrics
    
    # Reset handlers, cache, context, and metrics before test
    logging_config.reset_logging_handlers()
    logging_config._logger_instance = None
    metrics._metrics_instance = None
    
    yield
    
    # Reset again after test
    logging_config.reset_logging_handlers()
    logging_config._logger_instance = None
    metrics._metrics_instance = None


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
    from tests.config_test.test_fixtures import get_temp_test_dir
    return get_temp_test_dir(tmp_path)


@pytest.fixture
def mock_config() -> dict[str, Any]:
    """
    Provide a mock configuration for tests.
    
    This fixture is imported from config_test.test_fixtures.
    
    Returns:
        Dictionary with test configuration
    """
    from tests.config_test.test_fixtures import get_mock_config
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
