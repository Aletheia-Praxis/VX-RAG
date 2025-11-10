# Test Configuration

This directory contains all test-specific configurations for VX-RAG.

## Purpose

- **Separate test configuration from production**: Ensures tests don't interfere with production settings
- **Test-specific logging**: Prevents pollution of production logs
- **Isolated test environment**: Tests use their own data directories and settings

## Structure

```text
config_test/
├── __init__.py           # Package initialization
├── settings.yaml         # Test-specific settings (mirrors production config structure)
├── test_logging.py       # Logging configuration utilities for tests
├── test_fixtures.py      # Reusable test fixtures
└── README.md             # This file
```

## Files

### `settings.yaml`

Test-specific configuration that mirrors the structure of `config/settings.yaml` but with:

- Separate log directories (`tests/test_logs/`)
- Separate data directories (`tests/test_data/`)
- DEBUG log level for verbose test output
- Disabled metrics to reduce noise
- Test-optimized settings

### `test_logging.py`

Provides the `configure_test_logging()` function that:

- Configures separate log files for each test run
- Sets appropriate log levels (DEBUG for file, WARNING for console)
- Prevents test logs from polluting production logs
- Timestamps each test run log file

### `test_fixtures.py`

Contains reusable pytest fixtures:

- `get_temp_test_dir()` - Temporary directory for test files
- `get_mock_config()` - Mock configuration dictionary for tests

## Usage

The test configuration is automatically loaded by `tests/conftest.py`:

```python
from config_test.test_logging import configure_test_logging, get_test_config_path
```

Environment variables are set automatically:

- `VX_RAG_TEST_MODE=true`
- `VX_RAG_CONFIG_PATH=tests/config_test/settings.yaml`

## Key Differences from Production Config

| Setting | Production | Test |
|---------|-----------|------|
| Log Level | INFO | DEBUG |
| Log Directory | `logs/` | `tests/test_logs/` |
| Data Directory | `data/` | `tests/test_data/` |
| Metrics | Enabled | Disabled |
| Log File Naming | `vx_rag_TIMESTAMP.log` | `test_run_TIMESTAMP.log` |

## Adding New Test Configuration

When adding new configuration options:

1. Add them to `settings.yaml` first
2. If they require special handling, add utilities to `test_logging.py` or `test_fixtures.py`
3. Update this README to document the changes
4. Ensure production config (`config/settings.yaml`) is also updated if needed
