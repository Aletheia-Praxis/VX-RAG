# VX-RAG Tests

This directory contains all tests for the VX-RAG project, with a clean separation between test and production environments.

## Structure

```text
tests/
├── conftest.py           # Main pytest configuration file
├── config_test/          # Test-specific configuration
│   ├── settings.yaml     # Test settings (mirrors production structure)
│   ├── test_logging.py   # Logging configuration for tests
│   ├── test_fixtures.py  # Reusable test fixtures
│   └── README.md         # Configuration documentation
├── test_logs/            # Test log files (separate from production)
│   └── test_run_*.log    # Timestamped test execution logs
├── test_data/            # Test data directory (separate from production)
│   ├── raw/              # Raw test files
│   ├── processed/        # Processed test files
│   └── index/            # Test index files
├── unit_test/            # Unit tests
│   ├── mcp_test/         # MCP module tests
│   ├── rag_test/         # RAG module tests
│   └── utils_test/       # Utility module tests
└── integration_test/     # Integration tests

## Key Features

### Separate Configuration

Tests use their own configuration from `config_test/settings.yaml`:
- Separate log directory (`tests/test_logs/`)
- Separate data directory (`tests/test_data/`)
- DEBUG log level for verbose output
- Disabled metrics to reduce noise

### No Production Pollution

Tests **never** create files in production directories:
- Production logs (`logs/`) remain clean
- Production data (`data/`) is untouched
- Production config (`config/`) is not modified

### Clean Test Execution

- Each test run creates a timestamped log file
- Singleton logger is reset between tests
- Third-party warnings are suppressed
- Environment variables are automatically set

## Running Tests

### Run all tests

```bash
pytest
```

### Run specific test suite

```bash
# Unit tests only
pytest tests/unit_test/

# Integration tests only
pytest tests/integration_test/

# Specific module
pytest tests/unit_test/utils_test/
```

### Run with verbose output

```bash
pytest -v
```

### Run with coverage

```bash
pytest --cov=src --cov-report=html
```

## Naming Convention

All test directories follow the `*_test` pattern:

- `unit_test/` instead of `unit/`
- `integration_test/` instead of `integration/`
- `mcp_test/` instead of `mcp/`
- etc.

This makes it clear that these are test directories and prevents confusion with production code.

## Environment Variables

The following environment variables are automatically set during test execution:

- `VX_RAG_TEST_MODE=true` - Indicates test mode is active
- `VX_RAG_CONFIG_PATH=tests/config_test/settings.yaml` - Points to test configuration

## Fixtures

Common fixtures are available in all tests:

### `temp_test_dir`

Provides a temporary directory for test files (automatically cleaned up).

```python
def test_something(temp_test_dir):
    test_file = temp_test_dir / "test.txt"
    test_file.write_text("test data")
```

### `mock_config`

Provides a mock configuration dictionary.

```python
def test_with_config(mock_config):
    assert mock_config['log_level'] == 'DEBUG'
```

## Adding New Tests

1. Create test file in appropriate directory following `test_*.py` pattern
2. Import from `src.*` (not relative imports)
3. Use provided fixtures (`temp_test_dir`, `mock_config`)
4. Run tests to verify they work

Example:

```python
"""Test for my new feature."""

import pytest
from src.mymodule import MyClass


def test_my_feature():
    """Test that my feature works."""
    obj = MyClass()
    assert obj.do_something() == expected_result
```

## Troubleshooting

### Tests creating files in production directories

Check that `conftest.py` is properly setting environment variables:

- `VX_RAG_CONFIG_PATH` should point to `tests/config_test/settings.yaml`
- Verify `load_logging_config()` in `src/utils/logging_config.py` reads this variable

### Import errors

Ensure you're importing from `src.*`, not using relative imports between test files.

### Singleton logger issues

The `reset_singleton_logger` fixture should reset the logger between tests. If you still have issues, manually reset it in your test:

```python
def test_something():
    from src.utils import logging_config
    logging_config._logger_instance = None
    # Your test code here
```

## Best Practices

1. **Keep tests isolated**: Each test should be independent
2. **Use fixtures**: Leverage provided fixtures for common setup
3. **Clean up resources**: Use `tmp_path` for temporary files
4. **Mock external dependencies**: Don't make real API calls or access real resources
5. **Test one thing**: Each test should verify one specific behavior
6. **Clear naming**: Use descriptive test names that explain what is being tested

## Contributing

When adding new test utilities:

1. Add to appropriate module in `config_test/`
2. Update `config_test/README.md`
3. Add fixtures to `test_fixtures.py` if reusable
4. Document in this README if it's a common pattern

---

For more information about test configuration, see [`config_test/README.md`](config_test/README.md).
