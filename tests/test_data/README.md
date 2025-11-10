# Test Data Directory

This directory contains test data used during test runs.

## Purpose

- Separate test data from production data
- Provide isolated environment for tests
- Prevent accidental modification of production data

## Structure

```text
test_data/
├── raw/           # Raw test files
├── processed/     # Processed test files
├── index/         # Test index files
└── extracted_images/  # Test extracted images
```

## Note

This directory is automatically created by `conftest.py` if it doesn't exist.
Files in this directory can be safely deleted.
