# Test Logs Directory

This directory contains log files generated during test runs.

## Purpose

- Separate test logs from production logs
- Make debugging test failures easier
- Keep production logs clean

## Naming Convention

Log files follow the pattern: `test_run_YYYY-MM-DD_HH-MM-SS.log`

## Note

This directory is automatically created by `conftest.py` if it doesn't exist.
Files in this directory can be safely deleted.
