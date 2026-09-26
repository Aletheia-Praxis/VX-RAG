"""
E2E Test Runner Script for VX-RAG.

Allows running the end-to-end test suite directly via Python:
    python tests/e2e_test/run_e2e_tests.py [pytest_args]

Equivalent to:
    pytest tests/e2e_test/ [pytest_args]
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest


def run_e2e_suite(extra_args: List[str] | None = None) -> int:
    """
    Execute the E2E test suite using pytest.

    Args:
        extra_args: Optional additional command-line arguments to pass to pytest.

    Returns:
        Exit code from pytest execution (0 for pass).
    """
    e2e_dir = Path(__file__).resolve().parent
    args = [str(e2e_dir), "-v"]
    if extra_args:
        args.extend(extra_args)

    print(f"Executing VX-RAG E2E Test Suite with arguments: {args}")
    exit_code = pytest.main(args)
    return int(exit_code)


if __name__ == "__main__":
    cli_args = sys.argv[1:]
    sys.exit(run_e2e_suite(cli_args))
