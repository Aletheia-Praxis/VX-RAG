"""
Unit tests for ingest functions.

Note: src.rag.services was removed during refactoring in develop-den.
This test suite is marked skipped pending implementation of the new pipeline.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="src.rag.services.ingest_service was removed in refactoring"
)


def test_ingest_placeholder() -> None:
    """Placeholder test for retired ingest service."""
    pass
