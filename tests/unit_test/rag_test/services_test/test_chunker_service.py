"""
Unit tests for chunker service.

Note: src.rag.services was removed during refactoring in develop-den.
This test suite is marked skipped pending implementation of the new pipeline.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="src.rag.services.chunker_service was removed in refactoring"
)


def test_chunker_service_placeholder() -> None:
    """Placeholder test for retired chunker service."""
    pass
