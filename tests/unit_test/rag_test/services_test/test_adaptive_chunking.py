"""
Unit tests for adaptive chunking functionality.

Note: src.rag.services was removed during refactoring in develop-den.
This test suite is marked skipped pending implementation of the new pipeline.
"""

import pytest

pytestmark = pytest.mark.skip(
    reason="src.rag.services.chunker_service was removed in refactoring"
)


def test_adaptive_chunking_placeholder() -> None:
    """Placeholder test for retired adaptive chunking service."""
    pass
