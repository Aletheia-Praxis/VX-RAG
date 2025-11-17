"""
Unit tests for ContextAssembler service.
Tests the refactored implementation using LlamaIndex TokenCountingHandler.
"""

import pytest
from typing import Any, Dict, List

from src.rag.services.assembler_service.service import ContextAssembler


class TestContextAssembler:
    """Test suite for ContextAssembler with LlamaIndex TokenCountingHandler."""

    @pytest.fixture
    def assembler(self) -> ContextAssembler:
        """Create ContextAssembler instance for testing."""
        return ContextAssembler()

    @pytest.fixture
    def sample_documents(self) -> List[Dict[str, Any]]:
        """Sample documents for testing."""
        return [
            {
                'id': 'doc1',
                'text': 'This is a test document about cybersecurity.',
                'score': 0.9,
                'metadata': {'source': 'test.pdf', 'page': 1}
            },
            {
                'id': 'doc2',
                'text': 'Another document with different content.',
                'score': 0.7,
                'metadata': {'source': 'test2.pdf', 'page': 2}
            },
            {
                'id': 'doc3',
                'text': 'Third document for testing purposes.',
                'score': 0.5,
                'metadata': {'source': 'test3.pdf', 'page': 3}
            }
        ]

    def test_init(self, assembler: ContextAssembler):
        """Test ContextAssembler initialization."""
        assert assembler.default_token_budget == 2048  # From config
        assert assembler.model_name == "gpt-3.5-turbo"
        assert assembler.token_counter is not None

    def test_select_documents_by_budget(self, assembler: ContextAssembler, sample_documents: List[Dict[str, Any]]):
        """Test document selection within token budget."""
        selected, total_tokens = assembler._select_documents_by_budget(
            sample_documents, token_budget=100, max_items=2
        )

        assert len(selected) <= 2  # max_items limit
        assert total_tokens <= 100  # token budget limit
        assert selected[0]['score'] >= selected[1]['score']  # sorted by score descending

    def test_select_top_documents(self, assembler: ContextAssembler, sample_documents: List[Dict[str, Any]]):
        """Test top documents selection."""
        selected = assembler.select_top_documents(
            sample_documents, max_tokens=100, max_items=2
        )

        assert len(selected) <= 2
        assert selected[0]['score'] >= selected[1]['score']

    def test_reset_token_counts(self, assembler: ContextAssembler):
        """Test token counter reset."""
        # Should not raise an exception
        assembler.reset_token_counts()