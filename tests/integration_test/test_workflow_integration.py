"""
Integration tests for RAG Workflow migration.

Tests the new LlamaIndex Workflow implementation that replaces
custom orchestration layers with native async pipelines.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.rag.orchestrator import RAGWorkflow


class TestRAGWorkflow:
    """Test cases for RAG Workflow functionality."""

    @pytest.fixture
    def mock_query_engine(self):
        """Create a mock QueryEngine."""
        mock_engine = AsyncMock()
        # Mock the aretrieve method to return mock nodes
        mock_nodes = [
            Mock(text="Document 1 text", score=0.9, metadata={"source": "test"}, node_id="node1"),
            Mock(text="Document 2 text", score=0.8, metadata={"source": "test"}, node_id="node2"),
        ]
        mock_engine.aretrieve.return_value = mock_nodes
        return mock_engine

    @pytest.fixture
    def mock_assembler(self):
        """Create a mock ContextAssembler."""
        mock_assembler = Mock()
        mock_payload = Mock()
        mock_payload.total_tokens_estimate.return_value = 150
        mock_payload.context = [
            Mock(id="node1", text="Document 1 text", score=0.9, meta={"source": "test"}),
            Mock(id="node2", text="Document 2 text", score=0.8, meta={"source": "test"}),
        ]
        mock_assembler.assemble_context.return_value = mock_payload
        return mock_assembler

    @pytest.mark.asyncio
    async def test_workflow_retrieve_and_assemble_success(self, mock_query_engine, mock_assembler):
        """Test successful workflow execution."""
        workflow = RAGWorkflow(
            query_engine=mock_query_engine,
            assembler=mock_assembler
        )

        # Run workflow
        result = await workflow.run(
            query="test query",
            top_k=2,
            search_type="hybrid",
            token_budget=1000
        )

        # Verify result
        assert result is not None
        assert hasattr(result, 'context')
        assert len(result.context) == 2
        assert result.total_tokens_estimate() == 150

        # Verify query engine was called
        mock_query_engine.aretrieve.assert_called_once()

        # Verify assembler was called
        mock_assembler.assemble_context.assert_called_once_with(
            query="test query",
            documents=[
                {
                    'text': 'Document 1 text',
                    'score': 0.9,
                    'metadata': {'source': 'test'},
                    'node_id': 'node1'
                },
                {
                    'text': 'Document 2 text',
                    'score': 0.8,
                    'metadata': {'source': 'test'},
                    'node_id': 'node2'
                }
            ],
            token_budget=1000,
            max_items=2
        )

    @pytest.mark.asyncio
    async def test_workflow_without_assembler_fallback(self, mock_query_engine):
        """Test workflow execution without assembler (fallback)."""
        workflow = RAGWorkflow(
            query_engine=mock_query_engine,
            assembler=None
        )

        # Run workflow
        result = await workflow.run(
            query="test query",
            top_k=2,
            search_type="hybrid",
            token_budget=1000
        )

        # Verify result
        assert result is not None
        assert hasattr(result, 'context')
        assert len(result.context) == 2
        assert result.schema_version == "1.0"
        assert result.token_budget == 1000

    @pytest.mark.asyncio
    async def test_workflow_query_validation(self, mock_query_engine, mock_assembler):
        """Test workflow query validation."""
        workflow = RAGWorkflow(
            query_engine=mock_query_engine,
            assembler=mock_assembler
        )

        # Test with empty query
        with pytest.raises(ValueError, match="Query is required"):
            await workflow.run(query="")

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, mock_query_engine, mock_assembler):
        """Test workflow error handling."""
        # Make query engine raise an exception
        mock_query_engine.aretrieve.side_effect = Exception("Retrieval failed")

        workflow = RAGWorkflow(
            query_engine=mock_query_engine,
            assembler=mock_assembler
        )

        # Run workflow and expect exception
        with pytest.raises(Exception, match="Retrieval failed"):
            await workflow.run(query="test query")