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
        # Mock the aretrieve method to return mock NodeWithScore objects
        from unittest.mock import Mock
        from llama_index.core.schema import NodeWithScore, TextNode
        
        # Create proper NodeWithScore objects
        node1 = TextNode(text="Document 1 text", id_="node1", metadata={"source": "test"})
        node2 = TextNode(text="Document 2 text", id_="node2", metadata={"source": "test"})
        
        mock_nodes = [
            NodeWithScore(node=node1, score=0.9),
            NodeWithScore(node=node2, score=0.8),
        ]
        mock_engine.aretrieve.return_value = mock_nodes
        return mock_engine

    @pytest.fixture
    def mock_response_synthesizer(self):
        """Create a mock Response Synthesizer."""
        mock_synthesizer = AsyncMock()
        mock_response = Mock()
        mock_response.response = "Synthesized response text"
        mock_synthesizer.asynthesize.return_value = mock_response
        return mock_synthesizer

    @pytest.mark.asyncio
    async def test_workflow_retrieve_and_assemble_success(self, mock_query_engine, mock_response_synthesizer):
        """Test successful workflow execution."""
        workflow = RAGWorkflow(
            query_engine=mock_query_engine,
            response_synthesizer=mock_response_synthesizer
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
        assert result.query == "test query"
        assert result.schema_version == "1.0"

        # Verify query engine was called
        mock_query_engine.aretrieve.assert_called_once()

        # Verify response synthesizer was called
        mock_response_synthesizer.asynthesize.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_without_synthesizer_fallback(self, mock_query_engine):
        """Test workflow execution without response synthesizer (fallback)."""
        workflow = RAGWorkflow(
            query_engine=mock_query_engine,
            response_synthesizer=None
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
        assert result.query == "test query"
        assert result.schema_version == "1.0"
        assert result.token_budget == 1000
        # Fallback case has empty provenance
        assert hasattr(result, 'provenance')
        assert result.provenance == {}

    @pytest.mark.asyncio
    async def test_workflow_query_validation(self, mock_query_engine, mock_response_synthesizer):
        """Test workflow query validation."""
        workflow = RAGWorkflow(
            query_engine=mock_query_engine,
            response_synthesizer=mock_response_synthesizer
        )

        # Test with empty query
        with pytest.raises(ValueError, match="Query is required"):
            await workflow.run(query="")

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, mock_query_engine, mock_response_synthesizer):
        """Test workflow error handling."""
        # Make query engine raise an exception
        mock_query_engine.aretrieve.side_effect = Exception("Retrieval failed")

        workflow = RAGWorkflow(
            query_engine=mock_query_engine,
            response_synthesizer=mock_response_synthesizer
        )

        # Run workflow and expect exception
        with pytest.raises(Exception, match="Retrieval failed"):
            await workflow.run(query="test query")