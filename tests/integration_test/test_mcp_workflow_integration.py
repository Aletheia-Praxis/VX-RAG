"""
Integration tests for MCP handlers with Workflow migration.

Tests that MCP handlers correctly use the new async Workflow
instead of the old synchronous orchestrator methods.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch

from src.mcp.handlers import handle_query_knowledge_base
from src.mcp.schemas import QueryKnowledgeBaseRequest


class TestMCPHandlersWorkflowIntegration:
    """Test MCP handlers integration with Workflow."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock orchestrator with Workflow."""
        mock_orch = Mock()
        mock_workflow_result = Mock()
        mock_workflow_result.context = [
            Mock(id="node1", text="Document 1", score=0.9, meta={"source": "test"}),
            Mock(id="node2", text="Document 2", score=0.8, meta={"source": "test"}),
        ]
        mock_workflow_result.total_tokens_estimate.return_value = 150
        mock_orch.query_async = AsyncMock(return_value={
            'query': 'test query',
            'context': [
                {'id': 'node1', 'text': 'Document 1', 'score': 0.9, 'metadata': {'source': 'test'}},
                {'id': 'node2', 'text': 'Document 2', 'score': 0.8, 'metadata': {'source': 'test'}},
            ],
            'total_tokens_estimate': 150,
            'sources_count': 2,
            'retrieval_stats': {
                'total_duration_ms': 100.0,
                'search_type': 'hybrid',
                'note': 'Query executed via LlamaIndex Workflow'
            }
        })
        return mock_orch

    @pytest.mark.asyncio
    async def test_handle_query_knowledge_base_with_workflow(self, mock_orchestrator):
        """Test that query handler uses Workflow via orchestrator."""
        with patch('src.mcp.handlers._get_orchestrator', return_value=mock_orchestrator):
            params = QueryKnowledgeBaseRequest(
                query="test cybersecurity query",
                top_k=5,
                search_type="hybrid",
                token_budget=4000
            )

            result = await handle_query_knowledge_base(params)

            # Verify orchestrator was called with async method
            mock_orchestrator.query_async.assert_called_once_with(
                query="test cybersecurity query",
                top_k=5,
                search_type="hybrid",
                token_budget=4000
            )

            # Verify result is JSON string
            assert isinstance(result, str)
            assert '"query": "test cybersecurity query"' in result
            assert '"sources_count": 2' in result
            assert '"note": "Query executed via LlamaIndex Workflow"' in result

    @pytest.mark.asyncio
    async def test_handle_query_knowledge_base_error_handling(self, mock_orchestrator):
        """Test error handling in query handler."""
        # Make orchestrator raise an exception
        mock_orchestrator.query_async.side_effect = RuntimeError("Workflow not initialized")

        with patch('src.mcp.handlers._get_orchestrator', return_value=mock_orchestrator):
            params = QueryKnowledgeBaseRequest(
                query="test query",
                top_k=3,
                search_type="semantic",
                token_budget=2000
            )

            result = await handle_query_knowledge_base(params)

            # Verify error response
            assert isinstance(result, str)
            assert '"error_type": "service_unavailable"' in result
            assert '"error_message": "System not ready: Workflow not initialized"' in result