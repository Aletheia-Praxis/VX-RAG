"""
Tests for MCP server module.
"""

import pytest
from unittest.mock import Mock, patch
from src.mcp.server import mcp, query_documents, get_health_status, get_system_context


class TestMCPServer:
    """Test cases for MCP server functionality."""

    def test_query_documents_tool_success(self) -> None:
        """Test successful document query tool execution."""
        # Mock query handler
        mock_handler = Mock()
        mock_handler.query_documents.return_value = ("Test response", [("text1", 0.9, {"meta": "data"})])

        with patch('src.mcp.server.query_handler', mock_handler):
            from src.mcp.server import QueryParams
            params = QueryParams(query="test query", top_k=3)

            result = query_documents(params)  # type: ignore

            assert "Query: test query" in result  # nosec B101
            assert "Test response" in result  # nosec B101
            assert "Result 1" in result  # nosec B101
            mock_handler.query_documents.assert_called_once_with("test query", 3)

    def test_query_documents_tool_no_handler(self) -> None:
        """Test query tool when handler is not initialized."""
        with patch('src.mcp.server.query_handler', None):
            from src.mcp.server import QueryParams
            params = QueryParams(query="test query", top_k=3)

            result = query_documents(params)  # type: ignore

            assert "Error: Query service not initialized" in result  # nosec B101

    def test_query_documents_tool_exception(self) -> None:
        """Test query tool when exception occurs."""
        mock_handler = Mock()
        mock_handler.query_documents.side_effect = Exception("Test error")

        with patch('src.mcp.server.query_handler', mock_handler):
            from src.mcp.server import QueryParams
            params = QueryParams(query="test query", top_k=3)

            result = query_documents(params)  # type: ignore

            assert "Error: Query processing failed: Test error" in result  # nosec B101

    def test_get_health_status_resource(self) -> None:
        """Test health status resource."""
        mock_handler = Mock()
        mock_handler.index = Mock()  # Simulate loaded index

        with patch('src.mcp.server.query_handler', mock_handler):
            result = get_health_status()  # type: ignore

            assert '"status": "healthy"' in result  # nosec B101
            assert '"index_loaded": true' in result  # nosec B101
            assert '"version": "1.0.0"' in result  # nosec B101

    def test_get_health_status_no_index(self) -> None:
        """Test health status when index is not loaded."""
        mock_handler = Mock()
        mock_handler.index = None

        with patch('src.mcp.server.query_handler', mock_handler):
            result = get_health_status()  # type: ignore

            assert '"status": "degraded"' in result  # nosec B101
            assert '"index_loaded": false' in result  # nosec B101

    def test_get_system_context_resource(self) -> None:
        """Test system context resource."""
        result = get_system_context()  # type: ignore

        assert "VX-RAG is a Retrieval-Augmented Generation system" in result  # nosec B101
        assert "Semantic document search" in result  # nosec B101
        assert "PDF" in result  # nosec B101

    async def test_mcp_server_registration(self) -> None:
        """Test that MCP server has expected tools and resources."""
        # Check that tools are registered
        tools = await mcp.get_tools()
        assert len(tools) > 0  # nosec B101
        assert "query_documents" in [tool.name for tool in tools.values()]  # nosec B101

        # Check that resources are registered
        resources = await mcp.get_resource_templates()
        assert len(resources) > 0  # nosec B101

        resources_static = await mcp.get_resources()
        assert len(resources_static) > 0  # nosec B101
