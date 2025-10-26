"""
Tests for MCP server module.
"""

import pytest
from unittest.mock import Mock, patch
from src.mcp.server import mcp, query_documents, get_health_status, get_system_context, initialize_query_handler


class TestMCPServer:
    """Test cases for MCP server functionality."""

    def test_query_documents_tool_success(self):
        """Test successful document query tool execution."""
        # Mock query handler
        mock_handler = Mock()
        mock_handler.query_documents.return_value = ("Test response", [("text1", 0.9, {"meta": "data"})])

        with patch('src.mcp.server.query_handler', mock_handler):
            from src.mcp.server import QueryParams
            params = QueryParams(query="test query", top_k=3)

            result = query_documents(params)

            assert "Query: test query" in result
            assert "Test response" in result
            assert "Result 1" in result
            mock_handler.query_documents.assert_called_once_with("test query", 3)

    def test_query_documents_tool_no_handler(self):
        """Test query tool when handler is not initialized."""
        with patch('src.mcp.server.query_handler', None):
            from src.mcp.server import QueryParams
            params = QueryParams(query="test query", top_k=3)

            result = query_documents(params)

            assert "Error: Query service not initialized" in result

    def test_query_documents_tool_exception(self):
        """Test query tool when exception occurs."""
        mock_handler = Mock()
        mock_handler.query_documents.side_effect = Exception("Test error")

        with patch('src.mcp.server.query_handler', mock_handler):
            from src.mcp.server import QueryParams
            params = QueryParams(query="test query", top_k=3)

            result = query_documents(params)

            assert "Error: Query processing failed: Test error" in result

    def test_get_health_status_resource(self):
        """Test health status resource."""
        mock_handler = Mock()
        mock_handler.index = Mock()  # Simulate loaded index

        with patch('src.mcp.server.query_handler', mock_handler):
            result = get_health_status()

            assert '"status": "healthy"' in result
            assert '"index_loaded": true' in result
            assert '"version": "1.0.0"' in result

    def test_get_health_status_no_index(self):
        """Test health status when index is not loaded."""
        mock_handler = Mock()
        mock_handler.index = None

        with patch('src.mcp.server.query_handler', mock_handler):
            result = get_health_status()

            assert '"status": "degraded"' in result
            assert '"index_loaded": false' in result

    def test_get_system_context_resource(self):
        """Test system context resource."""
        result = get_system_context()

        assert "VX-RAG is a Retrieval-Augmented Generation system" in result
        assert "Semantic document search" in result
        assert "PDF" in result

    def test_initialize_query_handler(self):
        """Test query handler initialization."""
        with patch('src.mcp.server.query_handler', None):
            with patch('src.mcp.server.DocumentQuery') as mock_doc_query:
                mock_instance = Mock()
                mock_instance.load_index.return_value = True
                mock_doc_query.return_value = mock_instance

                handler = initialize_query_handler()

                assert handler is mock_instance
                mock_doc_query.assert_called_once()
                mock_instance.load_index.assert_called_once()

    def test_mcp_server_registration(self):
        """Test that MCP server has expected tools and resources."""
        # Check that tools are registered
        tools = mcp.get_tools()
        assert len(tools) > 0
        assert "query_documents" in [tool.name for tool in tools.values()]

        # Check that resources are registered
        resources = mcp.get_resource_templates()
        assert len(resources) > 0

        resources_static = mcp.get_resources()
        assert len(resources_static) > 0