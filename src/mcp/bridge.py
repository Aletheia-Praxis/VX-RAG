"""
MCP Bridge module for VX-RAG.

This module acts as the intermediary between external LLMs and the RAG engine.
Implements the Model Context Protocol for managing context and queries.
Provides a clean interface for MCP server operations.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import json
from datetime import datetime

from ..rag.query import DocumentQuery

logger = logging.getLogger(__name__)


class MCPBridge:
    """
    Bridge class for MCP communication with RAG system.
    
    Handles initialization of RAG components and provides methods
    for querying documents, health checks, and system context.
    """

    def __init__(self):
        """
        Initialize the MCP bridge with RAG query handler.
        """
        self.query_handler: Optional[DocumentQuery] = None
        self._initialize_query_handler()

    def _initialize_query_handler(self):
        """Initialize the document query handler."""
        if self.query_handler is None:
            logger.info("Initializing VX-RAG query handler in MCP bridge")
            self.query_handler = DocumentQuery()
            if not self.query_handler.load_index():
                logger.warning("Failed to load index in MCP bridge. Queries may not work.")
                self.query_handler = None

    def query_documents(self, query: str, top_k: int = 3) -> str:
        """
        Query the RAG system for relevant documents and generate a response.
        
        Args:
            query: The search query string
            top_k: Number of top results to return
            
        Returns:
            Formatted response with query results and LLM-generated answer
        """
        if self.query_handler is None:
            self._initialize_query_handler()
            if self.query_handler is None:
                return "Error: Query service not initialized"
        
        try:
            logger.info(f"Processing query via MCP bridge: {query} (top_k={top_k})")
            
            # Perform query
            response, results = self.query_handler.query_documents(query, top_k)
            
            if not response and not results:
                return "Error: Query execution failed"
            
            # Format results
            formatted_results = []
            for i, (text, score, metadata) in enumerate(results, 1):
                formatted_results.append(f"Result {i} (score: {score:.3f}):\n{text[:500]}...")
            
            result_text = "\n\n".join(formatted_results)
            
            full_response = f"Query: {query}\n\nLLM Response:\n{response}\n\nTop Results:\n{result_text}"
            
            logger.info(f"Query completed via MCP bridge: {len(results)} results returned")
            
            return full_response
            
        except Exception as e:
            logger.error(f"Query failed in MCP bridge: {e}")
            return f"Error: Query processing failed: {str(e)}"

    def get_health_status(self) -> str:
        """
        Get the health status of the RAG system.
        
        Returns:
            JSON-formatted health status including index load state
        """
        if self.query_handler is None:
            self._initialize_query_handler()
        
        index_loaded = self.query_handler is not None and self.query_handler.index is not None
        status = "healthy" if index_loaded else "degraded"
        
        health_data = {
            "status": status,
            "index_loaded": index_loaded,
            "version": "1.0.0"
        }
        
        return json.dumps(health_data, indent=2)

    def get_system_context(self) -> str:
        """
        Get information about the RAG system's capabilities and context.
        
        Returns:
            JSON-formatted system information
        """
        context_data = {
            "description": "VX-RAG is a Retrieval-Augmented Generation system specialized in VX Underground technical documents.",
            "capabilities": [
                "Semantic document search using FAISS vector database",
                "Text extraction from PDF documents",
                "LLM-powered response generation",
                "MCP protocol integration for IDE/LLM access"
            ],
            "supported_formats": ["PDF"]
        }
        
        return json.dumps(context_data, indent=2)

    def handle_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle a query from the external LLM (legacy method for compatibility).
        
        Args:
            query: The query string.
            context: Additional context from the LLM.
            
        Returns:
            Response dictionary with retrieved context and metadata.
        """
        response = self.query_documents(query)
        
        return {
            "query": query,
            "retrieved_context": response,
            "metadata": {
                "source": "vx-rag",
                "timestamp": datetime.now().isoformat() + "Z"
            }
        }

    def get_context(self, query: str) -> str:
        """
        Get context for a query (legacy method for compatibility).
        
        Args:
            query: Query string.
            
        Returns:
            Retrieved context as string.
        """
        return self.query_documents(query)


# Global bridge instance
_bridge_instance: Optional[MCPBridge] = None


def get_mcp_bridge() -> MCPBridge:
    """
    Get or create the global MCP bridge instance.
    
    Returns:
        The MCP bridge instance
    """
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = MCPBridge()
    return _bridge_instance
