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

from ..rag.services.vectordb_service.service import VectorStoreClient
from ..rag.services.retriever_service.service import RetrieverService
from ..rag.services.llm_proxy.service import LLMProxy

logger = logging.getLogger(__name__)


class MCPBridge:
    """
    Bridge class for MCP communication with RAG system.
    
    Handles initialization of RAG components and provides methods
    for querying documents, health checks, and system context.
    """

    def __init__(self) -> None:
        """
        Initialize the MCP bridge with RAG services.
        """
        self.vector_store: Optional[VectorStoreClient] = None
        self.retriever: Optional[RetrieverService] = None
        self.llm_proxy: Optional[LLMProxy] = None
        self._initialize_services()

    def _initialize_services(self) -> None:
        """Initialize the RAG services."""
        try:
            logger.info("Initializing VX-RAG services in MCP bridge")
            
            # Initialize vector store
            self.vector_store = VectorStoreClient(store_type="faiss")
            
            # Load index
            if self.vector_store.load_index():
                # Initialize retriever with loaded index
                if self.vector_store.index is not None:
                    self.retriever = RetrieverService(self.vector_store.index)
                    logger.info("Retriever service initialized with loaded index")
                else:
                    logger.warning("Index loaded but index object is None")
            else:
                logger.warning("Failed to load index. Retriever service not initialized.")
            
            # Initialize LLM proxy
            self.llm_proxy = LLMProxy()
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG services: {e}")

    def query_documents(self, query: str, top_k: int = 3) -> str:
        """
        Query the RAG system for relevant documents and generate a response.
        
        Args:
            query: The search query string
            top_k: Number of top results to return
            
        Returns:
            Formatted response with query results and LLM-generated answer
        """
        if self.retriever is None or self.llm_proxy is None:
            return "Error: RAG services not initialized"
        
        try:
            logger.info(f"Processing query via MCP bridge: {query} (top_k={top_k})")
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query, top_k)
            
            if not retrieved_docs:
                return f"Query: {query}\n\nNo relevant documents found."
            
            # Generate response using LLM
            llm_result = self.llm_proxy.generate_with_sources(query, retrieved_docs)
            
            # Format results
            formatted_results = []
            for i, source in enumerate(llm_result['sources'], 1):
                formatted_results.append(f"Result {i} (score: {source['score']:.3f}):\n{source['text']}")
            
            result_text = "\n\n".join(formatted_results)
            
            full_response = f"Query: {query}\n\nLLM Response:\n{llm_result['response']}\n\nTop Results:\n{result_text}"
            
            logger.info(f"Query completed via MCP bridge: {len(retrieved_docs)} results returned")
            
            return full_response
            
        except Exception as e:
            logger.error(f"Query failed in MCP bridge: {e}")
            return f"Error: Query processing failed: {str(e)}"

    def get_health_status(self) -> str:
        """
        Get the health status of the RAG system.
        
        Returns:
            JSON-formatted health status including service states
        """
        index_loaded = self.vector_store is not None and self.vector_store.index is not None
        retriever_ready = self.retriever is not None
        llm_ready = self.llm_proxy is not None
        
        status = "healthy" if (index_loaded and retriever_ready and llm_ready) else "degraded"
        
        health_data = {
            "status": status,
            "services": {
                "vector_store": {
                    "initialized": self.vector_store is not None,
                    "index_loaded": index_loaded
                },
                "retriever": {
                    "initialized": retriever_ready
                },
                "llm_proxy": {
                    "initialized": llm_ready
                }
            },
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
            "supported_formats": ["PDF"],
            "services": {
                "ingest": "Document loading and preprocessing",
                "embedder": "Text embedding generation",
                "vectordb": "Vector storage and retrieval",
                "retriever": "Document retrieval",
                "llm_proxy": "LLM interaction and response generation"
            }
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
