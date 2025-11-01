"""
MCP Bridge module for VX-RAG.

This module acts as the intermediary between external LLMs and the RAG engine.
Implements the Model Context Protocol for managing context and queries.
Provides a clean interface for MCP server operations.
"""

import logging
import json
from typing import Dict, Any, Optional
import re
from datetime import datetime

from ..rag.services.vectordb_service.service import VectorStoreClient
from ..rag.services.retriever_service.service import RetrieverService

# Import structured logging and metrics
from src.utils.logging_config import get_logger, log_index_event, log_service_health
from src.utils.metrics import get_metrics

logger = get_logger("mcp_bridge")
metrics = get_metrics()


def redact_sensitive_data(text: str) -> str:
    """
    Redact sensitive data from text according to VX-RAG standards.
    
    Automatically redacts emails and IP addresses. Does not redact names or code samples.
    
    Args:
        text: The text to redact
        
    Returns:
        Text with sensitive data redacted
    """
    # Redact email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text = re.sub(email_pattern, '[REDACTED_EMAIL]', text)
    
    # Redact IP addresses
    ip_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
    text = re.sub(ip_pattern, '[REDACTED_IP]', text)
    
    return text


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
                    log_service_health("vector_store", "loaded")
                    log_service_health("retriever", "initialized")
                else:
                    logger.warning("Index loaded but index object is None")
                    log_service_health("vector_store", "error", error="index_object_none")
            else:
                logger.warning("Failed to load index. Retriever service not initialized.")
                log_service_health("vector_store", "error", error="load_failed")
            
        except Exception as e:
            logger.error("Failed to initialize RAG services", error=str(e))
            log_service_health("bridge_initialization", "error", error=str(e))

    def query_documents(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Query the RAG system for relevant documents.
        
        Args:
            query: The search query string
            top_k: Number of top results to return
            
        Returns:
            Structured JSON response with query results and sources
        """
        import time
        start_time = time.time()
        
        if self.retriever is None:
            logger.error("Query failed: RAG services not initialized", query=query)
            return {
                "error": "RAG services not initialized",
                "query": query,
                "sources": []
            }
        
        try:
            logger.info("Processing query via MCP bridge", query=query, top_k=top_k)
            
            # Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(query, top_k)
            
            duration = time.time() - start_time
            
            if not retrieved_docs:
                logger.info("Query completed: no relevant documents found", 
                           query=query, results_count=0, duration_ms=duration * 1000)
                return {
                    "query": query,
                    "response": "No relevant documents found for the query.",
                    "sources": []
                }
            
            # Format sources with redaction
            sources = []
            context_parts = []
            for i, source in enumerate(retrieved_docs, 1):
                redacted_text = redact_sensitive_data(source['text'])
                source_info = {
                    "id": i,
                    "score": round(source['score'], 3),
                    "text": redacted_text,  # Redacted text
                    "metadata": source.get('metadata', {})
                }
                sources.append(source_info)
                context_parts.append(f"Source {i} (score: {source['score']:.3f}):\n{redacted_text}")
            
            # Combine redacted context for response
            full_context = "\n\n".join(context_parts)
            
            # Redact the response
            response = redact_sensitive_data(f"Based on the retrieved documents:\n\n{full_context}")
            
            result = {
                "query": query,
                "response": response,
                "sources": sources
            }
            
            # Log metrics
            results_count = len(retrieved_docs)
            metrics.increment("bridge_queries_total")
            metrics.histogram("bridge_query_duration_ms", duration * 1000)
            metrics.gauge("bridge_query_results_count", results_count)
            
            logger.info("Query completed via MCP bridge", 
                       query=query, 
                       results_count=results_count, 
                       duration_ms=duration * 1000)
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error("Query failed in MCP bridge", 
                        query=query, 
                        error=str(e), 
                        duration_ms=duration * 1000)
            return {
                "error": f"Query processing failed: {str(e)}",
                "query": query,
                "sources": []
            }

    def get_health_status(self) -> str:
        """
        Get the health status of the RAG system.
        
        Returns:
            JSON-formatted health status including service states
        """
        index_loaded = self.vector_store is not None and self.vector_store.index is not None
        retriever_ready = self.retriever is not None
        
        status = "healthy" if (index_loaded and retriever_ready) else "degraded"
        
        health_data = {
            "status": status,
            "services": {
                "vector_store": {
                    "initialized": self.vector_store is not None,
                    "index_loaded": index_loaded
                },
                "retriever": {
                    "initialized": retriever_ready
                }
            },
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat() + "Z"
        }
        
        # Log health status
        log_service_health("rag_system", status)
        
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
                "MCP protocol integration for IDE/LLM access"
            ],
            "supported_formats": ["PDF"],
            "services": {
                "ingest": "Document loading and preprocessing",
                "embedder": "Text embedding generation",
                "vectordb": "Vector storage and retrieval",
                "retriever": "Document retrieval"
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
            Retrieved context as JSON string.
        """
        result = self.query_documents(query)
        return json.dumps(result, indent=2, ensure_ascii=False)


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
