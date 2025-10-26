"""
FastMCP server module for VX-RAG system.

Provides MCP (Model Context Protocol) interface for querying the RAG system.
"""

import logging
from typing import List, Optional, Dict, Any

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from ..rag.query import DocumentQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global query handler
query_handler: Optional[DocumentQuery] = None


def initialize_query_handler():
    """Initialize the document query handler."""
    global query_handler
    if query_handler is None:
        logger.info("Initializing VX-RAG query handler")
        query_handler = DocumentQuery()
        if not query_handler.load_index():
            logger.warning("Failed to load index. Queries may not work.")
    return query_handler


# Create FastMCP server
mcp = FastMCP(
    name="VX-RAG MCP Server"
)


# Pydantic models for tool parameters
class QueryParams(BaseModel):
    """Parameters for document query tool."""
    query: str = Field(..., description="The search query")
    top_k: int = Field(3, ge=1, le=10, description="Number of top results to return")


@mcp.tool
def query_documents(params: QueryParams) -> str:
    """
    Query the RAG system for relevant documents and generate a response.
    
    Performs semantic search on the document index and uses LLM to generate
    a contextual response based on the most relevant documents.
    
    Args:
        params: Query parameters including query text and number of results
        
    Returns:
        Formatted response with query results and LLM-generated answer
    """
    global query_handler
    
    # Initialize if needed
    if query_handler is None:
        initialize_query_handler()
    
    if query_handler is None:
        return "Error: Query service not initialized"
    
    try:
        logger.info(f"Processing MCP query: {params.query} (top_k={params.top_k})")
        
        # Perform query
        response, results = query_handler.query_documents(params.query, params.top_k)
        
        if not response and not results:
            return "Error: Query execution failed"
        
        # Format results
        formatted_results = []
        for i, (text, score, metadata) in enumerate(results, 1):
            formatted_results.append(f"Result {i} (score: {score:.3f}):\n{text[:500]}...")
        
        result_text = "\n\n".join(formatted_results)
        
        full_response = f"Query: {params.query}\n\nLLM Response:\n{response}\n\nTop Results:\n{result_text}"
        
        logger.info(f"MCP query completed: {len(results)} results returned")
        
        return full_response
        
    except Exception as e:
        logger.error(f"MCP query failed: {e}")
        return f"Error: Query processing failed: {str(e)}"


@mcp.resource("health://status")
def get_health_status() -> str:
    """
    Get the health status of the RAG system.
    
    Returns:
        JSON-formatted health status including index load state
    """
    global query_handler
    
    if query_handler is None:
        initialize_query_handler()
    
    index_loaded = query_handler is not None and query_handler.index is not None
    status = "healthy" if index_loaded else "degraded"
    
    import json
    health_data = {
        "status": status,
        "index_loaded": index_loaded,
        "version": "1.0.0"
    }
    
    return json.dumps(health_data, indent=2)


@mcp.resource("context://system")
def get_system_context() -> str:
    """
    Get information about the RAG system's capabilities and context.
    
    Returns:
        JSON-formatted system information
    """
    import json
    
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


if __name__ == "__main__":
    logger.info("Starting VX-RAG MCP server")
    mcp.run()