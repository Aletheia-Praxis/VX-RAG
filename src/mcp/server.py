"""
FastMCP server module for VX-RAG system.

Provides MCP (Model Context Protocol) interface for querying the RAG system.
Delegates all RAG operations to the MCP bridge for clean separation of concerns.
"""

import logging
from typing import List, Optional, Dict, Any
import json

from fastmcp import FastMCP
from pydantic import BaseModel, Field
from .bridge import get_mcp_bridge

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global MCP bridge instance
mcp_bridge = get_mcp_bridge()


# Create FastMCP server
mcp = FastMCP(
    name="VX-RAG MCP Server"
)


# Pydantic models for tool parameters
class QueryParams(BaseModel):
    """Parameters for document query tool."""
    query: str = Field(..., description="The search query")
    top_k: int = Field(3, ge=1, le=10, description="Number of top results to return")


@mcp.tool  # type: ignore
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
    try:
        logger.info(f"Processing MCP query: {params.query} (top_k={params.top_k})")
        
        # Delegate to MCP bridge
        response = mcp_bridge.query_documents(params.query, params.top_k)
        
        logger.info("MCP query completed successfully")
        
        return response
        
    except Exception as e:
        logger.error(f"MCP query failed: {e}")
        return f"Error: Query processing failed: {str(e)}"


@mcp.resource("health://status")  # type: ignore
def get_health_status() -> str:
    """
    Get the health status of the RAG system.
    
    Returns:
        JSON-formatted health status including index load state
    """
    try:
        # Delegate to MCP bridge
        return mcp_bridge.get_health_status()
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return json.dumps({"status": "error", "error": str(e)})


@mcp.resource("context://system")  # type: ignore
def get_system_context() -> str:
    """
    Get information about the RAG system's capabilities and context.
    
    Returns:
        JSON-formatted system information
    """
    try:
        # Delegate to MCP bridge
        return mcp_bridge.get_system_context()
    except Exception as e:
        logger.error(f"System context retrieval failed: {e}")
        return json.dumps({"error": f"Failed to retrieve system context: {str(e)}"})


if __name__ == "__main__":
    logger.info("Starting VX-RAG MCP server")
    mcp.run()
