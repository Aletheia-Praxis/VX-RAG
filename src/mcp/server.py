"""
FastMCP server module for VX-RAG system.

Provides MCP (Model Context Protocol) interface for querying the RAG system.
Delegates all RAG operations to the MCP bridge for clean separation of concerns.
"""

import json
import time
import asyncio
import uuid
from typing import Dict, Any
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from .bridge import get_mcp_bridge

# Import structured logging and metrics
from src.utils.logging_config import get_logger, log_query_event, log_service_health
from src.utils.metrics import get_metrics
from src.utils.config_loader import get_mcp_config
from src.utils.rate_limiter import get_rate_limiter

# Get structured logger and metrics
logger = get_logger("mcp_server")
metrics = get_metrics()

# Global MCP bridge instance
mcp_bridge = get_mcp_bridge()

# Initialize rate limiter for concurrent request control
rate_limiter = get_rate_limiter(
    max_concurrent=2,       # Max 2 concurrent queries
    queue_size=10,          # Queue up to 10 requests
    default_timeout=600.0,  # 10 minutes default timeout
)


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
async def query_documents(params: QueryParams) -> str:
    """
    Query the RAG system for relevant documents and generate a response.
    
    Performs semantic search on the document index and uses LLM to generate
    a contextual response based on the most relevant documents.
    
    Rate limited to max 2 concurrent queries with queue for additional requests.
    
    Args:
        params: Query parameters including query text and number of results
        
    Returns:
        JSON-formatted response with query results, LLM-generated answer, and sources
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Load timeout from config
    mcp_config = get_mcp_config()
    query_timeout = mcp_config.get('query_timeout', 600.0)
    
    try:
        logger.info(
            "Processing MCP query with rate limiting",
            request_id=request_id,
            query=params.query,
            top_k=params.top_k,
        )
        
        # Define query handler
        async def _execute_query() -> Dict[str, Any]:
            return mcp_bridge.query_documents(params.query, params.top_k)
        
        # Execute with rate limiting and timeout
        response = await rate_limiter.execute(
            request_id=request_id,
            handler=_execute_query,
            timeout=query_timeout,
        )
        
        # Convert to JSON string for MCP response
        json_response = json.dumps(response, indent=2, ensure_ascii=False)
        
        duration = time.time() - start_time
        results_count = len(response.get('sources', []))
        
        # Log structured event and metrics
        log_query_event(params.query, params.top_k, results_count, duration)
        
        logger.info(
            "MCP query completed successfully", 
            request_id=request_id,
            query=params.query, 
            results_count=results_count, 
            duration_ms=duration * 1000,
        )
        
        return json_response
        
    except asyncio.TimeoutError:
        duration = time.time() - start_time
        logger.error(
            f"MCP query timed out after {query_timeout}s",
            request_id=request_id,
            query=params.query, 
            timeout=query_timeout,
            duration_ms=duration * 1000,
        )
        error_response = {
            "error": f"Query processing timed out after {query_timeout}s",
            "query": params.query,
            "sources": []
        }
        return json.dumps(error_response, indent=2, ensure_ascii=False)
    
    except RuntimeError as e:
        # Queue full error
        duration = time.time() - start_time
        logger.error(
            "MCP query queue full",
            request_id=request_id,
            query=params.query,
            error=str(e),
            duration_ms=duration * 1000,
        )
        error_response = {
            "error": "Server is busy, request queue is full. Please try again later.",
            "query": params.query,
            "sources": []
        }
        return json.dumps(error_response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "MCP query failed",
            request_id=request_id,
            query=params.query, 
            duration_ms=duration * 1000, 
            error=str(e),
            exc_info=True,
        )
        error_response = {
            "error": f"Query processing failed: {str(e)}",
            "query": params.query,
            "sources": []
        }
        return json.dumps(error_response, indent=2, ensure_ascii=False)


@mcp.resource("health://status")
def get_health_status() -> str:
    """
    Get the health status of the RAG system.
    
    Returns:
        JSON-formatted health status including index load state and rate limiter stats
    """
    try:
        # Delegate to MCP bridge
        health_data = mcp_bridge.get_health_status()
        
        # Parse health data
        health_dict = json.loads(health_data)
        
        # Add rate limiter statistics
        health_dict['rate_limiter'] = rate_limiter.get_stats()
        
        # Log service health
        log_service_health("mcp_server", "healthy")
        
        return json.dumps(health_dict, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        log_service_health("mcp_server", "error", error=str(e))
        return json.dumps({"status": "error", "error": str(e)})


@mcp.resource("context://system")
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
        logger.error("System context retrieval failed", error=str(e))
        return json.dumps({"error": f"Failed to retrieve system context: {str(e)}"})


if __name__ == "__main__":
    logger.info("Starting VX-RAG MCP server")
    log_service_health("mcp_server", "starting")
    mcp.run()
