"""
MCP Handlers - High-level tool handlers for MCP server.

This module implements the business logic for MCP tools exposed to LLMs.
Each handler is responsible for:
- Validating input parameters
- Delegating work to the RAG orchestrator
- Formatting responses for LLM consumption
- Error handling and logging

These handlers represent the PUBLIC API that LLMs interact with.
All administrative/utility operations have been moved to CLI.
"""

import time
import uuid
from typing import Dict, Any, Optional, Callable

from typing import TYPE_CHECKING

from .schemas import (
    QueryKnowledgeBaseRequest,
    SearchDocumentsRequest,
)
from .formatters import (
    format_query_response,
    format_search_response,
    format_health_status,
    format_system_context,
    format_error_response,
)

from src.utils.logging_config import get_logger, log_query_event
from src.utils.metrics import get_metrics
from src.utils.config_loader import get_mcp_defaults

if TYPE_CHECKING:
    from src.rag.orchestrator import RAGOrchestrator

logger = get_logger("mcp_handlers")
metrics = get_metrics()

# Load MCP default configuration
mcp_defaults = get_mcp_defaults()


def _get_orchestrator() -> "RAGOrchestrator":
    """Lazy import of orchestrator to avoid circular imports and slow startup."""
    from src.rag.orchestrator import get_orchestrator
    return get_orchestrator()


async def handle_query_knowledge_base(params: QueryKnowledgeBaseRequest) -> str:
    """
    Handle knowledge base query - the primary tool for LLMs.
    
    Executes a complete RAG pipeline: retrieval, reranking, context assembly.
    Returns formatted context optimized for LLM consumption.
    
    Args:
        params: Query parameters (query, top_k, search_type, token_budget)
        
    Returns:
        JSON-formatted response with context and sources
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        "Handling knowledge base query",
        request_id=request_id,
        query=params.query,
        top_k=params.top_k,
        search_type=params.search_type,
        token_budget=params.token_budget
    )
    
    try:
        # Get orchestrator instance (lazy import)
        orchestrator = _get_orchestrator()
        
        # Execute query pipeline using Workflow
        rag_result = await orchestrator.query_async(
            query=params.query,
            top_k=params.top_k,
            search_type=params.search_type,
            token_budget=params.token_budget
        )
        
        # Format response
        formatted_response = format_query_response(
            rag_result=rag_result,
            apply_redaction=mcp_defaults['apply_redaction']
        )
        
        # Log metrics
        duration = time.time() - start_time
        results_count = rag_result.get('sources_count', 0)
        
        log_query_event(
            query=params.query,
            top_k=params.top_k,
            results_count=results_count,
            duration=duration
        )
        
        metrics.increment("mcp_query_knowledge_base_total")
        metrics.histogram("mcp_query_knowledge_base_duration_ms", duration * 1000)
        
        logger.info(
            "Knowledge base query completed",
            request_id=request_id,
            results=results_count,
            duration_ms=duration * 1000
        )
        
        return formatted_response
        
    except RuntimeError as e:
        # Service initialization or availability errors
        duration = time.time() - start_time
        error_msg = f"System not ready: {str(e)}"
        
        logger.error(
            "Knowledge base query failed - system not ready",
            request_id=request_id,
            error=str(e),
            duration_ms=duration * 1000
        )
        
        metrics.increment("mcp_query_knowledge_base_errors_total")
        
        return format_error_response(
            error_message=error_msg,
            error_type="service_unavailable",
            details={
                "request_id": request_id,
                "query": params.query
            }
        )
        
    except Exception as e:
        # Unexpected errors
        duration = time.time() - start_time
        error_msg = f"Query processing failed: {str(e)}"
        
        logger.error(
            "Knowledge base query failed",
            request_id=request_id,
            query=params.query,
            error=str(e),
            duration_ms=duration * 1000,
            exc_info=True
        )
        
        metrics.increment("mcp_query_knowledge_base_errors_total")
        
        return format_error_response(
            error_message=error_msg,
            error_type="internal_error",
            details={
                "request_id": request_id,
                "query": params.query
            }
        )


async def handle_search_documents(params: SearchDocumentsRequest) -> str:
    """
    Handle document search - simple search without context assembly.
    
    Returns raw search results for exploration and discovery.
    Useful when LLM wants to browse available documents or explore topics.
    
    Args:
        params: Search parameters (query, top_k, search_type)
        
    Returns:
        JSON-formatted response with search results
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    logger.info(
        "Handling document search",
        request_id=request_id,
        query=params.query,
        top_k=params.top_k,
        search_type=params.search_type
    )
    
    try:
        # Get orchestrator instance (lazy import)
        orchestrator = _get_orchestrator()
        
        # Execute search
        results = orchestrator.search_documents(
            query=params.query,
            top_k=params.top_k,
            search_type=params.search_type
        )
        
        # Format response
        formatted_response = format_search_response(
            query=params.query,
            results=results,
            search_type=params.search_type,
            apply_redaction=mcp_defaults['apply_redaction']
        )
        
        # Log metrics
        duration = time.time() - start_time
        
        metrics.increment("mcp_search_documents_total")
        metrics.histogram("mcp_search_documents_duration_ms", duration * 1000)
        
        logger.info(
            "Document search completed",
            request_id=request_id,
            results=len(results),
            duration_ms=duration * 1000
        )
        
        return formatted_response
        
    except RuntimeError as e:
        # Service initialization or availability errors
        duration = time.time() - start_time
        error_msg = f"System not ready: {str(e)}"
        
        logger.error(
            "Document search failed - system not ready",
            request_id=request_id,
            error=str(e),
            duration_ms=duration * 1000
        )
        
        metrics.increment("mcp_search_documents_errors_total")
        
        return format_error_response(
            error_message=error_msg,
            error_type="service_unavailable",
            details={
                "request_id": request_id,
                "query": params.query
            }
        )
        
    except Exception as e:
        # Unexpected errors
        duration = time.time() - start_time
        error_msg = f"Search failed: {str(e)}"
        
        logger.error(
            "Document search failed",
            request_id=request_id,
            query=params.query,
            error=str(e),
            duration_ms=duration * 1000,
            exc_info=True
        )
        
        metrics.increment("mcp_search_documents_errors_total")
        
        return format_error_response(
            error_message=error_msg,
            error_type="internal_error",
            details={
                "request_id": request_id,
                "query": params.query
            }
        )


async def handle_health_check() -> str:
    """
    Handle health check request.
    
    Returns system health status including service availability.
    Useful for LLMs to understand system readiness before making queries.
    
    Returns:
        JSON-formatted health status
    """
    logger.info("Handling health check")
    
    try:
        # Get orchestrator instance (lazy import)
        orchestrator = _get_orchestrator()
        
        # Get health status
        health_data = orchestrator.get_health_status()
        
        # Format response
        formatted_response = format_health_status(health_data)
        
        logger.info(
            "Health check completed",
            status=health_data.get('overall_status', 'unknown')
        )
        
        metrics.increment("mcp_health_checks_total")
        
        return formatted_response
        
    except Exception as e:
        # Even health check can fail if orchestrator isn't initialized
        logger.error(
            "Health check failed",
            error=str(e),
            exc_info=True
        )
        
        return format_error_response(
            error_message=f"Health check failed: {str(e)}",
            error_type="internal_error"
        )


async def handle_get_system_context() -> str:
    """
    Handle system context request.
    
    Returns information about system capabilities, supported features,
    and corpus information. Helps LLMs understand what the system can do.
    
    Returns:
        JSON-formatted system context
    """
    logger.info("Handling system context request")
    
    try:
        # Format system context (static information)
        formatted_response = format_system_context()
        
        logger.info("System context request completed")
        
        metrics.increment("mcp_system_context_requests_total")
        
        return formatted_response
        
    except Exception as e:
        logger.error(
            "System context request failed",
            error=str(e),
            exc_info=True
        )
        
        return format_error_response(
            error_message=f"System context request failed: {str(e)}",
            error_type="internal_error"
        )

# Map of tool names to handler functions
TOOL_HANDLERS: Dict[str, Callable[..., Any]] = {
    "query_knowledge_base": handle_query_knowledge_base,
    "search_documents": handle_search_documents,
    "health_check": handle_health_check,
    "get_system_context": handle_get_system_context,
}


def get_handler(tool_name: str) -> Optional[Callable[..., Any]]:
    """
    Get handler function for a tool by name.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Handler function or None if not found
    """
    return TOOL_HANDLERS.get(tool_name)
