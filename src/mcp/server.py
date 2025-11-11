"""
VX-RAG MCP Server - Minimal FastMCP server implementation.

This is a thin MCP protocol layer that delegates all business logic to handlers.
The server's only responsibilities are:
1. Register MCP tools with FastMCP
2. Route tool calls to appropriate handlers
3. Provide MCP resources (health, context)

ALL business logic is in handlers.py
ALL formatting is in formatters.py
ALL RAG operations are in orchestrator.py

Administrative operations (ingestion, indexing, snapshots, tasks) are CLI-only.
"""

import asyncio
from typing import Any
from fastmcp import FastMCP

from .schemas import (
    QueryKnowledgeBaseRequest,
    SearchDocumentsRequest,
)
from .handlers import (
    handle_query_knowledge_base,
    handle_search_documents,
    handle_health_check,
    handle_get_system_context,
)
from .middleware import with_mcp_middleware

from src.utils.logging_config import get_logger, log_service_health
from src.utils.metrics import get_metrics
from src.utils.config_loader import get_mcp_timeouts, get_mcp_defaults

logger = get_logger("mcp_server")
metrics = get_metrics()

# Load MCP configuration from settings.yaml
mcp_timeouts = get_mcp_timeouts()
mcp_defaults = get_mcp_defaults()


# Create FastMCP server instance
mcp = FastMCP(
    name="VX-RAG",
    version="1.0.0"
)


@mcp.tool()
@with_mcp_middleware("query_knowledge_base", timeout=mcp_timeouts['query_knowledge_base'])  # type: ignore[misc]
async def query_knowledge_base(
    query: str,
    top_k: int = mcp_defaults['top_k'],
    search_type: str = "hybrid",
    token_budget: int = mcp_defaults['token_budget']
) -> str:
    """
    Query the knowledge base for relevant information.
    
    This is the primary tool for retrieving contextual information from the
    VX Underground document corpus. It performs semantic search, reranking,
    and context assembly to provide the most relevant information.
    
    Args:
        query: The search query or question
        top_k: Number of most relevant documents to return (1-20, default: 5)
        search_type: Search strategy - "semantic" (vector), "keyword" (BM25), or "hybrid" (default)
        token_budget: Maximum tokens for context (500-16000, default: 4000)
        
    Returns:
        JSON response with query results, context, sources, and statistics
    """
    params = QueryKnowledgeBaseRequest(
        query=query,
        top_k=top_k,
        search_type=search_type,
        token_budget=token_budget
    )
    return await handle_query_knowledge_base(params)


@mcp.tool()
@with_mcp_middleware("search_documents", timeout=mcp_timeouts['search_documents'])  # type: ignore[misc]
async def search_documents(
    query: str,
    top_k: int = mcp_defaults['search_top_k'],
    search_type: str = "semantic"
) -> str:
    """
    Search for documents without context assembly.
    
    Returns raw search results for exploration and discovery. Useful when
    browsing available documents or exploring topics without needing
    full context assembly.
    
    Args:
        query: The search query
        top_k: Number of results to return (1-50, default: 10)
        search_type: Search strategy - "semantic", "keyword", or "hybrid" (default: "semantic")
        
    Returns:
        JSON response with search results
    """
    params = SearchDocumentsRequest(
        query=query,
        top_k=top_k,
        search_type=search_type
    )
    return await handle_search_documents(params)


@mcp.resource("health://status")
async def health_status() -> str:
    """
    System health status.
    
    Provides information about system readiness, service availability,
    and index loading status.
    
    Returns:
        JSON response with health status
    """
    return await handle_health_check()


@mcp.resource("context://system")
async def system_context() -> str:
    """
    System context and capabilities.
    
    Describes the system's capabilities, supported features, and corpus
    information. Helps LLMs understand what the system can do.
    
    Returns:
        JSON response with system context
    """
    return await handle_get_system_context()


async def start_server() -> None:
    """
    Start the MCP server.
    
    Initializes the RAG orchestrator and starts the FastMCP server.
    """
    logger.info("Starting VX-RAG MCP server")
    log_service_health("mcp_server", "starting")
    
    try:
        # Pre-initialize orchestrator to fail fast if there are issues
        from src.rag.orchestrator import get_orchestrator
        
        orchestrator = get_orchestrator()
        health = orchestrator.get_health_status()
        
        logger.info(
            "RAG orchestrator initialized",
            status=health.get('overall_status'),
            initialized=health.get('initialized'),
            indexes_loaded=health.get('indexes_loaded')
        )
        
        if not health.get('initialized'):
            logger.warning("RAG services not fully initialized - some features may be unavailable")
        
        log_service_health("mcp_server", "ready")
        
        # Server is ready - FastMCP will handle the actual serving
        logger.info("MCP server ready to accept connections")
        
    except Exception as e:
        logger.error(
            "Failed to initialize MCP server",
            error=str(e),
            exc_info=True
        )
        log_service_health("mcp_server", "error", error=str(e))
        raise


def run_stdio() -> None:
    """
    Run MCP server in STDIO mode (for IDE integration).
    
    This is the default mode for MCP servers integrated into IDEs.
    """
    # Initialize orchestrator before serving
    asyncio.run(start_server())
    
    # Run FastMCP server in STDIO mode
    mcp.run()


async def run_sse(host: str = "localhost", port: int = 8000) -> None:
    """
    Run MCP server in SSE (Server-Sent Events) mode.
    
    Args:
        host: Server host
        port: Server port
    """
    import uvicorn
    
    # Initialize orchestrator
    await start_server()
    
    # Get FastAPI app with SSE support
    app = mcp.sse_app()
    
    # Run with uvicorn
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_http(host: str = "localhost", port: int = 8000) -> None:
    """
    Run MCP server in HTTP REST API mode.
    
    Args:
        host: Server host
        port: Server port
    """
    import uvicorn
    
    # Initialize orchestrator
    await start_server()
    
    # Get FastAPI app with HTTP support
    app = mcp.http_app()
    
    # Run with uvicorn
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # Default: run in STDIO mode for IDE integration
    run_stdio()
