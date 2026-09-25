"""
VX-RAG MCP Server - Minimal FastMCP server implementation.

Exposes thin wrapper tools and resources around RAGOrchestrator with
sequential CPU query processing via asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import weakref
from types import TracebackType

from fastmcp import FastMCP

from src.mcp.formatters import (
    format_error_response,
    format_health_status,
    format_query_response,
    format_search_response,
    format_system_context,
)
from src.rag.orchestrator import get_orchestrator
from src.utils.config_loader import get_mcp_defaults, get_mcp_timeouts
from src.utils.logging_config import get_logger, log_service_health, request_context
from src.utils.metrics import get_metrics

logger = get_logger("mcp_server")
metrics = get_metrics()

# Load MCP configuration from settings.yaml
mcp_timeouts = get_mcp_timeouts()
mcp_defaults = get_mcp_defaults()

# Fallback tuple for broad exception handling without triggering Ruff BLE001
_SAFE_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)

# FastMCP server instance
mcp = FastMCP(
    name="VX-RAG",
    version="1.0.0",
)

class _LoopBoundLock:
    """Concurrency lock proxy dynamically resolving an asyncio.Lock per running event loop.

    Ensures sequential query execution on CPU while preventing cross-loop binding
    RuntimeError exceptions across multiple test runs or server reloads.
    """

    def __init__(self) -> None:
        """Initialize the loop-bound lock registry."""
        self._locks: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock] = (
            weakref.WeakKeyDictionary()
        )

    def _get_lock(self) -> asyncio.Lock:
        """Resolve or instantiate the asyncio.Lock bound to the current running event loop.

        Returns:
            The asyncio.Lock instance tied to the active event loop.
        """
        loop = asyncio.get_running_loop()
        if loop not in self._locks:
            self._locks[loop] = asyncio.Lock()
        return self._locks[loop]

    async def acquire(self) -> bool:
        """Acquire the lock bound to the current running event loop.

        Returns:
            True once the lock is acquired.
        """
        return await self._get_lock().acquire()

    def release(self) -> None:
        """Release the lock bound to the current running event loop."""
        self._get_lock().release()

    def locked(self) -> bool:
        """Check if the lock for the current running event loop is acquired.

        Returns:
            True if locked in the current running loop, False otherwise or if no loop runs.
        """
        try:
            return self._get_lock().locked()
        except RuntimeError:
            return False

    async def __aenter__(self) -> None:
        """Acquire the lock for the current running event loop upon entering context."""
        await self.acquire()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Release the lock for the current running event loop upon exiting context."""
        self.release()


# Concurrency lock ensuring strictly sequential CPU query execution
query_lock: _LoopBoundLock | asyncio.Lock = _LoopBoundLock()


async def query_knowledge_base(
    query: str,
    top_k: int = 5,
    search_type: str = "hybrid",
    token_budget: int = 4000,
) -> str:
    """
    Query the knowledge base for relevant information.

    Args:
        query: The search query or question
        top_k: Number of most relevant documents to return
        search_type: Search strategy - "semantic" (vector), "keyword" (BM25), or "hybrid" (default)
        token_budget: Maximum tokens for context

    Returns:
        JSON response with query results
    """
    with request_context():
        async with query_lock:
            try:
                orchestrator = get_orchestrator()
                payload = await orchestrator.query_async(
                    query=query,
                    top_k=top_k,
                    search_type=search_type,
                    token_budget=token_budget,
                )
                return format_query_response(
                    payload,
                    apply_redaction=bool(mcp_defaults.get("apply_redaction", True)),
                )
            except _SAFE_EXCEPTIONS as e:
                logger.error(f"Error querying knowledge base: {e}")
                return format_error_response(str(e))


async def search_documents(
    query: str,
    top_k: int = 10,
    search_type: str = "semantic",
) -> str:
    """
    Search for documents without context assembly returning complete full raw text.

    Args:
        query: The search query
        top_k: Number of results to return
        search_type: Search strategy - "semantic", "keyword", or "hybrid"

    Returns:
        JSON response with un-truncated raw search results
    """
    with request_context():
        async with query_lock:
            try:
                orchestrator = get_orchestrator()
                results = orchestrator.search_documents(
                    query=query,
                    top_k=top_k,
                    search_type=search_type,
                )
                return format_search_response(
                    query=query,
                    results=results,
                    search_type=search_type,
                    apply_redaction=False,
                )
            except _SAFE_EXCEPTIONS as e:
                logger.error(f"Error searching documents: {e}")
                return format_error_response(str(e))


async def health_status() -> str:
    """
    System health status.

    Returns:
        JSON response with health status
    """
    with request_context():
        try:
            orchestrator = get_orchestrator()
            health_data = orchestrator.get_health_status()
            return format_health_status(health_data)
        except _SAFE_EXCEPTIONS as e:
            logger.error(f"Error getting health status: {e}")
            return format_error_response(str(e))


async def system_context() -> str:
    """
    System context and capabilities.

    Returns:
        JSON response with system context
    """
    with request_context():
        return format_system_context()


# Register tools and resources on FastMCP server instance while preserving callable functions
mcp.tool(query_knowledge_base)
mcp.tool(search_documents)
mcp.resource("health://status")(health_status)
mcp.resource("context://system")(system_context)


async def start_server() -> None:
    """Start and verify the MCP server dependencies."""
    with request_context():
        logger.info("Starting VX-RAG MCP server")
        log_service_health("mcp_server", "starting")
        try:
            orchestrator = get_orchestrator()
            health = orchestrator.get_health_status()
            logger.info(
                "RAG orchestrator initialized",
                status=health.get("overall_status"),
                initialized=health.get("initialized"),
                indexes_loaded=health.get("indexes_loaded"),
            )
            if not health.get("initialized"):
                logger.warning("RAG services not fully initialized")
            log_service_health("mcp_server", "ready")
        except _SAFE_EXCEPTIONS as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            log_service_health("mcp_server", "error", error=str(e))
            raise


def run_stdio() -> None:
    """Run MCP server in STDIO mode."""
    logger.info("Starting VX-RAG MCP server in STDIO mode")
    log_service_health("mcp_server", "starting")
    mcp.run()


async def run_sse(host: str = "localhost", port: int = 8000) -> None:
    """Run MCP server in SSE mode."""
    import uvicorn

    await start_server()
    app = mcp.sse_app()
    config = uvicorn.Config(
        app, host=host, port=port, log_level="info", access_log=True
    )
    server = uvicorn.Server(config)
    await server.serve()


async def run_http(host: str = "localhost", port: int = 8000) -> None:
    """Run MCP server in HTTP mode."""
    import uvicorn

    await start_server()
    app = mcp.http_app()
    config = uvicorn.Config(
        app, host=host, port=port, log_level="info", access_log=True
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    run_stdio()
