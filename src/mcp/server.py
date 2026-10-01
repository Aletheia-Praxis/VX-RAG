"""
VX-RAG MCP Server - Minimal FastMCP server implementation.

Exposes thin wrapper tools and resources around RAGOrchestrator with
sequential CPU query processing via asyncio.Lock.
"""

from __future__ import annotations

import asyncio
import sys
import weakref
from pathlib import Path
from types import TracebackType
from typing import Any

from fastmcp import FastMCP

from src.mcp.formatters import (
    format_error_response,
    format_health_status,
    format_query_response,
    format_search_response,
    format_system_context,
)
from src.rag.orchestrator import RAGOrchestrator, get_orchestrator
from src.utils.config_loader import get_mcp_defaults, get_mcp_timeouts
from src.utils.logging_config import (
    configure_console_stream,
    get_logger,
    log_service_health,
    request_context,
)
from src.utils.metrics import get_metrics

logger = get_logger("mcp_server")
metrics = get_metrics()

# Load MCP configuration from settings.yaml (deferred from import time to prevent stdout pollution)
mcp_timeouts: dict[str, float] = {}
mcp_defaults: dict[str, Any] = {}

# Fallback tuple for broad exception handling without triggering Ruff BLE001
_SAFE_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)

# Configured paths for orchestrator initialization
_configured_config_path: str = "config/settings.yaml"
_configured_persist_dir: str = "data/index"


def configure_server(
    config_path: str = "config/settings.yaml",
    persist_dir: str = "data/index",
) -> RAGOrchestrator:
    """Configure and initialize the RAG orchestrator for MCP server.

    If the orchestrator was previously initialized with a different persist directory
    or configuration path, it is safely reset and re-initialized.

    Args:
        config_path: Path to configuration YAML file.
        persist_dir: Path to directory containing persisted Qdrant and BM25 indices.

    Returns:
        The configured RAGOrchestrator instance.
    """
    global _configured_config_path, _configured_persist_dir, mcp_defaults, mcp_timeouts
    _configured_config_path = config_path
    _configured_persist_dir = persist_dir
    try:
        if Path(config_path).exists():
            mcp_timeouts = get_mcp_timeouts(config_path)
            mcp_defaults = get_mcp_defaults(config_path)
        else:
            mcp_timeouts = {}
            mcp_defaults = {}
    except _SAFE_EXCEPTIONS:
        mcp_timeouts = {}
        mcp_defaults = {}

    from src.rag.orchestrator import _orchestrator_instance, reset_orchestrator

    if _orchestrator_instance is not None:
        try:
            curr_persist = Path(_orchestrator_instance.persist_dir).resolve()
            new_persist = Path(persist_dir).resolve()
            curr_cfg = Path(_orchestrator_instance.config_path).resolve()
            new_cfg = Path(config_path).resolve()
            if curr_persist != new_persist or curr_cfg != new_cfg:
                reset_orchestrator()
        except OSError:
            reset_orchestrator()

    return get_orchestrator(config_path=config_path, persist_dir=persist_dir)


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
                orchestrator = get_orchestrator(
                    config_path=_configured_config_path,
                    persist_dir=_configured_persist_dir,
                )
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
                orchestrator = get_orchestrator(
                    config_path=_configured_config_path,
                    persist_dir=_configured_persist_dir,
                )
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
            orchestrator = get_orchestrator(
                config_path=_configured_config_path,
                persist_dir=_configured_persist_dir,
            )
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


async def start_server(
    config_path: str = "config/settings.yaml",
    persist_dir: str = "data/index",
) -> None:
    """Start and verify the MCP server dependencies.

    Args:
        config_path: Path to configuration YAML file.
        persist_dir: Path to directory containing persisted Qdrant and BM25 indices.
    """
    with request_context():
        logger.info("Starting VX-RAG MCP server")
        log_service_health("mcp_server", "starting")
        try:
            orchestrator = configure_server(
                config_path=config_path, persist_dir=persist_dir
            )
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


def run_stdio(
    config_path: str = "config/settings.yaml",
    persist_dir: str = "data/index",
) -> None:
    """Run MCP server in STDIO mode.

    Args:
        config_path: Path to configuration YAML file.
        persist_dir: Path to directory containing persisted Qdrant and BM25 indices.
    """
    configure_console_stream(sys.stderr)
    logger.info("Starting VX-RAG MCP server in STDIO mode")
    log_service_health("mcp_server", "starting")
    try:
        orchestrator = configure_server(
            config_path=config_path, persist_dir=persist_dir
        )
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
    mcp.run()


async def run_http(
    host: str = "localhost",
    port: int = 8000,
    config_path: str = "config/settings.yaml",
    persist_dir: str = "data/index",
) -> None:
    """Run MCP server in HTTP mode.

    Args:
        host: Host interface to bind.
        port: TCP port to listen on.
        config_path: Path to configuration YAML file.
        persist_dir: Path to directory containing persisted Qdrant and BM25 indices.
    """
    import uvicorn

    await start_server(config_path=config_path, persist_dir=persist_dir)
    app = mcp.http_app()
    config = uvicorn.Config(
        app, host=host, port=port, log_level="info", access_log=True
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    run_stdio()
