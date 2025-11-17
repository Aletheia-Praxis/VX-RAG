"""
Router Query Engine - Advanced query routing for VX-RAG.

Provides intelligent query routing capabilities using LlamaIndex RouterQueryEngine.
Supports routing between different data sources and query types for enhanced RAG performance.

This module enables:
- Multi-source query routing (future extensibility)
- Query type classification and routing
- Fallback mechanisms for query handling
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import PydanticSingleSelector, LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core import Settings

from src.utils.logging_config import get_logger
from src.utils.config_loader import get_router_config

logger = get_logger("router_query_engine")


class VXRouterQueryEngine:
    """
    Advanced query router for VX-RAG using LlamaIndex RouterQueryEngine.

    Provides intelligent routing between different query engines based on:
    - Query content analysis
    - Data source characteristics
    - Performance optimization

    Currently supports single FAISS index but designed for multi-source extensibility.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize router query engine.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = get_router_config(config_path)

        # Router components
        self.router_engine: Optional[RouterQueryEngine] = None
        self.query_tools: List[QueryEngineTool] = []

        # Selector configuration
        self.selector_type = self.config.get('selector_type', 'pydantic')  # 'pydantic' or 'llm'
        self.use_multi_select = self.config.get('use_multi_select', False)

        logger.info(
            f"Initialized VXRouterQueryEngine: selector={self.selector_type}, "
            f"multi_select={self.use_multi_select}"
        )

    def add_query_tool(
        self,
        query_engine: Any,
        name: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a query engine tool to the router.

        Args:
            query_engine: LlamaIndex QueryEngine instance
            name: Tool name for identification
            description: Description for routing decisions
            metadata: Additional metadata for the tool
        """
        tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name=name,
            description=description
        )

        self.query_tools.append(tool)
        logger.info(f"Added query tool: {name}")

    def build_router(self) -> RouterQueryEngine:
        """
        Build the RouterQueryEngine with configured selector and tools.

        Returns:
            Configured RouterQueryEngine instance
        """
        if not self.query_tools:
            raise ValueError("No query tools added. Add tools before building router.")

        # Select appropriate selector
        if self.selector_type == 'llm':
            selector = LLMSingleSelector.from_defaults(
                llm=Settings.llm if hasattr(Settings, 'llm') else None
            )
        else:  # pydantic (default)
            selector = PydanticSingleSelector.from_defaults()

        # Build router
        self.router_engine = RouterQueryEngine(
            selector=selector,
            query_engine_tools=self.query_tools,
            verbose=self.config.get('verbose', False)
        )

        logger.info(
            f"Built RouterQueryEngine with {len(self.query_tools)} tools "
            f"using {self.selector_type} selector"
        )

        return self.router_engine

    def query(self, query_str: str) -> Any:
        """
        Execute query using the router.

        Args:
            query_str: Query string

        Returns:
            Query response from appropriate engine
        """
        if not self.router_engine:
            raise RuntimeError("Router not built. Call build_router() first.")

        logger.info(f"Routing query: {query_str[:100]}...")

        try:
            response = self.router_engine.query(query_str)

            logger.info(
                f"Query routed successfully, response length: {len(str(response))}"
            )

            return response

        except Exception as e:
            logger.error(f"Query routing failed: {e}")
            raise

    def get_router_info(self) -> Dict[str, Any]:
        """
        Get information about the router configuration.

        Returns:
            Dictionary with router details
        """
        return {
            'selector_type': self.selector_type,
            'use_multi_select': self.use_multi_select,
            'num_tools': len(self.query_tools),
            'tool_names': [tool.metadata.name if hasattr(tool.metadata, 'name') else 'unnamed' for tool in self.query_tools],
            'built': self.router_engine is not None,
            'config': self.config
        }

    def reset_tools(self) -> None:
        """Reset all query tools."""
        self.query_tools = []
        self.router_engine = None
        logger.info("Reset all query tools")


# Convenience function for creating router with FAISS query engine
def create_faiss_router(
    faiss_query_engine: Any,
    config_path: Optional[str] = None
) -> VXRouterQueryEngine:
    """
    Create a router query engine configured for FAISS index.

    Args:
        faiss_query_engine: FAISS-based QueryEngine
        config_path: Path to configuration file

    Returns:
        Configured VXRouterQueryEngine
    """
    router = VXRouterQueryEngine(config_path)

    # Add FAISS tool
    router.add_query_tool(
        query_engine=faiss_query_engine,
        name="faiss_vector_search",
        description=(
            "Use this for semantic search and retrieval from cybersecurity documents. "
            "Best for questions requiring understanding of technical concepts, "
            "code analysis, or detailed explanations from the VX Underground collection."
        ),
        metadata={
            'source_type': 'vector_db',
            'index_type': 'faiss',
            'data_domain': 'cybersecurity'
        }
    )

    # Build router
    router.build_router()

    logger.info("Created FAISS router query engine")

    return router