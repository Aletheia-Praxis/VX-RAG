"""
Schemas package for VX-RAG.

This package exposes Pydantic models used throughout the RAG and MCP
components. Individual models are implemented in `embedder_schemas.py`
and `mcp_schemas.py` and re-exported here for convenience.
"""

from .embedder_schemas import (
	EmbeddingRequest,
	EmbeddingResponse,
	EmbeddingVector,
)
from .mcp_schemas import (
	ContextItem,
	MCPContextPayload,
	ContextAssemblyRequest,
	ContextAssemblyResponse,
)

__all__ = [
	"EmbeddingRequest",
	"EmbeddingResponse",
	"EmbeddingVector",
	"ContextItem",
	"MCPContextPayload",
	"ContextAssemblyRequest",
	"ContextAssemblyResponse",
]