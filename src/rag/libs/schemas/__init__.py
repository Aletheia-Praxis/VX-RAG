"""
Schemas package for VX-RAG.

This package exposes Pydantic models used throughout the RAG and MCP
components. MCP-specific models are implemented in `mcp_schemas.py`
and re-exported here for convenience.
"""

from .mcp_schemas import (
	ContextItem,
	MCPContextPayload,
	ContextAssemblyRequest,
	ContextAssemblyResponse,
)

__all__ = [
	"ContextItem",
	"MCPContextPayload",
	"ContextAssemblyRequest",
	"ContextAssemblyResponse",
]