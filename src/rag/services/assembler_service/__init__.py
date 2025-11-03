"""
Assembler Service for VX-RAG system.

Handles context assembly for LLM queries.
Provides MCP-compatible payload formatting.
"""

from .service import ContextAssembler

__all__ = ["ContextAssembler"]

# TODO: Implement MCP payload assembly
# TODO: Add token budgeting and context selection
# TODO: Support provenance tracking