"""
MCP (Model Context Protocol) module for VX-RAG.

This module provides MCP server functionality for integrating with external LLMs and IDEs.

Architecture:
- server.py: Minimal FastMCP protocol layer
- formatters.py: Response formatting and redaction
- schemas.py: Pydantic models for type safety
"""

from .server import mcp

__all__ = ["mcp"]