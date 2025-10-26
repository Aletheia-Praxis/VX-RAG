"""
MCP (Model Context Protocol) module for VX-RAG.

This module provides MCP server functionality and bridge components
for integrating with external LLMs and IDEs.
"""

from .bridge import MCPBridge, get_mcp_bridge
from .server import mcp

__all__ = ["MCPBridge", "get_mcp_bridge", "mcp"]