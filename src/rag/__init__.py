"""
VX-RAG System Initialization.

This module provides the main entry point for initializing and using
the VX-RAG (Retrieval-Augmented Generation) system components.
"""

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Note: The old RAGSystem class has been removed after migration to RAGOrchestrator.
# Use src.rag.orchestrator.RAGOrchestrator for RAG functionality.

__all__: list[str] = []
