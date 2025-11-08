"""
PaddleOCR Service for VX-RAG system.

Handles text extraction from images using PaddleOCR.
Integrates with Docling to extract text from images ignored during document parsing.
"""

from .service import PaddleOCRService

__all__ = ['PaddleOCRService']