"""
Ingest Service for VX-RAG system.

Handles data ingestion from various sources (PDF, DOCX, APIs).
Provides unified interface for loading and preprocessing documents.
"""

from .service import (
	DoclingReader,
)

__all__ = [
	"DoclingReader",
]