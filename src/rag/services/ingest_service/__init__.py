"""
Ingest Service for VX-RAG system.

Handles data ingestion from various sources (PDF, DOCX, APIs).
Provides unified interface for loading and preprocessing documents.
"""

from .service import PDFIngestAdapter

__all__ = ["PDFIngestAdapter"]

# TODO: Implement ingest adapters for different data sources
# TODO: Add text normalization and metadata extraction
# TODO: Support batch processing and error handling