"""
Ingest Service for VX-RAG system.

Handles data ingestion from various sources (PDF, DOCX, APIs).
Provides unified interface for loading and preprocessing documents.
"""

from .service import (
	PDFIngestAdapter,
	TXTIngestAdapter,
	MDIngestAdapter,
	process_and_save_documents,
	save_processed_text,
)

__all__ = [
	"PDFIngestAdapter",
	"TXTIngestAdapter",
	"MDIngestAdapter",
	"process_and_save_documents",
	"save_processed_text",
]