"""
Ingestion Pipeline Service using LlamaIndex IngestionPipeline.

This service provides a unified ingestion pipeline that leverages LlamaIndex's
IngestionPipeline for document processing while preserving specialized functionality
for Docling + OCR processing and custom transformations.
"""

from .service import IngestionPipelineService

__all__ = ["IngestionPipelineService"]