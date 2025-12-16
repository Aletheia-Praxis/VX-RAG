"""
IngestionPipelineService facade.

Provides a wrapper around LlamaIndex IngestionPipeline and SimpleDirectoryReader.
This implementation uses native LlamaIndex components for document loading,
cleaning, chunking, and embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict, Any
import asyncio
import logging

from llama_index.core import SimpleDirectoryReader, Document
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import BaseNode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.utils.config_loader import get_embedding_config, get_ingestion_config

# Import custom components
from src.rag.services.ingest_service.service import DoclingReader
from src.rag.services.boilerplate_removal_service.transform import BoilerplateCleaner
from src.rag.services.chunker_service.transform import AdaptiveChunker

logger = logging.getLogger(__name__)


class IngestionPipelineService:
    """
    Facade for ingestion pipeline functionality using LlamaIndex.
    """

    def __init__(self, config_path: Optional[str] = None, persist_dir: Optional[str] = None) -> None:
        self.config_path = config_path
        self.persist_dir = persist_dir
        self._embed_model: Optional[HuggingFaceEmbedding] = None

    def get_embedder(self) -> HuggingFaceEmbedding:
        """Get or initialize the embedding model."""
        if self._embed_model is None:
            embed_config = get_embedding_config(self.config_path)
            self._embed_model = HuggingFaceEmbedding(
                model_name=embed_config['model_name'],
                cache_folder=embed_config['cache_folder'],
                device=embed_config['device'],
                embed_batch_size=embed_config['batch_size']
            )
        return self._embed_model

    async def process_pdf_directory(self, directory: Path) -> List[BaseNode]:
        """
        Parse PDFs in the given directory using DoclingReader and run through IngestionPipeline.
        """
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Directory not found: {directory}")
            return []

        logger.info(f"Processing PDF directory: {directory}")

        # Get config
        ingestion_config = get_ingestion_config(self.config_path)
        ocr_enabled = ingestion_config.get("ocr_enabled", True)

        # Initialize SimpleDirectoryReader with DoclingReader
        reader = SimpleDirectoryReader(
            input_dir=str(directory),
            recursive=True,
            required_exts=[".pdf"],
            file_extractor={
                ".pdf": DoclingReader(ocr_enabled=ocr_enabled)
            }
        )

        # Load documents (in thread to avoid blocking)
        documents = await asyncio.to_thread(reader.load_data)
        logger.info(f"Loaded {len(documents)} PDF documents")

        if not documents:
            return []

        # Run pipeline
        return await self._run_pipeline(documents)

    async def process_text_directory(self, directory: Path, pattern: str = "*.txt") -> List[BaseNode]:
        """
        Parse text/markdown files and run through IngestionPipeline.
        """
        if not directory.exists() or not directory.is_dir():
            logger.warning(f"Directory not found: {directory}")
            return []

        logger.info(f"Processing text directory: {directory} with pattern {pattern}")

        # Determine extension from pattern
        ext = pattern.replace("*", "")
        
        # Initialize SimpleDirectoryReader
        reader = SimpleDirectoryReader(
            input_dir=str(directory),
            recursive=True,
            required_exts=[ext]
        )

        # Load documents
        documents = await asyncio.to_thread(reader.load_data)
        logger.info(f"Loaded {len(documents)} documents")

        if not documents:
            return []

        # Run pipeline
        return await self._run_pipeline(documents)

    async def _run_pipeline(self, documents: List[Document]) -> List[BaseNode]:
        """
        Run the ingestion pipeline on a list of documents.
        """
        pipeline = IngestionPipeline(
            transformations=[
                BoilerplateCleaner(aggressive_mode=True), # Config could be passed here
                AdaptiveChunker(), # Config could be passed here
                # Embedder is removed to avoid double computation in handle_index
                # self.get_embedder(),
            ]
        )

        # Run pipeline (arun for async)
        nodes = await pipeline.arun(documents=documents)
        logger.info(f"Pipeline processed {len(nodes)} nodes")
        
        return list(nodes)

    def persist_pipeline(self) -> None:
        """
        Persist pipeline state (if caching is enabled).
        Currently no-op as we recreate pipeline per batch, but could be extended.
        """
        pass
