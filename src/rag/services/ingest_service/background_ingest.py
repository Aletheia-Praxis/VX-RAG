"""
Background Ingestion Pipeline for VX-RAG.

Provides async wrapper for the full ingestion pipeline with progress tracking
and task queue integration. Designed for non-blocking background execution.
"""

import asyncio
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Awaitable

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


async def run_ingestion_pipeline(
    data_dir: str,
    persist_dir: str,
    config_path: str,
    progress_callback: Optional[Callable[[int, str], Awaitable[None]]] = None,
) -> Dict[str, Any]:
    """
    Execute full ingestion pipeline in background task with progress tracking.
    
    This function wraps the entire ingestion workflow (parsing, deduplication,
    chunking, embedding, indexing) and executes CPU-bound operations in thread pool
    to avoid blocking the event loop.
    
    Progress tracking is provided via optional callback with percentage (0-100)
    and status message.
    
    Args:
        data_dir: Directory containing raw documents (pdf/txt/md subdirs)
        persist_dir: Directory to persist vector index
        config_path: Path to settings.yaml configuration file
        progress_callback: Optional async callback(percentage: int, status: str)
        
    Returns:
        Dictionary with ingestion summary statistics:
            - parsed: Total documents parsed
            - unique: Unique documents after deduplication
            - chunks: Total chunks created
            - indexed: Whether index was successfully created
            - duration: Total execution time in seconds
            
    Raises:
        FileNotFoundError: If data_dir does not exist
        Exception: Any error during pipeline execution (propagated to TaskQueue)
        
    Example:
        >>> async def progress_handler(pct: int, msg: str) -> None:
        ...     print(f"[{pct}%] {msg}")
        >>> 
        >>> summary = await run_ingestion_pipeline(
        ...     data_dir="data/raw",
        ...     persist_dir="data/index",
        ...     config_path="config/settings.yaml",
        ...     progress_callback=progress_handler
        ... )
        >>> print(f"Ingested {summary['unique']} documents")
    """
    from rag.services.ingest_service.service import (
        PDFIngestAdapter,
        TXTIngestAdapter,
        MDIngestAdapter,
        process_and_save_documents,
    )
    from rag.services.duplicate_detection_service.service import DuplicateDetector
    from rag.services.chunker_service.service import Chunker
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from rag.services.vectordb_service.service import VectorStoreClient
    from rag.services.retriever_service.service import RetrieverService
    from src.utils.config_loader import (
        get_chunking_metadata,
        get_embedding_model_name,
        get_vector_store_type,
    )
    from llama_index.core.schema import Document as LlamaDocument
    
    data_path = Path(data_dir)
    processed_dir = Path("./data/processed")
    persist_path = Path(persist_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_path} not found")
    
    summary = {
        'parsed': 0,
        'unique': 0,
        'chunks': 0,
        'indexed': False,
        'duration': 0.0,
    }
    
    start_time = time.time()
    
    try:
        logger.info("Background ingestion: starting pipeline", data_dir=data_dir)
        
        # Step 1: Parse Documents (10-30%)
        if progress_callback:
            await progress_callback(10, "Parsing PDF documents...")
        
        pdf_adapter = PDFIngestAdapter()
        pdf_docs = await asyncio.to_thread(
            pdf_adapter.load_data,
            str(data_path / "pdf")
        )
        logger.info(f"Parsed {len(pdf_docs)} PDF documents")
        
        if progress_callback:
            await progress_callback(20, "Parsing TXT documents...")
        
        txt_adapter = TXTIngestAdapter()
        txt_docs = await asyncio.to_thread(
            txt_adapter.load_data,
            str(data_path / "txt")
        )
        logger.info(f"Parsed {len(txt_docs)} TXT documents")
        
        if progress_callback:
            await progress_callback(30, "Parsing MD documents...")
        
        md_adapter = MDIngestAdapter()
        md_docs = await asyncio.to_thread(
            md_adapter.load_data,
            str(data_path / "md")
        )
        logger.info(f"Parsed {len(md_docs)} MD documents")
        
        all_docs = pdf_docs + txt_docs + md_docs
        summary['parsed'] = len(all_docs)
        logger.info(f"Total documents parsed: {len(all_docs)}")
        
        # Step 1.5: OCR Processing (if enabled) (35%)
        from src.utils.config_loader import load_settings
        config = load_settings(config_path) if Path(config_path).exists() else {}
        paddle_ocr_config = config.get('paddle_ocr', {})
        
        if paddle_ocr_config.get('enabled', False):
            if progress_callback:
                await progress_callback(35, "Processing images with OCR...")
            
            try:
                # Note: OCR service is available but not yet fully implemented
                # Currently only detecting image placeholders
                # from rag.services.paddle_ocr_service.service import PaddleOCRService
                
                # Process documents with image placeholders
                ocr_count = 0
                for doc in all_docs:
                    text = doc.get('text', '')
                    if '<!-- image -->' in text or '<image>' in text:
                        logger.info(f"Document {doc.get('id', 'unknown')} contains image placeholders")
                        ocr_count += 1
                
                logger.info(f"OCR processing completed: {ocr_count} documents with image placeholders")
                
            except ImportError:
                logger.warning("PaddleOCR not installed, skipping OCR processing")
            except Exception as e:
                logger.error(f"OCR processing error: {e}")
        
        # Step 1.6: Boilerplate Removal (if enabled) (37%)
        boilerplate_config = config.get('boilerplate_removal', {})
        
        if boilerplate_config.get('enabled', False):
            if progress_callback:
                await progress_callback(37, "Removing boilerplate content...")
            
            try:
                from rag.services.boilerplate_removal_service.service import BoilerplateRemovalService
                
                boilerplate_service = BoilerplateRemovalService(
                    aggressive_mode=boilerplate_config.get('aggressive_mode', True)
                )
                
                total_removed = 0
                for doc in all_docs:
                    text = doc.get('text', '')
                    original_length = len(text)
                    cleaned_text = await asyncio.to_thread(
                        boilerplate_service.remove_boilerplate,
                        text
                    )
                    doc['text'] = cleaned_text
                    removed = original_length - len(cleaned_text)
                    total_removed += removed
                
                logger.info(f"Boilerplate removal completed: {total_removed} chars removed")
                
            except Exception as e:
                logger.error(f"Boilerplate removal error: {e}")
        
        # Step 2: Deduplication (40%)
        if progress_callback:
            await progress_callback(40, "Removing duplicates...")
        
        detector = DuplicateDetector()
        unique_docs = await asyncio.to_thread(
            detector.remove_duplicates,
            all_docs
        )
        summary['unique'] = len(unique_docs)
        removed = len(all_docs) - len(unique_docs)
        logger.info(
            f"Deduplication complete: {len(unique_docs)} unique ({removed} duplicates removed)"
        )
        
        # Step 3: Chunking (55%)
        if progress_callback:
            await progress_callback(55, "Chunking documents...")
        
        chunker = Chunker()
        chunks = await asyncio.to_thread(
            chunker.chunk_documents,
            unique_docs
        )
        summary['chunks'] = len(chunks)
        logger.info(f"Created {len(chunks)} chunks from {len(unique_docs)} documents")
        
        # Step 4: Save Processed Data (70%)
        if progress_callback:
            await progress_callback(70, "Saving processed data...")
        
        saved_count = await asyncio.to_thread(
            process_and_save_documents,
            unique_docs,
            processed_dir
        )
        logger.info(f"Saved {saved_count} processed documents to {processed_dir}")
        
        # Step 5: Create Vector Index (85%)
        if progress_callback:
            await progress_callback(85, "Building vector index...")
        
        # Configure global embedding model
        from src.utils.config_loader import get_embedding_config
        embed_config = get_embedding_config(config_path)
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_config['embedding_model'],
            embed_batch_size=embed_config['embedding_batch_size'],
            trust_remote_code=embed_config['embedding_trust_remote_code']
        )
        
        # Initialize vector store
        vector_config = {'index_dir': str(persist_path)}
        store_type = get_vector_store_type(config_path)
        vector_client = VectorStoreClient(
            store_type=store_type,
            config=vector_config
        )
        
        # Convert chunks to LlamaIndex documents
        llama_docs = [
            LlamaDocument(
                text=chunk['text'],
                metadata=chunk.get('metadata', {}),
                id_=chunk['id']
            )
            for chunk in chunks
        ]
        
        # Build index (CPU-intensive operation)
        index = await asyncio.to_thread(
            vector_client.build_index,
            llama_docs,
            Settings.embed_model
        )
        
        if index:
            # Save index to disk
            await asyncio.to_thread(vector_client.save_index)
            
            # Create snapshot with metadata
            embed_model_info = {"model_name": embed_config['embedding_model']}
            chunking_params = get_chunking_metadata()
            await asyncio.to_thread(
                vector_client.create_snapshot,
                None,  # Auto-generate snapshot name
                embed_model_info,
                chunking_params
            )
            
            summary['indexed'] = True
            logger.info(f"Vector index created and saved to {persist_path}")
        else:
            logger.error("Index creation failed: build_index returned None")
            raise RuntimeError("Failed to create vector index")
        
        # Step 6: Build BM25 Index (95%)
        if progress_callback:
            await progress_callback(95, "Building BM25 index for hybrid search...")
        
        retriever_service = RetrieverService(
            index=index,
            config_path=config_path
        )
        await asyncio.to_thread(
            retriever_service.build_bm25_index,
            llama_docs
        )
        logger.info("BM25 index built for hybrid search")
        
        # Step 7: Complete (100%)
        if progress_callback:
            await progress_callback(100, "Ingestion complete!")
        
        summary['duration'] = time.time() - start_time
        
        logger.info(
            "Background ingestion completed successfully",
            parsed=summary['parsed'],
            unique=summary['unique'],
            chunks=summary['chunks'],
            duration_sec=round(summary['duration'], 2)
        )
        
        return summary
        
    except Exception as e:
        summary['duration'] = time.time() - start_time
        logger.error(
            "Background ingestion failed",
            error=str(e),
            duration_sec=round(summary['duration'], 2),
            exc_info=True
        )
        raise
