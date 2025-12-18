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
    from src.rag.services.ingestion_pipeline_service.service import IngestionPipelineService
    from src.rag.services.ingest_service.service import process_and_save_documents
    # Chunker behavior is now provided via IngestionPipelineService's AdaptiveChunkerTransformation
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core import StorageContext, VectorStoreIndex
    from src.utils.config_loader import (
        get_chunking_metadata,
        get_embedding_model_name,
        get_vector_store_type,
    )
    from llama_index.core.schema import Document, BaseNode
    
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
        
        ingestion_service = IngestionPipelineService(config_path=config_path)
        pdf_docs = []
        if (data_path / "pdf").exists():
            pdf_nodes = await ingestion_service.process_pdf_directory(data_path / "pdf")
            # Convert nodes to dictionaries compatible with rest of pipeline
            for n in pdf_nodes:
                pdf_docs.append({
                    'id': getattr(n, 'id_', ''),
                    'source': n.metadata.get('file_path', ''),
                    'text': getattr(n, 'text', ''),
                    'lang': n.metadata.get('lang', 'en'),
                    'metadata': n.metadata
                })
        logger.info(f"Parsed {len(pdf_docs)} PDF documents")
        
        if progress_callback:
            await progress_callback(20, "Parsing TXT documents...")
        
        txt_docs = []
        if (data_path / "txt").exists():
            txt_nodes = await ingestion_service.process_text_directory(data_path / "txt", "*.txt")
            for n in txt_nodes:
                txt_docs.append({
                    'id': getattr(n, 'id_', ''),
                    'source': n.metadata.get('file_path', ''),
                    'text': getattr(n, 'text', ''),
                    'lang': n.metadata.get('lang', 'en'),
                    'metadata': n.metadata
                })
        logger.info(f"Parsed {len(txt_docs)} TXT documents")
        
        if progress_callback:
            await progress_callback(30, "Parsing MD documents...")
        
        md_docs = []
        if (data_path / "md").exists():
            md_nodes = await ingestion_service.process_text_directory(data_path / "md", "*.md")
            for n in md_nodes:
                md_docs.append({
                    'id': getattr(n, 'id_', ''),
                    'source': n.metadata.get('file_path', ''),
                    'text': getattr(n, 'text', ''),
                    'lang': n.metadata.get('lang', 'en'),
                    'metadata': n.metadata
                })
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
                # from src.rag.services.paddle_ocr_service.service import PaddleOCRService
                
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
        
        # Step 1.6: Boilerplate Removal (if enabled) (37%) - handled by pipeline transforms
        boilerplate_config = config.get('boilerplate_removal', {})
        
        if boilerplate_config.get('enabled', False):
            if progress_callback:
                await progress_callback(37, "Removing boilerplate content...")
            
            try:
                from src.rag.services.boilerplate_removal_service.service import BoilerplateRemovalService
                
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
        
        # Step 2: Deduplication (40%) - already performed by pipeline transformations
        if progress_callback:
            await progress_callback(40, "Removing duplicates...")
        
        # Ensure unique_docs is taken from pipeline or perform deduplication if needed
        unique_docs = all_docs
        summary['unique'] = len(unique_docs)
        removed = 0
        logger.info(
            f"Deduplication complete: {len(unique_docs)} unique ({removed} duplicates removed)"
        )
        
        # Step 3: Chunking (55%) - handled by AdaptiveChunkerTransformation within pipeline
        if progress_callback:
            await progress_callback(55, "Chunking documents...")
        
        # Convert processed documents (unique_docs) directly to chunks using the ingestion service
        # Unique_docs are already chunked if AdaptiveChunkerTransformation is enabled
        chunks = []
        for doc in unique_docs:
            # If doc already represents a chunk, add directly
            chunk_id = doc.get('id') if isinstance(doc, dict) else None
            if isinstance(doc, dict) and doc.get('metadata', {}).get('chunk_metadata'):
                chunks.append(doc)
            else:
                # Fallback: use Simple splitting by newline for raw text
                if isinstance(doc, dict):
                    text = doc.get('text', '')
                    # Simple fallback chunking - split lines into blocks of ~1000 chars
                    block_size = 1000
                    for i in range(0, len(text), block_size):
                        chunks.append({'id': f"{doc.get('id', 'doc')}_chunk_{i}", 'text': text[i:i+block_size], 'metadata': doc.get('metadata', {})})
        summary['chunks'] = len(chunks)
        logger.info(f"Prepared {len(chunks)} chunks from {len(unique_docs)} documents")
        
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
        
        # Initialize vector store directly with LlamaIndex
        vector_store = FaissVectorStore.from_persist_dir(str(persist_path))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Convert chunks to LlamaIndex documents
        llama_docs = [
            Document(
                text=chunk['text'],
                metadata=chunk.get('metadata', {}),
                id_=chunk['id']
            )
            for chunk in chunks
        ]
        
        # Build index (CPU-intensive operation)
        index = await asyncio.to_thread(
            VectorStoreIndex.from_documents,
            llama_docs,
            storage_context=storage_context,
            embed_model=Settings.embed_model
        )
        
        if index:
            # Save index to disk
            await asyncio.to_thread(index.storage_context.persist, persist_dir=str(persist_path))
            
            # Create snapshot with metadata
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            snapshot_name = f"snapshot_{timestamp}"
            snapshot_dir = persist_path / "snapshots" / snapshot_name
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Persist to snapshot directory
            await asyncio.to_thread(index.storage_context.persist, persist_dir=str(snapshot_dir))
            
            # Create manifest
            embed_model_info = {"model_name": embed_config['embedding_model']}
            chunking_params = get_chunking_metadata()
            manifest = {
                "timestamp": timestamp,
                "embed_model_name": embed_model_info['model_name'],
                "embed_dim": embed_config.get('embedding_dim', 384),
                "chunking_params": chunking_params,
                "files": list(snapshot_dir.glob("*"))
            }
            
            import json
            manifest_path = snapshot_dir / "manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            summary['indexed'] = True
            logger.info(f"Vector index created and saved to {persist_path}")
        else:
            logger.error("Index creation failed: VectorStoreIndex.from_documents returned None")
            raise RuntimeError("Failed to create vector index")
        
        # Step 6: Build BM25 Index (95%)
        if progress_callback:
            await progress_callback(95, "Building BM25 index for hybrid search...")
        
        from llama_index.retrievers.bm25 import BM25Retriever
        from src.utils.config_loader import get_bm25_config
        from typing import cast, List
        from llama_index.core.schema import BaseNode
        
        bm25_config = get_bm25_config(config_path)
        similarity_top_k = bm25_config.get('similarity_top_k', 20)
        
        bm25_retriever = await asyncio.to_thread(
            BM25Retriever.from_defaults,
            nodes=cast(List[BaseNode], llama_docs),
            similarity_top_k=similarity_top_k,
            verbose=True
        )
        
        # Persist BM25 retriever
        bm25_index_path = Path(persist_dir) / "bm25_index"
        await asyncio.to_thread(bm25_retriever.persist, str(bm25_index_path))
        
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
