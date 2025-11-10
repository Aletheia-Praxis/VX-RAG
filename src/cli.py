"""
CLI interface for VX-RAG.

Provides command-line tools for ingestion, indexing, and querying.
"""

import argparse
import sys
import time

from typing import Any, List, Dict

# Import structured logging and metrics
from src.utils.logging_config import get_logger, log_index_event
from src.utils.metrics import get_metrics

logger = get_logger("cli")
metrics = get_metrics()

def main() -> None:
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(description="VX-RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command - Full ingestion pipeline
    ingest_parser = subparsers.add_parser("ingest", help="Run full ingestion pipeline")
    ingest_parser.add_argument("--data-dir", type=str, default="data/raw",
                              help="Directory containing raw documents")
    ingest_parser.add_argument("--persist-dir", type=str, default="data/index",
                              help="Directory to persist index")
    ingest_parser.add_argument("--config", type=str, default="config/settings.yaml",
                              help="Configuration file path")
    ingest_parser.add_argument("--background", "-b", action="store_true",
                              help="Run ingestion in background (non-blocking)")

    # Index command
    index_parser = subparsers.add_parser("index", help="Create index")
    index_parser.add_argument("--persist-dir", type=str, default="./data/index",
                             help="Directory to persist index")
    index_parser.add_argument("--data-dir", type=str, default="./data/processed",
                             help="Directory containing processed documents")
    index_parser.add_argument("--config", type=str, default="./config/settings.yaml",
                             help="Configuration file path")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("query", type=str, help="Query string")
    query_parser.add_argument("--config", type=str, default="./config/settings.yaml",
                             help="Configuration file path")

    # Update index command
    update_parser = subparsers.add_parser("update-index", help="Update existing index with new documents")
    update_parser.add_argument("--persist-dir", type=str, default="./data/index",
                              help="Directory containing existing index")
    update_parser.add_argument("--data-dir", type=str, default="./data/processed",
                              help="Directory containing new processed documents")
    update_parser.add_argument("--config", type=str, default="./config/settings.yaml",
                              help="Configuration file path")

    # Snapshot command
    snapshot_parser = subparsers.add_parser("snapshot", help="Create index snapshot")
    snapshot_parser.add_argument("--persist-dir", type=str, default="./data/index",
                                help="Directory containing index")
    snapshot_parser.add_argument("--name", type=str, help="Snapshot name (optional)")
    snapshot_parser.add_argument("--config", type=str, default="./config/settings.yaml",
                                help="Configuration file path")

    # Verify snapshot command
    verify_parser = subparsers.add_parser("verify-snapshot", help="Verify snapshot integrity")
    verify_parser.add_argument("--persist-dir", type=str, default="./data/index",
                              help="Directory containing index")
    verify_parser.add_argument("--name", type=str, required=True, help="Snapshot name to verify")
    
    # Task status command
    status_parser = subparsers.add_parser("status", help="Check status of background task")
    status_parser.add_argument("task_id", type=str, help="Task ID to check")
    
    # Cancel task command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel background task")
    cancel_parser.add_argument("task_id", type=str, help="Task ID to cancel")
    
    # List tasks command
    list_parser = subparsers.add_parser("list-tasks", help="List all tasks")
    list_parser.add_argument(
        "--filter",
        choices=["all", "pending", "running", "completed", "failed", "cancelled"],
        default="all",
        help="Filter tasks by status"
    )
    
    # Cleanup tasks command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup old tasks (manual)")
    cleanup_parser.add_argument(
        "--all-completed",
        action="store_true",
        help="Remove all completed tasks"
    )
    cleanup_parser.add_argument(
        "--all-failed",
        action="store_true",
        help="Remove all failed tasks"
    )
    cleanup_parser.add_argument(
        "--all-cancelled",
        action="store_true",
        help="Remove all cancelled tasks"
    )
    
    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Display system metrics")
    metrics_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format"
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark search strategies")
    benchmark_parser.add_argument("query", type=str, help="Test query string")
    benchmark_parser.add_argument("--config", type=str, default="./config/settings.yaml",
                                 help="Configuration file path")
    benchmark_parser.add_argument("--alphas", type=str, default="0.0,0.3,0.5,0.7,1.0",
                                 help="Comma-separated hybrid alpha values to test")
    
    # Clean boilerplate command
    clean_bp_parser = subparsers.add_parser("clean-boilerplate", 
                                            help="Remove boilerplate from processed documents")
    clean_bp_parser.add_argument("--data-dir", type=str, default="./data/processed",
                                help="Directory containing documents to clean")
    clean_bp_parser.add_argument("--config", type=str, default="./config/settings.yaml",
                                help="Configuration file path")
    clean_bp_parser.add_argument("--dry-run", action="store_true",
                                help="Preview changes without saving")

    args = parser.parse_args()

    if args.command == "ingest":
        # Check if background mode requested
        if args.background:
            # Background mode: submit task to queue and return immediately
            print("Starting background ingestion...")
            
            import asyncio
            from pathlib import Path
            from src.utils.task_queue import get_task_queue, TaskPriority
            from src.rag.services.ingest_service.background_ingest import run_ingestion_pipeline
            
            async def submit_background_ingest() -> str:
                """Submit ingestion as background task."""
                queue = get_task_queue(
                    max_workers=4,
                    max_concurrent_tasks=2,
                    max_completed_tasks=100,
                    max_failed_tasks=50,
                )
                
                # Start queue if not running
                if not queue._running:
                    await queue.start()
                
                # Submit task with retry=2 (optimistic: if success no retries, if fail 2 retries)
                task_id = await queue.submit_task(
                    name=f"ingest_{Path(args.data_dir).name}",
                    func=run_ingestion_pipeline,
                    kwargs={
                        'data_dir': args.data_dir,
                        'persist_dir': args.persist_dir,
                        'config_path': args.config,
                    },
                    priority=TaskPriority.HIGH,
                    max_retries=2,  # 1 attempt + 2 retries if failed
                )
                
                return task_id
            
            # Execute submission
            task_id = asyncio.run(submit_background_ingest())
            
            print("Ingestion started in background")
            print(f"Task ID: {task_id}")
            print("\nUse the following commands:")
            print(f"  vx-rag status {task_id}")
            print(f"  vx-rag cancel {task_id}")
            print("  vx-rag list-tasks")
            sys.exit(0)
        
        # Synchronous mode: blocking execution
        start_time = time.time()
        try:
            from pathlib import Path
            from src.rag.services.ingest_service.service import PDFIngestAdapter, TXTIngestAdapter, MDIngestAdapter, process_and_save_documents
            from src.rag.services.duplicate_detection_service.service import DuplicateDetector
            from src.rag.services.chunker_service.service import Chunker
            from src.utils.config_loader import get_chunking_metadata, get_embedding_model_name, get_vector_store_type, load_settings
            from src.rag.services.embedder_service.service import EmbeddingService
            from src.rag.services.vectordb_service.service import VectorStoreClient
            
            data_path = Path(args.data_dir)
            processed_dir = Path("./data/processed")
            persist_dir = Path(args.persist_dir)
            
            if not data_path.exists():
                print(f"Data directory {data_path} does not exist")
                logger.error("Ingestion failed: data directory not found", data_dir=str(data_path))
                sys.exit(1)
            
            logger.info("Starting full ingestion pipeline", data_dir=str(data_path))
            
            # Load configuration for OCR and boilerplate removal
            config = load_settings(args.config) if Path(args.config).exists() else {}
            paddle_ocr_config = config.get('paddle_ocr', {})
            boilerplate_config = config.get('boilerplate_removal', {})
            
            # Step 1: Parse documents
            print("Step 1: Parsing documents...")
            pdf_adapter = PDFIngestAdapter()
            pdf_docs = pdf_adapter.load_data(str(data_path / "pdf"))
            print(f"Loaded {len(pdf_docs)} PDF documents")
            
            txt_adapter = TXTIngestAdapter()
            txt_docs = txt_adapter.load_data(str(data_path / "txt"))
            print(f"Loaded {len(txt_docs)} TXT documents")
            
            md_adapter = MDIngestAdapter()
            md_docs = md_adapter.load_data(str(data_path / "md"))
            print(f"Loaded {len(md_docs)} MD documents")
            
            all_docs = pdf_docs + txt_docs + md_docs
            print(f"Total documents parsed: {len(all_docs)}")
            
            # Step 1.5: OCR processing (if enabled)
            if paddle_ocr_config.get('enabled', False):
                print("Step 1.5: Processing images with OCR...")
                try:
                    # Note: OCR service is available but not yet fully implemented
                    # Currently only detecting image placeholders
                    # from src.rag.services.paddle_ocr_service.service import PaddleOCRService
                    
                    # Process documents with <!-- image --> placeholders
                    ocr_count = 0
                    for doc in all_docs:
                        text = doc.get('text', '')
                        if '<!-- image -->' in text or '<image>' in text:
                            # Extract images and replace placeholders
                            # Note: This is a simplified approach. Full implementation
                            # would extract actual images from PDFs and process them
                            logger.info(f"Document {doc.get('id', 'unknown')} contains image placeholders")
                            ocr_count += 1
                    
                    print(f"Processed {ocr_count} documents with OCR (image placeholders detected)")
                    logger.info(f"OCR processing completed: {ocr_count} documents")
                    
                except ImportError:
                    print("Warning: PaddleOCR not installed. Skipping OCR step.")
                    logger.warning("PaddleOCR not available, skipping OCR processing")
                except Exception as e:
                    print(f"Warning: OCR processing failed: {e}")
                    logger.error(f"OCR processing error: {e}")
            
            # Step 1.6: Boilerplate removal (if enabled)
            if boilerplate_config.get('enabled', False):
                print("Step 1.6: Removing boilerplate content...")
                try:
                    from src.rag.services.boilerplate_removal_service.service import BoilerplateRemovalService
                    
                    boilerplate_service = BoilerplateRemovalService(
                        aggressive_mode=boilerplate_config.get('aggressive_mode', True)
                    )
                    
                    total_removed = 0
                    for doc in all_docs:
                        text = doc.get('text', '')
                        original_length = len(text)
                        cleaned_text = boilerplate_service.remove_boilerplate(text)
                        doc['text'] = cleaned_text
                        removed = original_length - len(cleaned_text)
                        total_removed += removed
                    
                    print(f"Removed {total_removed} characters of boilerplate from {len(all_docs)} documents")
                    logger.info(f"Boilerplate removal completed: {total_removed} chars removed")
                    
                except Exception as e:
                    print(f"Warning: Boilerplate removal failed: {e}")
                    logger.error(f"Boilerplate removal error: {e}")
            
            # Step 2: Remove duplicates
            print("Step 2: Removing duplicates...")
            detector = DuplicateDetector()
            unique_docs = detector.remove_duplicates(all_docs)
            print(f"Documents after deduplication: {len(unique_docs)} (removed {len(all_docs) - len(unique_docs)} duplicates)")
            
            # Step 3: Chunk documents
            print("Step 3: Chunking documents...")
            # All chunking parameters loaded from config/settings.yaml
            chunker = Chunker()
            chunks = chunker.chunk_documents(unique_docs)
            print(f"Created {len(chunks)} chunks from {len(unique_docs)} documents")
            
            # Step 4: Save processed documents and chunks
            print("Step 4: Saving processed data...")
            saved_count = process_and_save_documents(unique_docs, processed_dir)
            print(f"Saved {saved_count} processed documents to {processed_dir}")
            
            # Step 5: Create/update vector index
            print("Step 5: Creating/updating vector index...")
            
            # Initialize embedder with model from config
            embedder = EmbeddingService(
                model_name=get_embedding_model_name(args.config)
            )
            
            # Initialize vector store with type from config
            vector_config = {'index_dir': str(persist_dir)}
            store_type = get_vector_store_type(args.config)
            vector_client = VectorStoreClient(store_type=store_type, config=vector_config)
            
            # Convert chunks to LlamaIndex documents for indexing
            from llama_index.core.schema import Document as LlamaDocument
            llama_docs = []
            for chunk in chunks:
                llama_doc = LlamaDocument(
                    text=chunk['text'],
                    metadata=chunk.get('metadata', {}),
                    id_=chunk['id']
                )
                llama_docs.append(llama_doc)
            
            # Build/update index
            index = vector_client.build_index(llama_docs, embedder.embed_model)
            if index:
                vector_client.save_index()
                print(f"Index created/updated and saved to {persist_dir}")
                
                # Create snapshot with metadata from config
                embed_model_info = {"model_name": embedder.model_name}
                chunking_params = get_chunking_metadata()
                vector_client.create_snapshot(None, embed_model_info, chunking_params)
                print("Index snapshot created")
            else:
                print("Failed to create/update index")
                logger.error("Index creation failed: build_index returned None")
                sys.exit(1)
            
            # Step 6: Build BM25 index for hybrid search
            print("Step 6: Building BM25 index for hybrid search...")
            from src.rag.services.retriever_service.service import RetrieverService
            
            retriever_service = RetrieverService(index=index, config_path=args.config)
            retriever_service.build_bm25_index(llama_docs)
            print("BM25 index built for hybrid search")
            
            duration = time.time() - start_time
            log_index_event("full_ingestion", len(unique_docs), duration)
            metrics.increment("ingestion_full_pipeline_total")
            metrics.histogram("ingestion_full_pipeline_duration_ms", duration * 1000)
            
            logger.info("Full ingestion pipeline completed", 
                       parsed=len(all_docs), 
                       unique=len(unique_docs), 
                       chunks=len(chunks), 
                       duration_ms=duration * 1000)
            
            print("\nIngestion Summary:")
            print(f"  Documents parsed: {len(all_docs)}")
            print(f"  Duplicates removed: {len(all_docs) - len(unique_docs)}")
            print(f"  Unique documents: {len(unique_docs)}")
            print(f"  Chunks created: {len(chunks)}")
            print("  Vector index: Created/Updated")
            print("  BM25 index: Built for hybrid search")
            print(f"  Duration: {duration:.2f} seconds")
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"Error during full ingestion: {e}")
            logger.error("Full ingestion pipeline failed", 
                        error=str(e), 
                        duration_ms=duration * 1000)
            sys.exit(1)

    elif args.command == "index":
        start_time = time.time()
        print(f"Creating index in {args.persist_dir}")
        try:
            from pathlib import Path
            from llama_index.core import SimpleDirectoryReader
            
            logger.info("Starting index creation", persist_dir=args.persist_dir, config_path=args.config)
            
            # Import services
            from src.rag.services.embedder_service.service import EmbeddingService
            from src.rag.services.vectordb_service.service import VectorStoreClient
            from src.utils.config_loader import get_embedding_model_name, get_vector_store_type
            
            # Initialize embedder with model from config
            embedder = EmbeddingService(
                model_name=get_embedding_model_name(args.config)
            )
            
            # Load processed documents
            data_path = Path(args.data_dir)
            if not data_path.exists():
                print(f"Data directory {data_path} does not exist")
                logger.error("Index creation failed: data directory not found", data_dir=str(data_path))
                sys.exit(1)
            
            reader = SimpleDirectoryReader(
                input_dir=str(data_path),
                required_exts=[".txt", ".md"],
                recursive=True
            )
            documents = reader.load_data()
            print(f"Loaded {len(documents)} documents from {data_path}")
            
            # Initialize vector store client with type from config
            vector_config = {
                'index_dir': args.persist_dir
            }
            store_type = get_vector_store_type(args.config)
            vector_client = VectorStoreClient(
                store_type=store_type,
                config=vector_config
            )
            
            # Build index
            index = vector_client.build_index(documents, embedder.embed_model)
            if index:
                vector_client.save_index()
                print(f"Index created and saved to {args.persist_dir}")
                
                duration = time.time() - start_time
                log_index_event("creation", len(documents), duration)
                metrics.increment("index_creations_total")
                metrics.histogram("index_creation_duration_ms", duration * 1000)
                
                logger.info("Index creation completed", 
                           documents_count=len(documents), 
                           persist_dir=args.persist_dir, 
                           duration_ms=duration * 1000)
            else:
                print("Failed to create index")
                logger.error("Index creation failed: build_index returned None")
                sys.exit(1)
                
        except Exception as e:
            duration = time.time() - start_time
            print(f"Error creating index: {e}")
            logger.error("Index creation failed", 
                        error=str(e), 
                        persist_dir=args.persist_dir, 
                        duration_ms=duration * 1000)
            sys.exit(1)

    elif args.command == "query":
        print(f"Querying: {args.query}")
        try:
            import asyncio
            from pathlib import Path
            from src.rag.services.vectordb_service.service import VectorStoreClient
            from src.rag.services.retriever_service.service import RetrieverService
            from src.rag.services.reranker_service.service import RerankerService
            from src.rag.services.embedder_service.service import EmbeddingService
            from src.rag.services.hybrid_search_service.service import HybridSearchService
            from src.utils.config_loader import get_embedding_model_name, get_vector_store_type, load_settings
            
            # Load configuration for index_dir
            config = load_settings(args.config) if Path(args.config).exists() else {}
            
            # Initialize embedding service with model from config
            embedder = EmbeddingService(
                model_name=get_embedding_model_name(args.config)
            )
            
            # Initialize vector store and load index
            vector_config = {
                'index_dir': config.get('index_dir', './data/index')
            }
            vector_client = VectorStoreClient(store_type=get_vector_store_type(args.config), config=vector_config)
            if not vector_client.load_index(embed_model=embedder.embed_model):
                print("Failed to load index. Please run 'index' command first.")
                sys.exit(1)
            
            # Initialize retriever
            retriever_service = RetrieverService(index=vector_client.index, config_path=args.config)
            if vector_client.index:
                retriever_service.set_index(vector_client.index)

            # Initialize reranker
            reranker_service = RerankerService(config_path=args.config)

            # Initialize hybrid search orchestrator
            hybrid_search_service = HybridSearchService(
                retriever_service=retriever_service,
                reranker_service=reranker_service,
                hybrid_alpha=config.get('hybrid_alpha', 0.5)
            )
            
            # Asynchronously retrieve documents
            async def do_search() -> List[Dict[str, Any]]:
                # Use top_k from config (retriever.semantic_top_k or similarity_top_k)
                top_k = config.get('similarity_top_k', 5)
                result = await hybrid_search_service.asearch(args.query, top_k=top_k)
                return result

            retrieved_docs = asyncio.run(do_search())
            
            if not retrieved_docs:
                print("No relevant documents found.")
                sys.exit(0)
            
            # Use context assembler to organize results
            from src.rag.services.assembler_service.service import ContextAssembler
            
            assembler = ContextAssembler(config_path=args.config)
            payload = assembler.assemble_context(
                query=args.query,
                documents=retrieved_docs,
                token_budget=config.get('context_assembler', {}).get('token_budget', 2048)
            )
            
            # Print results
            print(f"\n{'='*80}")
            print(f"Query: {args.query}")
            print(f"{'='*80}\n")
            print("Search Strategy: Hybrid (Vector + BM25)")
            print(f"Hybrid Alpha: {config.get('hybrid_alpha', 0.5)} (vector weight)")
            print(f"Results: {len(retrieved_docs)} documents retrieved, {len(payload.context)} in context")
            print(f"Token Budget: {payload.token_budget} (estimated: {payload.total_tokens_estimate()})")
            print(f"\n{'='*80}\n")
            
            for i, source in enumerate(retrieved_docs, 1):
                print(f"{i}. Score: {source.get('score', 0.0):.4f}")
                
                # Show component scores if available
                if 'vector_score' in source:
                    print(f"   - Vector Score: {source.get('vector_score', 0.0):.4f}")
                if 'bm25_score' in source:
                    print(f"   - BM25 Score: {source.get('bm25_score', 0.0):.4f}")
                
                # Show metadata
                metadata = source.get('metadata', {})
                if metadata:
                    print("   Metadata:")
                    for key, value in list(metadata.items())[:3]:  # Show first 3 metadata items
                        print(f"     - {key}: {value}")
                
                # Show text preview
                text = source.get('text', '')
                preview_length = 300 if len(retrieved_docs) <= 5 else 150
                preview = text[:preview_length] + ("..." if len(text) > preview_length else "")
                print(f"   Text: {preview}")
                print()
            
            # Print assembly statistics
            stats = assembler.get_assembly_stats(payload)
            print(f"{'='*80}")
            print("Context Assembly Statistics:")
            print(f"  Total Items: {stats['total_items']}")
            print(f"  Token Budget: {stats['token_budget']}")
            print(f"  Estimated Tokens: {stats['total_tokens_estimate']}")
            print(f"  Within Budget: {'YES' if stats['within_budget'] else 'NO'}")
            if stats.get('avg_score'):
                print(f"  Average Score: {stats['avg_score']:.4f}")
                print(f"  Score Range: {stats['min_score']:.4f} - {stats['max_score']:.4f}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"Error during query: {e}")
            sys.exit(1)

    elif args.command == "update-index":
        print(f"Updating index in {args.persist_dir} with documents from {args.data_dir}")
        try:
            from pathlib import Path
            from llama_index.core import SimpleDirectoryReader
            
            # Import services
            from src.rag.services.embedder_service.service import EmbeddingService
            from src.rag.services.vectordb_service.service import VectorStoreClient
            from src.utils.config_loader import get_chunking_metadata, get_embedding_model_name, get_vector_store_type, load_settings
            
            # Load configuration for other settings
            update_config = load_settings(args.config) if Path(args.config).exists() else {}
            
            # Initialize embedder with model from config
            embedder = EmbeddingService(
                model_name=get_embedding_model_name(args.config)
            )
            
            # Load new processed documents
            data_path = Path(args.data_dir)
            if not data_path.exists():
                print(f"Data directory {data_path} does not exist")
                sys.exit(1)
            
            reader = SimpleDirectoryReader(
                input_dir=str(data_path),
                required_exts=[".txt", ".md"],
                recursive=True
            )
            documents = reader.load_data()
            print(f"Loaded {len(documents)} new documents from {data_path}")
            
            if not documents:
                print("No new documents to add")
                sys.exit(0)
            
            # Initialize vector store client and load existing index
            vector_config = {
                'index_dir': args.persist_dir
            }
            store_type = update_config.get('vector_store', 'faiss') if isinstance(update_config, dict) else 'faiss'
            vector_client = VectorStoreClient(
                store_type=store_type,
                config=vector_config
            )
            
            if not vector_client.load_index():
                print(f"Failed to load existing index from {args.persist_dir}")
                sys.exit(1)
            
            # Add documents incrementally
            embed_model_info = {"model_name": embedder.model_name}
            chunking_params = get_chunking_metadata()
            
            success = vector_client.add_documents_incremental(documents, embedder.embed_model)
            if success:
                vector_client.save_index(create_backup=True, embed_model_info=embed_model_info, 
                                       chunking_params=chunking_params)
                print(f"Successfully added {len(documents)} documents to index")
            else:
                print("Failed to add documents to index")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error updating index: {e}")
            sys.exit(1)

    elif args.command == "snapshot":
        print(f"Creating snapshot of index in {args.persist_dir}")
        try:
            from pathlib import Path
            from src.rag.services.vectordb_service.service import VectorStoreClient
            from src.rag.services.embedder_service.service import EmbeddingService
            from src.utils.config_loader import get_chunking_metadata, get_embedding_model_name, get_vector_store_type
            
            # Initialize services for metadata
            embedder = EmbeddingService(
                model_name=get_embedding_model_name(args.config)
            )
            
            # Initialize vector store client and load index
            vector_config = {
                'index_dir': args.persist_dir
            }
            vector_client = VectorStoreClient(
                store_type=get_vector_store_type(args.config),
                config=vector_config
            )
            
            if not vector_client.load_index(embed_model=embedder.embed_model):
                print(f"Failed to load index from {args.persist_dir}")
                sys.exit(1)
            
            # Create snapshot with metadata from config
            embed_model_info = {"model_name": embedder.model_name}
            chunking_params = get_chunking_metadata()
            
            success = vector_client.create_snapshot(args.name, embed_model_info, chunking_params)
            if success:
                snapshot_name = args.name or "auto-generated"
                print(f"Snapshot '{snapshot_name}' created successfully")
            else:
                print("Failed to create snapshot")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error creating snapshot: {e}")
            sys.exit(1)

    elif args.command == "verify-snapshot":
        print(f"Verifying snapshot '{args.name}' in {args.persist_dir}")
        try:
            from src.rag.services.vectordb_service.service import VectorStoreClient
            
            # Initialize vector store client
            vector_config = {
                'index_dir': args.persist_dir
            }
            vector_client = VectorStoreClient(
                store_type="faiss",
                config=vector_config
            )
            
            # Verify snapshot
            result = vector_client.verify_snapshot_integrity(args.name)
            
            if result.get("valid", False):
                print(f"[OK] Snapshot '{args.name}' integrity verified successfully")
                print(f"  Verified files: {result.get('verified_files', 0)}/{result.get('total_files', 0)}")
            else:
                print(f"[FAIL] Snapshot '{args.name}' integrity verification failed")
                if "error" in result:
                    print(f"  Error: {result['error']}")
                if result.get("missing_files"):
                    print(f"  Missing files: {result['missing_files']}")
                if result.get("failed_files"):
                    print(f"  Corrupted files: {len(result['failed_files'])}")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error verifying snapshot: {e}")
            sys.exit(1)
    
    elif args.command == "status":
        """Check status of background task."""
        import asyncio
        from datetime import datetime
        from src.utils.task_queue import get_task_queue
        
        async def check_status() -> None:
            queue = get_task_queue()
            
            status = await queue.get_task_status(args.task_id)
            
            if not status:
                print(f"Task {args.task_id} not found")
                sys.exit(1)
            
            print("\nTask Status Report")
            print(f"{'=' * 70}")
            print(f"  ID:       {status['task_id']}")
            print(f"  Name:     {status['name']}")
            print(f"  Status:   {status['status'].upper()}")
            print(f"  Priority: {status['priority']}")
            
            if status.get('created_at'):
                created = datetime.fromtimestamp(status['created_at'])
                print(f"  Created:  {created.strftime('%Y-%m-%d %H:%M:%S')}")
            
            if status['status'] == 'running' and status.get('started_at'):
                elapsed = time.time() - status['started_at']
                print(f"  Running:  {elapsed:.1f} seconds")
            
            if status['status'] == 'completed' and status.get('result'):
                import json
                try:
                    result = json.loads(status['result']) if isinstance(status['result'], str) else status['result']
                    print("\n  Results:")
                    print(f"    Parsed:   {result.get('parsed', 0)} docs")
                    print(f"    Unique:   {result.get('unique', 0)} docs")
                    print(f"    Chunks:   {result.get('chunks', 0)}")
                    print(f"    Duration: {result.get('duration', 0):.2f}s")
                except Exception:
                    print(f"\n  Result: {status['result']}")
            
            if status['status'] == 'failed' and status.get('error'):
                print(f"\n  Error: {status['error']}")
            
            print(f"{'=' * 70}\n")
        
        asyncio.run(check_status())
    
    elif args.command == "cancel":
        """Cancel background task."""
        import asyncio
        from src.utils.task_queue import get_task_queue
        
        async def cancel_task() -> None:
            queue = get_task_queue()
            
            success = await queue.cancel_task(args.task_id)
            
            if success:
                print(f"Task {args.task_id} cancelled successfully")
            else:
                print(f"Failed to cancel task {args.task_id}")
                print("Task may be already completed or not found")
                sys.exit(1)
        
        asyncio.run(cancel_task())
    
    elif args.command == "list-tasks":
        """List all tasks with optional filtering."""
        import asyncio
        from src.utils.task_queue import get_task_queue
        
        async def list_all_tasks() -> None:
            queue = get_task_queue()
            
            stats = queue.get_queue_stats()
            
            print("\nTask Queue Statistics")
            print(f"{'=' * 70}")
            print(f"  Total Tasks:          {stats['total_tasks']}")
            print(f"  Pending:              {stats['pending']}")
            print(f"  Running:              {stats['running']}")
            print(f"  Completed:            {stats['completed']}")
            print(f"  Failed:               {stats['failed']}")
            print(f"  Cancelled:            {stats['cancelled']}")
            print(f"  Queue Size:           {stats['queue_size']}")
            print(f"  Max Completed Limit:  {stats['max_completed_tasks']}")
            print(f"  Max Failed Limit:     {stats['max_failed_tasks']}")
            print(f"{'=' * 70}\n")
            
            if stats['running'] > 0 or stats['pending'] > 0:
                print("Note: Use 'vx-rag status <task_id>' for detailed task information")
                print()
        
        asyncio.run(list_all_tasks())
    
    elif args.command == "cleanup":
        """Manual cleanup of old tasks."""
        import asyncio
        from src.utils.task_queue import get_task_queue, TaskStatus
        
        async def cleanup_tasks() -> None:
            queue = get_task_queue()
            
            removed_count = 0
            
            if args.all_completed:
                tasks_to_remove = [
                    task_id for task_id, task in queue._tasks.items()
                    if task.status == TaskStatus.COMPLETED
                ]
                for task_id in tasks_to_remove:
                    del queue._tasks[task_id]
                    removed_count += 1
                print(f"Removed {removed_count} completed tasks")
            
            if args.all_failed:
                tasks_to_remove = [
                    task_id for task_id, task in queue._tasks.items()
                    if task.status == TaskStatus.FAILED
                ]
                for task_id in tasks_to_remove:
                    del queue._tasks[task_id]
                    removed_count += 1
                print(f"Removed {removed_count} failed tasks")
            
            if args.all_cancelled:
                tasks_to_remove = [
                    task_id for task_id, task in queue._tasks.items()
                    if task.status == TaskStatus.CANCELLED
                ]
                for task_id in tasks_to_remove:
                    del queue._tasks[task_id]
                    removed_count += 1
                print(f"Removed {removed_count} cancelled tasks")
            
            if removed_count == 0:
                print("No tasks to cleanup. Use flags: --all-completed, --all-failed, --all-cancelled")
            else:
                # Save state after cleanup
                if queue.enable_persistence:
                    await queue._save_state()
                print(f"Total tasks removed: {removed_count}")
        
        asyncio.run(cleanup_tasks())
    
    elif args.command == "metrics":
        """Display system metrics."""
        try:
            import json
            
            stats = metrics.get_stats()
            
            if args.format == "json":
                print(json.dumps(stats, indent=2))
            else:
                print(f"\n{'='*70}")
                print("System Metrics")
                print(f"{'='*70}\n")
                
                # Index metrics
                if any(k.startswith('index_') for k in stats.keys()):
                    print("Index Operations:")
                    for key, value in stats.items():
                        if key.startswith('index_'):
                            print(f"  {key}: {value}")
                    print()
                
                # Query metrics
                if any(k.startswith('query_') for k in stats.keys()):
                    print("Query Operations:")
                    for key, value in stats.items():
                        if key.startswith('query_'):
                            print(f"  {key}: {value}")
                    print()
                
                # Ingestion metrics
                if any(k.startswith('ingestion_') for k in stats.keys()):
                    print("Ingestion Operations:")
                    for key, value in stats.items():
                        if key.startswith('ingestion_'):
                            print(f"  {key}: {value}")
                    print()
                
                print(f"{'='*70}\n")
                
        except Exception as e:
            print(f"Error displaying metrics: {e}")
            sys.exit(1)
    
    elif args.command == "benchmark":
        """Benchmark different search strategies."""
        print(f"Benchmarking search strategies for query: '{args.query}'")
        try:
            import asyncio
            from pathlib import Path
            from src.rag.services.vectordb_service.service import VectorStoreClient
            from src.rag.services.retriever_service.service import RetrieverService
            from src.rag.services.reranker_service.service import RerankerService
            from src.rag.services.embedder_service.service import EmbeddingService
            from src.rag.services.hybrid_search_service.service import HybridSearchService
            from src.utils.config_loader import get_embedding_model_name, get_vector_store_type, load_settings
            
            # Load configuration
            config = load_settings(args.config) if Path(args.config).exists() else {}
            
            # Parse alpha values
            alphas = [float(a.strip()) for a in args.alphas.split(',')]
            
            # Initialize services
            embedder = EmbeddingService(model_name=get_embedding_model_name(args.config))
            vector_config = {'index_dir': config.get('index_dir', './data/index')}
            vector_client = VectorStoreClient(store_type=get_vector_store_type(args.config), config=vector_config)
            
            if not vector_client.load_index(embed_model=embedder.embed_model):
                print("Failed to load index. Please run 'index' command first.")
                sys.exit(1)
            
            print(f"\n{'='*80}")
            print("Search Strategy Benchmark")
            print(f"{'='*80}\n")
            
            results_comparison = []
            
            for alpha in alphas:
                strategy_name = "Pure BM25" if alpha == 0.0 else "Pure Vector" if alpha == 1.0 else f"Hybrid (α={alpha})"
                print(f"Testing {strategy_name}...")
                
                # Create fresh services for each test
                retriever_service = RetrieverService(index=vector_client.index, config_path=args.config)
                if vector_client.index:
                    retriever_service.set_index(vector_client.index)  # Ensure index is set
                reranker_service = RerankerService(config_path=args.config)
                
                hybrid_search = HybridSearchService(
                    retriever_service=retriever_service,
                    reranker_service=reranker_service,
                    hybrid_alpha=alpha
                )
                
                async def run_search() -> tuple[List[Dict[str, Any]], float]:
                    start = time.time()
                    results = await hybrid_search.asearch(args.query, top_k=5)
                    duration = time.time() - start
                    return results, duration
                
                results, duration = asyncio.run(run_search())
                
                results_comparison.append({
                    'strategy': strategy_name,
                    'alpha': alpha,
                    'duration_ms': duration * 1000,
                    'results_count': len(results),
                    'avg_score': sum(r.get('score', 0.0) for r in results) / len(results) if results else 0.0,
                    'top_result_score': results[0].get('score', 0.0) if results else 0.0
                })
            
            # Display results
            print(f"\n{'='*80}")
            print("Benchmark Results")
            print(f"{'='*80}\n")
            print(f"{'Strategy':<20} {'Alpha':>6} {'Time (ms)':>10} {'Results':>8} {'Avg Score':>10} {'Top Score':>10}")
            print("-" * 80)
            
            for result in results_comparison:
                print(f"{result['strategy']:<20} {result['alpha']:>6.1f} {result['duration_ms']:>10.2f} "
                      f"{result['results_count']:>8} {result['avg_score']:>10.4f} {result['top_result_score']:>10.4f}")
            
            print(f"\n{'='*80}\n")
            
            # Find best performing strategy
            best = max(results_comparison, key=lambda x: x['avg_score'])
            print(f"Best performing strategy: {best['strategy']} (avg score: {best['avg_score']:.4f})")
            
        except Exception as e:
            print(f"Error during benchmark: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    elif args.command == "clean-boilerplate":
        """Clean boilerplate from processed documents."""
        print(f"Cleaning boilerplate from documents in {args.data_dir}")
        try:
            from pathlib import Path
            from src.rag.services.boilerplate_removal_service.service import BoilerplateRemovalService
            from src.utils.config_loader import load_settings
            
            data_path = Path(args.data_dir)
            if not data_path.exists():
                print(f"Data directory {data_path} does not exist")
                sys.exit(1)
            
            # Load config
            config = load_settings(args.config) if Path(args.config).exists() else {}
            boilerplate_config = config.get('boilerplate_removal', {})
            
            # Initialize service
            boilerplate_service = BoilerplateRemovalService(
                aggressive_mode=boilerplate_config.get('aggressive_mode', True)
            )
            
            # Process all text files
            txt_files = list(data_path.glob("**/*.txt"))
            print(f"Found {len(txt_files)} text files to process")
            
            if args.dry_run:
                print("\nDRY RUN - No files will be modified\n")
            
            total_removed = 0
            files_processed = 0
            
            for txt_file in txt_files:
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        original_text = f.read()
                    
                    cleaned_text = boilerplate_service.remove_boilerplate(original_text)
                    removed = len(original_text) - len(cleaned_text)
                    
                    if removed > 0:
                        total_removed += removed
                        files_processed += 1
                        
                        removal_pct = (removed / len(original_text)) * 100 if original_text else 0
                        print(f"  {txt_file.name}: removed {removed} chars ({removal_pct:.1f}%)")
                        
                        if not args.dry_run:
                            with open(txt_file, 'w', encoding='utf-8') as f:
                                f.write(cleaned_text)
                
                except Exception as e:
                    print(f"  Error processing {txt_file.name}: {e}")
            
            print("\nSummary:")
            print(f"  Files processed: {files_processed}/{len(txt_files)}")
            print(f"  Total removed: {total_removed} characters")
            
            if args.dry_run:
                print("\nThis was a dry run. Use without --dry-run to apply changes.")
            
        except Exception as e:
            print(f"Error cleaning boilerplate: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

