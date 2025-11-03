"""
CLI interface for VX-RAG.

Provides command-line tools for ingestion, indexing, and querying.
"""

import argparse
import sys
import time

from typing import Any, List, Dict, cast

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

    args = parser.parse_args()

    if args.command == "ingest":
        start_time = time.time()
        try:
            from pathlib import Path
            from rag.services.ingest_service.service import PDFIngestAdapter, TXTIngestAdapter, MDIngestAdapter, process_and_save_documents
            from rag.services.duplicate_detection_service.service import DuplicateDetector
            from rag.services.chunker_service.service import Chunker
            from rag.services.embedder_service.service import EmbeddingService
            from rag.services.vectordb_service.service import VectorStoreClient
            
            data_path = Path(args.data_dir)
            processed_dir = Path("./data/processed")
            persist_dir = Path(args.persist_dir)
            
            if not data_path.exists():
                print(f"Data directory {data_path} does not exist")
                logger.error("Ingestion failed: data directory not found", data_dir=str(data_path))
                sys.exit(1)
            
            logger.info("Starting full ingestion pipeline", data_dir=str(data_path))
            
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
            
            # Step 2: Remove duplicates
            print("Step 2: Removing duplicates...")
            detector = DuplicateDetector()
            unique_docs = detector.remove_duplicates(all_docs)
            print(f"Documents after deduplication: {len(unique_docs)} (removed {len(all_docs) - len(unique_docs)} duplicates)")
            
            # Step 3: Chunk documents
            print("Step 3: Chunking documents...")
            chunker = Chunker(chunk_size=1024, chunk_overlap=200, use_semantic_chunking=True)  # Adaptive chunking enabled by default
            chunks = chunker.chunk_documents(unique_docs)
            print(f"Created {len(chunks)} chunks from {len(unique_docs)} documents")
            
            # Step 4: Save processed documents and chunks
            print("Step 4: Saving processed data...")
            saved_count = process_and_save_documents(unique_docs, processed_dir)
            print(f"Saved {saved_count} processed documents to {processed_dir}")
            
            # Step 5: Create/update vector index
            print("Step 5: Creating/updating vector index...")
            
            # Load configuration
            import yaml
            index_config: dict[str, Any] = {}
            if Path(args.config).exists():
                with open(args.config, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        index_config = loaded_config
            
            # Initialize embedder
            embedder = EmbeddingService(
                model_name=index_config.get('embedding_model', 'all-MiniLM-L6-v2')
            )
            
            # Initialize vector store
            vector_config = {'index_dir': str(persist_dir)}
            store_type = index_config.get('vector_store', 'faiss')
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
                
                # Create snapshot with metadata
                embed_model_info = {"model_name": embedder.model_name}
                chunking_params = {"adaptive_chunking": True, "default_chunk_size": 1024, "technical_chunk_size": 384}
                vector_client.create_snapshot(None, embed_model_info, chunking_params)
                print("Index snapshot created")
            else:
                print("Failed to create/update index")
                logger.error("Index creation failed: build_index returned None")
                sys.exit(1)
            
            # Step 6: Build BM25 index for hybrid search
            print("Step 6: Building BM25 index for hybrid search...")
            from rag.services.retriever_service.service import RetrieverService
            
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
            import yaml
            from pathlib import Path
            from llama_index.core import SimpleDirectoryReader
            
            # Load configuration
            index_cmd_config: dict[str, Any] = {}
            if Path(args.config).exists():
                with open(args.config, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        index_cmd_config = loaded_config
            
            logger.info("Starting index creation", persist_dir=args.persist_dir, config_path=args.config)
            
            # Import services
            from rag.services.embedder_service.service import EmbeddingService
            from rag.services.vectordb_service.service import VectorStoreClient
            
            # Initialize embedder
            embedder = EmbeddingService(
                model_name=index_cmd_config.get('embedding_model', 'all-MiniLM-L6-v2') if isinstance(index_cmd_config, dict) else 'all-MiniLM-L6-v2'
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
            
            # Initialize vector store client
            vector_config = {
                'index_dir': args.persist_dir
            }
            store_type = index_cmd_config.get('vector_store', 'faiss') if isinstance(index_cmd_config, dict) else 'faiss'
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
            import yaml
            from pathlib import Path
            from rag.services.vectordb_service.service import VectorStoreClient
            from rag.services.retriever_service.service import RetrieverService
            from rag.services.reranker_service.service import RerankerService
            from rag.services.embedder_service.service import EmbeddingService
            from rag.services.hybrid_search_service.service import HybridSearchService
            
            # Load configuration
            query_config: dict[str, Any] = {}
            if Path(args.config).exists():
                with open(args.config, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        query_config = loaded_config
            
            # Initialize embedding service (needed for index loading)
            embedder = EmbeddingService(
                model_name=query_config.get('embedding_model', 'all-MiniLM-L6-v2') if isinstance(query_config, dict) else 'all-MiniLM-L6-v2'
            )
            
            # Initialize vector store and load index
            vector_config = {
                'index_dir': query_config.get('index_dir', './data/index') if isinstance(query_config, dict) else './data/index'
            }
            vector_client = VectorStoreClient(store_type="faiss", config=vector_config)
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
                hybrid_alpha=query_config.get('hybrid_alpha', 0.5)
            )
            
            # Asynchronously retrieve documents
            async def do_search() -> List[Dict[str, Any]]:
                result = await hybrid_search_service.asearch(args.query, top_k=5)
                return cast(List[Dict[str, Any]], result)

            retrieved_docs = asyncio.run(do_search())
            
            if not retrieved_docs:
                print("No relevant documents found.")
                sys.exit(0)
            
            # Print results
            print(f"\nQuery: {args.query}")
            print("\nTop Results:")
            for i, source in enumerate(retrieved_docs, 1):
                print(f"{i}. Score: {source.get('score', 0.0):.3f}")
                print(f"   Text: {source.get('text', '')[:200]}...")
                print()
            
        except Exception as e:
            print(f"Error during query: {e}")
            sys.exit(1)

    elif args.command == "update-index":
        print(f"Updating index in {args.persist_dir} with documents from {args.data_dir}")
        try:
            import yaml
            from pathlib import Path
            from llama_index.core import SimpleDirectoryReader
            
            # Load configuration
            update_config: dict[str, Any] = {}
            if Path(args.config).exists():
                with open(args.config, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        update_config = loaded_config
            
            # Import services
            from rag.services.embedder_service.service import EmbeddingService
            from rag.services.vectordb_service.service import VectorStoreClient
            
            # Initialize embedder
            embedder = EmbeddingService(
                model_name=update_config.get('embedding_model', 'all-MiniLM-L6-v2') if isinstance(update_config, dict) else 'all-MiniLM-L6-v2'
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
            chunking_params = {"adaptive_chunking": True, "default_chunk_size": 1024, "technical_chunk_size": 256}
            
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
            import yaml
            from pathlib import Path
            from rag.services.vectordb_service.service import VectorStoreClient
            from rag.services.embedder_service.service import EmbeddingService
            
            # Load configuration
            snapshot_config: dict[str, Any] = {}
            if Path(args.config).exists():
                with open(args.config, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        snapshot_config = loaded_config
            
            # Initialize services for metadata
            embedder = EmbeddingService(
                model_name=snapshot_config.get('embedding_model', 'all-MiniLM-L6-v2') if isinstance(snapshot_config, dict) else 'all-MiniLM-L6-v2'
            )
            
            # Initialize vector store client and load index
            vector_config = {
                'index_dir': args.persist_dir
            }
            vector_client = VectorStoreClient(
                store_type=snapshot_config.get('vector_store', 'faiss') if isinstance(snapshot_config, dict) else 'faiss',
                config=vector_config
            )
            
            if not vector_client.load_index():
                print(f"Failed to load index from {args.persist_dir}")
                sys.exit(1)
            
            # Create snapshot with metadata
            embed_model_info = {"model_name": embedder.model_name}
            chunking_params = {"adaptive_chunking": True, "default_chunk_size": 1024, "technical_chunk_size": 256}
            
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
            from rag.services.vectordb_service.service import VectorStoreClient
            
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
                print(f"✓ Snapshot '{args.name}' integrity verified successfully")
                print(f"  Verified files: {result.get('verified_files', 0)}/{result.get('total_files', 0)}")
            else:
                print(f"✗ Snapshot '{args.name}' integrity verification failed")
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

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
