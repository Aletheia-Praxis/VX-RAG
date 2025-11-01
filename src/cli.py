"""
CLI interface for VX-RAG.

Provides command-line tools for ingestion, indexing, and querying.
"""

import argparse
import sys
import time

from typing import Any

# Import structured logging and metrics
from .utils.logging_config import get_logger, log_index_event
from .utils.metrics import get_metrics

logger = get_logger("cli")
metrics = get_metrics()

def main() -> None:
    """
    Main CLI entry point.
    """
    parser = argparse.ArgumentParser(description="VX-RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument("--data-dir", type=str, default="../data/raw",
                              help="Directory containing raw documents")

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
            from rag.services.ingest_service.service import PDFIngestAdapter, TXTIngestAdapter, MDIngestAdapter, save_processed_text
            
            data_path = Path(args.data_dir)
            processed_dir = Path("./data/processed")
            
            if not data_path.exists():
                print(f"Data directory {data_path} does not exist")
                logger.error("Ingestion failed: data directory not found", data_dir=str(data_path))
                sys.exit(1)
            
            logger.info("Starting document ingestion", data_dir=str(data_path))
            
            # Process PDF files
            pdf_adapter = PDFIngestAdapter()
            pdf_docs = pdf_adapter.load_data(str(data_path / "pdf"))
            print(f"Loaded {len(pdf_docs)} PDF documents")
            
            # Process TXT files
            txt_adapter = TXTIngestAdapter()
            txt_docs = txt_adapter.load_data(str(data_path / "txt"))
            print(f"Loaded {len(txt_docs)} TXT documents")
            
            # Process MD files
            md_adapter = MDIngestAdapter()
            md_docs = md_adapter.load_data(str(data_path / "md"))
            print(f"Loaded {len(md_docs)} MD documents")
            
            # Combine all documents
            all_docs = pdf_docs + txt_docs + md_docs
            print(f"Total documents to process: {len(all_docs)}")
            
            # Save processed documents
            saved_count = save_processed_text(all_docs, processed_dir)
            print(f"Successfully saved {saved_count} processed documents to {processed_dir}")
            
            duration = time.time() - start_time
            log_index_event("ingestion", saved_count, duration)
            metrics.increment("ingestion_total")
            metrics.histogram("ingestion_duration_ms", duration * 1000)
            
            logger.info("Document ingestion completed", 
                       pdf_count=len(pdf_docs), 
                       txt_count=len(txt_docs), 
                       md_count=len(md_docs), 
                       total_saved=saved_count, 
                       duration_ms=duration * 1000)
            
        except Exception as e:
            duration = time.time() - start_time
            print(f"Error during ingestion: {e}")
            logger.error("Document ingestion failed", 
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
            from llama_index.core.node_parser import TokenTextSplitter
            
            # Load configuration
            config: dict[str, Any] = {}
            if Path(args.config).exists():
                with open(args.config, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        config = loaded_config
            
            logger.info("Starting index creation", persist_dir=args.persist_dir, config_path=args.config)
            
            # Create node parser with chunking settings
            chunk_size = config.get('chunk_size', 1024)
            chunk_overlap = config.get('chunk_overlap', 10)
            node_parser = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Import services
            from rag.services.embedder_service.service import EmbeddingService
            from rag.services.vectordb_service.service import VectorStoreClient
            
            # Initialize embedder
            embedder = EmbeddingService(
                model_name=config.get('embedding_model', 'all-MiniLM-L6-v2') if isinstance(config, dict) else 'all-MiniLM-L6-v2'
            )
            
            # Load processed documents
            data_path = Path(args.data_dir)
            if not data_path.exists():
                print(f"Data directory {data_path} does not exist")
                logger.error("Index creation failed: data directory not found", data_dir=str(data_path))
                sys.exit(1)
            
            reader = SimpleDirectoryReader(
                input_dir=str(data_path),
                required_exts=[".txt"],
                recursive=True
            )
            documents = reader.load_data()
            print(f"Loaded {len(documents)} documents from {data_path}")
            
            # Initialize vector store client
            vector_config = {
                'index_dir': args.persist_dir
            }
            store_type = config.get('vector_store', 'faiss') if isinstance(config, dict) else 'faiss'
            vector_client = VectorStoreClient(
                store_type=store_type,
                config=vector_config
            )
            
            # Build index
            index = vector_client.build_index(documents, embedder.embed_model, transformations=[node_parser])
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
            from rag.services.vectordb_service.service import VectorStoreClient
            from rag.services.retriever_service.service import RetrieverService
            
            # Initialize vector store and load index
            vector_client = VectorStoreClient(store_type="faiss")
            if not vector_client.load_index():
                print("Failed to load index. Please run 'index' command first.")
                sys.exit(1)
            
            # Initialize retriever
            retriever = RetrieverService(vector_client.index)
            
            # Retrieve documents
            retrieved_docs = retriever.retrieve(args.query, top_k=5)
            if not retrieved_docs:
                print("No relevant documents found.")
                sys.exit(0)
            
            # Print results
            print(f"\nQuery: {args.query}")
            print("\nTop Results:")
            for i, source in enumerate(retrieved_docs, 1):
                print(f"{i}. Score: {source['score']:.3f}")
                print(f"   Text: {source['text'][:200]}...")
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
            from llama_index.core.node_parser import TokenTextSplitter
            
            # Load configuration
            config: dict[str, Any] = {}
            if Path(args.config).exists():
                with open(args.config, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        config = loaded_config
            
            # Create node parser with chunking settings
            chunk_size = config.get('chunk_size', 1024)
            chunk_overlap = config.get('chunk_overlap', 10)
            node_parser = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Import services
            from rag.services.embedder_service.service import EmbeddingService
            from rag.services.vectordb_service.service import VectorStoreClient
            
            # Initialize embedder
            embedder = EmbeddingService(
                model_name=config.get('embedding_model', 'all-MiniLM-L6-v2') if isinstance(config, dict) else 'all-MiniLM-L6-v2'
            )
            
            # Load new processed documents
            data_path = Path(args.data_dir)
            if not data_path.exists():
                print(f"Data directory {data_path} does not exist")
                sys.exit(1)
            
            reader = SimpleDirectoryReader(
                input_dir=str(data_path),
                required_exts=[".txt"],
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
            store_type = config.get('vector_store', 'faiss') if isinstance(config, dict) else 'faiss'
            vector_client = VectorStoreClient(
                store_type=store_type,
                config=vector_config
            )
            
            if not vector_client.load_index():
                print(f"Failed to load existing index from {args.persist_dir}")
                sys.exit(1)
            
            # Add documents incrementally
            embed_model_info = {"model_name": embedder.model_name}
            chunking_params = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}
            
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
            config: dict[str, Any] = {}
            if Path(args.config).exists():
                with open(args.config, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        config = loaded_config
            
            # Initialize services for metadata
            embedder = EmbeddingService(
                model_name=config.get('embedding_model', 'all-MiniLM-L6-v2') if isinstance(config, dict) else 'all-MiniLM-L6-v2'
            )
            
            # Initialize vector store client and load index
            vector_config = {
                'index_dir': args.persist_dir
            }
            vector_client = VectorStoreClient(
                store_type=config.get('vector_store', 'faiss') if isinstance(config, dict) else 'faiss',
                config=vector_config
            )
            
            if not vector_client.load_index():
                print(f"Failed to load index from {args.persist_dir}")
                sys.exit(1)
            
            # Create snapshot with metadata
            embed_model_info = {"model_name": embedder.model_name}
            chunking_params = {
                "chunk_size": config.get('chunk_size', 1024),
                "chunk_overlap": config.get('chunk_overlap', 10)
            }
            
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
