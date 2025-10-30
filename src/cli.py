"""
CLI interface for VX-RAG.

Provides command-line tools for ingestion, indexing, and querying.
"""

import argparse
import sys

from typing import Any

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

    args = parser.parse_args()

    if args.command == "ingest":
        print(f"Ingesting documents from {args.data_dir}")
        # TODO: Implement ingestion
        # docs = load_documents(Path(args.data_dir))
        # processed = preprocess_documents(docs)
        # Save processed documents

    elif args.command == "index":
        print(f"Creating index in {args.persist_dir}")
        try:
            import yaml  # type: ignore
            from pathlib import Path
            from llama_index.core import SimpleDirectoryReader
            
            # Load configuration
            config: dict[str, Any] = {}
            if Path(args.config).exists():
                with open(args.config, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    if isinstance(loaded_config, dict):
                        config = loaded_config
            
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
            index = vector_client.build_index(documents, embedder.embed_model)
            if index:
                vector_client.save_index()
                print(f"Index created and saved to {args.persist_dir}")
            else:
                print("Failed to create index")
                sys.exit(1)
                
        except Exception as e:
            print(f"Error creating index: {e}")
            sys.exit(1)

    elif args.command == "query":
        print(f"Querying: {args.query}")
        # TODO: Implement querying
        # index = load_index()
        # engine = create_query_engine(index)
        # result = query_documents(args.query, engine)
        # print(result)

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
