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
        try:
            from pathlib import Path
            from rag.services.ingest_service.service import PDFIngestAdapter, TXTIngestAdapter, MDIngestAdapter, save_processed_text
            
            data_path = Path(args.data_dir)
            processed_dir = Path("./data/processed")
            
            if not data_path.exists():
                print(f"Data directory {data_path} does not exist")
                sys.exit(1)
            
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
            
        except Exception as e:
            print(f"Error during ingestion: {e}")
            sys.exit(1)

    elif args.command == "index":
        print(f"Creating index in {args.persist_dir}")
        try:
            import yaml
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
        try:
            from rag.services.vectordb_service.service import VectorStoreClient
            from rag.services.retriever_service.service import RetrieverService
            from rag.services.llm_proxy.service import LLMProxy
            
            # Initialize vector store and load index
            vector_client = VectorStoreClient(store_type="faiss")
            if not vector_client.load_index():
                print("Failed to load index. Please run 'index' command first.")
                sys.exit(1)
            
            # Initialize retriever
            retriever = RetrieverService(vector_client.index)
            
            # Initialize LLM proxy
            llm_proxy = LLMProxy()
            
            # Retrieve documents
            retrieved_docs = retriever.retrieve(args.query, top_k=5)
            if not retrieved_docs:
                print("No relevant documents found.")
                sys.exit(0)
            
            # Generate response
            result = llm_proxy.generate_with_sources(args.query, retrieved_docs)
            
            # Print results
            print(f"\nQuery: {args.query}")
            print(f"\nLLM Response:\n{result['response']}")
            print(f"\nTop Results:")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. Score: {source['score']:.3f}")
                print(f"   Text: {source['text'][:200]}...")
                print()
            
        except Exception as e:
            print(f"Error during query: {e}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
