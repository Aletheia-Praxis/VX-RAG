"""
CLI interface for VX-RAG.

Provides command-line tools for ingestion, indexing, and querying.
"""

import argparse
import sys
from pathlib import Path

# Import modules (will be available after setup)
# from rag.ingest import load_documents, preprocess_documents
# from rag.index import create_index
# from rag.query import create_query_engine, query_documents

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
    index_parser.add_argument("--persist-dir", type=str, default="../data/index",
                             help="Directory to persist index")

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
        # TODO: Implement indexing
        # docs = load_processed_documents()
        # index = create_index(docs, args.persist_dir)

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
