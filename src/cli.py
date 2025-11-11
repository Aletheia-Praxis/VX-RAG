"""
CLI interface for VX-RAG.

Provides command-line tools for ingestion, indexing, and querying.
"""

import argparse
import sys
import time
import asyncio
from pathlib import Path
from typing import Any, List, Dict

from src.utils.logging_config import get_logger, log_index_event
from src.utils.metrics import get_metrics

logger = get_logger("cli")
metrics = get_metrics()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="VX-RAG CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Run full ingestion pipeline")
    ingest_parser.add_argument("--data-dir", type=str, default="data/raw")
    ingest_parser.add_argument("--persist-dir", type=str, default="data/index")
    ingest_parser.add_argument("--config", type=str, default="config/settings.yaml")
    ingest_parser.add_argument("--background", "-b", action="store_true")

    # Index command
    index_parser = subparsers.add_parser("index", help="Create index")
    index_parser.add_argument("--persist-dir", type=str, default="./data/index")
    index_parser.add_argument("--data-dir", type=str, default="./data/processed")
    index_parser.add_argument("--config", type=str, default="./config/settings.yaml")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("query", type=str)
    query_parser.add_argument("--config", type=str, default="./config/settings.yaml")
    query_parser.add_argument("--top-k", type=int, default=5)

    # Update index command
    update_parser = subparsers.add_parser("update-index", help="Update existing index")
    update_parser.add_argument("--persist-dir", type=str, default="./data/index")
    update_parser.add_argument("--data-dir", type=str, default="./data/processed")
    update_parser.add_argument("--config", type=str, default="./config/settings.yaml")

    # Snapshot command
    snapshot_parser = subparsers.add_parser("snapshot", help="Create index snapshot")
    snapshot_parser.add_argument("--persist-dir", type=str, default="./data/index")
    snapshot_parser.add_argument("--name", type=str)
    snapshot_parser.add_argument("--config", type=str, default="./config/settings.yaml")

    # Verify snapshot command
    verify_parser = subparsers.add_parser("verify-snapshot", help="Verify snapshot integrity")
    verify_parser.add_argument("--persist-dir", type=str, default="./data/index")
    verify_parser.add_argument("--name", type=str, required=True)

    # Task status command
    status_parser = subparsers.add_parser("status", help="Check background task status")
    status_parser.add_argument("task_id", type=str)

    # Cancel task command
    cancel_parser = subparsers.add_parser("cancel", help="Cancel background task")
    cancel_parser.add_argument("task_id", type=str)

    # List tasks command
    list_parser = subparsers.add_parser("list-tasks", help="List all tasks")
    list_parser.add_argument("--filter", choices=["all", "pending", "running", "completed", "failed", "cancelled"], default="all")

    # Cleanup tasks command
    cleanup_parser = subparsers.add_parser("cleanup", help="Cleanup old tasks")
    cleanup_parser.add_argument("--all-completed", action="store_true")
    cleanup_parser.add_argument("--all-failed", action="store_true")
    cleanup_parser.add_argument("--all-cancelled", action="store_true")

    # Metrics command
    metrics_parser = subparsers.add_parser("metrics", help="Display system metrics")
    metrics_parser.add_argument("--format", choices=["text", "json"], default="text")

    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark search strategies")
    benchmark_parser.add_argument("query", type=str)
    benchmark_parser.add_argument("--config", type=str, default="./config/settings.yaml")
    benchmark_parser.add_argument("--alphas", type=str, default="0.0,0.3,0.5,0.7,1.0")

    # Clean boilerplate command
    clean_bp_parser = subparsers.add_parser("clean-boilerplate", help="Remove boilerplate from documents")
    clean_bp_parser.add_argument("--data-dir", type=str, default="./data/processed")
    clean_bp_parser.add_argument("--config", type=str, default="./config/settings.yaml")
    clean_bp_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "ingest":
        handle_ingest(args)
    elif args.command == "index":
        handle_index(args)
    elif args.command == "query":
        handle_query(args)
    elif args.command == "update-index":
        handle_update_index(args)
    elif args.command == "snapshot":
        handle_snapshot(args)
    elif args.command == "verify-snapshot":
        handle_verify_snapshot(args)
    elif args.command == "status":
        handle_status(args)
    elif args.command == "cancel":
        handle_cancel(args)
    elif args.command == "list-tasks":
        handle_list_tasks(args)
    elif args.command == "cleanup":
        handle_cleanup(args)
    elif args.command == "metrics":
        handle_metrics(args)
    elif args.command == "benchmark":
        handle_benchmark(args)
    elif args.command == "clean-boilerplate":
        handle_clean_boilerplate(args)
    else:
        parser.print_help()
        sys.exit(1)


def handle_ingest(args: argparse.Namespace) -> None:
    """Handle ingest command."""
    from src.rag.services.ingest_service.service import (
        PDFIngestAdapter, 
        TXTIngestAdapter, 
        MDIngestAdapter,
        save_processed_text
    )
    from src.rag.services.duplicate_detection_service.service import DuplicateDetector
    from src.rag.services.chunker_service.service import Chunker
    
    start_time = time.time()
    data_path = Path(args.data_dir)
    processed_dir = data_path / "processed"
    
    if not data_path.exists():
        print(f"Error: Data directory {data_path} does not exist")
        logger.error("Ingestion failed: data directory not found", data_dir=str(data_path))
        sys.exit(1)
    
    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting ingestion pipeline", data_dir=str(data_path))
    
    # Step 1: Parse documents
    print(f"\n[Step 1/4] Parsing documents from {data_path}...")
    
    pdf_adapter = PDFIngestAdapter()
    txt_adapter = TXTIngestAdapter()
    md_adapter = MDIngestAdapter()
    
    pdf_docs = pdf_adapter.load_data(str(data_path / "pdf"))
    txt_docs = txt_adapter.load_data(str(data_path / "txt"))
    md_docs = md_adapter.load_data(str(data_path / "md"))
    
    all_docs = pdf_docs + txt_docs + md_docs
    
    print(f"  PDF: {len(pdf_docs)}, TXT: {len(txt_docs)}, MD: {len(md_docs)}")
    print(f"  Total parsed: {len(all_docs)} documents")
    
    # Step 2: Remove duplicates
    print(f"\n[Step 2/4] Removing duplicates...")
    
    detector = DuplicateDetector(config_path=args.config)
    unique_docs = detector.remove_duplicates(all_docs)
    
    duplicates_removed = len(all_docs) - len(unique_docs)
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Unique documents: {len(unique_docs)}")
    
    # Step 3: Save processed documents
    print(f"\n[Step 3/4] Saving processed documents to {processed_dir}...")
    
    saved_count = save_processed_text(unique_docs, processed_dir)
    print(f"  Saved files: {saved_count}")
    
    # Step 4: Chunk documents
    print(f"\n[Step 4/4] Chunking documents...")
    
    chunker = Chunker(config_path=args.config)
    chunks = chunker.chunk_documents(unique_docs)
    
    chunking_stats = chunker.get_chunking_stats(chunks)
    print(f"  Total chunks: {chunking_stats['total_chunks']}")
    print(f"  Avg chunk length: {chunking_stats['avg_chunk_length']:.0f} chars")
    print(f"  Chunk distribution: {chunking_stats['chunk_size_distribution']}")
    
    # Save chunks to file for indexing
    import json
    chunks_file = data_path / "processed" / "chunks.json"
    
    # Filter out non-serializable LlamaIndex objects
    def make_serializable(obj):
        """Recursively remove non-serializable objects."""
        if isinstance(obj, dict):
            return {
                k: make_serializable(v) 
                for k, v in obj.items() 
                if k not in ['node_info', 'relationships', 'excluded_llm_metadata_keys', 
                             'excluded_embed_metadata_keys', 'metadata_seperator', 
                             'metadata_template', 'text_template']
            }
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    try:
        serializable_chunks = make_serializable(chunks)
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)
        print(f"  Saved chunks to: {chunks_file}")
    except TypeError as e:
        logger.error(f"Failed to serialize chunks: {e}")
        print(f"  Warning: Could not save chunks (serialization error)")
    
    # Summary
    duration = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Ingestion Summary:")
    print(f"{'='*50}")
    print(f"  Documents parsed:     {len(all_docs)}")
    print(f"  Duplicates removed:   {duplicates_removed}")
    print(f"  Unique documents:     {len(unique_docs)}")
    print(f"  Saved to disk:        {saved_count}")
    print(f"  Total chunks:         {len(chunks)}")
    print(f"  Chunks file:          {chunks_file}")
    print(f"  Duration:             {duration:.2f}s")
    print(f"{'='*50}\n")
    
    logger.info("Ingestion pipeline completed", 
               parsed=len(all_docs), 
               duplicates_removed=duplicates_removed,
               unique=len(unique_docs),
               saved=saved_count,
               chunks=len(chunks),
               duration_ms=duration * 1000)
    
    metrics.increment("ingestion_documents_parsed_total", len(all_docs))
    metrics.increment("ingestion_duplicates_removed_total", duplicates_removed)
    metrics.increment("ingestion_documents_saved_total", saved_count)
    metrics.increment("ingestion_chunks_created_total", len(chunks))
    metrics.histogram("ingestion_pipeline_duration_ms", duration * 1000)


def handle_index(args: argparse.Namespace) -> None:
    """Handle index command - create embeddings and build indexes."""
    import json
    from src.rag.services.embedder_service.service import EmbeddingService
    
    start_time = time.time()
    data_dir = Path(args.data_dir)
    persist_dir = Path(args.persist_dir)
    chunks_file = data_dir / "chunks.json"
    
    if not chunks_file.exists():
        print(f"Error: Chunks file not found at {chunks_file}")
        print("Please run 'ingest' command first to create chunks.")
        logger.error("Indexing failed: chunks file not found", chunks_file=str(chunks_file))
        sys.exit(1)
    
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting indexing pipeline", persist_dir=str(persist_dir))
    
    # Step 1: Load chunks
    print(f"\n[Step 1/3] Loading chunks from {chunks_file}...")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"  Loaded chunks: {len(chunks)}")
    
    # Step 2: Generate embeddings
    print(f"\n[Step 2/3] Generating embeddings...")
    
    embedder = EmbeddingService(config_path=args.config)
    
    # Extract text from chunks
    chunk_texts = [chunk.get('text', '') for chunk in chunks]
    
    print(f"  Embedding {len(chunk_texts)} chunks...")
    embeddings = embedder.embed(chunk_texts)
    
    print(f"  Generated embeddings: {len(embeddings)}")
    print(f"  Embedding dimension: {len(embeddings[0]) if embeddings else 0}")
    
    # Save embeddings
    embeddings_file = persist_dir / "embeddings.json"
    with open(embeddings_file, 'w', encoding='utf-8') as f:
        json.dump(embeddings, f)
    print(f"  Saved embeddings to: {embeddings_file}")
    
    # Summary
    duration = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Indexing Summary:")
    print(f"{'='*50}")
    print(f"  Chunks processed:     {len(chunks)}")
    print(f"  Embeddings created:   {len(embeddings)}")
    print(f"  Embedding dim:        {len(embeddings[0]) if embeddings else 0}")
    print(f"  Persist dir:          {persist_dir}")
    print(f"  Duration:             {duration:.2f}s")
    print(f"{'='*50}\n")
    
    logger.info("Indexing pipeline completed",
               chunks=len(chunks),
               embeddings=len(embeddings),
               duration_ms=duration * 1000)
    
    metrics.increment("indexing_chunks_processed_total", len(chunks))
    metrics.increment("indexing_embeddings_created_total", len(embeddings))
    metrics.histogram("indexing_pipeline_duration_ms", duration * 1000)


def handle_query(args: argparse.Namespace) -> None:
    """Handle query command."""
    print(f"Querying: {args.query}")


def handle_update_index(args: argparse.Namespace) -> None:
    """Handle update-index command."""
    print(f"Updating index in {args.persist_dir}")


def handle_snapshot(args: argparse.Namespace) -> None:
    """Handle snapshot command."""
    print(f"Creating snapshot in {args.persist_dir}")


def handle_verify_snapshot(args: argparse.Namespace) -> None:
    """Handle verify-snapshot command."""
    print(f"Verifying snapshot '{args.name}' in {args.persist_dir}")


def handle_status(args: argparse.Namespace) -> None:
    """Handle status command."""
    print(f"Checking status of task {args.task_id}")


def handle_cancel(args: argparse.Namespace) -> None:
    """Handle cancel command."""
    print(f"Cancelling task {args.task_id}")


def handle_list_tasks(args: argparse.Namespace) -> None:
    """Handle list-tasks command."""
    print("Listing all tasks")


def handle_cleanup(args: argparse.Namespace) -> None:
    """Handle cleanup command."""
    print("Cleaning up old tasks")


def handle_metrics(args: argparse.Namespace) -> None:
    """Handle metrics command."""
    print("Displaying system metrics")


def handle_benchmark(args: argparse.Namespace) -> None:
    """Handle benchmark command."""
    print(f"Benchmarking with query: {args.query}")


def handle_clean_boilerplate(args: argparse.Namespace) -> None:
    """Handle clean-boilerplate command."""
    print(f"Cleaning boilerplate from {args.data_dir}")


if __name__ == "__main__":
    main()
