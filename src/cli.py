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
        MDIngestAdapter
    )
    from src.rag.services.duplicate_detection_service.service import DuplicateDetector
    from src.rag.services.chunker_service.service import Chunker
    
    start_time = time.time()
    data_path = Path(args.data_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory {data_path} does not exist")
        logger.error("Ingestion failed: data directory not found", data_dir=str(data_path))
        sys.exit(1)
    
    logger.info("Starting ingestion pipeline", data_dir=str(data_path))
    
    # Step 1: Parse documents
    print(f"\n[Step 1/3] Parsing documents from {data_path}...")
    
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
    print(f"\n[Step 2/3] Removing duplicates...")
    
    detector = DuplicateDetector(config_path=args.config)
    unique_docs = detector.remove_duplicates(all_docs)
    
    duplicates_removed = len(all_docs) - len(unique_docs)
    print(f"  Duplicates removed: {duplicates_removed}")
    print(f"  Unique documents: {len(unique_docs)}")
    
    # Step 3: Chunk documents
    print(f"\n[Step 3/3] Chunking documents...")
    
    chunker = Chunker(config_path=args.config)
    chunks = chunker.chunk_documents(unique_docs)
    
    chunking_stats = chunker.get_chunking_stats(chunks)
    print(f"  Total chunks: {chunking_stats['total_chunks']}")
    print(f"  Avg chunk length: {chunking_stats['avg_chunk_length']:.0f} chars")
    print(f"  Chunk distribution: {chunking_stats['chunk_size_distribution']}")
    
    # Summary
    duration = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Ingestion Summary:")
    print(f"{'='*50}")
    print(f"  Documents parsed:     {len(all_docs)}")
    print(f"  Duplicates removed:   {duplicates_removed}")
    print(f"  Unique documents:     {len(unique_docs)}")
    print(f"  Total chunks:         {len(chunks)}")
    print(f"  Duration:             {duration:.2f}s")
    print(f"{'='*50}\n")
    
    logger.info("Ingestion pipeline completed", 
               parsed=len(all_docs), 
               duplicates_removed=duplicates_removed,
               unique=len(unique_docs),
               chunks=len(chunks),
               duration_ms=duration * 1000)
    
    metrics.increment("ingestion_documents_parsed_total", len(all_docs))
    metrics.increment("ingestion_duplicates_removed_total", duplicates_removed)
    metrics.increment("ingestion_chunks_created_total", len(chunks))
    metrics.histogram("ingestion_pipeline_duration_ms", duration * 1000)


def handle_index(args: argparse.Namespace) -> None:
    """Handle index command."""
    print(f"Creating index at {args.persist_dir}")


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
