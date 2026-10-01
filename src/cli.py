"""
CLI interface for VX-RAG.

Provides command-line tools for ingestion, indexing, and querying.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb

from src.rag.orchestrator import (
    get_orchestrator,
    load_intermediate_nodes,
    persist_intermediate_nodes,
)
from src.utils.logging_config import get_logger, request_context
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

    # Index command
    index_parser = subparsers.add_parser("index", help="Create index")
    index_parser.add_argument("--data-dir", type=str, default="data/raw")
    index_parser.add_argument("--persist-dir", type=str, default="data/index")
    index_parser.add_argument("--config", type=str, default="config/settings.yaml")

    # Query command
    query_parser = subparsers.add_parser("query", help="Query the system")
    query_parser.add_argument("query", type=str)
    query_parser.add_argument("--persist-dir", type=str, default="data/index")
    query_parser.add_argument("--config", type=str, default="config/settings.yaml")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.add_argument(
        "--search-type",
        choices=["hybrid", "semantic", "keyword"],
        default="hybrid",
        help="Search modality: hybrid (default), semantic (vector only), or keyword (BM25 only)",
    )

    # Serve MCP command
    serve_parser = subparsers.add_parser("serve", help="Start MCP server")
    serve_parser.add_argument("--host", type=str, default="localhost")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--config", type=str, default="config/settings.yaml")
    serve_parser.add_argument(
        "--persist-dir",
        type=str,
        default="data/index",
        help="Directory containing persisted FAISS and BM25 indices",
    )
    serve_parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    serve_parser.add_argument("--cors", action="store_true")
    serve_parser.add_argument("--allowed-origins", type=str, default="*")

    args = parser.parse_args()

    command_dispatch = {
        "ingest": handle_ingest,
        "index": handle_index,
        "query": handle_query,
        "serve": handle_serve,
    }

    if args.command in command_dispatch:
        with request_context():
            command_dispatch[args.command](args)
    else:
        parser.print_help()
        sys.exit(1)


def handle_ingest(args: argparse.Namespace) -> None:
    """Handle ingest command."""
    with request_context():
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            print(f"Error: Data directory not found at {data_dir}", file=sys.stderr)
            sys.exit(1)

        start_time = time.time()
        orchestrator = get_orchestrator(config_path=args.config, persist_dir=args.persist_dir)
        print(f"Ingesting documents from {args.data_dir}...")
        nodes = orchestrator.ingest(args.data_dir)

        # Save nodes to DuckDB in processed directory to avoid Windows Defender file locking on plaintext JSON
        processed_dir = data_dir.parent / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)
        try:
            nodes_file = persist_intermediate_nodes(nodes=list(nodes), directory=processed_dir)
        except (duckdb.Error, OSError) as err:
            print(f"Storage error persisting intermediate nodes: {err}", file=sys.stderr)
            sys.exit(1)

        duration = time.time() - start_time
        print(f"Ingestion complete. Processed {len(nodes)} nodes in {duration:.2f}s.")
        print(f"Saved to {nodes_file}")


def handle_index(args: argparse.Namespace) -> None:
    """Handle index command."""
    with request_context():
        start_time = time.time()
        orchestrator = get_orchestrator(config_path=args.config, persist_dir=args.persist_dir)

        processed_dir = Path(args.data_dir).parent / "processed"
        nodes_db = processed_dir / "nodes.duckdb"
        nodes_json = processed_dir / "nodes.json"

        try:
            nodes = load_intermediate_nodes(directory=processed_dir, legacy_fallback=True)
        except FileNotFoundError:
            print(
                f"Nodes file not found at {nodes_db} or {nodes_json}. Run 'ingest' first.",
                file=sys.stderr,
            )
            sys.exit(1)
        except (duckdb.Error, OSError) as err:
            print(f"Storage error loading intermediate nodes: {err}", file=sys.stderr)
            sys.exit(1)

        print(f"Building index from {len(nodes)} nodes...")
        try:
            orchestrator.index_nodes(nodes)
        except (duckdb.Error, OSError) as err:
            print(f"Storage error updating index: {err}", file=sys.stderr)
            sys.exit(1)

        duration = time.time() - start_time
        print(f"Indexing complete in {duration:.2f}s.")


def handle_query(args: argparse.Namespace) -> None:
    """Handle query command."""
    with request_context():
        start_time = time.time()
        orchestrator = get_orchestrator(config_path=args.config, persist_dir=args.persist_dir)

        print(f"Querying for: {args.query}")
        # Execute query with requested search modality (defaults to search_type="hybrid")
        result = orchestrator.query(
            query=args.query, top_k=args.top_k, search_type=args.search_type
        )

        for i, item in enumerate(result.context, 1):
            print(f"\n[{i}] Score: {item.score:.4f}")
            print(f"    ID: {item.id}")
            print(f"    Preview: {item.text[:150]}...")

        duration = time.time() - start_time
        print(f"\nQuery complete. Returned {len(result.context)} results in {duration:.2f}s.")


def handle_serve(args: argparse.Namespace) -> None:
    """Handle serve command."""
    from src.utils.logging_config import configure_console_stream

    is_stdio = args.transport == "stdio"
    banner_file = sys.stderr if is_stdio else sys.stdout

    if is_stdio:
        configure_console_stream(sys.stderr)

    print(f"Starting MCP server with transport: {args.transport}", file=banner_file)

    with request_context():
        import asyncio

        from src.mcp.server import configure_server, run_http, run_stdio

        configure_server(config_path=args.config, persist_dir=args.persist_dir)

        try:
            if args.transport == "stdio":
                run_stdio(config_path=args.config, persist_dir=args.persist_dir)
            elif args.transport == "http":
                asyncio.run(
                    run_http(
                        host=args.host,
                        port=args.port,
                        config_path=args.config,
                        persist_dir=args.persist_dir,
                    )
                )
        except KeyboardInterrupt:
            print("\nServer shutdown complete", file=banner_file)


if __name__ == "__main__":
    main()
