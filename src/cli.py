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
    query_parser.add_argument("--persist-dir", type=str, default="./data/index")
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
    """Handle index command - create embeddings and build FAISS + BM25 indexes."""
    import json
    import numpy as np
    from llama_index.core.schema import Document, TextNode
    from src.rag.services.embedder_service.service import EmbeddingService
    from src.rag.services.vectordb_service.service import VectorStoreClient
    from src.rag.services.bm25_service.service import BM25Service
    
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
    print(f"\n[Step 1/4] Loading chunks from {chunks_file}...")
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    print(f"  Loaded chunks: {len(chunks)}")
    logger.info(f"Loaded {len(chunks)} chunks from {chunks_file}")
    
    # Step 2: Generate embeddings (Module 5 - EmbedderService)
    print(f"\n[Step 2/4] Generating embeddings...")
    
    embedder = EmbeddingService(config_path=args.config)
    
    # Extract text from chunks
    chunk_texts = [chunk.get('text', '') for chunk in chunks]
    chunk_ids = [chunk.get('id', f"chunk_{i}") for i, chunk in enumerate(chunks)]
    
    print(f"  Embedding {len(chunk_texts)} chunks...")
    embeddings_list = embedder.embed_batch(chunk_texts)
    
    # Convert to numpy array for FAISS
    embeddings_array = np.array(embeddings_list, dtype=np.float32)
    
    embedding_dim = len(embeddings_list[0]) if embeddings_list else 0
    print(f"  Generated embeddings: {len(embeddings_list)}")
    print(f"  Embedding dimension: {embedding_dim}")
    logger.info(f"Generated {len(embeddings_list)} embeddings with dimension {embedding_dim}")
    
    # Step 3: Build FAISS vector index (Module 6 - VectorStoreClient)
    print(f"\n[Step 3/4] Building FAISS vector index...")
    
    # Convert chunks to LlamaIndex Document objects for VectorStoreClient
    documents = []
    for chunk in chunks:
        doc = Document(
            text=chunk.get('text', ''),
            metadata=chunk.get('metadata', {}),
            id_=chunk.get('id', '')
        )
        documents.append(doc)
    
    # Initialize VectorStoreClient with persist directory
    vector_store_config = {
        'index_dir': str(persist_dir / "faiss_index")
    }
    vector_client = VectorStoreClient(store_type="faiss", config=vector_store_config)
    
    # Build FAISS index
    vector_index = vector_client.build_index(
        documents=documents,
        embed_model=embedder.embed_model
    )
    
    # Save FAISS index
    vector_client.save_index(create_backup=False)
    
    faiss_index_path = persist_dir / "faiss_index"
    print(f"  FAISS index built: {len(documents)} vectors")
    print(f"  Saved to: {faiss_index_path}")
    logger.info(f"Built and saved FAISS index to {faiss_index_path}")
    
    # Step 4: Build BM25 index (Module 7 - BM25Service)
    print(f"\n[Step 4/4] Building BM25 index...")
    
    # Initialize BM25Service with persist directory
    bm25_index_path = persist_dir / "bm25_index"
    bm25_service = BM25Service(index_dir=str(bm25_index_path), config_path=args.config)
    
    # Build BM25 index with documents
    bm25_service.build_index(documents)
    
    # Save BM25 index
    bm25_service.save_index()
    
    print(f"  BM25 index built: {len(documents)} documents")
    print(f"  Saved to: {bm25_index_path}")
    logger.info(f"Built and saved BM25 index to {bm25_index_path}")
    
    # Save metadata for later use
    metadata = {
        'num_chunks': len(chunks),
        'num_embeddings': len(embeddings_list),
        'embedding_dim': embedding_dim,
        'faiss_index_path': str(faiss_index_path),
        'bm25_index_path': str(bm25_index_path),
        'indexed_at': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_file = persist_dir / "index_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  Saved metadata to: {metadata_file}")
    
    # Summary
    duration = time.time() - start_time
    
    print(f"\n{'='*50}")
    print(f"Indexing Summary:")
    print(f"{'='*50}")
    print(f"  Chunks processed:     {len(chunks)}")
    print(f"  Embeddings created:   {len(embeddings_list)}")
    print(f"  Embedding dim:        {embedding_dim}")
    print(f"  FAISS index:          {faiss_index_path}")
    print(f"  BM25 index:           {bm25_index_path}")
    print(f"  Metadata:             {metadata_file}")
    print(f"  Duration:             {duration:.2f}s")
    print(f"{'='*50}\n")
    
    logger.info("Indexing pipeline completed",
               chunks=len(chunks),
               embeddings=len(embeddings_list),
               faiss_index=str(faiss_index_path),
               bm25_index=str(bm25_index_path),
               duration_ms=duration * 1000)
    
    metrics.increment("indexing_chunks_processed_total", len(chunks))
    metrics.increment("indexing_embeddings_created_total", len(embeddings_list))
    metrics.histogram("indexing_pipeline_duration_ms", duration * 1000)


def handle_query(args: argparse.Namespace) -> None:
    """Handle query command - search documents using hybrid retrieval + reranking."""
    import json
    from llama_index.core import StorageContext, load_index_from_storage
    from src.rag.services.embedder_service.service import EmbeddingService
    from src.rag.services.vectordb_service.service import VectorStoreClient
    from src.rag.services.bm25_service.service import BM25Service
    from src.rag.services.retriever_service.service import RetrieverService
    from src.rag.services.reranker_service.service import RerankerService
    from src.rag.services.hybrid_search_service.service import HybridSearchService
    from src.rag.services.assembler_service.service import ContextAssembler
    
    start_time = time.time()
    persist_dir = Path(args.persist_dir)
    query = args.query
    top_k = args.top_k if hasattr(args, 'top_k') else 5
    
    # Validate index exists
    faiss_index_path = persist_dir / "faiss_index"
    bm25_index_path = persist_dir / "bm25_index"
    metadata_file = persist_dir / "index_metadata.json"
    
    if not faiss_index_path.exists() or not bm25_index_path.exists():
        print(f"Error: Indexes not found in {persist_dir}")
        print("Please run 'index' command first to create indexes.")
        logger.error("Query failed: indexes not found", persist_dir=str(persist_dir))
        sys.exit(1)
    
    logger.info("Starting query pipeline", query=query, top_k=top_k)
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    # Step 1: Load indexes (Module 8 - RetrieverService setup)
    print(f"[Step 1/5] Loading indexes...")
    
    # Load FAISS index
    embedder = EmbeddingService(config_path=args.config)
    vector_client = VectorStoreClient(store_type="faiss", config={'index_dir': str(faiss_index_path)})
    vector_client.load_index(embed_model=embedder.embed_model)
    
    print(f"  FAISS index loaded from: {faiss_index_path}")
    logger.info(f"Loaded FAISS index from {faiss_index_path}")
    
    # Load BM25 index
    bm25_service = BM25Service(index_dir=str(bm25_index_path), config_path=args.config)
    bm25_service.load_index()
    
    print(f"  BM25 index loaded from: {bm25_index_path}")
    logger.info(f"Loaded BM25 index from {bm25_index_path}")
    
    # Step 2: Initialize RetrieverService (Module 8)
    print(f"\n[Step 2/5] Initializing retriever...")
    
    retriever = RetrieverService(index=vector_client.index, config_path=args.config)
    if vector_client.index:
        retriever.set_index(vector_client.index)
        retriever.bm25_retriever = bm25_service.bm25_retriever
    else:
        print(f"  ERROR: Failed to load FAISS index")
        logger.error("FAISS index is None after loading")
        sys.exit(1)
    
    print(f"  Retriever initialized (vector + BM25)")
    logger.info("Retriever service initialized")
    
    # Step 3: Retrieve candidates (Module 8 - hybrid retrieval)
    print(f"\n[Step 3/5] Retrieving candidates...")
    
    # Retrieve from both sources
    initial_k = top_k * 4  # Over-retrieve for reranking
    
    try:
        vector_results = retriever.retrieve(query, top_k=initial_k, search_type="semantic")
        print(f"  Vector search: {len(vector_results)} candidates")
        
        bm25_results = retriever.retrieve(query, top_k=initial_k, search_type="keyword")
        print(f"  BM25 search: {len(bm25_results)} candidates")
        
        logger.info(f"Retrieved {len(vector_results)} vector + {len(bm25_results)} BM25 candidates")
    except Exception as e:
        print(f"  ERROR: Retrieval failed - {e}")
        logger.error(f"Retrieval failed: {e}")
        sys.exit(1)
    
    # Step 4: Rerank candidates (Module 9 - RerankerService)
    print(f"\n[Step 4/5] Reranking candidates...")
    
    reranker = RerankerService(config_path=args.config)
    
    # Combine results (simple merge for reranking)
    all_candidates = vector_results + bm25_results
    
    # Remove duplicates by node_id
    seen_ids = set()
    unique_candidates = []
    for doc in all_candidates:
        node_id = doc.get('node_id', doc.get('id', ''))
        if node_id not in seen_ids:
            seen_ids.add(node_id)
            unique_candidates.append(doc)
    
    print(f"  Unique candidates: {len(unique_candidates)}")
    
    # Rerank
    reranked_results = reranker.rerank(query, unique_candidates, top_k=top_k)
    
    print(f"  Reranked to top {len(reranked_results)} results")
    logger.info(f"Reranked {len(unique_candidates)} candidates to top {len(reranked_results)}")
    
    # Step 5: Assemble context (Module 11 - ContextAssembler)
    print(f"\n[Step 5/5] Assembling context...")
    
    assembler = ContextAssembler(config_path=args.config)
    context_payload = assembler.assemble_context(
        query=query,
        documents=reranked_results,
        token_budget=4000,  # Default budget
        max_items=top_k
    )
    
    print(f"  Context assembled: {len(context_payload.context)} items")
    print(f"  Estimated tokens: ~{context_payload.total_tokens_estimate()}")
    logger.info(f"Context assembled: {len(context_payload.context)} items")
    
    # Display results
    duration = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"{'='*60}\n")
    
    for i, item in enumerate(context_payload.context, 1):
        score_str = f"{item.score:.4f}" if item.score is not None else "N/A"
        print(f"[{i}] Score: {score_str}")
        print(f"    ID: {item.id}")
        print(f"    Preview: {item.text[:150]}...")
        if item.meta:
            print(f"    Metadata: {json.dumps(item.meta, indent=8)}")
        print()
    
    print(f"{'='*60}")
    print(f"Query Summary:")
    print(f"{'='*60}")
    print(f"  Query:              {query}")
    print(f"  Results returned:   {len(context_payload.context)}")
    print(f"  Token estimate:     ~{context_payload.total_tokens_estimate()}")
    print(f"  Duration:           {duration:.2f}s")
    print(f"{'='*60}\n")
    
    logger.info("Query pipeline completed",
               query=query,
               results=len(context_payload.context),
               tokens_estimate=context_payload.total_tokens_estimate(),
               duration_ms=duration * 1000)
    
    metrics.increment("query_requests_total")
    metrics.histogram("query_duration_ms", duration * 1000)
    metrics.histogram("query_results_count", len(context_payload.context))


def handle_update_index(args: argparse.Namespace) -> None:
    """Handle update-index command - add new documents to existing index incrementally."""
    import json
    from llama_index.core.schema import Document
    from src.rag.services.embedder_service.service import EmbeddingService
    from src.rag.services.vectordb_service.service import VectorStoreClient
    from src.rag.services.bm25_service.service import BM25Service
    from src.rag.services.ingest_service.service import (
        PDFIngestAdapter,
        TXTIngestAdapter,
        MDIngestAdapter
    )
    from src.rag.services.duplicate_detection_service.service import DuplicateDetector
    from src.rag.services.chunker_service.service import Chunker
    
    start_time = time.time()
    persist_dir = Path(args.persist_dir)
    data_dir = Path(args.data_dir)
    
    # Validate paths
    faiss_index_path = persist_dir / "faiss_index"
    bm25_index_path = persist_dir / "bm25_index"
    
    if not faiss_index_path.exists() or not bm25_index_path.exists():
        print(f"Error: Indexes not found in {persist_dir}")
        print("Please run 'index' command first to create initial indexes.")
        logger.error("Update index failed: indexes not found", persist_dir=str(persist_dir))
        sys.exit(1)
    
    logger.info("Starting incremental index update", persist_dir=str(persist_dir), data_dir=str(data_dir))
    print(f"\n{'='*60}")
    print(f"Incremental Index Update")
    print(f"{'='*60}\n")
    
    # Step 1: Parse new documents
    print(f"[Step 1/4] Parsing new documents from {data_dir}...")
    
    pdf_adapter = PDFIngestAdapter(config_path=args.config)
    txt_adapter = TXTIngestAdapter(config_path=args.config)
    md_adapter = MDIngestAdapter(config_path=args.config)
    
    new_docs = []
    
    # Parse new PDFs
    if (data_dir / "pdf").exists():
        pdf_files = list((data_dir / "pdf").glob("*.pdf"))
        for pdf_file in pdf_files:
            try:
                docs = pdf_adapter.load_data(str(pdf_file))
                new_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to parse {pdf_file.name}: {e}")
    
    print(f"  Found {len(new_docs)} new documents")
    
    if not new_docs:
        print("  No new documents to add.")
        return
    
    # Step 2: Remove duplicates
    print(f"\n[Step 2/4] Checking for duplicates...")
    deduplicator = DuplicateDetector()
    unique_docs = deduplicator.remove_duplicates(new_docs)
    print(f"  Unique new documents: {len(unique_docs)}")
    
    # Step 3: Chunk documents
    print(f"\n[Step 3/4] Chunking documents...")
    chunker = Chunker(config_path=args.config)
    chunks = chunker.chunk_documents(unique_docs)
    print(f"  Created {len(chunks)} chunks")
    
    # Step 4: Update indexes
    print(f"\n[Step 4/4] Updating indexes...")
    
    # Load embedder
    embedder = EmbeddingService(config_path=args.config)
    
    # Convert chunks to Documents
    documents = []
    for chunk in chunks:
        doc = Document(
            text=chunk.get('text', ''),
            metadata=chunk.get('metadata', {}),
            id_=chunk.get('id', '')
        )
        documents.append(doc)
    
    # Load and update FAISS index
    vector_client = VectorStoreClient(store_type="faiss", config={'index_dir': str(faiss_index_path)})
    vector_client.load_index(embed_model=embedder.embed_model)
    
    if vector_client.index:
        success = vector_client.add_documents_incremental(documents, embedder.embed_model)
        if success:
            vector_client.save_index(create_backup=True)
            print(f"  FAISS index updated: +{len(documents)} documents")
        else:
            print(f"  ERROR: Failed to update FAISS index")
            logger.error("Failed to update FAISS index")
    
    # Load and update BM25 index
    bm25_service = BM25Service(index_dir=str(bm25_index_path), config_path=args.config)
    bm25_service.load_index()
    
    # BM25 requires full rebuild (limitation of the current implementation)
    print(f"  Note: BM25 index requires full rebuild for updates")
    print(f"  Run 'index' command to rebuild BM25 with new documents")
    
    # Summary
    duration = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Update Summary:")
    print(f"{'='*60}")
    print(f"  New documents added:  {len(unique_docs)}")
    print(f"  New chunks created:   {len(chunks)}")
    print(f"  FAISS index updated:  Yes")
    print(f"  BM25 index updated:   Requires manual rebuild")
    print(f"  Duration:             {duration:.2f}s")
    print(f"{'='*60}\n")
    
    logger.info("Incremental index update completed",
               new_docs=len(unique_docs),
               new_chunks=len(chunks),
               duration_ms=duration * 1000)
    
    metrics.increment("index_update_documents_added", len(unique_docs))
    metrics.histogram("index_update_duration_ms", duration * 1000)


def handle_snapshot(args: argparse.Namespace) -> None:
    """Handle snapshot command - create versioned backup of indexes."""
    from src.rag.services.vectordb_service.service import VectorStoreClient
    from src.rag.services.embedder_service.service import EmbeddingService
    
    start_time = time.time()
    persist_dir = Path(args.persist_dir)
    snapshot_name = args.name if hasattr(args, 'name') and args.name else None
    
    faiss_index_path = persist_dir / "faiss_index"
    
    if not faiss_index_path.exists():
        print(f"Error: Index not found at {faiss_index_path}")
        logger.error("Snapshot failed: index not found", persist_dir=str(persist_dir))
        sys.exit(1)
    
    logger.info("Creating snapshot", persist_dir=str(persist_dir), snapshot_name=snapshot_name)
    print(f"\n{'='*60}")
    print(f"Creating Index Snapshot")
    print(f"{'='*60}\n")
    
    # Load embedder for model info
    embedder = EmbeddingService(config_path=args.config)
    embed_model_info = embedder.get_model_info()
    
    # Load vector store
    vector_client = VectorStoreClient(store_type="faiss", config={'index_dir': str(faiss_index_path)})
    vector_client.load_index(embed_model=embedder.embed_model)
    
    if not vector_client.index:
        print(f"ERROR: Failed to load index")
        logger.error("Failed to load index for snapshot")
        sys.exit(1)
    
    # Create snapshot
    snapshot_path = vector_client.create_snapshot(
        snapshot_name=snapshot_name,
        embed_model_info=embed_model_info
    )
    
    duration = time.time() - start_time
    
    print(f"  Snapshot created: {snapshot_path}")
    print(f"  Duration: {duration:.2f}s")
    print(f"{'='*60}\n")
    
    logger.info("Snapshot created successfully",
               snapshot_path=str(snapshot_path),
               duration_ms=duration * 1000)
    
    metrics.increment("snapshots_created_total")
    metrics.histogram("snapshot_creation_duration_ms", duration * 1000)


if __name__ == "__main__":
    main()
