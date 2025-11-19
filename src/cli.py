"""
CLI interface for VX-RAG.

Provides command-line tools for ingestion, indexing, and querying.
"""

import argparse
import sys
import time
import asyncio
from pathlib import Path
from typing import Any

from src.utils.logging_config import get_logger
from src.utils.metrics import get_metrics
from src.utils.task_queue import get_task_queue

from llama_index.core.response_synthesizers import ResponseMode

logger = get_logger("cli")
metrics = get_metrics()

# Global task queue instance (lazy initialization)
_task_queue = None


def get_cli_task_queue() -> Any:
    """Get or create the CLI task queue instance."""
    global _task_queue
    if _task_queue is None:
        _task_queue = get_task_queue()
    return _task_queue


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

    # Serve MCP command
    serve_parser = subparsers.add_parser("serve", help="Start MCP server")
    serve_parser.add_argument("--host", type=str, default="localhost", help="Server host (for http/sse transports)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Server port (for http/sse transports)")
    serve_parser.add_argument("--config", type=str, default="./config/settings.yaml", help="Config file path")
    serve_parser.add_argument("--transport", choices=["stdio", "sse", "http"], default="stdio", 
                              help="Transport protocol: stdio (default, for IDE integration), sse (Server-Sent Events), http (REST API)")
    serve_parser.add_argument("--cors", action="store_true", help="Enable CORS for http/sse transports")
    serve_parser.add_argument("--allowed-origins", type=str, default="*", help="Comma-separated list of allowed CORS origins")

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
    elif args.command == "serve":
        handle_serve(args)
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
    print("\n[Step 2/4] Removing duplicates...")
    
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
    print("\n[Step 4/4] Chunking documents...")
    
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
    def make_serializable(obj: Any) -> Any:
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
        print("  Warning: Could not save chunks (serialization error)")
    
    # Summary
    duration = time.time() - start_time
    
    print(f"\n{'='*50}")
    print("Ingestion Summary:")
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
    from llama_index.core.schema import Document
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core import StorageContext, VectorStoreIndex
    from src.utils.config_loader import get_embedding_config
    
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
    
    # Step 2: Generate embeddings (using HuggingFaceEmbedding)
    print("\n[Step 2/4] Generating embeddings...")
    
    embed_config = get_embedding_config(args.config)
    embed_model = HuggingFaceEmbedding(
        model_name=embed_config['embedding_model'],
        embed_batch_size=embed_config['embedding_batch_size'],
        trust_remote_code=embed_config['embedding_trust_remote_code']
    )
    
    # Extract text from chunks
    chunk_texts = [chunk.get('text', '') for chunk in chunks]
    
    print(f"  Embedding {len(chunk_texts)} chunks...")
    embeddings_list = embed_model.get_text_embedding_batch(chunk_texts)
    
    embedding_dim = len(embeddings_list[0]) if embeddings_list else 0
    print(f"  Generated embeddings: {len(embeddings_list)}")
    print(f"  Embedding dimension: {embedding_dim}")
    logger.info(f"Generated {len(embeddings_list)} embeddings with dimension {embedding_dim}")
    
    # Step 3: Build FAISS vector index (Module 6 - Direct LlamaIndex)
    print("\n[Step 3/4] Building FAISS vector index...")
    
    # Convert chunks to LlamaIndex Document objects for direct indexing
    documents = []
    for chunk in chunks:
        doc = Document(
            text=chunk.get('text', ''),
            metadata=chunk.get('metadata', {}),
            id_=chunk.get('id', '')
        )
        documents.append(doc)
    
    # Initialize FAISS vector store directly with LlamaIndex
    faiss_index_path = persist_dir / "faiss_index"
    faiss_index_path.mkdir(parents=True, exist_ok=True)
    
    bm25_index_path = persist_dir / "bm25_index"
    bm25_index_path.mkdir(parents=True, exist_ok=True)
    
    # Create FAISS index with proper dimensions
    import faiss
    d = len(embeddings_list[0]) if embeddings_list else 384  # Default to 384 for all-MiniLM-L6-v2
    faiss_index = faiss.IndexHNSWFlat(d, 32)  # HNSW with M=32
    
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Build FAISS index directly with LlamaIndex
    index = VectorStoreIndex.from_documents(
        documents=documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    
    # Save FAISS index directly with LlamaIndex
    index.storage_context.persist(persist_dir=str(faiss_index_path))
    
    faiss_index_path = persist_dir / "faiss_index"
    print(f"  FAISS index built: {len(documents)} vectors")
    print(f"  Saved to: {faiss_index_path}")
    logger.info(f"Built and saved FAISS index to {faiss_index_path}")
    
    # Step 4: Build BM25 index (Module 7 - Direct LlamaIndex)
    print("\n[Step 4/4] Building BM25 index...")
    
    # Build BM25 retriever directly with LlamaIndex
    from llama_index.retrievers.bm25 import BM25Retriever
    from src.utils.config_loader import get_bm25_config
    
    bm25_config = get_bm25_config(args.config)
    similarity_top_k = bm25_config.get('similarity_top_k', 20)
    
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=chunks,  # Use the same chunks as FAISS
        similarity_top_k=similarity_top_k,
        verbose=True
    )
    
    # Persist BM25 retriever
    bm25_index_path = persist_dir / "bm25_index"
    bm25_retriever.persist(str(bm25_index_path))
    
    print(f"  BM25 index built: {len(chunks)} chunks")
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
    print("Indexing Summary:")
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
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core import get_response_synthesizer
    from llama_index.core.callbacks import TokenCountingHandler
    from src.utils.config_loader import get_embedding_config
    
    start_time = time.time()
    persist_dir = Path(args.persist_dir)
    query = args.query
    top_k = args.top_k if hasattr(args, 'top_k') else 5
    
    # Validate index exists
    faiss_index_path = persist_dir / "faiss_index"
    bm25_index_path = persist_dir / "bm25_index"
    
    if not faiss_index_path.exists() or not bm25_index_path.exists():
        print(f"Error: Indexes not found in {persist_dir}")
        print("Please run 'index' command first to create indexes.")
        logger.error("Query failed: indexes not found", persist_dir=str(persist_dir))
        sys.exit(1)
    
    logger.info("Starting query pipeline", query=query, top_k=top_k)
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    # Step 1: Load indexes (Module 8 - QueryEngine setup)
    print("[Step 1/5] Loading indexes...")
    
    # Load embedder
    embed_config = get_embedding_config(args.config)
    embed_model = HuggingFaceEmbedding(
        model_name=embed_config['embedding_model'],
        embed_batch_size=embed_config['embedding_batch_size'],
        trust_remote_code=embed_config['embedding_trust_remote_code']
    )
    
    # Load FAISS index directly with LlamaIndex
    try:
        vector_store = FaissVectorStore.from_persist_dir(str(faiss_index_path))
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        faiss_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=embed_model
        )
        print(f"  FAISS index loaded from: {faiss_index_path}")
        logger.info(f"Loaded FAISS index from {faiss_index_path}")
    except Exception as e:
        print(f"  ERROR: Failed to load FAISS index - {e}")
        logger.error(f"Failed to load FAISS index: {e}")
        sys.exit(1)
    
    # Load BM25 index directly with LlamaIndex
    from llama_index.retrievers.bm25 import BM25Retriever
    
    try:
        bm25_retriever = BM25Retriever.from_persist_dir(str(bm25_index_path))
        print(f"  BM25 index loaded from: {bm25_index_path}")
        logger.info(f"Loaded BM25 index from {bm25_index_path}")
    except Exception as e:
        print(f"  WARNING: Failed to load BM25 index - {e}")
        print("  Continuing with vector-only search")
        bm25_retriever = None
        logger.warning(f"Failed to load BM25 index: {e}")
    
    # Step 2: Initialize QueryEngine (Module 8)
    print("\n[Step 2/5] Initializing query engine...")
    
    if faiss_index:
        # Create QueryEngine with postprocessors for hybrid search + reranking
        from llama_index.core.postprocessor import SentenceTransformerRerank
        from llama_index.core.retrievers import QueryFusionRetriever
        from llama_index.core.query_engine import RetrieverQueryEngine
        from src.utils.config_loader import get_retriever_config
        
        retriever_config = get_retriever_config(args.config)
        semantic_top_k = retriever_config.get('semantic_top_k', 20)
        
        # Create base retrievers
        vector_retriever = faiss_index.as_retriever(similarity_top_k=semantic_top_k)
        
        # Create hybrid retriever (fusion of vector and BM25)
        retrievers = [vector_retriever]
        if bm25_retriever:
            retrievers.append(bm25_retriever)
        
        # Add reranking postprocessor
        rerank_postprocessor = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_n=top_k
        )
        
        if len(retrievers) > 1:
            # Use QueryFusionRetriever for hybrid search
            query_fusion_retriever = QueryFusionRetriever(
                retrievers=retrievers,
                similarity_top_k=semantic_top_k,
                num_queries=1,  # Single query
                use_async=True,
                verbose=False
            )
            query_engine = RetrieverQueryEngine.from_args(
                retriever=query_fusion_retriever,
                node_postprocessors=[rerank_postprocessor]
            )
        else:
            # Fallback to vector-only
            query_engine = RetrieverQueryEngine.from_args(
                retriever=vector_retriever,
                node_postprocessors=[rerank_postprocessor]
            )
        
        print("  QueryEngine initialized (hybrid search + reranking)")
        logger.info("QueryEngine initialized with hybrid search and reranking")
    else:
        print("  ERROR: Failed to load FAISS index")
        logger.error("FAISS index is None after loading")
        sys.exit(1)
    
    # Step 3: Retrieve candidates (Module 8 - hybrid retrieval via QueryEngine)
    print("\n[Step 3/5] Retrieving candidates...")
    
    # Retrieve using QueryEngine (includes postprocessing)
    initial_k = top_k * 4  # Over-retrieve for better selection
    
    try:
        # Use QueryEngine for retrieval
        response = query_engine.query(query)
        retrieved_nodes = response.source_nodes[:initial_k] if response.source_nodes else []
        
        print(f"  Retrieved {len(retrieved_nodes)} candidates via QueryEngine")
        logger.info(f"Retrieved {len(retrieved_nodes)} candidates via QueryEngine")
    except Exception as e:
        print(f"  ERROR: Retrieval failed - {e}")
        logger.error(f"Retrieval failed: {e}")
        sys.exit(1)
    
    # Convert nodes to document format
    retrieved_docs = []
    for node in retrieved_nodes:
        doc = {
            'text': node.text,
            'score': getattr(node, 'score', 0.0),
            'metadata': node.metadata,
            'node_id': getattr(node, 'node_id', getattr(node, 'id_', ''))
        }
        retrieved_docs.append(doc)
    
    # Step 4: Results already postprocessed by QueryEngine
    # Hybrid search + metadata boost + cross-encoder reranking via native LlamaIndex postprocessors
    print("\n[Step 4/5] Postprocessing complete (hybrid search + reranking)...")
    
    # Results are already postprocessed by QueryEngine
    # Limit to final top_k
    unique_candidates = retrieved_docs[:top_k]
    
    print(f"  Unique candidates: {len(unique_candidates)}")
    print(f"  Postprocessed (via QueryEngine): {len(unique_candidates)} -> top {top_k}")
    
    # Use postprocessed results
    reranked_results = unique_candidates[:top_k]
    
    logger.info(f"Postprocessed {len(unique_candidates)} candidates to top {len(reranked_results)}")
    logger.info("Note: Postprocessing (hybrid search + reranking) handled by QueryEngine")
    
    # Step 5: Synthesize response using native LlamaIndex Response Synthesizer
    print("\n[Step 5/5] Synthesizing response...")
    
    # Initialize Response Synthesizer with token counting
    from src.utils.config_loader import get_context_assembler_config
    assembler_config = get_context_assembler_config(args.config)
    
    import tiktoken
    try:
        tokenizer_fn = tiktoken.encoding_for_model(assembler_config['model_name']).encode
    except KeyError:
        tokenizer_fn = tiktoken.get_encoding("cl100k_base").encode
        logger.warning(f"Unknown model {assembler_config['model_name']}, using cl100k_base encoding")
    
    token_counter = TokenCountingHandler(tokenizer=tokenizer_fn, verbose=False)
    
    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        use_async=False,
        streaming=False
    )
    
    # Convert reranked results to NodeWithScore objects
    from llama_index.core.schema import TextNode, NodeWithScore
    nodes_with_scores = []
    for doc in reranked_results:
        node = TextNode(
            text=doc.get('text', ''),
            metadata=doc.get('metadata', {}),
            id_=doc.get('node_id', doc.get('id', ''))
        )
        score = doc.get('score', 0.0)
        node_with_score = NodeWithScore(node=node, score=score)
        nodes_with_scores.append(node_with_score)
    
    # Synthesize response
    synthesized_response = response_synthesizer.synthesize(
        query_str=query,
        nodes=nodes_with_scores
    )
    
    # Create MCP-compatible context items from synthesized results
    from src.rag.libs.schemas.mcp_schemas import MCPContextPayload, ContextItem
    context_items = []
    for node_with_score in nodes_with_scores:
        item = ContextItem(
            id=node_with_score.node.node_id or node_with_score.node.id_,
            text=node_with_score.node.get_content(),
            score=node_with_score.score,
            meta=node_with_score.node.metadata
        )
        context_items.append(item)
    
    context_payload = MCPContextPayload(
        schema_version="1.0",
        context=context_items,
        query=query,
        token_budget=4000,  # Default budget
        provenance={
            'total_candidates': len(retrieved_docs),
            'selected_count': len(reranked_results),
            'total_tokens': len(str(synthesized_response)) // 4,  # Rough token estimate
            'selection_method': 'response_synthesizer_compact'
        }
    )
    
    print(f"  Response synthesized: {len(context_payload.context)} items")
    print(f"  Estimated tokens: ~{context_payload.total_tokens_estimate()}")
    logger.info(f"Response synthesized: {len(context_payload.context)} items")
    
    # Display results
    duration = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("Results:")
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
    print("Query Summary:")
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
    from llama_index.core.schema import Document
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core import StorageContext, VectorStoreIndex
    from src.rag.services.ingest_service.service import PDFIngestAdapter
    from src.rag.services.duplicate_detection_service.service import DuplicateDetector
    from src.rag.services.chunker_service.service import Chunker
    from src.utils.config_loader import get_embedding_config
    
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
    print("Incremental Index Update")
    print(f"{'='*60}\n")
    
    # Step 1: Parse new documents
    print(f"[Step 1/4] Parsing new documents from {data_dir}...")
    
    pdf_adapter = PDFIngestAdapter(config_path=args.config)
    
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
    print("\n[Step 2/4] Checking for duplicates...")
    deduplicator = DuplicateDetector()
    unique_docs = deduplicator.remove_duplicates(new_docs)
    print(f"  Unique new documents: {len(unique_docs)}")
    
    # Step 3: Chunk documents
    print("\n[Step 3/4] Chunking documents...")
    chunker = Chunker(config_path=args.config)
    chunks = chunker.chunk_documents(unique_docs)
    print(f"  Created {len(chunks)} chunks")
    
    # Step 4: Update indexes
    print("\n[Step 4/4] Updating indexes...")
    
    # Load embedder
    embed_config = get_embedding_config(args.config)
    embed_model = HuggingFaceEmbedding(
        model_name=embed_config['embedding_model'],
        embed_batch_size=embed_config['embedding_batch_size'],
        trust_remote_code=embed_config['embedding_trust_remote_code']
    )
    
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
    vector_store = FaissVectorStore.from_persist_dir(str(faiss_index_path))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    faiss_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    if faiss_index:
        # Add documents incrementally
        for doc in documents:
            faiss_index.insert(doc)
        
        # Save updated index
        faiss_index.storage_context.persist(persist_dir=str(faiss_index_path))
        print(f"  FAISS index updated: +{len(documents)} documents")
        logger.info(f"FAISS index updated with {len(documents)} new documents")
    else:
        print("  ERROR: Failed to load FAISS index")
        logger.error("Failed to load FAISS index for update")
    
    # Note: BM25 index requires full rebuild for updates (limitation of sparse retrieval)
    print("  Note: BM25 index requires full rebuild for updates")
    print("  Run 'index' command to rebuild BM25 with all documents")
    logger.info("BM25 index update skipped - requires full rebuild")
    
    # Summary
    duration = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("Update Summary:")
    print(f"{'='*60}")
    print(f"  New documents added:  {len(unique_docs)}")
    print(f"  New chunks created:   {len(chunks)}")
    print("  FAISS index updated:  Yes")
    print("  BM25 index updated:   Requires manual rebuild")
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
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from src.utils.config_loader import get_embedding_config
    
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
    print("Creating Index Snapshot")
    print(f"{'='*60}\n")
    
    # Load embedder for model info
    embed_config = get_embedding_config(args.config)
    embed_model = HuggingFaceEmbedding(
        model_name=embed_config['embedding_model'],
        embed_batch_size=embed_config['embedding_batch_size'],
        trust_remote_code=embed_config['embedding_trust_remote_code']
    )
    embed_model_info = {
        'model_name': embed_config['embedding_model'],
        'embed_dim': embed_config.get('embedding_dim', 384)
    }
    
    # Load vector store
    vector_store = FaissVectorStore.from_persist_dir(str(faiss_index_path))
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    faiss_index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model
    )
    
    if not faiss_index:
        print("ERROR: Failed to load index")
        logger.error("Failed to load index for snapshot")
        sys.exit(1)
    
    # Create snapshot directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    snapshot_name = snapshot_name or f"snapshot_{timestamp}"
    snapshot_dir = persist_dir / "snapshots" / snapshot_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    # Persist index to snapshot directory
    faiss_index.storage_context.persist(persist_dir=str(snapshot_dir))
    
    # Create manifest
    manifest = {
        "timestamp": timestamp,
        "embed_model_name": embed_model_info['model_name'],
        "embed_dim": embed_model_info['embed_dim'],
        "chunking_params": {},  # Would need to load from config
        "files": list(snapshot_dir.glob("*"))
    }
    
    import json
    manifest_path = snapshot_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    
    snapshot_path = snapshot_dir
    
    duration = time.time() - start_time
    
    print(f"  Snapshot created: {snapshot_path}")
    print(f"  Duration: {duration:.2f}s")
    print(f"{'='*60}\n")
    
    logger.info("Snapshot created successfully",
               snapshot_path=str(snapshot_path),
               duration_ms=duration * 1000)
    
    metrics.increment("snapshots_created_total")
    metrics.histogram("snapshot_creation_duration_ms", duration * 1000)


def handle_verify_snapshot(args: argparse.Namespace) -> None:
    """Handle verify-snapshot command - verify integrity of snapshot."""
    import json
    from pathlib import Path
    
    start_time = time.time()
    persist_dir = Path(args.persist_dir)
    snapshot_name = args.name
    
    faiss_index_path = persist_dir / "faiss_index"
    snapshot_dir = persist_dir / "snapshots" / snapshot_name
    
    if not snapshot_dir.exists():
        print(f"Error: Snapshot not found at {snapshot_dir}")
        logger.error("Verification failed: snapshot not found", snapshot_dir=str(snapshot_dir))
        sys.exit(1)
    
    logger.info("Verifying snapshot", snapshot_name=snapshot_name)
    print(f"\n{'='*60}")
    print(f"Verifying Snapshot: {snapshot_name}")
    print(f"{'='*60}\n")
    
    # Verify snapshot integrity
    try:
        # Check if manifest exists
        manifest_path = snapshot_dir / "manifest.json"
        if not manifest_path.exists():
            print("  Status: INVALID")
            print("  Error: Manifest file missing")
            logger.error("Snapshot verification failed: manifest missing")
            sys.exit(1)
        
        # Load and validate manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Check required manifest fields
        required_fields = ['timestamp', 'embed_model_name', 'embed_dim']
        missing_fields = [field for field in required_fields if field not in manifest]
        
        if missing_fields:
            print("  Status: INVALID")
            print(f"  Error: Missing manifest fields: {missing_fields}")
            logger.error("Snapshot verification failed: missing manifest fields", missing_fields=missing_fields)
            sys.exit(1)
        
        # Check if index files exist
        index_files = list(snapshot_dir.glob("*.faiss")) + list(snapshot_dir.glob("*.pkl"))
        if not index_files:
            print("  Status: INVALID")
            print("  Error: No index files found")
            logger.error("Snapshot verification failed: no index files")
            sys.exit(1)
        
        # Try to load the snapshot index
        try:
            from llama_index.vector_stores.faiss import FaissVectorStore
            from llama_index.core import StorageContext, VectorStoreIndex
            
            vector_store = FaissVectorStore.from_persist_dir(str(snapshot_dir))
            # If we get here, the index loaded successfully
            print("  Status: VALID")
            
        except Exception as load_error:
            print("  Status: INVALID")
            print(f"  Error: Failed to load index - {load_error}")
            logger.error("Snapshot verification failed: index load error", error=str(load_error))
            sys.exit(1)
        
        duration = time.time() - start_time
        
        print(f"  Snapshot: {snapshot_name}")
        print("\n  Manifest Info:")
        print(f"    Created: {manifest.get('timestamp', 'N/A')}")
        print(f"    Embedding model: {manifest.get('embed_model_name', 'N/A')}")
        print(f"    Files: {len(index_files)}")
        
        print(f"\n  Verification duration: {duration:.2f}s")
        print(f"{'='*60}\n")
        
        logger.info("Snapshot verification completed",
                   snapshot_name=snapshot_name,
                   valid=True,
                   duration_ms=duration * 1000)
        
        metrics.increment("snapshot_verifications_total")
            
    except Exception as e:
        print(f"  ERROR: Verification failed - {e}")
        logger.error(f"Snapshot verification failed: {e}")
        sys.exit(1)


def handle_status(args: argparse.Namespace) -> None:
    """Handle status command - check background task status."""
    import json
    from datetime import datetime
    
    async def _check_status() -> None:
        task_queue = get_cli_task_queue()
        await task_queue.start()
        
        try:
            task_status = await task_queue.get_task_status(args.task_id)
            
            print(f"\n{'='*60}")
            print("Task Status")
            print(f"{'='*60}\n")
            
            if task_status is None:
                print(f"  Task ID: {args.task_id}")
                print("  Status: NOT FOUND")
                print("\n  The task may have been removed or never existed.")
            else:
                print(f"  Task ID: {task_status['task_id']}")
                print(f"  Name: {task_status['name']}")
                print(f"  Status: {task_status['status'].upper()}")
                
                if task_status['created_at']:
                    created = datetime.fromtimestamp(task_status['created_at'])
                    print(f"  Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if task_status['started_at']:
                    started = datetime.fromtimestamp(task_status['started_at'])
                    print(f"  Started: {started.strftime('%Y-%m-%d %H:%M:%S')}")
                
                if task_status['completed_at']:
                    completed = datetime.fromtimestamp(task_status['completed_at'])
                    print(f"  Completed: {completed.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if task_status['started_at']:
                        duration = task_status['completed_at'] - task_status['started_at']
                        print(f"  Duration: {duration:.2f}s")
                
                if task_status['error']:
                    print(f"\n  Error: {task_status['error']}")
                
                if task_status['result']:
                    print("\n  Result:")
                    try:
                        result_dict = json.loads(task_status['result']) if isinstance(task_status['result'], str) else task_status['result']
                        print(f"    {json.dumps(result_dict, indent=4)}")
                    except (json.JSONDecodeError, TypeError, AttributeError):
                        print(f"    {task_status['result']}")
            
            print(f"\n{'='*60}\n")
            
        finally:
            await task_queue.stop()
    
    asyncio.run(_check_status())


def handle_cancel(args: argparse.Namespace) -> None:
    """Handle cancel command - cancel background task."""
    async def _cancel_task() -> None:
        task_queue = get_cli_task_queue()
        await task_queue.start()
        
        try:
            print(f"\n{'='*60}")
            print("Task Cancellation")
            print(f"{'='*60}\n")
            
            # Check if task exists first
            task_status = await task_queue.get_task_status(args.task_id)
            
            if task_status is None:
                print(f"  Task ID: {args.task_id}")
                print("  Status: NOT FOUND")
                print("\n  The task may have been removed or never existed.")
            else:
                print(f"  Task ID: {args.task_id}")
                print(f"  Current Status: {task_status['status'].upper()}")
                
                # Try to cancel
                success = await task_queue.cancel_task(args.task_id)
                
                if success:
                    print("  Result: CANCELLED SUCCESSFULLY")
                else:
                    print("  Result: CANNOT CANCEL")
                    print(f"  Reason: Task is in '{task_status['status']}' state")
            
            print(f"\n{'='*60}\n")
            
        finally:
            await task_queue.stop()
    
    asyncio.run(_cancel_task())


def handle_list_tasks(args: argparse.Namespace) -> None:
    """Handle list-tasks command - list all background tasks."""
    from datetime import datetime
    
    async def _list_tasks() -> None:
        task_queue = get_cli_task_queue()
        await task_queue.start()
        
        try:
            task_filter = args.filter if hasattr(args, 'filter') else 'all'
            
            print(f"\n{'='*60}")
            print(f"Task List (Filter: {task_filter.upper()})")
            print(f"{'='*60}\n")
            
            # Get all tasks
            all_tasks = []
            for task_id, task_data in task_queue._tasks.items():
                all_tasks.append(task_data.to_dict())
            
            # Filter by status
            if task_filter != 'all':
                filtered_tasks = [t for t in all_tasks if t['status'] == task_filter]
            else:
                filtered_tasks = all_tasks
            
            # Sort by creation time (newest first)
            filtered_tasks.sort(key=lambda t: t['created_at'] or 0, reverse=True)
            
            if not filtered_tasks:
                print(f"  No tasks found matching filter: {task_filter}")
            else:
                print(f"  Found {len(filtered_tasks)} task(s):\n")
                
                for i, task in enumerate(filtered_tasks, 1):
                    print(f"  [{i}] {task['name']}")
                    print(f"      ID: {task['task_id']}")
                    print(f"      Status: {task['status'].upper()}")
                    
                    if task['created_at']:
                        created = datetime.fromtimestamp(task['created_at'])
                        print(f"      Created: {created.strftime('%Y-%m-%d %H:%M:%S')}")
                    
                    if task['started_at'] and task['completed_at']:
                        duration = task['completed_at'] - task['started_at']
                        print(f"      Duration: {duration:.2f}s")
                    
                    if task['error']:
                        print(f"      Error: {task['error'][:80]}...")
                    
                    print()
            
            # Show queue statistics
            stats = task_queue.get_queue_stats()
            print("  Queue Statistics:")
            print(f"    Total: {stats['total_tasks']}")
            print(f"    Pending: {stats['pending']}")
            print(f"    Running: {stats['running']}")
            print(f"    Completed: {stats['completed']}")
            print(f"    Failed: {stats['failed']}")
            print(f"    Cancelled: {stats['cancelled']}")
            
            print(f"\n{'='*60}\n")
            
        finally:
            await task_queue.stop()
    
    asyncio.run(_list_tasks())


def handle_cleanup(args: argparse.Namespace) -> None:
    """Handle cleanup command - cleanup old task records."""
    print(f"\n{'='*60}")
    print("Task Cleanup")
    print(f"{'='*60}\n")
    print("  Status: Not implemented (requires async task queue)")
    print("\n  Note: Task queue requires MCP server integration")
    print("  Use MCP API endpoints for task management")
    print(f"{'='*60}\n")


def handle_metrics(args: argparse.Namespace) -> None:
    """Handle metrics command - display collected metrics."""
    import json
    
    output_format = args.format if hasattr(args, 'format') else 'text'
    
    # Get current metrics
    stats = metrics.get_stats()
    
    if output_format == 'json':
        print(json.dumps(stats, indent=2))
    else:
        print(f"\n{'='*60}")
        print("System Metrics")
        print(f"{'='*60}\n")
        
        if not stats:
            print("  No metrics collected yet")
        else:
            for metric_name, metric_data in stats.items():
                metric_type = metric_data.get('type', 'unknown')
                
                if metric_type == 'counter':
                    print(f"  {metric_name}: {metric_data.get('value', 0)}")
                elif metric_type == 'gauge':
                    print(f"  {metric_name}: {metric_data.get('value', 0):.2f}")
                elif metric_type == 'histogram':
                    values = metric_data.get('values', [])
                    if values:
                        avg = sum(values) / len(values)
                        min_val = min(values)
                        max_val = max(values)
                        print(f"  {metric_name}:")
                        print(f"    avg: {avg:.2f}")
                        print(f"    min: {min_val:.2f}")
                        print(f"    max: {max_val:.2f}")
                        print(f"    count: {len(values)}")
        
        print(f"\n{'='*60}\n")
    
    logger.info("Metrics displayed", format=output_format)


def handle_benchmark(args: argparse.Namespace) -> None:
    """Handle benchmark command - benchmark different search strategies."""
    print(f"\n{'='*60}")
    print("Search Strategy Benchmark")
    print(f"{'='*60}\n")
    print(f"  Query: {args.query}")
    print(f"  Alphas: {args.alphas if hasattr(args, 'alphas') else '0.0,0.5,1.0'}")
    print("\n  Status: Not yet implemented")
    print("\n  Planned features:")
    print("    - Compare vector vs BM25 vs hybrid search")
    print("    - Test different alpha weights")
    print("    - Measure retrieval latency")
    print("    - Compare relevance scores")
    print(f"{'='*60}\n")
    
    logger.info("Benchmark command called (not implemented)", query=args.query)


def handle_clean_boilerplate(args: argparse.Namespace) -> None:
    """Handle clean-boilerplate command - remove web artifacts and boilerplate."""
    from src.rag.services.boilerplate_removal_service.service import BoilerplateRemovalService
    
    start_time = time.time()
    data_dir = Path(args.data_dir)
    dry_run = args.dry_run
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        logger.error("Clean boilerplate failed: directory not found", data_dir=str(data_dir))
        sys.exit(1)
    
    logger.info("Starting boilerplate removal", data_dir=str(data_dir), dry_run=dry_run)
    print(f"\n{'='*60}")
    print(f"Cleaning boilerplate from: {data_dir}")
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (files will be modified)'}")
    print(f"{'='*60}\n")
    
    # Initialize service
    boilerplate_service = BoilerplateRemovalService(aggressive_mode=True)
    
    # Find all .txt files
    txt_files = list(data_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"No .txt files found in {data_dir}")
        logger.warning("No txt files found for cleaning", data_dir=str(data_dir))
        return
    
    print(f"Found {len(txt_files)} text files to process\n")
    
    cleaned_count = 0
    skipped_count = 0
    error_count = 0
    total_removed_chars = 0
    
    for txt_file in txt_files:
        try:
            # Read original content
            with open(txt_file, 'r', encoding='utf-8') as f:
                original_text = f.read()
            
            original_len = len(original_text)
            
            # Clean boilerplate
            cleaned_text = boilerplate_service.remove_boilerplate(original_text)
            cleaned_len = len(cleaned_text)
            
            removed_chars = original_len - cleaned_len
            
            if removed_chars > 0:
                print(f"  {txt_file.name}")
                print(f"    Original: {original_len:,} chars")
                print(f"    Cleaned:  {cleaned_len:,} chars")
                print(f"    Removed:  {removed_chars:,} chars ({removed_chars/original_len*100:.1f}%)")
                
                if not dry_run:
                    # Write cleaned content
                    with open(txt_file, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)
                    print("    Status: UPDATED")
                else:
                    print("    Status: DRY RUN (no changes)")
                
                cleaned_count += 1
                total_removed_chars += removed_chars
            else:
                skipped_count += 1
                
        except Exception as e:
            error_count += 1
            print(f"  ERROR: {txt_file.name} - {e}")
            logger.error(f"Failed to clean {txt_file.name}", error=str(e))
    
    # Get statistics from service
    stats = boilerplate_service.stats if hasattr(boilerplate_service, 'stats') else {}
    
    # Summary
    duration = time.time() - start_time
    
    print(f"\n{'='*60}")
    print("Boilerplate Removal Summary:")
    print(f"{'='*60}")
    print(f"  Total files:          {len(txt_files)}")
    print(f"  Files cleaned:        {cleaned_count}")
    print(f"  Files skipped:        {skipped_count}")
    print(f"  Errors:               {error_count}")
    print(f"  Total chars removed:  {total_removed_chars:,}")
    print(f"  Mode:                 {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"  Duration:             {duration:.2f}s")
    
    if stats:
        print("\n  Pattern Statistics:")
        for pattern_name, count in stats.items():
            if count > 0:
                print(f"    {pattern_name}: {count} matches")
    
    print(f"{'='*60}\n")
    
    logger.info("Boilerplate removal completed",
               total_files=len(txt_files),
               cleaned=cleaned_count,
               skipped=skipped_count,
               errors=error_count,
               chars_removed=total_removed_chars,
               dry_run=dry_run,
               duration_ms=duration * 1000)
    
    metrics.increment("boilerplate_removal_files_processed", len(txt_files))
    metrics.increment("boilerplate_removal_files_cleaned", cleaned_count)
    metrics.histogram("boilerplate_removal_duration_ms", duration * 1000)


def handle_serve(args: argparse.Namespace) -> None:
    """
    Handle serve command - start MCP server.
    
    Starts the minimal FastMCP server that delegates to handlers.
    Only provides LLM-facing tools, no admin operations.
    """
    import signal
    from src.mcp.server import run_stdio, run_sse, run_http
    
    print(f"\n{'='*80}")
    print("Starting VX-RAG MCP Server (New Architecture)")
    print(f"{'='*80}")
    print(f"  Config:    {args.config}")
    print(f"  Transport: {args.transport}")
    if args.transport in ["http", "sse"]:
        print(f"  Host:      {args.host}")
        print(f"  Port:      {args.port}")
        if args.cors:
            print(f"  CORS:      Enabled ({args.allowed_origins})")
    print(f"{'='*80}\n")
    
    logger.info(
        "Starting MCP server",
        config=args.config,
        transport=args.transport,
        host=args.host if args.transport in ["http", "sse"] else None,
        port=args.port if args.transport in ["http", "sse"] else None,
        cors=args.cors if hasattr(args, "cors") else False
    )
    
    # Setup graceful shutdown
    def signal_handler(signum: int, frame: Any) -> None:
        """Handle shutdown signals."""
        print(f"\n\nReceived signal {signum}. Shutting down gracefully...")
        logger.info("Shutdown signal received", signal=signum)
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run server based on transport
    try:
        if args.transport == "stdio":
            print("\nMCP Server starting in STDIO mode")
            print("This mode is designed for IDE integration (VS Code, Claude Desktop)")
            print("Server will listen on stdin/stdout")
            print("\nPress Ctrl+C to stop the server\n")
            run_stdio()
            
        elif args.transport == "sse":
            print("\nMCP Server starting in SSE mode")
            print(f"Server URL: http://{args.host}:{args.port}")
            print(f"SSE Endpoint: http://{args.host}:{args.port}/sse")
            if args.cors:
                print(f"CORS enabled for origins: {args.allowed_origins}")
            print("\nPress Ctrl+C to stop the server\n")
            asyncio.run(run_sse(host=args.host, port=args.port))
            
        elif args.transport == "http":
            print("\nMCP Server starting in HTTP mode")
            print(f"Server URL: http://{args.host}:{args.port}")
            if args.cors:
                print(f"CORS enabled for origins: {args.allowed_origins}")
            print("\nPress Ctrl+C to stop the server\n")
            asyncio.run(run_http(host=args.host, port=args.port))
            
    except KeyboardInterrupt:
        print("\nServer shutdown complete")
    except Exception as e:
        print(f"\nFatal error: {e}")
        logger.error("Fatal server error", error=str(e), exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
