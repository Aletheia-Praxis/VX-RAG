"""
Services for the RAG pipeline.

Note: 
- Embedding functionality is now provided via llama_index.core.Settings.embed_model
- Hybrid search is now provided via llama_index.core.retrievers.QueryFusionRetriever
- BM25 persistence is now provided via src.rag.libs.bm25_manager.BM25IndexManager
"""
from . import (
    assembler_service,
    chunker_service,
    duplicate_detection_service,
    ingest_service,
    reranker_service,
    retriever_service,
    vectordb_service,
)

__all__ = [
    "assembler_service",
    "chunker_service",
    "duplicate_detection_service",
    "ingest_service",
    "reranker_service",
    "retriever_service",
    "vectordb_service",
]
