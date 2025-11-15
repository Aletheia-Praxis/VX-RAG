"""
Services for the RAG pipeline.

Note: 
- Embedding functionality is now provided via llama_index.core.Settings.embed_model
- Hybrid search is now provided via llama_index.core.retrievers.QueryFusionRetriever
- BM25 persistence is now provided via src.rag.libs.bm25_manager.BM25IndexManager
- Reranking functionality is now provided via llama_index.postprocessor.SentenceTransformerRerank
  (integrated in RetrieverService)
"""
from . import (
    assembler_service,
    chunker_service,
    duplicate_detection_service,
    ingest_service,
    retriever_service,
    vectordb_service,
)

__all__ = [
    "assembler_service",
    "chunker_service",
    "duplicate_detection_service",
    "ingest_service",
    "retriever_service",
    "vectordb_service",
]
