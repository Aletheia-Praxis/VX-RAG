"""
Services for the RAG pipeline.

Note: 
- Embedding functionality is now provided via llama_index.core.Settings.embed_model
- Hybrid search is now provided via llama_index.core.retrievers.QueryFusionRetriever
- BM25 retrieval is now provided via llama_index.retrievers.bm25.BM25Retriever
- Reranking functionality is now provided via llama_index.postprocessor.SentenceTransformerRerank
  (integrated in native LlamaIndex QueryEngine postprocessors)
"""
from . import (
    assembler_service,
    chunker_service,
    duplicate_detection_service,
    ingest_service,
    vectordb_service,
)

__all__ = [
    "assembler_service",
    "chunker_service",
    "duplicate_detection_service",
    "ingest_service",
    "vectordb_service",
]
