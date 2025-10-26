"""
VX-RAG System Initialization.

This module provides the main entry point for initializing and using
the VX-RAG (Retrieval-Augmented Generation) system components.
"""

import logging
from typing import Optional

from .services.ingest_service.service import PDFIngestAdapter
from .services.embedder_service.service import EmbeddingService
from .services.vectordb_service.service import VectorStoreClient
from .services.retriever_service.service import RetrieverService
from .services.llm_proxy.service import LLMProxy

logger = logging.getLogger(__name__)

class RAGSystem:
    """Main RAG system orchestrator."""
    
    def __init__(self):
        self.ingest_adapter: Optional[PDFIngestAdapter] = None
        self.embedding_service: Optional[EmbeddingService] = None
        self.vector_store: Optional[VectorStoreClient] = None
        self.retriever: Optional[RetrieverService] = None
        self.llm_proxy: Optional[LLMProxy] = None
        
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all RAG services."""
        try:
            logger.info("Initializing VX-RAG system services")
            
            self.ingest_adapter = PDFIngestAdapter()
            self.embedding_service = EmbeddingService()
            self.vector_store = VectorStoreClient()
            self.llm_proxy = LLMProxy()
            
            # Try to load existing index
            if self.vector_store.load_index():
                self.retriever = RetrieverService(self.vector_store.index)
                logger.info("RAG system initialized with existing index")
            else:
                logger.info("RAG system initialized without index (run build_index to create)")
                
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def build_index(self, documents_dir: str = "data/processed") -> bool:
        """
        Build the vector index from processed documents.
        
        Args:
            documents_dir: Directory containing processed text files
            
        Returns:
            True if successful
        """
        if self.vector_store is None or self.embedding_service is None:
            logger.error("Required services not initialized")
            return False
        
        try:
            from llama_index.core import SimpleDirectoryReader
            
            logger.info(f"Building index from {documents_dir}")
            
            # Load documents
            reader = SimpleDirectoryReader(
                input_dir=documents_dir,
                required_exts=[".txt"]
            )
            documents = reader.load_data()
            
            if not documents:
                logger.error("No documents found for indexing")
                return False
            
            # Build index
            index = self.vector_store.build_index(documents, self.embedding_service.embed_model)
            
            # Save index
            self.vector_store.save_index()
            
            # Update retriever
            self.retriever = RetrieverService(index)
            
            logger.info("Index built and saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            return False
    
    def query(self, query: str, top_k: int = 3) -> dict:
        """
        Perform a RAG query.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            Dictionary with response and sources
        """
        if self.retriever is None or self.llm_proxy is None:
            return {"error": "RAG system not fully initialized"}
        
        try:
            # Retrieve documents
            retrieved_docs = self.retriever.retrieve(query, top_k)
            
            # Generate response
            result = self.llm_proxy.generate_with_sources(query, retrieved_docs)
            
            return result
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"error": str(e)}


# Global instance
_rag_instance: Optional[RAGSystem] = None

def get_rag_system() -> RAGSystem:
    """
    Get or create the global RAG system instance.
    
    Returns:
        The RAG system instance
    """
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = RAGSystem()
    return _rag_instance