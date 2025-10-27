"""
Vector Database Service implementation.

Provides classes for vector storage operations.
"""

from typing import List, Dict, Any, Optional, cast
import logging
from pathlib import Path

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

logger = logging.getLogger(__name__)

class VectorStoreClient:
    """Client for vector database operations."""
    
    def __init__(self, store_type: str = "faiss", config: Optional[Dict[str, Any]] = None):
        self.store_type = store_type
        self.config = config or {}
        self.index_dir = Path(self.config.get('index_dir', 'data/index'))
        self.index: Optional[VectorStoreIndex] = None
        
        # Create index directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        if store_type == "faiss":
            # Initialize FAISS vector store
            # We'll set the dimension when building the index
            pass  # Will be initialized when building index
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
    
    def build_index(self, documents: List[Any], embed_model: Any) -> VectorStoreIndex:
        """
        Build FAISS vector index from documents.
        
        Args:
            documents: List of documents to index
            embed_model: Embedding model to use
            
        Returns:
            The created VectorStoreIndex
        """
        if not documents:
            raise ValueError("No documents provided for indexing")

        try:
            logger.info("Initializing FAISS vector store")
            # Get embedding dimension by creating a test embedding
            test_embedding = embed_model.get_text_embedding("test")
            d = len(test_embedding)
            logger.info(f"Embedding dimension: {d}")
            
            # Initialize FAISS vector store
            faiss_index = faiss.IndexFlatL2(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            logger.info(f"Building index with {len(documents)} documents")
            # Build index with documents
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model,
                show_progress=True
            )

            self.index = index
            logger.info(f"Successfully built index with {len(documents)} documents")
            return index

        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise
    
    def save_index(self) -> None:
        """Save the index to disk."""
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        try:
            self.index.storage_context.persist(persist_dir=str(self.index_dir))
            logger.info(f"Index saved to {self.index_dir}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self) -> bool:
        """
        Load the index from disk.
        
        Returns:
            True if loading succeeded, False otherwise
        """
        if not self.index_dir.exists():
            logger.error(f"Index directory does not exist: {self.index_dir}")
            return False

        try:
            # Load storage context
            storage_context = StorageContext.from_defaults(persist_dir=str(self.index_dir))

            # Load index
            self.index = cast(VectorStoreIndex, load_index_from_storage(storage_context))
            logger.info(f"Index loaded from {self.index_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def store_vectors(self, vectors: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Store vectors with optional metadata."""
        # TODO: Implement vector storage for existing index
        logger.warning("store_vectors not implemented for FAISS index")
        return False
    
    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors."""
        if self.index is None:
            if not self.load_index():
                return []
        
        assert self.index is not None  # nosec B101
        try:
            # Use LlamaIndex query engine for search
            # Note: This is a simplified implementation
            # For pure vector search, we'd need direct FAISS access
            logger.warning("search_vectors using query engine, not pure vector search")
            return []
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs."""
        # TODO: Implement vector deletion
        logger.warning("delete_vectors not implemented for FAISS")
        return False

# TODO: Add persistence, snapshotting, replication
