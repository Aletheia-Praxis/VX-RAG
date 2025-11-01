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
        self.faiss_index: Optional[faiss.Index] = None  # Direct FAISS index access
        
        # Create index directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        if store_type == "faiss":
            # Will be initialized when building index
            pass
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
    
    def build_index(self, documents: List[Any], embed_model: Any, transformations: Optional[List[Any]] = None) -> VectorStoreIndex:
        """
        Build FAISS vector index from documents.
        
        Args:
            documents: List of documents to index
            embed_model: Embedding model to use
            transformations: Optional list of transformations (e.g., node parsers)
            
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
                transformations=transformations,
                show_progress=True
            )

            self.index = index
            self.faiss_index = faiss_index  # Store direct access to FAISS index
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
            
            # Try to get direct access to FAISS index from vector store
            try:
                # Note: Direct FAISS access may not be available through LlamaIndex API
                # We'll use LlamaIndex methods for operations
                logger.info("Using LlamaIndex API for vector operations")
            except Exception as e:
                logger.warning(f"Could not access FAISS index directly: {e}")
            
            logger.info(f"Index loaded from {self.index_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def store_vectors(self, vectors: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Store vectors with optional metadata using LlamaIndex."""
        if self.index is None:
            logger.error("No index available. Build or load index first.")
            return False
        
        try:
            # Convert vectors and metadata to LlamaIndex documents
            from llama_index.core.schema import Document
            
            documents = []
            for i, vector in enumerate(vectors):
                # Create a document with vector as embedding
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                doc = Document(
                    text="",  # Empty text since we have pre-computed vectors
                    metadata=doc_metadata,
                    embedding=vector
                )
                documents.append(doc)
            
            # Insert documents into existing index
            for doc in documents:
                self.index.insert(doc)
            
            logger.info(f"Successfully stored {len(vectors)} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            return False
    
    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors using LlamaIndex retriever."""
        if self.index is None:
            if not self.load_index():
                return []
        
        assert self.index is not None  # nosec B101
        try:
            # Create retriever
            # Since we have a vector, we need to create a query with embedding
            # For pure vector search, we'd need to use vector_store.query directly
            # But LlamaIndex retriever expects text queries
            
            # Alternative: Use vector_store.query with VectorStoreQuery
            from llama_index.core.vector_stores import VectorStoreQuery
            
            query = VectorStoreQuery(
                query_embedding=query_vector,
                similarity_top_k=top_k
            )
            results = self.index.vector_store.query(query)
            
            # Convert results to expected format
            formatted_results = []
            if results.nodes:
                for node in results.nodes:
                    formatted_results.append({
                        'id': node.id_,
                        'score': getattr(node, 'score', 0.0),  # May not be available
                        'metadata': node.metadata,
                        'text': node.get_content()
                    })
            return formatted_results
                
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs. Note: FAISS doesn't support deletion, rebuild index instead."""
        logger.warning("delete_vectors not supported for FAISS. Consider rebuilding the index without deleted items.")
        return False
    
    def create_snapshot(self, snapshot_name: Optional[str] = None) -> bool:
        """Create a snapshot of the current index."""
        if self.index is None:
            logger.error("No index to snapshot. Build index first.")
            return False
        
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_name = snapshot_name or f"snapshot_{timestamp}"
            snapshot_dir = self.index_dir.parent / "snapshots" / snapshot_name
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            self.index.storage_context.persist(persist_dir=str(snapshot_dir))
            logger.info(f"Snapshot created: {snapshot_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return False
    
    def list_snapshots(self) -> List[str]:
        """List available snapshots."""
        try:
            snapshots_dir = self.index_dir.parent / "snapshots"
            if not snapshots_dir.exists():
                return []
            return [d.name for d in snapshots_dir.iterdir() if d.is_dir()]
        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            return []
    
    def load_snapshot(self, snapshot_name: str) -> bool:
        """Load a snapshot as the current index."""
        try:
            snapshots_dir = self.index_dir.parent / "snapshots"
            snapshot_dir = snapshots_dir / snapshot_name
            if not snapshot_dir.exists():
                logger.error(f"Snapshot {snapshot_name} does not exist")
                return False
            
            storage_context = StorageContext.from_defaults(persist_dir=str(snapshot_dir))
            self.index = cast(VectorStoreIndex, load_index_from_storage(storage_context))
            logger.info(f"Snapshot {snapshot_name} loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load snapshot {snapshot_name}: {e}")
            return False
    
    def replicate_index(self, target_dir: str) -> bool:
        """Replicate index to another directory."""
        if self.index is None:
            logger.error("No index to replicate. Build or load index first.")
            return False
        
        try:
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(target_path))
            logger.info(f"Index replicated to {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to replicate index: {e}")
            return False

# TODO: Add persistence, snapshotting, replication
