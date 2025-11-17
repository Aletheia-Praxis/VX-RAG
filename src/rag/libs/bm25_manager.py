"""
BM25 Index Manager - Lightweight persistence wrapper for BM25Retriever.

This module provides a minimal wrapper around LlamaIndex's native BM25Retriever
with only essential persistence functionality. All core BM25 logic is handled
by LlamaIndex directly.
"""

from typing import Optional, List, Any
from pathlib import Path

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import Document, BaseNode
from llama_index.core.node_parser import SimpleNodeParser

from src.utils.logging_config import get_logger
from src.utils.config_loader import get_bm25_config

logger = get_logger("bm25_manager")


class BM25IndexManager:
    """
    Lightweight manager for BM25 index persistence.
    
    This class provides only persistence functionality around LlamaIndex's
    native BM25Retriever. Use BM25Retriever.from_defaults() directly for
    retrieval operations.
    
    Key differences from BM25Service:
    - No wrapper methods around BM25Retriever (use it directly)
    - Only handles save/load/exists operations
    - Uses BM25Retriever's native .persist() and .from_persist_dir()
    - Reduces code from 183 LOC to ~60 LOC
    """

    def __init__(self, index_dir: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize BM25 index manager.
        
        Args:
            index_dir: Directory for BM25 index persistence
            config_path: Path to settings.yaml
        """
        self.config_path = config_path
        
        if index_dir is None:
            config = get_bm25_config(config_path)
            index_dir = config['index_dir']
        
        if index_dir is None:
            raise ValueError("index_dir must be set")
        
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized BM25IndexManager: index_dir={self.index_dir}")

    def build_and_persist(self, documents: List[Document]) -> BM25Retriever:
        """
        Build BM25 retriever from documents and persist to disk.
        
        Args:
            documents: List of documents to index
            
        Returns:
            BM25Retriever instance
            
        Raises:
            ValueError: If no documents provided
        """
        if not documents:
            raise ValueError("No documents provided for indexing")

        try:
            logger.info(f"Building BM25 index with {len(documents)} documents")

            # Parse documents to nodes
            parser = SimpleNodeParser()
            nodes = parser.get_nodes_from_documents(documents)

            # Create BM25 retriever using LlamaIndex native method
            config = get_bm25_config(self.config_path)
            similarity_top_k = config.get('similarity_top_k', 20)
            
            retriever = BM25Retriever.from_defaults(
                nodes=nodes,
                similarity_top_k=similarity_top_k,
                verbose=True
            )

            # Persist using native BM25Retriever method
            retriever.persist(str(self.index_dir))
            
            logger.info(f"BM25 index built and persisted with {len(nodes)} nodes")
            return retriever

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            raise

    def build_from_docstore(self, docstore: Any, similarity_top_k: Optional[int] = None) -> BM25Retriever:
        """
        Build BM25 retriever from existing docstore and persist to disk.
        
        This method is useful when you already have a VectorStoreIndex with a docstore
        and want to create a BM25 retriever that works with the same documents.
        
        Args:
            docstore: LlamaIndex DocumentStore instance
            similarity_top_k: Number of top results to return (uses config if None)
            
        Returns:
            BM25Retriever instance
        """
        try:
            if similarity_top_k is None:
                config = get_bm25_config(self.config_path)
                similarity_top_k = config.get('similarity_top_k', 20)
            
            # Ensure it's an int and not None
            if similarity_top_k is None or not isinstance(similarity_top_k, int):
                similarity_top_k = 20 if similarity_top_k is None else int(similarity_top_k)
            
            logger.info(f"Building BM25 retriever from docstore with top_k={similarity_top_k}")
            
            # Create BM25 retriever from docstore using LlamaIndex native method
            retriever = BM25Retriever.from_defaults(
                docstore=docstore,
                similarity_top_k=similarity_top_k,  # Now guaranteed to be int
                verbose=True
            )

            # Persist using native BM25Retriever method
            retriever.persist(str(self.index_dir))
            
            logger.info(f"BM25 retriever built from docstore and persisted")
            return retriever

        except Exception as e:
            logger.error(f"Failed to build BM25 retriever from docstore: {e}")
            raise

    def load(self) -> Optional[BM25Retriever]:
        """
        Load BM25 retriever from disk.
        
        Returns:
            BM25Retriever instance if exists, None otherwise
        """
        if not self.exists():
            logger.warning(f"BM25 index does not exist in {self.index_dir}")
            return None

        try:
            # Load using native BM25Retriever method
            retriever = BM25Retriever.from_persist_dir(str(self.index_dir))
            
            logger.info(f"BM25 index loaded from {self.index_dir}")
            return retriever

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return None

    def exists(self) -> bool:
        """
        Check if BM25 index exists on disk.
        
        Returns:
            True if index directory exists and contains index files
        """
        # BM25Retriever creates these files on persist
        index_file = self.index_dir / "bm25_retriever.json"
        return index_file.exists()

    def get_or_load(self) -> Optional[BM25Retriever]:
        """
        Convenience method to load existing index or return None.
        
        Returns:
            BM25Retriever if exists, None otherwise
        """
        return self.load() if self.exists() else None
