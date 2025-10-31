"""
BM25 Service implementation.

Provides classes for BM25 indexing and retrieval operations.
"""

from typing import List, Dict, Any, Optional, cast
import logging
import pickle
from pathlib import Path

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import Document, BaseNode

logger = logging.getLogger(__name__)

class BM25Service:
    """Service for BM25 indexing and retrieval."""

    def __init__(self, index_dir: Optional[str] = None):
        self.index_dir = Path(index_dir) if index_dir else Path("data/index/bm25")
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "bm25_index.pkl"
        self.nodes_path = self.index_dir / "bm25_nodes.pkl"
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.nodes: List[BaseNode] = []

    def build_index(self, documents: List[Document]) -> None:
        """
        Build BM25 index from documents.

        Args:
            documents: List of documents to index
        """
        if not documents:
            raise ValueError("No documents provided for indexing")

        try:
            logger.info(f"Building BM25 index with {len(documents)} documents")

            # Convert documents to nodes if needed
            from llama_index.core.node_parser import SimpleNodeParser
            parser = SimpleNodeParser()
            self.nodes = parser.get_nodes_from_documents(documents)

            # Create BM25 retriever
            self.bm25_retriever = BM25Retriever.from_defaults(
                nodes=self.nodes,
                similarity_top_k=20,  # Default, can be overridden in retrieve
                verbose=True
            )

            logger.info(f"Successfully built BM25 index with {len(self.nodes)} nodes")

        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            raise

    def save_index(self) -> None:
        """Save the BM25 index and nodes to disk."""
        if self.bm25_retriever is None:
            raise ValueError("No index to save. Build index first.")

        try:
            # Save the retriever and nodes
            data = {
                'retriever': self.bm25_retriever,
                'nodes': self.nodes
            }

            with open(self.index_path, 'wb') as f:
                pickle.dump(data, f)

            logger.info(f"BM25 index saved to {self.index_path}")

        except Exception as e:
            logger.error(f"Failed to save BM25 index: {e}")
            raise

    def load_index(self) -> bool:
        """
        Load the BM25 index from disk.

        Returns:
            True if loading succeeded, False otherwise
        """
        if not self.index_path.exists():
            logger.warning(f"BM25 index file does not exist: {self.index_path}")
            return False

        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)

            self.bm25_retriever = data['retriever']
            self.nodes = data['nodes']

            logger.info(f"BM25 index loaded from {self.index_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents using BM25.

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores and metadata
        """
        if self.bm25_retriever is None:
            if not self.load_index():
                logger.error("No BM25 index available")
                return []

        assert self.bm25_retriever is not None  # nosec B101

        try:
            # Set top_k for this query
            self.bm25_retriever.similarity_top_k = top_k

            nodes = self.bm25_retriever.retrieve(query)

            results = []
            for node in nodes:
                result = {
                    'text': node.text,
                    'score': getattr(node, 'score', 0.0),
                    'metadata': node.metadata,
                    'node_id': node.id_
                }
                results.append(result)

            logger.info(f"BM25 retrieved {len(results)} documents for query")
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve with BM25: {e}")
            return []

    def is_index_built(self) -> bool:
        """Check if BM25 index exists."""
        return self.index_path.exists() and self.bm25_retriever is not None