#!/usr/bin/env python3
"""
Query module for VX-RAG system.

This module loads the existing FAISS index and performs semantic search
to retrieve relevant documents for a given query.
"""

import logging
from pathlib import Path
from typing import List, Tuple

from llama_index.core import load_index_from_storage, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentQuery:
    """Handles document querying against FAISS index."""

    def __init__(self, index_dir: str = "data/index"):
        """
        Initialize the document query handler.

        Args:
            index_dir: Directory containing the persisted FAISS index
        """
        self.index_dir = Path(index_dir)
        self.index = None

    def load_index(self) -> bool:
        """
        Load the FAISS index from disk.

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
            self.index = load_index_from_storage(storage_context)
            logger.info(f"Index loaded from {self.index_dir}")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False

    def query_documents(self, query: str, top_k: int = 3) -> List[Tuple[str, float, dict]]:
        """
        Query the index and retrieve top-k similar documents.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List of tuples containing (text, score, metadata) for each result
        """
        if self.index is None:
            if not self.load_index():
                return []

        try:
            # Create query engine
            query_engine = self.index.as_query_engine(similarity_top_k=top_k)

            # Perform query
            response = query_engine.query(query)

            # Extract results with metadata
            results = []
            for node in response.source_nodes:
                text = node.node.text[:500] + "..." if len(node.node.text) > 500 else node.node.text
                score = node.score
                metadata = node.node.metadata
                results.append((text, score, metadata))

            return results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []

    def print_results(self, query: str, results: List[Tuple[str, float, dict]]) -> None:
        """
        Print query results in a readable format.

        Args:
            query: The original query
            results: List of query results
        """
        print(f"\nQuery: {query}")
        print("=" * 50)

        if not results:
            print("No results found.")
            return

        for i, (text, score, metadata) in enumerate(results, 1):
            print(f"\nMatch #{i} (Score: {score:.4f})")
            print("-" * 30)

            # Print metadata if available
            if metadata:
                source = metadata.get('file_name', 'Unknown')
                print(f"Source: {source}")

            print(f"Text: {text}")
            print()


def main():
    """Main entry point for testing document queries."""
    query_handler = DocumentQuery()

    # Test query
    test_query = "Example query: malware analysis dataset"

    results = query_handler.query_documents(test_query)

    if not results:
        logger.error("Failed to perform query. Check if index exists and is properly built.")
        return 1

    query_handler.print_results(test_query, results)
    return 0


if __name__ == "__main__":
    exit(main())