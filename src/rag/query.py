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
from llama_index.llms.ollama import Ollama

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentQuery:
    """Handles document querying against FAISS index."""

    def __init__(self, index_dir: str = "data/index", llm_model: str = "llama3"):
        """
        Initialize the document query handler.

        Args:
            index_dir: Directory containing the persisted FAISS index
            llm_model: Name of the Ollama LLM model to use
        """
        self.index_dir = Path(index_dir)
        self.llm_model = llm_model
        self.index = None
        self.llm = Ollama(model=self.llm_model)

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

    def query_documents(self, query: str, top_k: int = 3) -> Tuple[str, List[Tuple[str, float, dict]]]:
        """
        Query the index and retrieve top-k similar documents with generated response.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            Tuple of (generated_response, list of (text, score, metadata) for each result)
        """
        if self.index is None:
            if not self.load_index():
                return "", []

        try:
            # Ensure index is loaded
            if self.index is None:
                if not self.load_index():
                    return "", []
            
            assert self.index is not None  # For type checker
            
            # Create query engine with LLM
            query_engine = self.index.as_query_engine(
                similarity_top_k=top_k,
                llm=self.llm
            )

            # Perform query with response generation
            response = query_engine.query(query)
            generated_response = str(response)

            # Extract source nodes for detailed results
            results = []
            for node in response.source_nodes:
                text = node.text[:500] + "..." if len(node.text) > 500 else node.text
                score = node.score
                metadata = node.metadata
                results.append((text, score, metadata))

            return generated_response, results

        except Exception as e:
            logger.error(f"Query failed: {e}")
            return "", []

    def print_results(self, query: str, response: str, results: List[Tuple[str, float, dict]]) -> None:
        """
        Print query results and generated response in a readable format.

        Args:
            query: The original query
            response: The generated response from LLM
            results: List of query results
        """
        print(f"\nQuery: {query}")
        print("=" * 50)
        print(f"Generated Response: {response}")
        print("=" * 50)

        if not results:
            print("No source documents found.")
            return

        print(f"\nSource Documents (Top {len(results)}):")
        for i, (text, score, metadata) in enumerate(results, 1):
            print(f"\nDocument #{i} (Similarity Score: {score:.4f})")
            print("-" * 40)

            # Print metadata if available
            if metadata:
                source = metadata.get('file_name', metadata.get('file_path', 'Unknown'))
                print(f"Source: {source}")

            print(f"Text: {text}")
            print()


def main():
    """Main entry point for testing document queries."""
    query_handler = DocumentQuery()

    # Test queries
    test_queries = [
        "What is malware analysis?",
        "What is the easiest way to transfer malware to the target system?",
        "How to detect malicious code?"
    ]

    for test_query in test_queries:
        logger.info(f"Testing query: {test_query}")
        response, results = query_handler.query_documents(test_query)

        if not response and not results:
            logger.error("Failed to perform query. Check if index exists and is properly built.")
            return 1

        query_handler.print_results(test_query, response, results)

    return 0


if __name__ == "__main__":
    exit(main())