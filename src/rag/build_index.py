"""
FAISS index building module for VX-RAG system.

This module loads processed text documents from data/processed/,
creates embeddings using HuggingFace ... model,
and builds a FAISS vector index persisted under data/index/.
"""

import os
import logging
from pathlib import Path
from typing import List

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """Handles FAISS vector index creation and persistence."""

    def __init__(
        self,
        processed_dir: str = "data/processed",
        index_dir: str = "data/index",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the index builder.

        Args:
            processed_dir: Directory containing processed text files
            index_dir: Directory to store the FAISS index
            embedding_model: HuggingFace embedding model name
        """
        self.processed_dir = Path(processed_dir)
        self.index_dir = Path(index_dir)
        self.embedding_model = embedding_model

        # Create index directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self.embed_model = HuggingFaceEmbedding(model_name=self.embedding_model)

    def load_documents(self) -> List:
        """
        Load documents from the processed directory.

        Returns:
            List of loaded documents
        """
        if not self.processed_dir.exists():
            logger.error(f"Processed directory does not exist: {self.processed_dir}")
            return []

        try:
            logger.info(f"Loading documents from {self.processed_dir}")
            documents = SimpleDirectoryReader(
                input_dir=str(self.processed_dir),
                required_exts=[".txt"]
            ).load_data()
            logger.info(f"Successfully loaded {len(documents)} documents from {self.processed_dir}")
            return documents
        except Exception as e:
            logger.error(f"Failed to load documents from {self.processed_dir}: {e}")
            return []

    def build_index(self, documents: List) -> VectorStoreIndex:
        """
        Build FAISS vector index from documents.

        Args:
            documents: List of documents to index

        Returns:
            The created VectorStoreIndex
        """
        if not documents:
            raise ValueError("No documents provided for indexing")

        try:
            logger.info(f"Initializing FAISS vector store with dimension {self.embed_model.embed_dim}")
            # Initialize FAISS vector store
            d = self.embed_model.embed_dim  # Get dimension from embedding model
            faiss_index = faiss.IndexFlatL2(d)
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            logger.info(f"Building index with {len(documents)} documents using {self.embedding_model}")
            # Build index with documents
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True
            )

            logger.info(f"Successfully built index with {len(documents)} documents")
            return index

        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise

    def save_index(self, index: VectorStoreIndex) -> None:
        """
        Save the index to disk.

        Args:
            index: The VectorStoreIndex to save
        """
        try:
            index.storage_context.persist(persist_dir=str(self.index_dir))
            logger.info(f"Index saved to {self.index_dir}")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise

    def build_and_save_index(self) -> bool:
        """
        Load documents, build index, and save to disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Starting index building process")
            
            # Load documents
            documents = self.load_documents()
            if not documents:
                logger.error("No documents to index. Run ingestion first.")
                return False

            # Build index
            index = self.build_index(documents)

            # Save index
            self.save_index(index)

            logger.info("Index building and saving completed successfully")
            return True

        except Exception as e:
            logger.error(f"Index building failed: {e}")
            return False


def main():
    """Main entry point for index building."""
    builder = IndexBuilder()
    success = builder.build_and_save_index()

    if not success:
        logger.error("Index building failed. Check logs for details.")
        return 1

    logger.info("Index building completed successfully.")
    return 0


if __name__ == "__main__":
    exit(main())