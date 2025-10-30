"""Embedder service for generating text embeddings.

This module provides the EmbeddingService class that uses LlamaIndex's
HuggingFaceEmbedding to generate embeddings for text chunks.
"""

import logging
from typing import Any

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using HuggingFace models.

    This class provides a wrapper around LlamaIndex's HuggingFaceEmbedding
    for compatibility with the RAG system.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_size: int = 1000):
        """Initialize the embedding service.

        Args:
            model_name: Name of the HuggingFace model to use.
            cache_size: Maximum number of cached embeddings (not used by HuggingFaceEmbedding).
        """
        self.model_name = model_name
        self.cache_size = cache_size

        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                embed_batch_size=10,
                cache_folder=None,
                trust_remote_code=False
            )
            logger.info(f"Initialized embedding service with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model {model_name}: {e}")
            raise RuntimeError(f"Could not load embedding model: {e}") from e

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the embedding model.

        Returns:
            Dictionary with model information.
        """
        return {
            "model_name": self.model_name,
            "provider": "HuggingFace",
            "cache_size": self.cache_size
        }

    def embed_single(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.
        """
        if not text.strip():
            return []
        result = self.embed_model.get_text_embedding(text)
        return list(result) if result else []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        if not texts:
            return []
        result = self.embed_model.get_text_embedding_batch(texts)
        return [list(vec) for vec in result] if result else []

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Main embedding interface.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self.embed_batch(texts)
