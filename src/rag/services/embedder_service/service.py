"""Embedder service for generating text embeddings.

This module provides the EmbeddingService class that uses LlamaIndex's
HuggingFaceEmbedding to generate embeddings for text chunks.
"""

import logging
from typing import Any, Optional

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from src.utils.config_loader import get_embedding_config

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using HuggingFace models.

    This class provides a wrapper around LlamaIndex's HuggingFaceEmbedding
    for compatibility with the RAG system.
    """

    def __init__(
        self, 
        model_name: Optional[str] = None, 
        cache_size: Optional[int] = None,
        batch_size: Optional[int] = None,
        trust_remote_code: Optional[bool] = None,
        config_path: Optional[str] = None
    ):
        """Initialize the embedding service.

        Args:
            model_name: Name of the HuggingFace model to use. If None, loads from config.
            cache_size: Maximum number of cached embeddings. If None, loads from config.
            batch_size: Batch size for embedding generation. If None, loads from config.
            trust_remote_code: Whether to trust remote code. If None, loads from config.
            config_path: Path to settings.yaml. If None, uses default location.
        """
        # Load config if parameters not provided
        if any(param is None for param in [model_name, cache_size, batch_size, trust_remote_code]):
            config = get_embedding_config(config_path)
            model_name = model_name or config['embedding_model']
            cache_size = cache_size or config['embedding_cache_size']
            batch_size = batch_size or config['embedding_batch_size']
            trust_remote_code = trust_remote_code if trust_remote_code is not None else config['embedding_trust_remote_code']
        
        # Ensure values are set
        assert model_name is not None, "model_name must be set"
        assert cache_size is not None, "cache_size must be set"
        assert batch_size is not None, "batch_size must be set"
        assert trust_remote_code is not None, "trust_remote_code must be set"
        
        self.model_name = model_name
        self.cache_size = cache_size
        self.batch_size = batch_size

        try:
            self.embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                embed_batch_size=batch_size,
                cache_folder=None,
                trust_remote_code=trust_remote_code
            )
            logger.info(
                f"Initialized embedding service: model={model_name}, "
                f"batch_size={batch_size}, cache_size={cache_size}"
            )
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
