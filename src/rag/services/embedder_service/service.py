"""
Embedder Service implementation.

Provides classes for text embedding generation.
"""

from typing import List, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Initialize embedding model
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        self.embed_model: HuggingFaceEmbedding = HuggingFaceEmbedding(model_name=self.model_name)
        logger.info(f"Initialized embedding model: {self.model_name}")
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts."""
        try:
            embeddings = []
            for text in texts:
                embedding = self.embed_model.get_text_embedding(text)
                embeddings.append(embedding)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        try:
            embedding = self.embed_model.get_text_embedding(text)
            logger.debug(f"Generated embedding for single text (dim: {len(embedding)})")
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            raise

# TODO: Add caching, batch processing, error handling
