"""
Embeddings module for VX-RAG.

This module handles local embedding generation using sentence-transformers.
Supports CPU-only inference.
"""

from sentence_transformers import SentenceTransformer

class LocalEmbeddings:
    """
    Local embeddings using sentence-transformers.
    """

    def __init__(self, model_name: str = "..."):
        """
        Initialize the embedding model.

        Args:
            model_name: Name of the sentence-transformer model.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

if __name__ == "__main__":
    # Example usage
    embedder = LocalEmbeddings()
    texts = ["This is a test document.", "Another document."]
    embeddings = embedder.encode(texts)
    print(f"Generated {len(embeddings)} embeddings")