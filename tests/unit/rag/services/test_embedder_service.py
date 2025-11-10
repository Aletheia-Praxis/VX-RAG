"""Unit tests for the embedder service.

This module contains comprehensive unit tests for the Embedder class,
covering initialization, embedding generation, caching, batch processing,
and error handling.
"""

import pytest
from unittest.mock import Mock, patch

from src.rag.services.embedder_service.service import EmbeddingService


class TestEmbeddingService:
    """Test suite for the EmbeddingService class."""

    @pytest.fixture
    def mock_huggingface_embedding(self) -> Mock:
        """Mock HuggingFaceEmbedding for testing."""
        mock_model = Mock()
        # Mock the embedding methods
        mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]
        mock_model.get_text_embedding_batch.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        return mock_model

    @pytest.fixture
    def embedder(self, mock_huggingface_embedding: Mock) -> EmbeddingService:
        """Create EmbeddingService instance with mocked model."""
        with patch('src.rag.services.embedder_service.service.HuggingFaceEmbedding') as mock_hf:
            mock_hf.return_value = mock_huggingface_embedding
            embedder = EmbeddingService(
                model_name="all-MiniLM-L6-v2", 
                cache_size=10, 
                batch_size=10, 
                trust_remote_code=False
            )
            return embedder

    def test_initialization_success(self) -> None:
        """Test successful initialization of EmbeddingService."""
        with patch('src.rag.services.embedder_service.service.HuggingFaceEmbedding') as mock_hf:
            mock_hf.return_value = Mock()
            embedder = EmbeddingService(
                model_name="test-model", 
                cache_size=5, 
                batch_size=10, 
                trust_remote_code=False
            )

            assert embedder.model_name == "test-model"
            assert embedder.cache_size == 5
            assert embedder.embed_model is not None
            mock_hf.assert_called_once()

    def test_initialization_model_load_failure(self) -> None:
        """Test initialization failure when model loading fails."""
        with patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding') as mock_hf:
            mock_hf.side_effect = Exception("Model load failed")

            with pytest.raises(RuntimeError, match="Could not load embedding model"):
                EmbeddingService(model_name="invalid-model")

    def test_embed_single_empty_text(self, embedder: EmbeddingService) -> None:
        """Test embedding single empty text."""
        result = embedder.embed_single("")
        assert result == []

        result = embedder.embed_single("   ")
        assert result == []

    def test_embed_single_success(self, embedder: EmbeddingService, monkeypatch) -> None:
        """Test successful single embedding generation."""
        text = "test text"
        expected_embedding = [0.1, 0.2, 0.3]

        monkeypatch.setattr(embedder.embed_model, 'get_text_embedding', lambda text: expected_embedding)
        result = embedder.embed_single(text)

        assert result == expected_embedding

    def test_embed_single_model_not_loaded(self, embedder: EmbeddingService) -> None:
        """Test embedding when model is not loaded."""
        # For EmbeddingService, model is always loaded in __init__
        # This test is not applicable
        pass

    def test_embed_single_encoding_failure(self, embedder: EmbeddingService, monkeypatch) -> None:
        """Test embedding failure during encoding."""
        def mock_get_text_embedding(text):
            raise Exception("Encoding failed")
        
        monkeypatch.setattr(embedder.embed_model, 'get_text_embedding', mock_get_text_embedding)

        with pytest.raises(Exception, match="Encoding failed"):
            embedder.embed_single("test")

    def test_embed_batch_empty_list(self, embedder: EmbeddingService) -> None:
        """Test batch embedding with empty list."""
        result = embedder.embed_batch([])
        assert result == []

    def test_embed_batch_with_batch_size(self, embedder: EmbeddingService, monkeypatch) -> None:
        """Test batch embedding (HuggingFaceEmbedding handles batching internally)."""
        texts = ["text1", "text2", "text3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        monkeypatch.setattr(embedder.embed_model, 'get_text_embedding_batch', lambda texts: embeddings)
        result = embedder.embed_batch(texts)

        assert result == embeddings

    def test_embed_batch_encoding_failure(self, embedder: EmbeddingService, monkeypatch) -> None:
        """Test batch embedding failure during encoding."""
        def mock_get_text_embedding_batch(texts):
            raise Exception("Batch encoding failed")
        
        monkeypatch.setattr(embedder.embed_model, 'get_text_embedding_batch', mock_get_text_embedding_batch)

        with pytest.raises(Exception, match="Batch encoding failed"):
            embedder.embed_batch(["text1"])

    def test_embed_main_interface(self, embedder: EmbeddingService, monkeypatch) -> None:
        """Test the main embed interface."""
        texts = ["text1", "text2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        monkeypatch.setattr(embedder.embed_model, 'get_text_embedding_batch', lambda texts: embeddings)
        result = embedder.embed(texts)

        assert result == embeddings

    def test_get_model_info_success(self, embedder: EmbeddingService, mock_huggingface_embedding: Mock) -> None:
        """Test getting model information successfully."""
        info = embedder.get_model_info()

        expected = {
            "model_name": "all-MiniLM-L6-v2",
            "provider": "HuggingFace",
            "cache_size": 10
        }
        assert info == expected

    def test_get_model_info_model_not_loaded(self, embedder: EmbeddingService) -> None:
        """Test getting model info when model is not loaded."""
        # For EmbeddingService, model is always loaded in __init__
        # This test is not applicable
        pass

    def test_embeddings_are_normalized(self) -> None:
        """Test that embeddings are automatically normalized to unit length.
        
        This test verifies that the embedding model (all-MiniLM-L6-v2) automatically
        normalizes all output vectors to unit length (L2 norm = 1.0).
        
        This is critical because:
        1. FAISS index uses METRIC_INNER_PRODUCT
        2. For normalized vectors, inner product = cosine similarity
        3. Without normalization, METRIC_INNER_PRODUCT would not give cosine similarity
        
        The all-MiniLM-L6-v2 model includes a 'Normalize' module in its architecture
        that ensures all embeddings have L2 norm = 1.0.
        """
        import numpy as np
        
        # Initialize real model (not mocked) for this test
        embedder = EmbeddingService(
            model_name="all-MiniLM-L6-v2",
            cache_size=10,
            batch_size=10,
            trust_remote_code=False
        )
        
        # Test various texts to ensure normalization is consistent
        test_texts = [
            "test",
            "malware analysis",
            "cybersecurity research document",
            "APT group uses zero-day exploit for credential harvesting",
            "a" * 100,  # Long repetitive text
        ]
        
        for text in test_texts:
            # Generate embedding
            embedding = embedder.embed_single(text)
            
            # Calculate L2 norm
            norm = np.linalg.norm(embedding)
            
            # Assert that norm is very close to 1.0
            # Using tolerance of 1e-6 for floating point comparison
            assert abs(norm - 1.0) < 1e-6, (
                f"Embedding for text '{text[:50]}...' is not normalized. "
                f"Expected norm=1.0, got norm={norm:.10f}"
            )
        
        # Test batch embeddings as well
        batch_embeddings = embedder.embed_batch(test_texts)
        
        for i, embedding in enumerate(batch_embeddings):
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 1e-6, (
                f"Batch embedding {i} for text '{test_texts[i][:50]}...' is not normalized. "
                f"Expected norm=1.0, got norm={norm:.10f}"
            )
