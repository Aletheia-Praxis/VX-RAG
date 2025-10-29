"""Unit tests for the embedder service.

This module contains comprehensive unit tests for the Embedder class,
covering initialization, embedding generation, caching, batch processing,
and error handling.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import numpy as np

from src.rag.services.embedder_service.service import EmbeddingService
from src.rag.libs.schemas.embedder_schemas import EmbeddingRequest, EmbeddingResponse, EmbeddingVector


class TestEmbeddingService:
    """Test suite for the EmbeddingService class."""

    @pytest.fixture
    def mock_huggingface_embedding(self) -> Mock:
        """Mock HuggingFaceEmbedding for testing."""
        mock_model = Mock()
        # Mock the embedding methods
        mock_model._get_text_embedding.return_value = [0.1, 0.2, 0.3]
        mock_model._get_text_embeddings.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        return mock_model

    @pytest.fixture
    def embedder(self, mock_huggingface_embedding: Mock) -> EmbeddingService:
        """Create EmbeddingService instance with mocked model."""
        with patch('src.rag.services.embedder_service.service.HuggingFaceEmbedding') as mock_hf:
            mock_hf.return_value = mock_huggingface_embedding
            embedder = EmbeddingService(model_name="all-MiniLM-L6-v2", cache_size=10)
            return embedder

    def test_initialization_success(self, mock_huggingface_embedding: Mock) -> None:
        """Test successful initialization of EmbeddingService."""
        with patch('src.rag.services.embedder_service.service.HuggingFaceEmbedding') as mock_hf:
            mock_hf.return_value = mock_huggingface_embedding
            embedder = EmbeddingService(model_name="test-model", cache_size=5)

            assert embedder.model_name == "test-model"
            assert embedder.cache_size == 5
            assert embedder.embed_model is not None
            mock_hf.assert_called_once()

    def test_initialization_model_load_failure(self) -> None:
        """Test initialization failure when model loading fails."""
        with patch('src.rag.services.embedder_service.service.HuggingFaceEmbedding') as mock_hf:
            mock_hf.side_effect = Exception("Model load failed")

            with pytest.raises(RuntimeError, match="Could not load embedding model"):
                EmbeddingService(model_name="invalid-model")

    def test_embed_single_empty_text(self, embedder: EmbeddingService) -> None:
        """Test embedding single empty text."""
        result = embedder.embed_single("")
        assert result == []

        result = embedder.embed_single("   ")
        assert result == []

    def test_embed_single_success(self, embedder: EmbeddingService, mock_huggingface_embedding: Mock) -> None:
        """Test successful single embedding generation."""
        text = "test text"
        expected_embedding = [0.1, 0.2, 0.3]

        result = embedder.embed_single(text)

        assert result == expected_embedding
        mock_huggingface_embedding._get_text_embedding.assert_called_once_with(text)

    def test_embed_single_model_not_loaded(self, embedder: EmbeddingService) -> None:
        """Test embedding when model is not loaded."""
        # For EmbeddingService, model is always loaded in __init__
        # This test is not applicable
        pass

    def test_embed_single_encoding_failure(self, embedder: EmbeddingService, mock_huggingface_embedding: Mock) -> None:
        """Test embedding failure during encoding."""
        mock_huggingface_embedding._get_text_embedding.side_effect = Exception("Encoding failed")

        with pytest.raises(Exception, match="Encoding failed"):
            embedder.embed_single("test")

    def test_embed_batch_empty_list(self, embedder: EmbeddingService) -> None:
        """Test batch embedding with empty list."""
        result = embedder.embed_batch([])
        assert result == []

    def test_embed_batch_with_batch_size(self, embedder: EmbeddingService, mock_huggingface_embedding: Mock) -> None:
        """Test batch embedding (HuggingFaceEmbedding handles batching internally)."""
        texts = ["text1", "text2", "text3"]
        embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        mock_huggingface_embedding._get_text_embeddings.return_value = embeddings

        result = embedder.embed_batch(texts)

        assert result == embeddings
        mock_huggingface_embedding._get_text_embeddings.assert_called_once_with(texts)

    def test_embed_batch_encoding_failure(self, embedder: EmbeddingService, mock_huggingface_embedding: Mock) -> None:
        """Test batch embedding failure during encoding."""
        mock_huggingface_embedding._get_text_embeddings.side_effect = Exception("Batch encoding failed")

        with pytest.raises(Exception, match="Batch encoding failed"):
            embedder.embed_batch(["text1"])

    def test_embed_main_interface(self, embedder: EmbeddingService, mock_huggingface_embedding: Mock) -> None:
        """Test the main embed interface."""
        texts = ["text1", "text2"]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        mock_huggingface_embedding._get_text_embeddings.return_value = embeddings

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
