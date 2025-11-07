"""
Unit tests for RerankerService.
"""

import pytest
from unittest.mock import Mock, patch
import tempfile
import yaml
from pathlib import Path

from src.rag.services.reranker_service.service import RerankerService


class TestRerankerService:
    """Test cases for RerankerService."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file."""
        config_data = {
            'reranker': {
                'model_name': 'cross-encoder/test-model',
                'top_k': 3,
                'device': 'cpu',
                'metadata_boost': 0.2,
                'enable_metadata_prioritization': True
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        yield config_path
        Path(config_path).unlink()

    def test_init_without_config(self):
        """Test initialization without config file."""
        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.return_value = Mock()

            service = RerankerService()
            assert service.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"
            assert service.model is not None
            mock_cross_encoder.assert_called_once()

    def test_init_with_config(self, temp_config_file):
        """Test initialization with config file."""
        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.return_value = Mock()

            service = RerankerService(config_path=temp_config_file)
            assert service.config['model_name'] == 'cross-encoder/test-model'
            assert service.config['top_k'] == 3
            mock_cross_encoder.assert_called_with('cross-encoder/test-model')

    def test_init_with_custom_model(self):
        """Test initialization with custom model name."""
        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.return_value = Mock()

            service = RerankerService(model_name="custom-model")
            assert service.model_name == "custom-model"
            mock_cross_encoder.assert_called_with("custom-model")

    def test_init_model_failure(self):
        """Test initialization when model loading fails."""
        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.side_effect = Exception("Model load failed")

            service = RerankerService()
            assert service.model is None

    def test_load_config_valid_file(self, temp_config_file):
        """Test loading valid config file."""
        service = RerankerService()
        config = service._load_config(temp_config_file)
        assert config['model_name'] == 'cross-encoder/test-model'
        assert config['enable_metadata_prioritization'] is True

    def test_load_config_invalid_file(self):
        """Test loading invalid config file."""
        service = RerankerService()
        config = service._load_config("nonexistent_file.yaml")
        assert config == {}

    def test_rerank_with_model(self):
        """Test reranking with loaded model."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7, 0.5]

        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.return_value = mock_model

            service = RerankerService()
            documents = [
                {"text": "doc1", "score": 0.5},
                {"text": "doc2", "score": 0.6},
                {"text": "doc3", "score": 0.4}
            ]

            result = service.rerank("test query", documents)

            assert len(result) == 3
            assert result[0]['score'] == 0.9
            assert result[1]['score'] == 0.7
            assert result[2]['score'] == 0.5
            mock_model.predict.assert_called_once()

    def test_rerank_without_model(self):
        """Test reranking when model is not loaded."""
        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.side_effect = Exception("Model failed")

            service = RerankerService()
            documents = [{"text": "doc1", "score": 0.5}]

            result = service.rerank("test query", documents)
            assert result == documents  # Should return documents as-is

    def test_rerank_empty_documents(self):
        """Test reranking with empty document list."""
        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.return_value = Mock()

            service = RerankerService()
            result = service.rerank("test query", [])
            assert result == []

    def test_rerank_with_top_k(self):
        """Test reranking with top_k limit."""
        mock_model = Mock()
        mock_model.predict.return_value = [0.9, 0.7, 0.5]

        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.return_value = mock_model

            service = RerankerService()
            documents = [
                {"text": "doc1", "score": 0.5},
                {"text": "doc2", "score": 0.6},
                {"text": "doc3", "score": 0.4}
            ]

            result = service.rerank("test query", documents, top_k=2)

            assert len(result) == 2
            assert result[0]['score'] == 0.9
            assert result[1]['score'] == 0.7

    def test_prioritize_by_metadata(self):
        """Test metadata-based prioritization."""
        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.return_value = Mock()

            service = RerankerService()
            documents = [
                {"text": "doc1", "score": 0.8, "metadata": {"source": "docs", "lang": "en"}},
                {"text": "doc2", "score": 0.7, "metadata": {"source": "web", "lang": "en"}},
                {"text": "doc3", "score": 0.9, "metadata": {"source": "docs", "lang": "fr"}}
            ]

            priority_rules = {"source": "docs", "lang": "en"}
            result = service.prioritize_by_metadata(documents, priority_rules)

            # Check that priority scores were added
            assert 'priority_score' in result[0]
            assert 'priority_score' in result[1]
            assert 'priority_score' in result[2]

            # Documents with matching metadata should be prioritized
            assert result[0]['priority_score'] >= result[1]['priority_score']

    def test_prioritize_by_metadata_empty_rules(self):
        """Test prioritization with empty rules."""
        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.return_value = Mock()

            service = RerankerService()
            documents = [{"text": "doc1", "score": 0.8}]

            result = service.prioritize_by_metadata(documents, {})
            assert result == documents

    def test_prioritize_by_metadata_range_rules(self):
        """Test prioritization with range-based rules."""
        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.return_value = Mock()

            service = RerankerService()
            documents = [
                {"text": "doc1", "score": 0.8, "metadata": {"year": 2020}},
                {"text": "doc2", "score": 0.7, "metadata": {"year": 2023}}
            ]

            priority_rules = {"year": {"min": 2022, "weight": 2.0}}
            result = service.prioritize_by_metadata(documents, priority_rules)

            # Document from 2023 should have higher priority
            assert result[0]['priority_score'] >= result[1]['priority_score']
