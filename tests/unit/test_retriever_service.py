"""
Unit tests for RetrieverService.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml

from src.rag.services.retriever_service.service import RetrieverService


class TestRetrieverService:
    """Test cases for RetrieverService."""

    @pytest.fixture
    def mock_index(self):
        """Create a mock VectorStoreIndex."""
        index = Mock()
        index.id = "test_index"
        return index

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file."""
        config_data = {
            'retriever': {
                'semantic_top_k': 15,
                'hybrid_alpha': 0.7,
                'metadata_filters': [],
                'enable_hybrid': True
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        yield config_path
        Path(config_path).unlink()

    def test_init_without_config(self):
        """Test initialization without config file."""
        service = RetrieverService()
        assert service.index is None
        assert service.vector_retriever is None
        assert service.hybrid_retriever is None
        assert isinstance(service.config, dict)

    def test_init_with_config(self, temp_config_file):
        """Test initialization with config file."""
        service = RetrieverService(config_path=temp_config_file)
        assert service.config['semantic_top_k'] == 15
        assert service.config['hybrid_alpha'] == 0.7

    def test_load_config_valid_file(self, temp_config_file):
        """Test loading valid config file."""
        service = RetrieverService()
        config = service._load_config(temp_config_file)
        assert config['semantic_top_k'] == 15
        assert config['enable_hybrid'] is True

    def test_load_config_invalid_file(self):
        """Test loading invalid config file."""
        service = RetrieverService()
        config = service._load_config("nonexistent_file.yaml")
        assert config == {}

    @patch('src.rag.services.retriever_service.service.VectorIndexRetriever')
    @patch('src.rag.services.retriever_service.service.QueryFusionRetriever')
    def test_set_index(self, mock_query_fusion, mock_vector_retriever, mock_index):
        """Test setting index and initializing retrievers."""
        # Setup mocks
        mock_vector_retriever.return_value = Mock()
        mock_query_fusion.return_value = Mock()

        service = RetrieverService()
        service.set_index(mock_index)

        # Verify retrievers were created
        assert mock_vector_retriever.call_count == 2  # Called twice for vector and hybrid
        mock_query_fusion.assert_called_once()

        assert service.vector_retriever is not None
        assert service.hybrid_retriever is not None

    def test_retrieve_without_index(self):
        """Test retrieval without setting index."""
        service = RetrieverService()
        results = service.retrieve("test query")
        assert results == []

    @patch('src.rag.services.retriever_service.service.VectorIndexRetriever')
    def test_retrieve_semantic_search(self, mock_vector_retriever, mock_index):
        """Test semantic search retrieval."""
        # Setup mock retriever
        mock_retriever_instance = Mock()
        mock_node = Mock()
        mock_node.text = "test text"
        mock_node.score = 0.8
        mock_node.metadata = {"source": "test"}
        mock_node.id_ = "node_1"
        mock_retriever_instance.retrieve.return_value = [mock_node]

        mock_vector_retriever.return_value = mock_retriever_instance

        service = RetrieverService()
        service.set_index(mock_index)

        results = service.retrieve("test query", search_type="semantic")

        assert len(results) == 1
        assert results[0]['text'] == "test text"
        assert results[0]['score'] == 0.8
        assert results[0]['metadata'] == {"source": "test"}

    @patch('src.rag.services.retriever_service.service.VectorIndexRetriever')
    def test_retrieve_with_filters(self, mock_vector_retriever, mock_index):
        """Test retrieval with metadata filters."""
        # Setup mock retriever
        mock_retriever_instance = Mock()
        mock_node = Mock()
        mock_node.text = "test text"
        mock_node.score = 0.8
        mock_node.metadata = {"lang": "en", "source": "docs"}
        mock_node.id_ = "node_1"
        mock_retriever_instance.retrieve.return_value = [mock_node]

        mock_vector_retriever.return_value = mock_retriever_instance

        service = RetrieverService()
        service.set_index(mock_index)

        filters = {"lang": "en"}
        results = service.retrieve("test query", filters=filters)

        assert len(results) == 1
        mock_retriever_instance.retrieve.assert_called_once()

    def test_matches_filters(self):
        """Test metadata filter matching."""
        service = RetrieverService()

        # Test exact match
        metadata = {"lang": "en", "source": "docs"}
        filters = {"lang": "en"}
        assert service._matches_filters(metadata, filters) is True

        # Test no match
        filters = {"lang": "fr"}
        assert service._matches_filters(metadata, filters) is False

        # Test list filter
        filters = {"lang": ["en", "de"]}
        assert service._matches_filters(metadata, filters) is True

    def test_hybrid_search(self, mock_index):
        """Test hybrid search method."""
        service = RetrieverService()
        service.set_index(mock_index)

        with patch.object(service, 'retrieve') as mock_retrieve:
            mock_retrieve.return_value = [{"text": "test", "score": 0.8}]

            results = service.hybrid_search("test query", top_k=5)

            mock_retrieve.assert_called_once_with("test query", 5, None, search_type="hybrid")
            assert results == [{"text": "test", "score": 0.8}]
