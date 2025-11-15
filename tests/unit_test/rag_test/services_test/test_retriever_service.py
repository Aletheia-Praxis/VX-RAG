"""
Unit tests for RetrieverService.
"""

import pytest
from unittest.mock import Mock, patch
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

    def test_postprocessors_initialized(self):
        """Test that postprocessors are initialized."""
        service = RetrieverService()
        # Metadata boost should be initialized
        assert service.metadata_boost is not None
        # Reranker may or may not be initialized depending on config

    @patch('llama_index.core.retrievers.QueryFusionRetriever')
    @patch('src.rag.services.retriever_service.service.VectorIndexRetriever')
    def test_set_index(self, mock_vector_retriever, mock_query_fusion, mock_index):
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

    def test_apply_filters(self):
        """Test metadata filter application using native _apply_filters."""
        service = RetrieverService()

        # Create mock nodes
        node1 = Mock()
        node1.metadata = {"lang": "en", "source": "docs"}
        
        node2 = Mock()
        node2.metadata = {"lang": "fr", "source": "docs"}
        
        node3 = Mock()
        node3.metadata = {"lang": "en", "source": "web"}
        
        nodes = [node1, node2, node3]
        
        # Test exact match filter
        filters = {"lang": "en"}
        result = service._apply_filters(nodes, filters)
        assert len(result) == 2  # node1 and node3
        
        # Test multiple filters
        filters = {"lang": "en", "source": "docs"}
        result = service._apply_filters(nodes, filters)
        assert len(result) == 1  # only node1

    def test_hybrid_search(self, mock_index):
        """Test hybrid search method."""
        service = RetrieverService()
        service.set_index(mock_index)

        with patch.object(service, 'retrieve') as mock_retrieve:
            mock_retrieve.return_value = [{"text": "test", "score": 0.8}]

            results = service.hybrid_search("test query", top_k=5)

            mock_retrieve.assert_called_once_with("test query", 5, None, search_type="hybrid")
            assert results == [{"text": "test", "score": 0.8}]
    
    def test_apply_postprocessors(self):
        """Test postprocessor chain application."""
        from llama_index.core.schema import NodeWithScore, TextNode
        
        service = RetrieverService()
        
        # Create real NodeWithScore objects
        node1 = NodeWithScore(
            node=TextNode(text="test1", metadata={"source": "docs", "lang": "en"}),
            score=0.8
        )
        node2 = NodeWithScore(
            node=TextNode(text="test2", metadata={}),
            score=0.7
        )
        
        nodes = [node1, node2]
        query = "test query"
        
        result = service._apply_postprocessors(query, nodes)
        
        # Should return nodes (possibly boosted/reranked)
        assert len(result) > 0
        assert result[0].score is not None
    
    def test_nodes_to_results(self):
        """Test node conversion to result format."""
        from llama_index.core.schema import NodeWithScore, TextNode
        
        service = RetrieverService()
        
        node = NodeWithScore(
            node=TextNode(text="test text", id_="node_1", metadata={"source": "test"}),
            score=0.85
        )
        
        results = service._nodes_to_results([node])
        
        assert len(results) == 1
        assert results[0]['text'] == "test text"
        assert results[0]['score'] == 0.85
        assert results[0]['metadata'] == {"source": "test"}
        assert results[0]['node_id'] == "node_1"
