"""
Integration tests for Retriever service with integrated postprocessors.

Note: RerankerService has been removed. Reranking is now handled by
native LlamaIndex postprocessors (SentenceTransformerRerank) within RetrieverService.
"""

import pytest
from unittest.mock import Mock, patch

from src.rag.services.retriever_service.service import RetrieverService


class TestRetrieverPostprocessorsIntegration:
    """Integration tests for Retriever service with native postprocessors."""

    @pytest.fixture
    def mock_index(self):
        """Create a mock VectorStoreIndex."""
        index = Mock()
        index.id = "test_index"
        return index

    @pytest.fixture
    def sample_nodes(self):
        """Create sample Node objects for testing."""
        
        nodes = []
        for i, doc in enumerate([
            {
                "text": "This is a document about security best practices for web applications.",
                "metadata": {"source": "documentation", "lang": "en", "topic": "security"}
            },
            {
                "text": "Database optimization techniques for better performance.",
                "metadata": {"source": "web", "lang": "en", "topic": "database"}
            },
            {
                "text": "Backup configuration guide for enterprise systems.",
                "metadata": {"source": "documentation", "lang": "en", "topic": "backup"}
            }
        ]):
            node = Mock()
            node.text = doc["text"]
            node.score = 0.8 - i * 0.1  # Decreasing scores
            node.metadata = doc["metadata"]
            node.id_ = f"node_{i+1}"
            nodes.append(node)
        
        return nodes

    def test_full_retrieval_postprocessing_workflow(self, mock_index, sample_nodes):
        """Test complete workflow from retrieval to postprocessing (metadata boost + reranking)."""
        # Setup retriever
        with patch('src.rag.services.retriever_service.service.VectorIndexRetriever') as mock_vector_retriever, \
             patch('llama_index.core.retrievers.QueryFusionRetriever') as mock_query_fusion:

            mock_retriever_instance = Mock()
            mock_retriever_instance.retrieve.return_value = sample_nodes
            mock_vector_retriever.return_value = mock_retriever_instance
            mock_query_fusion.return_value = mock_retriever_instance

            retriever = RetrieverService()
            retriever.set_index(mock_index)

            # Test full workflow with integrated postprocessors
            query = "security best practices"

            # Retrieve documents with postprocessing (metadata boost + reranking)
            retrieved_docs = retriever.retrieve(query, top_k=3, search_type="semantic")
            
            # Verify postprocessing workflow completed
            assert len(retrieved_docs) == 3
            assert 'score' in retrieved_docs[0]
            assert 'metadata' in retrieved_docs[0]
            
            # Verify metadata is preserved after postprocessing
            for doc in retrieved_docs:
                assert 'text' in doc
                assert 'metadata' in doc
                assert 'score' in doc or 'relevance' in doc

    def test_retrieval_with_filters_integration(self, mock_index):
        """Test retrieval with filters in integrated workflow."""
        with patch('src.rag.services.retriever_service.service.VectorIndexRetriever') as mock_vector_retriever:
            mock_retriever_instance = Mock()

            # Mock documents with different metadata
            docs_with_metadata = [
                Mock(text="English doc", score=0.8, metadata={"lang": "en", "source": "docs"}, id_="1"),
                Mock(text="French doc", score=0.7, metadata={"lang": "fr", "source": "docs"}, id_="2"),
                Mock(text="English web", score=0.6, metadata={"lang": "en", "source": "web"}, id_="3")
            ]
            mock_retriever_instance.retrieve.return_value = docs_with_metadata
            mock_vector_retriever.return_value = mock_retriever_instance

            retriever = RetrieverService()
            retriever.set_index(mock_index)

            # Test with language filter
            filters = {"lang": "en"}
            results = retriever.retrieve("test query", filters=filters)

            # Should only return English documents
            assert len(results) == 2
            assert all(doc['metadata']['lang'] == 'en' for doc in results)

    def test_services_config_integration(self, mock_index, tmp_path):
        """Test that RetrieverService properly loads and uses configuration."""
        # Create temporary config file
        config_data = {
            'retriever': {
                'semantic_top_k': 10,
                'enable_hybrid': True,
                'postprocessors': {
                    'metadata_boost': {'enabled': True},
                    'rerank': {'model': 'cross-encoder/test-model', 'top_n': 3}
                }
            }
        }

        config_file = tmp_path / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Test retriever config loading with postprocessors
        retriever = RetrieverService(config_path=str(config_file))
        assert retriever.config['semantic_top_k'] == 10
        assert retriever.config['enable_hybrid'] is True
        assert 'postprocessors' in retriever.config

    def test_error_handling_integration(self, mock_index):
        """Test error handling in integrated workflow."""
        # Test retriever error handling
        retriever = RetrieverService()
        # Don't set index - should handle gracefully
        results = retriever.retrieve("test query")
        assert results == []

        # Test postprocessor error handling
        retriever = RetrieverService()
        retriever.set_index(mock_index)
        
        # Should handle postprocessor failures gracefully
        # (RetrieverService has try-except around postprocessors)
        results = retriever.retrieve("test query", top_k=3)
        # Should return results even if postprocessing partially fails
        assert isinstance(results, list)

    def test_hybrid_search_integration(self, mock_index, sample_nodes):
        """Test hybrid search in integrated workflow."""
        with patch('src.rag.services.retriever_service.service.VectorIndexRetriever') as mock_vector_retriever, \
             patch('llama_index.core.retrievers.QueryFusionRetriever') as mock_query_fusion:

            mock_retriever_instance = Mock()
            mock_retriever_instance.retrieve.return_value = sample_nodes
            mock_vector_retriever.return_value = mock_retriever_instance
            mock_query_fusion.return_value = mock_retriever_instance

            retriever = RetrieverService()
            retriever.set_index(mock_index)

            # Test hybrid search
            results = retriever.retrieve("test query", search_type="hybrid")
            assert len(results) == 3

            # Test hybrid_search method
            hybrid_results = retriever.hybrid_search("test query", top_k=2)
            assert len(hybrid_results) == 2
    
    def test_metadata_boost_integration(self, mock_index, sample_nodes):
        """Test metadata boost postprocessor in integrated workflow."""
        with patch('src.rag.services.retriever_service.service.VectorIndexRetriever') as mock_vector_retriever, \
             patch('llama_index.core.retrievers.QueryFusionRetriever') as mock_query_fusion:

            mock_retriever_instance = Mock()
            mock_retriever_instance.retrieve.return_value = sample_nodes
            mock_vector_retriever.return_value = mock_retriever_instance
            mock_query_fusion.return_value = mock_retriever_instance

            # Setup retriever with integrated postprocessors
            retriever = RetrieverService()
            retriever.set_index(mock_index)

            # Test that postprocessing workflow completes successfully
            query = "security best practices"
            results = retriever.retrieve(query, top_k=3, search_type="semantic")

            # Verify that results were returned with postprocessing
            assert len(results) == 3

            # Verify postprocessing completed (metadata and scores preserved)
            assert 'score' in results[0] or 'relevance' in results[0]
            assert 'metadata' in results[0]
            
            # Verify metadata structure
            for doc in results:
                assert 'text' in doc
                assert 'metadata' in doc
    
    def test_postprocessor_metadata_boost_integration(self, mock_index):
        """Test MetadataBoostPostprocessor integration in RetrieverService."""
        with patch('src.rag.services.retriever_service.service.VectorIndexRetriever') as mock_vector_retriever:
            # Create mock nodes with different metadata
            from llama_index.core.schema import NodeWithScore, TextNode
            
            nodes = [
                NodeWithScore(node=TextNode(text="doc1", metadata={"source": "docs", "lang": "en"}), score=0.7),
                NodeWithScore(node=TextNode(text="doc2", metadata={}), score=0.8),
                NodeWithScore(node=TextNode(text="doc3", metadata={"source": "docs", "lang": "en", "topic": "security"}), score=0.75)
            ]
            
            mock_retriever_instance = Mock()
            mock_retriever_instance.retrieve.return_value = nodes
            mock_vector_retriever.return_value = mock_retriever_instance

            retriever = RetrieverService()
            retriever.set_index(mock_index)
            
            # Retrieve with postprocessors (including metadata boost)
            results = retriever.retrieve("test query", top_k=3)
            
            # Verify postprocessing completed
            assert len(results) == 3
            assert all('metadata' in doc for doc in results)
            
            # Verify metadata is preserved
            metadata_counts = [len(doc['metadata']) for doc in results]
            assert max(metadata_counts) >= 2  # At least one doc has multiple metadata fields