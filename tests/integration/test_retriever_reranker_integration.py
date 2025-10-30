"""
Integration tests for Retriever and Reranker services.
"""

import pytest
from unittest.mock import Mock, patch

from src.rag.services.retriever_service.service import RetrieverService
from src.rag.services.reranker_service.service import RerankerService


class TestRetrieverRerankerIntegration:
    """Integration tests for Retriever and Reranker services working together."""

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

    def test_full_retrieval_reranking_workflow(self, mock_index, sample_nodes):
        """Test complete workflow from retrieval to reranking."""
        # Setup retriever
        with patch('src.rag.services.retriever_service.service.VectorIndexRetriever') as mock_vector_retriever, \
             patch('src.rag.services.retriever_service.service.QueryFusionRetriever') as mock_query_fusion:

            mock_retriever_instance = Mock()
            mock_retriever_instance.retrieve.return_value = sample_nodes
            mock_vector_retriever.return_value = mock_retriever_instance
            mock_query_fusion.return_value = mock_retriever_instance

            retriever = RetrieverService()
            retriever.set_index(mock_index)

            # Setup reranker
            with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
                mock_model = Mock()
                # Simulate cross-encoder scores (higher for more relevant docs)
                mock_model.predict.return_value = [0.9, 0.6, 0.8]  # security, database, backup for security query
                mock_cross_encoder.return_value = mock_model

                reranker = RerankerService()

                # Test full workflow
                query = "security best practices"

                # Step 1: Retrieve documents
                retrieved_docs = retriever.retrieve(query, top_k=5, search_type="semantic")
                assert len(retrieved_docs) == 3

                # Step 2: Rerank documents
                reranked_docs = reranker.rerank(query, retrieved_docs, top_k=2)
                assert len(reranked_docs) == 2

                # Verify reranking worked (security doc should be first)
                assert reranked_docs[0]['score'] == 0.9
                assert reranked_docs[1]['score'] == 0.8

                # Step 3: Prioritize by metadata
                priority_rules = {"source": "documentation"}
                prioritized_docs = reranker.prioritize_by_metadata(reranked_docs, priority_rules)

                # Verify prioritization worked
                assert 'priority_score' in prioritized_docs[0]
                assert prioritized_docs[0]['metadata']['source'] == 'documentation'

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
        """Test that services properly load and use configuration."""
        # Create temporary config file
        config_data = {
            'retriever': {
                'semantic_top_k': 10,
                'enable_hybrid': True
            },
            'reranker': {
                'model_name': 'cross-encoder/test-model',
                'top_k': 3,
                'enable_metadata_prioritization': True
            }
        }

        config_file = tmp_path / "test_config.yaml"
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Test retriever config loading
        retriever = RetrieverService(config_path=str(config_file))
        assert retriever.config['semantic_top_k'] == 10

        # Test reranker config loading
        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.return_value = Mock()
            reranker = RerankerService(config_path=str(config_file))
            assert reranker.config['model_name'] == 'cross-encoder/test-model'
            assert reranker.config['top_k'] == 3

    def test_error_handling_integration(self, mock_index):
        """Test error handling in integrated workflow."""
        # Test retriever error handling
        retriever = RetrieverService()
        # Don't set index - should handle gracefully
        results = retriever.retrieve("test query")
        assert results == []

        # Test reranker error handling
        with patch('src.rag.services.reranker_service.service.CrossEncoder') as mock_cross_encoder:
            mock_cross_encoder.side_effect = Exception("Model failed")

            reranker = RerankerService()
            documents = [{"text": "test", "score": 0.5}]

            # Should return documents as-is when reranking fails
            result = reranker.rerank("query", documents)
            assert result == documents

    def test_hybrid_search_integration(self, mock_index, sample_nodes):
        """Test hybrid search in integrated workflow."""
        with patch('src.rag.services.retriever_service.service.VectorIndexRetriever') as mock_vector_retriever, \
             patch('src.rag.services.retriever_service.service.QueryFusionRetriever') as mock_query_fusion:

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