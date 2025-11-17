"""
Unit tests for RAGOrchestrator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import yaml

from src.rag.orchestrator import RAGOrchestrator


class TestRAGOrchestrator:
    """Test cases for RAGOrchestrator."""

    @pytest.fixture
    def temp_config_file(self):
        """Create a temporary config file."""
        config_data = {
            'retriever': {
                'semantic_top_k': 15,
                'enable_hybrid': True
            },
            'embedder': {
                'embedding_model': 'all-MiniLM-L6-v2',
                'embedding_batch_size': 32,
                'embedding_trust_remote_code': False
            },
            'bm25': {
                'index_dir': './data/index/bm25',
                'similarity_top_k': 20
            },
            'reranker': {
                'enable_metadata_prioritization': True,
                'model_name': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                'top_n': 5
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        yield config_path
        Path(config_path).unlink()

    @pytest.fixture
    def mock_index(self):
        """Create a mock VectorStoreIndex."""
        index = Mock()
        index.id = "test_index"
        return index

    def test_init_without_auto_load(self, temp_config_file):
        """Test initialization without auto loading."""
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            auto_load=False
        )
        assert not orchestrator._initialized
        assert not orchestrator._indexes_loaded
        assert orchestrator._vector_store is None
        assert orchestrator._retriever is None

    @patch('src.rag.orchestrator.VectorStoreClient')
    @patch('src.rag.orchestrator.BM25IndexManager')
    @patch('src.rag.orchestrator.RetrieverService')
    @patch('src.rag.orchestrator.ContextAssembler')
    @patch('src.rag.orchestrator.Settings')
    def test_query_hybrid_search(self, mock_settings, mock_assembler, mock_retriever,
                                mock_bm25_manager, mock_vector_store, temp_config_file, mock_index):
        """Test query method with hybrid search using QueryFusionRetriever."""
        from pathlib import Path
        
        # Setup mocks
        mock_vector_client = Mock()
        mock_vector_client.index = mock_index
        mock_vector_client.load_index.return_value = True
        mock_vector_store.return_value = mock_vector_client

        mock_bm25 = Mock()
        mock_bm25.exists.return_value = True
        mock_bm25.load.return_value = Mock()  # Mock BM25 retriever
        mock_bm25_manager.return_value = mock_bm25

        mock_retriever_instance = Mock()
        mock_retriever_instance.retrieve.return_value = [
            {
                'text': 'test document',
                'score': 0.8,
                'metadata': {'source': 'test'},
                'node_id': 'node_1'
            }
        ]
        mock_retriever.return_value = mock_retriever_instance

        mock_assembler_instance = Mock()
        mock_context_payload = Mock()
        mock_context_payload.context = [
            Mock(id='node_1', text='test document', score=0.8, meta={'source': 'test'})
        ]
        mock_context_payload.total_tokens_estimate.return_value = 100
        mock_assembler_instance.assemble_context.return_value = mock_context_payload
        mock_assembler.return_value = mock_assembler_instance

        # Create orchestrator without auto_load first
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir="data/index",
            auto_load=False
        )
        
        # Manually set up the orchestrator state for testing
        orchestrator._vector_store = mock_vector_client
        orchestrator._bm25_manager = mock_bm25
        orchestrator._retriever = mock_retriever_instance
        orchestrator._assembler = mock_assembler_instance
        orchestrator._initialized = True
        orchestrator._indexes_loaded = True

        # Execute query
        result = orchestrator.query(
            query="test query",
            top_k=5,
            search_type="hybrid"
        )

        # Verify results
        assert result['query'] == "test query"
        assert len(result['context']) == 1
        assert result['context'][0]['text'] == 'test document'
        assert result['retrieval_stats']['search_type'] == 'hybrid'
        assert 'note' in result['retrieval_stats']
        assert 'Hybrid search and postprocessing via native LlamaIndex components' in result['retrieval_stats']['note']

        # Verify retriever was called with correct parameters
        mock_retriever_instance.retrieve.assert_called_once_with(
            query="test query",
            top_k=20,  # top_k * 4 = 5 * 4
            search_type="hybrid"
        )

    def test_query_without_initialization(self, temp_config_file):
        """Test query fails when services not initialized."""
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            auto_load=False
        )

        with pytest.raises(RuntimeError, match="RAG services not initialized"):
            orchestrator.query("test query")

    def test_query_without_indexes_loaded(self, temp_config_file):
        """Test query fails when indexes not loaded."""
        with patch('src.rag.orchestrator.VectorStoreClient') as mock_vector_store:
            mock_vector_client = Mock()
            mock_vector_client.index = None  # No index loaded
            mock_vector_client.load_index.return_value = False
            mock_vector_store.return_value = mock_vector_client

            orchestrator = RAGOrchestrator(
                config_path=temp_config_file,
                persist_dir="data/index",
                auto_load=True
            )

            with pytest.raises(RuntimeError, match="Indexes not loaded"):
                orchestrator.query("test query")

    def test_search_documents(self, temp_config_file, mock_index):
        """Test document search method."""
        with patch('src.rag.orchestrator.RetrieverService') as mock_retriever:
            mock_retriever_instance = Mock()
            mock_retriever_instance.retrieve.return_value = [
                {'text': 'test', 'score': 0.8, 'metadata': {}}
            ]
            mock_retriever.return_value = mock_retriever_instance

            orchestrator = RAGOrchestrator(
                config_path=temp_config_file,
                auto_load=False
            )
            # Manually set retriever for test
            orchestrator._retriever = mock_retriever_instance
            orchestrator._initialized = True

            results = orchestrator.search_documents("test query", top_k=5)

            assert len(results) == 1
            mock_retriever_instance.retrieve.assert_called_once_with(
                "test query", top_k=5, search_type="semantic"
            )
