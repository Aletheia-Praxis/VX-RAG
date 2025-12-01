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
        assert orchestrator._query_engine is None

    @patch('src.rag.orchestrator.Settings')
    def test_query_hybrid_search(self, mock_settings, temp_config_file, mock_index):
        """Test query method with QueryEngine."""
        from pathlib import Path
        
        # Setup mocks
        mock_response = Mock()
        mock_node = Mock()
        mock_node.text = 'test document'
        mock_node.score = 0.8
        mock_node.metadata = {'source': 'test'}
        mock_node.node_id = 'node_1'
        mock_node.id_ = 'node_1'
        mock_response.source_nodes = [mock_node]

        # Create orchestrator without auto_load first
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir="data/index",
            auto_load=False
        )
        
        # Manually set up the orchestrator state for testing
        orchestrator._query_engine = Mock()
        orchestrator._query_engine.query.return_value = mock_response
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
        assert 'Query executed via native LlamaIndex QueryEngine' in result['retrieval_stats']['note']

        # Verify query engine was called
        orchestrator._query_engine.query.assert_called_once_with("test query")

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
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir="data/index",
            auto_load=False  # Don't auto-load to avoid initialization issues
        )

        # Manually set state to simulate no indexes loaded
        orchestrator._initialized = True
        orchestrator._indexes_loaded = False
        orchestrator._query_engine = None

        with pytest.raises(RuntimeError, match="Indexes not loaded"):
            orchestrator.query("test query")

    def test_search_documents(self, temp_config_file, mock_index):
        """Test document search method."""
        # Mock QueryEngine response
        mock_response = Mock()
        mock_node = Mock()
        mock_node.text = 'test'
        mock_node.score = 0.8
        mock_node.metadata = {}
        mock_node.node_id = 'node_1'
        mock_node.id_ = 'node_1'
        mock_response.source_nodes = [mock_node]

        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            auto_load=False
        )
        # Manually set query engine for test
        orchestrator._query_engine = Mock()
        orchestrator._query_engine.query.return_value = mock_response
        orchestrator._initialized = True

        results = orchestrator.search_documents("test query", top_k=5)

        assert len(results) == 1
        orchestrator._query_engine.query.assert_called_once_with("test query")

    def test_initialize_registers_global_token_counter(self, temp_config_file, monkeypatch):
        """RAG orchestrator should register a global TokenCountingHandler on init"""
        from src.rag.libs.utils.llamaindex_integration import get_global_token_counter
        from unittest.mock import patch

        # Ensure llama-index defaults to a MockLLM in tests to avoid OpenAI key dependency
        monkeypatch.setenv("IS_TESTING", "1")

        # Patch external dependencies to avoid heavy initializations
        with (
            patch('src.utils.config_loader.get_embedding_config', return_value={
                'embedding_model': 'all-MiniLM-L6-v2',
                'embedding_batch_size': 10,
                'embedding_trust_remote_code': False
            }),
            patch('src.utils.config_loader.get_context_assembler_config', return_value={'model_name': 'gpt-3.5-turbo'}),
            patch('llama_index.vector_stores.faiss.FaissVectorStore.from_persist_dir', side_effect=ValueError('not found')),
            patch('llama_index.core.indices.vector_store.base.VectorStoreIndex.from_vector_store', return_value=Mock()),
            patch('llama_index.core.get_response_synthesizer', return_value=Mock()),
            patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding', return_value=Mock())
        ):
            orchestrator = RAGOrchestrator(config_path=temp_config_file, auto_load=True)

        # Ensure global token counter now exists
        handler = get_global_token_counter()
        assert handler is not None

    def test_initialize_does_not_duplicate_handler(self, temp_config_file, monkeypatch):
        """Ensure orchestrator initialization does not create duplicate TokenCountingHandlers."""
        from src.rag.libs.utils.llamaindex_integration import ensure_global_token_counter
        from llama_index.core import Settings
        # Ensure testing LLM
        monkeypatch.setenv("IS_TESTING", "1")
        # Create a global handler first
        handler_before = ensure_global_token_counter(model_name="gpt-3.5-turbo", verbose=False)
        initial_count = len(Settings.callback_manager.handlers)

        # Initialize orchestrator (patched dependencies)
        with (
            patch('src.utils.config_loader.get_embedding_config', return_value={
                'embedding_model': 'all-MiniLM-L6-v2',
                'embedding_batch_size': 10,
                'embedding_trust_remote_code': False
            }),
            patch('src.utils.config_loader.get_context_assembler_config', return_value={'model_name': 'gpt-3.5-turbo'}),
            patch('llama_index.vector_stores.faiss.FaissVectorStore.from_persist_dir', side_effect=ValueError('not found')),
            patch('llama_index.core.indices.vector_store.base.VectorStoreIndex.from_vector_store', return_value=Mock()),
            patch('llama_index.core.get_response_synthesizer', return_value=Mock()),
            patch('llama_index.embeddings.huggingface.HuggingFaceEmbedding', return_value=Mock())
        ):
            orchestrator = RAGOrchestrator(config_path=temp_config_file, auto_load=True)

        # Ensure count hasn't increased
        assert len(Settings.callback_manager.handlers) == initial_count
