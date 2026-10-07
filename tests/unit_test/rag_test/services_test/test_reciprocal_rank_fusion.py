"""
Unit test suite for Qdrant Native RRF Hybrid Retrieval and CLI search type support.

Tests cover:
- Native Qdrant Fusion.RRF hybrid prefetch construction (dense and sparse).
- Native Qdrant query() and aquery() executing models.FusionQuery(fusion=models.Fusion.RRF).
- Orchestrator search_documents and query execution across hybrid, semantic, and keyword modes.
- CLI argument parsing for --search-type (hybrid, semantic, keyword, invalid choices).
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
)
from qdrant_client import models

from src.cli import main
from src.rag.libs.schemas.mcp_schemas import MCPContextPayload
from src.rag.orchestrator import (
    RAGOrchestrator,
    RobustQdrantVectorStore,
    reset_orchestrator,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.monkeypatch import MonkeyPatch


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Generator[str, None, None]:
    """Create a temporary configuration file for orchestrator tests.

    Args:
        tmp_path: Temporary path fixture.

    Yields:
        Path string to the created temporary YAML config file.
    """
    import yaml

    config_data: dict[str, Any] = {
        "retriever": {
            "semantic_top_k": 15,
            "enable_hybrid": True,
        },
        "embedder": {
            "embedding_model": "BAAI/bge-m3",
            "embedding_batch_size": 32,
            "embedding_trust_remote_code": False,
            "embedding_device": "cpu",
        },
        "embedding_device": "cpu",
        "vector_store": "qdrant",
        "qdrant": {
            "collection_name": "test_rrf",
            "path": str(tmp_path / "qdrant"),
            "distance": "Cosine",
        },
        "bm25": {
            "index_dir": str(tmp_path / "bm25"),
            "similarity_top_k": 20,
        },
        "reranker": {
            "enable_metadata_prioritization": True,
            "model_name": "BAAI/bge-reranker-v2-m3",
            "top_k": 5,
            "device": "cpu",
            "metadata_boost": 0.1,
        },
        "duplicate_detection": {
            "similarity_threshold": 0.95,
            "hash_algorithm": "sha256",
        },
    }

    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.dump(config_data), encoding="utf-8")
    yield str(config_path)


class TestQdrantNativeRRFRetrieval:
    """Test suite for native Qdrant engine-level Fusion.RRF hybrid retrieval."""

    def test_build_hybrid_rrf_prefetch(self) -> None:
        """Verify _build_hybrid_rrf_prefetch constructs dense and sparse Prefetch objects."""
        sparse_encoder = MagicMock(return_value=([[1, 2]], [[0.5, 0.8]]))
        vector_store = RobustQdrantVectorStore(
            collection_name="test_collection",
            client=MagicMock(),
            enable_hybrid=True,
            sparse_query_fn=sparse_encoder,
            sparse_doc_fn=sparse_encoder,
            sparse_vector_name="sparse",
        )
        query = VectorStoreQuery(
            query_str="test hybrid query",
            query_embedding=[0.1] * 1024,
            mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=5,
            sparse_top_k=5,
        )
        prefetch, top_k, _query_filter, _shard_key, _search_params = (
            vector_store._build_hybrid_rrf_prefetch(query)
        )
        assert len(prefetch) == 2
        assert prefetch[0].query == [0.1] * 1024
        assert prefetch[1].using == "sparse"
        assert top_k == 5

    def test_hybrid_query_executes_fusion_rrf(self) -> None:
        """Verify query() calls client.query_points with models.FusionQuery(fusion=models.Fusion.RRF)."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_client.query_points.return_value = mock_response

        sparse_encoder = MagicMock(return_value=([[1, 2]], [[0.5, 0.8]]))
        vector_store = RobustQdrantVectorStore(
            collection_name="test_collection",
            client=mock_client,
            enable_hybrid=True,
            sparse_query_fn=sparse_encoder,
            sparse_doc_fn=sparse_encoder,
            sparse_vector_name="sparse",
        )
        query = VectorStoreQuery(
            query_str="test query",
            query_embedding=[0.2] * 1024,
            mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=3,
        )
        res = vector_store.query(query)
        assert res.nodes == []
        mock_client.query_points.assert_called_once()
        call_kwargs = mock_client.query_points.call_args[1]
        assert isinstance(call_kwargs["query"], models.FusionQuery)
        assert call_kwargs["query"].fusion == models.Fusion.RRF

    @pytest.mark.asyncio
    async def test_hybrid_aquery_executes_fusion_rrf(self) -> None:
        """Verify aquery() calls aclient.query_points with models.FusionQuery(fusion=models.Fusion.RRF)."""
        mock_aclient = MagicMock()
        mock_response = MagicMock()
        mock_response.points = []
        mock_aclient.query_points = AsyncMock(return_value=mock_response)

        sparse_encoder = MagicMock(return_value=([[1, 2]], [[0.5, 0.8]]))
        vector_store = RobustQdrantVectorStore(
            collection_name="test_collection",
            client=MagicMock(),
            enable_hybrid=True,
            sparse_query_fn=sparse_encoder,
            sparse_doc_fn=sparse_encoder,
            sparse_vector_name="sparse",
        )
        vector_store._aclient = mock_aclient
        query = VectorStoreQuery(
            query_str="async query",
            query_embedding=[0.3] * 1024,
            mode=VectorStoreQueryMode.HYBRID,
            similarity_top_k=4,
        )
        res = await vector_store.aquery(query)
        assert res.nodes == []
        mock_aclient.query_points.assert_awaited_once()
        call_kwargs = mock_aclient.query_points.call_args[1]
        assert isinstance(call_kwargs["query"], models.FusionQuery)
        assert call_kwargs["query"].fusion == models.Fusion.RRF


class TestCLISearchTypeArgument:
    """Test suite for CLI --search-type argument parsing and dispatch."""

    def test_cli_query_help_contains_search_type(self, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
        """Verify that CLI query --help documents --search-type option."""
        monkeypatch.setattr("sys.argv", ["cli", "query", "--help"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--search-type" in captured.out
        assert "hybrid" in captured.out
        assert "semantic" in captured.out
        assert "keyword" in captured.out

    def test_cli_query_dispatches_with_search_type(self, monkeypatch: MonkeyPatch) -> None:
        """Verify that handle_query passes args.search_type to orchestrator.query."""
        mock_orchestrator = MagicMock()
        mock_payload = MCPContextPayload(
            schema_version="1.0",
            context=[],
            query="test query",
            token_budget=4000,
            provenance={"selected_count": 0, "total_tokens": 0, "selection_method": "query_engine"},
        )
        mock_orchestrator.query.return_value = mock_payload

        with patch("src.cli.get_orchestrator", return_value=mock_orchestrator):
            for mode in ["hybrid", "semantic", "keyword"]:
                monkeypatch.setattr("sys.argv", ["cli", "query", "cybersecurity", "--search-type", mode])
                main()
                mock_orchestrator.query.assert_called_with(
                    query="cybersecurity",
                    top_k=5,
                    search_type=mode,
                )

    def test_cli_query_defaults_to_hybrid(self, monkeypatch: MonkeyPatch) -> None:
        """Verify that CLI query defaults to search_type='hybrid' when flag is omitted."""
        mock_orchestrator = MagicMock()
        mock_payload = MCPContextPayload(
            schema_version="1.0",
            context=[],
            query="test query",
            token_budget=4000,
            provenance={"selected_count": 0, "total_tokens": 0, "selection_method": "query_engine"},
        )
        mock_orchestrator.query.return_value = mock_payload

        with patch("src.cli.get_orchestrator", return_value=mock_orchestrator):
            monkeypatch.setattr("sys.argv", ["cli", "query", "cybersecurity"])
            main()
            mock_orchestrator.query.assert_called_with(
                query="cybersecurity",
                top_k=5,
                search_type="hybrid",
            )

    def test_cli_query_invalid_search_type_rejected(self, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]) -> None:
        """Verify that invalid --search-type values are rejected by argparse."""
        monkeypatch.setattr("sys.argv", ["cli", "query", "cybersecurity", "--search-type", "unsupported_modality"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0
        captured = capsys.readouterr()
        assert "invalid choice" in captured.err


class TestOrchestratorModalitiesWithoutOpenAIKey:
    """Test suite ensuring orchestrator query and search_documents succeed without OPENAI_API_KEY."""

    @patch("llama_index.core.query_engine.RetrieverQueryEngine.aquery")
    def test_orchestrator_query_all_search_types(
        self,
        mock_aquery: MagicMock,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify orchestrator.query works for hybrid, semantic, and keyword modes without external LLM."""
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(tmp_path / "persist"),
            auto_load=False,
        )
        orchestrator._initialized = True
        orchestrator._indexes_loaded = True

        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        orchestrator._index = mock_index

        mock_bm25 = MagicMock()
        orchestrator._bm25_retriever = mock_bm25

        node = TextNode(text="Query hit content", id_="hit_1", metadata={"file_name": "report.pdf"})
        mock_response = MagicMock()
        mock_response.source_nodes = [NodeWithScore(node=node, score=0.88)]

        async def mock_async_aquery(*args: Any, **kwargs: Any) -> Any:
            return mock_response

        mock_aquery.side_effect = mock_async_aquery

        for mode in ["hybrid", "semantic", "keyword"]:
            payload = orchestrator.query("APT attack vector", top_k=2, search_type=mode)
            assert isinstance(payload, MCPContextPayload)
            assert payload.query == "APT attack vector"
            assert len(payload.context) == 1
            assert payload.context[0].id == "hit_1"

    def test_orchestrator_search_documents_all_search_types(
        self,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify orchestrator.search_documents operates across all three modalities."""
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(tmp_path / "persist"),
            auto_load=False,
        )
        orchestrator._initialized = True

        mock_index = MagicMock()
        mock_vec_retriever = MagicMock()
        node_1 = TextNode(text="Vector candidate", id_="v1", metadata={"file_name": "f1.txt"})
        mock_vec_retriever.retrieve.return_value = [NodeWithScore(node=node_1, score=0.9)]
        mock_index.as_retriever.return_value = mock_vec_retriever
        orchestrator._index = mock_index

        mock_bm25 = MagicMock()
        node_2 = TextNode(text="BM25 candidate", id_="b1", metadata={"file_name": "f2.txt"})
        mock_bm25.retrieve.return_value = [NodeWithScore(node=node_2, score=11.5)]
        orchestrator._bm25_retriever = mock_bm25

        for mode in ["hybrid", "semantic", "keyword"]:
            results = orchestrator.search_documents("lateral movement", top_k=5, search_type=mode)
            assert isinstance(results, list)
            assert len(results) > 0

    def test_reset_orchestrator_multiple_calls_safe(self) -> None:
        """Verify reset_orchestrator can be safely called multiple times sequentially."""
        reset_orchestrator()
        reset_orchestrator()

    @pytest.mark.asyncio
    @patch("llama_index.core.query_engine.RetrieverQueryEngine.aquery")
    async def test_mcp_query_knowledge_base_hybrid_without_openai(
        self,
        mock_aquery: MagicMock,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify MCP query_knowledge_base functions in hybrid search modality."""
        from src.mcp.server import query_knowledge_base

        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(tmp_path / "persist"),
            auto_load=False,
        )
        orchestrator._initialized = True
        orchestrator._indexes_loaded = True

        mock_index = MagicMock()
        mock_retriever = MagicMock()
        mock_index.as_retriever.return_value = mock_retriever
        orchestrator._index = mock_index

        mock_bm25 = MagicMock()
        orchestrator._bm25_retriever = mock_bm25

        node = TextNode(text="MCP knowledge result", id_="mcp_hit", metadata={"file_name": "intel.pdf"})
        mock_response = MagicMock()
        mock_response.source_nodes = [NodeWithScore(node=node, score=0.92)]

        async def mock_async_aquery(*args: Any, **kwargs: Any) -> Any:
            return mock_response

        mock_aquery.side_effect = mock_async_aquery

        with patch("src.mcp.server.get_orchestrator", return_value=orchestrator):
            response_json = await query_knowledge_base(query="ransomware attack vector", top_k=3, search_type="hybrid")
            assert "mcp_hit" in response_json or "MCP knowledge result" in response_json
