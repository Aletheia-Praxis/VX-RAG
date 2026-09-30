"""
Unit test suite for Standalone Reciprocal Rank Fusion (RRF) and CLI search type support.

Tests cover:
- RRF score formula correctness (1 / (k + rank)) with default k=60 and 1-based ranks.
- Merging of vector and BM25 candidate streams, including dual-match score accumulation.
- Boundary conditions: empty streams, disjoint candidates, deduplication within a run.
- Truncation to top_k.
- ReciprocalRankFusionRetriever synchronous and asynchronous retrieve execution.
- RAGOrchestrator.reciprocal_rank_fusion static method parity.
- Orchestrator search_documents and query execution across hybrid, semantic, and keyword modes.
- CLI argument parsing for --search-type (hybrid, semantic, keyword, invalid choices).
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from llama_index.core import QueryBundle
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode

from src.cli import main
from src.rag.libs.schemas.mcp_schemas import MCPContextPayload
from src.rag.orchestrator import (
    DEFAULT_RRF_K,
    RAGOrchestrator,
    ReciprocalRankFusionRetriever,
    reciprocal_rank_fusion,
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
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_batch_size": 32,
            "embedding_trust_remote_code": False,
            "embedding_device": "cpu",
        },
        "embedding_device": "cpu",
        "faiss": {
            "hnsw_m": 32,
            "metric": "inner_product",
        },
        "bm25": {
            "index_dir": str(tmp_path / "bm25"),
            "similarity_top_k": 20,
        },
        "reranker": {
            "enable_metadata_prioritization": True,
            "model_name": "BAAI/bge-reranker-base",
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


class TestReciprocalRankFusionFormula:
    """Test suite for the pure reciprocal_rank_fusion algorithm."""

    def test_rrf_single_stream_ranking(self) -> None:
        """Verify RRF calculates scores as 1.0 / (k + rank) with rank 1..N."""
        node_1 = TextNode(text="first doc", id_="node_1")
        node_2 = TextNode(text="second doc", id_="node_2")
        stream = [
            NodeWithScore(node=node_1, score=0.9),
            NodeWithScore(node=node_2, score=0.5),
        ]

        fused = reciprocal_rank_fusion([stream], k=60)

        assert len(fused) == 2
        assert fused[0].node.node_id == "node_1"
        assert fused[1].node.node_id == "node_2"
        # rank 1: 1 / (60 + 1) = 1/61
        assert fused[0].score == pytest.approx(1.0 / 61.0)
        # rank 2: 1 / (60 + 2) = 1/62
        assert fused[1].score == pytest.approx(1.0 / 62.0)

    def test_rrf_dual_stream_overlap_accumulates_scores(self) -> None:
        """Verify candidate appearing in both streams receives sum of reciprocal rank scores."""
        node_common = TextNode(text="appears in both", id_="common_node")
        node_vec_only = TextNode(text="only vector", id_="vec_node")
        node_bm25_only = TextNode(text="only bm25", id_="bm25_node")

        # Vector stream: common is rank 1, vec_only is rank 2
        vector_stream = [
            NodeWithScore(node=node_common, score=0.95),
            NodeWithScore(node=node_vec_only, score=0.80),
        ]
        # BM25 stream: common is rank 1, bm25_only is rank 2
        bm25_stream = [
            NodeWithScore(node=node_common, score=15.2),
            NodeWithScore(node=node_bm25_only, score=12.1),
        ]

        fused = reciprocal_rank_fusion([vector_stream, bm25_stream], k=60)

        assert len(fused) == 3
        # Common node must rank first with accumulated score: 1/61 + 1/61
        assert fused[0].node.node_id == "common_node"
        expected_common_score = (1.0 / 61.0) + (1.0 / 61.0)
        assert fused[0].score == pytest.approx(expected_common_score)

        # Single stream nodes each have 1/62
        assert fused[1].score == pytest.approx(1.0 / 62.0)
        assert fused[2].score == pytest.approx(1.0 / 62.0)

    def test_rrf_custom_k_parameter(self) -> None:
        """Verify custom smoothing constant k alters the scoring properly."""
        node_a = TextNode(text="doc a", id_="node_a")
        stream = [NodeWithScore(node=node_a, score=1.0)]

        fused = reciprocal_rank_fusion([stream], k=20)
        assert len(fused) == 1
        assert fused[0].score == pytest.approx(1.0 / (20.0 + 1.0))

    def test_rrf_top_k_truncation(self) -> None:
        """Verify top_k parameter truncates returned fused candidates."""
        nodes = [
            NodeWithScore(node=TextNode(text=f"item {i}", id_=f"n_{i}"), score=float(10 - i))
            for i in range(10)
        ]
        fused = reciprocal_rank_fusion([nodes], k=DEFAULT_RRF_K, top_k=3)
        assert len(fused) == 3
        assert [item.node.node_id for item in fused] == ["n_0", "n_1", "n_2"]

    def test_rrf_empty_inputs_handling(self) -> None:
        """Verify empty input streams gracefully return empty list."""
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[], []]) == []

    def test_rrf_duplicate_node_ids_within_single_stream_deduplicated(self) -> None:
        """Verify that duplicate occurrences of the same node_id in one run do not double count."""
        node_dup = TextNode(text="duplicate doc", id_="dup_id")
        stream = [
            NodeWithScore(node=node_dup, score=0.9),
            NodeWithScore(node=node_dup, score=0.7),
        ]
        fused = reciprocal_rank_fusion([stream], k=60)
        assert len(fused) == 1
        assert fused[0].score == pytest.approx(1.0 / 61.0)

    def test_orchestrator_static_method_parity(self) -> None:
        """Verify RAGOrchestrator.reciprocal_rank_fusion forwards identically to standalone function."""
        node = TextNode(text="sample", id_="s1")
        stream = [NodeWithScore(node=node, score=0.5)]

        fused_static = RAGOrchestrator.reciprocal_rank_fusion([stream], k=60, top_k=1)
        fused_func = reciprocal_rank_fusion([stream], k=60, top_k=1)

        assert len(fused_static) == len(fused_func)
        assert fused_static[0].node.node_id == fused_func[0].node.node_id
        assert fused_static[0].score == fused_func[0].score

    def test_rrf_one_stream_empty_other_populated(self) -> None:
        """Verify RRF when one stream is empty and one has candidates."""
        node_1 = TextNode(text="doc 1", id_="node_1")
        stream = [NodeWithScore(node=node_1, score=0.8)]

        # Vector has candidate, BM25 empty
        fused_1 = reciprocal_rank_fusion([stream, []], k=60)
        assert len(fused_1) == 1
        assert fused_1[0].node.node_id == "node_1"
        assert fused_1[0].score == pytest.approx(1.0 / 61.0)

        # BM25 has candidate, Vector empty
        fused_2 = reciprocal_rank_fusion([[], stream], k=60)
        assert len(fused_2) == 1
        assert fused_2[0].node.node_id == "node_1"
        assert fused_2[0].score == pytest.approx(1.0 / 61.0)

    def test_rrf_negative_k_raises_value_error(self) -> None:
        """Verify negative smoothing constant k raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            reciprocal_rank_fusion([], k=-1)

    def test_rrf_negative_top_k_returns_empty_list(self) -> None:
        """Verify negative top_k safely returns empty list without negative slice corruption."""
        node = TextNode(text="doc", id_="n1")
        stream = [NodeWithScore(node=node, score=0.5)]
        assert reciprocal_rank_fusion([stream], top_k=-1) == []
        assert reciprocal_rank_fusion([stream], top_k=0) == []

    def test_rrf_negative_and_none_score_ordering(self) -> None:
        """Verify nodes with explicit scores rank above nodes with None scores."""
        node_neg = TextNode(text="negative score doc", id_="neg_doc")
        node_none = TextNode(text="none score doc", id_="none_doc")
        stream = [
            NodeWithScore(node=node_none, score=None),
            NodeWithScore(node=node_neg, score=-0.5),
        ]
        fused = reciprocal_rank_fusion([stream], k=60)
        assert len(fused) == 2
        assert fused[0].node.node_id == "neg_doc"
        assert fused[1].node.node_id == "none_doc"


class TestReciprocalRankFusionRetriever:
    """Test suite for the ReciprocalRankFusionRetriever BaseRetriever class."""

    def test_sync_retrieve(self) -> None:
        """Verify synchronous _retrieve calls all sub-retrievers and fuses results."""
        retriever_1 = MagicMock(spec=BaseRetriever)
        retriever_2 = MagicMock(spec=BaseRetriever)

        node_a = TextNode(text="Content A", id_="doc_a")
        node_b = TextNode(text="Content B", id_="doc_b")

        retriever_1.retrieve.return_value = [NodeWithScore(node=node_a, score=0.8)]
        retriever_2.retrieve.return_value = [NodeWithScore(node=node_b, score=0.9)]

        rrf_retriever = ReciprocalRankFusionRetriever(
            retrievers=[retriever_1, retriever_2],
            similarity_top_k=5,
            k=60,
        )

        query_bundle = QueryBundle("search query")
        results = rrf_retriever.retrieve(query_bundle)

        assert len(results) == 2
        retriever_1.retrieve.assert_called_once_with(query_bundle)
        retriever_2.retrieve.assert_called_once_with(query_bundle)

    @pytest.mark.asyncio
    async def test_async_retrieve(self) -> None:
        """Verify asynchronous aretrieve calls sub-retrievers in parallel and fuses results."""
        retriever_1 = MagicMock(spec=BaseRetriever)
        retriever_2 = MagicMock(spec=BaseRetriever)

        node_a = TextNode(text="Content A", id_="doc_a")
        node_b = TextNode(text="Content B", id_="doc_b")

        retriever_1.aretrieve = AsyncMock(return_value=[NodeWithScore(node=node_a, score=0.8)])
        retriever_2.aretrieve = AsyncMock(return_value=[NodeWithScore(node=node_b, score=0.9)])

        rrf_retriever = ReciprocalRankFusionRetriever(
            retrievers=[retriever_1, retriever_2],
            similarity_top_k=5,
            k=60,
        )

        query_bundle = QueryBundle("async query")
        results = await rrf_retriever.aretrieve(query_bundle)

        assert len(results) == 2
        retriever_1.aretrieve.assert_awaited_once_with(query_bundle)
        retriever_2.aretrieve.assert_awaited_once_with(query_bundle)


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
