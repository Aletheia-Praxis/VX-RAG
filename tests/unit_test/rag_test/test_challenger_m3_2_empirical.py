"""Empirical adversarial validation suite for Milestone M3 (Reranker & Duplicate Detection).

Authored by challenger_m3_2 to empirically stress-test:
1. BGECrossEncoderReranker semantic reordering, sigmoid normalization in [0.0, 1.0],
   top_k candidate list truncation, node integrity preservation, and error fallbacks.
2. Adversarial edge cases: extreme scores (-1e6 to 1e6), empty lists, None/empty query bundles,
   identical candidates, and inference engine exceptions.
3. Ingest and Index duplicate detection: identical files in batch, already-indexed SHA-256 hashes,
   duplicate node IDs, case-insensitivity, and sequential idempotence.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core import QueryBundle
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode

from src.rag.libs.postprocessors import BGECrossEncoderReranker
from src.rag.orchestrator import RAGOrchestrator

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401


# Named constants to avoid magic values
DEFAULT_TOP_N: int = 5
SCORE_LOWER_BOUND: float = 0.0
SCORE_UPPER_BOUND: float = 1.0
SIGMOID_CLAMP_UPPER: float = 50.0
SIGMOID_CLAMP_LOWER: float = -50.0


@pytest.fixture
def mock_cross_encoder_model() -> MagicMock:
    """Create a mock CrossEncoder model for injection.

    Returns:
        MagicMock conforming to sentence_transformers.CrossEncoder interface.
    """
    mock = MagicMock()
    mock.predict.return_value = [0.0]
    return mock


@pytest.fixture
def temp_orchestrator_config(tmp_path: Path) -> Generator[str, None, None]:
    """Create a temporary configuration file for orchestrator tests.

    Args:
        tmp_path: Pytest temporary directory fixture.

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


class TestBGECrossEncoderRerankerEmpirical:
    """Empirical challenge suite for BGECrossEncoderReranker."""

    def test_semantic_reordering_descending_rank(self) -> None:
        """Verify candidate nodes are strictly sorted descending by normalized relevance score."""
        mock_model = MagicMock()
        # Raw logits for 5 nodes: -2.5, 4.2, 0.0, 1.8, -0.5
        # Expected rank order: Node 1 (4.2), Node 3 (1.8), Node 2 (0.0), Node 4 (-0.5), Node 0 (-2.5)
        raw_logits = [-2.5, 4.2, 0.0, 1.8, -0.5]
        mock_model.predict.return_value = raw_logits

        reranker = BGECrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            top_n=5,
            device="cpu",
            model=mock_model,
        )

        nodes = [
            NodeWithScore(node=TextNode(text=f"Passage {i}", id_=f"node_{i}"), score=0.5)
            for i in range(5)
        ]
        query = QueryBundle("Relevant cybersecurity query")
        reranked = reranker.postprocess_nodes(nodes, query)

        assert len(reranked) == 5
        assert reranked[0].node.id_ == "node_1"
        assert reranked[1].node.id_ == "node_3"
        assert reranked[2].node.id_ == "node_2"
        assert reranked[3].node.id_ == "node_4"
        assert reranked[4].node.id_ == "node_0"

        # Verify strict monotonicity of scores in output
        for idx in range(len(reranked) - 1):
            assert (reranked[idx].score or 0.0) >= (reranked[idx + 1].score or 0.0)

    @pytest.mark.parametrize(
        "raw_score, expected_normalized",
        [
            (-1_000_000.0, 0.0),
            (-1000.0, 0.0),
            (-100.0, 0.0),
            (-50.1, 0.0),
            (-50.0, 1.0 / (1.0 + math.exp(50.0))),
            (-10.0, 1.0 / (1.0 + math.exp(10.0))),
            (0.0, 0.5),
            (10.0, 1.0 / (1.0 + math.exp(-10.0))),
            (50.0, 1.0 / (1.0 + math.exp(-50.0))),
            (50.1, 1.0),
            (100.0, 1.0),
            (1000.0, 1.0),
            (1_000_000.0, 1.0),
        ],
    )
    def test_sigmoid_normalization_strict_bounds(
        self,
        raw_score: float,
        expected_normalized: float,
    ) -> None:
        """Verify sigmoid normalization bounds all scores strictly within [0.0, 1.0] without overflow."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [raw_score]

        reranker = BGECrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            top_n=5,
            device="cpu",
            model=mock_model,
        )

        candidate = NodeWithScore(node=TextNode(text="Test passage", id_="test_1"), score=0.0)
        reranked = reranker.postprocess_nodes([candidate], QueryBundle("Test query"))

        assert len(reranked) == 1
        score = reranked[0].score
        assert score is not None
        assert SCORE_LOWER_BOUND <= score <= SCORE_UPPER_BOUND
        assert pytest.approx(score, abs=1e-6) == expected_normalized

    @pytest.mark.parametrize("top_n", [1, 2, 5, 8, 15])
    def test_candidate_list_truncation_top_k(self, top_n: int) -> None:
        """Verify candidate list is truncated to exactly min(len(candidates), top_n)."""
        candidate_count = 10
        mock_model = MagicMock()
        mock_model.predict.return_value = [float(i) for i in range(candidate_count)]

        reranker = BGECrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            top_n=top_n,
            device="cpu",
            model=mock_model,
        )

        candidates = [
            NodeWithScore(node=TextNode(text=f"Passage {i}", id_=f"n_{i}"), score=0.1 * i)
            for i in range(candidate_count)
        ]
        result = reranker.postprocess_nodes(candidates, QueryBundle("Query"))

        expected_length = min(candidate_count, top_n)
        assert len(result) == expected_length

    def test_node_integrity_preservation(self) -> None:
        """Verify node ID, text content, and all strict 5 metadata fields are preserved intact."""
        metadata = {
            "file_name": "mimikatz_dump.txt",
            "file_type": "txt",
            "creation_date": "2026-03-01T12:00:00Z",
            "ingestion_date": "2026-03-01T12:05:00Z",
            "file_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        }
        text_content = "sekurlsa::logonpasswords credential extraction technique"
        node_id = "target_node_999"

        original_text_node = TextNode(text=text_content, id_=node_id, metadata=metadata)
        scored_node = NodeWithScore(node=original_text_node, score=0.25)

        mock_model = MagicMock()
        mock_model.predict.return_value = [3.5]

        reranker = BGECrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            top_n=5,
            device="cpu",
            model=mock_model,
        )

        reranked = reranker.postprocess_nodes([scored_node], QueryBundle("lsass dump"))

        assert len(reranked) == 1
        output_node = reranked[0]
        assert output_node.node.id_ == node_id
        assert output_node.node.get_content() == text_content
        assert output_node.node.metadata == metadata
        assert output_node.score != 0.25
        assert SCORE_LOWER_BOUND <= (output_node.score or 0.0) <= SCORE_UPPER_BOUND

    def test_adversarial_empty_candidate_list(self) -> None:
        """Verify empty candidate list returns empty list without calling model."""
        mock_model = MagicMock()
        reranker = BGECrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            top_n=5,
            device="cpu",
            model=mock_model,
        )

        result = reranker.postprocess_nodes([], QueryBundle("Non-empty query"))
        assert result == []
        mock_model.predict.assert_not_called()

    def test_adversarial_empty_and_none_query_bundle(self) -> None:
        """Verify None or empty query string returns candidates truncated to top_n without reranking."""
        mock_model = MagicMock()
        reranker = BGECrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            top_n=2,
            device="cpu",
            model=mock_model,
        )

        candidates = [
            NodeWithScore(node=TextNode(text=f"Text {i}", id_=f"id_{i}"), score=0.1 * i)
            for i in range(4)
        ]

        # Case A: query_bundle is None
        result_none = reranker.postprocess_nodes(candidates, None)
        assert len(result_none) == 2
        assert result_none[0].node.id_ == "id_0"
        assert result_none[1].node.id_ == "id_1"

        # Case B: query_str is empty string
        result_empty = reranker.postprocess_nodes(candidates, QueryBundle(""))
        assert len(result_empty) == 2
        assert result_empty[0].node.id_ == "id_0"
        assert result_empty[1].node.id_ == "id_1"

        mock_model.predict.assert_not_called()

    @pytest.mark.parametrize(
        "exception_type, exception_msg",
        [
            (RuntimeError, "CUDA out of memory error during cross-encoder forward pass"),
            (ValueError, "Input sentence pairs contain invalid encoding"),
            (Exception, "Unexpected compute runtime error"),
        ],
    )
    def test_error_fallback_when_cross_encoder_fails(
        self,
        exception_type: type[Exception],
        exception_msg: str,
    ) -> None:
        """Verify system gracefully falls back to initial retrieval candidates when CrossEncoder fails."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = exception_type(exception_msg)

        reranker = BGECrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            top_n=2,
            device="cpu",
            model=mock_model,
        )

        candidates = [
            NodeWithScore(node=TextNode(text="First candidate", id_="first"), score=0.9),
            NodeWithScore(node=TextNode(text="Second candidate", id_="second"), score=0.8),
            NodeWithScore(node=TextNode(text="Third candidate", id_="third"), score=0.7),
        ]

        result = reranker.postprocess_nodes(candidates, QueryBundle("Crash trigger query"))

        # Invariant: Must truncate to top_n and preserve original candidate objects and scores
        assert len(result) == 2
        assert result[0].node.id_ == "first"
        assert result[0].score == 0.9
        assert result[1].node.id_ == "second"
        assert result[1].score == 0.8


class TestDuplicateDetectionEmpirical:
    """Empirical challenge suite for duplicate detection in RAGOrchestrator."""

    @patch("src.rag.orchestrator.DoclingPipeline")
    def test_ingest_documents_skips_identical_files_in_same_batch(
        self,
        mock_pipeline_cls: MagicMock,
        temp_orchestrator_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify ingest_documents skips exact duplicate files in the source directory within the same session."""
        docs_dir = tmp_path / "batch_docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Create two files with IDENTICAL content (different names)
        identical_content = "Exact duplicate malware payload description and hashes."
        file_primary = docs_dir / "report_original.txt"
        file_primary.write_text(identical_content, encoding="utf-8")
        file_duplicate = docs_dir / "report_copy.txt"
        file_duplicate.write_text(identical_content, encoding="utf-8")

        expected_hash = hashlib.sha256(identical_content.encode("utf-8")).hexdigest()

        mock_pipeline_instance = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_instance

        mock_node = TextNode(
            text=identical_content,
            metadata={
                "file_name": "report_original.txt",
                "file_type": "txt",
                "creation_date": "2026-01-01T00:00:00Z",
                "ingestion_date": "2026-01-01T00:00:00Z",
                "file_hash": expected_hash,
            },
        )
        mock_pipeline_instance.ingest_file.return_value = [mock_node]

        orchestrator = RAGOrchestrator(
            config_path=temp_orchestrator_config,
            persist_dir=str(tmp_path / "index"),
            auto_load=False,
        )

        nodes = orchestrator.ingest_documents(docs_dir)

        # Invariant: Exactly one file must be parsed; the duplicate copy MUST be skipped
        assert mock_pipeline_instance.ingest_file.call_count == 1
        assert len(nodes) == 1
        assert nodes[0].metadata["file_hash"] == expected_hash

    @patch("src.rag.orchestrator.DoclingPipeline")
    def test_ingest_documents_skips_already_indexed_file_hashes(
        self,
        mock_pipeline_cls: MagicMock,
        temp_orchestrator_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify ingest_documents skips files whose SHA-256 hashes are already indexed in FAISS/manifest."""
        docs_dir = tmp_path / "mixed_docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        existing_content = "Already indexed content from previous run."
        file_existing = docs_dir / "existing.txt"
        file_existing.write_text(existing_content, encoding="utf-8")
        existing_hash = hashlib.sha256(existing_content.encode("utf-8")).hexdigest()

        new_content = "Brand new content not yet indexed."
        file_new = docs_dir / "new_doc.txt"
        file_new.write_text(new_content, encoding="utf-8")
        new_hash = hashlib.sha256(new_content.encode("utf-8")).hexdigest()

        mock_pipeline_instance = MagicMock()
        mock_pipeline_cls.return_value = mock_pipeline_instance

        new_node = TextNode(
            text=new_content,
            metadata={
                "file_name": "new_doc.txt",
                "file_type": "txt",
                "creation_date": "2026-01-01T00:00:00Z",
                "ingestion_date": "2026-01-01T00:00:00Z",
                "file_hash": new_hash,
            },
        )
        mock_pipeline_instance.ingest_file.return_value = [new_node]

        orchestrator = RAGOrchestrator(
            config_path=temp_orchestrator_config,
            persist_dir=str(tmp_path / "index"),
            auto_load=False,
        )
        # Pre-seed existing_hash in uppercase to test case-insensitive matching
        orchestrator.get_indexed_file_hashes = MagicMock(  # type: ignore[method-assign]
            return_value={existing_hash.upper()}
        )

        nodes = orchestrator.ingest_documents(docs_dir)

        # Invariant: existing file skipped; only new file parsed
        mock_pipeline_instance.ingest_file.assert_called_once_with(file_new)
        assert len(nodes) == 1
        assert nodes[0].metadata["file_hash"] == new_hash

    def test_index_nodes_skips_duplicate_hashes_and_existing_node_ids(
        self,
        temp_orchestrator_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify index_nodes filters out nodes matching already-indexed file hashes or node IDs."""
        orchestrator = RAGOrchestrator(
            config_path=temp_orchestrator_config,
            persist_dir=str(tmp_path / "index"),
            auto_load=False,
        )
        orchestrator._initialized = True

        mock_index = MagicMock()
        mock_storage = MagicMock()
        mock_index.storage_context = mock_storage
        mock_index.docstore.docs = {
            "existing_node_id": TextNode(text="Old", id_="existing_node_id")
        }
        orchestrator._index = mock_index

        indexed_hash = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        orchestrator.get_indexed_file_hashes = MagicMock(return_value={indexed_hash})  # type: ignore[method-assign]

        duplicate_hash_node = TextNode(
            text="Duplicate file hash",
            id_="new_id_1",
            metadata={"file_hash": indexed_hash},
        )
        duplicate_id_node = TextNode(
            text="Duplicate node ID",
            id_="existing_node_id",
            metadata={"file_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"},
        )
        fresh_node = TextNode(
            text="Fresh node",
            id_="fresh_id",
            metadata={"file_hash": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"},
        )

        orchestrator._bm25_retriever = MagicMock()
        orchestrator._generate_manifest = MagicMock(return_value={"version": "1.0"})  # type: ignore[method-assign]

        with patch("src.rag.orchestrator.BM25Retriever") as mock_bm25:
            mock_bm25.from_defaults.return_value = MagicMock()
            orchestrator.index_nodes([duplicate_hash_node, duplicate_id_node, fresh_node])

        # Invariant: Only fresh_node must be inserted into index
        mock_index.insert_nodes.assert_called_once()
        inserted_list: list[BaseNode] = mock_index.insert_nodes.call_args[0][0]
        assert len(inserted_list) == 1
        assert inserted_list[0].node_id == "fresh_id"

    def test_index_nodes_idempotence_on_complete_duplicate_batch(
        self,
        temp_orchestrator_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify calling index_nodes when all nodes are duplicates returns early without modifying storage."""
        orchestrator = RAGOrchestrator(
            config_path=temp_orchestrator_config,
            persist_dir=str(tmp_path / "index"),
            auto_load=False,
        )
        orchestrator._initialized = True

        mock_index = MagicMock()
        mock_index.docstore.docs = {"id_x": TextNode(text="Ex", id_="id_x")}
        orchestrator._index = mock_index

        duplicate_hash = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
        orchestrator.get_indexed_file_hashes = MagicMock(return_value={duplicate_hash})  # type: ignore[method-assign]

        duplicate_nodes = [
            TextNode(text="Duplicate", id_="new_1", metadata={"file_hash": duplicate_hash}),
            TextNode(text="Duplicate 2", id_="id_x", metadata={"file_hash": "different"}),
        ]

        orchestrator.index_nodes(duplicate_nodes)

        # Invariant: insert_nodes must NOT be called when all inputs are duplicates
        mock_index.insert_nodes.assert_not_called()
        mock_index.storage_context.persist.assert_not_called()

    def test_index_nodes_empty_list_no_op(
        self,
        temp_orchestrator_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify calling index_nodes with an empty list does nothing and returns cleanly."""
        orchestrator = RAGOrchestrator(
            config_path=temp_orchestrator_config,
            persist_dir=str(tmp_path / "index"),
            auto_load=False,
        )
        orchestrator._initialized = True
        mock_index = MagicMock()
        orchestrator._index = mock_index

        orchestrator.index_nodes([])
        mock_index.insert_nodes.assert_not_called()

