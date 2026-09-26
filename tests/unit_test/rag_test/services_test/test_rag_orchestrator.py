"""
Unit tests for RAGOrchestrator and Phase M3 RAG Core components.

Verifies:
- BGE Embedding configuration (BAAI/bge-small-en-v1.5, 384-dim, CPU, normalize=True)
- BGECrossEncoderReranker (BAAI/bge-reranker-base, sigmoid normalization, fallback on error)
- FAISS HNSW vector store fallback and incremental appends without rebuild
- BM25 synchronization retaining cumulative nodes from docstore
- Ingestion with DoclingPipeline and strict 5-field metadata
- Two-tier duplicate detection skipping indexed file hashes
- Manifest.json generation with SHA-256 checksums and versioned snapshots
- Query execution returning MCPContextPayload and search_documents returning full text/metadata
- Health status and singleton orchestrator lifecycle
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator
from unittest.mock import MagicMock, patch

import faiss
import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import NodeWithScore, TextNode

os.environ["IS_TESTING"] = "1"

from src.rag.libs.postprocessors import BGECrossEncoderReranker
from src.rag.libs.schemas.mcp_schemas import MCPContextPayload
from src.rag.orchestrator import (
    _EMBEDDING_DIMENSION_BY_MODEL,
    _FALLBACK_EMBEDDING_DIMENSION,
    RAGOrchestrator,
    get_orchestrator,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Generator[str, None, None]:
    """Create a temporary configuration file for tests.

    Args:
        tmp_path: Temporary path fixture.

    Yields:
        Path string to the created temporary YAML config file.
    """
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

    import yaml

    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.dump(config_data), encoding="utf-8")
    yield str(config_path)


class TestRAGOrchestratorInitialization:
    """Test suite for orchestrator initialization and model configuration."""

    def test_init_without_auto_load(self, temp_config_file: str, tmp_path: Path) -> None:
        """Verify orchestrator initialization when auto_load is disabled.

        Args:
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(persist_dir),
            auto_load=False,
        )

        assert not orchestrator._initialized
        assert not orchestrator._indexes_loaded
        assert orchestrator._vector_store is None
        assert orchestrator._storage_context is None
        assert orchestrator._index is None
        assert orchestrator._bm25_retriever is None
        assert orchestrator._reranker is None
        assert orchestrator.persist_dir == persist_dir

    def test_bge_small_embedding_dimension_mapping(self) -> None:
        """Verify BAAI/bge-small-en-v1.5 maps to 384 dimensions."""
        dim = _EMBEDDING_DIMENSION_BY_MODEL.get("BAAI/bge-small-en-v1.5", _FALLBACK_EMBEDDING_DIMENSION)
        assert dim == 384

    @patch("src.rag.orchestrator.HuggingFaceEmbedding")
    @patch("src.rag.orchestrator.BGECrossEncoderReranker")
    def test_initialize_services_configures_bge_models(
        self,
        mock_reranker_cls: MagicMock,
        mock_hf_embed: MagicMock,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify that _initialize_services configures BGE embedding and reranker.

        Args:
            mock_reranker_cls: Mocked BGECrossEncoderReranker class.
            mock_hf_embed: Mocked HuggingFaceEmbedding class.
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(persist_dir),
            auto_load=False,
        )
        mock_hf_embed.return_value = MagicMock(spec=BaseEmbedding)
        orchestrator._initialize_services()

        assert orchestrator._initialized
        mock_hf_embed.assert_called_once_with(
            model_name="BAAI/bge-small-en-v1.5",
            device="cpu",
            normalize=True,
            embed_batch_size=32,
            trust_remote_code=False,
        )
        mock_reranker_cls.assert_called_once_with(
            model_name="BAAI/bge-reranker-base",
            top_n=5,
            device="cpu",
        )


class TestFAISSHNSWVectorStoreFallback:
    """Test suite for FAISS HNSW vector store initialization and fallback."""

    def test_empty_vector_store_fallback_creates_hnsw_flat(
        self,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify empty vector store fallback avoids VectorStoreIndex.from_vector_store crash.

        Args:
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "empty_index"
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(persist_dir),
            auto_load=False,
        )

        vector_store, storage_context, index, loaded = orchestrator._load_vector_store(
            persist_dir / "faiss_index",
            "BAAI/bge-small-en-v1.5",
        )

        assert not loaded
        assert vector_store is not None
        assert storage_context is not None
        assert index is not None
        # Verify internal FAISS index is IndexHNSWFlat with dimension 384
        assert vector_store._faiss_index.d == 384
        assert isinstance(vector_store._faiss_index, faiss.IndexHNSWFlat)


class TestIndexedFileHashesAndDuplicateDetection:
    """Test suite for file hash duplicate detection."""

    def test_get_indexed_file_hashes_from_docstore_and_manifest(
        self,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify get_indexed_file_hashes aggregates hashes from docstore and manifest.

        Args:
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "index_with_manifest"
        persist_dir.mkdir(parents=True, exist_ok=True)

        manifest_data = {
            "version": "1.0",
            "timestamp": "2026-09-14T00:00:00Z",
            "model_name": "BAAI/bge-small-en-v1.5",
            "embedding_dimension": 384,
            "total_nodes": 1,
            "indexed_file_hashes": ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
            "files": {},
        }
        (persist_dir / "manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")

        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(persist_dir),
            auto_load=False,
        )

        mock_index = MagicMock()
        mock_doc = MagicMock()
        mock_doc.metadata = {"file_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
        mock_index.docstore.docs.values.return_value = [mock_doc]
        orchestrator._index = mock_index

        hashes = orchestrator.get_indexed_file_hashes()
        assert "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" in hashes
        assert "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" in hashes

    @patch("src.rag.orchestrator.DoclingPipeline")
    def test_ingest_documents_skips_duplicates(
        self,
        mock_pipeline_cls: MagicMock,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify ingest_documents skips files whose hashes are already indexed.

        Args:
            mock_pipeline_cls: Mocked DoclingPipeline class.
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)

        file1 = docs_dir / "doc1.txt"
        file1.write_text("Unique content for file 1", encoding="utf-8")
        hash1 = hashlib.sha256(b"Unique content for file 1").hexdigest()

        file2 = docs_dir / "doc2.txt"
        file2.write_text("Unique content for file 2", encoding="utf-8")
        hash2 = hashlib.sha256(b"Unique content for file 2").hexdigest()

        mock_instance = MagicMock()
        mock_pipeline_cls.return_value = mock_instance

        node2 = TextNode(
            text="Unique content for file 2",
            metadata={
                "file_name": "doc2.txt",
                "file_type": "txt",
                "creation_date": "2026-01-01T00:00:00Z",
                "ingestion_date": "2026-01-01T00:00:00Z",
                "file_hash": hash2,
            },
        )
        mock_instance.ingest_file.return_value = [node2]

        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(tmp_path / "index"),
            auto_load=False,
        )
        # Pre-seed hash1 as already indexed
        orchestrator.get_indexed_file_hashes = MagicMock(return_value={hash1})  # type: ignore[method-assign]

        ingested_nodes = orchestrator.ingest_documents(docs_dir)

        # File 1 must have been skipped; only file 2 ingested
        mock_instance.ingest_file.assert_called_once_with(file2)
        assert len(ingested_nodes) == 1
        assert ingested_nodes[0].metadata["file_hash"] == hash2


class TestIncrementalAppendsAndBM25Sync:
    """Test suite for incremental FAISS appends and BM25 synchronization."""

    @patch("src.rag.orchestrator.BM25Retriever")
    def test_index_nodes_incremental_append_and_bm25_cumulative(
        self,
        mock_bm25_cls: MagicMock,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify index_nodes incrementally appends to FAISS and uses cumulative docstore nodes for BM25.

        Args:
            mock_bm25_cls: Mocked BM25Retriever class.
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(persist_dir),
            auto_load=False,
        )
        orchestrator._initialized = True

        mock_index = MagicMock()
        mock_storage = MagicMock()
        mock_index.storage_context = mock_storage

        # Setup docstore with existing node and new node
        node1 = TextNode(
            text="Initial document",
            id_="n1",
            metadata={"file_hash": "hash_1"},
        )
        node2 = TextNode(
            text="Incrementally appended document",
            id_="n2",
            metadata={
                "file_name": "test2.txt",
                "file_type": "text/plain",
                "creation_date": "2026-09-14T10:00:00Z",
                "ingestion_date": "2026-09-14T10:00:00Z",
                "file_hash": "b" * 64,
            },
        )

        mock_index.docstore.docs = {"n1": node1}
        mock_index.insert_nodes.side_effect = lambda nodes: mock_index.docstore.docs.update(
            {n.id_: n for n in nodes}
        )
        orchestrator._index = mock_index
        orchestrator._storage_context = mock_storage
        orchestrator.get_indexed_file_hashes = MagicMock(return_value={"hash_1"})  # type: ignore[method-assign]

        mock_bm25_instance = MagicMock()
        mock_bm25_cls.from_defaults.return_value = mock_bm25_instance

        # Call index_nodes with both node1 (duplicate hash) and node2 (new)
        orchestrator.index_nodes([node1, node2])

        # Verify only node2 was inserted into FAISS index
        mock_index.insert_nodes.assert_called_once_with([node2])
        mock_storage.persist.assert_called_once()

        # Verify BM25 was rebuilt with cumulative nodes (both n1 and n2)
        mock_bm25_cls.from_defaults.assert_called_once()
        call_kwargs = mock_bm25_cls.from_defaults.call_args[1]
        cumulative_passed = call_kwargs["nodes"]
        assert len(cumulative_passed) == 2
        assert {n.id_ for n in cumulative_passed} == {"n1", "n2"}
        mock_bm25_instance.persist.assert_called_once()


class TestManifestAndSnapshotPersistence:
    """Test suite for manifest.json generation and snapshot versioning."""

    def test_generate_manifest_creates_valid_schema(
        self,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify _generate_manifest produces required schema and computes SHA-256 for index files.

        Args:
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        faiss_dir = persist_dir / "faiss_index"
        faiss_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = faiss_dir / "default__vector_store.json"
        content = b'{"test": "data"}'
        dummy_file.write_bytes(content)
        expected_checksum = hashlib.sha256(content).hexdigest()

        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(persist_dir),
            auto_load=False,
        )

        mock_index = MagicMock()
        mock_doc = MagicMock()
        mock_doc.metadata = {"file_hash": "abc123hash"}
        mock_index.docstore.docs = {"node_1": mock_doc}
        orchestrator._index = mock_index

        manifest = orchestrator._generate_manifest(persist_dir)

        assert manifest["version"] == "1.0"
        assert manifest["model_name"] == "BAAI/bge-small-en-v1.5"
        assert manifest["embedding_dimension"] == 384
        assert manifest["total_nodes"] == 1
        assert "abc123hash" in manifest["indexed_file_hashes"]
        assert f"faiss_index/{dummy_file.name}" in manifest["files"]
        assert manifest["files"][f"faiss_index/{dummy_file.name}"] == expected_checksum

    def test_create_snapshot_persists_isolated_version(
        self,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify create_snapshot writes manifest and copies index directories.

        Args:
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        (persist_dir / "faiss_index").mkdir(parents=True, exist_ok=True)
        (persist_dir / "faiss_index" / "store.bin").write_text("vector_data", encoding="utf-8")
        (persist_dir / "bm25_index").mkdir(parents=True, exist_ok=True)
        (persist_dir / "bm25_index" / "bm25.json").write_text("bm25_data", encoding="utf-8")

        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(persist_dir),
            auto_load=False,
        )
        orchestrator._index = MagicMock()
        orchestrator._index.docstore.docs = {}

        custom_snapshot_dir = tmp_path / "snapshots" / "test_snap"
        result_dir = orchestrator.create_snapshot(snapshot_dir=custom_snapshot_dir)

        assert result_dir.exists()
        assert (result_dir / "manifest.json").exists()
        assert (result_dir / "faiss_index" / "store.bin").exists()
        assert (result_dir / "bm25_index" / "bm25.json").exists()


class TestBGECrossEncoderReranker:
    """Test suite for BGECrossEncoderReranker postprocessor."""

    @patch("sentence_transformers.CrossEncoder")
    def test_reranker_reordering_and_sigmoid_normalization(
        self,
        mock_cross_encoder_cls: MagicMock,
    ) -> None:
        """Verify cross-encoder scores are sigmoid normalized into [0, 1] and truncated to top_n.

        Args:
            mock_cross_encoder_cls: Mocked CrossEncoder class.
        """
        from llama_index.core import QueryBundle

        mock_model = MagicMock()
        # Raw logits: relevant snippet gets +3.0, irrelevant gets -3.0
        mock_model.predict.return_value = [ -3.0, 3.0 ]
        mock_cross_encoder_cls.return_value = mock_model

        reranker = BGECrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            top_n=2,
            device="cpu",
        )

        node_low = TextNode(text="Irrelevant text", id_="low")
        node_high = TextNode(text="Highly relevant text", id_="high")
        candidates = [
            NodeWithScore(node=node_low, score=0.5),
            NodeWithScore(node=node_high, score=0.5),
        ]

        query = QueryBundle("Relevant query")
        reranked = reranker.postprocess_nodes(candidates, query)

        assert len(reranked) == 2
        # Highest score must be first
        assert reranked[0].node.id_ == "high"
        assert reranked[1].node.id_ == "low"
        # Scores must be bounded in [0.0, 1.0]
        assert 0.0 <= (reranked[0].score or 0.0) <= 1.0
        assert 0.0 <= (reranked[1].score or 0.0) <= 1.0
        assert (reranked[0].score or 0.0) > (reranked[1].score or 0.0)

    @patch("sentence_transformers.CrossEncoder")
    def test_reranker_fallback_on_inference_error(
        self,
        mock_cross_encoder_cls: MagicMock,
    ) -> None:
        """Verify cross-encoder gracefully returns original candidates on inference exception.

        Args:
            mock_cross_encoder_cls: Mocked CrossEncoder class.
        """
        from llama_index.core import QueryBundle

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Inference engine failure")
        mock_cross_encoder_cls.return_value = mock_model

        reranker = BGECrossEncoderReranker(
            model_name="BAAI/bge-reranker-base",
            top_n=2,
            device="cpu",
        )

        node_a = TextNode(text="Snippet A", id_="A")
        node_b = TextNode(text="Snippet B", id_="B")
        candidates = [
            NodeWithScore(node=node_a, score=0.88),
            NodeWithScore(node=node_b, score=0.74),
        ]

        query = QueryBundle("Query")
        result = reranker.postprocess_nodes(candidates, query)

        # Fallback contract: caught exception returns original candidates unchanged
        assert len(result) == 2
        assert result[0].node.id_ == "A"
        assert result[1].node.id_ == "B"


class TestQuerySearchAndHealthStatus:
    """Test suite for search_documents, query, health status, and singleton accessor."""

    def test_search_documents_returns_structured_list(
        self,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify search_documents retrieves and formats result dictionaries with raw text.

        Args:
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(tmp_path / "persist"),
            auto_load=False,
        )
        orchestrator._initialized = True

        mock_index = MagicMock()
        mock_retriever = MagicMock()
        node = TextNode(
            text="Raw payload signature content",
            id_="doc_node_1",
            metadata={"file_name": "signature.txt"},
        )
        mock_retriever.retrieve.return_value = [NodeWithScore(node=node, score=0.92)]
        mock_index.as_retriever.return_value = mock_retriever
        orchestrator._index = mock_index

        results = orchestrator.search_documents("signature", top_k=1)

        assert len(results) == 1
        assert results[0]["id"] == "doc_node_1"
        assert results[0]["text"] == "Raw payload signature content"
        assert results[0]["score"] == pytest.approx(1.012)
        assert results[0]["metadata"]["file_name"] == "signature.txt"

    @patch("llama_index.core.query_engine.RetrieverQueryEngine.aquery")
    def test_query_and_query_async(
        self,
        mock_aquery: MagicMock,
        temp_config_file: str,
        tmp_path: Path,
    ) -> None:
        """Verify query returns MCPContextPayload with populated context items.

        Args:
            mock_aquery: Mocked aquery method of RetrieverQueryEngine.
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
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

        node = TextNode(
            text="Simulated query response text",
            id_="res_1",
            metadata={"file_name": "test.txt"},
        )
        mock_response = MagicMock()
        mock_response.source_nodes = [NodeWithScore(node=node, score=0.85)]

        async def mock_async_return(*args: Any, **kwargs: Any) -> Any:
            return mock_response

        mock_aquery.side_effect = mock_async_return

        payload: MCPContextPayload = orchestrator.query("test query", top_k=1)

        assert isinstance(payload, MCPContextPayload)
        assert payload.query == "test query"
        assert len(payload.context) == 1
        assert payload.context[0].id == "res_1"
        assert payload.context[0].text == "Simulated query response text"
        assert payload.context[0].score == 0.85

    def test_get_health_status(self, temp_config_file: str, tmp_path: Path) -> None:
        """Verify get_health_status returns structured status indicators.

        Args:
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
        orchestrator = RAGOrchestrator(
            config_path=temp_config_file,
            persist_dir=str(tmp_path / "persist"),
            auto_load=False,
        )

        status = orchestrator.get_health_status()
        assert status["overall_status"] == "degraded"
        assert not status["initialized"]
        assert "services" in status
        assert "vector_store" in status["services"]
        assert "bm25_retriever" in status["services"]
        assert "reranker" in status["services"]

    def test_get_orchestrator_singleton(self, temp_config_file: str, tmp_path: Path) -> None:
        """Verify get_orchestrator returns a singleton instance.

        Args:
            temp_config_file: Path to temporary config YAML.
            tmp_path: Temporary directory fixture.
        """
        import src.rag.orchestrator as orch_module

        orch_module._orchestrator_instance = None
        instance1 = get_orchestrator(config_path=temp_config_file, persist_dir=str(tmp_path / "singleton"))
        instance2 = get_orchestrator(config_path=temp_config_file, persist_dir=str(tmp_path / "singleton"))
        assert instance1 is instance2
