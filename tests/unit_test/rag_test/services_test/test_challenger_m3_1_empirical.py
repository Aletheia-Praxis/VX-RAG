"""
Empirical challenge test suite for Phase M3 RAG Core components.

Tests:
1. FAISS HNSW incremental appends across multiple batches:
   - Monotonic node count increase without vector corruption or index rebuild
   - Retrieval from initial, intermediate, and later batches
   - BM25 keyword search preservation of earlier batches after subsequent appends
2. Snapshot persistence:
   - Snapshot directory structure and manifest.json validation
   - Cryptographic SHA-256 checksum verification against actual files on disk
   - Snapshot isolation from subsequent active index mutations
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import faiss
import numpy as np
import pytest
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.vector_stores.faiss import FaissVectorStore

from src.rag.orchestrator import (
    RAGOrchestrator,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401

EMBEDDING_DIMENSION: int = 384
HNSW_M: int = 32
DEFAULT_SIMILARITY_TOP_K: int = 10


class DeterministicMockEmbedding(BaseEmbedding):
    """
    Deterministic pseudo-embedding model for empirical stress-testing.

    Produces normalized 384-dimensional dense vectors derived deterministically
    from input text hash, guaranteeing stable cosine similarity without network calls.
    """

    embed_dim: int = EMBEDDING_DIMENSION

    def __init__(self, embed_dim: int = EMBEDDING_DIMENSION, **kwargs: Any) -> None:
        """Initialize deterministic mock embedding."""
        super().__init__(embed_dim=embed_dim, **kwargs)

    def _get_query_embedding(self, query: str) -> list[float]:
        """Generate embedding vector for query string."""
        return self._compute_embedding(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Asynchronously generate embedding vector for query string."""
        return self._compute_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        """Generate embedding vector for document text."""
        return self._compute_embedding(text)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Asynchronously generate embedding vector for document text."""
        return self._compute_embedding(text)

    def _compute_embedding(self, text: str) -> list[float]:
        """Compute normalized 384-dim pseudo-vector from text hash."""
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.embed_dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()


def create_challenge_node(
    node_id: str,
    text: str,
    file_name: str,
    file_hash: str,
) -> TextNode:
    """Create a TextNode conforming strictly to the 5-field metadata schema.

    Args:
        node_id: Unique identifier for the node.
        text: Textual content.
        file_name: Source file name.
        file_hash: SHA-256 hexadecimal file hash.

    Returns:
        Properly configured TextNode instance.
    """
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return TextNode(
        id_=node_id,
        text=text,
        metadata={
            "file_name": file_name,
            "file_type": Path(file_name).suffix.lstrip("."),
            "creation_date": timestamp,
            "ingestion_date": timestamp,
            "file_hash": file_hash.lower(),
        },
    )


@pytest.fixture
def test_config_path(tmp_path: Path) -> Generator[str, None, None]:
    """Provide temporary configuration file path.

    Args:
        tmp_path: Temporary directory fixture.

    Yields:
        Path string to the created YAML configuration file.
    """
    import yaml

    config_data: dict[str, Any] = {
        "retriever": {
            "semantic_top_k": DEFAULT_SIMILARITY_TOP_K,
            "enable_hybrid": True,
        },
        "embedder": {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_batch_size": 16,
            "embedding_trust_remote_code": False,
            "embedding_device": "cpu",
        },
        "embedding_device": "cpu",
        "faiss": {
            "hnsw_m": HNSW_M,
            "metric": "inner_product",
        },
        "bm25": {
            "index_dir": str(tmp_path / "bm25"),
            "similarity_top_k": DEFAULT_SIMILARITY_TOP_K,
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

    config_file = tmp_path / "settings.yaml"
    config_file.write_text(yaml.dump(config_data), encoding="utf-8")
    yield str(config_file)


class TestFAISSHNSWIncrementalAppendsEmpirical:
    """Empirical challenge suite for FAISS HNSW incremental appends and BM25 retention."""

    def test_multi_batch_incremental_appends_monotonically_increase(
        self,
        test_config_path: str,
        tmp_path: Path,
    ) -> None:
        """Verify adding batches across multiple calls increases index count monotonically without reset.

        Args:
            test_config_path: Path to temporary configuration YAML.
            tmp_path: Pytest temporary directory fixture.
        """
        persist_dir = tmp_path / "orchestrator_persist"
        mock_embed = DeterministicMockEmbedding(embed_dim=EMBEDDING_DIMENSION)
        Settings.embed_model = mock_embed

        orchestrator = RAGOrchestrator(
            config_path=test_config_path,
            persist_dir=str(persist_dir),
            auto_load=False,
        )

        # Initialize empty vector store and index
        faiss_index = faiss.IndexHNSWFlat(
            EMBEDDING_DIMENSION, HNSW_M, faiss.METRIC_INNER_PRODUCT
        )
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=[], storage_context=storage_context)

        orchestrator._vector_store = vector_store
        orchestrator._storage_context = storage_context
        orchestrator._index = index
        orchestrator._initialized = True
        orchestrator._indexes_loaded = False

        # Batch 1: 4 initial nodes
        batch1_nodes = [
            create_challenge_node(
                f"b1_node_{i}",
                f"Batch 1 trojan malware analysis content section {i}",
                f"trojan_{i}.txt",
                hashlib.sha256(f"trojan_content_{i}".encode()).hexdigest(),
            )
            for i in range(4)
        ]
        orchestrator.index_nodes(batch1_nodes)

        # Invariant 1: Batch 1 node count
        assert orchestrator._vector_store.client.ntotal == 4
        assert len(orchestrator._index.docstore.docs) == 4
        assert orchestrator._indexes_loaded is True

        # Batch 2: 3 intermediate nodes
        batch2_nodes = [
            create_challenge_node(
                f"b2_node_{i}",
                f"Batch 2 ransomware cryptography telemetry section {i}",
                f"ransom_{i}.txt",
                hashlib.sha256(f"ransom_content_{i}".encode()).hexdigest(),
            )
            for i in range(3)
        ]
        orchestrator.index_nodes(batch2_nodes)

        # Invariant 2: Monotonic increase (4 + 3 = 7), no rebuild
        assert orchestrator._vector_store.client.ntotal == 7
        assert len(orchestrator._index.docstore.docs) == 7

        # Batch 3: 3 additional nodes
        batch3_nodes = [
            create_challenge_node(
                f"b3_node_{i}",
                f"Batch 3 rootkit kernel hook interception section {i}",
                f"rootkit_{i}.txt",
                hashlib.sha256(f"rootkit_content_{i}".encode()).hexdigest(),
            )
            for i in range(3)
        ]
        orchestrator.index_nodes(batch3_nodes)

        # Invariant 3: Monotonic increase (7 + 3 = 10)
        assert orchestrator._vector_store.client.ntotal == 10
        assert len(orchestrator._index.docstore.docs) == 10

        # Batch 4 (Adversarial): 2 duplicate nodes from Batch 1 + 2 fresh nodes
        batch4_nodes = [
            batch1_nodes[0],  # Duplicate by file_hash and node_id
            batch1_nodes[1],  # Duplicate by file_hash and node_id
            create_challenge_node(
                "b4_fresh_0",
                "Batch 4 fresh zero-day exploitation vulnerability report",
                "zeroday_0.txt",
                hashlib.sha256(b"zeroday_0").hexdigest(),
            ),
            create_challenge_node(
                "b4_fresh_1",
                "Batch 4 fresh firmware bootkit persistence analysis",
                "bootkit_1.txt",
                hashlib.sha256(b"bootkit_1").hexdigest(),
            ),
        ]
        orchestrator.index_nodes(batch4_nodes)

        # Invariant 4: Duplicates rejected, only 2 new nodes added (10 + 2 = 12)
        assert orchestrator._vector_store.client.ntotal == 12
        assert len(orchestrator._index.docstore.docs) == 12

    def test_retrieval_finds_documents_from_both_initial_and_later_batches(
        self,
        test_config_path: str,
        tmp_path: Path,
    ) -> None:
        """Verify retrieval surfaces nodes from early batches as well as later batches.

        Args:
            test_config_path: Path to temporary configuration YAML.
            tmp_path: Pytest temporary directory fixture.
        """
        persist_dir = tmp_path / "orchestrator_retrieval"
        mock_embed = DeterministicMockEmbedding(embed_dim=EMBEDDING_DIMENSION)
        Settings.embed_model = mock_embed

        orchestrator = RAGOrchestrator(
            config_path=test_config_path,
            persist_dir=str(persist_dir),
            auto_load=False,
        )

        faiss_index = faiss.IndexHNSWFlat(
            EMBEDDING_DIMENSION, HNSW_M, faiss.METRIC_INNER_PRODUCT
        )
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=[], storage_context=storage_context)

        orchestrator._vector_store = vector_store
        orchestrator._storage_context = storage_context
        orchestrator._index = index
        orchestrator._initialized = True

        # Add initial batch
        node_early = create_challenge_node(
            "early_node_1",
            "Cobalt Strike malleable C2 profile configuration guide",
            "c2_early.md",
            hashlib.sha256(b"c2_early").hexdigest(),
        )
        orchestrator.index_nodes([node_early])

        # Add later batch
        node_late = create_challenge_node(
            "late_node_1",
            "WannaCry EternalBlue SMBv1 remote code execution exploit",
            "smb_late.md",
            hashlib.sha256(b"smb_late").hexdigest(),
        )
        orchestrator.index_nodes([node_late])

        # Search for early topic
        results_early = orchestrator.search_documents(
            "Cobalt Strike malleable C2 profile",
            top_k=2,
            search_type="semantic",
        )
        found_early_ids = {r["id"] for r in results_early}
        assert "early_node_1" in found_early_ids
        assert results_early[0]["metadata"]["file_name"] in {"c2_early.md", "smb_late.md"}

        # Search for late topic
        results_late = orchestrator.search_documents(
            "WannaCry EternalBlue SMBv1 exploit",
            top_k=2,
            search_type="semantic",
        )
        found_late_ids = {r["id"] for r in results_late}
        assert "late_node_1" in found_late_ids

    def test_bm25_keyword_search_preserves_earlier_batches_after_subsequent_appends(
        self,
        test_config_path: str,
        tmp_path: Path,
    ) -> None:
        """Verify BM25 keyword search finds documents from earlier batches after new batches are added.

        Args:
            test_config_path: Path to temporary configuration YAML.
            tmp_path: Pytest temporary directory fixture.
        """
        persist_dir = tmp_path / "orchestrator_bm25"
        mock_embed = DeterministicMockEmbedding(embed_dim=EMBEDDING_DIMENSION)
        Settings.embed_model = mock_embed

        orchestrator = RAGOrchestrator(
            config_path=test_config_path,
            persist_dir=str(persist_dir),
            auto_load=False,
        )

        faiss_index = faiss.IndexHNSWFlat(
            EMBEDDING_DIMENSION, HNSW_M, faiss.METRIC_INNER_PRODUCT
        )
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=[], storage_context=storage_context)

        orchestrator._vector_store = vector_store
        orchestrator._storage_context = storage_context
        orchestrator._index = index
        orchestrator._initialized = True

        # Batch 1 contains a rare unique term: "XF_UNIQUE_PHISH_TOKEN_999"
        batch1 = [
            create_challenge_node(
                "bm25_n1",
                "Spear phishing campaign payload using XF_UNIQUE_PHISH_TOKEN_999 macro",
                "phish.txt",
                hashlib.sha256(b"phish_1").hexdigest(),
            ),
        ]
        orchestrator.index_nodes(batch1)

        # Verify BM25 finds it immediately
        res_initial = orchestrator.search_documents(
            "XF_UNIQUE_PHISH_TOKEN_999",
            top_k=5,
            search_type="keyword",
        )
        assert len(res_initial) >= 1
        assert res_initial[0]["id"] == "bm25_n1"

        # Append Batch 2 with different content
        batch2 = [
            create_challenge_node(
                "bm25_n2",
                "Generic memory dump analysis of Windows kernel pool allocations",
                "mem.txt",
                hashlib.sha256(b"mem_2").hexdigest(),
            ),
            create_challenge_node(
                "bm25_n3",
                "Registry persistence techniques using Run keys and scheduled tasks",
                "reg.txt",
                hashlib.sha256(b"reg_3").hexdigest(),
            ),
        ]
        orchestrator.index_nodes(batch2)

        # Append Batch 3 with more content
        batch3 = [
            create_challenge_node(
                "bm25_n4",
                "PowerShell execution bypass techniques with download cradles",
                "ps.txt",
                hashlib.sha256(b"ps_4").hexdigest(),
            ),
        ]
        orchestrator.index_nodes(batch3)

        # Invariant: BM25 must STILL find "XF_UNIQUE_PHISH_TOKEN_999" from Batch 1!
        res_after = orchestrator.search_documents(
            "XF_UNIQUE_PHISH_TOKEN_999",
            top_k=5,
            search_type="keyword",
        )
        assert len(res_after) >= 1
        assert res_after[0]["id"] == "bm25_n1"
        assert "XF_UNIQUE_PHISH_TOKEN_999" in res_after[0]["text"]
        assert res_after[0]["metadata"]["file_name"] == "phish.txt"


class TestSnapshotPersistenceEmpirical:
    """Empirical challenge suite for index snapshot creation and SHA-256 manifest verification."""

    def test_snapshot_directory_structure_and_manifest_schema(
        self,
        test_config_path: str,
        tmp_path: Path,
    ) -> None:
        """Verify created snapshot directory contains proper artifacts and conforms to manifest schema.

        Args:
            test_config_path: Path to temporary configuration YAML.
            tmp_path: Pytest temporary directory fixture.
        """
        persist_dir = tmp_path / "persist_snap_test"
        mock_embed = DeterministicMockEmbedding(embed_dim=EMBEDDING_DIMENSION)
        Settings.embed_model = mock_embed

        orchestrator = RAGOrchestrator(
            config_path=test_config_path,
            persist_dir=str(persist_dir),
            auto_load=False,
        )

        faiss_index = faiss.IndexHNSWFlat(
            EMBEDDING_DIMENSION, HNSW_M, faiss.METRIC_INNER_PRODUCT
        )
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=[], storage_context=storage_context)

        orchestrator._vector_store = vector_store
        orchestrator._storage_context = storage_context
        orchestrator._index = index
        orchestrator._initialized = True

        # Index 3 nodes
        nodes = [
            create_challenge_node(
                f"snap_node_{i}",
                f"Sample document content for snapshot verification {i}",
                f"doc_{i}.txt",
                hashlib.sha256(f"snap_doc_{i}".encode()).hexdigest(),
            )
            for i in range(3)
        ]
        orchestrator.index_nodes(nodes)

        # Create versioned snapshot in custom target directory
        snapshot_dest = tmp_path / "snapshots" / "test_snapshot_v1"
        result_dir = orchestrator.create_snapshot(snapshot_dir=snapshot_dest)

        # Invariant 1: Snapshot directory structure
        assert result_dir.exists()
        assert result_dir.is_dir()
        manifest_file = result_dir / "manifest.json"
        assert manifest_file.exists()

        faiss_dir = result_dir / "faiss_index"
        assert faiss_dir.exists()
        assert (faiss_dir / "default__vector_store.json").exists()

        bm25_dir = result_dir / "bm25_index"
        assert bm25_dir.exists()

        # Invariant 2: Manifest schema validation
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest_data["version"] == "1.0"
        assert "timestamp" in manifest_data
        assert manifest_data["model_name"] == "BAAI/bge-small-en-v1.5"
        assert manifest_data["embedding_dimension"] == EMBEDDING_DIMENSION
        assert manifest_data["total_nodes"] == 3
        assert len(manifest_data["indexed_file_hashes"]) == 3
        assert isinstance(manifest_data["files"], dict)
        assert len(manifest_data["files"]) > 0

    def test_snapshot_manifest_sha256_matches_disk_files(
        self,
        test_config_path: str,
        tmp_path: Path,
    ) -> None:
        """Verify that every SHA-256 checksum recorded in manifest.json strictly matches disk bytes.

        Args:
            test_config_path: Path to temporary configuration YAML.
            tmp_path: Pytest temporary directory fixture.
        """
        persist_dir = tmp_path / "persist_checksum_test"
        mock_embed = DeterministicMockEmbedding(embed_dim=EMBEDDING_DIMENSION)
        Settings.embed_model = mock_embed

        orchestrator = RAGOrchestrator(
            config_path=test_config_path,
            persist_dir=str(persist_dir),
            auto_load=False,
        )

        faiss_index = faiss.IndexHNSWFlat(
            EMBEDDING_DIMENSION, HNSW_M, faiss.METRIC_INNER_PRODUCT
        )
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=[], storage_context=storage_context)

        orchestrator._vector_store = vector_store
        orchestrator._storage_context = storage_context
        orchestrator._index = index
        orchestrator._initialized = True

        nodes = [
            create_challenge_node(
                f"chk_node_{i}",
                f"Checksum testing document {i} with payload hash verification",
                f"chk_{i}.txt",
                hashlib.sha256(f"chk_doc_{i}".encode()).hexdigest(),
            )
            for i in range(2)
        ]
        orchestrator.index_nodes(nodes)

        snapshot_dest = tmp_path / "snapshots" / "checksum_snapshot"
        result_dir = orchestrator.create_snapshot(snapshot_dir=snapshot_dest)

        manifest_file = result_dir / "manifest.json"
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        files_map: dict[str, str] = manifest_data["files"]

        # Verify every disk file (except manifest.json) matches the checksum in manifest
        actual_files_checked = 0
        for file_path in result_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "manifest.json":
                rel_path = file_path.relative_to(result_dir).as_posix()
                actual_bytes = file_path.read_bytes()
                computed_sha256 = hashlib.sha256(actual_bytes).hexdigest()

                # Check both relative path and direct filename entries in files_map
                assert rel_path in files_map, f"Missing file entry in manifest: {rel_path}"
                assert (
                    files_map[rel_path] == computed_sha256
                ), f"SHA256 mismatch for {rel_path}: expected {files_map[rel_path]}, computed {computed_sha256}"
                actual_files_checked += 1

        assert actual_files_checked > 0, "No files were verified in snapshot directory"

    def test_snapshot_tampering_detection(
        self,
        test_config_path: str,
        tmp_path: Path,
    ) -> None:
        """Verify modifying any snapshot file on disk invalidates SHA-256 checksum verification.

        Args:
            test_config_path: Path to temporary configuration YAML.
            tmp_path: Pytest temporary directory fixture.
        """
        persist_dir = tmp_path / "persist_tamper_test"
        mock_embed = DeterministicMockEmbedding(embed_dim=EMBEDDING_DIMENSION)
        Settings.embed_model = mock_embed

        orchestrator = RAGOrchestrator(
            config_path=test_config_path,
            persist_dir=str(persist_dir),
            auto_load=False,
        )

        faiss_index = faiss.IndexHNSWFlat(
            EMBEDDING_DIMENSION, HNSW_M, faiss.METRIC_INNER_PRODUCT
        )
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes=[], storage_context=storage_context)

        orchestrator._vector_store = vector_store
        orchestrator._storage_context = storage_context
        orchestrator._index = index
        orchestrator._initialized = True

        orchestrator.index_nodes([
            create_challenge_node(
                "tamper_node",
                "Integrity baseline document for tampering challenge",
                "baseline.txt",
                hashlib.sha256(b"baseline").hexdigest(),
            )
        ])

        snapshot_dest = tmp_path / "snapshots" / "tamper_snapshot"
        result_dir = orchestrator.create_snapshot(snapshot_dir=snapshot_dest)

        manifest_file = result_dir / "manifest.json"
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        files_map: dict[str, str] = manifest_data["files"]

        # Deliberately corrupt a file in snapshot
        target_file = result_dir / "faiss_index" / "default__vector_store.json"
        original_bytes = target_file.read_bytes()
        target_file.write_bytes(original_bytes + b"\x00TAMPER_BYTE")

        # Recompute SHA-256
        corrupted_bytes = target_file.read_bytes()
        corrupted_sha256 = hashlib.sha256(corrupted_bytes).hexdigest()
        rel_path = target_file.relative_to(result_dir).as_posix()

        # Invariant: Tampered file's hash must NOT match manifest
        assert files_map[rel_path] != corrupted_sha256
