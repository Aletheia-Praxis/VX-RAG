"""Unit tests for DuckDB-backed docstore and intermediate node persistence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, patch

import pytest
import yaml
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.storage.docstore.simple_docstore import SimpleDocumentStore
from llama_index.storage.kvstore.duckdb import DuckDBKVStore

os.environ["IS_TESTING"] = "1"

from src.rag.orchestrator import (
    DEFAULT_DOCSTORE_DB_NAME,
    DEFAULT_INTERMEDIATE_NODES_DB_NAME,
    DEFAULT_INTERMEDIATE_NODES_TABLE_NAME,
    RAGOrchestrator,
    _close_duckdb_kvstore,
    _create_duckdb_kvstore,
    load_intermediate_nodes,
    persist_intermediate_nodes,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401


class MockDeterministicEmbedding(BaseEmbedding):
    """Deterministic mock embedding for unit tests."""

    dimension: int = 384

    def __init__(self, **kwargs: Any) -> None:
        """Initialize mock embedding model."""
        super().__init__(model_name="mock-deterministic", **kwargs)

    def _get_query_embedding(self, query: str) -> list[float]:
        """Generate deterministic 384-dimensional query embedding."""
        return [0.05] * self.dimension

    def _get_text_embedding(self, text: str) -> list[float]:
        """Generate deterministic 384-dimensional text embedding."""
        return [0.05] * self.dimension

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Generate deterministic async query embedding."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Generate deterministic async text embedding."""
        return self._get_text_embedding(text)


@pytest.fixture
def storage_temp_config(tmp_path: Path) -> Generator[str, None, None]:
    """Create a temporary configuration file configured for DuckDB docstore.

    Args:
        tmp_path: Pytest temporary path fixture.

    Yields:
        Path string to the created temporary settings.yaml.
    """
    config_data: dict[str, Any] = {
        "retriever": {
            "semantic_top_k": 5,
            "enable_hybrid": True,
        },
        "embedder": {
            "embedding_model": "BAAI/bge-small-en-v1.5",
            "embedding_batch_size": 10,
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
            "similarity_top_k": 5,
        },
        "reranker": {
            "enable_metadata_prioritization": False,
            "model_name": "BAAI/bge-reranker-base",
            "top_k": 5,
            "device": "cpu",
        },
        "duplicate_detection": {
            "similarity_threshold": 0.95,
            "hash_algorithm": "sha256",
        },
        "docstore": {
            "store_type": "duckdb",
            "db_name": "docstore.duckdb",
            "table_name": "docstore",
        },
    }

    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.dump(config_data), encoding="utf-8")
    yield str(config_path)


class TestIntermediateNodesDuckDBPersistence:
    """Test suite for intermediate node caching using DuckDB instead of JSON."""

    def test_persist_and_load_intermediate_nodes_duckdb(self, tmp_path: Path) -> None:
        """Verify saving and loading intermediate nodes via DuckDBKVStore.

        Args:
            tmp_path: Temporary directory fixture.
        """
        nodes = [
            TextNode(
                text="Document chunk 1",
                id_="node-001",
                metadata={"file_name": "guide.md", "file_hash": "hash_111"},
            ),
            TextNode(
                text="Document chunk 2",
                id_="node-002",
                metadata={"file_name": "manual.pdf", "file_hash": "hash_222"},
            ),
        ]

        saved_path = persist_intermediate_nodes(nodes, tmp_path)
        assert saved_path.exists()
        assert saved_path.name == DEFAULT_INTERMEDIATE_NODES_DB_NAME
        assert not (tmp_path / "nodes.json").exists()

        loaded_nodes = load_intermediate_nodes(tmp_path)
        assert len(loaded_nodes) == 2
        loaded_dict = {n.node_id: n for n in loaded_nodes}
        assert "node-001" in loaded_dict
        assert "node-002" in loaded_dict
        assert loaded_dict["node-001"].get_content() == "Document chunk 1"
        assert loaded_dict["node-001"].metadata["file_name"] == "guide.md"
        assert loaded_dict["node-002"].metadata["file_hash"] == "hash_222"

    def test_persist_intermediate_nodes_empty_list(self, tmp_path: Path) -> None:
        """Verify persisting an empty list creates valid DuckDB file without errors.

        Args:
            tmp_path: Temporary directory fixture.
        """
        saved_path = persist_intermediate_nodes([], tmp_path)
        assert saved_path.exists()
        loaded = load_intermediate_nodes(tmp_path)
        assert loaded == []

    def test_load_intermediate_nodes_fallback_to_legacy_json(self, tmp_path: Path) -> None:
        """Verify transparent fallback to legacy nodes.json when nodes.duckdb is absent.

        Args:
            tmp_path: Temporary directory fixture.
        """
        legacy_nodes = [
            TextNode(text="Legacy chunk content", id_="legacy-001", metadata={"source": "v1"}),
        ]
        legacy_file = tmp_path / "nodes.json"
        legacy_file.write_text(
            json.dumps([node.to_dict() for node in legacy_nodes]),
            encoding="utf-8",
        )

        assert not (tmp_path / DEFAULT_INTERMEDIATE_NODES_DB_NAME).exists()
        loaded = load_intermediate_nodes(tmp_path, legacy_fallback=True)
        assert len(loaded) == 1
        assert loaded[0].node_id == "legacy-001"
        assert loaded[0].get_content() == "Legacy chunk content"

    def test_load_intermediate_nodes_prefers_duckdb_over_legacy(self, tmp_path: Path) -> None:
        """Verify that nodes.duckdb takes precedence over legacy nodes.json if both exist.

        Args:
            tmp_path: Temporary directory fixture.
        """
        duckdb_nodes = [TextNode(text="From DuckDB", id_="duckdb-node")]
        persist_intermediate_nodes(duckdb_nodes, tmp_path)

        legacy_nodes = [TextNode(text="From Legacy JSON", id_="json-node")]
        (tmp_path / "nodes.json").write_text(
            json.dumps([node.to_dict() for node in legacy_nodes]),
            encoding="utf-8",
        )

        loaded = load_intermediate_nodes(tmp_path)
        assert len(loaded) == 1
        assert loaded[0].node_id == "duckdb-node"
        assert loaded[0].get_content() == "From DuckDB"

    def test_load_intermediate_nodes_missing_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError is raised when neither duckdb nor legacy json exists.

        Args:
            tmp_path: Temporary directory fixture.
        """
        with pytest.raises(FileNotFoundError) as exc_info:
            load_intermediate_nodes(tmp_path)
        assert "nodes.duckdb" in str(exc_info.value)
        assert "nodes.json" in str(exc_info.value)

    def test_windows_file_locking_release_after_persist(self, tmp_path: Path) -> None:
        """Verify that Windows file handle is released and file can be read or copied.

        Args:
            tmp_path: Temporary directory fixture.
        """
        nodes = [TextNode(text="Lock test chunk", id_="lock-node-1")]
        db_path = persist_intermediate_nodes(nodes, tmp_path)

        # On Windows, reading raw bytes or copying fails with WinError 32 if file handle is held
        file_bytes = db_path.read_bytes()
        assert len(file_bytes) > 0

        copy_destination = tmp_path / "nodes_copied.duckdb"
        shutil.copy2(db_path, copy_destination)
        assert copy_destination.exists()

    def test_persist_intermediate_nodes_overwrite(self, tmp_path: Path) -> None:
        """Verify that persisting with overwrite=True replaces previous records cleanly.

        Args:
            tmp_path: Temporary directory fixture.
        """
        nodes_batch1 = [TextNode(text="Batch 1 Node", id_="b1-node")]
        persist_intermediate_nodes(nodes_batch1, tmp_path, overwrite=True)

        loaded_batch1 = load_intermediate_nodes(tmp_path)
        assert len(loaded_batch1) == 1
        assert loaded_batch1[0].node_id == "b1-node"

        nodes_batch2 = [
            TextNode(text="Batch 2 Node A", id_="b2-node-a"),
            TextNode(text="Batch 2 Node B", id_="b2-node-b"),
        ]
        persist_intermediate_nodes(nodes_batch2, tmp_path, overwrite=True)

        loaded_batch2 = load_intermediate_nodes(tmp_path)
        assert len(loaded_batch2) == 2
        loaded_ids = {n.node_id for n in loaded_batch2}
        assert "b1-node" not in loaded_ids
        assert "b2-node-a" in loaded_ids
        assert "b2-node-b" in loaded_ids

    def test_duckdb_retry_on_concurrent_file_lock(self, tmp_path: Path) -> None:
        """Verify _create_duckdb_kvstore retries and succeeds when lock is briefly held.

        Args:
            tmp_path: Temporary directory fixture.
        """
        db_name = "contended.duckdb"
        child_code = f"""
import time
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
kv = DuckDBKVStore(database_name='{db_name}', persist_dir={str(tmp_path)!r})
kv.put('child_key', {{'data': 'initial'}})
time.sleep(0.4)
"""
        proc = subprocess.Popen([sys.executable, "-c", child_code])
        try:
            time.sleep(0.1)
            kv2 = _create_duckdb_kvstore(
                database_name=db_name,
                persist_dir=str(tmp_path),
                table_name=DEFAULT_INTERMEDIATE_NODES_TABLE_NAME,
                max_retries=6,
                initial_backoff=0.15,
            )
            kv2.put("parent_key", {"data": "success"})
            assert kv2.get("parent_key") is not None
            _close_duckdb_kvstore(kv2)
        finally:
            proc.wait()

    def test_duckdb_wal_recovery_after_process_crash(self, tmp_path: Path) -> None:
        """Verify DuckDB recovers uncheckpointed WAL records after sudden process termination.

        Args:
            tmp_path: Temporary directory fixture.
        """
        db_name = "crash_recovery.duckdb"
        table_name = "crash_table"

        crash_script = f"""
import os
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
kv = DuckDBKVStore(database_name='{db_name}', persist_dir={str(tmp_path)!r}, table_name='{table_name}')
kv.put('crash_key_1', {{'content': 'uncheckpointed_data'}})
os._exit(42)
"""
        proc = subprocess.run([sys.executable, "-c", crash_script], check=False)
        assert proc.returncode == 42

        kv_reopened = _create_duckdb_kvstore(
            database_name=db_name,
            persist_dir=str(tmp_path),
            table_name=table_name,
        )
        try:
            data = kv_reopened.get("crash_key_1")
            assert data is not None
            assert data["content"] == "uncheckpointed_data"
        finally:
            _close_duckdb_kvstore(kv_reopened)


class TestDuckDBKVStoreOrchestratorIntegration:
    """Test suite for RAGOrchestrator integration with DuckDBKVStore."""

    @patch("src.rag.orchestrator.HuggingFaceEmbedding", return_value=MockDeterministicEmbedding())
    @patch("src.rag.orchestrator.BGECrossEncoderReranker")
    def test_orchestrator_initializes_duckdb_docstore(
        self,
        mock_reranker: MagicMock,
        mock_embedding: MagicMock,
        storage_temp_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify orchestrator creates DuckDBKVStore and KVDocumentStore.

        Args:
            mock_reranker: Mocked reranker class.
            mock_embedding: Mocked embedding class.
            storage_temp_config: Path to test config.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        orchestrator = RAGOrchestrator(
            config_path=storage_temp_config,
            persist_dir=str(persist_dir),
            auto_load=True,
        )

        assert isinstance(orchestrator._docstore, KVDocumentStore)
        assert isinstance(orchestrator._kvstore, DuckDBKVStore)
        assert (persist_dir / DEFAULT_DOCSTORE_DB_NAME).exists()
        orchestrator.close()

    @patch("src.rag.orchestrator.HuggingFaceEmbedding", return_value=MockDeterministicEmbedding())
    @patch("src.rag.orchestrator.BGECrossEncoderReranker")
    def test_orchestrator_indexing_persists_to_duckdb_and_no_docstore_json(
        self,
        mock_reranker: MagicMock,
        mock_embedding: MagicMock,
        storage_temp_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify indexing stores nodes into DuckDB and does not create docstore.json.

        Args:
            mock_reranker: Mocked reranker class.
            mock_embedding: Mocked embedding class.
            storage_temp_config: Path to test config.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        orchestrator = RAGOrchestrator(
            config_path=storage_temp_config,
            persist_dir=str(persist_dir),
            auto_load=True,
        )

        nodes = [
            TextNode(
                text="Integration test chunk 1",
                id_="node-int-1",
                metadata={"file_name": "data.txt", "file_hash": "hash_abc_123"},
            ),
            TextNode(
                text="Integration test chunk 2",
                id_="node-int-2",
                metadata={"file_name": "data.txt", "file_hash": "hash_abc_123"},
            ),
        ]

        orchestrator.index_nodes(nodes)

        # Verify docstore.duckdb exists and has the documents
        db_file = persist_dir / DEFAULT_DOCSTORE_DB_NAME
        assert db_file.exists()
        assert orchestrator._docstore is not None
        assert len(orchestrator._docstore.docs) == 2

        # Verify NO docstore.json exists anywhere
        assert not (persist_dir / "docstore.json").exists()
        assert not (persist_dir / "faiss_index" / "docstore.json").exists()

        # Check health status
        health = orchestrator.get_health_status()
        assert health["overall_status"] == "healthy"
        assert health["services"]["docstore"]["available"] is True
        assert health["services"]["docstore"]["type"] == "duckdb"
        assert health["services"]["docstore"]["status"] == "healthy"

        orchestrator.close()

    @patch("src.rag.orchestrator.HuggingFaceEmbedding", return_value=MockDeterministicEmbedding())
    @patch("src.rag.orchestrator.BGECrossEncoderReranker")
    def test_orchestrator_reload_from_existing_duckdb_docstore(
        self,
        mock_reranker: MagicMock,
        mock_embedding: MagicMock,
        storage_temp_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify an existing DuckDB docstore is correctly reloaded on orchestrator restart.

        Args:
            mock_reranker: Mocked reranker class.
            mock_embedding: Mocked embedding class.
            storage_temp_config: Path to test config.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        orchestrator1 = RAGOrchestrator(
            config_path=storage_temp_config,
            persist_dir=str(persist_dir),
            auto_load=True,
        )
        nodes = [
            TextNode(
                text="Reload test content",
                id_="reload-node-1",
                metadata={"file_name": "test.md", "file_hash": "test_hash"},
            ),
        ]
        orchestrator1.index_nodes(nodes)
        orchestrator1.close()

        # Initialize fresh orchestrator instance pointing to the same persist dir
        orchestrator2 = RAGOrchestrator(
            config_path=storage_temp_config,
            persist_dir=str(persist_dir),
            auto_load=True,
        )
        assert orchestrator2._docstore is not None
        assert len(orchestrator2._docstore.docs) == 1
        assert "reload-node-1" in orchestrator2._docstore.docs
        assert orchestrator2._docstore.docs["reload-node-1"].get_content() == "Reload test content"
        orchestrator2.close()

    @patch("src.rag.orchestrator.HuggingFaceEmbedding", return_value=MockDeterministicEmbedding())
    @patch("src.rag.orchestrator.BGECrossEncoderReranker")
    def test_orchestrator_migrates_legacy_docstore_json(
        self,
        mock_reranker: MagicMock,
        mock_embedding: MagicMock,
        storage_temp_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify migration from legacy docstore.json into DuckDBKVStore on initial startup.

        Args:
            mock_reranker: Mocked reranker class.
            mock_embedding: Mocked embedding class.
            storage_temp_config: Path to test config.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        faiss_dir = persist_dir / "faiss_index"
        faiss_dir.mkdir(parents=True, exist_ok=True)

        # Pre-populate a legacy SimpleDocumentStore at faiss_index/docstore.json
        legacy_ds = SimpleDocumentStore()
        legacy_nodes = [
            TextNode(
                text="Legacy migrated content 1",
                id_="legacy-node-1",
                metadata={"file_name": "legacy.txt", "file_hash": "legacy_hash_1"},
            ),
            TextNode(
                text="Legacy migrated content 2",
                id_="legacy-node-2",
                metadata={"file_name": "legacy.txt", "file_hash": "legacy_hash_1"},
            ),
        ]
        legacy_ds.add_documents(legacy_nodes)
        legacy_ds.persist(str(faiss_dir / "docstore.json"))
        assert (faiss_dir / "docstore.json").exists()

        # Initialize RAGOrchestrator with DuckDB config
        orchestrator = RAGOrchestrator(
            config_path=storage_temp_config,
            persist_dir=str(persist_dir),
            auto_load=True,
        )

        # Documents should be migrated to DuckDB docstore
        assert orchestrator._docstore is not None
        assert len(orchestrator._docstore.docs) == 2
        assert "legacy-node-1" in orchestrator._docstore.docs
        assert "legacy-node-2" in orchestrator._docstore.docs

        # Hashes should be detectable from migrated docstore
        hashes = orchestrator.get_indexed_file_hashes()
        assert "legacy_hash_1" in hashes

        orchestrator.close()

    @patch("src.rag.orchestrator.HuggingFaceEmbedding", return_value=MockDeterministicEmbedding())
    @patch("src.rag.orchestrator.BGECrossEncoderReranker")
    def test_orchestrator_snapshot_creates_duckdb_in_snapshot_dir(
        self,
        mock_reranker: MagicMock,
        mock_embedding: MagicMock,
        storage_temp_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify create_snapshot properly copies docstore.duckdb and logs in manifest.json.

        Args:
            mock_reranker: Mocked reranker class.
            mock_embedding: Mocked embedding class.
            storage_temp_config: Path to test config.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        orchestrator = RAGOrchestrator(
            config_path=storage_temp_config,
            persist_dir=str(persist_dir),
            auto_load=True,
        )

        nodes = [
            TextNode(
                text="Snapshot test chunk",
                id_="snap-node-1",
                metadata={"file_name": "doc.md", "file_hash": "snap_hash"},
            ),
        ]
        orchestrator.index_nodes(nodes)

        snapshot_dir = orchestrator.create_snapshot(tmp_path / "snapshots" / "v1.0.0")
        assert snapshot_dir.exists()

        snapshot_duckdb = snapshot_dir / DEFAULT_DOCSTORE_DB_NAME
        assert snapshot_duckdb.exists()
        assert snapshot_duckdb.stat().st_size > 0

        manifest_file = snapshot_dir / "manifest.json"
        assert manifest_file.exists()
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert manifest_data["version"] == "1.0"
        assert DEFAULT_DOCSTORE_DB_NAME in manifest_data["files"]

        orchestrator.close()

    @patch("src.rag.orchestrator.HuggingFaceEmbedding", return_value=MockDeterministicEmbedding())
    @patch("src.rag.orchestrator.BGECrossEncoderReranker")
    def test_orchestrator_close_releases_duckdb_lock(
        self,
        mock_reranker: MagicMock,
        mock_embedding: MagicMock,
        storage_temp_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify orchestrator.close() releases DuckDB connection and locks on Windows.

        Args:
            mock_reranker: Mocked reranker class.
            mock_embedding: Mocked embedding class.
            storage_temp_config: Path to test config.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        orchestrator = RAGOrchestrator(
            config_path=storage_temp_config,
            persist_dir=str(persist_dir),
            auto_load=True,
        )
        nodes = [TextNode(text="Lock test", id_="l1", metadata={"file_name": "l.txt", "file_hash": "h1"})]
        orchestrator.index_nodes(nodes)

        orchestrator.close()

        # Attempt reading bytes and copying the file
        db_path = persist_dir / DEFAULT_DOCSTORE_DB_NAME
        raw_bytes = db_path.read_bytes()
        assert len(raw_bytes) > 0

        copy_path = tmp_path / "backup.duckdb"
        shutil.copy2(db_path, copy_path)
        assert copy_path.exists()

    @patch("src.rag.orchestrator.HuggingFaceEmbedding", return_value=MockDeterministicEmbedding())
    @patch("src.rag.orchestrator.BGECrossEncoderReranker")
    def test_orchestrator_queries_and_searches_with_duckdb(
        self,
        mock_reranker: MagicMock,
        mock_embedding: MagicMock,
        storage_temp_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify query and search_documents succeed with DuckDB KVStore without full scans.

        Args:
            mock_reranker: Mocked reranker class.
            mock_embedding: Mocked embedding class.
            storage_temp_config: Path to test config.
            tmp_path: Temporary directory fixture.
        """
        async def mock_apostprocess(nodes: Any, query_bundle: Any = None) -> Any:
            return nodes

        mock_reranker.return_value.postprocess_nodes.side_effect = lambda nodes, query_bundle=None: nodes
        mock_reranker.return_value.apostprocess_nodes.side_effect = mock_apostprocess
        persist_dir = tmp_path / "persist"
        orchestrator = RAGOrchestrator(
            config_path=storage_temp_config,
            persist_dir=str(persist_dir),
            auto_load=True,
        )
        nodes = [
            TextNode(
                text="LockBit ransomware analysis and IOCs",
                id_="node-malware-1",
                metadata={"file_name": "malware.txt", "file_hash": "hash_malware_1"},
            ),
            TextNode(
                text="APT29 phishing infrastructure and TTPs",
                id_="node-phishing-2",
                metadata={"file_name": "threats.txt", "file_hash": "hash_threats_2"},
            ),
        ]
        orchestrator.index_nodes(nodes)

        # Synchronous search_documents
        search_res = orchestrator.search_documents("LockBit", top_k=5, search_type="semantic")
        assert len(search_res) > 0
        node_ids = [r["id"] for r in search_res]
        assert "node-malware-1" in node_ids or "node-phishing-2" in node_ids

        # Synchronous query (which wraps query_async)
        query_payload = orchestrator.query("APT29", top_k=5, search_type="semantic")
        assert query_payload is not None
        assert len(query_payload.context) > 0

        orchestrator.close()

    @patch("src.rag.orchestrator.HuggingFaceEmbedding", return_value=MockDeterministicEmbedding())
    @patch("src.rag.orchestrator.BGECrossEncoderReranker")
    def test_create_snapshot_external_docstore_path(
        self,
        mock_reranker: MagicMock,
        mock_embedding: MagicMock,
        storage_temp_config: str,
        tmp_path: Path,
    ) -> None:
        """Verify create_snapshot properly copies docstore when docstore_path is external.

        Args:
            mock_reranker: Mocked reranker class.
            mock_embedding: Mocked embedding class.
            storage_temp_config: Path to test config.
            tmp_path: Temporary directory fixture.
        """
        persist_dir = tmp_path / "persist"
        external_dir = tmp_path / "external_docstore"
        external_dir.mkdir(parents=True, exist_ok=True)
        external_db = external_dir / "custom_docstore.duckdb"

        orchestrator = RAGOrchestrator(
            config_path=storage_temp_config,
            persist_dir=str(persist_dir),
            auto_load=True,
            docstore_path=external_db,
        )
        nodes = [
            TextNode(
                text="External docstore content",
                id_="ext-1",
                metadata={"file_name": "ext.md", "file_hash": "hash_ext"},
            ),
        ]
        orchestrator.index_nodes(nodes)
        assert external_db.exists()

        snapshot_dir = tmp_path / "snapshots" / "v2.0.0"
        created_snap = orchestrator.create_snapshot(snapshot_dir)
        assert created_snap.exists()

        # The external db should be copied into snapshot dir
        snap_db = created_snap / "custom_docstore.duckdb"
        assert snap_db.exists()
        assert snap_db.stat().st_size > 0

        manifest_file = created_snap / "manifest.json"
        assert manifest_file.exists()
        manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
        assert "custom_docstore.duckdb" in manifest_data["files"]

        orchestrator.close()

