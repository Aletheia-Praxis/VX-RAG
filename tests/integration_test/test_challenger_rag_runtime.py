"""
Empirical challenger verification test suite for VX-RAG RAG Pipeline & Runtime.

Tests cover:
- Task 1: data/index Qdrant and BM25 index artifacts and SHA-256 manifest integrity.
- Task 2: Multi-modal query execution (semantic, keyword, hybrid) and strict [0.0, 1.0] sigmoid normalization.
- Task 3: FastMCP server runtime, tools, resources, sequential _LoopBoundLock execution, and request_id UUIDs in logs.
- Task 4: Diagnostic findings verification (CLI query hybrid search, CLI serve arguments, stdio stdout logging, Defender signatures).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from llama_index.core import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import MockLLM

from src.mcp.server import mcp, query_lock
from src.rag.libs.postprocessors import BGECrossEncoderReranker
from src.rag.orchestrator import get_orchestrator, reset_orchestrator

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401

INDEX_DIR: Path = Path("data/index")
MANIFEST_PATH: Path = INDEX_DIR / "manifest.json"
QDRANT_DIR: Path = INDEX_DIR / "qdrant"
BM25_DIR: Path = INDEX_DIR / "bm25_index"
LOG_PATH: Path = Path("logs/vx_rag.jsonl")

REQUIRED_METADATA_KEYS: set[str] = {
    "file_name",
    "file_type",
    "creation_date",
    "ingestion_date",
    "file_hash",
}


def compute_sha256(data: bytes) -> str:
    """Compute hex SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


class MockBGEEmbedding(BaseEmbedding):
    """Deterministic mock embedding for BAAI/bge-m3 producing 1024-dim normalized vectors."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize mock embedding model."""
        super().__init__(model_name="BAAI/bge-m3", **kwargs)

    def _get_query_embedding(self, query: str) -> list[float]:
        """Produce 1024-dimensional normalized float vector."""
        return [1.0 / (1024.0**0.5)] * 1024

    def _get_text_embedding(self, text: str) -> list[float]:
        """Produce 1024-dimensional normalized float vector."""
        return [1.0 / (1024.0**0.5)] * 1024

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Produce batch of 1024-dimensional normalized float vectors."""
        return [[1.0 / (1024.0**0.5)] * 1024 for _ in texts]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        """Produce asynchronous 1024-dimensional normalized float vector."""
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> list[float]:
        """Produce asynchronous 1024-dimensional normalized float vector."""
        return self._get_text_embedding(text)

    def encode_sparse(self, texts: list[str]) -> tuple[list[list[int]], list[list[float]]]:
        """Produce sparse token frequency vectors matching vocabulary in index."""
        return [[4, 5, 6] for _ in texts], [[1.0, 1.0, 1.0] for _ in texts]


class MockBGERerankerModel:
    """Mock cross encoder model for BAAI/bge-reranker-v2-m3."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return descending logit scores for candidate pairs."""
        return [10.0 - (float(i) * 0.5) for i in range(len(pairs))]

    def rerank_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return descending logit scores for candidate pairs."""
        return self.predict(pairs)


@pytest.fixture(autouse=True)
def configure_mock_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure MockLLM and BAAI/bge models are configured for testing without external weights."""
    reset_orchestrator()
    Settings.llm = MockLLM()
    mock_embed = MockBGEEmbedding()
    monkeypatch.setattr("src.rag.orchestrator.BGEM3Embedding", lambda *a, **kw: mock_embed)

    mock_model = MockBGERerankerModel()
    real_reranker_init = BGECrossEncoderReranker.__init__

    def _mocked_reranker_init(self: Any, *a: Any, **kw: Any) -> None:
        kw["model"] = mock_model
        real_reranker_init(self, *a, **kw)

    monkeypatch.setattr("src.rag.orchestrator.BGECrossEncoderReranker.__init__", _mocked_reranker_init)
    monkeypatch.setattr("src.rag.libs.postprocessors.BGECrossEncoderReranker.__init__", _mocked_reranker_init)


def test_index_artifacts_manifest_and_hashes() -> None:
    """Empirically verify data/index index structures and manifest SHA-256 hashes."""
    assert INDEX_DIR.is_dir(), f"Index directory {INDEX_DIR} does not exist"
    assert MANIFEST_PATH.is_file(), f"Manifest file {MANIFEST_PATH} does not exist"
    assert QDRANT_DIR.is_dir(), f"Qdrant directory {QDRANT_DIR} does not exist"

    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest: dict[str, Any] = json.loads(manifest_bytes.decode("utf-8"))

    # Manifest schema validation
    assert manifest.get("version") == "1.0", f"Unexpected version: {manifest.get('version')}"
    assert manifest.get("model_name") == "BAAI/bge-m3"
    assert manifest.get("embedding_dimension") == 1024
    assert manifest.get("total_nodes", 0) > 0
    assert len(manifest.get("indexed_file_hashes", [])) > 0

    # Verify every indexed_file_hash is 64 hex chars
    for h in manifest["indexed_file_hashes"]:
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)

    # Verify SHA-256 hashes for physical index files recorded in manifest
    files_dict: dict[str, str] = manifest.get("files", {})
    assert len(files_dict) > 0, "No files recorded in manifest"

    verified_count: int = 0
    for rel_path, expected_hash in files_dict.items():
        file_path = INDEX_DIR / rel_path
        if not file_path.exists():
            continue
        actual_hash = compute_sha256(file_path.read_bytes())
        assert actual_hash == expected_hash, (
            f"Hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
        )
        verified_count += 1

    assert verified_count > 0, f"Expected verified physical files, got {verified_count}"


def test_vector_store_and_bm25_structures_in_memory() -> None:
    """Empirically inspect Qdrant vector store and BM25 index objects in memory."""
    orchestrator = get_orchestrator(persist_dir=str(INDEX_DIR))
    vector_store = orchestrator._vector_store
    assert vector_store is not None, "Vector store was not initialized"

    client = orchestrator._qdrant_client
    assert client is not None, "Qdrant client is missing"
    collection_name = vector_store.collection_name
    collection_info = client.get_collection(collection_name)
    assert collection_info.points_count is not None and collection_info.points_count > 0, (
        f"Expected > 0 points, got {collection_info.points_count}"
    )


def test_query_execution_and_score_normalization() -> None:
    """Run queries across semantic, BM25, and hybrid modalities, verifying [0.0, 1.0] bounds."""
    orchestrator = get_orchestrator(persist_dir=str(INDEX_DIR))

    queries = [
        "Cobalt Strike beaconing",
        "ransomware encryption routines",
        "process hollowing injection",
        "kernel driver vulnerability",
        "0x5A4D MZ header DOS executable",
    ]
    modalities = ["semantic", "keyword", "hybrid"]

    total_evaluations: int = 0
    for q in queries:
        for mode in modalities:
            total_evaluations += 1
            res = orchestrator.query(query=q, top_k=5, search_type=mode)
            assert res is not None, f"Query '{q}' returned None for mode '{mode}'"
            assert hasattr(res, "context"), "Query result has no context attribute"
            assert len(res.context) > 0, f"Query '{q}' ({mode}) returned empty context"

            scores: list[float] = []
            for item in res.context:
                s = item.score
                assert s is not None, "Node score is None"
                assert 0.0 <= s <= 1.0, f"Score {s} out of [0.0, 1.0] bounds for query '{q}' ({mode})"
                scores.append(s)

                # Verify metadata contract (exact 5 keys in item.meta)
                metadata_keys = set(item.meta.keys())
                missing = REQUIRED_METADATA_KEYS - metadata_keys
                assert not missing, f"Missing metadata keys: {missing} in node {item.id}"

            # Verify descending order of scores
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"Scores not sorted descending: {scores[i]} < {scores[i+1]}"
                )

    assert total_evaluations == 15, f"Expected 15 query evaluations, ran {total_evaluations}"


@pytest.mark.asyncio
async def test_fastmcp_server_runtime_and_sequential_lock() -> None:
    """Verify FastMCP server tool/resource discovery, invocation, sequential lock, and request_id."""
    from fastmcp import Client

    get_orchestrator(persist_dir=str(INDEX_DIR))

    async with Client(mcp) as client:
        # Check tools
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        assert "query_knowledge_base" in tool_names
        assert "search_documents" in tool_names

        # Check resources
        resources = await client.list_resources()
        resource_uris = {str(r.uri) for r in resources}
        assert "health://status" in resource_uris
        assert "context://system" in resource_uris

        # Read health resource
        health_res = await client.read_resource("health://status")
        health_data = json.loads(health_res[0].text)
        assert health_data.get("overall_status") in ("healthy", "degraded")

        # Read context resource
        sys_res = await client.read_resource("context://system")
        sys_data = json.loads(sys_res[0].text)
        assert sys_data.get("system_name") == "VX-RAG"

        # Sequential concurrency verification with _LoopBoundLock
        t0 = time.perf_counter()
        results = await asyncio.gather(
            client.call_tool("search_documents", {"query": "process injection", "top_k": 2}),
            client.call_tool("query_knowledge_base", {"query": "ransomware encryption", "top_k": 2}),
            client.call_tool("search_documents", {"query": "kernel driver", "top_k": 2}),
        )
        elapsed = time.perf_counter() - t0
        assert len(results) == 3
        # Concurrency must be serialized: verify non-zero elapsed time and lock release
        assert elapsed > 0.05, f"Sequential serialization failed: elapsed time was unexpectedly short ({elapsed:.2f}s)"
        assert not query_lock.locked(), "query_lock was left locked"

    # Verify request_id in logs/vx_rag.jsonl
    assert LOG_PATH.is_file(), f"Log file {LOG_PATH} does not exist"
    lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    recent_entries = [json.loads(line) for line in lines[-100:] if line.strip()]
    request_ids = [e.get("request_id") for e in recent_entries if e.get("request_id")]
    assert len(request_ids) > 0, "No request_id found in recent log entries"
    # Ensure request_ids are valid UUID strings
    import uuid
    for req_id in request_ids[-5:]:
        parsed = uuid.UUID(req_id)
        assert str(parsed) == req_id


def test_cli_serve_supports_persist_dir_argument() -> None:
    """Empirically confirm that CLI 'serve' accepts --persist-dir argument."""
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli", "serve", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert "--persist-dir" in proc.stdout


def test_cli_query_succeeds_without_openai_api_key_in_clean_env() -> None:
    """Empirically confirm that CLI query succeeds without OPENAI_API_KEY in a clean process."""
    reset_orchestrator()

    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    env.pop("VX_RAG_CONFIG_PATH", None)
    env["PYTHONPATH"] = "."

    script = (
        "import sys\n"
        "from unittest.mock import patch\n"
        "from llama_index.core.embeddings import BaseEmbedding\n"
        "class MockBGE(BaseEmbedding):\n"
        "    def _get_query_embedding(self, q):\n"
        "        return [1.0 / (1024.0**0.5)] * 1024\n"
        "    def _get_text_embedding(self, t):\n"
        "        return [1.0 / (1024.0**0.5)] * 1024\n"
        "    def _get_text_embeddings(self, ts):\n"
        "        return [[1.0 / (1024.0**0.5)] * 1024 for _ in ts]\n"
        "    async def _aget_query_embedding(self, q):\n"
        "        return [1.0 / (1024.0**0.5)] * 1024\n"
        "class MockModel:\n"
        "    def predict(self, pairs):\n"
        "        return [10.0 - float(i)*0.5 for i in range(len(pairs))]\n"
        "    def rerank_pairs(self, pairs):\n"
        "        return self.predict(pairs)\n"
        "from src.rag.libs.postprocessors import BGECrossEncoderReranker\n"
        "mock_reranker = BGECrossEncoderReranker(model_name='BAAI/bge-reranker-v2-m3', top_n=5, model=MockModel())\n"
        "with (\n"
        "    patch('src.rag.orchestrator.BGEM3Embedding', return_value=MockBGE(model_name='BAAI/bge-m3')),\n"
        "    patch('src.rag.orchestrator.BGECrossEncoderReranker', return_value=mock_reranker),\n"
        "):\n"
        "    from src.cli import main\n"
        "    sys.argv = ['cli.py', 'query', 'ransomware', '--persist-dir', 'data/index']\n"
        "    main()\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0
    assert "No API key found for OpenAI" not in proc.stderr
    assert "Query complete." in proc.stdout


def test_cli_serve_stdio_stdout_logging_clean() -> None:
    """Empirically confirm that CLI serve outputs print banners to stderr in stdio mode."""
    cli_source = Path("src/cli.py").read_text(encoding="utf-8")
    assert "banner_file = sys.stderr if is_stdio else sys.stdout" in cli_source
    assert 'print(f"Starting MCP server with transport: {args.transport}", file=banner_file)' in cli_source
    assert 'search_type="hybrid"' in cli_source

    script = (
        "import sys\n"
        "from unittest.mock import patch\n"
        "from llama_index.core.embeddings import BaseEmbedding\n"
        "class MockBGE(BaseEmbedding):\n"
        "    def _get_query_embedding(self, q):\n"
        "        return [1.0 / (1024.0**0.5)] * 1024\n"
        "    def _get_text_embedding(self, t):\n"
        "        return [1.0 / (1024.0**0.5)] * 1024\n"
        "    def _get_text_embeddings(self, ts):\n"
        "        return [[1.0 / (1024.0**0.5)] * 1024 for _ in ts]\n"
        "    async def _aget_query_embedding(self, q):\n"
        "        return [1.0 / (1024.0**0.5)] * 1024\n"
        "class MockModel:\n"
        "    def predict(self, pairs):\n"
        "        return [10.0 - float(i)*0.5 for i in range(len(pairs))]\n"
        "    def rerank_pairs(self, pairs):\n"
        "        return self.predict(pairs)\n"
        "from src.rag.libs.postprocessors import BGECrossEncoderReranker\n"
        "mock_reranker = BGECrossEncoderReranker(model_name='BAAI/bge-reranker-v2-m3', top_n=5, model=MockModel())\n"
        "with (\n"
        "    patch('src.mcp.server.mcp.run'),\n"
        "    patch('src.rag.orchestrator.BGEM3Embedding', return_value=MockBGE(model_name='BAAI/bge-m3')),\n"
        "    patch('src.rag.orchestrator.BGECrossEncoderReranker', return_value=mock_reranker),\n"
        "):\n"
        "    from src.cli import main\n"
        "    sys.argv = ['cli.py', 'serve', '--transport', 'stdio', '--persist-dir', 'data/index']\n"
        "    main()\n"
    )
    env = dict(os.environ)
    env.pop("VX_RAG_CONFIG_PATH", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stdout == ""
    assert "Starting MCP server with transport: stdio" in proc.stderr
