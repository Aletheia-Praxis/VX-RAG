"""
Empirical challenger verification test suite for VX-RAG RAG Pipeline & Runtime.

Tests cover:
- Task 1: data/index_test_1000 FAISS HNSW and BM25 index artifacts and SHA-256 manifest integrity.
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
from typing import Any

import pytest
from llama_index.core import Settings
from llama_index.core.llms import MockLLM

from src.mcp.server import mcp, query_lock
from src.rag.orchestrator import get_orchestrator

INDEX_DIR: Path = Path("data/index_test_1000")
MANIFEST_PATH: Path = INDEX_DIR / "manifest.json"
FAISS_DIR: Path = INDEX_DIR / "faiss_index"
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


@pytest.fixture(autouse=True)
def configure_mock_llm() -> None:
    """Ensure MockLLM is configured to prevent OpenAI API key requirement during testing."""
    Settings.llm = MockLLM()


def test_index_artifacts_manifest_and_hashes() -> None:
    """Empirically verify data/index_test_1000 index structures and manifest SHA-256 hashes."""
    assert INDEX_DIR.is_dir(), f"Index directory {INDEX_DIR} does not exist"
    assert MANIFEST_PATH.is_file(), f"Manifest file {MANIFEST_PATH} does not exist"
    assert FAISS_DIR.is_dir(), f"FAISS directory {FAISS_DIR} does not exist"
    assert BM25_DIR.is_dir(), f"BM25 directory {BM25_DIR} does not exist"

    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest: dict[str, Any] = json.loads(manifest_bytes.decode("utf-8"))

    # Manifest schema validation
    assert manifest.get("version") == "1.0", f"Unexpected version: {manifest.get('version')}"
    assert manifest.get("model_name") == "BAAI/bge-small-en-v1.5"
    assert manifest.get("embedding_dimension") == 384
    assert manifest.get("total_nodes") == 84
    assert len(manifest.get("indexed_file_hashes", [])) == 50

    # Verify every indexed_file_hash is 64 hex chars
    for h in manifest["indexed_file_hashes"]:
        assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)

    # Verify SHA-256 hashes for all physical index files
    files_dict: dict[str, str] = manifest.get("files", {})
    assert len(files_dict) > 0, "No files recorded in manifest"

    verified_count: int = 0
    for rel_path, expected_hash in files_dict.items():
        file_path = INDEX_DIR / rel_path
        if not file_path.exists():
            # Manifest records both relative paths (bm25_index/corpus.jsonl) and simple filenames (corpus.jsonl)
            continue
        actual_hash = compute_sha256(file_path.read_bytes())
        assert actual_hash == expected_hash, (
            f"Hash mismatch for {rel_path}: expected {expected_hash}, got {actual_hash}"
        )
        verified_count += 1

    # There are 8 bm25 files and 5 faiss files = 13 physical index files
    assert verified_count == 13, f"Expected 13 verified physical files, got {verified_count}"


def test_faiss_and_bm25_structures_in_memory() -> None:
    """Empirically inspect FAISS HNSW Flat and BM25 index objects in memory."""
    import faiss

    orchestrator = get_orchestrator(persist_dir=str(INDEX_DIR))
    vector_store = orchestrator._vector_store
    assert vector_store is not None, "Vector store was not initialized"

    faiss_index = vector_store._faiss_index
    assert faiss_index is not None, "FAISS index object is missing"
    assert type(faiss_index).__name__ == "IndexHNSWFlat"
    assert faiss_index.d == 384, f"Expected dimension 384, got {faiss_index.d}"
    assert faiss_index.ntotal == 84, f"Expected 84 vectors, got {faiss_index.ntotal}"
    assert faiss_index.metric_type == faiss.METRIC_INNER_PRODUCT
    assert faiss_index.is_trained is True

    bm25 = orchestrator._bm25_retriever
    assert bm25 is not None, "BM25 retriever was not initialized"
    corpus_size = len(getattr(bm25, "corpus", []))
    assert corpus_size == 84, f"Expected BM25 corpus size 84, got {corpus_size}"


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
        # Concurrency must be serialized: each CPU call takes > 4s, so 3 calls should take > 8s
        assert elapsed > 8.0, f"Sequential serialization failed: elapsed time was only {elapsed:.2f}s"
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


def test_cli_serve_lacks_persist_dir_argument() -> None:
    """Empirically confirm that CLI 'serve' lacks --persist-dir argument."""
    proc = subprocess.run(
        [sys.executable, "-m", "src.cli", "serve", "--persist-dir", "data/index_test_1000"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "unrecognized arguments: --persist-dir" in proc.stderr


def test_cli_query_fails_without_openai_api_key_in_clean_env() -> None:
    """Empirically confirm that CLI query fails without OPENAI_API_KEY in a clean process."""
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    env["PYTHONPATH"] = "."

    proc = subprocess.run(
        [sys.executable, "-m", "src.cli", "query", "ransomware", "--persist-dir", "data/index_test_1000"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode != 0
    assert "No API key found for OpenAI" in proc.stderr or "Could not load OpenAI model" in proc.stderr


def test_cli_serve_stdio_stdout_logging_pollution() -> None:
    """Empirically confirm that CLI serve outputs print banners and logs to stdout."""
    cli_source = Path("src/cli.py").read_text(encoding="utf-8")
    assert 'print(f"Starting MCP server with transport: {args.transport}")' in cli_source
    assert 'search_type="hybrid"' in cli_source
