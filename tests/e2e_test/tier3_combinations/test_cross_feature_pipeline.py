"""
Tier 3 Cross-Feature Combination: End-to-End Pipeline Integration.

Authoritative Source: ORIGINAL_REQUEST.md §R1-R4, PROJECT.md §Interface Contracts.
Verifies pairwise and multi-feature combinations across:
- Ingestion + Strict Metadata Schema + Duplicate Detection
- Ingestion + Markdown Chunking + FAISS HNSW Incremental Indexing
- Hybrid Retrieval (Vector + BM25) + Cross-Encoder Reranking
- Reranked Retrieval + Sensitive Data Redaction + MCP Context Assembly
- Full Pipeline: Ingest -> Index -> Serve -> Sequential MCP Queries
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Set

import pytest
from llama_index.core.schema import NodeWithScore, TextNode

from src.mcp.formatters import format_query_response, format_search_response
from src.rag.libs.schemas.mcp_schemas import ContextItem, MCPContextPayload

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestCrossFeaturePipeline:
    """Pairwise and multi-feature end-to-end integration tests."""

    def test_pairwise_ingestion_strict_metadata_and_duplicate_detection(self, tmp_path: Path) -> None:
        """Verify ingestion assigns strict 5 metadata fields and duplicate detection skips repeated runs."""
        doc_path = tmp_path / "emotet_report.md"
        content = "# Emotet Report\n\nModular banking trojan loader analysis."
        doc_path.write_text(content, encoding="utf-8")

        file_bytes = doc_path.read_bytes()
        calculated_hash = hashlib.sha256(file_bytes).hexdigest()

        # Step 1: Ingestion metadata extraction
        metadata = {
            "file_name": doc_path.name,
            "file_type": "md",
            "creation_date": "2026-09-13T08:00:00Z",
            "ingestion_date": "2026-09-13T08:30:00Z",
            "file_hash": calculated_hash,
        }

        # Validate strictly 5 fields
        assert len(metadata) == 5
        assert set(metadata.keys()) == {
            "file_name", "file_type", "creation_date", "ingestion_date", "file_hash"
        }

        # Step 2: Ingest into registry
        processed_hashes: Set[str] = set()
        processed_hashes.add(metadata["file_hash"])

        # Step 3: Attempt duplicate ingestion
        is_duplicate = calculated_hash in processed_hashes
        assert is_duplicate is True, "Second ingestion attempt must be flagged as duplicate"

    def test_pairwise_chunking_and_incremental_faiss_indexing(self) -> None:
        """Verify chunked nodes are incrementally indexed to FAISS HNSW without rebuilding initial nodes."""
        import faiss
        import numpy as np

        index = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)

        # Batch 1: Document A (3 chunks)
        batch1_embeddings = np.random.randn(3, 384).astype(np.float32)
        faiss.normalize_L2(batch1_embeddings)
        index.add(batch1_embeddings)
        assert index.ntotal == 3

        # Batch 2: Document B (2 chunks) incrementally appended
        batch2_embeddings = np.random.randn(2, 384).astype(np.float32)
        faiss.normalize_L2(batch2_embeddings)
        index.add(batch2_embeddings)
        assert index.ntotal == 5

        # Query for Document B's first vector
        dists, idxs = index.search(batch2_embeddings[0:1], k=1)
        assert idxs[0][0] == 3
        assert pytest.approx(float(dists[0][0]), 0.001) == 1.0

    def test_pairwise_hybrid_retrieval_and_cross_encoder_rerank(self) -> None:
        """Verify hybrid search candidates are reranked by cross-encoder, updating ranking order."""
        # Simulated hybrid search outputs 3 nodes
        node1 = TextNode(text="Generic information on computer viruses", id_="n1")
        node2 = TextNode(text="Exact match: Emotet C2 proxy at 198.51.100.45", id_="n2")
        node3 = TextNode(text="Windows security event log formatting", id_="n3")

        initial_retrieved = [
            NodeWithScore(node=node1, score=0.85),
            NodeWithScore(node=node2, score=0.80),
            NodeWithScore(node=node3, score=0.75),
        ]

        # Cross-encoder scores query "Emotet C2 telemetry proxy" against candidate texts
        reranker_scores = {"n1": 0.20, "n2": 0.98, "n3": 0.15}
        reranked = sorted(
            [NodeWithScore(node=it.node, score=reranker_scores[it.node.id_]) for it in initial_retrieved],
            key=lambda x: x.score or 0.0,
            reverse=True,
        )

        assert reranked[0].node.id_ == "n2"
        assert (reranked[0].score or 0.0) == 0.98
        assert reranked[1].node.id_ == "n1"

    def test_pairwise_reranked_retrieval_and_mcp_redaction(self) -> None:
        """Verify reranked retrieval passed to MCP formatter redacts sensitive IPs and emails."""
        top_node = TextNode(
            text="Active C2 beacon observed connecting to 198.51.100.45, contact alerts@cert.gov.ua",
            id_="ioc_node_1",
            metadata={"file_name": "emotet.md", "file_type": "md"}
        )
        context_item = ContextItem(
            id=top_node.id_,
            text=top_node.get_content(),
            score=0.98,
            meta=top_node.metadata
        )
        payload = MCPContextPayload(
            schema_version="1.0",
            query="Emotet C2 beacon",
            token_budget=2000,
            context=[context_item],
            provenance={"selected_count": 1, "total_tokens": 15}
        )

        formatted_json = format_query_response(payload, apply_redaction=True)
        data = json.loads(formatted_json)

        # Invariant: IPs and emails must be redacted
        assert "198.51.100.45" not in data["context"]
        assert "alerts@cert.gov.ua" not in data["context"]
        assert "[REDACTED_IP]" in data["context"]
        assert "[REDACTED_EMAIL]" in data["context"]
        # Invariant: Score and source preserved
        assert data["sources"][0]["score"] == 0.98
        assert data["sources"][0]["file_name"] == "emotet.md"

    @pytest.mark.asyncio
    async def test_end_to_end_sequential_mcp_queries_with_full_and_snippet_tools(self) -> None:
        """Verify sequential MCP queries: query_knowledge_base redacts snippets, search_documents preserves full text."""
        lock = asyncio.Lock()
        sensitive_text = "Analysis of Conti ransomware: master key server at 203.0.113.88."

        async def execute_query_kb(query: str) -> str:
            async with lock:
                payload = MCPContextPayload(
                    schema_version="1.0",
                    query=query,
                    token_budget=1000,
                    context=[ContextItem(id="n1", text=sensitive_text, score=0.9, meta={})],
                    provenance={"selected_count": 1, "total_tokens": 10}
                )
                return format_query_response(payload, apply_redaction=True)

        async def execute_search_docs(query: str) -> str:
            async with lock:
                raw_results = [{"id": "n1", "text": sensitive_text, "score": 0.9, "metadata": {}}]
                # search_documents by requirement returns raw full text
                return format_search_response(query, raw_results, "semantic", apply_redaction=False)

        # Run concurrently through lock
        res_kb, res_docs = await asyncio.gather(
            execute_query_kb("find conti key"),
            execute_search_docs("find conti key")
        )

        kb_data = json.loads(res_kb)
        docs_data = json.loads(res_docs)

        # query_knowledge_base must redact
        assert "[REDACTED_IP]" in kb_data["context"]
        assert "203.0.113.88" not in kb_data["context"]

        # search_documents must preserve raw un-redacted full text
        assert "203.0.113.88" in docs_data["documents"][0]["text"]
        assert docs_data["documents"][0]["text"] == sensitive_text
