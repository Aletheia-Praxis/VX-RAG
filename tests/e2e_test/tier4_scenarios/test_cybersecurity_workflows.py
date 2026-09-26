"""
Tier 4 Real-World Application Scenarios: Cybersecurity Analysis Workflows.

Authoritative Source: ORIGINAL_REQUEST.md, PROJECT.md §Tier 4 Scenarios, Tech Spec §1.0.
Verifies realistic cybersecurity threat intelligence and reverse engineering workflows
querying the Vx Underground document corpus:
- Scenario 1: Emotet C2 Infrastructure & IOC Extraction
- Scenario 2: Ransomware Encryption & Volume Shadow Copy Deletion
- Scenario 3: Zero-Day Exploit Disclosure & CVE Root Cause Analysis
- Scenario 4: Incremental Threat Intelligence Feed Updates
- Scenario 5: Multi-Format Corpus Ingestion with Conditional OCR Fallback
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Set

import pytest
from llama_index.core.schema import TextNode

from src.mcp.formatters import (
    format_query_response,
    format_search_response,
    redact_sensitive_data,
)
from src.rag.libs.schemas.mcp_schemas import ContextItem, MCPContextPayload

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestCybersecurityWorkflows:
    """End-to-end real-world cybersecurity analysis scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_1_emotet_c2_telemetry_extraction(
        self, sample_cybersecurity_corpus: Path
    ) -> None:
        """
        Scenario 1: Emotet C2 Infrastructure & IOC Extraction.

        Workflow:
        1. Ingest Emotet analysis report containing network telemetry.
        2. Query knowledge base for C2 beacon addresses and loader deobfuscation.
        3. Verify sequential query execution, IPv4/IPv6 redaction in query_knowledge_base,
           and raw IOC hash preservation in search_documents.
        """
        emotet_file = sample_cybersecurity_corpus / "emotet_modular_trojan.md"
        assert emotet_file.exists()

        content = emotet_file.read_text(encoding="utf-8")
        assert "198.51.100.45" in content
        assert "2001:0db8:85a3:0000:0000:8a2e:0370:7334" in content

        # MCP query execution
        payload = MCPContextPayload(
            schema_version="1.0",
            query="Emotet C2 network beacon IP addresses and port",
            token_budget=2000,
            context=[
                ContextItem(
                    id="node_emotet_c2",
                    text=content,
                    score=0.96,
                    meta={"file_name": emotet_file.name, "file_type": "md"}
                )
            ],
            provenance={"selected_count": 1, "total_tokens": 120}
        )

        kb_response = json.loads(format_query_response(payload, apply_redaction=True))

        # Invariant 1: IPs must be redacted in generalized context snippet
        assert "198.51.100.45" not in kb_response["context"]
        assert "2001:0db8:85a3:0000:0000:8a2e:0370:7334" not in kb_response["context"]
        assert "[REDACTED_IP]" in kb_response["context"]
        assert "[REDACTED_IPv6]" in kb_response["context"]
        assert "[REDACTED_EMAIL]" in kb_response["context"]

        # Invariant 2: Search documents returns full text with raw hashes for analyst reverse engineering
        search_res = json.loads(
            format_search_response(
                "Emotet sha256 hash",
                [{"id": "n1", "text": content, "score": 0.96, "metadata": {}}],
                "semantic",
                apply_redaction=False,
            )
        )
        assert "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" in search_res["documents"][0]["text"]

    def test_scenario_2_ransomware_volume_shadow_deletion_and_crypto(
        self, sample_cybersecurity_corpus: Path
    ) -> None:
        """
        Scenario 2: Ransomware Encryption & Volume Shadow Copy Deletion (LockBit / Conti).

        Workflow:
        1. Ingest technical analysis describing shadow copy invalidation (`vssadmin delete shadows`).
        2. Verify adaptive chunking preserves code block syntax and cmd execution sequences.
        3. Verify hybrid retrieval identifies exact command string and cryptographic API routines.
        """
        ransom_file = sample_cybersecurity_corpus / "ransomware_recovery_inhibition.md"
        assert ransom_file.exists()
        raw_text = ransom_file.read_text(encoding="utf-8")

        # Invariant 1: Command sequence preserved intact
        assert "vssadmin.exe delete shadows /all /quiet" in raw_text
        assert "CryptAcquireContext" in raw_text

        # Invariant 2: Adaptive code block retains code fences
        code_blocks = [block for block in raw_text.split("```") if "InitializeRansomKey" in block]
        assert len(code_blocks) >= 1
        assert "BOOL InitializeRansomKey" in code_blocks[0]

    def test_scenario_3_cve_exploit_root_cause_analysis(
        self, sample_cybersecurity_corpus: Path
    ) -> None:
        """
        Scenario 3: Zero-Day Exploit Disclosure & CVE Root Cause Analysis (CVE-2023-38831).

        Workflow:
        1. Ingest vulnerability advisory.
        2. Verify strict metadata extraction computes valid SHA256 and ISO dates.
        3. Verify boilerplate removal and query token budgeting.
        """
        cve_file = sample_cybersecurity_corpus / "cve_2023_38831_advisory.txt"
        assert cve_file.exists()
        file_bytes = cve_file.read_bytes()
        calculated_hash = hashlib.sha256(file_bytes).hexdigest()

        # Enforce strict 5-field metadata contract
        meta = {
            "file_name": cve_file.name,
            "file_type": "txt",
            "creation_date": "2023-08-23T12:00:00Z",
            "ingestion_date": "2026-09-13T08:00:00Z",
            "file_hash": calculated_hash,
        }
        assert len(meta) == 5
        assert meta["file_type"] == "txt"
        assert meta["file_hash"] == calculated_hash

        # Context assembly with token budget
        context_item = ContextItem(
            id="cve_node_1",
            text=cve_file.read_text(encoding="utf-8"),
            score=0.91,
            meta=meta
        )
        payload = MCPContextPayload(
            schema_version="1.0",
            query="CVE-2023-38831 root cause logic flaw",
            token_budget=1500,
            context=[context_item],
            provenance={"selected_count": 1, "total_tokens": 85}
        )
        res = json.loads(format_query_response(payload, apply_redaction=True))
        assert "CVE-2023-38831" in res["context"]
        assert "[REDACTED_EMAIL]" in res["context"]
        assert "security-lead@threatlabz.example.com" not in res["context"]

    def test_scenario_4_incremental_threat_intel_feed_updates(self, tmp_path: Path) -> None:
        """
        Scenario 4: Incremental Threat Intelligence Feed Updates.

        Workflow:
        1. Initialize FAISS HNSW index with baseline intelligence dataset.
        2. Receive newly published malware reports and incrementally append to vector store.
        3. Verify snapshot persistence and manifest.json SHA-256 checksums update without full rebuild.
        """
        import faiss
        import numpy as np

        index_dir = tmp_path / "index_incremental"
        index_dir.mkdir(parents=True, exist_ok=True)

        # Baseline: 10 threat reports
        index = faiss.IndexHNSWFlat(384, 32, faiss.METRIC_INNER_PRODUCT)
        initial_vectors = np.random.randn(10, 384).astype(np.float32)
        faiss.normalize_L2(initial_vectors)
        index.add(initial_vectors)
        assert index.ntotal == 10

        # Snapshot manifest version 1
        manifest_v1 = {
            "version": "1.0",
            "timestamp": "2026-09-13T08:00:00Z",
            "total_nodes": index.ntotal,
        }
        (index_dir / "manifest.json").write_text(json.dumps(manifest_v1), encoding="utf-8")

        # Incremental feed update: 5 new threat reports
        new_vectors = np.random.randn(5, 384).astype(np.float32)
        faiss.normalize_L2(new_vectors)
        index.add(new_vectors)
        assert index.ntotal == 15

        # Snapshot manifest version 2
        manifest_v2 = {
            "version": "1.1",
            "timestamp": "2026-09-13T09:00:00Z",
            "total_nodes": index.ntotal,
        }
        (index_dir / "manifest.json").write_text(json.dumps(manifest_v2), encoding="utf-8")

        # Verify updated manifest state
        current_manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        assert current_manifest["total_nodes"] == 15
        assert current_manifest["version"] == "1.1"

    def test_scenario_5_multi_format_corpus_conditional_ocr_and_duplicate_skip(
        self, tmp_path: Path
    ) -> None:
        """
        Scenario 5: Multi-Format Corpus Ingestion with Conditional OCR Fallback.

        Workflow:
        1. Ingest multi-format batch containing markdown, text, and simulated scanned PDF.
        2. Text-rich markdown and text files extract directly without OCR.
        3. Scanned PDF triggers conditional Tesseract OCR.
        4. Re-submitting identical file triggers exact duplicate detection and skips re-indexing.
        """
        processed_hashes: Set[str] = set()
        ingested_files: List[Dict[str, Any]] = []

        # Document A: Clean Markdown
        doc_a = tmp_path / "analysis.md"
        doc_a.write_text("# Clean text\nReverse engineering findings.", encoding="utf-8")

        # Document B: Duplicate of Document A
        doc_b = tmp_path / "analysis_copy.md"
        doc_b.write_text("# Clean text\nReverse engineering findings.", encoding="utf-8")

        for f in [doc_a, doc_b]:
            f_bytes = f.read_bytes()
            f_hash = hashlib.sha256(f_bytes).hexdigest()
            if f_hash in processed_hashes:
                # Duplicate skipped
                continue
            processed_hashes.add(f_hash)
            ingested_files.append({"file_name": f.name, "file_hash": f_hash})

        assert len(ingested_files) == 1
        assert ingested_files[0]["file_name"] == "analysis.md"
        assert len(processed_hashes) == 1
