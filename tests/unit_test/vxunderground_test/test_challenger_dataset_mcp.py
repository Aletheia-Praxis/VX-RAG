"""
Empirical challenger verification test suite for VX-RAG Dataset Sampling and MCP Configuration.

Tests cover:
- Source dataset zero-touch Merkle fingerprint and file count (18,823 files).
- Excluded stubs and duplicate quarantine integrity.
- Target sample 1,000 PDF files validity (%PDF- magic header, %%EOF trailer, 0 duplicates, 0 corrupt).
- Target sample 192 categories coverage and distribution.
- Target sample manifest.json schema and cryptographic SHA-256 integrity.
- MCP configuration files byte-identity, UTF-8 BOM absence, and FastMCP instantiation.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pytest
from fastmcp.mcp_config import MCPConfig, StdioMCPServer

SOURCE_DIR: Path = Path("data/dataset/vxunderground/downloads")
TARGET_DIR: Path = Path("data/dataset/test_sample_1000")
MANIFEST_PATH: Path = TARGET_DIR / "manifest.json"

EXPECTED_SOURCE_COUNT: int = 18823
EXPECTED_SOURCE_BYTES: int = 19456571751
EXPECTED_SOURCE_FINGERPRINT: str = (
    "7806f687cc4132bb9d3621c7fb303534a01b451cd87dd02d8a9c428d3553ede2"
)

EXPECTED_TARGET_PDF_COUNT: int = 1000
EXPECTED_TARGET_TOTAL_BYTES: int = 1144958964
EXPECTED_CATEGORIES_COUNT: int = 192

EXPECTED_MCP_CONFIG_SHA256: str = (
    "b8c5fd74237d2b24d4561120f2dceccbb4def832dce06e04c24a6f1f53b6881c"
)
AGENTS_MCP_CONFIG: Path = Path(".agents/mcp_config.json")
ROOT_MCP_CONFIG: Path = Path("mcp_config.json")


def compute_sha256(data: bytes) -> str:
    """Compute hex SHA-256 digest of bytes."""
    return hashlib.sha256(data).hexdigest()


@pytest.mark.unit
def test_source_dataset_zero_touch_fingerprint() -> None:
    """Verify nanosecond-precision Merkle fingerprint and file count of source dataset."""
    assert SOURCE_DIR.is_dir(), f"Source directory {SOURCE_DIR} does not exist"

    entries: list[tuple[str, int, int]] = []
    total_bytes: int = 0

    for root, _, files in os.walk(SOURCE_DIR):
        for f in files:
            fp = os.path.join(root, f)
            st = os.stat(fp)
            rel_p = os.path.relpath(fp, SOURCE_DIR)
            entries.append((rel_p, st.st_size, st.st_mtime_ns))
            total_bytes += st.st_size

    entries.sort(key=lambda x: x[0])
    hasher = hashlib.sha256()
    for rel_p, sz, mt in entries:
        hasher.update(f"{rel_p}:{sz}:{mt}\n".encode())

    fingerprint = hasher.hexdigest()

    assert len(entries) == EXPECTED_SOURCE_COUNT, (
        f"Expected {EXPECTED_SOURCE_COUNT} source files, found {len(entries)}"
    )
    assert total_bytes == EXPECTED_SOURCE_BYTES, (
        f"Expected {EXPECTED_SOURCE_BYTES} bytes, found {total_bytes}"
    )
    assert fingerprint == EXPECTED_SOURCE_FINGERPRINT, (
        f"Merkle fingerprint mismatch: expected {EXPECTED_SOURCE_FINGERPRINT}, got {fingerprint}"
    )


@pytest.mark.unit
def test_source_dataset_quarantine_integrity() -> None:
    """Verify that corrupted stubs and duplicates in source were quarantined and excluded from target."""
    quarantined_files: set[str] = {
        "Hydraq- In Depth Analysis.pdf",
        "Night Dragon Specific Protection Measures for Consideration.pdf",
        "6420_2019-05-25 - Analyzing ISFB - The Second Loader_8e16f02d.pdf",
    }

    # Verify they exist in source
    for root, _, files in os.walk(SOURCE_DIR):
        for f in files:
            quarantined_files.discard(f)
    assert len(quarantined_files) == 0, f"Quarantined files not found in source: {quarantined_files}"

    # Verify they NEVER leaked into target sample
    target_filenames: set[str] = {
        f
        for root, _, files in os.walk(TARGET_DIR)
        for f in files
        if f != "manifest.json"
    }
    leaked = target_filenames.intersection({
        "Hydraq- In Depth Analysis.pdf",
        "Night Dragon Specific Protection Measures for Consideration.pdf",
        "6420_2019-05-25 - Analyzing ISFB - The Second Loader_8e16f02d.pdf",
    })
    assert len(leaked) == 0, f"Corrupted or duplicate files leaked into target: {leaked}"


@pytest.mark.unit
def test_target_sample_1000_counts_and_magic_headers() -> None:
    """Verify target directory contains exactly 1,000 valid PDFs with magic headers and 0 duplicates."""
    assert TARGET_DIR.is_dir(), f"Target directory {TARGET_DIR} does not exist"

    all_files: list[Path] = []
    pdf_files: list[Path] = []
    other_files: list[Path] = []

    for root, _, files in os.walk(TARGET_DIR):
        for f in files:
            fp = Path(root) / f
            all_files.append(fp)
            if f == "manifest.json":
                other_files.append(fp)
            else:
                pdf_files.append(fp)

    assert len(all_files) == 1001, f"Expected 1001 files, found {len(all_files)}"
    assert len(pdf_files) == EXPECTED_TARGET_PDF_COUNT, (
        f"Expected {EXPECTED_TARGET_PDF_COUNT} PDFs, found {len(pdf_files)}"
    )
    assert len(other_files) == 1 and other_files[0] == MANIFEST_PATH

    seen_hashes: set[str] = set()
    total_bytes: int = 0

    for pdf_p in pdf_files:
        raw_data = pdf_p.read_bytes()
        sz = len(raw_data)
        total_bytes += sz

        # Verify not corrupt (size > 0)
        assert sz > 0, f"Empty PDF file found: {pdf_p}"

        # Verify %PDF- magic header
        assert raw_data[:5] == b"%PDF-", f"Invalid PDF magic header in {pdf_p}: {raw_data[:10]!r}"

        # Verify %%EOF marker within the last 4096 bytes
        assert b"%%EOF" in raw_data[-4096:], f"Missing %%EOF marker in trailer of {pdf_p}"

        # Verify uniqueness of SHA-256
        h = compute_sha256(raw_data)
        assert h not in seen_hashes, f"Duplicate hash found for {pdf_p}: {h}"
        seen_hashes.add(h)

    assert len(seen_hashes) == EXPECTED_TARGET_PDF_COUNT
    assert total_bytes == EXPECTED_TARGET_TOTAL_BYTES


@pytest.mark.unit
def test_target_sample_category_representation() -> None:
    """Verify target sample covers exactly 192 distinct categories with required distribution."""
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    subfolders: set[str] = set()
    categories: Counter[str] = Counter()
    phases: Counter[str] = Counter()

    for doc in manifest["documents"]:
        subfolders.add(doc["subfolder"])
        categories[doc["category"]] += 1
        phases[doc["sampling_phase"]] += 1

    assert len(subfolders) == EXPECTED_CATEGORIES_COUNT, (
        f"Expected {EXPECTED_CATEGORIES_COUNT} categories, found {len(subfolders)}"
    )

    assert phases["categorical_seed"] == 192
    assert phases["stratified_fill"] == 808
    assert sum(phases.values()) == EXPECTED_TARGET_PDF_COUNT

    expected_category_counts = {
        "Archive": 322,
        "Malware Analysis": 499,
        "Papers": 66,
        "Samples": 113,
    }
    assert dict(categories) == expected_category_counts


@pytest.mark.unit
def test_manifest_schema_and_cryptographic_integrity() -> None:
    """Verify manifest.json adheres to schema and matches file system contents cryptographically."""
    assert MANIFEST_PATH.is_file(), f"Manifest file {MANIFEST_PATH} does not exist"

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["manifest_version"] == "1.0.0"
    assert manifest["source_directory"] == "data/dataset/vxunderground/downloads"
    assert manifest["target_directory"] == "data/dataset/test_sample_1000"
    assert manifest["total_documents"] == EXPECTED_TARGET_PDF_COUNT
    assert manifest["total_bytes"] == EXPECTED_TARGET_TOTAL_BYTES
    assert manifest["prng_seed"] == 42
    assert manifest["source_state_fingerprint"] == EXPECTED_SOURCE_FINGERPRINT

    docs = manifest["documents"]
    assert len(docs) == EXPECTED_TARGET_PDF_COUNT

    for i, doc in enumerate(docs):
        assert doc["id"] == f"doc_{i + 1:04d}"
        assert doc["file_type"] == "pdf"
        assert doc["is_valid_pdf"] is True
        assert doc["magic_header"].startswith("%PDF-")

        target_file = TARGET_DIR / doc["relative_target_path"]
        source_file = SOURCE_DIR / doc["relative_source_path"]

        assert target_file.is_file(), f"Target file missing: {target_file}"
        assert source_file.is_file(), f"Source file missing: {source_file}"

        t_bytes = target_file.read_bytes()
        s_bytes = source_file.read_bytes()

        assert len(t_bytes) == doc["file_size"]
        assert len(s_bytes) == doc["file_size"]

        t_hash = compute_sha256(t_bytes)
        s_hash = compute_sha256(s_bytes)

        assert t_hash == doc["sha256"], f"Target hash mismatch for {doc['id']}"
        assert s_hash == doc["sha256"], f"Source hash mismatch for {doc['id']}"


@pytest.mark.unit
def test_mcp_config_byte_identical_and_no_bom() -> None:
    """Verify .agents/mcp_config.json and mcp_config.json are byte-identical, UTF-8, and have no BOM."""
    assert AGENTS_MCP_CONFIG.is_file(), f"Missing {AGENTS_MCP_CONFIG}"
    assert ROOT_MCP_CONFIG.is_file(), f"Missing {ROOT_MCP_CONFIG}"

    b_agents = AGENTS_MCP_CONFIG.read_bytes()
    b_root = ROOT_MCP_CONFIG.read_bytes()

    assert not b_agents.startswith(b"\xef\xbb\xbf"), ".agents/mcp_config.json contains UTF-8 BOM"
    assert not b_root.startswith(b"\xef\xbb\xbf"), "mcp_config.json contains UTF-8 BOM"

    assert b_agents == b_root, "Configuration files are not byte-identical"

    h_agents = compute_sha256(b_agents)
    assert h_agents == EXPECTED_MCP_CONFIG_SHA256, (
        f"Unexpected SHA-256 hash: {h_agents} vs {EXPECTED_MCP_CONFIG_SHA256}"
    )


@pytest.mark.unit
def test_mcp_config_fastmcp_instantiation() -> None:
    """Verify both MCP configuration files instantiate valid FastMCP MCPConfig and StdioMCPServer."""
    for cfg_path in [AGENTS_MCP_CONFIG, ROOT_MCP_CONFIG]:
        with open(cfg_path, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)

        config = MCPConfig.from_dict(raw_cfg)
        assert "vx-rag" in config.mcpServers, f"Server 'vx-rag' missing in {cfg_path}"

        server = config.mcpServers["vx-rag"]
        assert isinstance(server, StdioMCPServer), (
            f"Expected StdioMCPServer, got {type(server)} in {cfg_path}"
        )
        assert server.command == "python"
        assert server.args == ["-m", "src.cli", "serve", "--transport", "stdio"]
        assert server.env == {"PYTHONPATH": ".", "PYTHONUNBUFFERED": "1"}
        assert server.transport == "stdio"
