"""
Adversarial stress challenge test suite for src/rag/ingestion.py.

Empirically tests:
1. Markdown table integrity: tables with 2 to 50 rows are not fractured mid-table across chunks.
2. Hex dump preservation: memory dumps with offset and ascii columns are preserved verbatim.
3. Overlap compliance: adjacent chunks have token overlap strictly between 10% and 15%.
4. Duplicate file detection: ingest_directory deduplicates identical files by SHA-256 hash even if names differ.
5. Error recovery: directory traversal recovers and continues when corrupted files are present alongside valid files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from src.rag.ingestion import (
    DoclingPipeline,
    count_tokens,
    ingest_directory,
    ingest_file,
)
from src.rag.metadata import STRICT_METADATA_KEYS

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401


class TestAdversarialMarkdownTableIntegrity:
    """Adversarially challenge markdown table chunking boundaries across 2 to 50 rows."""

    @pytest.mark.parametrize("row_count", [2, 3, 5, 10, 20, 30, 40, 50])
    def test_markdown_tables_from_2_to_50_rows_never_fractured(self, row_count: int) -> None:
        """
        Verify markdown tables with 2-50 rows are never fractured mid-table across chunks.

        The table is flanked by long contextual prose to force chunk boundary splitting.
        """
        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)

        # Build table with specified row count
        header = "| Index | Module Name | Base Address | Size | Entry Point |"
        divider = "|---|---|---|---|---|"
        rows: list[str] = []
        for r in range(row_count):
            rows.append(
                f"| Row_{r:02d} | ntdll_{r:02d}.dll | 0x7FFF000{r:02X}0000 | 0x1A0000 | 0x7FFF000{r:02X}1234 |"
            )
        table_text = "\n".join([header, divider, *rows])

        # Flank with substantial prose before and after to trigger chunk division
        prefix = " ".join([f"Preceding telemetry context statement {i}." for i in range(75)])
        suffix = " ".join([f"Succeeding forensic observation statement {i}." for i in range(75)])
        full_markdown = f"{prefix}\n\n{table_text}\n\n{suffix}"

        nodes = pipeline.chunk_markdown(full_markdown)
        assert len(nodes) >= 2, "Expected multiple chunks due to length"

        # Locate chunks that contain any portion of the table
        table_containing_chunks = [n for n in nodes if "| Row_00 |" in n.text]
        assert len(table_containing_chunks) >= 1, "Table Row_00 must appear in at least one chunk"

        for chunk_node in table_containing_chunks:
            # Header and divider must be present
            assert header in chunk_node.text, "Table header missing in table chunk"
            assert divider in chunk_node.text, "Table divider missing in table chunk"

            # All rows must be present in the same chunk without mid-table fracture
            for r in range(row_count):
                row_signature = f"| Row_{r:02d} |"
                assert (
                    row_signature in chunk_node.text
                ), f"Row {r} was fractured out of chunk containing table with {row_count} rows"

    def test_consecutive_adjacent_tables_remain_discrete_and_intact(self) -> None:
        """Verify back-to-back markdown tables are kept intact without corruption."""
        table1 = (
            "| Service | Port | Protocol |\n"
            "|---|---|---|\n"
            "| SSH | 22 | TCP |\n"
            "| HTTPS | 443 | TCP |"
        )
        table2 = (
            "| User | Privilege | Shell |\n"
            "|---|---|---|\n"
            "| root | 0 | /bin/bash |\n"
            "| daemon | 1 | /usr/sbin/nologin |"
        )
        doc = f"# Infrastructure Overview\n\n{table1}\n\n{table2}\n\nConclusion summary."

        pipeline = DoclingPipeline()
        nodes = pipeline.chunk_markdown(doc)

        assert len(nodes) >= 1
        full_text = "\n\n".join(n.text for n in nodes)
        assert "| SSH | 22 | TCP |" in full_text
        assert "| root | 0 | /bin/bash |" in full_text


class TestAdversarialHexDumpPreservation:
    """Adversarially challenge memory hex dump preservation verbatim."""

    def test_canonical_pe_hex_dump_preserved_verbatim(self) -> None:
        """Verify hex memory dump with offset and ascii columns is preserved verbatim."""
        hex_dump_lines = [
            "00000000  4d 5a 90 00 03 00 00 00  04 00 00 00 ff ff 00 00  |MZ..............|",
            "00000010  b8 00 00 00 00 00 00 00  40 00 00 00 00 00 00 00  |........@.......|",
            "00000020  00 00 00 00 00 00 00 00  00 00 00 00 00 00 00 00  |................|",
            "00000030  00 00 00 00 00 00 00 00  00 00 00 00 80 00 00 00  |................|",
            "00000040  0e 1f ba 0e 00 b4 09 cd  21 b8 01 4c cd 21 54 68  |........!..L.!Th|",
        ]
        verbatim_hex = "\n".join(hex_dump_lines)

        pipeline = DoclingPipeline()
        nodes = pipeline.chunk_markdown(verbatim_hex)

        assert len(nodes) == 1
        node_text = nodes[0].text
        for line in hex_dump_lines:
            assert line in node_text, f"Hex line '{line}' was modified or missing"

    def test_hex_dump_embedded_in_heavy_boilerplate(self) -> None:
        """Verify aggressive boilerplate stripping leaves raw hex dump lines 100% intact."""
        hex_dump = (
            "00401000  55 8b ec 83 ec 10 53 56  57 8d 7d f0 33 c0 b9 04  |U.....SVW.}.3...|\n"
            "00401010  f3 ab 8b 45 08 85 c0 74  12 8b 40 04 85 c0 74 0b  |...E...t..@...t.|\n"
            "00401020  50 e8 23 01 00 00 59 85  c0 75 02 33 c0 5f 5e 5b  |P.#...Y..u.3._^[|"
        )
        raw_doc = (
            "CONFIDENTIAL - DO NOT DISTRIBUTE\n"
            "Page 1 of 3\n"
            "TOP SECRET - DO NOT DISTRIBUTE\n\n"
            "# Disassembly Dump\n\n"
            f"{hex_dump}\n\n"
            "Page 2 of 3\n"
            "RESTRICTED - DO NOT DISTRIBUTE\n"
            "<!-- internal note: review required -->\n\n"
            "End of dump."
        )

        pipeline = DoclingPipeline()
        cleaned = pipeline.clean_boilerplate(raw_doc)

        assert "CONFIDENTIAL" not in cleaned
        assert "Page 1 of 3" not in cleaned
        assert "TOP SECRET" not in cleaned
        assert "RESTRICTED" not in cleaned
        assert "<!-- internal note" not in cleaned

        # Verify hex dump lines remain intact in cleaned text
        for line in hex_dump.splitlines():
            assert line in cleaned, f"Hex line was destroyed in boilerplate cleaning: {line}"

        # Verify chunking also preserves them intact
        nodes = pipeline.chunk_markdown(cleaned)
        assert any(hex_dump in n.text for n in nodes)


class TestAdversarialOverlapCompliance:
    """Adversarially challenge chunk overlap token bounds strictly between 10% and 15%."""

    def test_token_overlap_strictly_between_10_and_15_percent_for_natural_prose(self) -> None:
        """
        Verify consecutive chunks have token overlap strictly within [10%, 15%] of chunk size.

        Uses natural prose where semantic sentence units fit within the overlap window.
        """
        chunk_size = 1024
        overlap = 128  # Target 12.5%
        pipeline = DoclingPipeline(default_chunk_size=chunk_size, overlap=overlap)

        # Build natural prose with ~34 tokens per paragraph (100 paragraphs)
        paras: list[str] = []
        for p in range(100):
            paras.append(
                f"Paragraph {p:02d}: The advanced persistent threat group leverages spear-phishing "
                f"attachments containing weaponized macro documents to execute arbitrary shellcode "
                f"in memory, avoiding disk-based antivirus detection."
            )
        text = "\n\n".join(paras)

        nodes = pipeline.chunk_markdown(text)
        assert len(nodes) >= 3, "Expected at least 3 chunks to verify multiple transitions"

        for i in range(len(nodes) - 1):
            c1 = nodes[i].text
            c2 = nodes[i + 1].text

            # Determine matching overlap string between c1 suffix and c2 prefix
            overlap_str = ""
            for k in range(1, len(c2) + 1):
                prefix = c2[:k]
                if c1.endswith(prefix):
                    overlap_str = prefix

            assert (
                len(overlap_str) > 0
            ), f"No overlap found between chunk {i} and chunk {i + 1}"

            toks_overlap = count_tokens(overlap_str)
            overlap_ratio = toks_overlap / chunk_size

            assert 0.10 <= overlap_ratio <= 0.15, (
                f"Overlap ratio {overlap_ratio:.2%} ({toks_overlap} tokens) between "
                f"chunk {i} and {i + 1} is outside the required [10%, 15%] window!"
            )

    def test_overlap_ratio_calculation_with_custom_parameters(self) -> None:
        """Verify custom chunk size and overlap configurations maintain ratio compliance."""
        chunk_size = 500
        overlap = 60  # 60 / 500 = 12%
        pipeline = DoclingPipeline(default_chunk_size=chunk_size, overlap=overlap)

        paras: list[str] = [
            f"Telemetry item {i:03d} confirms inbound beaconing from external IP 198.51.100.{i}."
            for i in range(70)
        ]
        text = "\n\n".join(paras)

        nodes = pipeline.chunk_markdown(text)
        assert len(nodes) >= 2

        for i in range(len(nodes) - 1):
            c1 = nodes[i].text
            c2 = nodes[i + 1].text
            overlap_str = ""
            for k in range(1, len(c2) + 1):
                prefix = c2[:k]
                if c1.endswith(prefix):
                    overlap_str = prefix

            toks_overlap = count_tokens(overlap_str)
            ratio = toks_overlap / chunk_size
            assert 0.10 <= ratio <= 0.15, (
                f"Custom overlap ratio {ratio:.2%} out of bounds [10%, 15%]"
            )

    def test_overlap_drops_to_zero_for_paragraphs_exceeding_overlap_budget(self) -> None:
        """
        Verify that paragraphs exceeding overlap budget (128 tokens) yield 0% overlap.

        Demonstrates that treating whole paragraphs as indivisible units causes complete
        loss of chunk overlap when paragraph length exceeds eff_overlap.
        """
        chunk_size = 1024
        overlap = 128
        pipeline = DoclingPipeline(default_chunk_size=chunk_size, overlap=overlap)

        # 20 paragraphs of 150 words each (~190 tokens per paragraph)
        paras: list[str] = [
            " ".join([f"ForensicTelemetryEvidence_{i}_{p}" for i in range(150)]) + "."
            for p in range(20)
        ]
        text = "\n\n".join(paras)

        nodes = pipeline.chunk_markdown(text)
        assert len(nodes) >= 2

        # Check overlap between consecutive chunks
        zero_overlap_transitions = 0
        for i in range(len(nodes) - 1):
            c1 = nodes[i].text
            c2 = nodes[i + 1].text
            overlap_str = ""
            for k in range(1, len(c2) + 1):
                prefix = c2[:k]
                if c1.endswith(prefix):
                    overlap_str = prefix

            toks_overlap = count_tokens(overlap_str)
            if toks_overlap == 0:
                zero_overlap_transitions += 1

        assert zero_overlap_transitions > 0, "Expected zero overlap transitions for large paragraphs"


class TestAdversarialDuplicateFileDetection:
    """Adversarially challenge duplicate file detection by SHA-256 hash."""

    def test_ingest_directory_skips_duplicates_across_subdirectories_and_different_names(
        self, tmp_path: Path
    ) -> None:
        """
        Verify identical files with completely different names and extensions are deduplicated.
        """
        corpus_dir = tmp_path / "threat_corpus"
        corpus_dir.mkdir()
        sub_dir = corpus_dir / "archive" / "reports"
        sub_dir.mkdir(parents=True)

        content_alpha = (
            "# Threat Report Alpha\n\nIdentified malicious executable payload hash in telemetry."
        )
        content_beta = (
            "# Threat Report Beta\n\nDistinct threat campaign targeting financial institutions."
        )

        # Create identical files with different names
        (corpus_dir / "original_alpha.md").write_text(content_alpha, encoding="utf-8")
        (corpus_dir / "copy_of_alpha.txt").write_text(content_alpha, encoding="utf-8")
        (sub_dir / "nested_clone_alpha.markdown").write_text(content_alpha, encoding="utf-8")

        # Create unique file
        (corpus_dir / "original_beta.md").write_text(content_beta, encoding="utf-8")
        (sub_dir / "renamed_beta.txt").write_text(content_beta, encoding="utf-8")

        nodes = ingest_directory(corpus_dir, recursive=True)

        # There are 5 total files, but only 2 unique SHA-256 hashes
        hashes = {node.metadata["file_hash"] for node in nodes}
        assert len(hashes) == 2, f"Expected exactly 2 unique hashes, got {len(hashes)}"

        # Every node must have strict 5 fields
        for node in nodes:
            assert set(node.metadata.keys()) == STRICT_METADATA_KEYS

    def test_ingest_directory_preserves_distinct_files(self, tmp_path: Path) -> None:
        """Verify non-duplicate files are not mistakenly skipped."""
        corpus_dir = tmp_path / "unique_corpus"
        corpus_dir.mkdir()

        for i in range(5):
            (corpus_dir / f"doc_{i}.txt").write_text(
                f"Distinct unique content payload {i}", encoding="utf-8"
            )

        nodes = ingest_directory(corpus_dir, recursive=False)
        assert len(nodes) == 5
        hashes = {n.metadata["file_hash"] for n in nodes}
        assert len(hashes) == 5


class TestAdversarialErrorRecovery:
    """Adversarially challenge directory traversal and file parsing error recovery."""

    def test_directory_traversal_with_corrupted_files_alongside_valid_files(
        self, tmp_path: Path
    ) -> None:
        """
        Verify corrupted files do not abort traversal of valid files.

        Tests:
        - Corrupted PDF binary header
        - 0-byte empty PDF
        - Binary corrupted file
        - Unreadable image triggering OCR failure
        """
        corpus = tmp_path / "mixed_corpus"
        corpus.mkdir()

        # Valid files
        (corpus / "valid_1.txt").write_text(
            "Valid telemetry document 1 with actionable intelligence.", encoding="utf-8"
        )
        (corpus / "valid_2.md").write_text(
            "# Valid Report 2\n\nThreat actor analysis and indicators.", encoding="utf-8"
        )

        # Corrupted PDF (invalid header)
        (corpus / "corrupted_header.pdf").write_bytes(b"\x00\xFF\xFE\x00_NOT_A_PDF_FILE")

        # Empty 0-byte PDF
        (corpus / "zero_byte.pdf").write_bytes(b"")

        # Corrupted image file
        (corpus / "corrupt_bitmap.png").write_bytes(b"INVALID_PNG_BYTES")

        # Unsupported file formats that should be ignored without error
        (corpus / "tool.exe").write_bytes(b"MZ\x90\x00")
        (corpus / "data.bin").write_bytes(b"\x01\x02\x03\x04")

        pipeline = DoclingPipeline()
        nodes = pipeline.ingest_directory(corpus, recursive=True)

        ingested_names = {node.metadata["file_name"] for node in nodes}
        assert "valid_1.txt" in ingested_names
        assert "valid_2.md" in ingested_names
        assert "corrupted_header.pdf" not in ingested_names
        assert "zero_byte.pdf" not in ingested_names
        assert "tool.exe" not in ingested_names

    def test_single_file_ingest_graceful_on_ocr_failure(self, tmp_path: Path) -> None:
        """Verify ingest_file returns empty list gracefully on OCR engine crash."""
        image_file = tmp_path / "unreadable_scan.png"
        image_file.write_bytes(b"FAKE_IMAGE_DATA")

        pipeline = DoclingPipeline()
        with patch.object(
            pipeline,
            "_ocr_extract",
            side_effect=RuntimeError("Tesseract OCR process terminated unexpectedly"),
        ):
            nodes = pipeline.ingest_file(image_file)

        assert nodes == [], "Failed OCR must gracefully yield empty node list"

    def test_empty_text_document_yields_empty_node_list(self, tmp_path: Path) -> None:
        """Verify 0-byte or whitespace-only text files return empty list without crash."""
        empty_md = tmp_path / "empty.md"
        empty_md.write_text("   \n\t  \n  ", encoding="utf-8")

        nodes = ingest_file(empty_md)
        assert nodes == []
