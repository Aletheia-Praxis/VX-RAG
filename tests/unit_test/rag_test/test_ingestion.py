"""
Unit tests for Docling conditional ingestion pipeline and adaptive markdown chunker.

Authoritative Source: ORIGINAL_REQUEST.md §R2, PROJECT.md Features 6-10, Tech Spec §4.1-§4.3.
Verifies:
- Programmatic extraction without OCR on text PDF, TXT, MD
- Conditional OCR fallback when text is empty, image-only, or copy-protected
- Boilerplate removal preserving markdown headers and code blocks
- Markdown chunking adhering to sentence boundaries and 1024 token budget
- Technical adaptive chunking keeping code blocks within 256-512 tokens with intact fences
- Strict 5-field metadata assignment on all generated chunk TextNodes
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
from llama_index.core.schema import TextNode

from src.rag.exceptions import DocumentParsingError
from src.rag.ingestion import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CODE_CHUNK_SIZE,
    MAX_CODE_CHUNK_SIZE,
    MIN_CODE_CHUNK_SIZE,
    CorruptedDocumentError,
    DoclingPipeline,
    balance_code_fences,
    break_unbroken_strings,
    chunk_markdown,
    count_tokens,
    get_default_pipeline,
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


class TestDoclingStandardExtraction:
    """Tests for standard programmatic document extraction without OCR."""

    def test_programmatic_extraction_markdown_file(self, tmp_path: Path) -> None:
        """Verify standard extraction handles Markdown files and retains structure."""
        md_file = tmp_path / "threat_report.md"
        content = (
            "# Threat Actor Overview\n\n"
            "Active phishing campaign identified.\n\n"
            "- IOC 1: 198.51.100.1\n"
            "- IOC 2: example.com\n"
        )
        md_file.write_text(content, encoding="utf-8")

        pipeline = DoclingPipeline()
        extracted = pipeline.extract_text(md_file)

        assert "# Threat Actor Overview" in extracted
        assert "Active phishing campaign identified." in extracted
        assert "- IOC 1: 198.51.100.1" in extracted
        assert not pipeline.ocr_invoked, "OCR must not be invoked for markdown files"

    def test_programmatic_extraction_plaintext_file(self, tmp_path: Path) -> None:
        """Verify standard extraction handles plaintext files without OCR."""
        txt_file = tmp_path / "raw_log.txt"
        content = "2026-09-13 10:00:00 [INFO] Payload executed in sandbox environment.\n"
        txt_file.write_text(content, encoding="utf-8")

        pipeline = DoclingPipeline()
        extracted = pipeline.extract_text(txt_file)

        assert "Payload executed in sandbox" in extracted
        assert not pipeline.ocr_invoked, "OCR must not be invoked for plaintext files"

    def test_programmatic_extraction_searchable_pdf(self, tmp_path: Path) -> None:
        """Verify OCR is bypassed when PDF contains a valid programmatic text stream."""
        pdf_file = tmp_path / "searchable.pdf"
        # Simulate minimal valid PDF header
        pdf_file.write_bytes(b"%PDF-1.7\n%Fake search layer\n%%EOF")

        pipeline = DoclingPipeline()
        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Searchable memory dump report."
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            extracted = pipeline.extract_text(pdf_file)

        assert "Searchable memory dump report." in extracted
        assert not pipeline.ocr_invoked, "OCR must not be invoked for searchable PDFs"

    def test_extraction_corrupted_pdf_header_raises_error(self, tmp_path: Path) -> None:
        """Verify extraction raises CorruptedDocumentError when PDF header is invalid."""
        corrupt_file = tmp_path / "corrupted.pdf"
        corrupt_file.write_bytes(b"\x00\xFF\xFE\x00INVALID_HEADER_GARBAGE")

        pipeline = DoclingPipeline()
        with pytest.raises(CorruptedDocumentError, match="Corrupted PDF header"):
            pipeline.extract_text(corrupt_file)

    def test_extraction_empty_file_raises_parsing_error(self, tmp_path: Path) -> None:
        """Verify extraction raises DocumentParsingError on empty 0-byte PDF."""
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")

        pipeline = DoclingPipeline()
        with pytest.raises(DocumentParsingError, match="Empty 0-byte"):
            pipeline.extract_text(empty_pdf)

    def test_extraction_nonexistent_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """Verify FileNotFoundError is raised when file does not exist."""
        missing = tmp_path / "nonexistent.pdf"
        pipeline = DoclingPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.extract_text(missing)

    def test_extraction_directory_path_raises_is_a_directory_error(self, tmp_path: Path) -> None:
        """Verify IsADirectoryError is raised when path is a directory."""
        pipeline = DoclingPipeline()
        with pytest.raises(IsADirectoryError):
            pipeline.extract_text(tmp_path)


class TestConditionalOCRFallback:
    """Tests for conditional Tesseract OCR fallback behavior."""

    def test_conditional_ocr_triggered_for_scanned_image_only_pdf(
        self, tmp_path: Path
    ) -> None:
        """Verify OCR is triggered when standard extraction yields 0 text characters."""
        scanned_pdf = tmp_path / "scanned_invoice.pdf"
        scanned_pdf.write_bytes(b"%PDF-1.7\n%Scanned page\n%%EOF")

        pipeline = DoclingPipeline()
        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""  # Empty text stream
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader), patch.object(
            pipeline, "_ocr_extract", return_value="Recovered OCR text from scanned image"
        ):
            extracted = pipeline.extract_text(scanned_pdf)

        assert pipeline.ocr_invoked, "OCR must trigger when standard text layer is empty"
        assert "Recovered OCR text from scanned image" in extracted
        assert extracted.startswith("## OCR Ingested Content")

    def test_conditional_ocr_triggered_for_copy_protected_pdf(self, tmp_path: Path) -> None:
        """Verify OCR is triggered when PDF permissions or DRM prevent text extraction."""
        protected_pdf = tmp_path / "protected.pdf"
        protected_pdf.write_bytes(b"%PDF-1.7\n%Protected\n%%EOF")

        pipeline = DoclingPipeline()
        mock_reader = MagicMock()
        mock_reader.is_encrypted = True
        mock_reader.decrypt.return_value = 0  # Failed empty password decrypt

        with patch("pypdf.PdfReader", return_value=mock_reader), patch.object(
            pipeline, "_ocr_extract", return_value="Text from protected PDF render"
        ):
            extracted = pipeline.extract_text(protected_pdf)

        assert pipeline.ocr_invoked, "OCR must trigger for copy-protected documents"
        assert "Text from protected PDF render" in extracted

    def test_conditional_ocr_triggered_for_image_files(self, tmp_path: Path) -> None:
        """Verify OCR is directly invoked for image file formats."""
        png_file = tmp_path / "screenshot.png"
        png_file.write_bytes(b"\x89PNG\r\n\x1a\nIMAGE_BYTES")

        pipeline = DoclingPipeline()
        with patch.object(
            pipeline, "_ocr_extract", return_value="Extracted text from PNG screenshot"
        ):
            extracted = pipeline.extract_text(png_file)

        assert pipeline.ocr_invoked, "OCR must trigger for image files"
        assert "Extracted text from PNG screenshot" in extracted

    def test_conditional_ocr_failure_gracefully_returns_empty_string(
        self, tmp_path: Path
    ) -> None:
        """Verify OCR engine failure logs warning without crashing, returning empty string."""
        scanned_pdf = tmp_path / "failed_scan.pdf"
        scanned_pdf.write_bytes(b"%PDF-1.7\n%EOF")

        pipeline = DoclingPipeline()
        mock_reader = MagicMock()
        mock_reader.is_encrypted = False
        mock_reader.pages = []

        with patch("pypdf.PdfReader", return_value=mock_reader), patch.object(
            pipeline, "_ocr_extract", side_effect=RuntimeError("Tesseract binary missing")
        ):
            extracted = pipeline.extract_text(scanned_pdf)

        assert pipeline.ocr_invoked
        assert extracted == "", "Failed OCR must return empty string gracefully"


class TestBoilerplateRemoval:
    """Tests for aggressive boilerplate removal while preserving structural markdown."""

    def test_boilerplate_strips_page_numbers(self) -> None:
        """Verify standalone and formatted page numbers are stripped."""
        text = (
            "Critical vulnerability analysis.\n\n"
            "Page 3 of 15\n\n"
            "Secondary vulnerability analysis.\n\n"
            "- 4 -\n\n"
            "Final notes."
        )
        pipeline = DoclingPipeline()
        cleaned = pipeline.clean_boilerplate(text)

        assert "Page 3 of 15" not in cleaned
        assert "- 4 -" not in cleaned
        assert "Critical vulnerability analysis." in cleaned
        assert "Secondary vulnerability analysis." in cleaned
        assert "Final notes." in cleaned

    def test_boilerplate_strips_recurring_banners(self) -> None:
        """Verify recurring confidentiality and classification banners are removed."""
        text = (
            "CONFIDENTIAL - DO NOT DISTRIBUTE\n\n"
            "Operational threat intel.\n\n"
            "TOP SECRET - DO NOT DISTRIBUTE\n\n"
            "Additional telemetry."
        )
        pipeline = DoclingPipeline()
        cleaned = pipeline.clean_boilerplate(text)

        assert "CONFIDENTIAL - DO NOT DISTRIBUTE" not in cleaned
        assert "TOP SECRET - DO NOT DISTRIBUTE" not in cleaned
        assert "Operational threat intel." in cleaned
        assert "Additional telemetry." in cleaned

    def test_boilerplate_preserves_markdown_headers(self) -> None:
        """Verify markdown headers (#, ##, ###) are strictly preserved."""
        text = (
            "# Main Incident Response Plan\n\n"
            "## Phase 1: Containment\n\n"
            "### Step 1.1: Isolate Endpoints\n\n"
            "Details regarding network isolation."
        )
        pipeline = DoclingPipeline()
        cleaned = pipeline.clean_boilerplate(text)

        assert "# Main Incident Response Plan" in cleaned
        assert "## Phase 1: Containment" in cleaned
        assert "### Step 1.1: Isolate Endpoints" in cleaned

    def test_boilerplate_preserves_code_blocks(self) -> None:
        """Verify code blocks and indentation are 100% preserved."""
        code_text = (
            "Script:\n"
            "```python\n"
            "def inject_dll(pid: int, path: str) -> bool:\n"
            "    # 12 is a number inside code\n"
            "    return True\n"
            "```\n"
        )
        pipeline = DoclingPipeline()
        cleaned = pipeline.clean_boilerplate(code_text)

        assert "```python" in cleaned
        assert "def inject_dll(pid: int, path: str) -> bool:" in cleaned
        assert "    # 12 is a number inside code" in cleaned

    def test_boilerplate_preserves_markdown_tables(self) -> None:
        """Verify markdown tables are preserved intact."""
        table = (
            "| CVE ID | Severity | Vector |\n"
            "|--------|----------|--------|\n"
            "| CVE-2026-0001 | Critical | Network |\n"
            "| CVE-2026-0002 | High | Local |\n"
        )
        pipeline = DoclingPipeline()
        cleaned = pipeline.clean_boilerplate(table)

        assert "| CVE ID | Severity | Vector |" in cleaned
        assert "| CVE-2026-0001 | Critical | Network |" in cleaned


class TestMarkdownAwareChunking:
    """Tests for general markdown chunking (1024 token budget, 10-15% overlap, sentence preservation)."""

    def test_chunking_default_token_budget_1024(self) -> None:
        """Verify text chunks do not exceed 1024 tokens."""
        assert DEFAULT_CHUNK_SIZE == 1024
        assert DEFAULT_CODE_CHUNK_SIZE == 512
        assert DEFAULT_CHUNK_OVERLAP == 128
        words = ["payload"] * 2500
        long_text = " ".join(words)

        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)
        nodes = pipeline.chunk_markdown(long_text)

        assert len(nodes) > 1
        for node in nodes:
            token_count = len(node.text.split())
            assert token_count <= 1024, f"Chunk exceeded 1024 tokens: {token_count}"

    def test_chunking_overlap_ratio_within_10_to_15_percent(self) -> None:
        """Verify default overlap is between 10% and 15% of chunk size."""
        chunk_size = 1024
        overlap = 128  # 128 / 1024 = 12.5%
        ratio = overlap / chunk_size
        assert 0.10 <= ratio <= 0.15

        pipeline = DoclingPipeline(default_chunk_size=chunk_size, overlap=overlap)
        assert pipeline.overlap == 128

    def test_chunking_preserves_sentence_boundaries(self) -> None:
        """Verify chunk boundaries never split a sentence mid-phrase."""
        s1 = "The LockBit ransomware encrypts files using AES-256."
        s2 = "A ransom note is dropped into each targeted directory."
        s3 = "Telemetry indicates lateral movement via compromised RDP credentials."
        doc = f"{s1} {s2} {s3}"

        pipeline = DoclingPipeline(default_chunk_size=20, overlap=5)
        nodes = pipeline.chunk_markdown(doc)

        assert len(nodes) >= 2
        for node in nodes:
            # Each chunk text should end with a period or complete sentence boundary
            assert node.text.endswith(".")
            assert (
                node.text.startswith("The LockBit")
                or node.text.startswith("A ransom note")
                or node.text.startswith("Telemetry indicates")
            )

    def test_chunking_preserves_heading_context(self) -> None:
        """Verify chunk split respects markdown headings."""
        doc = "# Chapter 1: Reconnaissance\nScanning ports.\n\n# Chapter 2: Weaponization\nCrafting exploit."
        pipeline = DoclingPipeline(default_chunk_size=15, overlap=3)
        nodes = pipeline.chunk_markdown(doc)

        assert any("# Chapter 1: Reconnaissance" in n.text for n in nodes)
        assert any("# Chapter 2: Weaponization" in n.text for n in nodes)

    def test_chunking_rejects_overlap_greater_than_or_equal_chunk_size(self) -> None:
        """Verify ValueError is raised when overlap >= chunk_size."""
        with pytest.raises(ValueError, match="overlap"):
            DoclingPipeline(default_chunk_size=500, overlap=500)

        with pytest.raises(ValueError, match="overlap"):
            DoclingPipeline(default_chunk_size=500, overlap=600)

    def test_chunking_rejects_negative_overlap(self) -> None:
        """Verify ValueError is raised when overlap is negative."""
        with pytest.raises(ValueError, match="overlap"):
            DoclingPipeline(default_chunk_size=500, overlap=-1)


class TestTechnicalAdaptiveChunking:
    """Tests for technical content chunking (256-512 tokens, code fences, tables, hex dumps)."""

    def test_code_block_adaptive_budget_window(self) -> None:
        """Verify code blocks adhere to 256 to 512 token budget window."""
        assert MIN_CODE_CHUNK_SIZE == 256
        assert MAX_CODE_CHUNK_SIZE == 512
        pipeline = DoclingPipeline()
        assert pipeline.code_chunk_size == 512

    def test_small_code_block_never_split(self) -> None:
        """Verify a code block under 512 tokens is kept in a single chunk."""
        cpp_code = (
            "```cpp\n"
            "#include <windows.h>\n"
            "int WINAPI WinMain(HINSTANCE hInst, HINSTANCE hPrev, LPSTR lpCmd, int nShow) {\n"
            "    MessageBoxA(NULL, \"Malware Test\", \"Alert\", MB_OK);\n"
            "    return 0;\n"
            "}\n"
            "```"
        )
        nodes = chunk_markdown(cpp_code, code_chunk_size=512)

        assert len(nodes) == 1
        assert nodes[0].text.startswith("```cpp")
        assert nodes[0].text.endswith("```")

    def test_large_code_block_partitioning_preserves_code_fences(self) -> None:
        """Verify large code blocks (>512 tokens) are partitioned with matching fences."""
        lines = [f"    int register_var_{i} = 0x{i:04X}; // statement {i}" for i in range(120)]
        large_code = "```cpp\nvoid LongFunction() {\n" + "\n".join(lines) + "\n}\n```"

        assert count_tokens(large_code) > 512

        nodes = chunk_markdown(large_code, code_chunk_size=256)
        assert len(nodes) > 1

        for node in nodes:
            # Each sub-chunk must retain opening and closing fences
            opening = len(re.findall(r"^```", node.text, flags=re.MULTILINE))
            closing = len(re.findall(r"^```$", node.text, flags=re.MULTILINE))
            assert opening >= 1, "Sub-chunk must have opening code fence"
            assert closing >= 1, "Sub-chunk must have closing code fence"

    def test_empty_code_block_handling(self) -> None:
        """Verify empty code block does not cause IndexError or crash."""
        empty_code = "```cpp\n```"
        nodes = chunk_markdown(empty_code)
        assert len(nodes) == 1
        assert nodes[0].text == "```cpp\n```"

    def test_unclosed_code_fence_balancing(self) -> None:
        """Verify unclosed code fences are balanced by appending closing fence."""
        unclosed = "## Heading\n```python\nimport os\nprint(os.getpid())\n"
        balanced = balance_code_fences(unclosed)

        fences = re.findall(r"^```", balanced, flags=re.MULTILINE)
        assert len(fences) % 2 == 0
        assert balanced.rstrip().endswith("```")

    def test_markdown_tables_preserved_intact(self) -> None:
        """Verify markdown tables are not fractured across chunks."""
        table = (
            "| Register | Purpose |\n"
            "|----------|---------|\n"
            "| RAX      | Accumulator |\n"
            "| RBX      | Base |\n"
            "| RCX      | Counter |\n"
            "| RDX      | Data |\n"
        )
        nodes = chunk_markdown(table)
        assert len(nodes) == 1
        assert "| RAX      | Accumulator |" in nodes[0].text
        assert "| RDX      | Data |" in nodes[0].text

    def test_hex_dumps_preserved_intact(self) -> None:
        """Verify hex memory dumps are preserved with offsets and ascii representations."""
        hex_dump = (
            "00000000  4d 5a 90 00 03 00 00 00  04 00 00 00 ff ff 00 00  |MZ..............|\n"
            "00000010  b8 00 00 00 00 00 00 00  40 00 00 00 00 00 00 00  |........@.......|\n"
        )
        nodes = chunk_markdown(hex_dump)
        assert len(nodes) == 1
        assert "|MZ..............|" in nodes[0].text
        assert "00000010" in nodes[0].text

    def test_extremely_long_unbroken_string_does_not_loop_infinitely(self) -> None:
        """Verify unbroken strings exceeding 1000 chars are split safely without hangs."""
        long_string = "A" * 5000
        broken = break_unbroken_strings(long_string, max_chunk_chars=1000)
        parts = broken.split(" ")
        assert len(parts) == 5
        assert all(len(p) <= 1000 for p in parts)


class TestStrictMetadataAssignment:
    """Tests for strict 5-field metadata assignment on generated nodes."""

    def test_ingest_file_assigns_exact_five_metadata_fields(self, tmp_path: Path) -> None:
        """Verify each TextNode produced by ingest_file contains exactly the 5 metadata fields."""
        test_file = tmp_path / "emotet_report.md"
        test_file.write_text(
            "# Emotet IOC Report\n\nIdentified C2 domain: bad.example.com",
            encoding="utf-8",
        )

        pipeline = DoclingPipeline()
        nodes = pipeline.ingest_file(test_file)

        assert len(nodes) >= 1
        for node in nodes:
            assert isinstance(node, TextNode)
            assert set(node.metadata.keys()) == STRICT_METADATA_KEYS
            assert len(node.metadata) == 5
            assert node.metadata["file_name"] == "emotet_report.md"
            assert node.metadata["file_type"] == "md"
            assert len(node.metadata["file_hash"]) == 64
            assert re.match(r"^\d{4}-\d{2}-\d{2}T", node.metadata["creation_date"])
            assert re.match(r"^\d{4}-\d{2}-\d{2}T", node.metadata["ingestion_date"])

    def test_ingest_directory_skips_duplicates_and_assigns_metadata(
        self, tmp_path: Path
    ) -> None:
        """Verify ingest_directory detects duplicate files and assigns valid metadata."""
        dir_path = tmp_path / "corpus"
        dir_path.mkdir()

        f1 = dir_path / "doc1.txt"
        f1.write_text("Unique content 1", encoding="utf-8")

        # Duplicate file with different name
        f2 = dir_path / "doc2.txt"
        f2.write_text("Unique content 1", encoding="utf-8")

        f3 = dir_path / "doc3.txt"
        f3.write_text("Unique content 2", encoding="utf-8")

        pipeline = DoclingPipeline()
        nodes = pipeline.ingest_directory(dir_path)

        # f1 and f2 are duplicates; only one should be ingested along with f3
        file_hashes = {n.metadata["file_hash"] for n in nodes}
        assert len(file_hashes) == 2

        for node in nodes:
            assert set(node.metadata.keys()) == STRICT_METADATA_KEYS

    def test_standalone_ingest_functions(self, tmp_path: Path) -> None:
        """Verify module-level standalone ingest_file and ingest_directory functions."""
        sample_file = tmp_path / "sample.md"
        sample_file.write_text("# Standalone Test\nContent here.", encoding="utf-8")

        default_pipe = get_default_pipeline()
        assert default_pipe is not None

        nodes = ingest_file(sample_file)
        assert len(nodes) >= 1
        assert set(nodes[0].metadata.keys()) == STRICT_METADATA_KEYS

        nodes_dir = ingest_directory(tmp_path)
        assert len(nodes_dir) >= 1
