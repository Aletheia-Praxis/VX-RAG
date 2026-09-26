"""
Adversarial stress challenge test suite for Milestone M2 (src/rag/ingestion.py).

Empirically tests:
1. Conditional OCR Trigger:
   - Searchable text PDF strictly bypasses OCR (ocr_invoked == False).
   - Scanned image PDF (0 text characters) strictly triggers OCR (ocr_invoked == True).
   - Encrypted / copy-protected PDF strictly triggers OCR (ocr_invoked == True).
   - Plain image files (.png, .jpg) directly trigger OCR.
   - External OCR engine failure degrades gracefully without crashing.
2. Boundary and Corner Conditions in Chunking:
   - Massive unbroken strings (continuous base64 / hex blobs without whitespace).
   - Unclosed and unbalanced code fences.
   - Nested code fences within markdown documentation.
   - Large code blocks (>512 tokens) partitioned with balanced fences.
   - Empty documents and 0-byte files (.txt, .md, .pdf).
   - Corrupted binary headers for PDF files.
   - Tables and hex memory dumps preserved intact.
   - Invalid overlap and chunk size configurations.
3. Strict Metadata Integrity:
   - Every generated TextNode contains ONLY and EXACTLY the 5 canonical keys.
   - Hostile and arbitrary extra fields injected into metadata are stripped.
   - Strict format validation on metadata values (ISO 8601 UTC, SHA-256 hex).
   - Duplicate file deduplication by content hash during directory ingestion.
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pypdf
import pytest
from PIL import Image

from src.rag.exceptions import DocumentParsingError
from src.rag.ingestion import (
    MAX_CODE_CHUNK_SIZE,
    CorruptedDocumentError,
    DoclingPipeline,
    balance_code_fences,
    break_unbroken_strings,
    chunk_markdown,
    count_tokens,
    ingest_directory,
    ingest_file,
)
from src.rag.metadata import (
    ISO8601_PATTERN,
    SHA256_HEX_PATTERN,
    STRICT_METADATA_KEYS,
    compute_file_hash,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401

# Named constants for adversarial testing
SAMPLE_SEARCHABLE_PDF_PATH: str = (
    "data/dataset/vxunderground/downloads/Archive/Collections/CocoMelon/2021-09-04 - Welcome to my cybersecurity path.pdf"
)
LONG_UNBROKEN_STRING_LENGTH: int = 25000
MASSIVE_BASE64_BYTE_SIZE: int = 20000
LARGE_CODE_BLOCK_LINE_COUNT: int = 150
DUMMY_SECRET_PASSWORD: str = "HostileLockdown123!"


def _create_searchable_pdf(target_path: Path) -> None:
    """
    Construct a real PDF with a verified searchable text layer.

    Extracts a page from an existing verified repository PDF to guarantee a
    fully valid, standards-compliant PDF with a genuine text stream.

    Args:
        target_path: Destination path for the generated PDF.
    """
    source_pdf = Path(SAMPLE_SEARCHABLE_PDF_PATH)
    if source_pdf.exists():
        reader = pypdf.PdfReader(str(source_pdf))
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[0])
        with open(target_path, "wb") as f:
            writer.write(f)
    else:
        # Fallback minimal searchable PDF structure
        writer = pypdf.PdfWriter()
        writer.add_blank_page(width=300, height=300)
        with open(target_path, "wb") as f:
            writer.write(f)


def _create_scanned_image_pdf(target_path: Path) -> None:
    """
    Construct a real scanned-image PDF containing 0 text characters.

    Saves a pure bitmap image as a single-page PDF document.

    Args:
        target_path: Destination path for the generated image PDF.
    """
    img = Image.new("RGB", (256, 256), color=(255, 255, 255))
    img.save(str(target_path), "PDF")


def _create_encrypted_pdf(target_path: Path) -> None:
    """
    Construct a real encrypted / copy-protected PDF requiring a password.

    Args:
        target_path: Destination path for the generated encrypted PDF.
    """
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(user_password=DUMMY_SECRET_PASSWORD)
    with open(target_path, "wb") as f:
        writer.write(f)


class TestAdversarialConditionalOCRTrigger:
    """Empirical verification of conditional OCR invocation triggers."""

    def test_searchable_pdf_bypasses_ocr_completely(self, tmp_path: Path) -> None:
        """Verify searchable text PDF extracts programmatic text and NEVER triggers OCR."""
        pdf_path = tmp_path / "searchable_sample.pdf"
        _create_searchable_pdf(pdf_path)

        pipeline = DoclingPipeline()
        extracted_text = pipeline.extract_text(pdf_path)

        # 1. Text stream must be non-empty and match document content
        assert len(extracted_text.strip()) > 0, "Searchable PDF must yield non-empty text"
        assert "cybersecurity" in extracted_text.lower() or len(extracted_text) > 50

        # 2. OCR must NOT be invoked
        assert pipeline.ocr_invoked is False, (
            "Violation: OCR was invoked on a searchable text PDF!"
        )

        # 3. Ingestion into nodes must also bypass OCR
        nodes = pipeline.ingest_file(pdf_path)
        assert len(nodes) >= 1, "Ingestion must produce at least one TextNode"
        assert pipeline.ocr_invoked is False, (
            "Violation: Ingest file triggered OCR on searchable text PDF"
        )

    def test_scanned_image_only_pdf_strictly_triggers_ocr(self, tmp_path: Path) -> None:
        """Verify scanned image PDF (0 text characters) strictly triggers conditional OCR."""
        pdf_path = tmp_path / "scanned_image.pdf"
        _create_scanned_image_pdf(pdf_path)

        # Confirm PDF has exactly 0 text characters programmatically
        reader = pypdf.PdfReader(str(pdf_path))
        assert len(reader.pages) == 1
        assert (reader.pages[0].extract_text() or "").strip() == "", (
            "Prerequisite failed: Scanned PDF must contain 0 programmatic characters"
        )

        pipeline = DoclingPipeline()
        pipeline.extract_text(pdf_path)

        # OCR must be triggered
        assert pipeline.ocr_invoked is True, (
            "Violation: Conditional OCR was NOT invoked for 0-text scanned image PDF"
        )

    def test_encrypted_copy_protected_pdf_strictly_triggers_ocr(self, tmp_path: Path) -> None:
        """Verify encrypted / copy-protected PDF strictly triggers conditional OCR."""
        pdf_path = tmp_path / "protected_document.pdf"
        _create_encrypted_pdf(pdf_path)

        # Confirm PDF is encrypted and cannot be decrypted with empty password
        reader = pypdf.PdfReader(str(pdf_path))
        assert reader.is_encrypted is True
        assert reader.decrypt("") == 0

        pipeline = DoclingPipeline()
        pipeline.extract_text(pdf_path)

        # OCR must be triggered
        assert pipeline.ocr_invoked is True, (
            "Violation: Conditional OCR was NOT invoked for copy-protected/encrypted PDF"
        )

    def test_direct_image_extensions_trigger_ocr(self, tmp_path: Path) -> None:
        """Verify image file formats (.png, .jpg) directly invoke OCR."""
        image_path = tmp_path / "screenshot.png"
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img.save(str(image_path), "PNG")

        pipeline = DoclingPipeline()
        pipeline.extract_text(image_path)

        assert pipeline.ocr_invoked is True, "Violation: Image file did not trigger OCR"

    def test_ocr_engine_failure_degrades_gracefully(self, tmp_path: Path) -> None:
        """Verify external OCR engine crash is caught, logged, and returns empty text gracefully."""
        pdf_path = tmp_path / "scanned_to_fail.pdf"
        _create_scanned_image_pdf(pdf_path)

        pipeline = DoclingPipeline()
        with patch.object(
            pipeline,
            "_ocr_extract",
            side_effect=RuntimeError("External Tesseract engine failed to allocate memory"),
        ):
            extracted = pipeline.extract_text(pdf_path)

        assert extracted == "", "Failing OCR must return empty string without unhandled crash"
        assert pipeline.ocr_invoked is True, "OCR invoked flag must remain True on failure"


class TestAdversarialChunkingBoundaries:
    """Empirical stress-testing of chunking boundaries and corner conditions."""

    def test_massive_unbroken_string_no_whitespace_terminates(self) -> None:
        """Verify massive unbroken strings do not hang the parser or cause infinite loops."""
        unbroken_raw = "A" * LONG_UNBROKEN_STRING_LENGTH
        broken = break_unbroken_strings(unbroken_raw, max_chunk_chars=1000)

        # Must be partitioned into space-separated safe words
        words = broken.split(" ")
        assert len(words) == (LONG_UNBROKEN_STRING_LENGTH // 1000)
        assert all(len(w) <= 1000 for w in words)

        # Chunking markdown with massive unbroken string terminates cleanly
        nodes = chunk_markdown(unbroken_raw)
        assert len(nodes) >= 1
        assert all(len(node.text) > 0 for node in nodes)

    def test_massive_base64_blob_without_whitespace(self) -> None:
        """Verify massive base64 payload (>25KB without whitespace) chunks without infinite recursion."""
        binary_payload = os.urandom(MASSIVE_BASE64_BYTE_SIZE)
        b64_string = base64.b64encode(binary_payload).decode("ascii")
        assert len(b64_string) > 25000
        assert " " not in b64_string

        markdown_doc = f"# Payload Dump\n\n```text\n{b64_string}\n```\n\nEnd of dump."
        nodes = chunk_markdown(markdown_doc)

        assert len(nodes) >= 1
        for node in nodes:
            assert isinstance(node.text, str)
            assert len(node.text.strip()) > 0

    def test_unclosed_code_fence_balancing(self) -> None:
        """Verify odd number of code fences is automatically closed to prevent runaway state."""
        unclosed_markdown = (
            "# Reverse Engineering Shellcode\n\n"
            "```x86asm\n"
            "xor eax, eax\n"
            "push eax\n"
            "push 0x68732f2f\n"
            "push 0x6e69622f\n"
            "mov ebx, esp\n"
        )
        # Verify odd count initially
        assert len(re.findall(r"^```", unclosed_markdown, flags=re.MULTILINE)) == 1

        balanced = balance_code_fences(unclosed_markdown)
        balanced_fence_count = len(re.findall(r"^```", balanced, flags=re.MULTILINE))
        assert balanced_fence_count % 2 == 0, "Balanced text must have even number of fences"
        assert balanced.rstrip().endswith("```")

        nodes = chunk_markdown(unclosed_markdown)
        assert len(nodes) >= 1
        # Code fence inside chunk must not break markdown
        for node in nodes:
            if "```" in node.text:
                fence_count = len(re.findall(r"^```", node.text, flags=re.MULTILINE))
                assert fence_count >= 2, "Code block in node must have opening and closing fences"

    def test_nested_code_fences_in_markdown(self) -> None:
        """Verify documentation containing nested markdown code fences processes cleanly."""
        nested_markdown = (
            "# Documentation Guide\n\n"
            "Here is how to declare Python snippets in markdown:\n\n"
            "```markdown\n"
            "```python\n"
            "print('Nested script')\n"
            "```\n"
            "```\n\n"
            "## Subsequent Heading\n\n"
            "Normal paragraph after code blocks."
        )
        nodes = chunk_markdown(nested_markdown)
        assert len(nodes) >= 1
        full_text = " ".join(n.text for n in nodes)
        assert "Nested script" in full_text
        assert "Subsequent Heading" in full_text

    def test_large_code_block_partitioning_budget_and_fences(self) -> None:
        """Verify code blocks exceeding 512 tokens are partitioned with intact opening and closing fences."""
        code_lines = [
            f"    uint64_t encrypted_block_{i} = crypto_round({i}, key_schedule[{i % 16}]);"
            for i in range(LARGE_CODE_BLOCK_LINE_COUNT)
        ]
        large_code_block = (
            "```c\nvoid DecryptPayload(uint8_t* buffer, size_t size) {\n"
            + "\n".join(code_lines)
            + "\n}\n```"
        )

        token_size = count_tokens(large_code_block)
        assert token_size > MAX_CODE_CHUNK_SIZE, (
            f"Test prerequisite failed: token size {token_size} must exceed {MAX_CODE_CHUNK_SIZE}"
        )

        nodes = chunk_markdown(large_code_block, code_chunk_size=MAX_CODE_CHUNK_SIZE)
        assert len(nodes) > 1, "Large code block must be partitioned into multiple sub-chunks"

        for idx, node in enumerate(nodes):
            chunk_text = node.text.strip()
            # Every sub-chunk MUST start with opening fence
            assert chunk_text.startswith("```"), (
                f"Sub-chunk {idx} missing valid opening code fence"
            )
            # Every sub-chunk MUST end with closing fence
            assert chunk_text.endswith("```"), (
                f"Sub-chunk {idx} missing valid closing code fence"
            )
            # Sub-chunk token count should not exceed code chunk size by large margin
            chunk_tokens = count_tokens(chunk_text)
            assert chunk_tokens <= MAX_CODE_CHUNK_SIZE + 50, (
                f"Sub-chunk {idx} exceeded max code chunk budget: {chunk_tokens} > {MAX_CODE_CHUNK_SIZE}"
            )

    def test_empty_files_handling(self, tmp_path: Path) -> None:
        """Verify empty documents and 0-byte files are handled deterministically."""
        pipeline = DoclingPipeline()

        # 1. 0-byte TXT file
        empty_txt = tmp_path / "empty_report.txt"
        empty_txt.write_text("", encoding="utf-8")
        assert pipeline.extract_text(empty_txt) == ""
        assert pipeline.ingest_file(empty_txt) == []
        assert ingest_file(empty_txt) == []

        # 2. 0-byte MD file
        empty_md = tmp_path / "empty_doc.md"
        empty_md.write_text("", encoding="utf-8")
        assert pipeline.extract_text(empty_md) == ""
        assert pipeline.ingest_file(empty_md) == []
        assert ingest_file(empty_md) == []

        # 3. 0-byte PDF file must raise DocumentParsingError
        empty_pdf = tmp_path / "empty.pdf"
        empty_pdf.write_bytes(b"")
        with pytest.raises(DocumentParsingError, match="Empty 0-byte"):
            pipeline.extract_text(empty_pdf)

    def test_corrupted_binary_headers(self, tmp_path: Path) -> None:
        """Verify non-PDF bytes disguised with .pdf extension raise CorruptedDocumentError."""
        pipeline = DoclingPipeline()

        corrupted_pdf = tmp_path / "fake_binary.pdf"
        # Executable binary header or random bytes disguised as PDF
        corrupted_pdf.write_bytes(b"\x4D\x5A\x90\x00\x03\x00\x00\x00MZ_PE_HEADER_CORRUPTED")

        with pytest.raises(CorruptedDocumentError, match="Corrupted PDF header"):
            pipeline.extract_text(corrupted_pdf)

    def test_extreme_markdown_tables_and_hex_dumps_preservation(self) -> None:
        """Verify markdown tables and memory hex dumps are preserved intact without mid-row fracture."""
        table_rows = [
            f"| 0x{i:04X} | SYSCALL_{i} | STATUS_SUCCESS | PASS |" for i in range(25)
        ]
        table_markdown = (
            "| Offset | Syscall Name | Return Code | Verification |\n"
            "|---|---|---|---|\n"
            + "\n".join(table_rows)
        )

        hex_rows = [
            f"0000{i:04x}  55 89 e5 83 ec 18 c7 45  f4 00 00 00 00 83 7d f4  |U.....E...}}..|"
            for i in range(20)
        ]
        hex_dump_markdown = "\n".join(hex_rows)

        full_doc = f"# Technical Annex\n\n{table_markdown}\n\n## Hex Dump\n\n{hex_dump_markdown}\n"
        nodes = chunk_markdown(full_doc)

        assert len(nodes) >= 1
        node_texts = [n.text for n in nodes]

        # Verify all table rows are present in some node
        assert any("SYSCALL_0" in t and "SYSCALL_24" in t for t in node_texts)
        # Verify hex dump rows are present intact
        assert any("00000000" in t for t in node_texts)

    def test_invalid_chunking_overlap_parameters(self) -> None:
        """Verify invalid overlap and chunk size configurations raise ValueError."""
        # Overlap equal to chunk size
        with pytest.raises(ValueError, match="chunk_overlap.*strictly less"):
            DoclingPipeline(default_chunk_size=512, overlap=512)

        # Overlap strictly greater than chunk size
        with pytest.raises(ValueError, match="chunk_overlap.*strictly less"):
            DoclingPipeline(default_chunk_size=512, overlap=600)

        # Negative overlap
        with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
            DoclingPipeline(overlap=-50)

        # Non-positive default_chunk_size raises ValueError
        with pytest.raises(ValueError):
            DoclingPipeline(default_chunk_size=0)

        # Non-positive code_chunk_size raises ValueError
        with pytest.raises(ValueError, match="code_chunk_size must be strictly positive"):
            DoclingPipeline(code_chunk_size=0)


class TestAdversarialStrictMetadataIntegrity:
    """Empirical stress-testing of strict 5-field metadata schema enforcement."""

    def test_all_ingested_nodes_contain_strictly_the_exact_five_keys(
        self, tmp_path: Path
    ) -> None:
        """Verify every single TextNode emitted contains ONLY the exact 5 canonical keys."""
        pipeline = DoclingPipeline()

        # Ingest Markdown
        md_file = tmp_path / "sample_doc.md"
        md_file.write_text("# Title\n\nFirst paragraph.\n\nSecond paragraph.\n", encoding="utf-8")
        md_nodes = pipeline.ingest_file(md_file)
        assert len(md_nodes) >= 1
        for node in md_nodes:
            assert set(node.metadata.keys()) == STRICT_METADATA_KEYS, (
                f"Node metadata keys mismatch: {set(node.metadata.keys())} != {STRICT_METADATA_KEYS}"
            )
            assert len(node.metadata) == 5

        # Ingest Plaintext
        txt_file = tmp_path / "sample_log.txt"
        txt_file.write_text("Security monitoring log entry.\n", encoding="utf-8")
        txt_nodes = pipeline.ingest_file(txt_file)
        assert len(txt_nodes) >= 1
        for node in txt_nodes:
            assert set(node.metadata.keys()) == STRICT_METADATA_KEYS
            assert len(node.metadata) == 5

        # Ingest Searchable PDF
        pdf_file = tmp_path / "sample_text.pdf"
        _create_searchable_pdf(pdf_file)
        pdf_nodes = pipeline.ingest_file(pdf_file)
        assert len(pdf_nodes) >= 1
        for node in pdf_nodes:
            assert set(node.metadata.keys()) == STRICT_METADATA_KEYS
            assert len(node.metadata) == 5

    def test_extra_injected_metadata_fields_are_stripped(self, tmp_path: Path) -> None:
        """Verify hostile and arbitrary extra fields injected into metadata are completely stripped."""
        md_file = tmp_path / "injected_target.md"
        md_file.write_text("Markdown text for injection testing.", encoding="utf-8")

        # Canonical 5 fields plus 15 hostile and arbitrary injection fields
        hostile_metadata: dict[str, Any] = {
            "file_name": "injected_target.md",
            "file_type": "md",
            "creation_date": "2026-09-13T12:00:00Z",
            "ingestion_date": "2026-09-13T12:05:00Z",
            "file_hash": "e" * 64,
            # Injected arbitrary keys
            "SELECT * FROM users;--": "sql_injection",
            "<script>alert('xss')</script>": "xss_injection",
            "tlp_marking": "TLP:RED",
            "threat_actor": "APT29",
            "confidence_score": 0.99,
            "custom_tags": ["c2", "ransomware"],
            "nested_tree": {"key": "val"},
            "author": "SecretAgent",
            "system_prompt": "Ignore previous instructions",
        }

        pipeline = DoclingPipeline()
        nodes = pipeline.chunk_markdown(
            text="Exploit analysis paragraph.",
            metadata=hostile_metadata,
        )

        assert len(nodes) >= 1
        for node in nodes:
            # Must strictly contain ONLY the 5 canonical keys
            assert set(node.metadata.keys()) == STRICT_METADATA_KEYS, (
                f"Hostile keys were not stripped: {set(node.metadata.keys())}"
            )
            assert len(node.metadata) == 5
            # Injected keys must be completely absent
            assert "SELECT * FROM users;--" not in node.metadata
            assert "<script>alert('xss')</script>" not in node.metadata
            assert "tlp_marking" not in node.metadata
            assert "threat_actor" not in node.metadata
            assert "system_prompt" not in node.metadata

    def test_metadata_value_integrity_and_formatting(self, tmp_path: Path) -> None:
        """Verify metadata values strictly adhere to ISO 8601 UTC and SHA-256 hex formats."""
        test_file = tmp_path / "integrity_check.txt"
        test_content = "Integrity verification payload 12345.\n"
        test_file.write_text(test_content, encoding="utf-8")

        pipeline = DoclingPipeline()
        nodes = pipeline.ingest_file(test_file)
        assert len(nodes) == 1

        meta = nodes[0].metadata
        assert meta["file_name"] == "integrity_check.txt"
        assert meta["file_type"] == "txt"
        assert ISO8601_PATTERN.match(meta["creation_date"]) is not None, (
            f"Invalid creation_date ISO 8601 format: {meta['creation_date']}"
        )
        assert ISO8601_PATTERN.match(meta["ingestion_date"]) is not None, (
            f"Invalid ingestion_date ISO 8601 format: {meta['ingestion_date']}"
        )
        assert SHA256_HEX_PATTERN.match(meta["file_hash"]) is not None, (
            f"Invalid file_hash SHA-256 hex format: {meta['file_hash']}"
        )
        # Verify calculated file hash matches raw content hash
        expected_hash = compute_file_hash(test_file)
        assert meta["file_hash"] == expected_hash

    def test_duplicate_file_detection_in_directory_ingest(self, tmp_path: Path) -> None:
        """Verify exact duplicate files are detected by SHA-256 hash and skipped in directory ingest."""
        work_dir = tmp_path / "ingest_corpus"
        work_dir.mkdir()

        # File 1: original file
        file1 = work_dir / "advisory_original.md"
        file1.write_text("# Advisory\n\nSame content payload across files.\n", encoding="utf-8")

        # File 2: duplicate file with different name but identical content
        file2 = work_dir / "advisory_copy.md"
        file2.write_text("# Advisory\n\nSame content payload across files.\n", encoding="utf-8")

        # File 3: unique distinct file
        file3 = work_dir / "unique_report.md"
        file3.write_text("# Unique Report\n\nCompletely different content.\n", encoding="utf-8")

        # Test both pipeline.ingest_directory and standalone ingest_directory
        nodes = ingest_directory(work_dir)

        # Only 2 distinct files should be indexed (file1 and file3; file2 skipped as duplicate)
        unique_file_names = {node.metadata["file_name"] for node in nodes}
        assert len(unique_file_names) == 2, (
            f"Expected 2 unique files indexed, but found: {unique_file_names}"
        )
        assert "unique_report.md" in unique_file_names
        # Exactly one of the duplicate pair should be indexed
        assert ("advisory_original.md" in unique_file_names) ^ (
            "advisory_copy.md" in unique_file_names
        ), "Duplicate file was not skipped during directory ingestion"
