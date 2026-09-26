"""
Unit tests for strict 5-field metadata schema, extraction, sanitization, and duplicate detection.

Authoritative Source: ORIGINAL_REQUEST.md §R3, §AC, PROJECT.md Features 11-12, Tech Spec §4.3.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from pydantic import ValidationError

from src.rag.libs.postprocessors import MetadataBoostPostprocessor
from src.rag.metadata import (
    ISO8601_PATTERN,
    STRICT_METADATA_KEYS,
    StrictMetadata,
    compute_file_hash,
    extract_document_metadata,
    get_file_creation_date,
    is_duplicate_hash,
    sanitize_metadata_to_strict_schema,
    sanitize_node_metadata,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture  # noqa: F401

EMPTY_FILE_SHA256: str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


class TestMetadataExtraction:
    """Tests for extracting strict metadata from files."""

    def test_extract_document_metadata_returns_all_five_canonical_fields(
        self, tmp_path: Path
    ) -> None:
        """Verify extract_document_metadata returns StrictMetadata with exactly the 5 schema fields."""
        file_path = tmp_path / "apt29_analysis.pdf"
        content = b"APT29 cyber espionage telemetry report"
        file_path.write_bytes(content)

        metadata = extract_document_metadata(file_path)

        assert isinstance(metadata, StrictMetadata)
        assert set(metadata.keys()) == STRICT_METADATA_KEYS
        assert len(metadata) == 5
        assert metadata.file_name == "apt29_analysis.pdf"
        assert metadata.file_type == "pdf"
        assert metadata.file_hash == hashlib.sha256(content).hexdigest()
        assert re.match(ISO8601_PATTERN, metadata.creation_date) is not None
        assert re.match(ISO8601_PATTERN, metadata.ingestion_date) is not None

    def test_extract_document_metadata_dictionary_access_and_conversion(
        self, tmp_path: Path
    ) -> None:
        """Verify StrictMetadata supports dictionary key access, items, and to_dict."""
        file_path = tmp_path / "threat_intel.json"
        file_path.write_bytes(b'{"ioc": "192.0.2.1"}')

        metadata = extract_document_metadata(file_path)

        # Mapping / dict-like access
        assert metadata["file_name"] == "threat_intel.json"
        assert metadata["file_type"] == "json"
        assert metadata.get("file_name") == "threat_intel.json"
        assert metadata.get("nonexistent", "default_val") == "default_val"
        assert "file_name" in metadata
        assert "arbitrary_tag" not in metadata

        # to_dict conversion
        as_dict = metadata.to_dict()
        assert isinstance(as_dict, dict)
        assert set(as_dict.keys()) == STRICT_METADATA_KEYS
        assert len(as_dict) == 5

        # Unknown key access raises KeyError
        with pytest.raises(KeyError):
            _ = metadata["invalid_field"]

    def test_extract_document_metadata_nonexistent_file_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Verify attempting metadata extraction on a missing file raises FileNotFoundError."""
        missing_file = tmp_path / "does_not_exist.txt"
        with pytest.raises(FileNotFoundError, match="File not found"):
            extract_document_metadata(missing_file)

    def test_extract_document_metadata_directory_path_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Verify passing a directory path raises IsADirectoryError."""
        sub_dir = tmp_path / "sub_folder"
        sub_dir.mkdir()
        with pytest.raises(IsADirectoryError, match="Target path is a directory"):
            extract_document_metadata(sub_dir)


class TestMetadataSanitization:
    """Tests for sanitizing metadata dictionaries and BaseNodes to strict schema."""

    def test_sanitize_metadata_to_strict_schema_strips_disallowed_keys(self) -> None:
        """Verify arbitrary extra keys are stripped while preserving mandatory keys."""
        raw_meta: dict[str, Any] = {
            "file_name": "rootkit.sys",
            "file_type": "sys",
            "creation_date": "2026-09-13T09:00:00Z",
            "ingestion_date": "2026-09-13T09:15:00Z",
            "file_hash": "a" * 64,
            # Extra arbitrary keys to strip
            "author": "malware_analyst",
            "source": "virustotal",
            "threat_actor": "lazarus",
            "file_size": 1048576,
            "tags": ["driver", "rootkit"],
        }

        sanitized = sanitize_metadata_to_strict_schema(raw_meta)

        assert set(sanitized.keys()) == STRICT_METADATA_KEYS
        assert len(sanitized) == 5
        assert "author" not in sanitized
        assert "source" not in sanitized
        assert "threat_actor" not in sanitized
        assert "file_size" not in sanitized
        assert "tags" not in sanitized

    def test_sanitize_metadata_to_strict_schema_missing_mandatory_key_raises_error(self) -> None:
        """Verify sanitize_metadata_to_strict_schema raises ValueError if required keys are missing."""
        incomplete_meta: dict[str, Any] = {
            "file_name": "kernel_exploit.c",
            "file_type": "c",
            "creation_date": "2026-09-13T09:00:00Z",
            # Missing ingestion_date and file_hash
        }
        with pytest.raises(ValueError, match="Missing mandatory metadata fields"):
            sanitize_metadata_to_strict_schema(incomplete_meta)

    def test_sanitize_node_metadata_strips_extraneous_llamaindex_metadata(self) -> None:
        """Verify sanitize_node_metadata enforces strictly 5 fields on BaseNode."""
        node = TextNode(
            text="Payload shellcode execution section",
            metadata={
                "file_name": "dropper.exe",
                "file_type": "exe",
                "creation_date": "2026-09-13T09:00:00Z",
                "ingestion_date": "2026-09-13T09:30:00Z",
                "file_hash": "b" * 64,
                # Extraneous fields typical from LlamaIndex loaders
                "file_path": "/var/data/dropper.exe",
                "file_size": 4096,
                "last_modified_date": "2026-09-13T09:00:00Z",
                "author": "vx-team",
            },
        )

        sanitized_node = sanitize_node_metadata(node)

        assert set(sanitized_node.metadata.keys()) == STRICT_METADATA_KEYS
        assert len(sanitized_node.metadata) == 5
        assert "file_path" not in sanitized_node.metadata
        assert "file_size" not in sanitized_node.metadata
        assert "last_modified_date" not in sanitized_node.metadata
        assert "author" not in sanitized_node.metadata

    def test_sanitize_node_metadata_derives_fields_from_file_path(
        self, tmp_path: Path
    ) -> None:
        """Verify sanitize_node_metadata automatically derives missing fields if file_path exists on disk."""
        target_file = tmp_path / "ransomware_notes.txt"
        target_file.write_text("All your files are encrypted", encoding="utf-8")

        node = TextNode(
            text="Decryption instructions snippet",
            metadata={
                "file_path": str(target_file),
                # file_name, file_type, creation_date, ingestion_date, file_hash are missing
            },
        )

        sanitized_node = sanitize_node_metadata(node)

        assert set(sanitized_node.metadata.keys()) == STRICT_METADATA_KEYS
        assert len(sanitized_node.metadata) == 5
        assert sanitized_node.metadata["file_name"] == "ransomware_notes.txt"
        assert sanitized_node.metadata["file_type"] == "txt"
        assert sanitized_node.metadata["file_hash"] == compute_file_hash(target_file)
        assert re.match(ISO8601_PATTERN, sanitized_node.metadata["creation_date"]) is not None
        assert re.match(ISO8601_PATTERN, sanitized_node.metadata["ingestion_date"]) is not None
        assert "file_path" not in sanitized_node.metadata

    def test_sanitize_node_metadata_missing_irrecoverable_fields_raises_error(self) -> None:
        """Verify sanitize_node_metadata raises ValueError if mandatory fields cannot be resolved."""
        node = TextNode(
            text="Fragment without origin",
            metadata={"random_tag": "orphaned_chunk"},
        )
        with pytest.raises(ValueError, match="Missing mandatory metadata fields"):
            sanitize_node_metadata(node)


class TestDuplicateDetection:
    """Tests for exact duplicate file detection using SHA-256."""

    def test_compute_file_hash_generates_correct_sha256(self, tmp_path: Path) -> None:
        """Verify compute_file_hash computes accurate SHA-256 hexadecimal string."""
        sample_file = tmp_path / "beacon.bin"
        payload = b"\x90\x90\x90\xcc\xc3"
        sample_file.write_bytes(payload)

        expected_hash = hashlib.sha256(payload).hexdigest()
        computed = compute_file_hash(sample_file)

        assert computed == expected_hash
        assert len(computed) == 64

    def test_compute_file_hash_nonexistent_file_raises_error(
        self, tmp_path: Path
    ) -> None:
        """Verify compute_file_hash raises FileNotFoundError for nonexistent file."""
        missing = tmp_path / "ghost.bin"
        with pytest.raises(FileNotFoundError, match="File not found"):
            compute_file_hash(missing)

    def test_compute_file_hash_directory_raises_error(self, tmp_path: Path) -> None:
        """Verify compute_file_hash raises IsADirectoryError when target is a directory."""
        dir_path = tmp_path / "directory_target"
        dir_path.mkdir()
        with pytest.raises(IsADirectoryError, match="Target path is a directory"):
            compute_file_hash(dir_path)

    def test_is_duplicate_hash_detects_existing_hash(self) -> None:
        """Verify is_duplicate_hash returns True when hash matches indexed registry."""
        known_hash = "c" * 64
        indexed_set: set[str] = {known_hash, "d" * 64}

        assert is_duplicate_hash(known_hash, indexed_set) is True
        # Case insensitive check
        assert is_duplicate_hash(known_hash.upper(), indexed_set) is True

    def test_is_duplicate_hash_allows_novel_hash(self) -> None:
        """Verify is_duplicate_hash returns False when hash is absent from registry."""
        indexed_set: set[str] = {"e" * 64}
        novel_hash = "f" * 64

        assert is_duplicate_hash(novel_hash, indexed_set) is False


class TestBoundaryCases:
    """Tests for edge and boundary cases (empty files, corrupted timestamps, strict validation)."""

    def test_empty_file_metadata_extraction_and_hash(self, tmp_path: Path) -> None:
        """Verify empty 0-byte file extraction generates correct hash and empty file_type."""
        empty_file = tmp_path / "zero_bytes.dat"
        empty_file.write_bytes(b"")

        meta = extract_document_metadata(empty_file)

        assert meta.file_hash == EMPTY_FILE_SHA256
        assert meta.file_name == "zero_bytes.dat"
        assert meta.file_type == "dat"
        assert len(meta.to_dict()) == 5

    def test_corrupted_negative_timestamp_falls_back_gracefully(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        """Verify negative or out-of-range timestamp falls back to valid ISO 8601 UTC timestamp."""
        test_file = tmp_path / "corrupted_timestamp.txt"
        test_file.write_text("Telemetry with bad stat timestamps", encoding="utf-8")

        class MockCorruptedStat:
            """Mock stat result with corrupted creation timestamp."""

            st_birthtime: float = -1000.0
            st_ctime: float = -1000.0
            st_mtime: float = -1000.0

        monkeypatch.setattr(Path, "stat", lambda _self: MockCorruptedStat())

        creation_date = get_file_creation_date(test_file)
        assert re.match(ISO8601_PATTERN, creation_date) is not None

        # Verify full metadata extraction survives corrupted timestamps
        meta = extract_document_metadata(test_file)
        assert re.match(ISO8601_PATTERN, meta.creation_date) is not None

    def test_strict_metadata_pydantic_validation_rejects_extra_fields(self) -> None:
        """Verify StrictMetadata rejects extra unauthorized metadata fields at instantiation."""
        with pytest.raises(ValidationError):
            StrictMetadata(  # type: ignore[call-arg]
                file_name="exploit.py",
                file_type="py",
                creation_date="2026-09-13T09:00:00Z",
                ingestion_date="2026-09-13T09:30:00Z",
                file_hash="0" * 64,
                unauthorized_tag="disallowed",
            )

    def test_strict_metadata_pydantic_validation_rejects_invalid_hash(self) -> None:
        """Verify StrictMetadata rejects non-SHA-256 formatted hash strings."""
        with pytest.raises(ValidationError):
            StrictMetadata(
                file_name="exploit.py",
                file_type="py",
                creation_date="2026-09-13T09:00:00Z",
                ingestion_date="2026-09-13T09:30:00Z",
                file_hash="short_hash_123",
            )

    def test_strict_metadata_pydantic_validation_rejects_invalid_iso_date(self) -> None:
        """Verify StrictMetadata rejects non-ISO-8601 date strings."""
        with pytest.raises(ValidationError):
            StrictMetadata(
                file_name="exploit.py",
                file_type="py",
                creation_date="invalid-non-iso-date",
                ingestion_date="2026-09-13T09:30:00Z",
                file_hash="0" * 64,
            )

    def test_strict_metadata_pydantic_validation_rejects_semantically_invalid_calendar_date(
        self,
    ) -> None:
        """Verify StrictMetadata rejects semantically impossible calendar dates with valid regex syntax."""
        with pytest.raises(ValidationError, match="Invalid ISO 8601 calendar date/time"):
            StrictMetadata(
                file_name="exploit.py",
                file_type="py",
                creation_date="2026-13-45T99:99:99Z",
                ingestion_date="2026-09-13T09:30:00Z",
                file_hash="0" * 64,
            )

    def test_validate_file_type_strips_leading_trailing_spaces_and_dot(self) -> None:
        """Verify file_type validator strips leading/trailing spaces and leading dot properly."""
        meta = StrictMetadata(
            file_name="exploit.docx",
            file_type="  .DoCx   ",
            creation_date="2026-09-13T09:00:00Z",
            ingestion_date="2026-09-13T09:30:00Z",
            file_hash="0" * 64,
        )
        assert meta.file_type == "docx"

    def test_sanitize_node_metadata_normalizes_file_type_and_hash(self) -> None:
        """Verify sanitize_node_metadata writes normalized file_type and file_hash directly into node.metadata."""
        node = TextNode(
            text="Unnormalized metadata node payload",
            metadata={
                "file_name": "SAMPLE.PDF",
                "file_type": "  .PDF  ",
                "creation_date": "2026-09-13T09:00:00Z",
                "ingestion_date": "2026-09-13T09:30:00Z",
                "file_hash": "A" * 64,
            },
        )
        sanitized_node = sanitize_node_metadata(node)
        assert sanitized_node.metadata["file_type"] == "pdf"
        assert sanitized_node.metadata["file_hash"] == "a" * 64


class TestMetadataBoostPostprocessorAlignment:
    """Tests for MetadataBoostPostprocessor aligned with the strict 5-field schema."""

    def test_metadata_boost_postprocessor_boosts_populated_strict_fields(self) -> None:
        """Verify score boosting operates on populated strict schema fields."""
        node = TextNode(
            text="Document chunk text",
            metadata={
                "file_name": "advisory.pdf",
                "file_type": "pdf",
                "creation_date": "2026-09-13T09:00:00Z",
                "ingestion_date": "2026-09-13T09:30:00Z",
                "file_hash": "a" * 64,
            },
        )
        node_with_score = NodeWithScore(node=node, score=1.0)
        postprocessor = MetadataBoostPostprocessor(boost_factor=0.1)

        result = postprocessor.postprocess_nodes(
            [node_with_score], query_bundle=QueryBundle(query_str="test")
        )

        # 5 populated fields * 0.1 boost = 1.0 * (1 + 0.5) = 1.5
        assert len(result) == 1
        assert pytest.approx(result[0].score, 0.01) == 1.5

    def test_metadata_boost_postprocessor_gracefully_ignores_legacy_fields(self) -> None:
        """Verify postprocessor ignores legacy fields (source, lang, author) without raising errors."""
        node = TextNode(
            text="Document chunk text",
            metadata={
                # Only legacy fields populated, no strict schema fields
                "source": "threat_intel",
                "lang": "en",
                "author": "vx-team",
            },
        )
        node_with_score = NodeWithScore(node=node, score=1.0)
        # Passing legacy priority_fields should be filtered out
        postprocessor = MetadataBoostPostprocessor(
            boost_factor=0.1,
            priority_fields=["source", "lang", "topic", "author", "year"],
        )

        result = postprocessor.postprocess_nodes([node_with_score])

        # Since legacy fields are filtered out and no strict fields are populated, score remains 1.0
        assert len(result) == 1
        assert pytest.approx(result[0].score, 0.01) == 1.0
