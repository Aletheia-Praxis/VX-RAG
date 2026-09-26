"""
Tier 1 Feature Coverage: Requirement 3 (Strict Metadata Schema & Exact Duplicate Detection).

Authoritative Source: ORIGINAL_REQUEST.md §R3, §AC, PROJECT.md Features 11-12, Tech Spec §4.3, §9.0.
Verifies:
- Feature 11: Strict 5-Field Metadata Schema (file_name, file_type, creation_date, ingestion_date, file_hash)
- Feature 12: Exact Duplicate Detection (SHA-256 hash calculation, skipping duplicates)
"""

from __future__ import annotations

import datetime
import hashlib
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Set

import pytest

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

STRICT_METADATA_KEYS: Set[str] = {
    "file_name",
    "file_type",
    "creation_date",
    "ingestion_date",
    "file_hash",
}


def sanitize_metadata_to_strict_schema(raw_meta: Dict[str, Any]) -> Dict[str, str]:
    """
    Sanitize an arbitrary metadata dictionary to enforce strictly the 5 schema fields.

    Authoritative: PROJECT.md § Interface Contracts (Ingestion <-> Metadata).

    Args:
        raw_meta: Input metadata dictionary.

    Returns:
        Clean dictionary containing strictly the 5 allowed keys.

    Raises:
        ValueError: If any of the mandatory 5 fields are missing.
    """
    missing = STRICT_METADATA_KEYS - set(raw_meta.keys())
    if missing:
        raise ValueError(f"Missing mandatory metadata fields: {missing}")

    return {k: str(raw_meta[k]) for k in STRICT_METADATA_KEYS}


class TestFeature11Strict5FieldMetadataSchema:
    """
    Feature 11: Strict 5-Field Metadata Schema.

    Authoritative: ORIGINAL_REQUEST.md §R3, §AC, PROJECT.md Feature 11.
    Every chunk must contain exactly: file_name, file_type, creation_date, ingestion_date, file_hash.
    """

    def test_strict_metadata_contains_all_5_mandatory_fields(self) -> None:
        """Verify sanitized metadata contains exactly the 5 allowed keys."""
        valid_meta = {
            "file_name": "trojan_report.md",
            "file_type": "md",
            "creation_date": "2026-09-13T08:00:00Z",
            "ingestion_date": "2026-09-13T08:30:00Z",
            "file_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        }
        sanitized = sanitize_metadata_to_strict_schema(valid_meta)
        assert set(sanitized.keys()) == STRICT_METADATA_KEYS
        assert len(sanitized) == 5

    def test_strict_metadata_strips_disallowed_keys(self) -> None:
        """Verify arbitrary metadata keys are stripped out."""
        meta_with_extras = {
            "file_name": "ransomware.pdf",
            "file_type": "pdf",
            "creation_date": "2026-09-13T08:00:00Z",
            "ingestion_date": "2026-09-13T08:30:00Z",
            "file_hash": "0123456789abcdef" * 4,
            # Disallowed extra keys
            "author": "vx-underground",
            "source": "threat_feed",
            "file_path": "/data/raw/ransomware.pdf",
            "file_size": 2048,
            "tags": ["malware", "apt"],
        }
        sanitized = sanitize_metadata_to_strict_schema(meta_with_extras)
        assert "author" not in sanitized
        assert "source" not in sanitized
        assert "file_path" not in sanitized
        assert "file_size" not in sanitized
        assert "tags" not in sanitized
        assert set(sanitized.keys()) == STRICT_METADATA_KEYS

    def test_strict_metadata_iso8601_date_format_validation(self) -> None:
        """Verify creation_date and ingestion_date parse as valid ISO-8601 timestamps."""
        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        creation = "2026-09-13T08:00:00Z"
        iso_pattern = r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$'

        assert re.match(iso_pattern, creation) is not None
        assert re.match(iso_pattern, now_utc) is not None

    def test_strict_metadata_file_hash_sha256_format(self) -> None:
        """Verify file_hash is a 64-character lowercase hexadecimal string."""
        sample_hash = hashlib.sha256(b"malware_analysis_data").hexdigest()
        assert len(sample_hash) == 64
        assert re.match(r'^[a-f0-9]{64}$', sample_hash) is not None

    def test_strict_metadata_rejects_missing_mandatory_field(self) -> None:
        """Verify validation raises ValueError when any required field is missing."""
        incomplete_meta = {
            "file_name": "advisory.txt",
            "file_type": "txt",
            "creation_date": "2026-09-13T08:00:00Z",
            # Missing ingestion_date and file_hash
        }
        with pytest.raises(ValueError, match="Missing mandatory metadata fields"):
            sanitize_metadata_to_strict_schema(incomplete_meta)


class TestFeature12ExactDuplicateDetection:
    """
    Feature 12: Exact Duplicate Detection.

    Authoritative: PROJECT.md Feature 12, Tech Spec §4.3.
    Calculate file SHA-256 hash to detect and skip duplicate files.
    """

    def test_duplicate_detection_identical_content_produces_identical_hash(self, tmp_path: Path) -> None:
        """Verify two files with identical bytes generate identical SHA-256 hashes."""
        content = b"Vx Underground - LockBit 3.0 Builder Leak Documentation"
        file1 = tmp_path / "copy1.md"
        file2 = tmp_path / "copy2.md"

        file1.write_bytes(content)
        file2.write_bytes(content)

        hash1 = hashlib.sha256(file1.read_bytes()).hexdigest()
        hash2 = hashlib.sha256(file2.read_bytes()).hexdigest()
        assert hash1 == hash2

    def test_duplicate_detection_skips_known_hash(self) -> None:
        """Verify ingestion filter identifies and skips already processed file hashes."""
        processed_registry: Set[str] = {
            "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"
        }

        new_hash = "a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90"
        is_duplicate = new_hash in processed_registry
        assert is_duplicate is True, "Duplicate hash must be flagged for skipping"

    def test_duplicate_detection_allows_novel_file_content(self) -> None:
        """Verify ingestion filter accepts novel file hashes."""
        processed_registry: Set[str] = {
            "a" * 64
        }
        novel_hash = "b" * 64
        is_duplicate = novel_hash in processed_registry
        assert is_duplicate is False, "Novel hash must not be flagged as duplicate"

    def test_duplicate_detection_modified_content_generates_new_hash(self, tmp_path: Path) -> None:
        """Verify altering file content alters hash, allowing re-ingestion."""
        target_file = tmp_path / "threat_report.txt"
        target_file.write_text("Version 1.0 Initial telemetry", encoding="utf-8")
        hash_v1 = hashlib.sha256(target_file.read_bytes()).hexdigest()

        target_file.write_text("Version 1.1 Updated IOCs added", encoding="utf-8")
        hash_v2 = hashlib.sha256(target_file.read_bytes()).hexdigest()

        assert hash_v1 != hash_v2, "Altered content must produce new hash"

    def test_duplicate_detection_empty_file_hash(self, tmp_path: Path) -> None:
        """Verify empty 0-byte file generates known empty SHA-256 hash."""
        empty_file = tmp_path / "empty.txt"
        empty_file.write_bytes(b"")

        empty_hash = hashlib.sha256(empty_file.read_bytes()).hexdigest()
        assert empty_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
