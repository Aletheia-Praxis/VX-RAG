"""
Strict 5-field metadata schema and validation for VX-RAG.

Authoritative Source: ORIGINAL_REQUEST.md §R3, §AC, PROJECT.md Feature 11-12, Tech Spec §4.3.
Enforces strictly the 5 allowed metadata fields:
- file_name: File name with extension (e.g. report.pdf)
- file_type: Lowercase file extension without dot (e.g. pdf, txt, md)
- creation_date: File creation timestamp in ISO 8601 UTC format
- ingestion_date: Ingestion timestamp in ISO 8601 UTC format
- file_hash: SHA-256 hexadecimal digest of raw file bytes
"""

from __future__ import annotations

import datetime
import hashlib
import re
import sys
from collections.abc import Collection
from pathlib import Path
from typing import Any

from llama_index.core.schema import BaseNode
from pydantic import BaseModel, ConfigDict, Field, field_validator

STRICT_METADATA_KEYS_ORDERED: tuple[str, ...] = (
    "file_name",
    "file_type",
    "creation_date",
    "ingestion_date",
    "file_hash",
)

STRICT_METADATA_KEYS: set[str] = set(STRICT_METADATA_KEYS_ORDERED)

HASH_CHUNK_SIZE: int = 65536
MAX_TIMESTAMP_SECONDS: float = 253402300799.0

ISO8601_PATTERN: re.Pattern[str] = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)
SHA256_HEX_PATTERN: re.Pattern[str] = re.compile(r"^[a-f0-9]{64}$")


class StrictMetadata(BaseModel):
    """
    Strict 5-field metadata schema container.

    Enforces exactly 5 fields without any extraneous arbitrary tags.
    Provides dictionary-like access and conversion methods.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    file_name: str = Field(..., min_length=1, description="Document file name with extension")
    file_type: str = Field(..., description="Lowercase file extension without leading dot")
    creation_date: str = Field(..., description="File creation timestamp in ISO 8601 UTC format")
    ingestion_date: str = Field(..., description="Ingestion timestamp in ISO 8601 UTC format")
    file_hash: str = Field(..., description="SHA-256 hexadecimal hash digest of file content")

    @field_validator("file_type")
    @classmethod
    def validate_file_type(cls, v: str) -> str:
        """
        Validate and normalize file extension.

        Args:
            v: Input file type string.

        Returns:
            Normalized lowercase file type without leading dot.
        """
        return v.strip().lstrip(".").lower()

    @field_validator("creation_date", "ingestion_date")
    @classmethod
    def validate_iso8601_date(cls, v: str) -> str:
        """
        Validate timestamp adheres to ISO 8601 format and valid calendar date.

        Args:
            v: Timestamp string to validate.

        Returns:
            Validated timestamp string.

        Raises:
            ValueError: If timestamp does not match ISO 8601 format or calendar date is invalid.
        """
        if not ISO8601_PATTERN.match(v):
            raise ValueError(f"Invalid ISO 8601 timestamp format: {v}")
        try:
            datetime.datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid ISO 8601 calendar date/time: {v}") from exc
        return v

    @field_validator("file_hash")
    @classmethod
    def validate_file_hash(cls, v: str) -> str:
        """
        Validate file_hash is a 64-character SHA-256 hexadecimal string.

        Args:
            v: File hash string to validate.

        Returns:
            Normalized lowercase 64-character hexadecimal hash string.

        Raises:
            ValueError: If hash does not conform to SHA-256 format.
        """
        clean = v.strip().lower()
        if not SHA256_HEX_PATTERN.match(clean):
            raise ValueError(
                f"Invalid SHA-256 hash format (expected 64 lowercase hex chars): {v}"
            )
        return clean

    def to_dict(self) -> dict[str, str]:
        """
        Convert metadata to a strict 5-field dictionary.

        Returns:
            Dictionary containing exactly the 5 allowed metadata fields.
        """
        return {
            "file_name": self.file_name,
            "file_type": self.file_type,
            "creation_date": self.creation_date,
            "ingestion_date": self.ingestion_date,
            "file_hash": self.file_hash,
        }

    def __getitem__(self, key: str) -> str:
        """
        Retrieve a metadata field value using dictionary key syntax.

        Args:
            key: Metadata field name.

        Returns:
            String value for the given field.

        Raises:
            KeyError: If key is not one of the strict 5 fields.
        """
        if key in STRICT_METADATA_KEYS:
            return str(getattr(self, key))
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        """
        Check if a field name exists in the strict metadata schema.

        Args:
            key: Key to check.

        Returns:
            True if key is an allowed field name, False otherwise.
        """
        return key in STRICT_METADATA_KEYS

    def __len__(self) -> int:
        """
        Return total field count (always 5).

        Returns:
            Fixed field count of 5.
        """
        return len(STRICT_METADATA_KEYS_ORDERED)

    def get(self, key: str, default: str | None = None) -> str | None:
        """
        Retrieve a field value with an optional default.

        Args:
            key: Field name.
            default: Fallback value if field is absent or not in schema.

        Returns:
            Field value string or default value.
        """
        if key in STRICT_METADATA_KEYS:
            return str(getattr(self, key))
        return default

    def keys(self) -> list[str]:
        """
        Return list of canonical field names.

        Returns:
            List of the 5 canonical field names.
        """
        return list(STRICT_METADATA_KEYS_ORDERED)

    def values(self) -> list[str]:
        """
        Return list of metadata values in canonical order.

        Returns:
            List of the 5 metadata values.
        """
        return [str(getattr(self, k)) for k in STRICT_METADATA_KEYS_ORDERED]

    def items(self) -> list[tuple[str, str]]:
        """
        Return list of (key, value) pairs in canonical order.

        Returns:
            List of (key, value) tuples for all 5 fields.
        """
        return [(k, str(getattr(self, k))) for k in STRICT_METADATA_KEYS_ORDERED]


def compute_file_hash(file_path: Path | str) -> str:
    """
    Calculate SHA-256 hexadecimal hash digest for a given file.

    Reads file in chunks to handle arbitrary file sizes efficiently.

    Args:
        file_path: Path to the target file.

    Returns:
        64-character lowercase hexadecimal SHA-256 hash string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the target path is a directory.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Target path is a directory, not a file: {path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def is_duplicate_hash(file_hash: str, indexed_hashes: Collection[str]) -> bool:
    """
    Check whether a file hash already exists within indexed hashes collection.

    Args:
        file_hash: SHA-256 hash string of the file to check.
        indexed_hashes: Collection or set of previously indexed file hashes.

    Returns:
        True if the hash is already present in indexed_hashes, False otherwise.
    """
    target = file_hash.strip().lower()
    if isinstance(indexed_hashes, (set, frozenset)):
        if target in indexed_hashes:
            return True
        return any(target == h.lower() for h in indexed_hashes)
    return any(target == h.lower() for h in indexed_hashes)


def get_file_creation_date(file_path: Path) -> str:
    """
    Extract file creation timestamp in ISO 8601 UTC format.

    Falls back to last modified date if creation date is unavailable on the OS,
    or current UTC timestamp if filesystem timestamps are corrupted or invalid.

    Args:
        file_path: Path to the target file.

    Returns:
        ISO 8601 formatted UTC timestamp string (e.g. '2026-09-13T08:00:00Z').
    """
    stat = file_path.stat()
    ts: float | None = getattr(stat, "st_birthtime", None)
    if ts is None:
        if sys.platform == "win32":
            ts = stat.st_ctime
        else:
            ts = stat.st_mtime

    try:
        if ts < 0 or ts > MAX_TIMESTAMP_SECONDS:
            raise ValueError(f"Timestamp out of bounds: {ts}")
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OSError, OverflowError):
        try:
            mtime = stat.st_mtime
            if 0 <= mtime <= MAX_TIMESTAMP_SECONDS:
                dt = datetime.datetime.fromtimestamp(mtime, tz=datetime.timezone.utc)
                return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, OSError, OverflowError):
            pass
        return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def extract_document_metadata(file_path: Path | str) -> StrictMetadata:
    """
    Extract strictly validated 5-field metadata from a document file.

    Extracts exactly:
    - file_name: File name with extension
    - file_type: Lowercase extension without leading dot
    - creation_date: File creation timestamp in ISO 8601 UTC format
    - ingestion_date: Ingestion timestamp in ISO 8601 UTC format
    - file_hash: SHA-256 hexadecimal hash digest of file content

    All other arbitrary metadata is disallowed.

    Args:
        file_path: Path to the target document.

    Returns:
        StrictMetadata instance with exactly the 5 allowed fields.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the target path is a directory.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Target path is a directory, not a file: {path}")

    file_name = path.name
    file_type = path.suffix.lstrip(".").lower()
    creation_date = get_file_creation_date(path)
    ingestion_date = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    file_hash = compute_file_hash(path)

    return StrictMetadata(
        file_name=file_name,
        file_type=file_type,
        creation_date=creation_date,
        ingestion_date=ingestion_date,
        file_hash=file_hash,
    )


def sanitize_metadata_to_strict_schema(raw_meta: dict[str, Any]) -> dict[str, str]:
    """
    Sanitize an arbitrary metadata dictionary to enforce strictly the 5 schema fields.

    Authoritative: ORIGINAL_REQUEST.md §R3, PROJECT.md Interface Contracts.

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

    return {k: str(raw_meta[k]) for k in STRICT_METADATA_KEYS_ORDERED}


def sanitize_node_metadata(node: BaseNode) -> BaseNode:
    """
    Sanitize a LlamaIndex BaseNode's metadata to strictly enforce the 5-field schema.

    Strips any extraneous default LlamaIndex metadata (e.g. file_path, file_size,
    last_modified_date, tags) and ensures node.metadata contains ONLY and EXACTLY
    the 5 mandatory schema fields: file_name, file_type, creation_date,
    ingestion_date, file_hash.

    If file_path is present in metadata but file_name is missing, attempts to derive
    missing fields before enforcing strict validation.

    Args:
        node: BaseNode instance whose metadata should be sanitized.

    Returns:
        The same BaseNode instance with node.metadata sanitized to exactly 5 fields.

    Raises:
        ValueError: If any of the 5 mandatory metadata fields are missing and cannot be derived.
    """
    meta: dict[str, Any] = dict(node.metadata) if node.metadata else {}

    # Attempt to derive file_name if missing but file_path exists
    if "file_name" not in meta and "file_path" in meta:
        p = Path(str(meta["file_path"]))
        meta["file_name"] = p.name

    # Attempt to derive file_type if missing but file_name exists
    if "file_type" not in meta and "file_name" in meta:
        meta["file_type"] = Path(str(meta["file_name"])).suffix.lstrip(".").lower()

    # Attempt to derive ingestion_date if missing
    if "ingestion_date" not in meta:
        meta["ingestion_date"] = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )

    # Attempt to derive creation_date if missing and file_path exists
    if "creation_date" not in meta:
        if "file_path" in meta and Path(str(meta["file_path"])).is_file():
            meta["creation_date"] = get_file_creation_date(Path(str(meta["file_path"])))
        elif "last_modified_date" in meta:
            raw_dt = str(meta["last_modified_date"])
            if ISO8601_PATTERN.match(raw_dt):
                meta["creation_date"] = raw_dt
            else:
                meta["creation_date"] = datetime.datetime.now(
                    datetime.timezone.utc
                ).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Attempt to derive file_hash if missing and file_path exists
    if "file_hash" not in meta and "file_path" in meta and Path(str(meta["file_path"])).is_file():
        meta["file_hash"] = compute_file_hash(Path(str(meta["file_path"])))

    # Validate against strict schema
    sanitized = sanitize_metadata_to_strict_schema(meta)

    # Validate with StrictMetadata model to guarantee format validity
    validated = StrictMetadata(**sanitized)

    # Assign sanitized dict directly to node.metadata
    node.metadata = validated.to_dict()

    # Clean up excluded metadata keys if present on node
    if hasattr(node, "excluded_embed_metadata_keys"):
        node.excluded_embed_metadata_keys = [
            k for k in node.excluded_embed_metadata_keys if k in STRICT_METADATA_KEYS
        ]
    if hasattr(node, "excluded_llm_metadata_keys"):
        node.excluded_llm_metadata_keys = [
            k for k in node.excluded_llm_metadata_keys if k in STRICT_METADATA_KEYS
        ]

    return node

