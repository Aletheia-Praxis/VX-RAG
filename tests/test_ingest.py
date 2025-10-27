"""
Tests for the ingestion service using pytest.
"""

import pytest
from pathlib import Path
from typing import Tuple
from unittest.mock import patch
from src.rag.services.ingest_service.service import PDFIngestAdapter, save_processed_text
from llama_index.core import Document


@pytest.fixture
def temp_dirs(tmp_path: Path) -> Tuple[Path, Path]:
    """Create temporary directories for testing."""
    raw_dir = tmp_path / "raw" / "pdf"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    return raw_dir, processed_dir


def test_load_data_nonexistent_dir() -> None:
    """Test load_data with nonexistent directory."""
    adapter = PDFIngestAdapter()
    documents = adapter.load_data("nonexistent")
    assert documents == []  # nosec B101


def test_load_data_empty_dir(temp_dirs: Tuple[Path, Path]) -> None:
    """Test load_data with empty directory."""
    adapter = PDFIngestAdapter()
    raw_dir, _ = temp_dirs
    with patch('src.rag.services.ingest_service.service.SimpleDirectoryReader') as mock_reader:
        mock_reader.return_value.load_data.return_value = []
        documents = adapter.load_data(str(raw_dir))
        assert documents == []  # nosec B101
        mock_reader.assert_called_once()


def test_load_data_with_documents(temp_dirs: Tuple[Path, Path]) -> None:
    """Test load_data with mock documents."""
    adapter = PDFIngestAdapter()
    raw_dir, _ = temp_dirs
    mock_docs = [
        Document(text="Test content 1", metadata={"file_path": str(raw_dir / "test1.pdf")}),
        Document(text="Test content 2", metadata={"file_path": str(raw_dir / "test2.pdf")})
    ]
    with patch('src.rag.services.ingest_service.service.SimpleDirectoryReader') as mock_reader:
        mock_reader.return_value.load_data.return_value = mock_docs
        documents = adapter.load_data(str(raw_dir))
        assert len(documents) == 2  # nosec B101
        assert documents[0]['text'] == "Test content 1"  # nosec B101


def test_load_data_exception(temp_dirs: Tuple[Path, Path]) -> None:
    """Test load_data handles exceptions."""
    adapter = PDFIngestAdapter()
    raw_dir, _ = temp_dirs
    with patch('src.rag.services.ingest_service.service.SimpleDirectoryReader') as mock_reader:
        mock_reader.return_value.load_data.side_effect = Exception("Load error")
        documents = adapter.load_data(str(raw_dir))
        assert documents == []  # nosec B101


def test_save_processed_text_empty_list(temp_dirs: Tuple[Path, Path]) -> None:
    """Test save_processed_text with empty document list."""
    _, processed_dir = temp_dirs
    saved_count = save_processed_text([], processed_dir)
    assert saved_count == 0  # nosec B101


def test_save_processed_text_valid_documents(temp_dirs: Tuple[Path, Path]) -> None:
    """Test save_processed_text with valid documents."""
    _, processed_dir = temp_dirs
    documents = [
        {'text': "Content 1", 'metadata': {"file_path": "test1.pdf"}},
        {'text': "Content 2", 'metadata': {"file_path": "test2.pdf"}}
    ]
    saved_count = save_processed_text(documents, processed_dir)
    assert saved_count == 2  # nosec B101
    assert (processed_dir / "test1.txt").exists()  # nosec B101
    assert (processed_dir / "test2.txt").exists()  # nosec B101
    with open(processed_dir / "test1.txt") as f:
        assert f.read() == "Content 1"  # nosec B101


def test_save_processed_text_empty_text(temp_dirs: Tuple[Path, Path]) -> None:
    """Test save_processed_text skips documents with empty text."""
    _, processed_dir = temp_dirs
    documents = [
        {'text': "", 'metadata': {"file_path": "empty.pdf"}},
        {'text': "Valid content", 'metadata': {"file_path": "valid.pdf"}}
    ]
    saved_count = save_processed_text(documents, processed_dir)
    assert saved_count == 1  # nosec B101
    assert not (processed_dir / "empty.txt").exists()  # nosec B101
    assert (processed_dir / "valid.txt").exists()  # nosec B101


def test_save_processed_text_exception(temp_dirs: Tuple[Path, Path]) -> None:
    """Test save_processed_text handles file write exceptions."""
    _, processed_dir = temp_dirs
    documents = [
        {'text': "Content", 'metadata': {"file_path": "test.pdf"}}
    ]
    with patch('builtins.open', side_effect=Exception("Write error")):
        saved_count = save_processed_text(documents, processed_dir)
        assert saved_count == 0  # nosec B101
