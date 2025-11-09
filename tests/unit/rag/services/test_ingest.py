"""
Tests for the ingestion service using pytest.
"""

import pytest
from pathlib import Path
from typing import Tuple
from unittest.mock import patch, MagicMock
from src.rag.services.ingest_service.service import (
    PDFIngestAdapter, TXTIngestAdapter, MDIngestAdapter,
    save_processed_text
)


@pytest.fixture
def temp_dirs(tmp_path: Path) -> Tuple[Path, Path]:
    """Create temporary directories for testing."""
    raw_dir = tmp_path / "raw" / "pdf"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    return raw_dir, processed_dir


def test_pdf_load_data_nonexistent_dir() -> None:
    """Test PDF load_data with nonexistent directory."""
    adapter = PDFIngestAdapter()
    documents = adapter.load_data("nonexistent")
    assert documents == []  # nosec B101


def test_pdf_load_data_empty_dir(temp_dirs: Tuple[Path, Path]) -> None:
    """Test PDF load_data with empty directory."""
    adapter = PDFIngestAdapter()
    raw_dir, _ = temp_dirs
    with patch('src.rag.services.ingest_service.service.DocumentConverter') as mock_converter:
        mock_converter.return_value.convert.return_value.document.export_to_markdown.return_value = ""
        documents = adapter.load_data(str(raw_dir))
        assert documents == []  # nosec B101
        # DocumentConverter should not be called for empty directory
        mock_converter.assert_not_called()


def test_pdf_load_data_with_documents(temp_dirs: Tuple[Path, Path]) -> None:
    """Test PDF load_data with mock documents."""
    raw_dir, _ = temp_dirs
    
    # Create a mock PDF file
    pdf_file = raw_dir / "test1.pdf"
    pdf_file.write_bytes(b"dummy pdf content")
    
    # Mock the conversion result
    mock_document = type('MockDocument', (), {
        'export_to_markdown': MagicMock(return_value="Test content 1"),
        'pages': ['page1', 'page2']
    })()
    mock_conversion_result = type('MockConversionResult', (), {'document': mock_document})()
    
    with patch('src.rag.services.ingest_service.service.DocumentConverter') as mock_converter_class:
        mock_converter_instance = mock_converter_class.return_value
        mock_converter_instance.convert.return_value = mock_conversion_result
        
        # Create adapter after patching
        adapter = PDFIngestAdapter()
        
        documents = adapter.load_data(str(raw_dir))
        assert len(documents) == 1  # nosec B101
        assert documents[0]['text'] == "Test content 1"  # nosec B101
        assert 'id' in documents[0]  # nosec B101
        assert 'source' in documents[0]  # nosec B101
        assert 'lang' in documents[0]  # nosec B101
        assert 'metadata' in documents[0]  # nosec B101
        assert documents[0]['metadata']['file_type'] == 'pdf'  # nosec B101
        assert documents[0]['metadata']['page_count'] == 2  # nosec B101
        mock_converter_class.assert_called_once()


def test_pdf_load_data_exception(temp_dirs: Tuple[Path, Path]) -> None:
    """Test PDF load_data handles exceptions."""
    adapter = PDFIngestAdapter()
    raw_dir, _ = temp_dirs
    
    # Create a mock PDF file
    pdf_file = raw_dir / "test1.pdf"
    pdf_file.write_bytes(b"dummy pdf content")
    
    with patch('src.rag.services.ingest_service.service.DocumentConverter') as mock_converter:
        mock_converter.return_value.convert.side_effect = Exception("Load error")
        documents = adapter.load_data(str(raw_dir))
        assert documents == []  # nosec B101


def test_txt_load_data_nonexistent_dir() -> None:
    """Test TXT load_data with nonexistent directory."""
    adapter = TXTIngestAdapter()
    documents = adapter.load_data("nonexistent")
    assert documents == []  # nosec B101


def test_txt_load_data_with_files(tmp_path: Path) -> None:
    """Test TXT load_data with text files."""
    adapter = TXTIngestAdapter()
    raw_dir = tmp_path / "raw" / "txt"
    raw_dir.mkdir(parents=True)
    
    # Create test files with ASCII content
    file1 = raw_dir / "test1.txt"
    file1.write_text("Hello world")
    file2 = raw_dir / "test2.txt"
    file2.write_text("Second file content")
    
    documents = adapter.load_data(str(raw_dir))
    assert len(documents) == 2  # nosec B101
    assert documents[0]['text'] == "Hello world"  # nosec B101
    assert documents[1]['text'] == "Second file content"  # nosec B101
    # Language detection may vary, just check that it's present
    assert 'lang' in documents[0]  # nosec B101
    assert 'lang' in documents[1]  # nosec B101


def test_md_load_data_with_files(tmp_path: Path) -> None:
    """Test MD load_data with markdown files."""
    adapter = MDIngestAdapter()
    raw_dir = tmp_path / "raw" / "md"
    raw_dir.mkdir(parents=True)
    
    # Create test file
    file1 = raw_dir / "test.md"
    file1.write_text("# Header\n\nSome content")
    
    documents = adapter.load_data(str(raw_dir))
    assert len(documents) == 1  # nosec B101
    assert "# Header" in documents[0]['text']  # nosec B101
    assert documents[0]['metadata']['file_type'] == "markdown"  # nosec B101


def test_save_processed_text_empty_list(temp_dirs: Tuple[Path, Path]) -> None:
    """Test save_processed_text with empty document list."""
    _, processed_dir = temp_dirs
    saved_count = save_processed_text([], processed_dir)
    assert saved_count == 0  # nosec B101


def test_save_processed_text_valid_documents(temp_dirs: Tuple[Path, Path]) -> None:
    """Test save_processed_text with valid documents."""
    _, processed_dir = temp_dirs
    documents = [
        {'id': 'doc1', 'source': 'test1.pdf', 'text': "Content 1", 'lang': 'en', 'metadata': {"file_path": "test1.pdf"}},
        {'id': 'doc2', 'source': 'test2.pdf', 'text': "Content 2", 'lang': 'en', 'metadata': {"file_path": "test2.pdf"}}
    ]
    saved_count = save_processed_text(documents, processed_dir)
    assert saved_count == 2  # nosec B101
    assert (processed_dir / "doc1.txt").exists()  # nosec B101
    assert (processed_dir / "doc1.json").exists()  # nosec B101


def test_save_processed_text_empty_text(temp_dirs: Tuple[Path, Path]) -> None:
    """Test save_processed_text skips documents with empty text."""
    _, processed_dir = temp_dirs
    documents = [
        {'id': 'empty', 'source': 'empty.pdf', 'text': "", 'lang': 'en', 'metadata': {}},
        {'id': 'valid', 'source': 'valid.pdf', 'text': "Valid content", 'lang': 'en', 'metadata': {}}
    ]
    saved_count = save_processed_text(documents, processed_dir)
    assert saved_count == 1  # nosec B101
    assert not (processed_dir / "empty.txt").exists()  # nosec B101
    assert (processed_dir / "valid.txt").exists()  # nosec B101
