"""
Tests for the ingestion module using pytest.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.rag.ingest import load_pdfs, save_processed_text, main
from llama_index.core import Document


@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary directories for testing."""
    raw_dir = tmp_path / "raw" / "pdf"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True)
    return raw_dir, processed_dir


def test_load_pdfs_nonexistent_dir():
    """Test load_pdfs with nonexistent directory."""
    nonexistent = Path("nonexistent")
    documents = load_pdfs(nonexistent)
    assert documents == []


def test_load_pdfs_empty_dir(temp_dirs):
    """Test load_pdfs with empty directory."""
    raw_dir, _ = temp_dirs
    with patch('src.rag.ingest.SimpleDirectoryReader') as mock_reader:
        mock_reader.return_value.load_data.return_value = []
        documents = load_pdfs(raw_dir)
        assert documents == []
        mock_reader.assert_called_once()


def test_load_pdfs_with_documents(temp_dirs):
    """Test load_pdfs with mock documents."""
    raw_dir, _ = temp_dirs
    mock_docs = [
        Document(text="Test content 1", metadata={"file_path": str(raw_dir / "test1.pdf")}),
        Document(text="Test content 2", metadata={"file_path": str(raw_dir / "test2.pdf")})
    ]
    with patch('src.rag.ingest.SimpleDirectoryReader') as mock_reader:
        mock_reader.return_value.load_data.return_value = mock_docs
        documents = load_pdfs(raw_dir)
        assert len(documents) == 2
        assert documents[0].text == "Test content 1"


def test_load_pdfs_exception(temp_dirs):
    """Test load_pdfs handles exceptions."""
    raw_dir, _ = temp_dirs
    with patch('src.rag.ingest.SimpleDirectoryReader') as mock_reader:
        mock_reader.return_value.load_data.side_effect = Exception("Load error")
        documents = load_pdfs(raw_dir)
        assert documents == []


def test_save_processed_text_empty_list(temp_dirs):
    """Test save_processed_text with empty document list."""
    _, processed_dir = temp_dirs
    saved_count = save_processed_text([], processed_dir)
    assert saved_count == 0


def test_save_processed_text_valid_documents(temp_dirs):
    """Test save_processed_text with valid documents."""
    _, processed_dir = temp_dirs
    documents = [
        Document(text="Content 1", metadata={"file_path": "test1.pdf"}),
        Document(text="Content 2", metadata={"file_path": "test2.pdf"})
    ]
    saved_count = save_processed_text(documents, processed_dir)
    assert saved_count == 2
    assert (processed_dir / "test1.txt").exists()
    assert (processed_dir / "test2.txt").exists()
    with open(processed_dir / "test1.txt") as f:
        assert f.read() == "Content 1"


def test_save_processed_text_empty_text(temp_dirs):
    """Test save_processed_text skips documents with empty text."""
    _, processed_dir = temp_dirs
    documents = [
        Document(text="", metadata={"file_path": "empty.pdf"}),
        Document(text="Valid content", metadata={"file_path": "valid.pdf"})
    ]
    saved_count = save_processed_text(documents, processed_dir)
    assert saved_count == 1
    assert not (processed_dir / "empty.txt").exists()
    assert (processed_dir / "valid.txt").exists()


def test_save_processed_text_exception(temp_dirs):
    """Test save_processed_text handles file write exceptions."""
    _, processed_dir = temp_dirs
    documents = [
        Document(text="Content", metadata={"file_path": "test.pdf"})
    ]
    with patch('builtins.open', side_effect=Exception("Write error")):
        saved_count = save_processed_text(documents, processed_dir)
        assert saved_count == 0


def test_main_success(temp_dirs, caplog):
    """Test main function with successful execution."""
    raw_dir, processed_dir = temp_dirs
    # Create mock PDF file
    pdf_file = raw_dir / "test.pdf"
    pdf_file.write_text("dummy")  # Not a real PDF, but for directory existence

    mock_docs = [Document(text="Test content", metadata={"file_path": str(pdf_file)})]

    with patch('src.rag.ingest.load_pdfs', return_value=mock_docs), \
         patch('src.rag.ingest.save_processed_text', return_value=1) as mock_save:
        exit_code = main()
        assert exit_code == 0
        mock_save.assert_called_once()


def test_main_no_documents(caplog):
    """Test main function when no documents are loaded."""
    with patch('src.rag.ingest.load_pdfs', return_value=[]):
        exit_code = main()
        assert exit_code == 1
        assert "No documents were loaded" in caplog.text


def test_main_save_failure(caplog):
    """Test main function when saving fails."""
    mock_docs = [Document(text="Content", metadata={"file_path": "test.pdf"})]
    with patch('src.rag.ingest.load_pdfs', return_value=mock_docs), \
         patch('src.rag.ingest.save_processed_text', return_value=0):
        exit_code = main()
        assert exit_code == 1
        assert "No documents were successfully saved" in caplog.text
