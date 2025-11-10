"""
Unit tests for ingest service.
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

from src.rag.services.ingest_service.service import PDFIngestAdapter, TXTIngestAdapter, MDIngestAdapter, save_processed_text


class TestPDFIngestAdapter:
    """Test PDF ingest adapter."""
    
    def test_load_data_nonexistent_dir(self) -> None:
        """Test loading from non-existent directory."""
        adapter = PDFIngestAdapter()
        result = adapter.load_data("/nonexistent/path")
        assert result == []
    
    def test_load_data_empty_dir(self) -> None:
        """Test loading from empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = PDFIngestAdapter()
            result = adapter.load_data(temp_dir)
            assert result == []
    
    @patch('src.rag.services.ingest_service.service.DocumentConverter')
    def test_load_data_success(self, mock_converter_class: MagicMock) -> None:
        """Test successful PDF loading."""
        # Mock converter instance
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter
        
        # Mock conversion result
        mock_result = MagicMock()
        mock_document = MagicMock()
        mock_document.export_to_markdown.return_value = "Test content"
        mock_document.pages = [MagicMock()]  # One page
        mock_result.document = mock_document
        mock_converter.convert.return_value = mock_result
        
        # Mock utility functions
        with patch('src.rag.services.ingest_service.service.normalize_text', return_value="Test content"), \
             patch('src.rag.services.ingest_service.service.detect_language', return_value="en"):
            
            adapter = PDFIngestAdapter()
            
            # Create temp dir with a pdf file
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_file = Path(temp_dir) / "test.pdf"
                pdf_file.write_text("")  # Empty file, since we're mocking
                
                result = adapter.load_data(temp_dir)
                
                assert len(result) == 1
                assert result[0]['id'] == "test.pdf_0"
                assert result[0]['text'] == "Test content"
                assert result[0]['metadata']['file_type'] == 'pdf'


class TestTXTIngestAdapter:
    """Test TXT ingest adapter."""
    
    def test_load_data_success(self) -> None:
        """Test successful TXT loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_file = Path(temp_dir) / "test.txt"
            txt_file.write_text("Test content")
            
            adapter = TXTIngestAdapter()
            result = adapter.load_data(temp_dir)
            
            assert len(result) == 1
            assert result[0]['text'] == "Test content"
            assert result[0]['metadata']['file_type'] == 'txt'


class TestMDIngestAdapter:
    """Test MD ingest adapter."""
    
    def test_load_data_success(self) -> None:
        """Test successful MD loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            md_file = Path(temp_dir) / "test.md"
            md_file.write_text("# Test Header\n\nTest content")
            
            adapter = MDIngestAdapter()
            result = adapter.load_data(temp_dir)
            
            assert len(result) == 1
            assert "Test Header" in result[0]['text']
            assert result[0]['metadata']['file_type'] == 'markdown'


class TestSaveProcessedText:
    """Test save processed text function."""
    
    def test_save_processed_text(self) -> None:
        """Test saving processed documents."""
        docs = [
            {
                'id': 'test1',
                'text': 'Test content 1',
                'source': 'test.pdf',
                'lang': 'en',
                'metadata': {'title': 'Test 1'}
            },
            {
                'id': 'test2',
                'text': 'Test content 2',
                'source': 'test2.pdf',
                'lang': 'en',
                'metadata': {'title': 'Test 2'}
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir) / "processed"
            count = save_processed_text(docs, processed_dir)
            
            assert count == 2
            assert (processed_dir / "test1.txt").exists()
            assert (processed_dir / "test1.json").exists()
            assert (processed_dir / "test2.txt").exists()
            assert (processed_dir / "test2.json").exists()
    
    def test_save_empty_text(self) -> None:
        """Test saving document with empty text."""
        docs = [
            {
                'id': 'empty',
                'text': '',
                'metadata': {'title': 'Empty'}
            }
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            processed_dir = Path(temp_dir) / "processed"
            count = save_processed_text(docs, processed_dir)
            
            assert count == 0  # Should skip empty text
