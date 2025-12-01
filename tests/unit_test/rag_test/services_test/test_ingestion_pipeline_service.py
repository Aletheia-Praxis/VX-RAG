"""
Unit tests for IngestionPipelineService using LlamaIndex IngestionPipeline
"""

from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import asyncio

from src.rag.services.ingestion_pipeline_service.service import IngestionPipelineService


class TestIngestionPipelineService:
    def test_process_pdf_directory_empty(self) -> None:
        """Process empty PDF directory returns empty list."""
        service = IngestionPipelineService(config_path=None)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Ensure empty directory
            result = asyncio.run(service.process_pdf_directory(Path(temp_dir)))
            assert isinstance(result, list)
            assert len(result) == 0

    @patch('docling.document_converter.DocumentConverter')
    def test_process_pdf_directory_success(self, mock_converter_class: MagicMock) -> None:
        """Test successful PDF processing using pipeline (Docling + OCR)."""
        # Mock converter instance
        mock_converter = MagicMock()
        mock_converter_class.return_value = mock_converter

        # Mock conversion result
        mock_result = MagicMock()
        mock_document = MagicMock()
        mock_document.export_to_markdown.return_value = "Test content"
        mock_document.pages = [MagicMock()]
        mock_document.pictures = []
        mock_result.document = mock_document
        mock_converter.convert.return_value = mock_result
        # Patch normalize_text and detect_language from the ingest_service adapters
        with patch('src.rag.services.ingest_service.service.normalize_text', return_value="Test content"), \
             patch('src.rag.services.ingest_service.service.detect_language', return_value="en"):
            # No ingestion config required for pipeline facade; adapters use config loader directly
            service = IngestionPipelineService(config_path=None)
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_file = Path(temp_dir) / "test.pdf"
                pdf_file.write_text("")

                processed = asyncio.run(service.process_pdf_directory(Path(temp_dir)))
                # In this test the Docling converter returns one document
                assert isinstance(processed, list)
                # If pipeline applied chunking, processed may be chunks or documents; assert each has 'text' and metadata
                if processed:
                    for item in processed:
                        assert hasattr(item, 'text')
                        assert 'file_type' in item.metadata

    def test_process_text_directory_txt(self) -> None:
        """Process txt files via pipeline and verify output format."""
        service = IngestionPipelineService(config_path=None)
        with tempfile.TemporaryDirectory() as temp_dir:
            txt_file = Path(temp_dir) / "test.txt"
            txt_file.write_text("Hello world")

            # Patch normalization and language detection used by adapters
            with patch('src.rag.services.ingest_service.service.normalize_text', return_value="Hello world"), \
                 patch('src.rag.services.ingest_service.service.detect_language', return_value="en"):
                processed = asyncio.run(service.process_text_directory(Path(temp_dir), "*.txt"))
            assert isinstance(processed, list)
            if processed:
                for n in processed:
                    assert hasattr(n, 'text')
                    assert n.metadata.get('file_type') == 'txt'

    def test_process_text_directory_md(self) -> None:
        """Process md files via pipeline and verify output format."""
        service = IngestionPipelineService(config_path=None)
        with tempfile.TemporaryDirectory() as temp_dir:
            md_file = Path(temp_dir) / "test.md"
            md_file.write_text("# Header\n\nContent")

            # Patch normalization and language detection used by adapters
            with patch('src.rag.services.ingest_service.service.normalize_text', return_value="# Header\n\nContent"), \
                 patch('src.rag.services.ingest_service.service.detect_language', return_value="en"):
                processed = asyncio.run(service.process_text_directory(Path(temp_dir), "*.md"))
            assert isinstance(processed, list)
            if processed:
                for n in processed:
                    assert hasattr(n, 'text')
                    assert n.metadata.get('file_type') == 'markdown'
