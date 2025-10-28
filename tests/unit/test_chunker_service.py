"""
Unit tests for chunker_service module.
"""

import pytest
from typing import TYPE_CHECKING
from unittest.mock import patch, MagicMock

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

from src.rag.services.chunker_service.service import Chunker, ChunkMetadata


class TestChunker:
    """Test cases for Chunker class."""

    def test_init_default(self) -> None:
        """Test Chunker initialization with default parameters."""
        chunker = Chunker()
        assert chunker.chunk_size == 1500
        assert chunker.chunk_overlap == 200
        assert chunker.separator == "\n"
        assert not chunker.use_semantic_chunking

    def test_init_custom(self) -> None:
        """Test Chunker initialization with custom parameters."""
        chunker = Chunker(chunk_size=1000, chunk_overlap=100, separator=".", use_semantic_chunking=True)
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 100
        assert chunker.separator == "."
        assert chunker.use_semantic_chunking

    def test_chunk_documents_empty(self) -> None:
        """Test chunk_documents with empty list."""
        chunker = Chunker()
        result = chunker.chunk_documents([])
        assert result == []

    def test_chunk_documents_single(self) -> None:
        """Test chunk_documents with single document."""
        chunker = Chunker()
        documents = [{
            'id': 'doc1',
            'source': 'test.txt',
            'text': 'This is a short test document.',
            'lang': 'en',
            'metadata': {'test': True}
        }]
        
        with patch('llama_index.core.node_parser.SentenceSplitter.get_nodes_from_documents') as mock_get_nodes:
            # Mock node
            mock_node = MagicMock()
            mock_node.get_content.return_value = 'This is a short test document.'
            mock_node.start_char_idx = 0
            mock_node.end_char_idx = 30
            mock_node.node_info = {}
            mock_node.relationships = {}
            
            mock_get_nodes.return_value = [mock_node]
            
            result = chunker.chunk_documents(documents)
            
            assert len(result) == 1
            assert result[0]['id'] == 'doc1_chunk_0'
            assert result[0]['text'] == 'This is a short test document.'
            assert result[0]['metadata']['chunk_metadata']['parent_id'] == 'doc1'
            assert result[0]['metadata']['chunk_metadata']['lang'] == 'en'

    def test_chunk_documents_multiple(self) -> None:
        """Test chunk_documents with multiple documents."""
        chunker = Chunker()
        documents = [
            {
                'id': 'doc1',
                'source': 'test1.txt',
                'text': 'First document.',
                'lang': 'en',
                'metadata': {}
            },
            {
                'id': 'doc2',
                'source': 'test2.txt',
                'text': 'Second document.',
                'lang': 'uk',
                'metadata': {}
            }
        ]
        
        with patch('llama_index.core.node_parser.SentenceSplitter.get_nodes_from_documents') as mock_get_nodes:
            # Mock nodes for first doc
            mock_node1 = MagicMock()
            mock_node1.get_content.return_value = 'First document.'
            mock_node1.start_char_idx = 0
            mock_node1.end_char_idx = 15
            mock_node1.node_info = {}
            mock_node1.relationships = {}
            
            # Mock nodes for second doc
            mock_node2 = MagicMock()
            mock_node2.get_content.return_value = 'Second document.'
            mock_node2.start_char_idx = 0
            mock_node2.end_char_idx = 16
            mock_node2.node_info = {}
            mock_node2.relationships = {}
            
            mock_get_nodes.side_effect = [[mock_node1], [mock_node2]]
            
            result = chunker.chunk_documents(documents)
            
            assert len(result) == 2
            assert result[0]['metadata']['chunk_metadata']['lang'] == 'en'
            assert result[1]['metadata']['chunk_metadata']['lang'] == 'uk'

    def test_chunk_documents_empty_text(self) -> None:
        """Test chunk_documents with document having empty text."""
        chunker = Chunker()
        documents = [{
            'id': 'doc1',
            'source': 'test.txt',
            'text': '',
            'lang': 'en',
            'metadata': {}
        }]
        
        result = chunker.chunk_documents(documents)
        assert result == []

    def test_preprocess_text_english(self) -> None:
        """Test _preprocess_text for English."""
        chunker = Chunker()
        text = "  Hello   world!  \n\n  Test.  "
        result = chunker._preprocess_text(text, 'en')
        assert result == "Hello world! Test."

    def test_preprocess_text_ukrainian(self) -> None:
        """Test _preprocess_text for Ukrainian."""
        chunker = Chunker()
        text = "  Привіт   світ!  \n\n  Тест.  "
        result = chunker._preprocess_text(text, 'uk')
        assert result == "Привіт світ! Тест."

    def test_get_chunking_stats_empty(self) -> None:
        """Test get_chunking_stats with empty chunks."""
        chunker = Chunker()
        stats = chunker.get_chunking_stats([])
        assert stats['total_chunks'] == 0
        assert stats['avg_chunk_length'] == 0

    def test_get_chunking_stats_with_chunks(self) -> None:
        """Test get_chunking_stats with chunks."""
        chunker = Chunker()
        chunks = [
            {
                'text': 'Short text',
                'metadata': {'chunk_metadata': {'lang': 'en'}}
            },
            {
                'text': 'This is a longer text for testing purposes',
                'metadata': {'chunk_metadata': {'lang': 'uk'}}
            }
        ]
        stats = chunker.get_chunking_stats(chunks)
        assert stats['total_chunks'] == 2
        assert stats['avg_chunk_length'] == (10 + 42) / 2  # 26.0
        assert stats['languages'] == {'en': 1, 'uk': 1}
        assert 'chunk_size_distribution' in stats


class TestChunkMetadata:
    """Test cases for ChunkMetadata dataclass."""

    def test_chunk_metadata_creation(self) -> None:
        """Test ChunkMetadata creation."""
        metadata = ChunkMetadata(
            chunk_id='test_chunk_0',
            parent_id='parent_doc',
            source='test.txt',
            lang='en',
            start_offset=0,
            end_offset=100,
            chunk_index=0,
            total_chunks=5,
            additional_metadata={'test': True}
        )
        
        assert metadata.chunk_id == 'test_chunk_0'
        assert metadata.parent_id == 'parent_doc'
        assert metadata.lang == 'en'
        assert metadata.start_offset == 0
        assert metadata.end_offset == 100
        assert metadata.chunk_index == 0
        assert metadata.total_chunks == 5
        assert metadata.additional_metadata == {'test': True}
