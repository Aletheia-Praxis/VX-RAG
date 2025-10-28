"""
Chunker Service implementation.

Provides classes and functions for text chunking and preprocessing.
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument

from ...libs.utils.text_utils import normalize_text

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    chunk_id: str
    parent_id: str
    source: str
    lang: str
    start_offset: int
    end_offset: int
    chunk_index: int
    total_chunks: int
    additional_metadata: Dict[str, Any]


class Chunker:
    """
    Text chunker with adaptive sizing and metadata preservation.
    
    Supports different chunking strategies based on language and content type.
    """
    
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        separator: str = "\n",
        use_semantic_chunking: bool = False
    ):
        """
        Initialize the chunker.
        
        Args:
            chunk_size: Default chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            separator: Separator for character-based splitting
            use_semantic_chunking: Whether to use sentence-based semantic chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.use_semantic_chunking = use_semantic_chunking
        
        # Initialize LlamaIndex parsers
        if use_semantic_chunking:
            self.parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            self.parser = SimpleNodeParser.from_defaults(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separator=separator
            )
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk a list of documents into smaller pieces.
        
        Args:
            documents: List of document dictionaries from ingest service
            
        Returns:
            List of chunk dictionaries with metadata
        """
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_single_document(doc)
            chunks.extend(doc_chunks)
        
        logger.info(f"Successfully chunked {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def _chunk_single_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single document.
        
        Args:
            document: Document dictionary from ingest
            
        Returns:
            List of chunk dictionaries
        """
        text = document.get('text', '')
        if not text:
            logger.warning(f"Document {document.get('id', 'unknown')} has no text to chunk")
            return []
        
        # Additional normalization if needed
        normalized_text = self._preprocess_text(text, document.get('lang', 'unknown'))
        
        # Create LlamaIndex document
        llama_doc = LlamaDocument(
            text=normalized_text,
            metadata=document.get('metadata', {}),
            id_=document.get('id', '')
        )
        
        # Parse into nodes (chunks)
        nodes = self.parser.get_nodes_from_documents([llama_doc])
        
        # Convert to our format
        chunks = []
        for i, node in enumerate(nodes):
            chunk_metadata = ChunkMetadata(
                chunk_id=f"{document['id']}_chunk_{i}",
                parent_id=document['id'],
                source=document['source'],
                lang=document['lang'],
                start_offset=getattr(node, 'start_char_idx', 0) or 0,
                end_offset=getattr(node, 'end_char_idx', len(node.get_content())) or len(node.get_content()),
                chunk_index=i,
                total_chunks=len(nodes),
                additional_metadata={
                    **document.get('metadata', {}),
                    'node_info': getattr(node, 'node_info', {}),
                    'relationships': getattr(node, 'relationships', {})
                }
            )
            
            chunk = {
                'id': chunk_metadata.chunk_id,
                'text': node.get_content(),
                'metadata': {
                    'chunk_metadata': chunk_metadata.__dict__,
                    'original_metadata': document.get('metadata', {})
                }
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def _preprocess_text(self, text: str, lang: str) -> str:
        """
        Additional preprocessing before chunking.
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            Preprocessed text
        """
        # Basic normalization already done in ingest
        # Add language-specific preprocessing if needed
        
        if lang in ['uk', 'ru']:
            # For Cyrillic languages, ensure proper handling of punctuation
            text = text.replace('...', '…')  # Replace multiple dots with ellipsis
        elif lang == 'en':
            # English-specific preprocessing if needed
            pass
        
        # Remove excessive whitespace that might affect chunking
        text = normalize_text(text)
        
        return text
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the chunking process.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not chunks:
            return {'total_chunks': 0, 'avg_chunk_length': 0, 'languages': {}}
        
        total_length = sum(len(chunk['text']) for chunk in chunks)
        languages = {}
        
        for chunk in chunks:
            lang = chunk['metadata']['chunk_metadata']['lang']
            languages[lang] = languages.get(lang, 0) + 1
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': total_length / len(chunks),
            'languages': languages,
            'chunk_size_distribution': self._get_length_distribution(chunks)
        }
    
    def _get_length_distribution(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get distribution of chunk lengths.
        
        Args:
            chunks: List of chunks
            
        Returns:
            Length distribution
        """
        distribution = {'short': 0, 'medium': 0, 'long': 0}
        
        for chunk in chunks:
            length = len(chunk['text'])
            if length < 500:
                distribution['short'] += 1
            elif length < 1500:
                distribution['medium'] += 1
            else:
                distribution['long'] += 1
        
        return distribution
