"""
Chunker Service implementation.

Provides classes and functions for text chunking and preprocessing.
"""

from typing import List, Dict, Any, Union, Optional
from src.utils.logging_config import get_logger
import re
from dataclasses import dataclass

from llama_index.core.node_parser import (
    SimpleNodeParser,
    SentenceSplitter,
    TokenTextSplitter,
    MarkdownNodeParser,
    CodeSplitter,
)
from llama_index.core.schema import Document as LlamaDocument

from ...libs.utils.text_utils import normalize_text
from src.utils.config_loader import load_chunking_config

logger = get_logger(__name__)


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


@dataclass
class HierarchicalChunkMetadata:
    """Metadata for hierarchical chunks."""
    chunk_id: str
    parent_id: str
    source: str
    lang: str
    start_offset: int
    end_offset: int
    chunk_index: int
    total_chunks: int
    hierarchy_path: List[str]  # Path of section headers
    content_type: str  # 'heading', 'paragraph', 'list', 'code_block', 'table'
    heading_level: int = 0  # For headings, the level (1-6)
    additional_metadata: Optional[Dict[str, Any]] = None


class Chunker:
    """
    Text chunker with adaptive sizing based on content type and metadata preservation.
    
    Supports different chunking strategies based on language and content type.
    Uses 1024 tokens for general content with adaptive sizing for technical content:
    - 768 tokens for large code blocks (assembly, scripts)
    - 512 tokens for tables and structured data
    - 384 tokens for short technical snippets
    - 896 tokens for hexdumps and memory dumps
    
    Maintains consistent 19.5% overlap across all chunk sizes for cybersecurity content.
    """
    
    parser: Union[SimpleNodeParser, SentenceSplitter, TokenTextSplitter]
    
    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separator: str = "\n",
        use_semantic_chunking: bool = True,
        use_hierarchical_chunking: bool = False,
        adaptive_config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ) -> None:
        """
        Initialize the chunker.
        
        All chunking parameters are loaded from config/settings.yaml by default.
        You can override them by passing explicit values.
        
        Args:
            chunk_size: Default chunk size in tokens. If None, loads from config (default: 1024)
            chunk_overlap: Overlap between chunks in tokens. If None, loads from config (default: 200)
            separator: Separator for character-based splitting (not used for token splitting)
            use_semantic_chunking: Whether to use sentence-based semantic chunking
            use_hierarchical_chunking: Whether to use hierarchical markdown chunking
            adaptive_config: Configuration for adaptive chunking by content type. If None, loads from config
            config_path: Path to settings.yaml. If None, uses default config/settings.yaml
        """
        # Load config from settings.yaml if parameters not explicitly provided
        if chunk_size is None or chunk_overlap is None or adaptive_config is None:
            try:
                config = load_chunking_config(config_path)
                if chunk_size is None:
                    chunk_size = config['chunk_size']
                    logger.debug(f"Loaded chunk_size from config: {chunk_size}")
                if chunk_overlap is None:
                    chunk_overlap = config['chunk_overlap']
                    logger.debug(f"Loaded chunk_overlap from config: {chunk_overlap}")
                if adaptive_config is None:
                    adaptive_config = config['adaptive_chunking']
                    enabled_status = adaptive_config.get('enabled', False) if isinstance(adaptive_config, dict) else False
                    logger.debug(f"Loaded adaptive_config from config: enabled={enabled_status}")
            except (FileNotFoundError, ValueError) as e:
                logger.warning(f"Failed to load config, using hardcoded defaults: {e}")
                # Fallback to hardcoded defaults only if config loading fails
                if chunk_size is None:
                    chunk_size = 1024
                if chunk_overlap is None:
                    chunk_overlap = 200
                if adaptive_config is None:
                    adaptive_config = {
                        'enabled': True,
                        'large_code_blocks': {'chunk_size': 768, 'chunk_overlap': 150, 'min_lines': 30},
                        'tables': {'chunk_size': 512, 'chunk_overlap': 100, 'preserve_integrity': True},
                        'short_snippets': {'chunk_size': 384, 'chunk_overlap': 75, 'max_lines': 15},
                        'hex_dumps': {'chunk_size': 896, 'chunk_overlap': 175, 'preserve_structure': True}
                    }
        
        # Ensure we have valid values (for type checker)
        if chunk_size is None:
            raise ValueError("chunk_size must be set")
        if chunk_overlap is None:
            raise ValueError("chunk_overlap must be set")
        if adaptive_config is None:
            raise ValueError("adaptive_config must be set")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self.use_semantic_chunking = use_semantic_chunking
        self.use_hierarchical_chunking = use_hierarchical_chunking
        self.adaptive_config = adaptive_config
        
        # Initialize parsers (all LlamaIndex native)
        if use_hierarchical_chunking:
            # Use native LlamaIndex MarkdownNodeParser for hierarchical markdown
            self.markdown_parser = MarkdownNodeParser()
            logger.debug("Initialized MarkdownNodeParser for hierarchical chunking")
        elif use_semantic_chunking:
            self.parser = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            self.parser = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        # Initialize CodeSplitter for code block detection if available
        try:
            self.code_splitter = CodeSplitter(
                language="python",  # Will be detected dynamically
                chunk_lines=40,
                chunk_lines_overlap=15,
                max_chars=1500,
            )
            logger.debug("Initialized CodeSplitter for code block handling")
        except Exception as e:
            logger.warning(f"CodeSplitter initialization failed (tree_sitter not available?): {e}")
            self.code_splitter = None
        logger.debug("Initialized CodeSplitter for code block handling")
    
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
        
        # Check if document is markdown from LlamaParse
        is_markdown = document.get('metadata', {}).get('content_type') == 'markdown' or document.get('metadata', {}).get('parsed_with') == 'llama_parse'
        
        # Use native MarkdownNodeParser for markdown content
        if self.use_hierarchical_chunking and is_markdown:
            return self._chunk_markdown_with_llamaindex(document)
        
        # Additional normalization if needed
        normalized_text = self._preprocess_text(text, document.get('lang', 'unknown'))
        
        # Detect content type for adaptive chunking
        content_analysis = self._analyze_content_type(normalized_text)
        content_type = content_analysis['type']
        content_subtype = content_analysis['subtype']
        
        # Use CodeSplitter for code-heavy content
        if content_subtype in ['large_code_blocks', 'short_snippets'] and self.adaptive_config.get('enabled', True):
            # Detect programming language
            detected_lang = self._detect_code_language(normalized_text)
            code_splitter = CodeSplitter(
                language=detected_lang,
                chunk_lines=40 if content_subtype == 'large_code_blocks' else 20,
                chunk_lines_overlap=15,
                max_chars=1500,
            )
            adaptive_parser = code_splitter
            adaptive_chunk_size = self.adaptive_config.get(content_subtype, {}).get('chunk_size', self.chunk_size)
            adaptive_overlap = self.adaptive_config.get(content_subtype, {}).get('chunk_overlap', self.chunk_overlap)
            logger.debug(
                f"Using CodeSplitter for {content_subtype}: lang={detected_lang}, "
                f"chunk_size={adaptive_chunk_size}"
            )
        elif self.adaptive_config.get('enabled', True):
            # Standard adaptive chunking for non-code content
            adaptive_chunk_size, adaptive_overlap = self._get_adaptive_chunk_params(
                content_type, content_subtype, content_analysis
            )
            if adaptive_chunk_size != self.chunk_size or adaptive_overlap != self.chunk_overlap:
                if self.use_semantic_chunking:
                    adaptive_parser = SentenceSplitter(
                        chunk_size=adaptive_chunk_size,
                        chunk_overlap=adaptive_overlap
                    )
                else:
                    adaptive_parser = TokenTextSplitter(
                        chunk_size=adaptive_chunk_size,
                        chunk_overlap=adaptive_overlap
                    )
            else:
                adaptive_parser = self.parser
            logger.debug(
                f"Adaptive chunking: type={content_type}, subtype={content_subtype}, "
                f"chunk_size={adaptive_chunk_size}, overlap={adaptive_overlap}"
            )
        else:
            adaptive_chunk_size = self.chunk_size
            adaptive_overlap = self.chunk_overlap
            adaptive_parser = self.parser
            logger.debug(f"Adaptive chunking disabled, using default: chunk_size={adaptive_chunk_size}")
        
        # Create LlamaIndex document
        llama_doc = LlamaDocument(
            text=normalized_text,
            metadata=document.get('metadata', {}),
            id_=document.get('id', '')
        )
        
        # Parse into nodes (chunks)
        nodes = adaptive_parser.get_nodes_from_documents([llama_doc])
        
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
                    'relationships': getattr(node, 'relationships', {}),
                    'content_type': content_type,
                    'content_subtype': content_subtype,
                    'adaptive_chunk_size': adaptive_chunk_size,
                    'adaptive_overlap': adaptive_overlap
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
    
    def _chunk_markdown_with_llamaindex(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk markdown document using native LlamaIndex MarkdownNodeParser.
        
        Args:
            document: Document dictionary from ingest
            
        Returns:
            List of chunk dictionaries
        """
        text = document.get('text', '')
        if not text:
            logger.warning(f"Document {document.get('id', 'unknown')} has no text to chunk")
            return []
        
        # Create LlamaIndex document
        llama_doc = LlamaDocument(
            text=text,
            metadata=document.get('metadata', {}),
            id_=document.get('id', '')
        )
        
        # Parse with MarkdownNodeParser (preserves markdown structure)
        nodes = self.markdown_parser.get_nodes_from_documents([llama_doc])
        
        # Convert to our format with metadata
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
                    'relationships': getattr(node, 'relationships', {}),
                    'content_type': 'markdown',
                    'parser': 'MarkdownNodeParser'
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
        
        logger.debug(f"MarkdownNodeParser created {len(chunks)} chunks from document {document.get('id')}")
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
    
    def _analyze_content_type(self, text: str) -> Dict[str, Any]:
        """
        Analyze content type in detail for adaptive chunking.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with content analysis:
            - type: 'general' or 'technical'
            - subtype: 'large_code_blocks', 'tables', 'short_snippets', 'hex_dumps', or None
            - metadata: Additional analysis metadata
        """
        analysis: Dict[str, Any] = {
            'type': 'general',
            'subtype': None,
            'metadata': {}
        }
        
        # Check for hexdumps and memory dumps (high priority)
        hex_patterns = [
            r'(?:[0-9A-Fa-f]{2}\s){8,}',  # Hex bytes sequence
            r'0x[0-9A-Fa-f]+:\s+(?:[0-9A-Fa-f]{2}\s)+',  # Memory address + hex
            r'[0-9A-Fa-f]{8,}\s+[0-9A-Fa-f]{8,}',  # Hex dump format
        ]
        
        hex_matches = sum(len(re.findall(p, text)) for p in hex_patterns)
        if hex_matches > 5:  # Significant hex content
            analysis['type'] = 'technical'
            analysis['subtype'] = 'hex_dumps'
            analysis['metadata']['hex_matches'] = hex_matches
            return analysis
        
        # Check for tables (structured data)
        table_patterns = [
            r'\|[^\n]+\|[\r\n]+\|[\s\-\|:]+\|[\r\n]+(?:\|[^\n]+\|[\r\n]*)+',  # Markdown tables (multiline)
            r'<table[\s\S]*?</table>',  # HTML tables
        ]
        
        table_matches = 0
        for pattern in table_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            table_matches += len(matches)
        
        if table_matches > 0:
            analysis['type'] = 'technical'
            analysis['subtype'] = 'tables'
            analysis['metadata']['table_matches'] = table_matches
            return analysis
        
        # Check for code blocks
        code_block_patterns = [
            r'```[\s\S]*?```',  # Markdown code blocks
            r'    [\s\S]*?(?=\n\S|\n\n|$)',  # Indented code blocks
            r'<code>[\s\S]*?</code>',  # HTML code tags
            r'<pre>[\s\S]*?</pre>',  # HTML pre tags
        ]
        
        code_blocks = []
        for pattern in code_block_patterns:
            code_blocks.extend(re.findall(pattern, text, re.MULTILINE))
        
        if code_blocks:
            # Analyze code block sizes
            total_code_lines = sum(block.count('\n') for block in code_blocks)
            avg_code_lines = total_code_lines / len(code_blocks) if code_blocks else 0
            
            config_large = self.adaptive_config.get('large_code_blocks', {})
            config_short = self.adaptive_config.get('short_snippets', {})
            
            min_lines_large = config_large.get('min_lines', 30)
            max_lines_short = config_short.get('max_lines', 15)
            
            if avg_code_lines >= min_lines_large:
                analysis['type'] = 'technical'
                analysis['subtype'] = 'large_code_blocks'
                analysis['metadata']['code_blocks'] = len(code_blocks)
                analysis['metadata']['avg_lines'] = avg_code_lines
                return analysis
            elif avg_code_lines > 0 and avg_code_lines <= max_lines_short:
                analysis['type'] = 'technical'
                analysis['subtype'] = 'short_snippets'
                analysis['metadata']['code_blocks'] = len(code_blocks)
                analysis['metadata']['avg_lines'] = avg_code_lines
                return analysis
            else:
                # Medium-sized code - treat as general with note
                analysis['type'] = 'technical'
                analysis['subtype'] = None  # Will use default chunk size
                analysis['metadata']['code_blocks'] = len(code_blocks)
                analysis['metadata']['avg_lines'] = avg_code_lines
                return analysis
        
        # Check for technical keywords (lower priority, more selective)
        # Only trigger on Win32 API and assembly - not common programming keywords
        high_specificity_keywords = [
            'VirtualAlloc', 'WriteProcessMemory', 'CreateRemoteThread',
            'LoadLibrary', 'GetProcAddress', 'RegSetValue', 'RegOpenKey',
            'mov ', 'push ', 'pop ', 'call ', 'jmp ', 'ret ', 'lea ',
            'HKEY_LOCAL_MACHINE', 'HKEY_CURRENT_USER',
        ]
        
        keyword_count = sum(1 for keyword in high_specificity_keywords 
                          if keyword in text)  # Case-sensitive for API names
        
        if keyword_count >= 3:  # Require multiple matches
            analysis['type'] = 'technical'
            analysis['metadata']['keyword_count'] = keyword_count
        
        return analysis
    
    def _get_adaptive_chunk_params(
        self, 
        content_type: str, 
        content_subtype: Optional[str],
        content_analysis: Dict[str, Any]
    ) -> tuple[int, int]:
        """
        Get adaptive chunk size and overlap based on content analysis.
        
        Args:
            content_type: 'general' or 'technical'
            content_subtype: Specific subtype of technical content
            content_analysis: Full content analysis from _analyze_content_type
            
        Returns:
            Tuple of (chunk_size, chunk_overlap)
        """
        # General content uses defaults
        if content_type == 'general':
            return self.chunk_size, self.chunk_overlap
        
        # Technical content with specific subtype
        if content_subtype and content_subtype in self.adaptive_config:
            config = self.adaptive_config[content_subtype]
            chunk_size = config.get('chunk_size', self.chunk_size)
            chunk_overlap = config.get('chunk_overlap', self.chunk_overlap)
            return chunk_size, chunk_overlap
        
        # Technical content without specific subtype - use defaults
        return self.chunk_size, self.chunk_overlap
    
    def _detect_code_language(self, text: str) -> str:
        """
        Detect programming language in code blocks.
        
        Args:
            text: Text containing code
            
        Returns:
            Detected language ('python', 'javascript', 'c', etc.)
        """
        # Simple heuristic-based detection
        language_indicators = {
            'python': ['def ', 'import ', 'class ', 'print(', 'self.'],
            'javascript': ['function ', 'const ', 'let ', 'var ', '=>', 'console.log'],
            'c': ['#include', 'int main', 'printf', 'malloc', 'struct '],
            'cpp': ['#include', 'std::', 'namespace ', 'template<'],
            'java': ['public class', 'private ', 'public static void main'],
            'go': ['func ', 'package ', 'import (', ':='],
        }
        
        for lang, indicators in language_indicators.items():
            if sum(1 for indicator in indicators if indicator in text) >= 2:
                return lang
        
        # Default to python for cybersecurity content
        return 'python'
    
    def _detect_content_type(self, text: str) -> str:
        """
        Legacy method for backward compatibility.
        Detect the content type of the text.
        
        Args:
            text: Input text
            
        Returns:
            'technical' if contains code blocks or tables, 'general' otherwise
        """
        analysis = self._analyze_content_type(text)
        return str(analysis['type'])
    
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
        languages: Dict[str, int] = {}
        
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
