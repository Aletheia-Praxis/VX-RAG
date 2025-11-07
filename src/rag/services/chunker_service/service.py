"""
Chunker Service implementation.

Provides classes and functions for text chunking and preprocessing.
"""

from typing import List, Dict, Any, Union, Optional
import logging
import re
from dataclasses import dataclass

from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter, TokenTextSplitter
from llama_index.core.schema import Document as LlamaDocument

from ...libs.utils.text_utils import normalize_text
from src.utils.config_loader import load_chunking_config

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
        
        # Initialize parsers
        if use_hierarchical_chunking:
            self.hierarchical_chunker = MarkdownHierarchicalChunker(max_chunk_size=chunk_size)
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
        
        # Use hierarchical chunking for markdown content
        if self.use_hierarchical_chunking and is_markdown:
            return self.hierarchical_chunker._chunk_markdown_document(document)
        
        # Additional normalization if needed
        normalized_text = self._preprocess_text(text, document.get('lang', 'unknown'))
        
        # Detect content type for adaptive chunking
        content_analysis = self._analyze_content_type(normalized_text)
        content_type = content_analysis['type']
        content_subtype = content_analysis['subtype']
        
        # Adaptive chunk sizing based on detailed content analysis
        if self.adaptive_config.get('enabled', True):
            adaptive_chunk_size, adaptive_overlap = self._get_adaptive_chunk_params(
                content_type, content_subtype, content_analysis
            )
            logger.debug(
                f"Content analysis for document {document.get('id')}: "
                f"type={content_type}, subtype={content_subtype}, "
                f"chunk_size={adaptive_chunk_size}, overlap={adaptive_overlap}"
            )
        else:
            adaptive_chunk_size = self.chunk_size
            adaptive_overlap = self.chunk_overlap
            logger.debug(
                f"Adaptive chunking disabled, using default: "
                f"chunk_size={adaptive_chunk_size}, overlap={adaptive_overlap}"
            )
        
        # Create adaptive parser if needed
        adaptive_parser: Union[SimpleNodeParser, SentenceSplitter, TokenTextSplitter]
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
        # Check for code blocks (markdown or other formats)
        code_block_patterns = [
            r'```[\s\S]*?```',               # Markdown code blocks
            r'    [\s\S]*?(?=\n\S|\n\n|$)',  # Indented code blocks
            r'<code>[\s\S]*?</code>',        # HTML code tags
            r'<pre>[\s\S]*?</pre>',          # HTML pre tags
        ]
        
        # Check for tables
        table_patterns = [
            r'\|.*\|\n\|[\s\-\|:]+\|\n(?:\|.*\|\n)*',  # Markdown tables
            r'<table[\s\S]*?</table>',                 # HTML tables
        ]
        
        # Check for technical keywords that indicate code-like content
        technical_keywords = [
            'function', 'class', 'def ', 'import ', 'from ', 'return ', 
            'if ', 'for ', 'while ', 'try:', 'except:', 'with ',
            'SELECT ', 'INSERT ', 'UPDATE ', 'DELETE ', 'CREATE ', 'DROP ',
            'public static', 'private ', 'protected ', 'interface ', 'extends ',
            'function(', 'const ', 'let ', 'var ', '=>', 'async ', 'await '
        ]
        
        # Count technical elements
        technical_score = 0
        
        # Check code blocks
        for pattern in code_block_patterns:
            if re.findall(pattern, text, re.IGNORECASE | re.MULTILINE):
                technical_score += 10
        
        # Check tables
        for pattern in table_patterns:
            if re.findall(pattern, text, re.IGNORECASE | re.MULTILINE):
                technical_score += 5
        
        # Check technical keywords (but not too many to avoid false positives)
        keyword_count = sum(1 for keyword in technical_keywords 
                          if keyword.lower() in text.lower())
        if keyword_count > 2:  # Lower threshold for considering it technical
            technical_score += keyword_count // 2  # Less weight for keywords
        
        # Determine content type
        return 'technical' if technical_score >= 10 else 'general'
    
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


class MarkdownHierarchicalChunker:
    """
    Hierarchical chunker for Markdown content from LlamaParse.
    
    Creates chunks based on document structure (headings, lists, code blocks)
    while preserving semantic integrity and adding hierarchical metadata.
    """
    
    def __init__(
        self,
        max_chunk_size: Optional[int] = None,
        preserve_code_blocks: Optional[bool] = None,
        preserve_tables: Optional[bool] = None,
        config_path: Optional[str] = None
    ) -> None:
        """
        Initialize the hierarchical chunker.
        
        Args:
            max_chunk_size: Maximum size of a chunk in characters. If None, loads from config.
            preserve_code_blocks: Whether to keep code blocks intact. If None, loads from config.
            preserve_tables: Whether to keep tables intact. If None, loads from config.
            config_path: Path to settings.yaml. If None, uses default location.
        """
        # Load from config if parameters not provided
        if max_chunk_size is None or preserve_code_blocks is None or preserve_tables is None:
            from src.utils.config_loader import get_hierarchical_chunker_config
            config = get_hierarchical_chunker_config(config_path)
            if max_chunk_size is None:
                max_chunk_size = config['max_chunk_size']
            if preserve_code_blocks is None:
                preserve_code_blocks = config['preserve_code_blocks']
            if preserve_tables is None:
                preserve_tables = config['preserve_tables']
        
        if max_chunk_size is None:
            raise ValueError("max_chunk_size must be set")
        if preserve_code_blocks is None:
            raise ValueError("preserve_code_blocks must be set")
        if preserve_tables is None:
            raise ValueError("preserve_tables must be set")
        
        self.max_chunk_size = max_chunk_size
        self.preserve_code_blocks = preserve_code_blocks
        self.preserve_tables = preserve_tables
        
        # Regex patterns for markdown elements
        self.heading_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        self.table_pattern = re.compile(r'\|.*\|\n\|[\s\-\|:]+\|\n(?:\|.*\|\n)*', re.MULTILINE)
        self.list_pattern = re.compile(r'^(?:\d+\.|\-|\*)\s+.+(?:\n(?:\d+\.|\-|\*)\s+.+)*', re.MULTILINE)
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk markdown documents hierarchically.
        
        Args:
            documents: List of document dictionaries from ingest service
            
        Returns:
            List of hierarchical chunk dictionaries
        """
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_markdown_document(doc)
            chunks.extend(doc_chunks)
        
        logger.info(f"Successfully chunked {len(documents)} markdown documents into {len(chunks)} hierarchical chunks")
        return chunks
    
    def _chunk_markdown_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk a single markdown document hierarchically.
        
        Args:
            document: Document dictionary from ingest
            
        Returns:
            List of hierarchical chunk dictionaries
        """
        text = document.get('text', '')
        if not text:
            logger.warning(f"Document {document.get('id', 'unknown')} has no text to chunk")
            return []
        
        # Parse markdown structure
        sections = self._parse_markdown_structure(text)
        
        # Build hierarchy and create chunks
        chunks = []
        hierarchy_stack: List[str] = []
        chunk_index = 0
        
        for section in sections:
            # Update hierarchy
            while len(hierarchy_stack) >= section['level']:
                hierarchy_stack.pop()
            hierarchy_stack.extend([section['title']] if section['level'] > len(hierarchy_stack) else [])
            
            # Create chunks for this section
            section_chunks = self._create_chunks_for_section(
                section, hierarchy_stack.copy(), document, chunk_index
            )
            chunks.extend(section_chunks)
            chunk_index += len(section_chunks)
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk['metadata']['hierarchical_metadata'].total_chunks = len(chunks)
        
        return chunks
    
    def _parse_markdown_structure(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse markdown text into structured sections.
        
        Args:
            text: Markdown text
            
        Returns:
            List of section dictionaries
        """
        sections = []
        lines = text.split('\n')
        current_section = None
        current_content: List[str] = []
        
        for i, line in enumerate(lines):
            heading_match = self.heading_pattern.match(line)
            if heading_match:
                # Save previous section
                if current_section:
                    current_section['content'] = '\n'.join(current_content).strip()
                    sections.append(current_section)
                
                # Start new section
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                current_section = {
                    'level': level,
                    'title': title,
                    'start_line': i,
                    'content': '',
                    'type': 'heading'
                }
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Add last section
        if current_section:
            current_section['content'] = '\n'.join(current_content).strip()
            sections.append(current_section)
        
        # If no headings found, treat whole document as one section
        if not sections:
            sections = [{
                'level': 1,
                'title': 'Document',
                'start_line': 0,
                'content': text,
                'type': 'document'
            }]
        
        return sections
    
    def _create_chunks_for_section(
        self, 
        section: Dict[str, Any], 
        hierarchy_path: List[str], 
        document: Dict[str, Any],
        start_index: int
    ) -> List[Dict[str, Any]]:
        """
        Create chunks for a section, preserving semantic units.
        
        Args:
            section: Section dictionary
            hierarchy_path: Current hierarchy path
            document: Parent document
            start_index: Starting chunk index
            
        Returns:
            List of chunk dictionaries
        """
        content = section['content']
        chunks: List[Dict[str, Any]] = []
        
        # Split content into semantic units
        semantic_units = self._split_into_semantic_units(content)
        
        current_chunk = ""
        chunk_start = 0
        
        for unit in semantic_units:
            # Check if adding this unit would exceed max size
            if len(current_chunk) + len(unit) > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk_metadata = HierarchicalChunkMetadata(
                    chunk_id=f"{document['id']}_chunk_{start_index + len(chunks)}",
                    parent_id=document['id'],
                    source=document['source'],
                    lang=document['lang'],
                    start_offset=chunk_start,
                    end_offset=chunk_start + len(current_chunk),
                    chunk_index=start_index + len(chunks),
                    total_chunks=0,  # Will be updated later
                    hierarchy_path=hierarchy_path,
                    content_type='mixed',
                    additional_metadata={
                        **document.get('metadata', {}),
                        'section_title': section['title'],
                        'section_level': section['level']
                    }
                )
                
                chunks.append({
                    'id': chunk_metadata.chunk_id,
                    'text': current_chunk.strip(),
                    'metadata': {
                        'hierarchical_metadata': chunk_metadata,
                        'original_metadata': document.get('metadata', {})
                    }
                })
                
                current_chunk = unit
                chunk_start += len(current_chunk)
            else:
                current_chunk += unit
        
        # Add remaining content
        if current_chunk.strip():
            chunk_metadata = HierarchicalChunkMetadata(
                chunk_id=f"{document['id']}_chunk_{start_index + len(chunks)}",
                parent_id=document['id'],
                source=document['source'],
                lang=document['lang'],
                start_offset=chunk_start,
                end_offset=chunk_start + len(current_chunk),
                chunk_index=start_index + len(chunks),
                total_chunks=0,  # Will be updated later
                hierarchy_path=hierarchy_path,
                content_type='mixed',
                additional_metadata={
                    **document.get('metadata', {}),
                    'section_title': section['title'],
                    'section_level': section['level']
                }
            )
            
            chunks.append({
                'id': chunk_metadata.chunk_id,
                'text': current_chunk.strip(),
                'metadata': {
                    'hierarchical_metadata': chunk_metadata,
                    'original_metadata': document.get('metadata', {})
                }
            })
        
        return chunks
    
    def _split_into_semantic_units(self, text: str) -> List[str]:
        """
        Split text into semantic units (paragraphs, lists, code blocks, etc.).
        
        Args:
            text: Text to split
            
        Returns:
            List of semantic units
        """
        units = []
        
        # Find code blocks first (preserve them)
        if self.preserve_code_blocks:
            code_blocks = self.code_block_pattern.findall(text)
            text = self.code_block_pattern.sub('{{CODE_BLOCK}}', text)
            code_placeholder = '{{CODE_BLOCK}}'
        else:
            code_blocks = []
            code_placeholder = ''
        
        # Find tables
        if self.preserve_tables:
            tables = self.table_pattern.findall(text)
            text = self.table_pattern.sub('{{TABLE}}', text)
            table_placeholder = '{{TABLE}}'
        else:
            tables = []
            table_placeholder = ''
        
        # Split by paragraphs and lists
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if it's a list
            if self.list_pattern.match(para):
                units.append(para + '\n\n')
            else:
                # Split into sentences but avoid breaking within structures
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sentence in sentences:
                    if sentence.strip():
                        units.append(sentence.strip() + ' ')
        
        # Restore code blocks and tables
        final_units = []
        code_idx = 0
        table_idx = 0
        
        for unit in units:
            if code_placeholder in unit and code_idx < len(code_blocks):
                unit = unit.replace(code_placeholder, code_blocks[code_idx], 1)
                code_idx += 1
            if table_placeholder in unit and table_idx < len(tables):
                unit = unit.replace(table_placeholder, tables[table_idx], 1)
                table_idx += 1
            final_units.append(unit)
        
        return final_units
