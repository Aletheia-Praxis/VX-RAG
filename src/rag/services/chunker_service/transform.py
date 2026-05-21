"""
Adaptive chunking transformation for LlamaIndex pipeline.

This module implements a LlamaIndex NodeParser that performs adaptive chunking
based on content type (code, tables, hex dumps, etc.), preserving the logic
from the original ChunkerService.
"""

import re
from typing import List, Any, Dict, Optional, Sequence
from src.utils.logging_config import get_logger

from llama_index.core.node_parser import (
    NodeParser,
    SentenceSplitter,
    TokenTextSplitter,
    MarkdownNodeParser,
    CodeSplitter,
)
from llama_index.core.schema import BaseNode, TextNode, NodeRelationship, Document
from llama_index.core.callbacks.base import CallbackManager

logger = get_logger(__name__)

class AdaptiveChunker(NodeParser):
    """
    NodeParser that adapts chunk size based on content type.
    
    It analyzes the content of each node and selects the appropriate splitting strategy:
    - Large code blocks -> CodeSplitter (768 tokens)
    - Tables -> TokenTextSplitter (512 tokens)
    - Short snippets -> TokenTextSplitter (384 tokens)
    - Hex dumps -> TokenTextSplitter (896 tokens)
    - General text -> SentenceSplitter (1024 tokens)
    """
    
    def __init__(
        self,
        chunk_size: int = 1024,
        chunk_overlap: int = 200,
        adaptive_config: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> None:
        """
        Initialize the adaptive chunker.
        
        Args:
            chunk_size: Default chunk size for general text.
            chunk_overlap: Default overlap.
            adaptive_config: Configuration for adaptive sizing.
        """
        super().__init__(
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager or CallbackManager([]),
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Default adaptive config if not provided
        self.adaptive_config = adaptive_config or {
            'enabled': True,
            'large_code_blocks': {'chunk_size': 768, 'chunk_overlap': 150, 'min_lines': 30},
            'tables': {'chunk_size': 512, 'chunk_overlap': 100, 'preserve_integrity': True},
            'short_snippets': {'chunk_size': 384, 'chunk_overlap': 75, 'max_lines': 15},
            'hex_dumps': {'chunk_size': 896, 'chunk_overlap': 175, 'preserve_structure': True}
        }
        
        # Initialize default parsers
        self.default_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            callback_manager=callback_manager or CallbackManager([])
        )
        
        self.markdown_parser = MarkdownNodeParser(callback_manager=callback_manager or CallbackManager([]))

    def _parse_nodes(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> List[BaseNode]:
        """
        Parse nodes into smaller chunks adaptively.
        """
        all_nodes: List[BaseNode] = []
        
        for node in nodes:
            # Skip non-text nodes
            if not isinstance(node, TextNode):
                all_nodes.append(node)
                continue
                
            # Check if it's markdown (from LlamaParse or metadata)
            is_markdown = (
                node.metadata.get('content_type') == 'markdown' or 
                node.metadata.get('parsed_with') == 'llama_parse'
            )
            
            if is_markdown:
                # Use Markdown parser for markdown content
                # Wrap TextNode in Document for MarkdownNodeParser
                doc = Document(text=node.text, metadata=node.metadata)
                sub_nodes = self.markdown_parser.get_nodes_from_documents([doc])
            else:
                # Analyze content and choose parser
                sub_nodes = self._chunk_adaptively(node)
                
            all_nodes.extend(sub_nodes)
            
        return all_nodes

    def _chunk_adaptively(self, node: TextNode) -> List[BaseNode]:
        """Chunk a single node based on its content."""
        text = node.get_content()
        analysis = self._analyze_content_type(text)
        content_type = analysis['type']
        content_subtype = analysis['subtype']
        
        parser = self.default_parser
        
        if self.adaptive_config.get('enabled', True):
            if content_subtype in ['large_code_blocks', 'short_snippets']:
                # Code content
                detected_lang = self._detect_code_language(text)
                chunk_lines = 40 if content_subtype == 'large_code_blocks' else 20
                
                try:
                    parser = CodeSplitter(
                        language=detected_lang,
                        chunk_lines=chunk_lines,
                        chunk_lines_overlap=15,
                        max_chars=1500,
                        callback_manager=self.callback_manager
                    )
                except Exception:
                    # Fallback if tree-sitter not available
                    parser = self.default_parser
                    
            elif content_subtype in self.adaptive_config:
                # Other technical content (tables, hex dumps)
                config = self.adaptive_config[content_subtype]
                chunk_size = config.get('chunk_size', self.chunk_size)
                chunk_overlap = config.get('chunk_overlap', self.chunk_overlap)
                
                # Use TokenTextSplitter for structured data to respect token limits strictly
                parser = TokenTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    callback_manager=self.callback_manager
                )
        
        # Parse the node
        # We wrap the node in a list because get_nodes_from_documents expects a list
        # But wait, get_nodes_from_documents expects Documents, not Nodes.
        # However, TextNode is a subclass of BaseNode, and get_nodes_from_documents takes Sequence[Document].
        # Actually, most parsers have get_nodes_from_documents.
        # But we are inside a NodeParser, processing a Node.
        # We should use `get_nodes_from_documents` passing the node as a Document-like object
        # OR use `parser._parse_nodes` if available, but that's internal.
        # The standard way is to treat the node as a document for the sub-parser.
        
        # Create a temporary document from the node to pass to the sub-parser
        # We need to preserve the node's ID and metadata
        from llama_index.core import Document
        temp_doc = Document(
            text=text,
            metadata=node.metadata,
            id_=node.node_id,
            excluded_embed_metadata_keys=node.excluded_embed_metadata_keys,
            excluded_llm_metadata_keys=node.excluded_llm_metadata_keys,
            relationships=node.relationships
        )
        
        sub_nodes = parser.get_nodes_from_documents([temp_doc])
        
        # Update metadata with analysis info
        for sub_node in sub_nodes:
            sub_node.metadata.update({
                'content_type': content_type,
                'content_subtype': content_subtype,
                'adaptive_strategy': parser.__class__.__name__
            })
            
        return sub_nodes

    def _analyze_content_type(self, text: str) -> Dict[str, Any]:
        """
        Analyze content type for adaptive chunking.
        (Logic ported from original ChunkerService)
        """
        analysis: Dict[str, Any] = {
            'type': 'general',
            'subtype': None,
            'metadata': {}
        }
        
        # Check for hexdumps
        hex_patterns = [
            r'(?:[0-9A-Fa-f]{2}\s){8,}',
            r'0x[0-9A-Fa-f]+:\s+(?:[0-9A-Fa-f]{2}\s)+',
            r'[0-9A-Fa-f]{8,}\s+[0-9A-Fa-f]{8,}',
        ]
        hex_matches = sum(len(re.findall(p, text)) for p in hex_patterns)
        if hex_matches > 5:
            analysis['type'] = 'technical'
            analysis['subtype'] = 'hex_dumps'
            return analysis
            
        # Check for tables
        table_patterns = [
            r'\|[^\n]+\|[\r\n]+\|[\s\-\|:]+\|[\r\n]+(?:\|[^\n]+\|[\r\n]*)+',
            r'<table[\s\S]*?</table>',
        ]
        table_matches = sum(len(re.findall(p, text, re.MULTILINE | re.DOTALL)) for p in table_patterns)
        if table_matches > 0:
            analysis['type'] = 'technical'
            analysis['subtype'] = 'tables'
            return analysis
            
        # Check for code blocks
        code_block_patterns = [
            r'```[\s\S]*?```',
            r'    [\s\S]*?(?=\n\S|\n\n|$)',
            r'<code>[\s\S]*?</code>',
            r'<pre>[\s\S]*?</pre>',
        ]
        code_blocks = []
        for p in code_block_patterns:
            code_blocks.extend(re.findall(p, text, re.MULTILINE))
            
        if code_blocks:
            total_lines = sum(b.count('\n') for b in code_blocks)
            avg_lines = total_lines / len(code_blocks)
            
            config_large = self.adaptive_config.get('large_code_blocks', {})
            config_short = self.adaptive_config.get('short_snippets', {})
            
            if avg_lines >= config_large.get('min_lines', 30):
                analysis['type'] = 'technical'
                analysis['subtype'] = 'large_code_blocks'
            elif avg_lines <= config_short.get('max_lines', 15):
                analysis['type'] = 'technical'
                analysis['subtype'] = 'short_snippets'
            else:
                analysis['type'] = 'technical'
                # Medium code, use default
                
            return analysis
            
        return analysis

    def _detect_code_language(self, text: str) -> str:
        """Detect programming language."""
        indicators = {
            'python': ['def ', 'import ', 'class ', 'print(', 'self.'],
            'javascript': ['function ', 'const ', 'let ', 'var ', '=>', 'console.log'],
            'c': ['#include', 'int main', 'printf', 'malloc', 'struct '],
            'cpp': ['#include', 'std::', 'namespace ', 'template<'],
            'java': ['public class', 'private ', 'public static void main'],
            'go': ['func ', 'package ', 'import (', ':='],
        }
        
        for lang, patterns in indicators.items():
            if sum(1 for p in patterns if p in text) >= 2:
                return lang
        return 'python'
