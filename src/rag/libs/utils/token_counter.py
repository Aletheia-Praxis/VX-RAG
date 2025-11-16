"""
Token counting utilities using LlamaIndex TokenCountingHandler.

Provides a unified interface for token counting and document budgeting
using LlamaIndex native token counting infrastructure.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import tiktoken

from llama_index.core.callbacks import TokenCountingHandler

logger = logging.getLogger(__name__)


class LlamaIndexTokenCounter:
    """
    Token counter using LlamaIndex TokenCountingHandler.
    
    Provides compatibility layer between VX-RAG token budgeting logic
    and LlamaIndex native token counting infrastructure.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", verbose: bool = False):
        """
        Initialize token counter with LlamaIndex TokenCountingHandler.
        
        Args:
            model_name: Name of the model for token encoding (e.g., 'gpt-3.5-turbo', 'gpt-4')
            verbose: If True, prints token usage to console
        """
        self.model_name = model_name
        self.verbose = verbose
        
        try:
            tokenizer_fn = tiktoken.encoding_for_model(model_name).encode
        except KeyError:
            # Fallback to cl100k_base for newer models
            tokenizer_fn = tiktoken.get_encoding("cl100k_base").encode
            logger.warning(f"Unknown model {model_name}, using cl100k_base encoding")
        
        # Initialize LlamaIndex TokenCountingHandler
        self.token_counter = TokenCountingHandler(
            tokenizer=tokenizer_fn,
            verbose=verbose
        )
        
        logger.info(
            f"Initialized LlamaIndex token counter: model={model_name}, verbose={verbose}"
        )
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text using LlamaIndex tokenizer.
        
        Args:
            text: Input text to count tokens
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
        
        # Use the underlying tokenizer function
        tokens = self.token_counter.tokenizer(text)
        return len(tokens)
    
    def estimate_document_tokens(self, document: Dict[str, Any]) -> int:
        """
        Estimate tokens for a document dictionary.
        
        Args:
            document: Document dict with 'text' and optional 'metadata'
            
        Returns:
            Estimated token count including metadata overhead
        """
        text = document.get('text', '')
        
        # Count main text tokens
        text_tokens = self.count_tokens(text)
        
        # Add overhead for metadata formatting (conservative estimate)
        metadata = document.get('metadata', {})
        if metadata:
            metadata_str = str(metadata)
            metadata_tokens = len(metadata_str) // 10  # Rough estimate
        else:
            metadata_tokens = 0
        
        return text_tokens + metadata_tokens
    
    def select_documents_by_budget(
        self,
        documents: List[Dict[str, Any]],
        token_budget: int,
        max_items: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Select documents within token budget, prioritizing by relevance score.
        
        This method implements VX-RAG specific business logic:
        - Filters by minimum score threshold
        - Sorts by relevance (score descending)
        - Selects documents greedily until budget exhausted
        
        Args:
            documents: List of document dictionaries with optional 'score' field
            token_budget: Maximum token budget
            max_items: Maximum number of documents to select
            min_score: Minimum relevance score to consider
            
        Returns:
            Tuple of (selected_documents, total_tokens)
        """
        # Filter by minimum score if specified
        if min_score is not None:
            filtered_docs = [
                doc for doc in documents 
                if doc.get('score', 0.0) >= min_score
            ]
        else:
            filtered_docs = documents
        
        # Sort by score descending (highest relevance first)
        sorted_docs = sorted(
            filtered_docs,
            key=lambda x: x.get('score', 0.0),
            reverse=True
        )
        
        # Greedy selection within budget
        selected: List[Dict[str, Any]] = []
        total_tokens = 0
        
        for doc in sorted_docs:
            doc_tokens = self.estimate_document_tokens(doc)
            
            # Check if adding this document would exceed budget
            if total_tokens + doc_tokens > token_budget:
                break
            
            # Check max items limit
            if max_items and len(selected) >= max_items:
                break
            
            selected.append(doc)
            total_tokens += doc_tokens
        
        logger.info(
            f"Selected {len(selected)}/{len(documents)} documents "
            f"with {total_tokens}/{token_budget} tokens"
        )
        
        return selected, total_tokens
    
    def select_documents_by_relevance(
        self,
        documents: List[Dict[str, Any]],
        max_items: int = 10,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Select top documents by relevance score without token budgeting.
        
        Args:
            documents: List of document dictionaries
            max_items: Maximum number of documents
            min_score: Minimum score threshold
            
        Returns:
            Selected documents sorted by score
        """
        # Filter by minimum score
        if min_score is not None:
            filtered = [
                doc for doc in documents 
                if doc.get('score', 0.0) >= min_score
            ]
        else:
            filtered = documents
        
        # Sort by score descending
        sorted_docs = sorted(
            filtered,
            key=lambda x: x.get('score', 0.0),
            reverse=True
        )
        
        return sorted_docs[:max_items]
    
    def reset_counts(self) -> None:
        """Reset all accumulated token counts."""
        self.token_counter.reset_counts()
        logger.debug("Token counts reset")
    
    def get_total_embedding_tokens(self) -> int:
        """Get total embedding tokens counted."""
        return self.token_counter.total_embedding_token_count
    
    def get_total_llm_tokens(self) -> int:
        """Get total LLM tokens (prompt + completion) counted."""
        return self.token_counter.total_llm_token_count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive token usage statistics.
        
        Returns:
            Dictionary with token counts and metadata
        """
        return {
            'model_name': self.model_name,
            'total_embedding_tokens': self.token_counter.total_embedding_token_count,
            'prompt_llm_tokens': self.token_counter.prompt_llm_token_count,
            'completion_llm_tokens': self.token_counter.completion_llm_token_count,
            'total_llm_tokens': self.token_counter.total_llm_token_count,
            'llm_event_count': len(self.token_counter.llm_token_counts),
            'embedding_event_count': len(self.token_counter.embedding_token_counts)
        }
