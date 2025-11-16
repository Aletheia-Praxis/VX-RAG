"""
Token utilities for context assembly and budgeting.

Provides functions for estimating token counts and selecting documents within token limits.
Uses LlamaIndex TokenCountingHandler for native integration.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional

from .token_counter import LlamaIndexTokenCounter

logger = logging.getLogger(__name__)


class TokenBudgeter:
    """
    Handles token budgeting and document selection for context assembly.
    
    This class now uses LlamaIndex TokenCountingHandler internally
    for better integration with the LlamaIndex ecosystem.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", verbose: bool = False):
        """
        Initialize token budgeter with LlamaIndex integration.
        
        Args:
            model_name: Name of the model for token encoding (e.g., 'gpt-3.5-turbo', 'gpt-4')
            verbose: If True, prints token usage to console
        """
        self.model_name = model_name
        self._counter = LlamaIndexTokenCounter(model_name=model_name, verbose=verbose)
        logger.info(f"Initialized TokenBudgeter with LlamaIndex integration: model={model_name}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using LlamaIndex tokenizer."""
        return self._counter.count_tokens(text)
    
    def estimate_document_tokens(self, document: Dict[str, Any]) -> int:
        """Estimate tokens for a document dictionary."""
        return self._counter.estimate_document_tokens(document)
    
    def select_documents_by_budget(
        self,
        documents: List[Dict[str, Any]],
        token_budget: int,
        max_items: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Select documents within token budget, prioritizing by relevance score.
        
        Args:
            documents: List of document dictionaries with optional 'score' field
            token_budget: Maximum token budget
            max_items: Maximum number of documents to select
            min_score: Minimum relevance score to consider
            
        Returns:
            Tuple of (selected_documents, total_tokens)
        """
        return self._counter.select_documents_by_budget(
            documents, token_budget, max_items, min_score
        )
    
    def select_documents_by_relevance(
        self,
        documents: List[Dict[str, Any]],
        max_items: int = 10,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Select top documents by relevance score.
        
        Args:
            documents: List of document dictionaries
            max_items: Maximum number of documents
            min_score: Minimum score threshold
            
        Returns:
            Selected documents sorted by score
        """
        return self._counter.select_documents_by_relevance(documents, max_items, min_score)
    
    def reset_counts(self) -> None:
        """Reset all accumulated token counts."""
        self._counter.reset_counts()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive token usage statistics from LlamaIndex.
        
        Returns:
            Dictionary with embedding, LLM, and total token counts
        """
        return self._counter.get_stats()


def budget_and_assemble(
    results: List[Dict[str, Any]],
    token_budget: int = 2048,
    query: str = "",
    max_items: Optional[int] = None,
    min_score: Optional[float] = None,
    model_name: str = "gpt-3.5-turbo"
) -> Dict[str, Any]:
    """
    Budget and assemble context from retrieval results using LlamaIndex token counting.
    
    This is the main function for Step F according to the standard.
    Now uses LlamaIndex TokenCountingHandler for accurate token tracking.
    
    Args:
        results: Retrieval results with documents
        token_budget: Maximum token budget
        query: Original query (for MCP payload)
        max_items: Maximum number of items
        min_score: Minimum relevance score
        model_name: Model name for token encoding (default: gpt-3.5-turbo)
        
    Returns:
        MCP-compatible context payload
    """
    budgeter = TokenBudgeter(model_name=model_name)
    
    # Extract documents from results (handle different formats)
    documents = []
    for result in results:
        documents.append(result)
    
    # Select documents within budget using LlamaIndex token counting
    selected_docs, total_tokens = budgeter.select_documents_by_budget(
        documents, token_budget, max_items, min_score
    )
    
    # Normalize scores to 0.0-1.0 range (CrossEncoder can return values outside this range)
    if selected_docs:
        scores = [doc.get('score', 0.0) for doc in selected_docs if doc.get('score') is not None]
        if scores:
            max_score = max(scores)
            min_score_val = min(scores)
            score_range = max_score - min_score_val if max_score > min_score_val else 1.0
            
            for doc in selected_docs:
                if 'score' in doc and doc['score'] is not None:
                    # Normalize to 0.0-1.0 (min-max scaling)
                    doc['score'] = (doc['score'] - min_score_val) / score_range
    
    # Convert to MCP ContextItems
    from ..schemas.mcp_schemas import ContextItem, MCPContextPayload
    
    context_items = []
    for doc in selected_docs:
        item = ContextItem(
            id=doc.get('id', ''),
            text=doc.get('text', ''),
            score=doc.get('score'),
            meta=doc.get('metadata', {})
        )
        context_items.append(item)
    
    # Create MCP payload
    payload = MCPContextPayload(
        schema_version="1.0",
        context=context_items,
        query=query,
        token_budget=token_budget,
        provenance={
            'total_candidates': len(documents),
            'selected_count': len(selected_docs),
            'total_tokens': total_tokens,
            'selection_method': 'token_budget_relevance'
        }
    )
    
    logger.info(f"Assembled context with {len(context_items)} items, "
               f"{total_tokens} tokens (budget: {token_budget})")
    
    return payload.dict()
