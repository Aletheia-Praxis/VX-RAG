"""
Token utilities for context assembly and budgeting.

Provides functions for estimating token counts and selecting documents within token limits.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import tiktoken  # For accurate token counting

logger = logging.getLogger(__name__)


class TokenBudgeter:
    """Handles token budgeting and document selection for context assembly."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        """
        Initialize token budgeter.
        
        Args:
            model_name: Name of the model for token encoding (e.g., 'gpt-3.5-turbo', 'gpt-4')
        """
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Unknown model {model_name}, using cl100k_base encoding")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def estimate_document_tokens(self, document: Dict[str, Any]) -> int:
        """Estimate tokens for a document dictionary."""
        text = document.get('text', '')
        # Add some overhead for metadata and formatting
        metadata_overhead = len(str(document.get('metadata', {}))) // 10
        return self.count_tokens(text) + metadata_overhead
    
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
        # Filter by minimum score if specified
        if min_score is not None:
            filtered_docs = [doc for doc in documents if doc.get('score', 0.0) >= min_score]
        else:
            filtered_docs = documents
        
        # Sort by score descending (highest relevance first)
        sorted_docs = sorted(
            filtered_docs,
            key=lambda x: x.get('score', 0.0),
            reverse=True
        )
        
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
        
        logger.info(f"Selected {len(selected)} documents with {total_tokens} tokens "
                   f"(budget: {token_budget})")
        
        return selected, total_tokens
    
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
        # Filter and sort
        if min_score is not None:
            filtered = [doc for doc in documents if doc.get('score', 0.0) >= min_score]
        else:
            filtered = documents
        
        sorted_docs = sorted(
            filtered,
            key=lambda x: x.get('score', 0.0),
            reverse=True
        )
        
        return sorted_docs[:max_items]


def budget_and_assemble(
    results: List[Dict[str, Any]],
    token_budget: int = 2048,
    query: str = "",
    max_items: Optional[int] = None,
    min_score: Optional[float] = None
) -> Dict[str, Any]:
    """
    Budget and assemble context from retrieval results.
    
    This is the main function for Step F according to the standard.
    
    Args:
        results: Retrieval results with documents
        token_budget: Maximum token budget
        query: Original query (for MCP payload)
        max_items: Maximum number of items
        min_score: Minimum relevance score
        
    Returns:
        MCP-compatible context payload
    """
    budgeter = TokenBudgeter()
    
    # Extract documents from results (handle different formats)
    documents = []
    for result in results:
        if isinstance(result, dict):
            # Assume result has document structure
            documents.append(result)
        else:
            # Try to extract from object attributes
            doc = {
                'id': getattr(result, 'id', str(id(result))),
                'text': getattr(result, 'text', str(result)),
                'score': getattr(result, 'score', None),
                'metadata': getattr(result, 'metadata', {})
            }
            documents.append(doc)
    
    # Select documents within budget
    selected_docs, total_tokens = budgeter.select_documents_by_budget(
        documents, token_budget, max_items, min_score
    )
    
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
