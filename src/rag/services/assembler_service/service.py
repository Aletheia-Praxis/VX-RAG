"""
Assembler Service implementation.

Provides classes for context assembly and MCP payload creation.
Integrates with LlamaIndex Node objects and token counting infrastructure.
"""

from typing import List, Dict, Any, Optional, Union
import logging

from llama_index.core.schema import NodeWithScore, BaseNode

from ...libs.schemas.mcp_schemas import (
    MCPContextPayload, ContextAssemblyRequest, ContextAssemblyResponse, ContextItem
)
from ...libs.utils.token_utils import TokenBudgeter, budget_and_assemble
from src.utils.config_loader import get_context_assembler_config

logger = logging.getLogger(__name__)


class ContextAssembler:
    """
    Service for assembling MCP-compatible context for LLM queries.
    
    Uses LlamaIndex TokenCountingHandler for accurate token tracking
    and budgeting across the RAG pipeline.
    """
    
    def __init__(
        self, 
        token_budget: Optional[int] = None,
        model_name: Optional[str] = None,
        config_path: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize context assembler with LlamaIndex integration.
        
        Args:
            token_budget: Default token budget for context. If None, loads from config.
            model_name: Model name for token counting. If None, loads from config.
            config_path: Path to settings.yaml. If None, uses default location.
            verbose: If True, prints token usage to console
        """
        # Load config if parameters not provided
        if token_budget is None or model_name is None:
            config = get_context_assembler_config(config_path)
            token_budget = token_budget or config['token_budget']
            model_name = model_name or config['model_name']
        
        # Ensure values are set
        if token_budget is None:
            raise ValueError("token_budget must be set")
        if model_name is None:
            raise ValueError("model_name must be set")
        
        self.default_token_budget = token_budget
        self.model_name = model_name
        self.token_budgeter = TokenBudgeter(model_name=model_name, verbose=verbose)
        
        logger.info(
            f"Initialized ContextAssembler with LlamaIndex: "
            f"token_budget={token_budget}, model={model_name}, verbose={verbose}"
        )
    
    def assemble_context(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        token_budget: Optional[int] = None,
        max_items: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> MCPContextPayload:
        """
        Assemble MCP-compatible context payload from documents using LlamaIndex token counting.
        
        Args:
            query: Original user query
            documents: List of document dictionaries with metadata
            token_budget: Maximum token budget (overrides default)
            max_items: Maximum number of context items
            min_score: Minimum relevance score
            
        Returns:
            MCPContextPayload with selected context
        """
        budget = token_budget or self.default_token_budget
        
        # Use LlamaIndex-integrated budget_and_assemble function
        payload_dict = budget_and_assemble(
            results=documents,
            token_budget=budget,
            query=query,
            max_items=max_items,
            min_score=min_score,
            model_name=self.model_name
        )
        
        # Convert back to Pydantic model for validation
        payload = MCPContextPayload(**payload_dict)
        
        logger.info(f"Assembled context: {len(payload.context)} items, "
                   f"~{payload.total_tokens_estimate()} tokens")
        
        return payload
    
    def assemble_from_request(self, request: ContextAssemblyRequest) -> ContextAssemblyResponse:
        """
        Assemble context from a structured request.
        
        Args:
            request: ContextAssemblyRequest with query and documents
            
        Returns:
            ContextAssemblyResponse with payload and metadata
        """
        payload = self.assemble_context(
            query=request.query,
            documents=request.documents,
            token_budget=request.token_budget,
            max_items=request.max_items,
            min_score=request.min_score
        )
        
        response = ContextAssemblyResponse(
            payload=payload,
            selected_count=len(payload.context),
            total_tokens=payload.total_tokens_estimate(),
            within_budget=payload.is_within_budget()
        )
        
        return response
    
    def select_top_documents(
        self,
        documents: List[Dict[str, Any]],
        max_tokens: int,
        max_items: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Select documents within token and item limits.
        
        Args:
            documents: List of document dictionaries
            max_tokens: Maximum token budget
            max_items: Maximum number of items
            min_score: Minimum relevance score
            
        Returns:
            Selected documents
        """
        selected, _ = self.token_budgeter.select_documents_by_budget(
            documents, max_tokens, max_items, min_score
        )
        return selected
    
    def validate_payload(self, payload: MCPContextPayload) -> bool:
        """
        Validate MCP payload structure and constraints.
        
        Args:
            payload: Payload to validate
            
        Returns:
            True if valid
        """
        try:
            # Pydantic validation happens automatically
            if not payload.is_within_budget():
                logger.warning(f"Payload exceeds token budget: "
                             f"{payload.total_tokens_estimate()} > {payload.token_budget}")
                return False
            
            if not payload.context:
                logger.warning("Payload has no context items")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Payload validation failed: {e}")
            return False
    
    def get_assembly_stats(self, payload: MCPContextPayload) -> Dict[str, Any]:
        """
        Get statistics about the assembled context including LlamaIndex token counts.
        
        Args:
            payload: Assembled payload
            
        Returns:
            Statistics dictionary with LlamaIndex metrics
        """
        scores = [item.score for item in payload.context if item.score is not None]
        
        # Get LlamaIndex token statistics
        llamaindex_stats = self.token_budgeter.get_stats()
        
        return {
            'total_items': len(payload.context),
            'total_tokens_estimate': payload.total_tokens_estimate(),
            'token_budget': payload.token_budget,
            'within_budget': payload.is_within_budget(),
            'avg_score': sum(scores) / len(scores) if scores else None,
            'min_score': min(scores) if scores else None,
            'max_score': max(scores) if scores else None,
            'provenance': payload.provenance,
            'llamaindex_stats': llamaindex_stats
        }
    
    def reset_token_counts(self) -> None:
        """Reset accumulated token counts in LlamaIndex counter."""
        self.token_budgeter.reset_counts()
        logger.debug("Reset token counts")
    
    def assemble_from_nodes(
        self,
        query: str,
        nodes: List[Union[NodeWithScore, BaseNode]],
        token_budget: Optional[int] = None,
        max_items: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> MCPContextPayload:
        """
        Assemble MCP context payload directly from LlamaIndex Node objects.
        
        This method provides seamless integration with LlamaIndex retrieval results,
        converting NodeWithScore objects to VX-RAG MCP format.
        
        Args:
            query: Original user query
            nodes: List of LlamaIndex Node or NodeWithScore objects
            token_budget: Maximum token budget (overrides default)
            max_items: Maximum number of context items
            min_score: Minimum relevance score
            
        Returns:
            MCPContextPayload with selected context
        """
        budget = token_budget or self.default_token_budget
        
        # Convert LlamaIndex nodes to VX-RAG document format
        documents = []
        for node in nodes:
            # Handle both NodeWithScore and BaseNode
            if isinstance(node, NodeWithScore):
                node_obj = node.node
                score = node.score
            else:
                node_obj = node
                score = None
            
            # Extract metadata
            metadata = node_obj.metadata if hasattr(node_obj, 'metadata') else {}
            
            doc = {
                'id': node_obj.node_id if hasattr(node_obj, 'node_id') else node_obj.id_,
                'text': node_obj.get_content(),
                'score': score,
                'metadata': metadata
            }
            documents.append(doc)
        
        # Use standard assembly with LlamaIndex token counting
        payload_dict = budget_and_assemble(
            results=documents,
            token_budget=budget,
            query=query,
            max_items=max_items,
            min_score=min_score,
            model_name=self.model_name
        )
        
        payload = MCPContextPayload(**payload_dict)
        
        logger.info(
            f"Assembled context from {len(nodes)} LlamaIndex nodes: "
            f"{len(payload.context)} items selected, "
            f"~{payload.total_tokens_estimate()} tokens"
        )
        
        return payload
    
    def nodes_to_context_items(
        self,
        nodes: List[Union[NodeWithScore, BaseNode]]
    ) -> List[ContextItem]:
        """
        Convert LlamaIndex Node objects to MCP ContextItem objects.
        
        Args:
            nodes: List of LlamaIndex Node or NodeWithScore objects
            
        Returns:
            List of ContextItem objects
        """
        context_items = []
        
        for node in nodes:
            # Handle both NodeWithScore and BaseNode
            if isinstance(node, NodeWithScore):
                node_obj = node.node
                score = node.score
            else:
                node_obj = node
                score = None
            
            # Extract metadata
            metadata = node_obj.metadata if hasattr(node_obj, 'metadata') else {}
            
            item = ContextItem(
                id=node_obj.node_id if hasattr(node_obj, 'node_id') else node_obj.id_,
                text=node_obj.get_content(),
                score=score,
                meta=metadata
            )
            context_items.append(item)
        
        return context_items
