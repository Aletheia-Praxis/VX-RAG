"""
RAG Workflow - LlamaIndex Workflow implementation for RAG query pipeline.

This module defines the workflow logic for executing RAG queries using
native LlamaIndex components (QueryEngine, ResponseSynthesizer).
"""

from typing import Any, List
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle

from src.utils.logging_config import get_logger
from src.rag.libs.schemas.mcp_schemas import MCPContextPayload, ContextItem

logger = get_logger("rag_workflow")


# Custom Events for RAG Workflow
class RetrieveEvent(Event):
    """Event triggered after retrieval step."""
    nodes: List[Any]


class RerankEvent(Event):
    """Event triggered after reranking step."""
    nodes: List[Any]


class AssembleEvent(Event):
    """Event triggered after context assembly."""
    context_payload: Any


class RAGWorkflow(Workflow):
    """
    LlamaIndex Workflow for RAG query pipeline.
    
    Implements the RAG pipeline using native LlamaIndex QueryEngine and Response Synthesizer:
    - retrieve: Get documents using QueryEngine
    - synthesize: Generate response using native Response Synthesizer
    """

    def __init__(self, query_engine: Any, response_synthesizer: Any, **kwargs):
        super().__init__(**kwargs)
        self.query_engine = query_engine
        self.response_synthesizer = response_synthesizer

    @step
    async def retrieve_and_assemble(
        self, ctx: Context, ev: StartEvent
    ) -> StopEvent:
        """Retrieve documents and synthesize response using native LlamaIndex components."""
        query = ev.get("query")
        top_k = ev.get("top_k", 5)
        search_type = ev.get("search_type", "hybrid")
        token_budget = ev.get("token_budget", 4000)

        if not query:
            raise ValueError("Query is required")

        logger.info(f"Workflow step: query='{query}', top_k={top_k}, search_type={search_type}")

        try:
            # Use QueryEngine for retrieval (includes postprocessing)
            query_bundle = QueryBundle(query_str=query)
            
            # Retrieve nodes
            response = await self.query_engine.aretrieve(query_bundle)
            nodes = response[:top_k * 4]  # Over-retrieve for better selection

            # Convert nodes to NodeWithScore objects
            nodes_with_scores = []
            for node in nodes:
                if isinstance(node, NodeWithScore):
                    nodes_with_scores.append(node)
                else:
                    # Create NodeWithScore if not already
                    score = getattr(node, 'score', 0.0)
                    node_with_score = NodeWithScore(node=node, score=score)
                    nodes_with_scores.append(node_with_score)

            # Limit to final top_k
            final_nodes = nodes_with_scores[:top_k]

            # Use Response Synthesizer to generate context
            if self.response_synthesizer:
                synthesized_response = await self.response_synthesizer.asynthesize(
                    query_str=query,
                    nodes=final_nodes
                )
                
                # Create MCP-compatible payload
                context_items = []
                for node_with_score in final_nodes:
                    item = ContextItem(
                        id=node_with_score.node.node_id or node_with_score.node.id_,
                        text=node_with_score.node.get_content(),
                        score=node_with_score.score,
                        meta=node_with_score.node.metadata
                    )
                    context_items.append(item)
                
                context_payload = MCPContextPayload(
                    schema_version="1.0",
                    context=context_items,
                    query=query,
                    token_budget=token_budget,
                    provenance={
                        'total_candidates': len(nodes),
                        'selected_count': len(final_nodes),
                        'total_tokens': len(str(synthesized_response)) // 4,  # Rough token estimate
                        'selection_method': 'response_synthesizer_async'
                    }
                )
            else:
                # Fallback
                context_items = []
                for node_with_score in final_nodes:
                    item = ContextItem(
                        id=node_with_score.node.node_id or node_with_score.node.id_,
                        text=node_with_score.node.get_content(),
                        score=node_with_score.score,
                        meta=node_with_score.node.metadata
                    )
                    context_items.append(item)
                context_payload = MCPContextPayload(
                    query=query,
                    context=context_items,
                    schema_version="1.0",
                    token_budget=token_budget
                )

            return StopEvent(result=context_payload)

        except Exception as e:
            logger.error(f"Workflow step failed: {e}")
            raise
