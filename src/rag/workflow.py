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
)
from llama_index.core.schema import NodeWithScore
from llama_index.core import QueryBundle

from src.utils.logging_config import get_logger
from src.rag.libs.schemas.mcp_schemas import MCPContextPayload, ContextItem

logger = get_logger("rag_workflow")

_SCORE_MIN: float = 0.0
_SCORE_MAX: float = 1.0
# Multiplier applied to top_k to over-retrieve candidates before final trimming.
_OVER_RETRIEVE_FACTOR: int = 4


def _clip_score(raw_score: float | None) -> float:
    """
    Clip a retrieval score to the valid [0.0, 1.0] range.

    FAISS inner-product scores are not bounded to [0, 1] by default, so raw
    values must be clipped before being stored in ``ContextItem`` which
    enforces that invariant via a Pydantic validator.

    Args:
        raw_score: Raw retrieval score from FAISS / BM25, or ``None``.

    Returns:
        Score clamped to ``[0.0, 1.0]``.
    """
    return max(_SCORE_MIN, min(_SCORE_MAX, raw_score or _SCORE_MIN))


def _build_context_items(nodes_with_scores: List[NodeWithScore]) -> List[ContextItem]:
    """
    Convert a list of ``NodeWithScore`` objects into ``ContextItem`` instances.

    Scores are clipped to ``[0.0, 1.0]`` so that ``ContextItem``'s Pydantic
    validator never raises for out-of-range FAISS inner-product scores.

    Args:
        nodes_with_scores: Retrieved nodes with raw similarity scores.

    Returns:
        List of ``ContextItem`` objects ready for ``MCPContextPayload``.
    """
    return [
        ContextItem(
            id=node_with_score.node.node_id or node_with_score.node.id_,
            text=node_with_score.node.get_content(),
            score=_clip_score(node_with_score.score),
            meta=node_with_score.node.metadata,
        )
        for node_with_score in nodes_with_scores
    ]


class RAGWorkflow(Workflow):
    """
    LlamaIndex Workflow for the RAG query pipeline.

    Implements the RAG pipeline using native LlamaIndex QueryEngine and
    ResponseSynthesizer in a single ``retrieve_and_assemble`` step:

    1. Retrieve nodes via ``query_engine.aretrieve`` (or ``retrieve`` fallback).
    2. Normalise raw scores to ``[0.0, 1.0]``.
    3. Synthesise a response via ``response_synthesizer.asynthesize``.
    4. Return an ``MCPContextPayload`` wrapped in ``StopEvent``.
    """

    def __init__(self, query_engine: Any, response_synthesizer: Any, **kwargs: Any) -> None:
        """
        Initialise the workflow.

        Args:
            query_engine: A LlamaIndex ``QueryEngine`` (must support
                ``aretrieve`` or ``retrieve``).
            response_synthesizer: A LlamaIndex ``ResponseSynthesizer``.
            **kwargs: Forwarded to the parent ``Workflow.__init__``.
        """
        super().__init__(**kwargs)
        self.query_engine = query_engine
        self.response_synthesizer = response_synthesizer

    @step
    async def retrieve_and_assemble(
        self, ctx: Context, ev: StartEvent
    ) -> StopEvent:
        """
        Retrieve documents and synthesise the MCP context payload.

        Args:
            ctx: LlamaIndex workflow context (unused but required by ``@step``).
            ev: Start event carrying ``query``, ``top_k``, ``search_type``,
                and ``token_budget`` keys.

        Returns:
            ``StopEvent`` whose ``result`` is an ``MCPContextPayload``.

        Raises:
            ValueError: If ``query`` is missing from the event.
            AttributeError: If ``query_engine`` supports neither ``aretrieve``
                nor ``retrieve``.
        """
        query: str = ev.get("query")
        top_k: int = ev.get("top_k", 5)
        search_type: str = ev.get("search_type", "hybrid")
        token_budget: int = ev.get("token_budget", 4000)

        if not query:
            raise ValueError("Query is required")

        logger.info(
            f"Workflow step: query='{query}', top_k={top_k}, search_type={search_type}"
        )

        try:
            query_bundle = QueryBundle(query_str=query)

            # Guard: prefer async retrieval, fall back to sync if unavailable.
            if hasattr(self.query_engine, "aretrieve"):
                raw_nodes = await self.query_engine.aretrieve(query_bundle)
            elif hasattr(self.query_engine, "retrieve"):
                raw_nodes = self.query_engine.retrieve(query_bundle)
            else:
                raise AttributeError(
                    f"{type(self.query_engine).__name__} supports neither "
                    "'aretrieve' nor 'retrieve'"
                )

            # Over-retrieve for better selection before trimming to top_k
            candidate_nodes: List[Any] = raw_nodes[: top_k * _OVER_RETRIEVE_FACTOR]

            # Normalise to NodeWithScore
            nodes_with_scores: List[NodeWithScore] = []
            for node in candidate_nodes:
                if isinstance(node, NodeWithScore):
                    nodes_with_scores.append(node)
                else:
                    nodes_with_scores.append(
                        NodeWithScore(node=node, score=getattr(node, "score", 0.0))
                    )

            final_nodes = nodes_with_scores[:top_k]

            if self.response_synthesizer:
                synthesized_response = await self.response_synthesizer.asynthesize(
                    query_str=query,
                    nodes=final_nodes,
                )

                context_items = _build_context_items(final_nodes)
                # Rough token estimate: 4 chars ≈ 1 token
                estimated_tokens = len(str(synthesized_response)) // 4

                context_payload = MCPContextPayload(
                    schema_version="1.0",
                    context=context_items,
                    query=query,
                    token_budget=token_budget,
                    provenance={
                        "total_candidates": len(candidate_nodes),
                        "selected_count": len(final_nodes),
                        "total_tokens": estimated_tokens,
                        "selection_method": "response_synthesizer_async",
                    },
                )
            else:
                # Fallback: build context directly from retrieved nodes
                context_items = _build_context_items(final_nodes)
                context_payload = MCPContextPayload(
                    query=query,
                    context=context_items,
                    schema_version="1.0",
                    token_budget=token_budget,
                )

            return StopEvent(result=context_payload)

        except Exception as e:
            logger.error(f"Workflow step failed: {e}")
            raise
