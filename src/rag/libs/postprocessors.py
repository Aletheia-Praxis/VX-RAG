"""
Custom LlamaIndex postprocessors for VX-RAG.

This module provides custom postprocessor implementations that extend
LlamaIndex's standard postprocessor functionality with VX-RAG specific logic.
"""

from __future__ import annotations

import math
from typing import Any

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import PrivateAttr

from src.rag.metadata import STRICT_METADATA_KEYS
from src.utils.config_loader import get_reranker_config
from src.utils.logging_config import get_logger

logger = get_logger("postprocessors")

_INFERENCE_FALLBACK_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)

__all__ = ["BGECrossEncoderReranker", "MetadataBoostPostprocessor"]


class MetadataBoostPostprocessor(BaseNodePostprocessor):  # type: ignore[misc]
    """
    Postprocessor that boosts node scores based on metadata completeness.

    Prioritizes nodes with populated strict metadata fields
    (file_name, file_type, creation_date, ingestion_date, file_hash) by applying
    a multiplicative boost to their retrieval scores. Legacy metadata fields
    (source, lang, topic, author, year) are no longer queried.

    Args:
        boost_factor: Multiplicative factor per populated priority field (default: 0.1)
        priority_fields: List of metadata fields to check (default: strict schema fields)
        config_path: Path to config file for loading boost_factor

    Example:
        >>> postprocessor = MetadataBoostPostprocessor(boost_factor=0.15)
        >>> boosted_nodes = postprocessor.postprocess_nodes(nodes, query_bundle)
    """

    # Private attributes avoid Pydantic field restrictions on BaseNodePostprocessor
    # while allowing super().__init__() to be called normally (B-13).
    _boost_factor: float = PrivateAttr(default=0.1)
    _priority_fields: list[str] = PrivateAttr(default_factory=list)

    def __init__(
        self,
        boost_factor: float | None = None,
        priority_fields: list[str] | None = None,
        config_path: str | None = None,
    ) -> None:
        """
        Initialize the metadata boost postprocessor.

        Args:
            boost_factor: Multiplicative boost per populated metadata field.
                Loaded from config if not provided.
            priority_fields: Metadata keys to inspect for completeness.
                Defaults to strict 5-field schema. If legacy fields are passed,
                they are filtered against the strict schema.
            config_path: Path to settings.yaml; uses default if None.
        """
        # Call Pydantic's __init__ first so the model is fully initialized
        super().__init__()

        if boost_factor is None:
            config = get_reranker_config(config_path)
            boost_factor = float(config.get("metadata_boost", 0.1))

        self._boost_factor = boost_factor
        if priority_fields is None:
            self._priority_fields = list(STRICT_METADATA_KEYS)
        else:
            # Filter against strict schema; drop legacy keys gracefully
            valid_keys = [k for k in priority_fields if k in STRICT_METADATA_KEYS]
            self._priority_fields = valid_keys if valid_keys else list(STRICT_METADATA_KEYS)

        logger.info(
            f"Initialized MetadataBoostPostprocessor: boost_factor={self._boost_factor}, "
            f"priority_fields={self._priority_fields}"
        )

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """
        Apply metadata completeness boost to node scores.

        Args:
            nodes: Retrieved nodes with scores from retriever.
            query_bundle: Optional query bundle.

        Returns:
            List of nodes with boosted scores, sorted descending by score.
        """
        if not nodes:
            return []

        try:
            for node_with_score in nodes:
                node = node_with_score.node
                metadata = getattr(node, "metadata", {}) or {}

                # Count populated fields that match strict schema
                populated_count = sum(
                    1
                    for key in self._priority_fields
                    if key in metadata and metadata[key] is not None and str(metadata[key]).strip()
                )

                # Multiplicative boost: score * (1 + populated_count * boost_factor)
                multiplier = 1.0 + (populated_count * self._boost_factor)
                original_score = node_with_score.score if node_with_score.score is not None else 1.0
                node_with_score.score = original_score * multiplier

            # Sort descending by score
            nodes.sort(key=lambda n: n.score if n.score is not None else 0.0, reverse=True)
            return nodes

        except (AttributeError, TypeError, KeyError, ValueError) as e:
            logger.warning(
                f"Failed to apply metadata boost: {e}, returning original nodes"
            )
            return nodes


class BGECrossEncoderReranker(BaseNodePostprocessor):  # type: ignore[misc]
    """
    Cross-Encoder Reranker using BAAI/bge-reranker-base.

    Computes relevance scores between query and candidate nodes using a CrossEncoder,
    normalizes scores into [0.0, 1.0] using sigmoid: 1.0 / (1.0 + math.exp(-raw_score)),
    reorders nodes descending by score, and truncates the candidate list to top_n.
    Preserves node text content, node_id, and metadata intact.
    Gracefully falls back to returning the original candidates if an inference error occurs.
    """

    model_name: str = "BAAI/bge-reranker-base"
    top_n: int = 5
    device: str = "cpu"
    _model: Any = PrivateAttr(default=None)

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        top_n: int = 5,
        device: str = "cpu",
        model: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the BGE cross-encoder reranker.

        Args:
            model_name: Name or HuggingFace ID of the cross-encoder model.
            top_n: Maximum number of top candidates to return.
            device: Compute device ('cpu' or 'cuda').
            model: Optional pre-loaded model instance (for dependency injection or testing).
            **kwargs: Additional keyword arguments passed to BaseNodePostprocessor.
        """
        super().__init__()
        self.model_name = model_name
        self.top_n = top_n
        self.device = device
        self._model = model

    def _get_model(self) -> Any:
        """
        Lazily load and cache the SentenceTransformer CrossEncoder model.

        Returns:
            Instantiated CrossEncoder model.
        """
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(
                f"Loading CrossEncoder model '{self.model_name}' on device '{self.device}'"
            )
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ) -> list[NodeWithScore]:
        """
        Rerank candidate nodes using cross-encoder relevance scores.

        Args:
            nodes: Candidate nodes with initial retrieval scores.
            query_bundle: Query bundle containing the query string.

        Returns:
            Reordered list of nodes with normalized sigmoid scores, truncated to top_n.
        """
        if not nodes:
            return []
        if query_bundle is None or not query_bundle.query_str:
            return nodes[: self.top_n]

        query_str = query_bundle.query_str
        try:
            model = self._get_model()
            pairs = [(query_str, node.node.get_content()) for node in nodes]
            raw_scores = model.predict(pairs)

            reranked_nodes: list[NodeWithScore] = []
            for idx, raw_score in enumerate(raw_scores):
                s = float(raw_score)
                # Sigmoid normalization: 1.0 / (1.0 + math.exp(-raw_score))
                if s > 50.0:
                    normalized_score = 1.0
                elif s < -50.0:
                    normalized_score = 0.0
                else:
                    normalized_score = 1.0 / (1.0 + math.exp(-s))

                node_with_score = nodes[idx]
                node_with_score.score = normalized_score
                reranked_nodes.append(node_with_score)

            reranked_nodes.sort(key=lambda x: x.score or 0.0, reverse=True)
            return reranked_nodes[: self.top_n]

        except _INFERENCE_FALLBACK_EXCEPTIONS as e:
            logger.warning(
                f"Cross-encoder inference failed ({e}), falling back to initial retrieval candidates"
            )
            return nodes[: self.top_n]
