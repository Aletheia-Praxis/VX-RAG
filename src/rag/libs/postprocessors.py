"""
Custom LlamaIndex postprocessors for VX-RAG.

This module provides custom postprocessor implementations that extend
LlamaIndex's standard postprocessor functionality with VX-RAG specific logic.
"""

from typing import List, Optional
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle

from src.utils.logging_config import get_logger
from src.utils.config_loader import get_reranker_config

logger = get_logger("postprocessors")


class MetadataBoostPostprocessor(BaseNodePostprocessor):
    """
    Postprocessor that boosts node scores based on metadata completeness.
    
    This is a VX-RAG-specific feature that prioritizes documents with richer
    metadata (source, language, topic, author, year) by applying a multiplicative
    boost to their retrieval scores.
    
    Args:
        boost_factor: Multiplicative factor per populated priority field (default: 0.1)
        priority_fields: List of metadata fields to check (default: VX-RAG fields)
        config_path: Path to config file for loading boost_factor
    
    Example:
        >>> postprocessor = MetadataBoostPostprocessor(boost_factor=0.15)
        >>> boosted_nodes = postprocessor.postprocess_nodes(nodes, query_bundle)
    """
    
    def __init__(
        self,
        boost_factor: Optional[float] = None,
        priority_fields: Optional[List[str]] = None,
        config_path: Optional[str] = None
    ):
        """Initialize the metadata boost postprocessor."""
        super().__init__()
        
        # Load config if boost_factor not provided
        if boost_factor is None:
            config = get_reranker_config(config_path)
            boost_factor = float(config.get('metadata_boost', 0.1))
        
        self.boost_factor: float = boost_factor
        self.priority_fields = priority_fields or [
            'source', 'lang', 'topic', 'author', 'year'
        ]
        
        logger.info(
            f"Initialized MetadataBoostPostprocessor: "
            f"boost_factor={self.boost_factor}, "
            f"priority_fields={self.priority_fields}"
        )
    
    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None
    ) -> List[NodeWithScore]:
        """
        Apply metadata-based score boosting to nodes.
        
        Args:
            nodes: List of nodes with scores
            query_bundle: Query bundle (unused but required by interface)
            
        Returns:
            Nodes with boosted scores based on metadata completeness
        """
        try:
            for node_with_score in nodes:
                metadata = node_with_score.node.metadata
                current_score = node_with_score.score or 0.0
                
                # Count populated priority fields
                populated_fields = sum(
                    1 for field in self.priority_fields
                    if field in metadata and metadata[field]
                )
                
                # Apply boost
                if populated_fields > 0:
                    boost_multiplier = 1.0 + (self.boost_factor * populated_fields)
                    boosted_score = current_score * boost_multiplier
                    node_with_score.score = boosted_score
                    
                    logger.debug(
                        f"Metadata boost applied: {current_score:.4f} -> {boosted_score:.4f} "
                        f"(fields: {populated_fields}, multiplier: {boost_multiplier:.2f})"
                    )
            
            logger.info(
                f"Applied metadata boost to {len(nodes)} nodes "
                f"(factor={self.boost_factor})"
            )
            return nodes
            
        except Exception as e:
            logger.warning(
                f"Failed to apply metadata boost: {e}, returning original nodes"
            )
            return nodes
