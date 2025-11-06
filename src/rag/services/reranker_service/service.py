"""
Reranker Service implementation.

Provides classes for re-ranking retrieved documents.
"""

from typing import List, Dict, Any, Optional, cast
import logging

from sentence_transformers import CrossEncoder
from src.utils.config_loader import get_reranker_config

logger = logging.getLogger(__name__)

class RerankerService:
    """Service for re-ranking documents using cross-encoders."""
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """Initialize the reranker service.
        
        Args:
            model_name: Cross-encoder model name. If None, loads from config.
            device: Device to use (cpu/cuda). If None, loads from config.
            config_path: Path to settings.yaml. If None, uses default location.
        """
        # Load config if parameters not provided
        if model_name is None or device is None:
            config = get_reranker_config(config_path)
            model_name = model_name or config['model_name']
            device = device or config['device']
        
        # Ensure values are set
        assert model_name is not None, "model_name must be set"
        assert device is not None, "device must be set"
        
        self.model_name = model_name
        self.device = device
        self.config = get_reranker_config(config_path)
        
        try:
            self.model = CrossEncoder(model_name, device=device)
            logger.info(f"Initialized cross-encoder: model={model_name}, device={device}")
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model {model_name}: {e}")
            self.model = None
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        DEPRECATED: Use get_reranker_config from config_loader instead.
        """
        logger.warning("_load_config is deprecated, use get_reranker_config instead")
        return get_reranker_config(config_path)
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Re-rank documents based on query relevance using cross-encoder.
        
        Args:
            query: The search query
            documents: List of documents with 'text' field
            top_k: Number of top documents to return, if None return all
        
        Returns:
            Re-ranked list of documents with updated scores
        """
        if self.model is None:
            logger.warning("Cross-encoder model not loaded, returning documents as-is")
            return documents
        
        if not documents:
            return documents
        
        try:
            # Prepare input pairs
            pairs = [(query, doc.get('text', '')) for doc in documents]
            
            # Get scores from cross-encoder
            scores = self.model.predict(pairs)
            
            # Update documents with new scores
            reranked_docs = []
            for doc, score in zip(documents, scores):
                updated_doc = doc.copy()
                updated_doc['score'] = float(score)
                reranked_docs.append(updated_doc)
            
            # Sort by score (higher is better for cross-encoder)
            reranked_docs.sort(key=lambda x: x['score'], reverse=True)
            
            # Limit to top_k if specified
            if top_k is not None:
                reranked_docs = reranked_docs[:top_k]
            
            logger.info(f"Re-ranked {len(documents)} documents, returned top {len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.error(f"Failed to rerank documents: {e}")
            return documents
    
    def prioritize_by_metadata(self, documents: List[Dict[str, Any]], priority_rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize documents based on metadata rules.
        
        Args:
            documents: List of documents
            priority_rules: Dict with metadata keys and priority values
        
        Returns:
            Prioritized list of documents
        """
        if not priority_rules:
            return documents
        
        try:
            # Calculate priority scores
            for doc in documents:
                priority_score = 0.0
                metadata = doc.get('metadata', {})
                
                for key, priority_value in priority_rules.items():
                    if key in metadata:
                        if isinstance(priority_value, dict):
                            # Range-based priority
                            value = metadata[key]
                            if 'min' in priority_value and value >= priority_value['min']:
                                priority_score += priority_value.get('weight', 1.0)
                            if 'max' in priority_value and value <= priority_value['max']:
                                priority_score += priority_value.get('weight', 1.0)
                        else:
                            # Exact match priority
                            if metadata[key] == priority_value:
                                priority_score += 1.0
                
                doc['priority_score'] = priority_score
            
            # Sort by priority score, then by original score
            documents.sort(key=lambda x: (x.get('priority_score', 0), x.get('score', 0)), reverse=True)
            
            logger.info(f"Prioritized {len(documents)} documents based on metadata rules")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to prioritize documents: {e}")
            return documents
