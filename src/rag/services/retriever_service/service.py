"""
Retriever Service implementation.

Provides classes for document retrieval operations.
"""

from typing import List, Dict, Any, Optional
import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.retrievers import QueryFusionRetriever

logger = logging.getLogger(__name__)

class RetrieverService:
    """Service for retrieving documents from index."""
    
    def __init__(self, index: Optional[VectorStoreIndex] = None):
        self.index = index
        self.vector_retriever = None
        self.hybrid_retriever = None
    
    def set_index(self, index: VectorStoreIndex) -> None:
        """Set the index for retrieval."""
        self.index = index
        # Initialize retrievers
        self.vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10
        )
        # For hybrid, we can use QueryFusionRetriever with multiple retrievers
        # For now, using vector retriever as hybrid fallback
        self.hybrid_retriever = self.vector_retriever
    
    def retrieve(self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None, search_type: str = "semantic") -> List[Dict[str, Any]]:
        """Retrieve top-k relevant documents for query.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filters: Metadata filters to apply
            search_type: "semantic", "keyword", or "hybrid"
        
        Returns:
            List of retrieved documents with scores and metadata
        """
        if self.index is None:
            logger.error("No index set for retrieval")
            return []
        
        try:
            if search_type == "hybrid":
                retriever = self.hybrid_retriever
            elif search_type == "keyword":
                # For now, fall back to vector retriever for keyword
                logger.warning("Keyword search not implemented, using semantic search")
                retriever = self.vector_retriever
            else:
                retriever = self.vector_retriever
            
            if retriever is None:
                logger.error("Retriever not initialized")
                return []
            
            # Apply filters if provided
            if filters:
                # Note: LlamaIndex filters need to be adapted based on metadata structure
                logger.info(f"Applying filters: {filters}")
                # For now, retrieve and filter post-hoc
                nodes = retriever.retrieve(query)
                filtered_nodes = []
                for node in nodes:
                    if self._matches_filters(node.metadata, filters):
                        filtered_nodes.append(node)
                        if len(filtered_nodes) >= top_k:
                            break
                nodes = filtered_nodes
            else:
                nodes = retriever.retrieve(query)
            
            # Limit to top_k
            nodes = nodes[:top_k]
            
            # Extract results
            results = []
            for node in nodes:
                result = {
                    'text': node.text,
                    'score': getattr(node, 'score', 0.0),
                    'metadata': node.metadata,
                    'node_id': node.id_
                }
                results.append(result)
            
            logger.info(f"Retrieved {len(results)} documents for query using {search_type} search")
            return results
            
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the given filters."""
        for key, value in filters.items():
            if key not in metadata:
                return False
            if isinstance(value, list):
                if metadata[key] not in value:
                    return False
            else:
                if metadata[key] != value:
                    return False
        return True
    
    def hybrid_search(self, query: str, vector_query: Optional[List[float]] = None, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform hybrid vector + keyword search."""
        return self.retrieve(query, top_k, filters, search_type="hybrid")
