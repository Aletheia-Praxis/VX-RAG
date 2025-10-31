"""
Retriever Service implementation.

Provides classes for document retrieval operations.
"""

from typing import List, Dict, Any, Optional, cast
import logging
import yaml

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever, BaseRetriever

logger = logging.getLogger(__name__)

class RetrieverService:
    """Service for retrieving documents from index."""
    
    def __init__(self, index: Optional[VectorStoreIndex] = None, config_path: Optional[str] = None):
        self.index = index
        self.vector_retriever: Optional[BaseRetriever] = None
        self.hybrid_retriever: Optional[BaseRetriever] = None
        self.reranker = None
        self.config = self._load_config(config_path)
        self._initialize_reranker()
    
    def _initialize_reranker(self) -> None:
        """Initialize the reranker if configured."""
        reranker_config = self.config.get('reranker', {})
        if reranker_config.get('enable_metadata_prioritization', False):
            try:
                from sentence_transformers import CrossEncoder
                model_name = reranker_config.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
                self.reranker = CrossEncoder(model_name)
                logger.info(f"Initialized reranker with model: {model_name}")
            except ImportError:
                logger.warning("sentence_transformers not available, reranker disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            config_path = "config/settings.yaml"
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = cast(Dict[str, Any], yaml.safe_load(f))
            if isinstance(loaded_config, dict):
                retriever_config = loaded_config.get('retriever', {})
                return cast(Dict[str, Any], retriever_config)
            else:
                logger.warning(f"Config file {config_path} does not contain a valid dict")
                return {}
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return {}
    
    def set_index(self, index: VectorStoreIndex) -> None:
        """Set the index for retrieval."""
        self.index = index
        # Initialize retrievers
        semantic_top_k = self.config.get('semantic_top_k', 20)
        
        self.vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=semantic_top_k
        )
        
        # For hybrid search, we'll use QueryFusionRetriever with multiple vector retrievers
        # Note: True hybrid (vector + keyword) requires BM25Retriever which may not be available
        # For now, using multiple vector retrievers with different parameters as approximation
        vector_retriever_2 = VectorIndexRetriever(
            index=index,
            similarity_top_k=semantic_top_k
        )
        
        try:
            self.hybrid_retriever = QueryFusionRetriever(
                [self.vector_retriever, vector_retriever_2],
                similarity_top_k=semantic_top_k,
                num_queries=1,
                llm=None,  # Disable LLM to avoid API key issues
                use_async=True,
                verbose=False
            )
            logger.info("Initialized hybrid retriever with QueryFusionRetriever")
        except Exception as e:
            logger.warning(f"Failed to initialize QueryFusionRetriever: {e}")
            self.hybrid_retriever = cast(Optional[QueryFusionRetriever], self.vector_retriever)
    
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
            
            # Apply reranking if available
            if self.reranker and len(nodes) > 1:
                nodes = self._rerank_nodes(query, nodes)
            
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
    
    def _rerank_nodes(self, query: str, nodes: List[Any]) -> List[Any]:
        """Rerank nodes using cross-encoder."""
        if not self.reranker:
            return nodes
        
        try:
            # Prepare pairs for reranking
            pairs = [[query, node.text] for node in nodes]
            scores = self.reranker.predict(pairs)
            
            # Sort nodes by reranker scores
            scored_nodes = list(zip(nodes, scores))
            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            
            reranked_nodes = [node for node, score in scored_nodes]
            logger.info(f"Reranked {len(reranked_nodes)} nodes")
            return reranked_nodes
            
        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original nodes")
            return nodes
    
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
