"""
Retriever Service implementation.

Provides classes for document retrieval operations.
"""

from typing import List, Dict, Any, Optional, cast, TYPE_CHECKING
import importlib.util

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

# Import structured logging and metrics
from src.utils.logging_config import get_logger
from src.utils.metrics import get_metrics
from src.utils.config_loader import get_retriever_config, get_bm25_config, get_reranker_config

logger = get_logger("retriever_service")
metrics = get_metrics()

# Check BM25 manager availability
_bm25_available = importlib.util.find_spec("src.rag.libs.bm25_manager") is not None
if not _bm25_available:
    logger.warning("BM25IndexManager not available")

class RetrieverService:
    """Service for retrieving documents from index."""
    
    def __init__(self, index: Optional[VectorStoreIndex] = None, config_path: Optional[str] = None):
        self.index = index
        self.vector_retriever: Optional[BaseRetriever] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.hybrid_retriever: Optional[BaseRetriever] = None
        self.bm25_manager: Optional[Any] = None
        self.reranker: Optional['CrossEncoder'] = None
        self.config_path = config_path
        self.config = get_retriever_config(config_path)
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize BM25 manager and reranker."""
        # Initialize BM25 manager if available
        if _bm25_available:
            from src.rag.libs.bm25_manager import BM25IndexManager
            bm25_config = get_bm25_config(self.config_path)
            index_dir = bm25_config['index_dir']
            self.bm25_manager = BM25IndexManager(index_dir=index_dir)
            logger.info("Initialized BM25 manager")
        else:
            logger.warning("BM25 service not available")
        
        # Initialize reranker
        self._initialize_reranker()
    
    def _initialize_reranker(self) -> None:
        """Initialize the reranker if configured."""
        reranker_config = get_reranker_config(self.config_path)
        if reranker_config.get('enable_metadata_prioritization', False):
            try:
                from sentence_transformers import CrossEncoder
                model_name = reranker_config['model_name']
                device = reranker_config['device']
                self.reranker = CrossEncoder(model_name, device=device)
                logger.info(f"Initialized reranker: model={model_name}, device={device}")
            except ImportError:
                logger.warning("sentence_transformers not available, reranker disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from YAML file.
        
        DEPRECATED: Use get_retriever_config from config_loader instead.
        """
        logger.warning("_load_config is deprecated, use get_retriever_config instead")
        return get_retriever_config(config_path)
    
    def set_index(self, index: VectorStoreIndex) -> None:
        """Set the index for retrieval."""
        self.index = index
        # Initialize retrievers
        semantic_top_k = self.config.get('semantic_top_k', 20)
        
        self.vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=semantic_top_k
        )
        
        # Initialize BM25 retriever if manager is available
        if self.bm25_manager:
            try:
                self.bm25_retriever = self.bm25_manager.load()
                if self.bm25_retriever:
                    logger.info("BM25 retriever loaded from manager")
            except Exception as e:
                logger.warning(f"Failed to load BM25 retriever: {e}")
        
        # Initialize hybrid retriever if both vector and BM25 are available
        if self.vector_retriever and self.bm25_retriever:
            try:
                from llama_index.core.retrievers import QueryFusionRetriever
                self.hybrid_retriever = QueryFusionRetriever(
                    [self.vector_retriever, self.bm25_retriever],
                    similarity_top_k=semantic_top_k,
                    num_queries=1,
                    llm=None,  # Disable LLM to avoid API key issues
                    use_async=True,
                    verbose=False
                )
                logger.info("Initialized hybrid retriever with vector and BM25")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid retriever: {e}")
                self.hybrid_retriever = cast(Optional[QueryFusionRetriever], self.vector_retriever)
        else:
            # Fallback to QueryFusion with multiple vector retrievers
            try:
                from llama_index.core.retrievers import QueryFusionRetriever
                vector_retriever_2 = VectorIndexRetriever(
                    index=index,
                    similarity_top_k=semantic_top_k
                )
                self.hybrid_retriever = QueryFusionRetriever(
                    [self.vector_retriever, vector_retriever_2],
                    similarity_top_k=semantic_top_k,
                    num_queries=1,
                    llm=None,
                    use_async=True,
                    verbose=False
                )
                logger.info("Initialized hybrid retriever with QueryFusionRetriever (vector only)")
            except Exception as e:
                logger.warning(f"Failed to initialize QueryFusionRetriever: {e}")
                self.hybrid_retriever = cast(Optional[QueryFusionRetriever], self.vector_retriever)
    
    def build_bm25_index(self, documents: List[Any]) -> None:
        """Build BM25 index from documents.
        
        DEPRECATED: Use BM25IndexManager.build_and_persist() directly instead.
        This method is kept for backward compatibility but will be removed.
        """
        import time
        start_time = time.time()
        
        if self.bm25_manager:
            try:
                self.bm25_manager.build_and_persist(documents)
                # Re-initialize retrievers with new BM25
                if self.index:
                    self.set_index(self.index)
                
                duration = time.time() - start_time
                metrics.increment("bm25_index_builds_total")
                metrics.histogram("bm25_index_build_duration_ms", duration * 1000)
                
                logger.info("BM25 index built and saved", 
                           documents_count=len(documents), 
                           duration_ms=duration * 1000)
            except Exception as e:
                duration = time.time() - start_time
                logger.error("Failed to build BM25 index", 
                           error=str(e), 
                           duration_ms=duration * 1000)
        else:
            logger.warning("BM25 manager not available")
    
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
        import time
        start_time = time.time()
        
        if self.index is None:
            logger.error("Retrieval failed: no index set", query=query)
            return []
        
        try:
            if search_type == "hybrid":
                retriever = self.hybrid_retriever
            elif search_type == "keyword":
                if self.bm25_retriever:
                    retriever = self.bm25_retriever
                else:
                    logger.warning("BM25 retriever not available, using semantic search", query=query)
                    retriever = self.vector_retriever
            else:
                retriever = self.vector_retriever
            
            if retriever is None:
                logger.error("Retrieval failed: retriever not initialized", query=query)
                return []
            
            # Apply filters if provided
            if filters:
                logger.info("Applying filters to retrieval", query=query, filters=filters)
                # Note: LlamaIndex filters need to be adapted based on metadata structure
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
            
            # Apply metadata boost and reranking if available
            if self.reranker and len(nodes) > 1:
                # Apply metadata boost before reranking
                nodes = self._apply_metadata_boost_to_nodes(nodes)
                # Then rerank with cross-encoder
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
            
            duration = time.time() - start_time
            
            # Log metrics
            metrics.increment("retrieval_queries_total")
            metrics.histogram("retrieval_duration_ms", duration * 1000)
            metrics.gauge("retrieval_results_count", len(results))
            
            logger.info("Documents retrieved successfully", 
                       query=query, 
                       search_type=search_type, 
                       results_count=len(results), 
                       duration_ms=duration * 1000)
            
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error("Document retrieval failed", 
                        query=query, 
                        search_type=search_type, 
                        error=str(e), 
                        duration_ms=duration * 1000)
            return []
    
    def _apply_metadata_boost_to_nodes(self, nodes: List[Any]) -> List[Any]:
        """Apply metadata-based score boosting to nodes.
        
        Args:
            nodes: List of nodes with scores and metadata
            
        Returns:
            Nodes with boosted scores
        """
        try:
            reranker_config = get_reranker_config(self.config_path)
            boost_factor = reranker_config.get('metadata_boost', 0.1)
            
            # Default priority fields for VX-RAG
            priority_fields = ['source', 'lang', 'topic', 'author', 'year']
            
            for node in nodes:
                metadata = node.metadata
                current_score = getattr(node, 'score', 0.0)
                
                # Count populated priority fields
                populated_fields = sum(
                    1 for field in priority_fields
                    if field in metadata and metadata[field]
                )
                
                # Apply boost
                if populated_fields > 0:
                    boost_multiplier = 1.0 + (boost_factor * populated_fields)
                    boosted_score = current_score * boost_multiplier
                    node.score = boosted_score
                    
                    logger.debug(
                        f"Metadata boost applied to node: {current_score:.4f} -> {boosted_score:.4f} "
                        f"(fields: {populated_fields}, multiplier: {boost_multiplier:.2f})"
                    )
            
            logger.info(f"Applied metadata boost to {len(nodes)} nodes (factor={boost_factor})")
            return nodes
            
        except Exception as e:
            logger.warning(f"Failed to apply metadata boost: {e}, returning original nodes")
            return nodes
    
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
