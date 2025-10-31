"""
Retriever Service implementation.

Provides classes for document retrieval operations.
"""

from typing import List, Dict, Any, Optional, cast, TYPE_CHECKING
import logging
import yaml

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever, BaseRetriever
from llama_index.retrievers.bm25 import BM25Retriever

if TYPE_CHECKING:
    from ..bm25_service.service import BM25Service

logger = logging.getLogger(__name__)

# Import BM25 service
try:
    from ..bm25_service.service import BM25Service
    _bm25_available = True
except ImportError:
    logger.warning("BM25Service not available")
    _bm25_available = False

class RetrieverService:
    """Service for retrieving documents from index."""
    
    def __init__(self, index: Optional[VectorStoreIndex] = None, config_path: Optional[str] = None):
        self.index = index
        self.vector_retriever: Optional[BaseRetriever] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.hybrid_retriever: Optional[BaseRetriever] = None
        self.bm25_service: Optional[Any] = None
        self.reranker = None
        self.config = self._load_config(config_path)
        self._initialize_services()
    
    def _initialize_services(self) -> None:
        """Initialize BM25 service and reranker."""
        # Initialize BM25 service if available
        if _bm25_available:
            from ..bm25_service.service import BM25Service as BM25ServiceClass
            bm25_config = self.config.get('bm25', {})
            index_dir = bm25_config.get('index_dir', 'data/index/bm25')
            self.bm25_service = BM25ServiceClass(index_dir=index_dir)
            logger.info("Initialized BM25 service")
        else:
            logger.warning("BM25 service not available")
        
        # Initialize reranker
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
        
        # Initialize BM25 retriever if service is available
        if self.bm25_service and self.bm25_service.is_index_built():
            try:
                self.bm25_retriever = self.bm25_service.bm25_retriever
                logger.info("BM25 retriever initialized from service")
            except Exception as e:
                logger.warning(f"Failed to initialize BM25 retriever: {e}")
        
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
        """Build BM25 index from documents."""
        if self.bm25_service:
            try:
                self.bm25_service.build_index(documents)
                self.bm25_service.save_index()
                # Re-initialize retrievers with new BM25
                if self.index:
                    self.set_index(self.index)
                logger.info("BM25 index built and saved")
            except Exception as e:
                logger.error(f"Failed to build BM25 index: {e}")
        else:
            logger.warning("BM25 service not available")
    
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
                if self.bm25_retriever:
                    retriever = self.bm25_retriever
                else:
                    logger.warning("BM25 retriever not available, using semantic search")
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
