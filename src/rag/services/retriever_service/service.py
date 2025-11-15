"""
Retriever Service implementation.

Provides lightweight orchestration of LlamaIndex native retrievers and postprocessors
with VX-RAG specific enhancements (metadata boost, BM25 integration).
"""

from typing import List, Dict, Any, Optional
import importlib.util

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.vector_stores import MetadataFilters, FilterCondition, MetadataFilter
from llama_index.retrievers.bm25 import BM25Retriever

# Import structured logging and metrics
from src.utils.logging_config import get_logger
from src.utils.metrics import get_metrics
from src.utils.config_loader import get_retriever_config, get_bm25_config, get_reranker_config
from src.rag.libs.postprocessors import MetadataBoostPostprocessor

logger = get_logger("retriever_service")
metrics = get_metrics()

# Check BM25 manager availability
_bm25_available = importlib.util.find_spec("src.rag.libs.bm25_manager") is not None
if not _bm25_available:
    logger.warning("BM25IndexManager not available")

class RetrieverService:
    """
    Lightweight orchestrator for LlamaIndex retrievers and postprocessors.
    
    This service initializes and coordinates:
    - Vector/BM25/Hybrid retrievers (native LlamaIndex)
    - Postprocessor chain (metadata boost + cross-encoder reranking)
    - BM25IndexManager integration
    - Metrics and logging
    """
    
    def __init__(self, index: Optional[VectorStoreIndex] = None, config_path: Optional[str] = None):
        """Initialize retriever service with config."""
        self.index = index
        self.vector_retriever: Optional[BaseRetriever] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.hybrid_retriever: Optional[BaseRetriever] = None
        self.bm25_manager: Optional[Any] = None
        self.config_path = config_path
        self.config = get_retriever_config(config_path)
        
        # Initialize native LlamaIndex postprocessors
        self.metadata_boost: Optional[MetadataBoostPostprocessor] = None
        self.reranker: Optional[SentenceTransformerRerank] = None
        self._initialize_postprocessors()
        
        # Initialize BM25 manager if available
        self._initialize_bm25_manager()
    
    def _initialize_postprocessors(self) -> None:
        """Initialize LlamaIndex native postprocessors."""
        reranker_config = get_reranker_config(self.config_path)
        
        # Initialize metadata boost postprocessor (custom VX-RAG feature)
        try:
            self.metadata_boost = MetadataBoostPostprocessor(config_path=self.config_path)
            logger.info("Initialized MetadataBoostPostprocessor")
        except Exception as e:
            logger.warning(f"Failed to initialize metadata boost: {e}")
        
        # Initialize native LlamaIndex reranker
        if reranker_config.get('enable_metadata_prioritization', False):
            try:
                model_name = reranker_config['model_name']
                top_n = reranker_config.get('top_n', 5)
                self.reranker = SentenceTransformerRerank(
                    model=model_name,
                    top_n=top_n
                )
                logger.info(f"Initialized SentenceTransformerRerank: model={model_name}, top_n={top_n}")
            except ImportError:
                logger.warning("sentence_transformers not available, reranker disabled")
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
    
    def _initialize_bm25_manager(self) -> None:
        """Initialize BM25 manager if available."""
        if _bm25_available:
            try:
                from src.rag.libs.bm25_manager import BM25IndexManager
                bm25_config = get_bm25_config(self.config_path)
                index_dir = bm25_config['index_dir']
                self.bm25_manager = BM25IndexManager(index_dir=index_dir)
                logger.info("Initialized BM25IndexManager")
            except Exception as e:
                logger.warning(f"Failed to initialize BM25 manager: {e}")
        else:
            logger.warning("BM25IndexManager not available")
    
    def set_index(self, index: VectorStoreIndex) -> None:
        """Set the index and initialize native LlamaIndex retrievers."""
        self.index = index
        semantic_top_k = self.config.get('semantic_top_k', 20)
        
        # Initialize vector retriever
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
        
        # Initialize hybrid retriever (native QueryFusionRetriever)
        self._initialize_hybrid_retriever(semantic_top_k)
    
    def _initialize_hybrid_retriever(self, semantic_top_k: int) -> None:
        """Initialize QueryFusionRetriever for hybrid search."""
        from llama_index.core.retrievers import QueryFusionRetriever
        
        if self.vector_retriever and self.bm25_retriever:
            # Hybrid: vector + BM25
            try:
                self.hybrid_retriever = QueryFusionRetriever(
                    [self.vector_retriever, self.bm25_retriever],
                    similarity_top_k=semantic_top_k,
                    num_queries=1,
                    llm=None,  # Disable LLM to avoid API key issues
                    use_async=True,
                    verbose=False
                )
                logger.info("Initialized QueryFusionRetriever with vector + BM25")
            except Exception as e:
                logger.warning(f"Failed to initialize hybrid retriever: {e}")
                self.hybrid_retriever = self.vector_retriever
        else:
            # Fallback: vector only
            self.hybrid_retriever = self.vector_retriever
            logger.info("Hybrid retriever initialized with vector only (BM25 unavailable)")
    
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
    
    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        filters: Optional[Dict[str, Any]] = None, 
        search_type: str = "semantic"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant documents using native LlamaIndex retrievers + postprocessors.
        
        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filters: Metadata filters (converted to MetadataFilters)
            search_type: "semantic", "keyword", or "hybrid"
        
        Returns:
            List of retrieved documents with scores and metadata
        """
        import time
        from llama_index.core.schema import QueryBundle
        
        start_time = time.time()
        
        if self.index is None:
            logger.error("Retrieval failed: no index set", query=query)
            return []
        
        try:
            # Select retriever
            retriever = self._select_retriever(search_type)
            if retriever is None:
                logger.error("Retrieval failed: retriever not initialized", query=query)
                return []
            
            # Retrieve nodes (initial candidates)
            nodes = retriever.retrieve(query)
            
            # Apply native LlamaIndex filters if provided
            if filters:
                nodes = self._apply_filters(nodes, filters)
            
            # Apply postprocessor chain: metadata boost → reranking
            nodes = self._apply_postprocessors(query, nodes)
            
            # Limit to top_k
            nodes = nodes[:top_k]
            
            # Convert to result format
            results = self._nodes_to_results(nodes)
            
            # Log metrics
            duration = time.time() - start_time
            metrics.increment("retrieval_queries_total")
            metrics.histogram("retrieval_duration_ms", duration * 1000)
            metrics.gauge("retrieval_results_count", len(results))
            
            logger.info(
                "Documents retrieved successfully", 
                query=query, 
                search_type=search_type, 
                results_count=len(results), 
                duration_ms=duration * 1000
            )
            
            return results
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Document retrieval failed", 
                query=query, 
                search_type=search_type, 
                error=str(e), 
                duration_ms=duration * 1000
            )
            return []
    
    def _select_retriever(self, search_type: str) -> Optional[BaseRetriever]:
        """Select retriever based on search type."""
        if search_type == "hybrid":
            return self.hybrid_retriever
        elif search_type == "keyword":
            if self.bm25_retriever:
                return self.bm25_retriever
            else:
                logger.warning("BM25 retriever not available, using semantic search")
                return self.vector_retriever
        else:
            return self.vector_retriever
    
    def _apply_filters(self, nodes: List[Any], filters: Dict[str, Any]) -> List[Any]:
        """Apply metadata filters to nodes."""
        logger.info(f"Applying filters: {filters}")
        filtered_nodes = [
            node for node in nodes
            if all(
                node.metadata.get(key) == value
                for key, value in filters.items()
            )
        ]
        logger.info(f"Filtered {len(nodes)} → {len(filtered_nodes)} nodes")
        return filtered_nodes
    
    def _apply_postprocessors(self, query: str, nodes: List[Any]) -> List[Any]:
        """Apply postprocessor chain: metadata boost → reranking."""
        from llama_index.core.schema import QueryBundle
        
        if len(nodes) <= 1:
            return nodes
        
        query_bundle = QueryBundle(query_str=query)
        
        # Step 1: Metadata boost (custom VX-RAG feature)
        if self.metadata_boost:
            try:
                nodes = self.metadata_boost.postprocess_nodes(nodes, query_bundle)
            except Exception as e:
                logger.warning(f"Metadata boost failed: {e}")
        
        # Step 2: Cross-encoder reranking (native LlamaIndex)
        if self.reranker:
            try:
                nodes = self.reranker.postprocess_nodes(nodes, query_bundle)
            except Exception as e:
                logger.warning(f"Reranking failed: {e}")
        
        return nodes
    
    def _nodes_to_results(self, nodes: List[Any]) -> List[Dict[str, Any]]:
        """Convert nodes to result format."""
        return [
            {
                'text': node.text,
                'score': getattr(node, 'score', 0.0),
                'metadata': node.metadata,
                'node_id': node.id_
            }
            for node in nodes
        ]
    

    
    def hybrid_search(self, query: str, vector_query: Optional[List[float]] = None, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Perform hybrid vector + keyword search."""
        return self.retrieve(query, top_k, filters, search_type="hybrid")
