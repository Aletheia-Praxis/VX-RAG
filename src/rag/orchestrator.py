"""
RAG Orchestrator - High-level coordinator for RAG pipeline operations.

This module provides a clean, high-level interface for coordinating all RAG services.
It acts as the central orchestration layer between the MCP interface and individual
RAG services, managing the complete document retrieval and query pipeline.

Key responsibilities:
- Coordinate retrieval pipeline (embedder -> retriever -> postprocessors -> assembler)
- Manage service lifecycle and initialization
- Provide health checks and system status
- Handle errors and logging at the orchestration level

Note: Reranking is now handled by native LlamaIndex postprocessors within RetrieverService.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import time

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from .services.vectordb_service.service import VectorStoreClient
from src.rag.libs.bm25_manager import BM25IndexManager
from .services.retriever_service.service import RetrieverService
from .services.assembler_service.service import ContextAssembler
from .exceptions import (
    ServiceInitializationError,
    RetrievalError,
)

from src.utils.logging_config import get_logger, log_service_health
from src.utils.metrics import get_metrics

logger = get_logger("rag_orchestrator")
metrics = get_metrics()


class RAGOrchestrator:
    """
    High-level orchestrator for RAG pipeline operations.
    
    Coordinates all RAG services and provides a unified interface for:
    - Document querying with hybrid retrieval
    - Health checks and system status
    - Service initialization and management
    """

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        persist_dir: str = "data/index",
        auto_load: bool = True
    ) -> None:
        """
        Initialize the RAG orchestrator.
        
        Args:
            config_path: Path to configuration file
            persist_dir: Directory containing persisted indexes
            auto_load: Whether to automatically load indexes on initialization
        """
        self.config_path = config_path
        self.persist_dir = Path(persist_dir)
        
        # Service instances (lazy initialization)
        # Note: Embedding model is now managed via Settings.embed_model (global LlamaIndex config)
        # Note: Reranking is now handled by RetrieverService postprocessors (no separate service)
        self._vector_store: Optional[VectorStoreClient] = None
        self._bm25_manager: Optional[BM25IndexManager] = None
        self._retriever: Optional[RetrieverService] = None
        self._assembler: Optional[ContextAssembler] = None
        
        # Status flags
        self._initialized = False
        self._indexes_loaded = False
        
        logger.info(
            "RAG Orchestrator created",
            config_path=config_path,
            persist_dir=str(persist_dir),
            auto_load=auto_load
        )
        
        if auto_load:
            self._initialize_services()

    def _initialize_services(self) -> None:
        """Initialize all RAG services and load indexes."""
        if self._initialized:
            logger.debug("Services already initialized")
            return
        
        try:
            logger.info("Initializing RAG services")
            start_time = time.time()
            
            # Configure global embedding model via Settings
            from src.utils.config_loader import get_embedding_config
            embed_config = get_embedding_config(self.config_path)
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=embed_config['embedding_model'],
                embed_batch_size=embed_config['embedding_batch_size'],
                trust_remote_code=embed_config['embedding_trust_remote_code']
            )
            logger.info(
                "Embedding model configured",
                model=embed_config['embedding_model'],
                batch_size=embed_config['embedding_batch_size']
            )
            log_service_health("embed_model", "initialized")
            
            # Initialize vector store
            faiss_index_path = self.persist_dir / "faiss_index"
            self._vector_store = VectorStoreClient(
                store_type="faiss",
                config={'index_dir': str(faiss_index_path)}
            )
            
            # Load FAISS index
            if faiss_index_path.exists():
                index_loaded = self._vector_store.load_index(
                    embed_model=Settings.embed_model
                )
                if index_loaded and self._vector_store.index:
                    self._indexes_loaded = True
                    logger.info("FAISS index loaded successfully")
                    log_service_health("vector_store", "loaded")
                else:
                    logger.warning("FAISS index exists but failed to load")
                    log_service_health("vector_store", "error", error="load_failed")
            else:
                logger.warning(f"FAISS index not found at {faiss_index_path}")
                log_service_health("vector_store", "not_found")
            
            # Initialize BM25 manager
            bm25_index_path = self.persist_dir / "bm25_index"
            self._bm25_manager = BM25IndexManager(
                index_dir=str(bm25_index_path),
                config_path=self.config_path
            )
            
            # Load BM25 index
            if bm25_index_path.exists():
                bm25_retriever = self._bm25_manager.load()
                if bm25_retriever:
                    logger.info("BM25 index loaded successfully")
                    log_service_health("bm25_manager", "loaded")
                else:
                    logger.warning("BM25 index load returned None")
                    log_service_health("bm25_manager", "load_failed")
            else:
                logger.warning(f"BM25 index not found at {bm25_index_path}")
                log_service_health("bm25_manager", "not_found")
            
            # Initialize retriever (requires loaded indexes)
            # Note: Reranking now integrated as postprocessors in RetrieverService
            if self._indexes_loaded and self._vector_store.index:
                self._retriever = RetrieverService(
                    index=self._vector_store.index,
                    config_path=self.config_path
                )
                # BM25 retriever and reranking postprocessors managed internally
                logger.info(
                    "Retriever service initialized with hybrid search and postprocessors"
                )
                log_service_health("retriever", "initialized")
            else:
                logger.error("Cannot initialize retriever: indexes not loaded")
                log_service_health("retriever", "error", error="indexes_not_loaded")
            
            # Initialize context assembler
            self._assembler = ContextAssembler(config_path=self.config_path)
            logger.info("Context assembler initialized")
            log_service_health("assembler", "initialized")
            
            self._initialized = True
            duration = time.time() - start_time
            
            logger.info(
                "RAG services initialization complete",
                duration_ms=duration * 1000,
                indexes_loaded=self._indexes_loaded
            )
            
            metrics.histogram("orchestrator_init_duration_ms", duration * 1000)
            
        except (ImportError, ModuleNotFoundError) as e:
            error_msg = f"Missing required dependency: {e}"
            logger.error("Failed to initialize RAG services", error=error_msg, exc_info=True)
            log_service_health("orchestrator", "error", error=error_msg)
            raise ServiceInitializationError("orchestrator", error_msg) from e
        
        except (FileNotFoundError, IOError) as e:
            error_msg = f"File system error: {e}"
            logger.error("Failed to initialize RAG services", error=error_msg, exc_info=True)
            log_service_health("orchestrator", "error", error=error_msg)
            raise ServiceInitializationError("orchestrator", error_msg) from e
        
        except Exception as e:
            error_msg = f"Unexpected error during initialization: {e}"
            logger.error("Failed to initialize RAG services", error=error_msg, exc_info=True)
            log_service_health("orchestrator", "error", error=error_msg)
            raise ServiceInitializationError("orchestrator", error_msg) from e

    def query(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        token_budget: int = 4000
    ) -> Dict[str, Any]:
        """
        Execute a complete query pipeline: retrieve, rerank, assemble context.
        
        Args:
            query: The search query string
            top_k: Number of top results to return after reranking
            search_type: Type of search ("semantic", "keyword", "hybrid")
            token_budget: Maximum token budget for assembled context
            
        Returns:
            Dictionary containing:
                - query: Original query string
                - context: List of context items with text, score, metadata
                - total_tokens_estimate: Estimated token count
                - sources_count: Number of source documents
                - retrieval_stats: Performance statistics
        
        Raises:
            RuntimeError: If services are not initialized or indexes not loaded
        """
        if not self._initialized:
            raise RuntimeError("RAG services not initialized")
        
        if not self._indexes_loaded:
            raise RuntimeError("Indexes not loaded")
        
        if not self._retriever:
            raise RuntimeError("Retriever service not available")
        
        start_time = time.time()
        request_id = f"query_{int(time.time() * 1000)}"
        
        logger.info(
            "Starting query pipeline",
            request_id=request_id,
            query=query,
            top_k=top_k,
            search_type=search_type
        )
        
        try:
            # Step 1: Initial retrieval (over-retrieve for reranking)
            initial_k = top_k * 4
            retrieve_start = time.time()
            
            if search_type == "hybrid":
                # Hybrid search: combine vector and BM25
                vector_results = self._retriever.retrieve(
                    query,
                    top_k=initial_k,
                    search_type="semantic"
                )
                bm25_results = self._retriever.retrieve(
                    query,
                    top_k=initial_k,
                    search_type="keyword"
                )
                
                # Merge and deduplicate
                all_candidates = vector_results + bm25_results
                seen_ids = set()
                unique_candidates = []
                for doc in all_candidates:
                    node_id = doc.get('node_id', doc.get('id', ''))
                    if node_id and node_id not in seen_ids:
                        seen_ids.add(node_id)
                        unique_candidates.append(doc)
                
                retrieved_docs = unique_candidates
            else:
                # Single search type
                retrieved_docs = self._retriever.retrieve(
                    query,
                    top_k=initial_k,
                    search_type=search_type
                )
            
            retrieve_duration = time.time() - retrieve_start
            
            logger.info(
                "Retrieval complete",
                request_id=request_id,
                candidates=len(retrieved_docs),
                duration_ms=retrieve_duration * 1000
            )
            
            # Step 2: Results already reranked by RetrieverService postprocessors
            # RetrieverService applies: metadata boost -> cross-encoder reranking
            # So retrieved_docs are already in optimal order
            reranked_docs = retrieved_docs[:top_k]
            
            logger.info(
                "Postprocessing complete (via RetrieverService)",
                request_id=request_id,
                results=len(reranked_docs)
            )
            
            # Step 3: Assemble context
            assemble_start = time.time()
            
            if self._assembler:
                context_payload = self._assembler.assemble_context(
                    query=query,
                    documents=reranked_docs,
                    token_budget=token_budget,
                    max_items=top_k
                )
            else:
                # Fallback: create simple context
                from .libs.schemas.mcp_schemas import MCPContextPayload, ContextItem
                context_items = []
                for doc in reranked_docs[:top_k]:
                    item = ContextItem(
                        id=doc.get('node_id', doc.get('id', '')),
                        text=doc.get('text', ''),
                        score=doc.get('score'),
                        meta=doc.get('metadata', {})
                    )
                    context_items.append(item)
                context_payload = MCPContextPayload(
                    query=query,
                    context=context_items,
                    schema_version="1.0",
                    token_budget=token_budget
                )
            
            assemble_duration = time.time() - assemble_start
            total_duration = time.time() - start_time
            
            # Build response
            response = {
                'query': query,
                'context': [
                    {
                        'id': item.id,
                        'text': item.text,
                        'score': item.score,
                        'metadata': item.meta
                    }
                    for item in context_payload.context
                ],
                'total_tokens_estimate': context_payload.total_tokens_estimate(),
                'sources_count': len(context_payload.context),
                'retrieval_stats': {
                    'retrieve_duration_ms': round(retrieve_duration * 1000, 2),
                    'assemble_duration_ms': round(assemble_duration * 1000, 2),
                    'total_duration_ms': round(total_duration * 1000, 2),
                    'candidates_retrieved': len(retrieved_docs),
                    'results_postprocessed': len(reranked_docs),
                    'search_type': search_type,
                    'note': 'Postprocessing (metadata boost + reranking) included in retrieve_duration'
                }
            }
            
            logger.info(
                "Query pipeline complete",
                request_id=request_id,
                results=len(context_payload.context),
                tokens=context_payload.total_tokens_estimate(),
                duration_ms=total_duration * 1000
            )
            
            # Record metrics
            metrics.increment("orchestrator_queries_total")
            metrics.histogram("orchestrator_query_duration_ms", total_duration * 1000)
            metrics.histogram("orchestrator_retrieve_duration_ms", retrieve_duration * 1000)
            # Note: reranking metrics now included in retrieve_duration
            metrics.gauge("orchestrator_results_count", len(context_payload.context))
            
            return response
            
        except ValueError as e:
            # Query validation or parameter errors
            duration = time.time() - start_time
            logger.error(
                "Query pipeline failed: invalid parameters",
                request_id=request_id,
                query=query,
                error=str(e),
                duration_ms=duration * 1000
            )
            metrics.increment("orchestrator_query_errors_total")
            raise RetrievalError(query, f"Invalid parameters: {e}") from e
        
        except (KeyError, AttributeError) as e:
            # Missing data or attribute errors
            duration = time.time() - start_time
            logger.error(
                "Query pipeline failed: data structure error",
                request_id=request_id,
                query=query,
                error=str(e),
                duration_ms=duration * 1000,
                exc_info=True
            )
            metrics.increment("orchestrator_query_errors_total")
            raise RetrievalError(query, f"Data structure error: {e}") from e
        
        except Exception as e:
            # Unexpected errors
            duration = time.time() - start_time
            logger.error(
                "Query pipeline failed: unexpected error",
                request_id=request_id,
                query=query,
                error=str(e),
                duration_ms=duration * 1000,
                exc_info=True
            )
            metrics.increment("orchestrator_query_errors_total")
            raise RetrievalError(query, f"Unexpected error: {e}") from e

    def search_documents(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "semantic"
    ) -> List[Dict[str, Any]]:
        """
        Simple document search without reranking or context assembly.
        
        Args:
            query: The search query string
            top_k: Number of results to return
            search_type: Type of search ("semantic", "keyword", "hybrid")
            
        Returns:
            List of document dictionaries with text, score, and metadata
        """
        if not self._initialized or not self._retriever:
            raise RuntimeError("RAG services not initialized")
        
        logger.info(
            "Executing document search",
            query=query,
            top_k=top_k,
            search_type=search_type
        )
        
        try:
            results = self._retriever.retrieve(
                query,
                top_k=top_k,
                search_type=search_type
            )
            
            logger.info(
                "Document search complete",
                query=query,
                results=len(results)
            )
            
            return results
            
        except ValueError as e:
            logger.error("Document search failed: invalid parameters", query=query, error=str(e))
            raise RetrievalError(query, f"Invalid parameters: {e}") from e
        
        except Exception as e:
            logger.error("Document search failed: unexpected error", query=query, error=str(e), exc_info=True)
            raise RetrievalError(query, f"Search failed: {e}") from e

    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of all RAG services.
        
        Returns:
            Dictionary with health status of each service and overall system status
        """
        status = {
            'overall_status': 'healthy' if self._indexes_loaded else 'degraded',
            'initialized': self._initialized,
            'indexes_loaded': self._indexes_loaded,
            'services': {
                'embed_model': {
                    'available': Settings.embed_model is not None,
                    'status': 'healthy' if Settings.embed_model else 'not_initialized'
                },
                'vector_store': {
                    'available': self._vector_store is not None,
                    'index_loaded': (
                        self._vector_store.index is not None 
                        if self._vector_store else False
                    ),
                    'status': (
                        'healthy' if (
                            self._vector_store and self._vector_store.index
                        ) else 'degraded'
                    )
                },
                'bm25_manager': {
                    'available': self._bm25_manager is not None,
                    'index_loaded': (
                        self._bm25_manager.exists() 
                        if self._bm25_manager else False
                    ),
                    'status': (
                        'healthy' if (
                            self._bm25_manager and 
                            self._bm25_manager.exists()
                        ) else 'degraded'
                    )
                },
                'retriever': {
                    'available': self._retriever is not None,
                    'status': 'healthy' if self._retriever else 'not_initialized'
                },
                'postprocessors': {
                    'available': self._retriever is not None,
                    'status': 'integrated_in_retriever',
                    'note': 'Metadata boost and reranking via RetrieverService postprocessors'
                },
                'assembler': {
                    'available': self._assembler is not None,
                    'status': 'healthy' if self._assembler else 'not_initialized'
                }
            },
            'persist_dir': str(self.persist_dir),
            'config_path': self.config_path
        }
        
        overall_status: str = str(status['overall_status'])
        log_service_health("orchestrator", overall_status)
        
        return status

    def reload_indexes(self) -> bool:
        """
        Reload indexes from disk.
        
        Returns:
            True if indexes were successfully reloaded, False otherwise
        """
        logger.info("Reloading indexes")
        
        try:
            self._indexes_loaded = False
            self._retriever = None
            
            # Reload FAISS
            if self._vector_store and Settings.embed_model:
                index_loaded = self._vector_store.load_index(
                    embed_model=Settings.embed_model
                )
                if index_loaded and self._vector_store.index:
                    self._indexes_loaded = True
                    logger.info("FAISS index reloaded")
            
            # Reload BM25
            if self._bm25_manager:
                bm25_retriever = self._bm25_manager.load()
                if bm25_retriever:
                    logger.info("BM25 index reloaded")
                else:
                    logger.warning("BM25 index reload returned None")
            
            # Reinitialize retriever
            if self._indexes_loaded and self._vector_store and self._vector_store.index:
                self._retriever = RetrieverService(
                    index=self._vector_store.index,
                    config_path=self.config_path
                )
                # BM25 retriever is now managed by RetrieverService internally
                logger.info("Retriever reinitialized")
            
            logger.info("Index reload complete", success=self._indexes_loaded)
            return self._indexes_loaded
            
        except FileNotFoundError as e:
            logger.error("Index reload failed: index files not found", error=str(e))
            return False
        
        except (IOError, OSError) as e:
            logger.error("Index reload failed: file system error", error=str(e), exc_info=True)
            return False
        
        except Exception as e:
            logger.error("Index reload failed: unexpected error", error=str(e), exc_info=True)
            return False


# Global orchestrator instance
_orchestrator_instance: Optional[RAGOrchestrator] = None


def get_orchestrator(
    config_path: str = "config/settings.yaml",
    persist_dir: str = "data/index",
    auto_load: bool = True
) -> RAGOrchestrator:
    """
    Get or create the global RAG orchestrator instance.
    
    Args:
        config_path: Path to configuration file
        persist_dir: Directory containing persisted indexes
        auto_load: Whether to automatically load indexes
        
    Returns:
        The global RAG orchestrator instance
    """
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        _orchestrator_instance = RAGOrchestrator(
            config_path=config_path,
            persist_dir=persist_dir,
            auto_load=auto_load
        )
    
    return _orchestrator_instance
