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

Note: Reranking is now handled by native LlamaIndex postprocessors within QueryEngine.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import time

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.workflow import (
    Context,
    Workflow,
    StartEvent,
    StopEvent,
    step,
    Event,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from src.rag.libs.bm25_manager import BM25IndexManager
from .services.assembler_service.service import ContextAssembler
from .exceptions import (
    ServiceInitializationError,
    RetrievalError,
)

from src.utils.logging_config import get_logger, log_service_health
from src.utils.metrics import get_metrics


# Custom Events for RAG Workflow
class RetrieveEvent(Event):
    """Event triggered after retrieval step."""
    nodes: List[Any]


class RerankEvent(Event):
    """Event triggered after reranking step."""
    nodes: List[Any]


class AssembleEvent(Event):
    """Event triggered after context assembly."""
    context_payload: Any


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
        
        # Initialize native LlamaIndex components
        self._vector_store: Optional[FaissVectorStore] = None
        self._storage_context: Optional[StorageContext] = None
        self._index: Optional[VectorStoreIndex] = None
        self._query_engine: Optional[Any] = None
        
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
            
            # Initialize vector store and storage context
            faiss_index_path = self.persist_dir / "faiss_index"
            
            # Try to load existing vector store, create empty if not exists
            try:
                self._vector_store = FaissVectorStore.from_persist_dir(str(faiss_index_path))
                self._storage_context = StorageContext.from_defaults(
                    vector_store=self._vector_store,
                    persist_dir=str(faiss_index_path)
                )
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self._vector_store,
                    storage_context=self._storage_context
                )
                self._indexes_loaded = True
                logger.info("FAISS index loaded successfully")
                log_service_health("vector_store", "loaded")
            except (ValueError, FileNotFoundError) as e:
                logger.warning(f"No existing FAISS index found: {e}, creating empty index")
                # Create empty FAISS vector store
                import faiss
                d = 384  # Dimension for all-MiniLM-L6-v2
                faiss_index = faiss.IndexFlatIP(d)  # Inner product for cosine similarity
                self._vector_store = FaissVectorStore(faiss_index=faiss_index)
                # Create empty storage context
                self._storage_context = StorageContext.from_defaults(
                    vector_store=self._vector_store
                )
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self._vector_store,
                    storage_context=self._storage_context
                )
                self._indexes_loaded = False
                log_service_health("vector_store", "created_empty")
            
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
            
            # Initialize query engine directly from index
            # Note: Reranking now integrated as postprocessors in QueryEngine
            if self._indexes_loaded and self._index:
                from src.utils.config_loader import get_retriever_config
                retriever_config = get_retriever_config(self.config_path)
                semantic_top_k = retriever_config.get('semantic_top_k', 20)
                
                self._query_engine = self._index.as_query_engine(
                    similarity_top_k=semantic_top_k,
                    response_mode="compact"
                )
                logger.info(
                    "QueryEngine initialized with native LlamaIndex components"
                )
                log_service_health("query_engine", "initialized")
            else:
                logger.error("Cannot initialize query engine: indexes not loaded")
                log_service_health("query_engine", "error", error="indexes_not_loaded")
            
            # Initialize context assembler
            self._assembler = ContextAssembler(config_path=self.config_path)
            logger.info("Context assembler initialized")
            log_service_health("assembler", "initialized")
            
            # Initialize RAG Workflow with QueryEngine
            if self._indexes_loaded and self._query_engine:
                self._workflow = RAGWorkflow(
                    query_engine=self._query_engine,
                    assembler=self._assembler
                )
                logger.info("RAG Workflow initialized with QueryEngine")
                log_service_health("workflow", "initialized")
            else:
                logger.warning("Cannot initialize Workflow: query engine not available")
                log_service_health("workflow", "not_initialized")
            
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
        
        if not self._query_engine:
            raise RuntimeError("Query engine not available")
        
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
            # Step 1: Retrieve documents using QueryEngine
            initial_k = top_k * 4  # Over-retrieve for better selection
            retrieve_start = time.time()
            
            # Use QueryEngine to get response with nodes
            response = self._query_engine.query(query)
            retrieved_nodes = response.source_nodes[:initial_k] if response.source_nodes else []
            
            retrieve_duration = time.time() - retrieve_start
            
            logger.info(
                "Retrieval complete",
                request_id=request_id,
                candidates=len(retrieved_nodes),
                duration_ms=retrieve_duration * 1000
            )
            
            # Convert nodes to document format
            retrieved_docs = []
            for node in retrieved_nodes:
                doc = {
                    'text': node.text,
                    'score': getattr(node, 'score', 0.0),
                    'metadata': node.metadata,
                    'node_id': getattr(node, 'node_id', getattr(node, 'id_', ''))
                }
                retrieved_docs.append(doc)
            
            # Step 2: Results already postprocessed by QueryEngine
            # (metadata boost + cross-encoder reranking via native LlamaIndex postprocessors)
            # Limit to final top_k
            final_docs = retrieved_docs[:top_k]
            
            logger.info(
                "Postprocessing complete (via QueryEngine)",
                request_id=request_id,
                results=len(final_docs)
            )
            
            # Step 3: Assemble context
            assemble_start = time.time()
            
            if self._assembler:
                context_payload = self._assembler.assemble_context(
                    query=query,
                    documents=final_docs,
                    token_budget=token_budget,
                    max_items=top_k
                )
            else:
                # Fallback: create simple context
                from .libs.schemas.mcp_schemas import MCPContextPayload, ContextItem
                context_items = []
                for doc in final_docs[:top_k]:
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
                    'results_postprocessed': len(final_docs),
                    'search_type': search_type,
                    'note': 'Query executed via native LlamaIndex QueryEngine'
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
        
    async def query_async(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        token_budget: int = 4000
    ) -> Dict[str, Any]:
        """
        Execute query pipeline using LlamaIndex Workflow (async).
        
        Args:
            query: The search query string
            top_k: Number of top results to return after reranking
            search_type: Type of search ("semantic", "keyword", "hybrid")
            token_budget: Maximum token budget for assembled context
            
        Returns:
            Dictionary containing query results
        """
        if not self._workflow:
            raise RuntimeError("RAG Workflow not initialized")
        
        start_time = time.time()
        request_id = f"query_{int(time.time() * 1000)}"
        
        logger.info(
            "Starting async query pipeline with Workflow",
            request_id=request_id,
            query=query,
            top_k=top_k,
            search_type=search_type
        )
        
        try:
            # Run workflow
            result = await self._workflow.run(
                query=query,
                top_k=top_k,
                search_type=search_type,
                token_budget=token_budget
            )
            
            context_payload = result
            
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
                    'total_duration_ms': round(total_duration * 1000, 2),
                    'search_type': search_type,
                    'note': 'Query executed via LlamaIndex Workflow'
                }
            }
            
            logger.info(
                "Async query pipeline complete",
                request_id=request_id,
                results=len(context_payload.context),
                tokens=context_payload.total_tokens_estimate(),
                duration_ms=total_duration * 1000
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Async query pipeline failed",
                request_id=request_id,
                query=query,
                error=str(e),
                duration_ms=duration * 1000,
                exc_info=True
            )
            raise RetrievalError(query, f"Workflow query failed: {e}") from e

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
        if not self._initialized or not self._query_engine:
            raise RuntimeError("RAG services not initialized")
        
        logger.info(
            "Executing document search",
            query=query,
            top_k=top_k,
            search_type=search_type
        )
        
        try:
            # Use QueryEngine for search
            response = self._query_engine.query(query)
            nodes = response.source_nodes[:top_k] if response.source_nodes else []
            
            results = []
            for node in nodes:
                result = {
                    'text': node.text,
                    'score': getattr(node, 'score', 0.0),
                    'metadata': node.metadata,
                    'node_id': getattr(node, 'node_id', getattr(node, 'id_', ''))
                }
                results.append(result)
            
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
                    'index_loaded': self._index is not None,
                    'status': (
                        'healthy' if self._index else 'degraded'
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
                    'available': self._query_engine is not None,
                    'status': 'healthy' if self._query_engine else 'not_initialized'
                },
                'postprocessors': {
                    'available': self._query_engine is not None,
                    'status': 'integrated_in_query_engine',
                    'note': 'Metadata boost and reranking via native LlamaIndex postprocessors'
                },
                'assembler': {
                    'available': self._assembler is not None,
                    'status': 'healthy' if self._assembler else 'not_initialized'
                },
                'workflow': {
                    'available': self._workflow is not None,
                    'status': 'healthy' if self._workflow else 'not_initialized'
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
            
            # Reload FAISS index
            faiss_index_path = self.persist_dir / "faiss_index"
            try:
                self._vector_store = FaissVectorStore.from_persist_dir(str(faiss_index_path))
                self._storage_context = StorageContext.from_defaults(
                    vector_store=self._vector_store,
                    persist_dir=str(faiss_index_path)
                )
                self._index = VectorStoreIndex.from_vector_store(
                    vector_store=self._vector_store,
                    storage_context=self._storage_context
                )
                self._indexes_loaded = True
                logger.info("FAISS index reloaded")
            except Exception as e:
                logger.error(f"Failed to reload FAISS index: {e}")
                return False
            
            # Reload BM25
            if self._bm25_manager:
                bm25_retriever = self._bm25_manager.load()
                if bm25_retriever:
                    logger.info("BM25 index reloaded")
                else:
                    logger.warning("BM25 index reload returned None")
            
            # Reinitialize query engine
            if self._indexes_loaded and self._index:
                from src.utils.config_loader import get_retriever_config
                retriever_config = get_retriever_config(self.config_path)
                semantic_top_k = retriever_config.get('semantic_top_k', 20)
                
                self._query_engine = self._index.as_query_engine(
                    similarity_top_k=semantic_top_k,
                    response_mode="compact"
                )
                logger.info("QueryEngine reinitialized")
            
            # Reinitialize workflow
            if self._indexes_loaded and self._query_engine:
                self._workflow = RAGWorkflow(
                    query_engine=self._query_engine,
                    assembler=self._assembler
                )
                logger.info("Workflow reinitialized with QueryEngine")
            else:
                logger.warning("Cannot reinitialize Workflow: query engine not available")
            
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


class RAGWorkflow(Workflow):
    """
    LlamaIndex Workflow for RAG query pipeline.
    
    Implements the RAG pipeline using native LlamaIndex QueryEngine:
    - retrieve: Get documents using QueryEngine
    - assemble: Assemble context using ContextAssembler
    """

    def __init__(self, query_engine: Any, assembler: Any, **kwargs):
        super().__init__(**kwargs)
        self.query_engine = query_engine
        self.assembler = assembler

    @step
    async def retrieve_and_assemble(
        self, ctx: Context, ev: StartEvent
    ) -> StopEvent:
        """Retrieve documents and assemble context using LlamaIndex QueryEngine."""
        query = ev.get("query")
        top_k = ev.get("top_k", 5)
        search_type = ev.get("search_type", "hybrid")
        token_budget = ev.get("token_budget", 4000)

        if not query:
            raise ValueError("Query is required")

        logger.info(f"Workflow step: query='{query}', top_k={top_k}, search_type={search_type}")

        try:
            # Use QueryEngine for retrieval (includes postprocessing)
            from llama_index.core import QueryBundle
            query_bundle = QueryBundle(query_str=query)
            
            # Retrieve nodes
            response = await self.query_engine.aretrieve(query_bundle)
            nodes = response[:top_k * 4]  # Over-retrieve for better selection

            # Convert nodes to document format
            documents = []
            for node in nodes:
                doc = {
                    'text': node.text,
                    'score': getattr(node, 'score', 0.0),
                    'metadata': node.metadata,
                    'node_id': getattr(node, 'node_id', getattr(node, 'id_', ''))
                }
                documents.append(doc)

            # Limit to final top_k
            final_docs = documents[:top_k]

            # Assemble context
            if self.assembler:
                context_payload = self.assembler.assemble_context(
                    query=query,
                    documents=final_docs,
                    token_budget=token_budget,
                    max_items=top_k
                )
            else:
                # Fallback
                from .libs.schemas.mcp_schemas import MCPContextPayload, ContextItem
                context_items = []
                for doc in final_docs:
                    item = ContextItem(
                        id=doc.get('node_id', ''),
                        text=doc.get('text', ''),
                        score=doc.get('score', 0.0),
                        meta=doc.get('metadata', {})
                    )
                    context_items.append(item)
                context_payload = MCPContextPayload(
                    query=query,
                    context=context_items,
                    schema_version="1.0",
                    token_budget=token_budget
                )

            return StopEvent(result=context_payload)

        except Exception as e:
            logger.error(f"Workflow step failed: {e}")
            raise


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
