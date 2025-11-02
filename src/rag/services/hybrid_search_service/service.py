"""
Hybrid Search Service for combining vector and keyword search results.
"""

from typing import List, Dict, Any, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.utils.logging_config import get_logger
from src.rag.services.retriever_service.service import RetrieverService
from src.rag.services.reranker_service.service import RerankerService

logger = get_logger(__name__)

class HybridSearchService:
    """
    Service for orchestrating hybrid search.
    """
    def __init__(
        self,
        retriever_service: RetrieverService,
        reranker_service: Optional[RerankerService] = None,
        hybrid_alpha: float = 0.5,
    ):
        self.retriever_service = retriever_service
        self.reranker_service = reranker_service
        self.hybrid_alpha = hybrid_alpha
        self._executor = ThreadPoolExecutor()

    async def asearch(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously perform a hybrid search.

        Args:
            query (str): The search query.
            top_k (int): The number of results to return.
            filters (Optional[Dict[str, Any]]): Metadata filters.

        Returns:
            List[Dict[str, Any]]: A list of ranked and filtered documents.
        """
        logger.info(f"Performing hybrid search for query: {query}")

        # Run vector and BM25 search in parallel
        loop = asyncio.get_running_loop()
        vector_task = loop.run_in_executor(
            self._executor, self.retriever_service.retrieve, query, top_k * 2, filters, "semantic"
        )
        bm25_task = loop.run_in_executor(
            self._executor, self.retriever_service.retrieve, query, top_k * 2, filters, "keyword"
        )

        vector_results, bm25_results = await asyncio.gather(vector_task, bm25_task)

        # Combine results
        combined_results = self._combine_results(vector_results, bm25_results)

        # Rerank if reranker is available
        if self.reranker_service:
            logger.info("Reranking combined results.")
            # Assuming reranker_service has a `rerank` method
            combined_results = self.reranker_service.rerank(query, combined_results)

        return combined_results[:top_k]

    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Combine results from vector and BM25 search using reciprocal rank fusion.
        """
        
        all_docs = {}
        for doc in vector_results:
            all_docs[doc['node_id']] = doc
        for doc in bm25_results:
            if doc['node_id'] not in all_docs:
                all_docs[doc['node_id']] = doc

        # Simple combination and de-duplication
        # A more sophisticated fusion method could be used here.
        combined = list(all_docs.values())
        
        # Sort by score as a default ranking before reranking
        combined.sort(key=lambda x: x.get('score', 0.0), reverse=True)

        logger.info(f"Combined {len(vector_results)} vector results and {len(bm25_results)} BM25 results into {len(combined)} unique results.")
        return combined

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=True)
