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
        combined_results = self._combine_results(vector_results, bm25_results, self.hybrid_alpha)

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
        hybrid_alpha: float,
    ) -> List[Dict[str, Any]]:
        """
        Combine results from vector and BM25 search using hybrid alpha weighting.
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search  
            hybrid_alpha: Weight for vector search (0.0 = BM25 only, 1.0 = vector only)
        
        Returns:
            Combined and ranked results
        """
        # Create lookup dicts by node_id
        vector_dict = {doc['node_id']: doc for doc in vector_results}
        bm25_dict = {doc['node_id']: doc for doc in bm25_results}
        
        # Combine all unique documents
        all_node_ids = set(vector_dict.keys()) | set(bm25_dict.keys())
        combined = []
        
        for node_id in all_node_ids:
            vector_doc = vector_dict.get(node_id)
            bm25_doc = bm25_dict.get(node_id)
            
            if vector_doc and bm25_doc:
                # Document found in both results - combine scores
                vector_score = vector_doc.get('score', 0.0)
                bm25_score = bm25_doc.get('score', 0.0)
                
                # Normalize BM25 score (BM25 can be > 1, vector is typically 0-1)
                # Simple normalization: scale BM25 to 0-1 range based on max score
                max_bm25 = max((doc.get('score', 0.0) for doc in bm25_results), default=1.0)
                normalized_bm25 = bm25_score / max_bm25 if max_bm25 > 0 else 0.0
                
                # Combine scores using hybrid_alpha
                combined_score = hybrid_alpha * vector_score + (1 - hybrid_alpha) * normalized_bm25
                
                combined_doc = vector_doc.copy()
                combined_doc['score'] = combined_score
                combined_doc['vector_score'] = vector_score
                combined_doc['bm25_score'] = bm25_score
                combined.append(combined_doc)
                
            elif vector_doc:
                # Only in vector results
                combined_doc = vector_doc.copy()
                combined_doc['score'] = hybrid_alpha * vector_doc.get('score', 0.0)
                combined_doc['vector_score'] = vector_doc.get('score', 0.0)
                combined_doc['bm25_score'] = 0.0
                combined.append(combined_doc)
                
            elif bm25_doc:
                # Only in BM25 results
                max_bm25 = max((doc.get('score', 0.0) for doc in bm25_results), default=1.0)
                normalized_bm25 = bm25_doc.get('score', 0.0) / max_bm25 if max_bm25 > 0 else 0.0
                combined_doc = bm25_doc.copy()
                combined_doc['score'] = (1 - hybrid_alpha) * normalized_bm25
                combined_doc['vector_score'] = 0.0
                combined_doc['bm25_score'] = bm25_doc.get('score', 0.0)
                combined.append(combined_doc)
        
        # Sort by combined score
        combined.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        logger.info(f"Combined {len(vector_results)} vector and {len(bm25_results)} BM25 results into {len(combined)} documents using alpha={hybrid_alpha}")
        return combined

    def shutdown(self) -> None:
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=True)
