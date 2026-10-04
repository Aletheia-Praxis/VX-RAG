"""
RAG Orchestrator - High-level coordinator for RAG pipeline operations.

This module provides a clean, high-level interface for coordinating all RAG services.
It uses native LlamaIndex abstractions: RetrieverQueryEngine, ReciprocalRankFusionRetriever,
BM25Retriever, Qdrant vector store, and node post-processors.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import shutil
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import duckdb
import qdrant_client
from llama_index.core import (
    QueryBundle,
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.llms import MockLLM
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import BaseNode, NodeWithScore, TextNode
from llama_index.core.storage.docstore import BaseDocumentStore, SimpleDocumentStore
from llama_index.core.storage.docstore.keyval_docstore import KVDocumentStore
from llama_index.core.vector_stores.types import (
    VectorStoreQuery,
    VectorStoreQueryMode,
    VectorStoreQueryResult,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.storage.kvstore.duckdb import DuckDBKVStore
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.vector_stores.qdrant.utils import (
    BatchSparseEncoding,
    SparseEncoderCallable,
    fastembed_sparse_encoder,
)
from qdrant_client import models

from src.rag.exceptions import ServiceInitializationError
from src.rag.ingestion import DoclingPipeline
from src.rag.libs.postprocessors import (
    BGECrossEncoderReranker,
    MetadataBoostPostprocessor,
)
from src.rag.libs.schemas.mcp_schemas import ContextItem, MCPContextPayload
from src.rag.metadata import (
    compute_file_hash,
    is_duplicate_hash,
    sanitize_node_metadata,
)
from src.utils.config_loader import (
    get_docstore_config,
    get_embedding_config,
    get_qdrant_config,
    get_reranker_config,
)
from src.utils.logging_config import get_logger

logger = get_logger("rag_orchestrator")

_EMBEDDING_DIMENSION_BY_MODEL: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "BAAI/bge-base-en-v1.5": 768,
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-small-en": 384,
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "nomic-embed-text-v1": 768,
    "nomic-embed-text-v1.5": 768,
}
_FALLBACK_EMBEDDING_DIMENSION = 1024


def _to_qdrant_id(raw_id: str | int) -> str | int:
    """Ensure point ID conforms to Qdrant requirements (valid UUID or integer).

    Args:
        raw_id: Original string or integer ID.

    Returns:
        A valid UUID string or integer acceptable by Qdrant.
    """
    if isinstance(raw_id, int):
        return raw_id
    raw_str = str(raw_id)
    try:
        uuid.UUID(raw_str)
        return raw_str
    except (ValueError, AttributeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_str))


def _create_bge_m3_sparse_encoder(
    model_name: str = "BAAI/bge-m3",
) -> SparseEncoderCallable:
    """Create a sparse encoder for BAAI/bge-m3 generating lexical token weights.

    Args:
        model_name: Embedding or sparse model identifier.

    Returns:
        SparseEncoderCallable taking a list of text strings and returning (indices, values).
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except (ImportError, OSError, ValueError, RuntimeError) as e:
        logger.warning(
            f"Could not load tokenizer for {model_name}, falling back to fastembed BM25: {e}"
        )
        try:
            return fastembed_sparse_encoder("Qdrant/bm25")
        except (ImportError, OSError, ValueError, RuntimeError):
            def _dummy_encoder(texts: list[str]) -> BatchSparseEncoding:
                return [[] for _ in texts], [[] for _ in texts]

            return _dummy_encoder

    def compute_vectors(texts: list[str]) -> BatchSparseEncoding:
        """Compute sparse token frequency and importance vectors."""
        all_indices: list[list[int]] = []
        all_values: list[list[float]] = []

        if not texts:
            return all_indices, all_values

        try:
            encoded = tokenizer(
                texts,
                add_special_tokens=False,
                truncation=True,
                max_length=8192,
            )
            for input_ids in encoded["input_ids"]:
                freqs: dict[int, int] = {}
                for tid in input_ids:
                    freqs[tid] = freqs.get(tid, 0) + 1
                indices = list(freqs.keys())
                values = [float(freqs[tid]) for tid in indices]
                all_indices.append(indices)
                all_values.append(values)
        except (OSError, ValueError, TypeError, RuntimeError) as err:
            logger.warning(f"Error computing sparse vectors for batch: {err}")
            for _ in texts:
                all_indices.append([])
                all_values.append([])

        return all_indices, all_values

    return compute_vectors


def _get_sparse_encoder(model_name: str | None = None) -> SparseEncoderCallable:
    """Resolve sparse document/query encoder function based on model name.

    Args:
        model_name: Sparse model name or None for default.

    Returns:
        SparseEncoderCallable suitable for QdrantVectorStore.
    """
    selected_model = model_name or "BAAI/bge-m3"

    try:
        from fastembed.sparse.sparse_text_embedding import SparseTextEmbedding

        supported_models = [
            m["model"].lower() for m in SparseTextEmbedding.list_supported_models()
        ]
        if selected_model.lower() in supported_models:
            return fastembed_sparse_encoder(model_name=selected_model)
    except (ImportError, OSError, ValueError, AttributeError) as e:
        logger.debug(f"Fastembed sparse model check notice: {e}")

    return _create_bge_m3_sparse_encoder(selected_model)


class RobustQdrantVectorStore(QdrantVectorStore):
    """Qdrant vector store with robust ID translation and safe local lifecycle.

    Ensures arbitrary string node IDs (e.g. from tests or legacy sources) are
    deterministically converted to valid UUIDs for Qdrant point IDs, while
    preserving the original node_id in payloads for exact round-trip deserialization.
    """

    def __init__(
        self,
        collection_name: str,
        client: Any | None = None,
        enable_hybrid: bool = False,
        fastembed_sparse_model: str | None = None,
        sparse_doc_fn: Any | None = None,
        sparse_query_fn: Any | None = None,
        sparse_vector_name: str = "sparse",
        **kwargs: Any,
    ) -> None:
        """Initialize RobustQdrantVectorStore."""
        if enable_hybrid and (sparse_doc_fn is None or sparse_query_fn is None):
            encoder = _get_sparse_encoder(fastembed_sparse_model)
            sparse_doc_fn = sparse_doc_fn or encoder
            sparse_query_fn = sparse_query_fn or encoder

        super().__init__(
            collection_name=collection_name,
            client=client,
            enable_hybrid=enable_hybrid,
            fastembed_sparse_model=fastembed_sparse_model,
            sparse_doc_fn=sparse_doc_fn,
            sparse_query_fn=sparse_query_fn,
            sparse_vector_name=sparse_vector_name,
            **kwargs,
        )

    def _build_points(
        self, nodes: Sequence[BaseNode], sparse_vector_name: str
    ) -> tuple[list[Any], list[str]]:
        """Build Qdrant points with translated UUID-compatible point IDs."""
        points, ids = super()._build_points(list(nodes), sparse_vector_name)
        for p in points:
            p.id = _to_qdrant_id(p.id)
        return points, ids

    def delete_nodes(
        self,
        node_ids: Sequence[str] | None = None,
        filters: Any | None = None,
        shard_identifier: Any | None = None,
        **delete_kwargs: Any,
    ) -> None:
        """Delete nodes with mapped point IDs."""
        translated_ids = (
            [str(_to_qdrant_id(nid)) for nid in node_ids]
            if node_ids is not None
            else None
        )
        super().delete_nodes(
            node_ids=translated_ids,
            filters=filters,
            shard_identifier=shard_identifier,
            **delete_kwargs,
        )

    def _build_hybrid_rrf_prefetch(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> tuple[list[models.Prefetch], int, models.Filter | None, Any, models.SearchParams | None]:
        """Construct Prefetch requests and parameters for native Qdrant Fusion.RRF hybrid retrieval.

        Args:
            query: VectorStoreQuery specifying mode, embedding, and similarity limits.
            **kwargs: Extra query options such as qdrant_filters or search_params.

        Returns:
            Tuple of (prefetch_list, top_k, query_filter, shard_key, search_params).
        """
        query_embedding = cast(list[float], query.query_embedding)
        assert self._sparse_query_fn is not None
        assert query.query_str is not None
        sparse_indices, sparse_values = self._sparse_query_fn([query.query_str])

        qdrant_filters = kwargs.get("qdrant_filters")
        if qdrant_filters is not None:
            query_filter = cast(models.Filter | None, qdrant_filters)
        else:
            query_filter = cast(models.Filter | None, self._build_query_filter(query))

        shard_identifier = kwargs.get("shard_identifier")
        shard_key = (
            self._generate_shard_key_selector(shard_identifier)
            if shard_identifier is not None and hasattr(self, "_generate_shard_key_selector")
            else None
        )

        search_params = kwargs.get("search_params")
        if search_params is not None and isinstance(search_params, dict):
            search_params = models.SearchParams(**search_params)
        search_params = cast(models.SearchParams | None, search_params)

        top_k = query.hybrid_top_k or query.similarity_top_k
        candidate_k = kwargs.get("candidate_k") or query.similarity_top_k
        sparse_top_k = query.sparse_top_k or candidate_k

        sparse_idx_0: list[int] = sparse_indices[0] if sparse_indices else []
        sparse_val_0: list[float] = sparse_values[0] if sparse_values else []

        dense_prefetch = models.Prefetch(
            query=query_embedding,
            using=self.dense_vector_name or None,
            limit=candidate_k,
            filter=query_filter,
            params=search_params,
        )
        sparse_prefetch = models.Prefetch(
            query=models.SparseVector(
                indices=sparse_idx_0,
                values=sparse_val_0,
            ),
            using=self.sparse_vector_name,
            limit=sparse_top_k,
            filter=query_filter,
            params=search_params,
        )
        prefetch = [dense_prefetch, sparse_prefetch]
        return prefetch, top_k, query_filter, shard_key, search_params

    def query(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Query index for top k most similar nodes using native Qdrant RRF for hybrid search.

        Args:
            query: VectorStoreQuery containing embedding, mode, and parameters.
            **kwargs: Additional parameters passed to Qdrant query.

        Returns:
            VectorStoreQueryResult parsed from Qdrant scored points.
        """
        if (
            query.mode == VectorStoreQueryMode.HYBRID
            and self.enable_hybrid
            and self._sparse_query_fn is not None
            and query.query_str is not None
            and query.query_embedding is not None
        ):
            prefetch, top_k, query_filter, shard_key, search_params = (
                self._build_hybrid_rrf_prefetch(query, **kwargs)
            )
            response = self._client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                query_filter=query_filter,
                shard_key_selector=shard_key,
                search_params=search_params,
                with_payload=True,
            )
            return self.parse_to_query_result(response.points)

        return super().query(query, **kwargs)

    async def aquery(
        self,
        query: VectorStoreQuery,
        **kwargs: Any,
    ) -> VectorStoreQueryResult:
        """Asynchronously query the vector store using native Qdrant RRF for hybrid search.

        Falls back to thread executor if asynchronous client is not available.

        Args:
            query: VectorStoreQuery containing embedding, mode, and parameters.
            **kwargs: Additional parameters passed to Qdrant query.

        Returns:
            VectorStoreQueryResult parsed from Qdrant scored points.
        """
        if (
            query.mode == VectorStoreQueryMode.HYBRID
            and self.enable_hybrid
            and self._sparse_query_fn is not None
            and query.query_str is not None
            and query.query_embedding is not None
        ):
            if getattr(self, "_aclient", None) is not None:
                prefetch, top_k, query_filter, shard_key, search_params = (
                    self._build_hybrid_rrf_prefetch(query, **kwargs)
                )
                response = await self._aclient.query_points(
                    collection_name=self.collection_name,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_k,
                    query_filter=query_filter,
                    shard_key_selector=shard_key,
                    search_params=search_params,
                    with_payload=True,
                )
                return self.parse_to_query_result(response.points)
            return await asyncio.to_thread(self.query, query, **kwargs)

        if getattr(self, "_aclient", None) is not None:
            return await super().aquery(query, **kwargs)
        return await asyncio.to_thread(self.query, query, **kwargs)

    async def async_add(
        self,
        nodes: Sequence[BaseNode],
        shard_identifier: Any | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Asynchronously add nodes to vector store, falling back to thread executor if aclient is not set."""
        node_list = list(nodes)
        if getattr(self, "_aclient", None) is not None:
            return await super().async_add(node_list, shard_identifier=shard_identifier, **kwargs)
        return await asyncio.to_thread(self.add, node_list, shard_identifier=shard_identifier, **kwargs)

    async def adelete_nodes(
        self,
        node_ids: list[str] | None = None,
        filters: Any | None = None,
        shard_identifier: Any | None = None,
        **delete_kwargs: Any,
    ) -> None:
        """Asynchronously delete nodes, falling back to thread executor if aclient is not set."""
        if getattr(self, "_aclient", None) is not None:
            await super().adelete_nodes(
                node_ids=node_ids,
                filters=filters,
                shard_identifier=shard_identifier,
                **delete_kwargs,
            )
            return
        await asyncio.to_thread(
            self.delete_nodes,
            node_ids=node_ids,
            filters=filters,
            shard_identifier=shard_identifier,
            **delete_kwargs,
        )


def _get_embedding_dimension(
    embedding_model_name: str, embed_model: Any | None = None
) -> int:
    """Resolve embedding dimension dynamically from model or name mapping.

    Args:
        embedding_model_name: Name of the embedding model.
        embed_model: Optional instantiated embedding model instance.

    Returns:
        Embedding vector dimension.
    """
    if embed_model is not None:
        if hasattr(embed_model, "_model"):
            model = embed_model._model
            for method_name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
                if hasattr(model, method_name):
                    try:
                        dim = getattr(model, method_name)()
                        if isinstance(dim, int) and dim > 0:
                            return dim
                    except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                        logger.debug(f"Could not retrieve sentence embedding dimension from model: {e}")
        if hasattr(embed_model, "embed_dim") and isinstance(embed_model.embed_dim, int):
            return embed_model.embed_dim

    return _EMBEDDING_DIMENSION_BY_MODEL.get(
        embedding_model_name, _FALLBACK_EMBEDDING_DIMENSION
    )


DEFAULT_DOCSTORE_DB_NAME: str = "docstore.duckdb"
DEFAULT_DOCSTORE_TABLE_NAME: str = "docstore"
DEFAULT_INTERMEDIATE_NODES_DB_NAME: str = "nodes.duckdb"
DEFAULT_INTERMEDIATE_NODES_TABLE_NAME: str = "intermediate_nodes"
DEFAULT_RRF_K: int = 60


def _close_duckdb_kvstore(kvstore: DuckDBKVStore | None) -> None:
    """Safely close DuckDB connection handles to prevent Windows file locks.

    Args:
        kvstore: DuckDBKVStore instance to close.
    """
    if kvstore is None:
        return
    try:
        if (
            hasattr(kvstore, "_thread_local")
            and hasattr(kvstore._thread_local, "conn")
            and kvstore._thread_local.conn
        ):
            kvstore._thread_local.conn.close()
            kvstore._thread_local.conn = None
        if hasattr(kvstore, "_shared_conn") and kvstore._shared_conn:
            kvstore._shared_conn.close()
            kvstore._shared_conn = None
    except (duckdb.Error, OSError, RuntimeError) as e:
        logger.debug(f"Error closing DuckDBKVStore connection: {e}")


def _create_duckdb_kvstore(
    database_name: str,
    persist_dir: str,
    table_name: str,
    max_retries: int = 5,
    initial_backoff: float = 0.2,
) -> DuckDBKVStore:
    """Create a DuckDBKVStore with exponential backoff on file lock contention.

    On Windows, concurrent processes or background virus scanners can momentarily
    hold an exclusive file lock. This helper retries with jittered backoff to avoid
    prematurely failing operations.

    Args:
        database_name: Name of the DuckDB database file.
        persist_dir: Directory where the database file resides.
        table_name: Table name for KV pairs.
        max_retries: Maximum retry attempts on lock contention.
        initial_backoff: Initial sleep duration in seconds.

    Returns:
        Instantiated DuckDBKVStore instance.

    Raises:
        duckdb.IOException: If file lock contention cannot be resolved after retries.
    """
    backoff = initial_backoff
    for attempt in range(max_retries):
        try:
            return DuckDBKVStore(
                database_name=database_name,
                persist_dir=persist_dir,
                table_name=table_name,
            )
        except (duckdb.IOException, duckdb.Error) as e:
            if attempt == max_retries - 1:
                logger.error(
                    f"Failed to acquire DuckDBKVStore for {database_name} after "
                    f"{max_retries} attempts: {e}"
                )
                raise
            logger.debug(
                f"DuckDB lock contention on {database_name}, retrying in {backoff:.2f}s "
                f"(attempt {attempt + 1}/{max_retries}): {e}"
            )
            time.sleep(backoff)
            backoff *= 1.5

    return DuckDBKVStore(
        database_name=database_name,
        persist_dir=persist_dir,
        table_name=table_name,
    )


def persist_intermediate_nodes(
    nodes: Sequence[BaseNode],
    directory: Path | str,
    db_name: str = DEFAULT_INTERMEDIATE_NODES_DB_NAME,
    table_name: str = DEFAULT_INTERMEDIATE_NODES_TABLE_NAME,
    overwrite: bool = True,
) -> Path:
    """Persist intermediate ingestion nodes directly to DuckDB to avoid Defender locks.

    Args:
        nodes: Sequence of BaseNode instances to persist.
        directory: Directory where the DuckDB database will be stored.
        db_name: Database file name.
        table_name: DuckDB table name.
        overwrite: Whether to clear existing records in intermediate nodes table.

    Returns:
        Path to the saved DuckDB database file.
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    db_path = target_dir / db_name

    kv = _create_duckdb_kvstore(
        database_name=db_name,
        persist_dir=str(target_dir),
        table_name=table_name,
    )
    try:
        if overwrite:
            try:
                kv.client.execute(f"DELETE FROM {table_name};")  # nosec B608  # Table name is from trusted config
            except (duckdb.Error, OSError, RuntimeError) as e:
                logger.debug(f"Clear intermediate nodes table notice: {e}")
        if nodes:
            kv.put_all([(node.node_id, node.to_dict()) for node in nodes])
        try:
            kv.client.execute("CHECKPOINT;")
        except (duckdb.Error, OSError, RuntimeError) as e:
            logger.debug(f"Checkpoint notice: {e}")
    finally:
        _close_duckdb_kvstore(kv)

    return db_path


def load_intermediate_nodes(
    directory: Path | str,
    db_name: str = DEFAULT_INTERMEDIATE_NODES_DB_NAME,
    table_name: str = DEFAULT_INTERMEDIATE_NODES_TABLE_NAME,
    legacy_fallback: bool = True,
) -> list[BaseNode]:
    """Load intermediate nodes from DuckDB or fallback to legacy nodes.json.

    Args:
        directory: Directory containing intermediate node files.
        db_name: DuckDB database file name.
        table_name: DuckDB table name.
        legacy_fallback: Whether to attempt fallback to nodes.json if DuckDB is absent.

    Returns:
        List of loaded BaseNode instances.

    Raises:
        FileNotFoundError: If neither DuckDB nor legacy nodes file is found.
    """
    dir_path = Path(directory)
    db_path = dir_path / db_name
    legacy_json = dir_path / "nodes.json"

    if db_path.exists():
        kv = _create_duckdb_kvstore(
            database_name=db_name,
            persist_dir=str(dir_path),
            table_name=table_name,
        )
        try:
            records = kv.get_all()
            return [cast(BaseNode, TextNode.from_dict(rec)) for rec in records.values()]
        finally:
            _close_duckdb_kvstore(kv)

    if legacy_fallback and legacy_json.exists():
        logger.info(f"Loading intermediate nodes from legacy JSON: {legacy_json}")
        with open(legacy_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [cast(BaseNode, TextNode.from_dict(rec)) for rec in data]

    raise FileNotFoundError(
        f"Nodes file not found at {db_path} or {legacy_json}. Run 'ingest' first."
    )


def reciprocal_rank_fusion(
    results: Sequence[Sequence[NodeWithScore]],
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[NodeWithScore]:
    """Fuse multiple lists of retrieved nodes using Reciprocal Rank Fusion (RRF).

    Computes:
        RRF_Score = sum(1.0 / (k + rank))
    where rank is the 1-based index (1, 2, ..., N) of each node in each
    retriever's ranked list.

    Args:
        results: Ranked node candidate sequences from individual retrievers.
        k: Smoothing constant to penalize low-ranked candidates (default: 60).
        top_k: Optional maximum number of fused candidates to return.

    Returns:
        List of fused NodeWithScore objects ordered descending by RRF score.
    """
    if k < 0:
        raise ValueError(f"RRF smoothing constant k must be non-negative, got {k}")

    fused_scores: dict[str, float] = {}
    node_map: dict[str, NodeWithScore] = {}

    for node_list in results:
        sorted_nodes = sorted(
            node_list,
            key=lambda item: item.score if item.score is not None else float("-inf"),
            reverse=True,
        )
        seen_in_run: set[str] = set()
        rank = 1
        for item in sorted_nodes:
            node_id = item.node.node_id
            if node_id in seen_in_run:
                continue
            seen_in_run.add(node_id)
            if node_id not in node_map:
                node_map[node_id] = item
            fused_scores[node_id] = fused_scores.get(node_id, 0.0) + (1.0 / (k + rank))
            rank += 1

    sorted_node_ids = sorted(
        fused_scores.keys(),
        key=lambda nid: fused_scores[nid],
        reverse=True,
    )

    reranked_nodes: list[NodeWithScore] = [
        NodeWithScore(
            node=node_map[node_id].node,
            score=fused_scores[node_id],
        )
        for node_id in sorted_node_ids
    ]

    if top_k is not None:
        return reranked_nodes[: max(0, top_k)]
    return reranked_nodes


class ReciprocalRankFusionRetriever(BaseRetriever):  # type: ignore[misc]
    """Standalone Reciprocal Rank Fusion (RRF) retriever combining multiple retrievers.

    Fuses retrieved candidates from multiple retrievers (e.g. dense vector and sparse BM25)
    using the reciprocal rank score formula:
        RRF_Score = sum(1.0 / (k + rank))
    without any dependency on external LLM services or API keys.
    """

    def __init__(
        self,
        retrievers: Sequence[BaseRetriever],
        similarity_top_k: int = 10,
        k: int = DEFAULT_RRF_K,
        callback_manager: Any | None = None,
    ) -> None:
        """Initialize ReciprocalRankFusionRetriever.

        Args:
            retrievers: Sequence of BaseRetriever instances to query and fuse.
            similarity_top_k: Number of fused candidates to return.
            k: Reciprocal rank fusion constant (default: 60).
            callback_manager: Optional callback manager for tracing.
        """
        super().__init__(callback_manager=callback_manager)
        self._retrievers: list[BaseRetriever] = list(retrievers)
        self._similarity_top_k: int = similarity_top_k
        self._k: int = k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve candidates synchronously from all retrievers and fuse them.

        Args:
            query_bundle: QueryBundle containing the query string.

        Returns:
            List of fused NodeWithScore candidates.
        """
        all_results: list[list[NodeWithScore]] = [
            retriever.retrieve(query_bundle) for retriever in self._retrievers
        ]
        return reciprocal_rank_fusion(
            results=all_results,
            k=self._k,
            top_k=self._similarity_top_k,
        )

    async def _aretrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve candidates asynchronously in parallel from all retrievers and fuse them.

        Args:
            query_bundle: QueryBundle containing the query string.

        Returns:
            List of fused NodeWithScore candidates.
        """
        tasks = [retriever.aretrieve(query_bundle) for retriever in self._retrievers]
        all_results: list[list[NodeWithScore]] = await asyncio.gather(*tasks)
        return reciprocal_rank_fusion(
            results=all_results,
            k=self._k,
            top_k=self._similarity_top_k,
        )


class RAGOrchestrator:
    """
    High-level orchestrator for RAG pipeline operations using LlamaIndex native components.

    Coordinates document ingestion via Docling, embedding generation with BAAI/bge-m3,
    incremental Qdrant vector indexing, BM25 indexing, cross-encoder reranking
    with BAAI/bge-reranker-v2-m3, snapshot persistence, and query execution.
    """

    @staticmethod
    def reciprocal_rank_fusion(
        results: Sequence[Sequence[NodeWithScore]],
        k: int = DEFAULT_RRF_K,
        top_k: int | None = None,
    ) -> list[NodeWithScore]:
        """Fuse multiple lists of retrieved nodes using Reciprocal Rank Fusion (RRF).

        Args:
            results: Sequences of retrieved nodes from individual retrievers.
            k: Smoothing constant for reciprocal rank score (default: 60).
            top_k: Optional maximum number of fused nodes to return.

        Returns:
            List of fused NodeWithScore candidates sorted descending by RRF score.
        """
        return reciprocal_rank_fusion(results=results, k=k, top_k=top_k)

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        persist_dir: str = "data/index",
        auto_load: bool = True,
        docstore_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the RAG orchestrator.

        Args:
            config_path: Path to configuration YAML file.
            persist_dir: Directory path for persisting Qdrant and BM25 indexes.
            auto_load: Whether to automatically initialize services on instantiation.
            docstore_path: Optional explicit path to docstore DuckDB database.
        """
        self.config_path = config_path
        self.persist_dir = Path(persist_dir)
        self.docstore_path = Path(docstore_path) if docstore_path else None

        self._vector_store: QdrantVectorStore | None = None
        self._qdrant_client: qdrant_client.QdrantClient | None = None
        self._storage_context: StorageContext | None = None
        self._docstore: BaseDocumentStore | None = None
        self._kvstore: DuckDBKVStore | None = None
        self._index: VectorStoreIndex | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._reranker: BGECrossEncoderReranker | None = None

        self._initialized = False
        self._indexes_loaded = False

        # Explicitly configure MockLLM to eliminate external OpenAI API key requirements
        Settings.llm = MockLLM()

        if auto_load:
            self._initialize_services()

    def _checkpoint_docstore(self, kvstore: DuckDBKVStore | None = None) -> None:
        """Checkpoint DuckDB WAL without closing the connection.

        Args:
            kvstore: Optional DuckDBKVStore instance; defaults to self._kvstore.
        """
        target = kvstore or self._kvstore
        if target is not None:
            try:
                target.client.execute("CHECKPOINT;")
            except (duckdb.Error, OSError, RuntimeError) as e:
                logger.debug(f"DuckDB checkpoint notice: {e}")

    def _checkpoint_and_flush_docstore(self) -> None:
        """Checkpoint DuckDB WAL and flush connection to allow safe file access on Windows."""
        if self._kvstore is not None:
            self._checkpoint_docstore(self._kvstore)
            _close_duckdb_kvstore(self._kvstore)

    def close(self) -> None:
        """Close any open storage resources, Qdrant client, DuckDB connections, and release file locks."""
        self._checkpoint_and_flush_docstore()
        if self._qdrant_client is not None:
            try:
                self._qdrant_client.close()
            except (OSError, RuntimeError) as e:
                logger.debug(f"Error closing Qdrant client: {e}")
            self._qdrant_client = None

    def __del__(self) -> None:
        """Destructor to clean up resources."""
        try:
            self.close()
        except (duckdb.Error, OSError, RuntimeError) as e:
            logger.debug(f"Destructor cleanup error: {e}")

    def _initialize_services(self) -> None:
        """Initialize all RAG services, models, and load indexes."""
        if self._initialized:
            return

        try:
            # Configure MockLLM to avoid OpenAI API key checks in LlamaIndex response synthesizers
            Settings.llm = MockLLM()

            embed_config = get_embedding_config(self.config_path)
            embedding_model_name: str = str(embed_config.get("embedding_model", "BAAI/bge-m3"))
            manifest_path = self.persist_dir / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest_model = manifest_data.get("model_name")
                    if manifest_model and isinstance(manifest_model, str):
                        embedding_model_name = manifest_model
                except (OSError, ValueError, KeyError):
                    pass
            embedding_device: str = embed_config.get("embedding_device", "cpu")
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model_name,
                device=embedding_device,
                normalize=True,
                embed_batch_size=embed_config.get("embedding_batch_size", 10),
                trust_remote_code=embed_config.get("embedding_trust_remote_code", False),
            )

            reranker_config = get_reranker_config(self.config_path)
            reranker_model = reranker_config.get("model_name", "BAAI/bge-reranker-v2-m3")
            reranker_device = reranker_config.get("device", "cpu")
            reranker_top_k = int(reranker_config.get("top_k", 5))
            self._reranker = BGECrossEncoderReranker(
                model_name=reranker_model,
                top_n=reranker_top_k,
                device=reranker_device,
            )

            qdrant_index_path = self.persist_dir / "qdrant"
            self._vector_store, self._storage_context, self._index, self._indexes_loaded = (
                self._load_vector_store(qdrant_index_path, embedding_model_name)
            )

            bm25_index_path = self.persist_dir / "bm25_index"
            self._bm25_retriever = self._load_bm25_retriever(bm25_index_path)

            self._initialized = True

        except Exception as e:
            raise ServiceInitializationError("orchestrator", str(e)) from e

    def _get_docstore_and_kvstore(
        self,
        storage_path: Path | str,
    ) -> tuple[BaseDocumentStore, DuckDBKVStore | None]:
        """Initialize or load DuckDBKVStore and KVDocumentStore with legacy JSON migration.

        Args:
            storage_path: Path to index storage directory.

        Returns:
            Tuple of (docstore, kvstore).
        """
        docstore_config = get_docstore_config(self.config_path)
        store_type = str(docstore_config.get("store_type", "duckdb")).lower()
        db_name = str(docstore_config.get("db_name", DEFAULT_DOCSTORE_DB_NAME))
        table_name = str(docstore_config.get("table_name", DEFAULT_DOCSTORE_TABLE_NAME))

        target_dir = Path(storage_path) if str(storage_path) != ":memory:" else self.persist_dir

        if store_type == "simple":
            if (target_dir / "docstore.json").exists():
                return SimpleDocumentStore.from_persist_dir(str(target_dir)), None
            return SimpleDocumentStore(), None

        if self.docstore_path is not None:
            docstore_dir = self.docstore_path.parent
            db_name = self.docstore_path.name
        else:
            if (target_dir / db_name).exists() and not (self.persist_dir / db_name).exists():
                docstore_dir = target_dir
            else:
                docstore_dir = self.persist_dir

        docstore_dir.mkdir(parents=True, exist_ok=True)
        kvstore = _create_duckdb_kvstore(
            database_name=db_name,
            persist_dir=str(docstore_dir),
            table_name=table_name,
        )
        docstore = KVDocumentStore(kvstore=kvstore)

        # Efficient emptiness check without deserializing all documents
        is_empty = False
        try:
            res = kvstore.client.execute(
                f"SELECT 1 FROM {table_name} WHERE collection = 'docstore/data' LIMIT 1"  # nosec B608  # table_name is trusted
            ).fetchone()
            is_empty = res is None
        except (duckdb.Error, OSError, ValueError, RuntimeError):
            is_empty = len(docstore.docs) == 0

        # Legacy JSON migration if DuckDB docstore is currently empty
        if is_empty:
            legacy_candidates = [
                target_dir / "docstore.json",
                self.persist_dir / "docstore.json",
            ]
            for legacy_json in legacy_candidates:
                if legacy_json.exists():
                    try:
                        legacy_ds = SimpleDocumentStore.from_persist_dir(str(legacy_json.parent))
                        if legacy_ds.docs:
                            logger.info(
                                f"Migrating {len(legacy_ds.docs)} documents from legacy "
                                f"{legacy_json} to DuckDBKVStore"
                            )
                            docstore.add_documents(list(legacy_ds.docs.values()))
                            self._checkpoint_docstore(kvstore)
                            break
                    except (duckdb.Error, OSError, ValueError, RuntimeError) as e:
                        logger.warning(
                            f"Could not migrate legacy docstore from {legacy_json}: {e}"
                        )

        return docstore, kvstore

    def _load_vector_store(
        self,
        storage_path: Path | str,
        embedding_model_name: str,
    ) -> tuple[QdrantVectorStore | None, StorageContext | None, VectorStoreIndex | None, bool]:
        """Load an existing Qdrant vector store or initialize a new collection.

        Args:
            storage_path: Path to directory containing Qdrant files or ':memory:'.
            embedding_model_name: Name of the embedding model to resolve vector dimension.

        Returns:
            Tuple of (vector_store, storage_context, index, indexes_loaded).
        """
        qdrant_config = get_qdrant_config(self.config_path)
        collection_name = str(qdrant_config.get("collection_name", "vx_rag_collection"))
        cfg_distance = str(qdrant_config.get("distance", "Cosine"))
        cfg_path = str(qdrant_config.get("path", "./data/index/qdrant"))
        enable_hybrid = bool(qdrant_config.get("enable_hybrid", True))
        sparse_model = str(qdrant_config.get("sparse_model", "BAAI/bge-m3"))

        self._docstore, self._kvstore = self._get_docstore_and_kvstore(storage_path)

        is_memory = (
            str(storage_path) == ":memory:"
            or str(self.persist_dir) == ":memory:"
            or cfg_path == ":memory:"
        )

        if is_memory:
            if self._qdrant_client is None:
                self._qdrant_client = qdrant_client.QdrantClient(":memory:")
            logger.info("Initialized in-memory Qdrant client")
        else:
            qdrant_dir = Path(storage_path)
            qdrant_dir.mkdir(parents=True, exist_ok=True)
            if self._qdrant_client is None:
                self._qdrant_client = qdrant_client.QdrantClient(path=str(qdrant_dir))
            logger.info(f"Initialized local disk Qdrant client at {qdrant_dir}")

        embedding_dimension = _get_embedding_dimension(
            embedding_model_name, getattr(Settings, "embed_model", None)
        )

        distance_mapping = {
            "cosine": qdrant_client.http.models.Distance.COSINE,
            "euclid": qdrant_client.http.models.Distance.EUCLID,
            "dot": qdrant_client.http.models.Distance.DOT,
        }
        distance = distance_mapping.get(
            cfg_distance.lower(), qdrant_client.http.models.Distance.COSINE
        )

        existing_collections = [
            c.name for c in self._qdrant_client.get_collections().collections
        ]
        collection_exists = collection_name in existing_collections

        if not collection_exists:
            sparse_vectors_config = (
                {"sparse": qdrant_client.http.models.SparseVectorParams()}
                if enable_hybrid
                else None
            )
            self._qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=qdrant_client.http.models.VectorParams(
                    size=embedding_dimension,
                    distance=distance,
                ),
                sparse_vectors_config=sparse_vectors_config,
            )
            logger.info(
                f"Created Qdrant collection '{collection_name}' "
                f"(dim={embedding_dimension}, distance={cfg_distance}, enable_hybrid={enable_hybrid})"
            )
        elif enable_hybrid:
            try:
                col_info = self._qdrant_client.get_collection(collection_name)
                sparse_cfg = getattr(getattr(col_info, "config", None), "params", None)
                sparse_vectors = getattr(sparse_cfg, "sparse_vectors", None)
                if not sparse_vectors or "sparse" not in sparse_vectors:
                    self._qdrant_client.create_vector_name(
                        collection_name=collection_name,
                        vector_name="sparse",
                        vector_name_config=qdrant_client.http.models.SparseVectorNameConfig(
                            sparse=qdrant_client.http.models.SparseVectorConfig()
                        ),
                    )
                    logger.info(
                        f"Added sparse vector configuration to existing collection '{collection_name}'"
                    )
            except (OSError, ValueError, KeyError, RuntimeError, AttributeError) as e:
                logger.debug(f"Notice ensuring sparse vector in existing collection: {e}")

        vector_store = RobustQdrantVectorStore(
            client=self._qdrant_client,
            collection_name=collection_name,
            enable_hybrid=enable_hybrid,
            fastembed_sparse_model=sparse_model,
            sparse_vector_name="sparse",
        )

        storage_context = StorageContext.from_defaults(
            docstore=self._docstore,
            vector_store=vector_store,
        )

        index = VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
        )

        point_count = 0
        try:
            point_count = self._qdrant_client.count(collection_name).count
        except (OSError, ValueError, RuntimeError) as e:
            logger.debug(f"Could not get point count for {collection_name}: {e}")

        indexes_loaded = point_count > 0
        if indexes_loaded:
            logger.info(
                f"Loaded existing Qdrant vector store collection '{collection_name}' with {point_count} points"
            )
        else:
            logger.info(
                f"Initialized empty Qdrant vector store collection '{collection_name}'"
            )

        return vector_store, storage_context, index, indexes_loaded

    def _load_bm25_retriever(self, bm25_index_path: Path) -> BM25Retriever | None:
        """
        Load persisted BM25 retriever from disk.

        Args:
            bm25_index_path: Path to persisted BM25 directory.

        Returns:
            Instantiated BM25Retriever or None if not found or corrupted.
        """
        if not bm25_index_path.exists():
            return None
        try:
            return BM25Retriever.from_persist_dir(str(bm25_index_path))
        except (OSError, ValueError, KeyError) as e:
            logger.warning(f"Failed to load BM25 retriever from {bm25_index_path}: {e}")
            return None

    def get_indexed_file_hashes(self) -> set[str]:
        """
        Retrieve the set of unique SHA-256 file hashes currently indexed.

        Inspects the active docstore and manifest.json if present.

        Returns:
            Set of lowercase 64-character hexadecimal SHA-256 digests.
        """
        hashes: set[str] = set()
        if self._kvstore is not None:
            try:
                query = (
                    f"SELECT DISTINCT json_extract_string(value, '$.__data__.metadata.file_hash') "  # nosec B608  # trusted column
                    f"FROM {self._kvstore.table_name} "
                    "WHERE collection = 'docstore/data'"  # nosec B608  # trusted table name
                )
                rows = self._kvstore.client.execute(query).fetchall()
                for row in rows:
                    if row and row[0]:
                        hashes.add(str(row[0]).lower())
            except (duckdb.Error, OSError, ValueError, RuntimeError) as e:
                logger.debug(f"Direct DuckDB hash query failed, falling back to docstore.docs: {e}")
                if self._index is not None and hasattr(self._index, "docstore") and self._index.docstore is not None:
                    for doc in self._index.docstore.docs.values():
                        h = doc.metadata.get("file_hash")
                        if h:
                            hashes.add(str(h).lower())
        elif self._index is not None and hasattr(self._index, "docstore") and self._index.docstore is not None:
            for doc in self._index.docstore.docs.values():
                h = doc.metadata.get("file_hash")
                if h:
                    hashes.add(str(h).lower())

        manifest_path = self.persist_dir / "manifest.json"
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                for h in data.get("indexed_file_hashes", []):
                    hashes.add(str(h).lower())
            except (OSError, ValueError, json.JSONDecodeError) as e:
                logger.debug(f"Could not load indexed hashes from manifest: {e}")

        return hashes

    def ingest_documents(
        self,
        directory: Path | str,
        recursive: bool = True,
        supported_extensions: tuple[str, ...] = (
            ".pdf",
            ".txt",
            ".md",
            ".markdown",
            ".png",
            ".jpg",
            ".jpeg",
        ),
    ) -> list[TextNode]:
        """
        Ingest documents from a directory using DoclingPipeline with duplicate detection.

        Checks SHA-256 file hashes against already indexed documents to skip duplicates
        before parsing. Parses non-duplicate files through DoclingPipeline.ingest_file()
        and sanitizes node metadata to strictly conform to the 5-field schema.

        Args:
            directory: Directory path containing source documents.
            recursive: Whether to scan directory recursively.
            supported_extensions: File extensions to process.

        Returns:
            List of sanitized TextNodes ready for indexing.
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        if not dir_path.is_dir():
            raise ValueError(f"Path is not a directory: {dir_path}")

        candidates = list(dir_path.rglob("*")) if recursive else list(dir_path.glob("*"))
        files = [p for p in candidates if p.is_file() and p.suffix.lower() in supported_extensions]
        files.sort()

        indexed_hashes = self.get_indexed_file_hashes()
        pipeline = DoclingPipeline(config_path=self.config_path)

        all_nodes: list[TextNode] = []
        session_hashes: set[str] = set()

        for file_path in files:
            try:
                file_hash = compute_file_hash(file_path)
                if is_duplicate_hash(file_hash, indexed_hashes) or file_hash in session_hashes:
                    logger.info(f"Skipping duplicate file: {file_path.name} (hash: {file_hash[:8]}...)")
                    continue

                session_hashes.add(file_hash)
                nodes = pipeline.ingest_file(file_path)
                for node in nodes:
                    sanitize_node_metadata(node)
                    all_nodes.append(node)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning(f"Error ingesting file {file_path}: {e}")
                continue

        logger.info(f"Ingested {len(all_nodes)} nodes from {len(files)} files in {dir_path}")
        return all_nodes

    def ingest(self, data_dir: str) -> list[BaseNode]:
        """
        Backward-compatible document ingestion alias delegating to ingest_documents.

        Args:
            data_dir: Path string to directory containing document files.

        Returns:
            List of BaseNode instances conforming to the strict metadata schema.
        """
        nodes = self.ingest_documents(directory=Path(data_dir))
        return list(nodes)

    def _generate_manifest(self, directory: Path) -> dict[str, Any]:
        """
        Generate a manifest dictionary containing index metadata and SHA-256 file checksums.

        Args:
            directory: Directory containing index files to checksum.

        Returns:
            Manifest dictionary conforming to Tech Spec §5.2.
        """
        embed_config = get_embedding_config(self.config_path)
        model_name = str(embed_config.get("embedding_model", "BAAI/bge-m3"))
        dim = _get_embedding_dimension(model_name, getattr(Settings, "embed_model", None))

        total_nodes = 0
        indexed_hashes: set[str] = set()
        if self._kvstore is not None:
            try:
                count_res = self._kvstore.client.execute(
                    f"SELECT COUNT(*) FROM {self._kvstore.table_name} WHERE collection = 'docstore/data'"  # nosec B608  # trusted table name
                ).fetchone()
                if count_res:
                    total_nodes = int(count_res[0])
                hash_res = self._kvstore.client.execute(
                    f"SELECT DISTINCT json_extract_string(value, '$.__data__.metadata.file_hash') "  # nosec B608  # trusted column
                    f"FROM {self._kvstore.table_name} WHERE collection = 'docstore/data'"
                ).fetchall()
                for r in hash_res:
                    if r and r[0]:
                        indexed_hashes.add(str(r[0]).lower())
            except (duckdb.Error, OSError, ValueError, RuntimeError) as e:
                logger.debug(f"Direct DuckDB manifest query failed: {e}")
            self._checkpoint_and_flush_docstore()
        elif self._index is not None and hasattr(self._index, "docstore") and self._index.docstore is not None:
            docs = self._index.docstore.docs
            total_nodes = len(docs)
            for d in docs.values():
                h = d.metadata.get("file_hash")
                if h:
                    indexed_hashes.add(str(h).lower())
            self._checkpoint_and_flush_docstore()

        files_map: dict[str, str] = {}
        if directory.exists():
            for p in directory.rglob("*"):
                if (
                    p.is_file()
                    and p.name != "manifest.json"
                    and not p.name.endswith(".lock")
                    and p.name != ".lock"
                ):
                    rel_path = p.relative_to(directory).as_posix()
                    try:
                        file_bytes = p.read_bytes()
                        file_sha256 = hashlib.sha256(file_bytes).hexdigest()
                        files_map[rel_path] = file_sha256
                        # Also record top-level filename for direct lookup
                        files_map[p.name] = file_sha256
                    except (PermissionError, OSError) as err:
                        logger.warning(f"Could not compute checksum for {p}: {err}")

        manifest: dict[str, Any] = {
            "version": "1.0",
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "model_name": model_name,
            "embedding_dimension": dim,
            "total_nodes": total_nodes,
            "indexed_file_hashes": sorted(indexed_hashes),
            "files": files_map,
        }
        return manifest

    def create_snapshot(self, snapshot_dir: Path | str | None = None) -> Path:
        """
        Create a versioned snapshot directory containing index artifacts and manifest.json.

        Args:
            snapshot_dir: Optional custom snapshot destination directory. If None,
                creates a versioned directory under 'data/snapshots/snapshot_<timestamp>/'.

        Returns:
            Path to the created snapshot directory.
        """
        if snapshot_dir is None:
            ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
            target_dir = Path("data/snapshots") / f"snapshot_{ts}"
        else:
            target_dir = Path(snapshot_dir)

        target_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint and flush DuckDB docstore before copying to prevent Win32 file sharing lock
        self._checkpoint_and_flush_docstore()

        if self.persist_dir.exists():
            for item in self.persist_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(
                        item,
                        target_dir / item.name,
                        dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.lock", ".lock"),
                    )
                elif (
                    item.is_file()
                    and item.name != "manifest.json"
                    and not item.name.endswith(".lock")
                    and item.name != ".lock"
                ):
                    shutil.copy2(item, target_dir / item.name)

        if self.docstore_path and self.docstore_path.exists():
            try:
                if not self.docstore_path.is_relative_to(self.persist_dir):
                    shutil.copy2(self.docstore_path, target_dir / self.docstore_path.name)
            except (ValueError, TypeError):
                shutil.copy2(self.docstore_path, target_dir / self.docstore_path.name)

        manifest_data = self._generate_manifest(target_dir)
        manifest_file = target_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        # Also write active manifest in persist_dir
        if self.persist_dir.exists():
            active_manifest = self.persist_dir / "manifest.json"
            active_manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        logger.info(f"Created index snapshot at {target_dir}")
        return target_dir

    def index_nodes(self, nodes: Sequence[BaseNode]) -> None:
        """
        Incrementally append nodes to Qdrant and BM25 indexes with duplicate prevention.

        Filters out nodes from files that are already indexed or nodes with matching node_ids.
        Appends new vector embeddings to the Qdrant collection without rebuilding.
        Updates BM25 using cumulative nodes from docstore so prior documents are preserved.
        Persists updated indexes and writes manifest.json.

        Args:
            nodes: Sequence of BaseNode instances to append.
        """
        if not nodes:
            return

        if not self._initialized:
            self._initialize_services()

        storage_context = self._storage_context or (
            getattr(self._index, "storage_context", None) if self._index else None
        )
        if self._index is None or storage_context is None:
            raise RuntimeError("RAG index is not initialized")

        # Duplicate filtering tier 2: check against existing docstore hashes and node IDs
        already_indexed_hashes = self.get_indexed_file_hashes()
        existing_node_ids: set[str] = set()
        if self._kvstore is not None:
            try:
                id_rows = self._kvstore.client.execute(
                    f"SELECT key FROM {self._kvstore.table_name} WHERE collection = 'docstore/data'"  # nosec B608  # trusted table name
                ).fetchall()
                existing_node_ids = {r[0] for r in id_rows if r and r[0]}
            except (duckdb.Error, OSError, ValueError, RuntimeError):
                existing_node_ids = set(self._index.docstore.docs.keys())
        elif self._index is not None and hasattr(self._index, "docstore") and self._index.docstore is not None:
            existing_node_ids = set(self._index.docstore.docs.keys())

        nodes_to_insert: list[BaseNode] = []
        for node in nodes:
            node_hash = node.metadata.get("file_hash")
            if node_hash and str(node_hash).lower() in already_indexed_hashes:
                continue
            if node.node_id in existing_node_ids:
                continue
            try:
                sanitize_node_metadata(node)
            except ValueError:
                pass
            nodes_to_insert.append(node)

        if not nodes_to_insert:
            logger.info("No new nodes to index (all were duplicates or already indexed)")
            return

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Incrementally append nodes to Qdrant index and docstore
        self._index.insert_nodes(nodes_to_insert)
        if self._docstore is not None:
            self._docstore.add_documents(nodes_to_insert)
        elif self._index.docstore is not None:
            self._index.docstore.add_documents(nodes_to_insert)

        self._index.storage_context.persist(persist_dir=str(self.persist_dir))
        self._indexes_loaded = True

        # Checkpoint and flush DuckDB docstore before manifest calculation on Windows
        self._checkpoint_and_flush_docstore()

        # Generate and save manifest.json
        manifest_data = self._generate_manifest(self.persist_dir)
        (self.persist_dir / "manifest.json").write_text(
            json.dumps(manifest_data, indent=2), encoding="utf-8"
        )
        active_docstore = self._docstore or (self._index.docstore if self._index else None)
        total_count = len(nodes_to_insert)
        if active_docstore is not None and active_docstore.docs:
            total_count = len(active_docstore.docs)

        logger.info(
            f"Successfully indexed {len(nodes_to_insert)} nodes. Total nodes: {total_count}"
        )

    async def query_async(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        token_budget: int = 4000,
    ) -> MCPContextPayload:
        """Execute an asynchronous query against the indexed documents.

        Args:
            query: The user query string.
            top_k: Number of context items to return.
            search_type: Search modality: 'hybrid', 'semantic', or 'keyword'.
            token_budget: Upper limit for token consumption in the payload.

        Returns:
            MCPContextPayload containing retrieved and reranked context items.

        Raises:
            RuntimeError: If services are not initialized or indexes are not loaded.
        """
        if not self._initialized:
            raise RuntimeError("RAG services not initialized")
        if not self._indexes_loaded or not self._index:
            raise RuntimeError("Indexes not loaded")

        total_docs = top_k
        if self._kvstore is not None:
            try:
                res = self._kvstore.client.execute(
                    f"SELECT COUNT(*) FROM {self._kvstore.table_name} WHERE collection = 'docstore/data'"  # nosec B608  # trusted table name
                ).fetchone()
                if res and res[0] > 0:
                    total_docs = int(res[0])
            except (duckdb.Error, OSError, ValueError, RuntimeError):
                total_docs = top_k
        elif self._index and hasattr(self._index, "docstore"):
            try:
                total_docs = len(self._index.docstore.docs)
            except (TypeError, AttributeError):
                total_docs = top_k
        candidate_k = max(1, min(total_docs, max(top_k * 2, 20)))

        retriever: BaseRetriever
        if search_type == "hybrid" and self._bm25_retriever is not None:
            self._bm25_retriever.similarity_top_k = candidate_k
            vector_retriever = self._index.as_retriever(similarity_top_k=candidate_k)
            retriever = ReciprocalRankFusionRetriever(
                retrievers=[vector_retriever, self._bm25_retriever],
                similarity_top_k=candidate_k,
                k=DEFAULT_RRF_K,
            )
        elif search_type == "keyword" and self._bm25_retriever is not None:
            self._bm25_retriever.similarity_top_k = candidate_k
            retriever = self._bm25_retriever
        else:
            is_hybrid = getattr(self._vector_store, "enable_hybrid", False)
            query_mode = VectorStoreQueryMode.DEFAULT
            if search_type == "hybrid" and is_hybrid:
                query_mode = VectorStoreQueryMode.HYBRID
            elif search_type == "keyword" and is_hybrid:
                query_mode = VectorStoreQueryMode.SPARSE
            retriever = self._index.as_retriever(
                similarity_top_k=candidate_k,
                vector_store_query_mode=query_mode,
            )

        node_postprocessors: list[BaseNodePostprocessor] = []
        if self._reranker is not None:
            node_postprocessors.append(self._reranker)
        node_postprocessors.extend([
            MetadataBoostPostprocessor(config_path=self.config_path),
            SimilarityPostprocessor(similarity_cutoff=0.6),
        ])

        synthesizer = get_response_synthesizer(response_mode=ResponseMode.NO_TEXT)
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=synthesizer,
            node_postprocessors=node_postprocessors,
        )

        response = await query_engine.aquery(query)

        context_items: list[ContextItem] = []
        for node in response.source_nodes[:top_k]:
            context_items.append(
                ContextItem(
                    id=node.node.node_id,
                    text=node.node.get_content(),
                    score=max(0.0, min(1.0, node.score or 0.0)),
                    meta=node.node.metadata,
                )
            )

        return MCPContextPayload(
            schema_version="1.0",
            context=context_items,
            query=query,
            token_budget=token_budget,
            provenance={
                "selected_count": len(context_items),
                "total_tokens": sum(len(c.text) // 4 for c in context_items),
                "selection_method": "query_engine",
            },
        )

    def query(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        token_budget: int = 4000,
    ) -> MCPContextPayload:
        """Execute a synchronous query against the indexed documents.

        Args:
            query: The user query string.
            top_k: Number of context items to return.
            search_type: Search modality: 'hybrid', 'semantic', or 'keyword'.
            token_budget: Upper limit for token consumption in the payload.

        Returns:
            MCPContextPayload containing retrieved context items.
        """
        return asyncio.run(self.query_async(query, top_k, search_type, token_budget))

    def search_documents(
        self,
        query: str,
        top_k: int = 10,
        search_type: str = "semantic",
    ) -> list[dict[str, Any]]:
        """Search documents and return structured dictionaries with full metadata.

        Args:
            query: Search query string.
            top_k: Maximum number of results to return.
            search_type: Search modality: 'semantic', 'hybrid', or 'keyword'.

        Returns:
            List of dictionaries containing document text, score, id, and metadata.

        Raises:
            RuntimeError: If services are not initialized or indexes are not loaded.
        """
        if not self._initialized or not self._index:
            raise RuntimeError("RAG services not initialized or indexes not loaded")

        total_docs = top_k
        if self._kvstore is not None:
            try:
                res = self._kvstore.client.execute(
                    f"SELECT COUNT(*) FROM {self._kvstore.table_name} WHERE collection = 'docstore/data'"  # nosec B608  # trusted table name
                ).fetchone()
                if res and res[0] > 0:
                    total_docs = int(res[0])
            except (duckdb.Error, OSError, ValueError, RuntimeError):
                total_docs = top_k
        elif self._index and hasattr(self._index, "docstore"):
            try:
                total_docs = len(self._index.docstore.docs)
            except (TypeError, AttributeError):
                total_docs = top_k
        candidate_k = max(1, min(total_docs, max(top_k * 3, 20)))
        retriever: BaseRetriever
        if search_type == "hybrid" and self._bm25_retriever is not None:
            vector_retriever = self._index.as_retriever(similarity_top_k=candidate_k)
            self._bm25_retriever.similarity_top_k = candidate_k
            retriever = ReciprocalRankFusionRetriever(
                retrievers=[vector_retriever, self._bm25_retriever],
                similarity_top_k=candidate_k,
                k=DEFAULT_RRF_K,
            )
        elif search_type == "keyword" and self._bm25_retriever is not None:
            self._bm25_retriever.similarity_top_k = candidate_k
            retriever = self._bm25_retriever
        else:
            is_hybrid = getattr(self._vector_store, "enable_hybrid", False)
            query_mode = VectorStoreQueryMode.DEFAULT
            if search_type == "hybrid" and is_hybrid:
                query_mode = VectorStoreQueryMode.HYBRID
            elif search_type == "keyword" and is_hybrid:
                query_mode = VectorStoreQueryMode.SPARSE
            retriever = self._index.as_retriever(
                similarity_top_k=candidate_k,
                vector_store_query_mode=query_mode,
            )

        query_bundle = QueryBundle(query)
        nodes = retriever.retrieve(query_bundle)

        if self._reranker is not None:
            nodes = self._reranker.postprocess_nodes(nodes, query_bundle)

        postprocessor = MetadataBoostPostprocessor(config_path=self.config_path)
        nodes = postprocessor.postprocess_nodes(nodes, query_bundle)

        final_nodes = nodes[:top_k]

        results: list[dict[str, Any]] = []
        for node in final_nodes:
            results.append(
                {
                    "id": node.node.node_id,
                    "node_id": node.node.node_id,
                    "text": node.node.get_content(),
                    "score": float(node.score) if node.score is not None else 0.0,
                    "metadata": node.node.metadata,
                }
            )
        return results

    def get_health_status(self) -> dict[str, Any]:
        """Return comprehensive health status of RAG services and vector store.

        Returns:
            Dictionary containing health status, initialization state, and service details.
        """
        has_docstore = self._docstore is not None or (
            self._index is not None and hasattr(self._index, "docstore") and self._index.docstore is not None
        )
        return {
            "overall_status": "healthy" if self._indexes_loaded else "degraded",
            "initialized": self._initialized,
            "indexes_loaded": self._indexes_loaded,
            "services": {
                "vector_store": {
                    "available": self._vector_store is not None,
                    "status": "healthy" if self._index else "degraded",
                },
                "docstore": {
                    "available": has_docstore,
                    "type": "duckdb" if self._kvstore is not None else "simple",
                    "status": "healthy" if has_docstore else "degraded",
                },
                "bm25_retriever": {
                    "available": self._bm25_retriever is not None,
                    "status": "healthy" if self._bm25_retriever else "not_loaded",
                },
                "reranker": {
                    "available": self._reranker is not None,
                    "status": "healthy" if self._reranker else "not_configured",
                },
            },
            "persist_dir": str(self.persist_dir),
        }


_orchestrator_instance: RAGOrchestrator | None = None


def get_orchestrator(
    config_path: str = "config/settings.yaml",
    persist_dir: str = "data/index",
    docstore_path: str | Path | None = None,
) -> RAGOrchestrator:
    """Retrieve or initialize the singleton RAGOrchestrator instance.

    Args:
        config_path: Path to the configuration YAML file.
        persist_dir: Directory where index files are persisted.
        docstore_path: Optional explicit path to docstore DuckDB database.

    Returns:
        The singleton RAGOrchestrator instance.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = RAGOrchestrator(
            config_path=config_path,
            persist_dir=persist_dir,
            docstore_path=docstore_path,
        )
    return _orchestrator_instance


def reset_orchestrator() -> None:
    """Reset and close the singleton RAGOrchestrator instance."""
    global _orchestrator_instance
    if _orchestrator_instance is not None:
        _orchestrator_instance.close()
        _orchestrator_instance = None

