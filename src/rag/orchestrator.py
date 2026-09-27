"""
RAG Orchestrator - High-level coordinator for RAG pipeline operations.

This module provides a clean, high-level interface for coordinating all RAG services.
It uses native LlamaIndex abstractions: RetrieverQueryEngine, QueryFusionRetriever,
BM25Retriever, FAISS HNSW vector store, and node post-processors.
"""

from __future__ import annotations

import asyncio
import datetime
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, cast

import faiss
from llama_index.core import (
    QueryBundle,
    Settings,
    StorageContext,
    VectorStoreIndex,
    get_response_synthesizer,
)
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers.fusion_retriever import (
    FUSION_MODES,
    QueryFusionRetriever,
)
from llama_index.core.schema import BaseNode, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.faiss import FaissVectorStore

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
    get_bm25_config,
    get_embedding_config,
    get_faiss_config,
    get_reranker_config,
)
from src.utils.logging_config import get_logger

logger = get_logger("rag_orchestrator")

_EMBEDDING_DIMENSION_BY_MODEL: dict[str, int] = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-small-en": 384,
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "all-mpnet-base-v2": 768,
    "nomic-embed-text-v1": 768,
    "nomic-embed-text-v1.5": 768,
}
_FALLBACK_EMBEDDING_DIMENSION = 384



class RAGOrchestrator:
    """
    High-level orchestrator for RAG pipeline operations using LlamaIndex native components.

    Coordinates document ingestion via Docling, embedding generation with BAAI/bge-small-en-v1.5,
    incremental FAISS HNSW vector indexing, BM25 indexing, cross-encoder reranking
    with BAAI/bge-reranker-base, snapshot persistence, and query execution.
    """

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        persist_dir: str = "data/index",
        auto_load: bool = True,
    ) -> None:
        """
        Initialize the RAG orchestrator.

        Args:
            config_path: Path to configuration YAML file.
            persist_dir: Directory path for persisting FAISS and BM25 indexes.
            auto_load: Whether to automatically initialize services on instantiation.
        """
        self.config_path = config_path
        self.persist_dir = Path(persist_dir)

        self._vector_store: FaissVectorStore | None = None
        self._storage_context: StorageContext | None = None
        self._index: VectorStoreIndex | None = None
        self._bm25_retriever: BM25Retriever | None = None
        self._reranker: BGECrossEncoderReranker | None = None

        self._initialized = False
        self._indexes_loaded = False

        if auto_load:
            self._initialize_services()

    def _initialize_services(self) -> None:
        """Initialize all RAG services, models, and load indexes."""
        if self._initialized:
            return

        try:
            embed_config = get_embedding_config(self.config_path)
            embedding_model_name: str = embed_config.get("embedding_model", "BAAI/bge-small-en-v1.5")
            embedding_device: str = embed_config.get("embedding_device", "cpu")
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=embedding_model_name,
                device=embedding_device,
                normalize=True,
                embed_batch_size=embed_config.get("embedding_batch_size", 10),
                trust_remote_code=embed_config.get("embedding_trust_remote_code", False),
            )

            reranker_config = get_reranker_config(self.config_path)
            reranker_model = reranker_config.get("model_name", "BAAI/bge-reranker-base")
            reranker_device = reranker_config.get("device", "cpu")
            reranker_top_k = int(reranker_config.get("top_k", 5))
            self._reranker = BGECrossEncoderReranker(
                model_name=reranker_model,
                top_n=reranker_top_k,
                device=reranker_device,
            )

            faiss_index_path = self.persist_dir / "faiss_index"
            self._vector_store, self._storage_context, self._index, self._indexes_loaded = (
                self._load_vector_store(faiss_index_path, embedding_model_name)
            )

            bm25_index_path = self.persist_dir / "bm25_index"
            self._bm25_retriever = self._load_bm25_retriever(bm25_index_path)

            self._initialized = True

        except Exception as e:
            raise ServiceInitializationError("orchestrator", str(e)) from e

    def _load_vector_store(
        self,
        faiss_index_path: Path,
        embedding_model_name: str,
    ) -> tuple[FaissVectorStore | None, StorageContext | None, VectorStoreIndex | None, bool]:
        """
        Load an existing FAISS vector store or initialize an empty HNSW index.

        Args:
            faiss_index_path: Path to directory containing persisted FAISS index files.
            embedding_model_name: Name of the embedding model to resolve vector dimension.

        Returns:
            Tuple of (vector_store, storage_context, index, indexes_loaded).
        """
        faiss_config = get_faiss_config(self.config_path)
        hnsw_m = int(faiss_config.get("hnsw_m", 32))

        try:
            from llama_index.core import load_index_from_storage

            if not (faiss_index_path / "default__vector_store.json").exists():
                raise FileNotFoundError(f"FAISS index not found at {faiss_index_path}")

            vector_store = FaissVectorStore.from_persist_dir(str(faiss_index_path))
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(faiss_index_path),
            )
            index = cast(VectorStoreIndex, load_index_from_storage(storage_context=storage_context))
            logger.info(f"Loaded existing FAISS vector store from {faiss_index_path}")
            return vector_store, storage_context, index, True

        except (ValueError, FileNotFoundError, OSError, RuntimeError) as e:
            logger.info(f"Initializing new empty FAISS HNSW vector store ({e})")
            embedding_dimension = _EMBEDDING_DIMENSION_BY_MODEL.get(
                embedding_model_name, _FALLBACK_EMBEDDING_DIMENSION
            )
            faiss_index = faiss.IndexHNSWFlat(
                embedding_dimension, hnsw_m, faiss.METRIC_INNER_PRODUCT
            )
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
            )
            return vector_store, storage_context, index, False

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
        if self._index is not None and hasattr(self._index, "docstore") and self._index.docstore is not None:
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
        model_name = str(embed_config.get("embedding_model", "BAAI/bge-small-en-v1.5"))
        dim = _EMBEDDING_DIMENSION_BY_MODEL.get(model_name, _FALLBACK_EMBEDDING_DIMENSION)

        total_nodes = 0
        indexed_hashes: set[str] = set()
        if self._index is not None and hasattr(self._index, "docstore") and self._index.docstore is not None:
            docs = self._index.docstore.docs
            total_nodes = len(docs)
            for d in docs.values():
                h = d.metadata.get("file_hash")
                if h:
                    indexed_hashes.add(str(h).lower())

        files_map: dict[str, str] = {}
        if directory.exists():
            for p in directory.rglob("*"):
                if p.is_file() and p.name != "manifest.json":
                    rel_path = p.relative_to(directory).as_posix()
                    file_bytes = p.read_bytes()
                    file_sha256 = hashlib.sha256(file_bytes).hexdigest()
                    files_map[rel_path] = file_sha256
                    # Also record top-level filename for direct lookup
                    files_map[p.name] = file_sha256

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

    def create_snapshot(self, snapshot_dir: Path | None = None) -> Path:
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

        if self.persist_dir.exists():
            for item in self.persist_dir.iterdir():
                if item.is_dir():
                    shutil.copytree(item, target_dir / item.name, dirs_exist_ok=True)
                elif item.is_file() and item.name != "manifest.json":
                    shutil.copy2(item, target_dir / item.name)

        manifest_data = self._generate_manifest(target_dir)
        manifest_file = target_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        # Also write active manifest in persist_dir
        if self.persist_dir.exists():
            active_manifest = self.persist_dir / "manifest.json"
            active_manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

        logger.info(f"Created index snapshot at {target_dir}")
        return target_dir

    def index_nodes(self, nodes: list[BaseNode]) -> None:
        """
        Incrementally append nodes to FAISS HNSW and BM25 indexes with duplicate prevention.

        Filters out nodes from files that are already indexed or nodes with matching node_ids.
        Appends new vector embeddings to the FAISS HNSW graph without rebuilding.
        Updates BM25 using cumulative nodes from docstore so prior documents are preserved.
        Persists updated indexes and writes manifest.json.

        Args:
            nodes: List of BaseNode instances to append.
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
        faiss_index_path = self.persist_dir / "faiss_index"
        bm25_index_path = self.persist_dir / "bm25_index"

        # Incrementally append nodes to FAISS HNSW index and docstore
        self._index.insert_nodes(nodes_to_insert)
        self._index.storage_context.persist(persist_dir=str(faiss_index_path))

        # Re-index BM25 using cumulative nodes from docstore to ensure no docs are lost
        cumulative_nodes = list(self._index.docstore.docs.values())
        bm25_config = get_bm25_config(self.config_path)
        similarity_top_k = int(bm25_config.get("similarity_top_k", 20))

        self._bm25_retriever = BM25Retriever.from_defaults(
            nodes=cumulative_nodes,
            similarity_top_k=similarity_top_k,
            verbose=False,
        )
        bm25_index_path.mkdir(parents=True, exist_ok=True)
        self._bm25_retriever.persist(str(bm25_index_path))
        self._indexes_loaded = True

        # Generate and save manifest.json
        manifest_data = self._generate_manifest(self.persist_dir)
        (self.persist_dir / "manifest.json").write_text(
            json.dumps(manifest_data, indent=2), encoding="utf-8"
        )
        logger.info(
            f"Successfully indexed {len(nodes_to_insert)} nodes. Total nodes: {len(cumulative_nodes)}"
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
        if self._index and hasattr(self._index, "docstore"):
            try:
                total_docs = len(self._index.docstore.docs)
            except (TypeError, AttributeError):
                total_docs = top_k
        candidate_k = max(1, min(total_docs, max(top_k * 2, 20)))
        vector_retriever = self._index.as_retriever(similarity_top_k=candidate_k)

        retriever: BaseRetriever
        if search_type == "hybrid" and self._bm25_retriever:
            retriever = QueryFusionRetriever(
                [vector_retriever, self._bm25_retriever],
                similarity_top_k=candidate_k,
                num_queries=1,
                mode=FUSION_MODES.RECIPROCAL_RANK,
            )
        elif search_type == "keyword" and self._bm25_retriever:
            self._bm25_retriever.similarity_top_k = candidate_k
            retriever = self._bm25_retriever
        else:
            retriever = vector_retriever

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
        if self._index and hasattr(self._index, "docstore"):
            try:
                total_docs = len(self._index.docstore.docs)
            except (TypeError, AttributeError):
                total_docs = top_k
        candidate_k = max(1, min(total_docs, max(top_k * 3, 20)))
        vector_retriever = self._index.as_retriever(similarity_top_k=candidate_k)
        retriever: BaseRetriever
        if search_type == "hybrid" and self._bm25_retriever:
            retriever = QueryFusionRetriever(
                [vector_retriever, self._bm25_retriever],
                similarity_top_k=candidate_k,
                num_queries=1,
                mode=FUSION_MODES.RECIPROCAL_RANK,
            )
        elif search_type == "keyword" and self._bm25_retriever:
            self._bm25_retriever.similarity_top_k = candidate_k
            retriever = self._bm25_retriever
        else:
            retriever = vector_retriever

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
        return {
            "overall_status": "healthy" if self._indexes_loaded else "degraded",
            "initialized": self._initialized,
            "indexes_loaded": self._indexes_loaded,
            "services": {
                "vector_store": {
                    "available": self._vector_store is not None,
                    "status": "healthy" if self._index else "degraded",
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
) -> RAGOrchestrator:
    """Retrieve or initialize the singleton RAGOrchestrator instance.

    Args:
        config_path: Path to the configuration YAML file.
        persist_dir: Directory where index files are persisted.

    Returns:
        The singleton RAGOrchestrator instance.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = RAGOrchestrator(
            config_path=config_path,
            persist_dir=persist_dir,
        )
    return _orchestrator_instance
