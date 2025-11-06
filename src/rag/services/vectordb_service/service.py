"""
Vector Database Service implementation.

Provides classes for vector storage operations.
"""

from typing import List, Dict, Any, Optional, cast
import logging
from pathlib import Path
import hashlib
import json
import datetime

from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

from src.utils.config_loader import get_data_directories, get_faiss_config

logger = logging.getLogger(__name__)

class VectorStoreClient:
    """
    Client for vector database operations.
    
    Supports FAISS vector storage with incremental updates, snapshotting, and integrity verification.
    Implements the incremental update strategy from the technical standard, allowing documents to be
    added to existing indexes without full rebuilds.
    
    Key features:
    - Incremental document addition (add_documents_incremental)
    - Snapshot creation with manifest.json and checksums
    - Integrity verification of snapshots
    - Automatic backups on index updates
    """
    
    def __init__(self, store_type: str = "faiss", config: Optional[Dict[str, Any]] = None):
        self.store_type = store_type
        self.config = config or {}
        
        # Load index_dir from config, or use provided value
        if 'index_dir' in self.config:
            index_dir = self.config['index_dir']
        else:
            dirs = get_data_directories()
            index_dir = dirs['index_dir']
        
        assert index_dir is not None, "index_dir must be set"
        self.index_dir = Path(index_dir)
        
        self.index: Optional[VectorStoreIndex] = None
        self.faiss_index: Optional[faiss.Index] = None  # Direct FAISS index access
        
        # Create index directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        if store_type == "faiss":
            # Will be initialized when building index
            pass
        else:
            raise ValueError(f"Unsupported store type: {store_type}")
    
    def build_index(self, documents: List[Any], embed_model: Any, transformations: Optional[List[Any]] = None) -> VectorStoreIndex:
        """
        Build FAISS vector index from documents.
        
        Args:
            documents: List of documents to index
            embed_model: Embedding model to use
            transformations: Optional list of transformations (e.g., node parsers)
            
        Returns:
            The created VectorStoreIndex
        """
        if not documents:
            raise ValueError("No documents provided for indexing")

        try:
            logger.info("Initializing FAISS vector store")
            # Get embedding dimension by creating a test embedding
            test_embedding = embed_model.get_text_embedding("test")
            d = len(test_embedding)
            logger.info(f"Embedding dimension: {d}")
            
            # Load FAISS configuration
            faiss_config = get_faiss_config()
            hnsw_m = faiss_config['hnsw_m']
            metric = faiss_config['metric']
            
            """
            Initialize FAISS HNSW index
            The technical standard requires HNSW for its high-speed, high-recall retrieval capabilities.
            The hnsw_m parameter represents the number of neighbors for each node in the graph.
            METRIC_INNER_PRODUCT is used for cosine similarity,
            as required by the technical standard for normalized embeddings.
            """
            metric_type = faiss.METRIC_INNER_PRODUCT if metric == 'inner_product' else faiss.METRIC_L2
            faiss_index = faiss.IndexHNSWFlat(d, hnsw_m, metric_type)
            vector_store = FaissVectorStore(faiss_index=faiss_index)

            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            logger.info(f"Building index with {len(documents)} documents")
            # Build index with documents
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model,
                transformations=transformations,
                show_progress=True
            )

            self.index = index
            self.faiss_index = faiss_index  # Store direct access to FAISS index
            logger.info(f"Successfully built index with {len(documents)} documents")
            return index

        except Exception as e:
            logger.error(f"Failed to build index: {e}")
            raise
    
    def add_documents_incremental(self, documents: List[Any], embed_model: Any) -> bool:
        """
        Add new documents to existing FAISS index incrementally.
        
        This method appends new documents to the existing index without rebuilding it,
        following the incremental update strategy specified in the technical standard.
        
        Args:
            documents: List of new documents to add
            embed_model: Embedding model to use for new documents
            
        Returns:
            True if documents were added successfully, False otherwise
        """
        if self.index is None:
            logger.error("No existing index found. Use build_index first or load existing index.")
            return False
        
        if not documents:
            logger.warning("No documents provided for incremental addition")
            return True  # Not an error, just nothing to do
        
        try:
            logger.info(f"Adding {len(documents)} documents incrementally to existing index")
            
            # Insert documents into existing index
            # LlamaIndex handles embedding generation and FAISS index updates internally
            for doc in documents:
                self.index.insert(doc)
            
            logger.info(f"Successfully added {len(documents)} documents to index")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents incrementally: {e}")
            return False
    
    def save_index(self, create_backup: bool = True, embed_model_info: Optional[Dict[str, Any]] = None,
                  chunking_params: Optional[Dict[str, Any]] = None) -> None:
        """
        Save the index to disk.
        
        Args:
            create_backup: Whether to create a backup snapshot before saving
            embed_model_info: Information about embedding model (for backup manifest)
            chunking_params: Chunking parameters (for backup manifest)
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        try:
            # Create backup if requested (following technical standard)
            if create_backup:
                backup_name = f"backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                self.create_snapshot(backup_name, embed_model_info, chunking_params)
            
            # Custom JSON encoder to handle non-serializable objects
            import json
            from llama_index.core.schema import RelatedNodeInfo
            
            class CustomJSONEncoder(json.JSONEncoder):
                def default(self, obj: Any) -> Any:
                    if isinstance(obj, RelatedNodeInfo):
                        # Convert RelatedNodeInfo to a string representation to avoid unhashable issues
                        return f"RelatedNodeInfo(node_id={obj.node_id})"
                    # Handle other non-serializable objects
                    try:
                        return str(obj)
                    except Exception:
                        return f"<{type(obj).__name__} object>"
            
            # Temporarily patch json.dumps to use our custom encoder
            original_dumps = json.dumps
            json.dumps = lambda obj, **kwargs: original_dumps(obj, cls=CustomJSONEncoder, **kwargs)
            
            try:
                self.index.storage_context.persist(persist_dir=str(self.index_dir))
                logger.info(f"Index saved to {self.index_dir}")
            finally:
                # Restore original json.dumps
                json.dumps = original_dumps
                
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            raise
    
    def load_index(self, embed_model: Optional[Any] = None) -> bool:
        """
        Load the index from disk.
        
        Args:
            embed_model: Embedding model to use for the index
            
        Returns:
            True if loading succeeded, False otherwise
        """
        if not self.index_dir.exists():
            logger.error(f"Index directory does not exist: {self.index_dir}")
            return False

        try:
            # Try to load using LlamaIndex's standard method with embed_model
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(self.index_dir))
                self.index = cast(VectorStoreIndex, load_index_from_storage(storage_context, embed_model=embed_model))
                logger.info(f"Index loaded from {self.index_dir} using standard method")
                return True
            except Exception as e:
                logger.warning(f"Standard loading failed: {e}, trying alternative method")
            
            # Alternative method: Load FAISS index directly and reconstruct
            faiss_index_path = self.index_dir / "default__vector_store.json"
            docstore_path = self.index_dir / "docstore.json"
            index_store_path = self.index_dir / "index_store.json"
            
            if not faiss_index_path.exists():
                logger.error(f"FAISS index file not found: {faiss_index_path}")
                return False
            
            # Load FAISS index directly
            import faiss
            faiss_index = faiss.read_index(str(faiss_index_path))
            
            # Create vector store with loaded index
            vector_store = FaissVectorStore(faiss_index=faiss_index)
            
            # Try to load docstore and index_store if they exist
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Load docstore if exists
            if docstore_path.exists():
                try:
                    with open(docstore_path, 'r', encoding='utf-8') as f:
                        docstore_data = json.load(f)
                    # Reconstruct docstore from data
                    from llama_index.core.storage.docstore import SimpleDocumentStore
                    storage_context.docstore = SimpleDocumentStore.from_dict(docstore_data)
                except Exception as e:
                    logger.warning(f"Could not load docstore: {e}")
            
            # Load index_store if exists  
            if index_store_path.exists():
                try:
                    with open(index_store_path, 'r', encoding='utf-8') as f:
                        index_store_data = json.load(f)
                    # Reconstruct index_store from data
                    from llama_index.core.storage.index_store import SimpleIndexStore
                    storage_context.index_store = SimpleIndexStore.from_dict(index_store_data)
                except Exception as e:
                    logger.warning(f"Could not load index_store: {e}")
            
            # Create index from storage context with embed_model
            self.index = cast(VectorStoreIndex, load_index_from_storage(storage_context, embed_model=embed_model))
            self.faiss_index = faiss_index
            
            logger.info(f"Index loaded from {self.index_dir} using alternative method")
            return True

        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return False
    
    def store_vectors(self, vectors: List[List[float]], metadata: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Store vectors with optional metadata using LlamaIndex."""
        if self.index is None:
            logger.error("No index available. Build or load index first.")
            return False
        
        try:
            # Convert vectors and metadata to LlamaIndex documents
            from llama_index.core.schema import Document
            
            documents = []
            for i, vector in enumerate(vectors):
                # Create a document with vector as embedding
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                doc = Document(
                    text="",  # Empty text since we have pre-computed vectors
                    metadata=doc_metadata,
                    embedding=vector
                )
                documents.append(doc)
            
            # Insert documents into existing index
            for doc in documents:
                self.index.insert(doc)
            
            logger.info(f"Successfully stored {len(vectors)} vectors")
            return True
        except Exception as e:
            logger.error(f"Failed to store vectors: {e}")
            return False
    
    def search_vectors(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors using LlamaIndex retriever."""
        if self.index is None:
            if not self.load_index():
                return []
        
        assert self.index is not None  # nosec B101
        try:
            # Create retriever
            # Since we have a vector, we need to create a query with embedding
            # For pure vector search, we'd need to use vector_store.query directly
            # But LlamaIndex retriever expects text queries
            
            # Alternative: Use vector_store.query with VectorStoreQuery
            from llama_index.core.vector_stores import VectorStoreQuery
            
            query = VectorStoreQuery(
                query_embedding=query_vector,
                similarity_top_k=top_k
            )
            results = self.index.vector_store.query(query)
            
            # Convert results to expected format
            formatted_results = []
            if results.nodes:
                for node in results.nodes:
                    formatted_results.append({
                        'id': node.id_,
                        'score': getattr(node, 'score', 0.0),  # May not be available
                        'metadata': node.metadata,
                        'text': node.get_content()
                    })
            return formatted_results
                
        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by IDs. Note: FAISS doesn't support deletion, rebuild index instead."""
        logger.warning("delete_vectors not supported for FAISS. Consider rebuilding the index without deleted items.")
        return False
    
    def create_snapshot(self, snapshot_name: Optional[str] = None, 
                       embed_model_info: Optional[Dict[str, Any]] = None,
                       chunking_params: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a snapshot of the current index with manifest.json.
        
        Following the technical standard, creates a timestamped snapshot with:
        - Index files persistence
        - manifest.json with metadata and checksums
        
        Args:
            snapshot_name: Optional custom snapshot name
            embed_model_info: Information about the embedding model used
            chunking_params: Parameters used for text chunking
            
        Returns:
            True if snapshot was created successfully
        """
        if self.index is None:
            logger.error("No index to snapshot. Build index first.")
            return False
        
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            snapshot_name = snapshot_name or f"snapshot_{timestamp}"
            snapshot_dir = self.index_dir.parent / "snapshots" / snapshot_name
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            
            # Custom JSON encoder to handle non-serializable objects
            import json
            from llama_index.core.schema import RelatedNodeInfo
            
            class CustomJSONEncoder(json.JSONEncoder):
                def default(self, obj: Any) -> Any:
                    if isinstance(obj, RelatedNodeInfo):
                        # Convert RelatedNodeInfo to a string representation to avoid unhashable issues
                        return f"RelatedNodeInfo(node_id={obj.node_id})"
                    # Handle other non-serializable objects
                    try:
                        return str(obj)
                    except Exception:
                        return f"<{type(obj).__name__} object>"
            
            # Temporarily patch json.dumps to use our custom encoder
            original_dumps = json.dumps
            json.dumps = lambda obj, **kwargs: original_dumps(obj, cls=CustomJSONEncoder, **kwargs)
            
            try:
                # Persist the index
                self.index.storage_context.persist(persist_dir=str(snapshot_dir))
                
                # Create manifest.json with metadata and checksums
                manifest = self._create_manifest(snapshot_dir, embed_model_info, chunking_params)
                
                # Save manifest
                manifest_path = snapshot_dir / "manifest.json"
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(manifest, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Snapshot created: {snapshot_dir} with manifest.json")
                return True
            finally:
                # Restore original json.dumps
                json.dumps = original_dumps
                
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}")
            return False
    
    def _create_manifest(self, snapshot_dir: Path, embed_model_info: Optional[Dict[str, Any]] = None,
                        chunking_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create manifest.json with metadata and checksums for the snapshot.
        
        Args:
            snapshot_dir: Directory containing the snapshot files
            embed_model_info: Information about embedding model
            chunking_params: Parameters used for chunking
            
        Returns:
            Manifest dictionary
        """
        manifest = {
            "snapshot_info": {
                "created_at": datetime.datetime.now().isoformat(),
                "version": "1.0"
            },
            "embedding_model": embed_model_info or {},
            "chunking_parameters": chunking_params or {},
            "files": {}
        }
        
        # Calculate checksums for all files in snapshot
        for file_path in snapshot_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "manifest.json":
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    
                    # Calculate both MD5 and SHA256
                    md5_hash = hashlib.md5(file_content, usedforsecurity=False).hexdigest()
                    sha256_hash = hashlib.sha256(file_content).hexdigest()
                    
                    # Store relative path from snapshot directory
                    rel_path = file_path.relative_to(snapshot_dir)
                    manifest["files"][str(rel_path)] = {
                        "md5": md5_hash,
                        "sha256": sha256_hash,
                        "size": len(file_content)
                    }
                except Exception as e:
                    logger.warning(f"Could not calculate checksum for {file_path}: {e}")
        
        return manifest
    
    def list_snapshots(self) -> List[str]:
        """List available snapshots."""
        try:
            snapshots_dir = self.index_dir.parent / "snapshots"
            if not snapshots_dir.exists():
                return []
            return [d.name for d in snapshots_dir.iterdir() if d.is_dir()]
        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}")
            return []
    
    def load_snapshot(self, snapshot_name: str) -> bool:
        """Load a snapshot as the current index."""
        try:
            snapshots_dir = self.index_dir.parent / "snapshots"
            snapshot_dir = snapshots_dir / snapshot_name
            if not snapshot_dir.exists():
                logger.error(f"Snapshot {snapshot_name} does not exist")
                return False
            
            storage_context = StorageContext.from_defaults(persist_dir=str(snapshot_dir))
            self.index = cast(VectorStoreIndex, load_index_from_storage(storage_context))
            logger.info(f"Snapshot {snapshot_name} loaded")
            return True
        except Exception as e:
            logger.error(f"Failed to load snapshot {snapshot_name}: {e}")
            return False
    
    def verify_snapshot_integrity(self, snapshot_name: str) -> Dict[str, Any]:
        """
        Verify the integrity of a snapshot by checking file checksums against manifest.json.
        
        Args:
            snapshot_name: Name of the snapshot to verify
            
        Returns:
            Dictionary with verification results
        """
        try:
            snapshots_dir = self.index_dir.parent / "snapshots"
            snapshot_dir = snapshots_dir / snapshot_name
            manifest_path = snapshot_dir / "manifest.json"
            
            if not snapshot_dir.exists():
                return {"valid": False, "error": f"Snapshot {snapshot_name} does not exist"}
            
            if not manifest_path.exists():
                return {"valid": False, "error": "manifest.json not found in snapshot"}
            
            # Load manifest
            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)
            
            verification_results = {
                "valid": True,
                "total_files": len(manifest.get("files", {})),
                "verified_files": 0,
                "failed_files": [],
                "missing_files": [],
                "snapshot_info": manifest.get("snapshot_info", {})
            }
            
            # Verify each file in manifest
            for file_path_str, expected_hashes in manifest.get("files", {}).items():
                file_path = snapshot_dir / file_path_str
                
                if not file_path.exists():
                    verification_results["missing_files"].append(file_path_str)
                    verification_results["valid"] = False
                    continue
                
                try:
                    with open(file_path, 'rb') as f:
                        file_content = f.read()
                    
                    # Check MD5
                    actual_md5 = hashlib.md5(file_content, usedforsecurity=False).hexdigest()
                    expected_md5 = expected_hashes.get("md5")
                    
                    # Check SHA256
                    actual_sha256 = hashlib.sha256(file_content).hexdigest()
                    expected_sha256 = expected_hashes.get("sha256")
                    
                    if actual_md5 != expected_md5 or actual_sha256 != expected_sha256:
                        verification_results["failed_files"].append({
                            "file": file_path_str,
                            "expected_md5": expected_md5,
                            "actual_md5": actual_md5,
                            "expected_sha256": expected_sha256,
                            "actual_sha256": actual_sha256
                        })
                        verification_results["valid"] = False
                    else:
                        verification_results["verified_files"] += 1
                        
                except Exception as e:
                    verification_results["failed_files"].append({
                        "file": file_path_str,
                        "error": str(e)
                    })
                    verification_results["valid"] = False
            
            if verification_results["valid"]:
                logger.info(f"Snapshot {snapshot_name} integrity verified successfully")
            else:
                logger.error(f"Snapshot {snapshot_name} integrity verification failed")
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Failed to verify snapshot {snapshot_name} integrity: {e}")
            return {"valid": False, "error": str(e)}
    
    def replicate_index(self, target_dir: str) -> bool:
        """Replicate index to another directory."""
        if self.index is None:
            logger.error("No index to replicate. Build or load index first.")
            return False
        
        try:
            target_path = Path(target_dir)
            target_path.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(target_path))
            logger.info(f"Index replicated to {target_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to replicate index: {e}")
            return False

# Features implemented:
# - Incremental updates to FAISS index (add_documents_incremental)
# - Snapshotting with manifest.json and checksums (create_snapshot, verify_snapshot_integrity)
# - Automatic backups on index updates (save_index with create_backup=True)
# - Persistence and replication (save_index, load_index, replicate_index)
