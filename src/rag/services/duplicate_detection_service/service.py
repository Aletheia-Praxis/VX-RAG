"""
Duplicate Detection Service implementation.

Provides functionality to identify and handle duplicate documents
based on content similarity and exact matches.
"""

from typing import List, Dict, Any, Set, Optional, Union
import hashlib
from src.utils.logging_config import get_logger
from difflib import SequenceMatcher

from llama_index.core.schema import BaseNode

from src.utils.config_loader import get_duplicate_detection_config

logger = get_logger(__name__)

class DuplicateDetector:
    """
    Service for detecting and handling duplicate documents.

    Features:
    - Exact duplicate detection using content hashing
    - Near-duplicate detection using text similarity
    - Configurable similarity thresholds
    """

    def __init__(self, similarity_threshold: Optional[float] = None, hash_algorithm: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the duplicate detector.

        Args:
            similarity_threshold: Threshold for near-duplicate detection (0.0-1.0). If None, loads from config.
            hash_algorithm: Hash algorithm for content hashing. If None, loads from config.
            config_path: Path to settings.yaml. If None, uses default location.
        """
        # Load from config if parameters not provided
        if similarity_threshold is None or hash_algorithm is None:
            config = get_duplicate_detection_config(config_path)
            if similarity_threshold is None:
                similarity_threshold = config['similarity_threshold']
            if hash_algorithm is None:
                hash_algorithm = config['hash_algorithm']
        
        if similarity_threshold is None:
            raise ValueError("similarity_threshold must be set")
        if hash_algorithm is None:
            raise ValueError("hash_algorithm must be set")
        
        self.similarity_threshold = similarity_threshold
        self.hash_algorithm = hash_algorithm

    def remove_duplicates(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        Remove duplicate nodes from the list based on content hash.

        Args:
            nodes: List of LlamaIndex BaseNode objects

        Returns:
            List of unique nodes
        """
        if not nodes:
            return []

        logger.info(f"Starting duplicate detection on {len(nodes)} nodes")

        # First pass: exact duplicates by content hash
        unique_by_hash = self._remove_exact_duplicates(nodes)

        logger.info(f"Removed duplicates: {len(nodes)} -> {len(unique_by_hash)} nodes")
        return unique_by_hash

    def _remove_exact_duplicates(self, nodes: List[BaseNode]) -> List[BaseNode]:
        """
        Remove exact duplicates based on content hashing.

        Args:
            nodes: List of LlamaIndex BaseNode objects

        Returns:
            List of unique nodes
        """
        seen_hashes: Set[str] = set()
        unique_nodes = []

        for node in nodes:
            content = node.get_content()
            content_hash = self._hash_content(content)

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_nodes.append(node)
            else:
                # Log duplicate found (optional: log ID)
                # logger.debug(f"Removed exact duplicate node: {node.node_id}")
                pass

        return unique_nodes

    def _hash_content(self, content: str) -> str:
        """
        Generate hash of document content.

        Args:
            content: Text content to hash

        Returns:
            Content hash
        """
        # Normalize content for hashing (remove extra whitespace)
        normalized = ' '.join(content.split())
        return hashlib.new(self.hash_algorithm, normalized.encode('utf-8')).hexdigest()

    def find_similar_documents(self, nodes: List[BaseNode]) -> List[Dict[str, Any]]:
        """
        Find near-duplicate nodes using text similarity.

        Note: This is computationally expensive and should be used sparingly.

        Args:
            nodes: List of LlamaIndex BaseNode objects

        Returns:
            List of similarity groups (each group contains similar nodes)
        """
        similarity_groups: List[Dict[str, Any]] = []

        # Simple pairwise comparison (O(n^2) - only suitable for small datasets)
        processed = set()

        for i, node1 in enumerate(nodes):
            if i in processed:
                continue

            group = [node1]
            processed.add(i)

            for j, node2 in enumerate(nodes):
                if j in processed or i == j:
                    continue

                similarity = self._calculate_similarity(
                    node1.get_content(),
                    node2.get_content()
                )

                if similarity >= self.similarity_threshold:
                    group.append(node2)
                    processed.add(j)

            if len(group) > 1:
                similarity_groups.append({
                    'group_id': f"group_{len(similarity_groups)}",
                    'nodes': group,
                    'similarity_score': similarity
                })

        return similarity_groups

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using sequence matching.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0-1.0)
        """
        return SequenceMatcher(None, text1, text2).ratio()