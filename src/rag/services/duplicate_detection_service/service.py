"""
Duplicate Detection Service implementation.

Provides functionality to identify and handle duplicate documents
based on content similarity and exact matches.
"""

from typing import List, Dict, Any, Set, Optional
import hashlib
import logging
from difflib import SequenceMatcher

from src.utils.config_loader import get_duplicate_detection_config

logger = logging.getLogger(__name__)

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
        
        assert similarity_threshold is not None, "similarity_threshold must be set"
        assert hash_algorithm is not None, "hash_algorithm must be set"
        
        self.similarity_threshold = similarity_threshold
        self.hash_algorithm = hash_algorithm

    def remove_duplicates(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate documents from the list.

        Args:
            documents: List of document dictionaries

        Returns:
            List of unique documents
        """
        if not documents:
            return []

        logger.info(f"Starting duplicate detection on {len(documents)} documents")

        # First pass: exact duplicates by content hash
        unique_by_hash = self._remove_exact_duplicates(documents)

        # Second pass: near-duplicates by similarity (optional, can be expensive)
        # For now, we'll skip similarity-based detection as it's computationally expensive
        # and exact duplicates are more common in document collections

        logger.info(f"Removed duplicates: {len(documents)} -> {len(unique_by_hash)} documents")
        return unique_by_hash

    def _remove_exact_duplicates(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove exact duplicates based on content hashing.

        Args:
            documents: List of document dictionaries

        Returns:
            List of unique documents
        """
        seen_hashes: Set[str] = set()
        unique_docs = []

        for doc in documents:
            content_hash = self._hash_content(doc.get('text', ''))

            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
            else:
                logger.info(f"Removed exact duplicate: {doc.get('id', 'unknown')}")

        return unique_docs

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

    def find_similar_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find near-duplicate documents using text similarity.

        Note: This is computationally expensive and should be used sparingly.

        Args:
            documents: List of document dictionaries

        Returns:
            List of similarity groups (each group contains similar documents)
        """
        similarity_groups: List[Dict[str, Any]] = []

        # Simple pairwise comparison (O(n^2) - only suitable for small datasets)
        processed = set()

        for i, doc1 in enumerate(documents):
            if i in processed:
                continue

            group = [doc1]
            processed.add(i)

            for j, doc2 in enumerate(documents):
                if j in processed or i == j:
                    continue

                similarity = self._calculate_similarity(
                    doc1.get('text', ''),
                    doc2.get('text', '')
                )

                if similarity >= self.similarity_threshold:
                    group.append(doc2)
                    processed.add(j)

            if len(group) > 1:
                similarity_groups.append({
                    'group_id': f"group_{len(similarity_groups)}",
                    'documents': group,
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