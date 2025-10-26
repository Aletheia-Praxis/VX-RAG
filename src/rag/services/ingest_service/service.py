"""
Ingest Service implementation.

Provides classes and functions for document ingestion.
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class IngestAdapter:
    """Base class for ingest adapters."""
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        """Load data from source and return unified format."""
        raise NotImplementedError

# TODO: Implement PDFAdapter, HTMLAdapter, APIAdapter, etc.