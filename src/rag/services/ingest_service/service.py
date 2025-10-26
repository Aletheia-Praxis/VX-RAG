"""
Ingest Service implementation.

Provides classes and functions for document ingestion.
"""

from typing import List, Dict, Any
import logging
from pathlib import Path

from llama_index.core import Document, SimpleDirectoryReader

logger = logging.getLogger(__name__)

class IngestAdapter:
    """Base class for ingest adapters."""
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        """Load data from source and return unified format."""
        raise NotImplementedError


class PDFIngestAdapter(IngestAdapter):
    """Adapter for loading PDF documents."""
    
    def load_data(self, source: str) -> List[Dict[str, Any]]:
        """
        Load PDF documents from the specified directory.
        
        Args:
            source: Path to the directory containing PDF files
            
        Returns:
            List of document dictionaries with text and metadata
        """
        raw_pdf_dir = Path(source)
        
        if not raw_pdf_dir.exists():
            logger.error(f"Raw PDF directory does not exist: {raw_pdf_dir}")
            return []

        if not raw_pdf_dir.is_dir():
            logger.error(f"Raw PDF path is not a directory: {raw_pdf_dir}")
            return []

        try:
            logger.info(f"Scanning directory {raw_pdf_dir} for PDF files")
            # List all files to log ignored ones
            all_files = list(raw_pdf_dir.glob("*"))
            pdf_files = [f for f in all_files if f.suffix.lower() == '.pdf']
            other_files = [f for f in all_files if f.suffix.lower() != '.pdf' and f.is_file()]
            
            if other_files:
                logger.info(f"Found {len(pdf_files)} PDF files and {len(other_files)} other files (ignored): {[f.name for f in other_files]}")
            else:
                logger.info(f"Found {len(pdf_files)} PDF files, no other files to ignore")
            
            reader = SimpleDirectoryReader(
                input_dir=str(raw_pdf_dir),
                required_exts=[".pdf"],
                recursive=False  # Only process files directly in the directory
            )
            documents = reader.load_data()
            logger.info(f"Successfully loaded {len(documents)} PDF documents from {raw_pdf_dir}")
            
            # Convert to unified format
            result = []
            for doc in documents:
                result.append({
                    'text': doc.text,
                    'metadata': doc.metadata,
                    'id': doc.id_
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to load PDF documents from {raw_pdf_dir}: {e}")
            return []


def save_processed_text(documents: List[Dict[str, Any]], processed_dir: Path) -> int:
    """
    Save the text content of documents to processed directory as .txt files.

    Args:
        documents: List of document dictionaries
        processed_dir: Directory to save processed text files

    Returns:
        Number of successfully saved files
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0
    total_docs = len(documents)

    logger.info(f"Starting to save {total_docs} documents to {processed_dir}")

    for i, doc in enumerate(documents, 1):
        try:
            # Extract filename from metadata or use a default
            file_path = doc['metadata'].get('file_path', f'document_{i}.pdf')
            filename = Path(file_path).stem + ".txt"
            output_path = processed_dir / filename

            # Check if text is not empty
            text = doc.get('text', '').strip()
            if not text:
                logger.warning(f"Document {file_path} has no extractable text, skipping")
                continue

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)

            logger.info(f"Saved processed text ({i}/{total_docs}) to {output_path}")
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save document {doc['metadata'].get('file_path', f'document_{i}')}: {e}")

    logger.info(f"Successfully saved {saved_count}/{total_docs} documents")
    return saved_count