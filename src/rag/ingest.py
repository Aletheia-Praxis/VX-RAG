"""
Document ingestion module for VX-RAG system.

This module reads PDF files from data/raw/pdf/, extracts text content using LlamaIndex,
and stores processed text files in data/processed/ for indexing.
"""

import logging
from pathlib import Path
from typing import List

from llama_index.core import Document, SimpleDirectoryReader

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_pdfs(raw_pdf_dir: Path) -> List[Document]:
    """
    Load PDF documents from the raw directory using LlamaIndex SimpleDirectoryReader.

    Args:
        raw_pdf_dir: Path to the directory containing raw PDF files

    Returns:
        List of Document objects loaded from PDF files
    """
    if not raw_pdf_dir.exists():
        logger.error(f"Raw PDF directory does not exist: {raw_pdf_dir}")
        return []

    try:
        reader = SimpleDirectoryReader(
            input_dir=str(raw_pdf_dir),
            required_exts=[".pdf"],
            recursive=False  # Only process files directly in the directory
        )
        documents = reader.load_data()
        logger.info(f"Loaded {len(documents)} PDF documents from {raw_pdf_dir}")
        return documents
    except Exception as e:
        logger.error(f"Failed to load PDF documents: {e}")
        return []


def save_processed_text(documents: List[Document], processed_dir: Path) -> int:
    """
    Save the text content of documents to processed directory as .txt files.

    Args:
        documents: List of Document objects
        processed_dir: Directory to save processed text files

    Returns:
        Number of successfully saved files
    """
    processed_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    for doc in documents:
        try:
            # Extract filename from metadata or use a default
            file_path = doc.metadata.get('file_path', 'unknown.pdf')
            filename = Path(file_path).stem + ".txt"
            output_path = processed_dir / filename

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(doc.text)

            logger.info(f"Saved processed text to {output_path}")
            saved_count += 1
        except Exception as e:
            logger.error(f"Failed to save document {doc.metadata.get('file_path', 'unknown')}: {e}")

    return saved_count


def main() -> int:
    """
    Main entry point for document ingestion pipeline.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    logger.info("Starting document ingestion pipeline")

    raw_pdf_dir = Path("data/raw/pdf")
    processed_dir = Path("data/processed")

    # TODO: Add support for TXT files here
    # Use SimpleDirectoryReader with required_exts=[".txt"] or a custom text loader

    # TODO: Add support for MD files here
    # Use SimpleDirectoryReader with required_exts=[".md"] or a custom markdown loader

    documents = load_pdfs(raw_pdf_dir)

    if not documents:
        logger.warning("No documents were loaded. Check the raw PDF directory.")
        return 1

    saved_count = save_processed_text(documents, processed_dir)

    if saved_count == 0:
        logger.error("No documents were successfully saved.")
        return 1

    logger.info(f"Ingestion complete. Processed {saved_count} documents.")
    return 0


if __name__ == "__main__":
    exit(main())
