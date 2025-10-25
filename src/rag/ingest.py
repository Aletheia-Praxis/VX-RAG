#!/usr/bin/env python3
"""
Document ingestion module for VX-RAG system.

This module reads PDF files from data/raw/pdf/, extracts text content,
and stores processed text files in data/processed/ for indexing.
"""

import os
import logging
from pathlib import Path
from pypdf import PdfReader
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIngester:
    """Handles PDF document ingestion and text extraction."""

    def __init__(self, raw_dir: str = "data/raw/pdf", processed_dir: str = "data/processed"):
        """
        Initialize the document ingester.

        Args:
            raw_dir: Directory containing raw PDF files
            processed_dir: Directory to store processed text files
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """
        Extract text content from a PDF file.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Extracted text content or None if extraction fails
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to extract text from {pdf_path}: {e}")
            return None

    def process_pdf(self, pdf_path: Path) -> bool:
        """
        Process a single PDF file and save extracted text.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            True if processing succeeded, False otherwise
        """
        text = self.extract_text_from_pdf(pdf_path)
        if text is None:
            return False

        # Create output filename
        output_filename = pdf_path.stem + ".txt"
        output_path = self.processed_dir / output_filename

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Processed {pdf_path} -> {output_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save processed text for {pdf_path}: {e}")
            return False

    def ingest_all_pdfs(self) -> int:
        """
        Process all PDF files in the raw directory.

        Returns:
            Number of successfully processed files
        """
        if not self.raw_dir.exists():
            logger.error(f"Raw directory does not exist: {self.raw_dir}")
            return 0

        pdf_files = list(self.raw_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.raw_dir}")
            return 0

        logger.info(f"Found {len(pdf_files)} PDF files to process")

        processed_count = 0
        for pdf_file in pdf_files:
            if self.process_pdf(pdf_file):
                processed_count += 1

        logger.info(f"Successfully processed {processed_count}/{len(pdf_files)} files")
        return processed_count


def main():
    """Main entry point for document ingestion."""
    ingester = DocumentIngester()
    processed_count = ingester.ingest_all_pdfs()

    if processed_count == 0:
        logger.error("No files were processed. Check your data directory and PDF files.")
        return 1

    logger.info(f"Ingestion complete. Processed {processed_count} documents.")
    return 0


if __name__ == "__main__":
    exit(main())
