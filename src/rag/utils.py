"""
Utility functions for VX-RAG.

Helper functions for file handling, logging, etc.
"""

import logging
from pathlib import Path

def setup_logging(log_level: str = "INFO"):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, etc.).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def list_files(directory: Path, extensions: list[str] = None) -> list[Path]:
    """
    List files in a directory with optional extension filter.

    Args:
        directory: Directory to list files from.
        extensions: List of file extensions to include (e.g., ['.pdf', '.txt']).

    Returns:
        List of file paths.
    """
    if extensions:
        return [f for f in directory.rglob("*") if f.is_file() and f.suffix in extensions]
    return [f for f in directory.rglob("*") if f.is_file()]

def read_text_file(file_path: Path) -> str:
    """
    Read text from a file.

    Args:
        file_path: Path to the file.

    Returns:
        File contents as string.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Utils module loaded")