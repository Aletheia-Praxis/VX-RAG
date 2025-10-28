"""
Text processing utilities for the RAG system.

Provides functions for text normalization, language detection, and other text-related operations.
"""

import re
import unicodedata
import logging
from typing import Optional

try:
    from langdetect import detect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False


def normalize_text(text: str) -> str:
    """
    Normalize text by removing extra whitespace, normalizing Unicode, and cleaning up formatting.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    if not text:
        return ""

    # Normalize Unicode (e.g., convert special characters to standard form)
    text = unicodedata.normalize('NFKC', text)

    # Remove extra whitespace and normalize line breaks
    text = re.sub(r'\s+', ' ', text.strip())

    # Remove control characters except common whitespace
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    return text


def detect_language(text: str) -> str:
    """
    Detect the language of the given text, focusing on Ukrainian, Russian, and English.

    Args:
        text: Input text for language detection

    Returns:
        Language code: 'uk' for Ukrainian, 'ru' for Russian, 'en' for English, 'unknown' otherwise
    """
    if not text or len(text.strip()) < 10:
        return 'unknown'

    text_lower = text.lower()

    if LANGDETECT_AVAILABLE:
        try:
            lang = detect(text)
            # Map to our supported languages
            if lang in ['uk', 'ru', 'en']:
                return lang
            else:
                return 'unknown'
        except Exception:
            logging.warning("Language detection failed, falling back to heuristic")

    # Heuristic detection for Ukrainian, Russian, English
    # Ukrainian specific characters
    ukrainian_chars = re.search(r'[іїєґ]', text_lower)
    # Russian specific characters (excluding Ukrainian ones)
    russian_chars = re.search(r'[ъыэюяё]', text_lower)
    # Common Cyrillic (shared)
    cyrillic_chars = re.search(r'[а-я]', text_lower)
    # English words
    english_words = re.search(r'\b(the|and|or|but|in|on|at|to|for|of|with|by|is|are|was|were|this|that|it|he|she|they|we|you|i|me|my|your|his|her|their|our)\b', text_lower)

    if ukrainian_chars:
        return 'uk'
    elif russian_chars:
        return 'ru'
    elif english_words:
        return 'en'
    elif cyrillic_chars:
        # If Cyrillic but no specific chars, assume Russian (more common)
        return 'ru'
    else:
        return 'unknown'
