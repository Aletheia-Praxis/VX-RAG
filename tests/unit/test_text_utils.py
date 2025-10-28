"""
Tests for text_utils module using pytest.
"""

import pytest
from typing import TYPE_CHECKING
from unittest.mock import patch

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

from src.rag.libs.utils.text_utils import normalize_text, detect_language


def test_normalize_text_empty() -> None:
    """Test normalize_text with empty string."""
    assert normalize_text("") == ""


def test_normalize_text_basic() -> None:
    """Test normalize_text with basic text."""
    text = "  Hello   world!  "
    expected = "Hello world!"
    assert normalize_text(text) == expected


def test_normalize_text_unicode() -> None:
    """Test normalize_text with Unicode characters."""
    text = "café naïve résumé"
    # NFKC normalization should handle these
    result = normalize_text(text)
    assert "é" in result  # Should preserve accented characters


def test_normalize_text_whitespace() -> None:
    """Test normalize_text with various whitespace."""
    text = "Line 1\n\nLine 2\t\tLine 3"
    expected = "Line 1 Line 2 Line 3"
    assert normalize_text(text) == expected


def test_normalize_text_control_chars() -> None:
    """Test normalize_text removes control characters."""
    text = "Hello\x00world\x01test"
    expected = "Helloworldtest"
    assert normalize_text(text) == expected


def test_detect_language_empty() -> None:
    """Test detect_language with empty text."""
    assert detect_language("") == "unknown"


def test_detect_language_short() -> None:
    """Test detect_language with short text."""
    assert detect_language("Hi") == "unknown"


def test_detect_language_english() -> None:
    """Test detect_language with English text."""
    text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
    result = detect_language(text)
    assert result == "en"


def test_detect_language_ukrainian() -> None:
    """Test detect_language with Ukrainian text."""
    text = "Привіт світ. Це тестове речення українською мовою з літерами ії та є."
    result = detect_language(text)
    assert result == "uk"


def test_detect_language_russian() -> None:
    """Test detect_language with Russian text."""
    text = "Привет мир. Это тестовое предложение на русском языке с буквами ё и ъ."
    result = detect_language(text)
    assert result == "ru"


def test_detect_language_mixed_uk_ru() -> None:
    """Test detect_language with mixed Ukrainian/Russian text."""
    # Text with Ukrainian specific chars should be detected as Ukrainian
    text = "Привіт, как дела? Це мішаний текст з і та є."
    result = detect_language(text)
    assert result == "uk"


def test_detect_language_unknown() -> None:
    """Test detect_language with unknown language."""
    text = "Dies ist ein Test auf Deutsch. C'est un test en français."
    result = detect_language(text)
    assert result == "unknown"


def test_detect_language_no_langdetect() -> None:
    """Test detect_language when langdetect is not available."""
    with patch('src.rag.libs.utils.text_utils.LANGDETECT_AVAILABLE', False):
        # English
        assert detect_language("The quick brown fox jumps over the lazy dog.") == "en"
        # Ukrainian
        assert detect_language("Привіт світ. Це тестове речення.") == "uk"
        # Russian
        assert detect_language("Привет мир. Это тестовое предложение.") == "ru"
        # Unknown
        assert detect_language("xyz123") == "unknown"


def test_detect_language_langdetect_success() -> None:
    """Test detect_language with successful langdetect."""
    with patch('src.rag.libs.utils.text_utils.LANGDETECT_AVAILABLE', True):
        with patch('src.rag.libs.utils.text_utils.detect') as mock_detect:
            mock_detect.return_value = 'uk'
            text = "Тестовий текст"
            assert detect_language(text) == "uk"


def test_detect_language_langdetect_unsupported() -> None:
    """Test detect_language with langdetect returning unsupported language."""
    with patch('src.rag.libs.utils.text_utils.LANGDETECT_AVAILABLE', True):
        with patch('src.rag.libs.utils.text_utils.detect') as mock_detect:
            mock_detect.return_value = 'de'  # German, not supported
            text = "Test text"
            assert detect_language(text) == "unknown"


def test_detect_language_langdetect_exception() -> None:
    """Test detect_language handles langdetect exceptions."""
    with patch('src.rag.libs.utils.text_utils.LANGDETECT_AVAILABLE', True):
        with patch('src.rag.libs.utils.text_utils.detect', side_effect=Exception("Detection failed")):
            text = "The quick brown fox jumps over the lazy dog."
            # Should fall back to heuristic
            result = detect_language(text)
            assert result in ["en", "uk", "ru", "unknown"]