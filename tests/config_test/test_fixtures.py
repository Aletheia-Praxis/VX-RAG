"""
Reusable test fixtures for VX-RAG tests.

This module provides common fixtures that can be used across all tests.
"""

from pathlib import Path
from typing import Any


def get_temp_test_dir(tmp_path: Path) -> Path:
    """
    Provide a temporary directory for test files.
    
    Args:
        tmp_path: Pytest's built-in temporary path fixture
        
    Returns:
        Path to temporary test directory
    """
    test_dir = tmp_path / "test_workspace"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def get_mock_config() -> dict[str, Any]:
    """
    Provide a mock configuration for tests.
    
    Returns:
        Dictionary with test configuration
    """
    return {
        'log_level': 'DEBUG',
        'log_format': 'json',
        'log_structured_events': True,
        'log_metrics': False,
        'chunk_size': 512,
        'chunk_overlap': 50,
        'similarity_top_k': 3,
    }
