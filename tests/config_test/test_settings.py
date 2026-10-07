"""
Tests for configuration files and settings validation.

Validates that both the production config/settings.yaml and the test
tests/config_test/settings.yaml adhere strictly to the VXRAGSettings schema
and that config_loader functions successfully retrieve configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import yaml

from src.utils.config_loader import (
    get_context_assembler_config,
    get_data_directories,
    get_docstore_config,
    get_embedding_config,
    get_embedding_dimension,
    get_embedding_dimensions,
    get_logging_config,
    get_mcp_config,
    get_metrics_config,
    get_qdrant_config,
    get_rate_limiter_config,
    get_reranker_config,
    get_task_queue_config,
    get_validated_settings,
    load_settings,
)
from src.utils.config_schemas import VXRAGSettings

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401


def test_production_settings_file_exists() -> None:
    """Verify that the production settings.yaml file exists on disk."""
    prod_path = Path("config/settings.yaml")
    assert prod_path.exists(), f"Configuration file not found at {prod_path}"
    assert prod_path.is_file(), f"Path is not a regular file: {prod_path}"


def test_test_settings_file_exists() -> None:
    """Verify that the test settings.yaml file exists on disk."""
    test_path = Path("tests/config_test/settings.yaml")
    assert test_path.exists(), f"Test configuration file not found at {test_path}"
    assert test_path.is_file(), f"Path is not a regular file: {test_path}"


def test_production_settings_validation() -> None:
    """Verify that production settings.yaml validates cleanly against VXRAGSettings."""
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    settings = VXRAGSettings.model_validate(data)
    assert settings is not None
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_dimensions == {"BAAI/bge-m3": 1024}
    assert settings.reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert settings.qdrant.collection_name == "vx_rag_collection"
    assert settings.mcp.host == "127.0.0.1"


def test_test_settings_validation() -> None:
    """Verify that test settings.yaml validates cleanly against VXRAGSettings."""
    with open("tests/config_test/settings.yaml", "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    settings = VXRAGSettings.model_validate(data)
    assert settings is not None
    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_dimensions == {"BAAI/bge-m3": 1024}
    assert settings.reranker.model_name == "BAAI/bge-reranker-v2-m3"
    assert settings.qdrant.collection_name == "test_vx_rag_collection"


def test_load_settings_prod() -> None:
    """Verify that load_settings successfully parses production settings."""
    config = load_settings("config/settings.yaml", force_reload=True)
    assert isinstance(config, dict)
    assert "embedding_model" in config
    assert "qdrant" in config
    assert "mcp" in config


def test_load_settings_test() -> None:
    """Verify that load_settings successfully parses test settings."""
    config = load_settings("tests/config_test/settings.yaml", force_reload=True)
    assert isinstance(config, dict)
    assert "embedding_model" in config
    assert "qdrant" in config


def test_config_loader_getters_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that config_loader getters return valid dictionaries with test config."""
    monkeypatch.setenv("VX_RAG_CONFIG_PATH", "tests/config_test/settings.yaml")
    # Force reload with the env var
    load_settings(force_reload=True)

    emb_cfg = get_embedding_config()
    assert emb_cfg["embedding_model"] == "BAAI/bge-m3"
    assert get_embedding_dimension() == 1024
    assert get_embedding_dimensions() == {"BAAI/bge-m3": 1024}

    qdrant_cfg = get_qdrant_config()
    assert qdrant_cfg["collection_name"] == "test_vx_rag_collection"

    reranker_cfg = get_reranker_config()
    assert reranker_cfg["model_name"] == "BAAI/bge-reranker-v2-m3"

    mcp_cfg = get_mcp_config()
    assert "host" in mcp_cfg
    assert "port" in mcp_cfg

    rl_cfg = get_rate_limiter_config()
    assert "max_concurrent" in rl_cfg

    tq_cfg = get_task_queue_config()
    assert "max_concurrent_tasks" in tq_cfg

    log_cfg = get_logging_config()
    assert "level" in log_cfg or "log_level" in log_cfg

    metrics_cfg = get_metrics_config()
    assert "metrics_enabled" in metrics_cfg

    data_dirs = get_data_directories()
    assert "data_dir" in data_dirs

    docstore_cfg = get_docstore_config()
    assert "store_type" in docstore_cfg or "type" in docstore_cfg

    ca_cfg = get_context_assembler_config()
    assert "token_budget" in ca_cfg

    validated = get_validated_settings()
    assert isinstance(validated, VXRAGSettings)
