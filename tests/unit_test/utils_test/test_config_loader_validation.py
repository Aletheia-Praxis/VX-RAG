"""
Unit tests for configuration loader with Pydantic validation.

Tests the config_loader module's ability to load and validate configuration.
"""

import pytest
import yaml
from pathlib import Path

from src.utils.config_loader import load_settings, get_validated_settings
from src.utils.config_schemas import VXRAGSettings


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary valid config file."""
    config_data = {
        "data_dir": "./data",
        "raw_data_dir": "./data/raw",
        "processed_data_dir": "./data/processed",
        "index_dir": "./data/index",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_device": "cpu",
        "embedding_batch_size": 10,
        "chunk_size": 1024,
        "chunk_overlap": 200,
        "vector_store": "faiss",
        "similarity_top_k": 5,
        "query_temperature": 0.1,
        "metrics_enabled": True,
        "metrics_log_interval": 300,
        "faiss": {
            "hnsw_m": 32,
            "metric": "inner_product"
        },
        "mcp": {
            "host": "127.0.0.1",
            "port": 25191,
            "debug": True
        }
    }
    
    config_file = tmp_path / "settings.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f)
    
    return config_file


@pytest.fixture
def invalid_config_file(tmp_path: Path) -> Path:
    """Create a temporary invalid config file (overlap >= chunk_size)."""
    config_data = {
        "chunk_size": 512,
        "chunk_overlap": 512  # Invalid: overlap >= chunk_size
    }
    
    config_file = tmp_path / "invalid_settings.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_data, f)
    
    return config_file


@pytest.fixture
def malformed_yaml_file(tmp_path: Path) -> Path:
    """Create a temporary malformed YAML file."""
    config_file = tmp_path / "malformed.yaml"
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write("invalid: yaml: content: [unclosed")
    
    return config_file


class TestLoadSettings:
    """Tests for load_settings function."""
    
    def test_load_valid_config(self, temp_config_file: Path) -> None:
        """Test loading a valid configuration file."""
        config = load_settings(str(temp_config_file))
        
        assert isinstance(config, dict)
        assert config["embedding_model"] == "all-MiniLM-L6-v2"
        assert config["chunk_size"] == 1024
        assert config["chunk_overlap"] == 200
    
    def test_load_config_with_validation(self, temp_config_file: Path) -> None:
        """Test that loaded config is validated by Pydantic."""
        config = load_settings(str(temp_config_file))
        
        # Verify nested configs are properly validated
        assert "faiss" in config
        assert config["faiss"]["hnsw_m"] == 32
        assert "mcp" in config
        assert config["mcp"]["port"] == 25191
    
    def test_load_invalid_config(self, invalid_config_file: Path) -> None:
        """Test loading an invalid configuration raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_settings(str(invalid_config_file))
        
        assert "validation errors" in str(exc_info.value).lower()
        assert "chunk_overlap" in str(exc_info.value).lower()
    
    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        """Test loading a nonexistent file raises FileNotFoundError."""
        nonexistent = tmp_path / "does_not_exist.yaml"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_settings(str(nonexistent))
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_load_malformed_yaml(self, malformed_yaml_file: Path) -> None:
        """Test loading malformed YAML raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            load_settings(str(malformed_yaml_file))
        
        assert "yaml" in str(exc_info.value).lower()
    
    def test_load_config_with_defaults(self, tmp_path: Path) -> None:
        """Test loading minimal config applies defaults."""
        # Create minimal config with only required fields
        minimal_config = {
            "data_dir": "./data"
        }
        
        config_file = tmp_path / "minimal.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(minimal_config, f)
        
        config = load_settings(str(config_file))
        
        # Verify defaults are applied
        assert config["embedding_model"] == "all-MiniLM-L6-v2"
        assert config["chunk_size"] == 1024
        assert config["chunk_overlap"] == 200
    
    def test_validation_catches_out_of_range_values(self, tmp_path: Path) -> None:
        """Test validation catches values outside allowed ranges."""
        invalid_config = {
            "embedding_batch_size": 2000  # Above max of 1000
        }
        
        config_file = tmp_path / "out_of_range.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ValueError) as exc_info:
            load_settings(str(config_file))
        
        assert "embedding_batch_size" in str(exc_info.value)
    
    def test_validation_catches_invalid_device(self, tmp_path: Path) -> None:
        """Test validation catches invalid device values."""
        invalid_config = {
            "embedding_device": "invalid_device"
        }
        
        config_file = tmp_path / "invalid_device.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(invalid_config, f)
        
        with pytest.raises(ValueError) as exc_info:
            load_settings(str(config_file))
        
        assert "embedding_device" in str(exc_info.value)


class TestGetValidatedSettings:
    """Tests for get_validated_settings function."""
    
    def test_get_settings_after_load(self, temp_config_file: Path) -> None:
        """Test getting validated settings after loading."""
        # Load settings first
        load_settings(str(temp_config_file))
        
        # Get validated settings
        settings = get_validated_settings()
        
        assert isinstance(settings, VXRAGSettings)
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.chunk_size == 1024
    
    def test_get_settings_auto_loads(self) -> None:
        """Test that get_validated_settings auto-loads if not already loaded."""
        # This test assumes there's a default config at config/settings.yaml
        # If running in a test environment without it, it should handle gracefully
        try:
            settings = get_validated_settings()
            assert isinstance(settings, VXRAGSettings)
        except FileNotFoundError:
            # Expected if no default config exists
            pytest.skip("No default config file available")
    
    def test_settings_persistence(self, temp_config_file: Path) -> None:
        """Test that settings persist across multiple calls."""
        load_settings(str(temp_config_file))
        
        settings1 = get_validated_settings()
        settings2 = get_validated_settings()
        
        # Should return the same instance
        assert settings1 is settings2


class TestConfigValidationIntegration:
    """Integration tests for configuration validation."""
    
    def test_complete_valid_config_loads(self, tmp_path: Path) -> None:
        """Test loading a complete valid configuration."""
        complete_config = {
            "data_dir": "./data",
            "raw_data_dir": "./data/raw",
            "processed_data_dir": "./data/processed",
            "index_dir": "./data/index",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_device": "cpu",
            "embedding_batch_size": 10,
            "embedding_cache_size": 1000,
            "embedding_trust_remote_code": False,
            "chunk_size": 1024,
            "chunk_overlap": 200,
            "vector_store": "faiss",
            "similarity_top_k": 5,
            "query_temperature": 0.1,
            "metrics_enabled": True,
            "metrics_log_interval": 300,
            "metrics_file": "metrics.log",
            "faiss": {
                "hnsw_m": 32,
                "metric": "inner_product"
            },
            "adaptive_chunking": {
                "enabled": True,
                "max_chunk_size": 2000,
                "large_code_blocks": {
                    "chunk_size": 512,
                    "chunk_overlap": 50,
                    "preserve_integrity": True
                }
            },
            "retriever": {
                "semantic_top_k": 20,
                "hybrid_alpha": 0.5,
                "enable_hybrid": True
            },
            "bm25": {
                "index_dir": "./data/index/bm25",
                "similarity_top_k": 20,
                "enable_persistence": True
            },
            "reranker": {
                "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "top_k": 5,
                "device": "cpu",
                "metadata_boost": 0.1
            },
            "context_assembler": {
                "token_budget": 2048,
                "model_name": "gpt-3.5-turbo"
            },
            "mcp": {
                "host": "127.0.0.1",
                "port": 25191,
                "debug": True,
                "rate_limit": {
                    "max_concurrent": 2,
                    "queue_size": 10,
                    "default_timeout": 600.0
                }
            },
            "duplicate_detection": {
                "similarity_threshold": 0.95,
                "hash_algorithm": "sha256"
            },
            "boilerplate_removal": {
                "enabled": True,
                "aggressive_mode": True,
                "position": "after_ocr"
            },
            "logging": {
                "log_level": "INFO",
                "log_file": "logs/vx_rag.log",
                "log_format": "json"
            }
        }
        
        config_file = tmp_path / "complete.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(complete_config, f)
        
        config = load_settings(str(config_file))
        
        # Verify all sections loaded
        assert config["embedding_model"] == "all-MiniLM-L6-v2"
        assert config["faiss"]["hnsw_m"] == 32
        assert config["mcp"]["port"] == 25191
        assert config["retriever"]["hybrid_alpha"] == 0.5
    
    def test_partial_config_with_defaults(self, tmp_path: Path) -> None:
        """Test that partial config uses defaults for missing values."""
        partial_config = {
            "chunk_size": 2048,
            "chunk_overlap": 400,
            "mcp": {
                "port": 30000
            }
        }
        
        config_file = tmp_path / "partial.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(partial_config, f)
        
        config = load_settings(str(config_file))
        
        # Verify custom values
        assert config["chunk_size"] == 2048
        assert config["mcp"]["port"] == 30000
        
        # Verify defaults
        assert config["embedding_model"] == "all-MiniLM-L6-v2"
        assert config["mcp"]["host"] == "127.0.0.1"
