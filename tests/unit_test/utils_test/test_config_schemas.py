"""
Unit tests for configuration validation schemas.

Tests Pydantic models that validate configuration loaded from settings.yaml.
"""

import pytest
from pydantic import ValidationError

from src.utils.config_schemas import (
    VXRAGSettings,
    EmbeddingConfig,
    ChunkingConfig,
    AdaptiveChunkingProfile,
    FAISSConfig,
    RetrieverConfig,
    RerankerConfig,
    MCPConfig,
    RateLimitConfig,
    DuplicateDetectionConfig,
    PaddleOCRConfig,
)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig validation."""
    
    def test_valid_config(self) -> None:
        """Test valid embedding configuration."""
        config = EmbeddingConfig(
            embedding_model="all-MiniLM-L6-v2",
            embedding_device="cpu",
            embedding_batch_size=10
        )
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.embedding_device == "cpu"
        assert config.embedding_batch_size == 10
    
    def test_batch_size_validation(self) -> None:
        """Test batch size must be within valid range."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingConfig(embedding_batch_size=0)
        assert "embedding_batch_size" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingConfig(embedding_batch_size=2000)
        assert "embedding_batch_size" in str(exc_info.value)
    
    def test_device_validation(self) -> None:
        """Test device must be cpu or cuda."""
        with pytest.raises(ValidationError) as exc_info:
            EmbeddingConfig(embedding_device="invalid")  # type: ignore
        assert "embedding_device" in str(exc_info.value)


class TestChunkingConfig:
    """Tests for ChunkingConfig validation."""
    
    def test_valid_config(self) -> None:
        """Test valid chunking configuration."""
        config = ChunkingConfig(
            chunk_size=1024,
            chunk_overlap=200
        )
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 200
    
    def test_overlap_less_than_size(self) -> None:
        """Test overlap must be less than chunk size."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(chunk_size=512, chunk_overlap=512)
        assert "chunk_overlap" in str(exc_info.value).lower()
    
    def test_chunk_size_bounds(self) -> None:
        """Test chunk size must be within valid range."""
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(chunk_size=100)  # Too small
        assert "chunk_size" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            ChunkingConfig(chunk_size=5000)  # Too large
        assert "chunk_size" in str(exc_info.value)


class TestAdaptiveChunkingProfile:
    """Tests for AdaptiveChunkingProfile validation."""
    
    def test_valid_profile(self) -> None:
        """Test valid adaptive chunking profile."""
        profile = AdaptiveChunkingProfile(
            chunk_size=512,
            chunk_overlap=50
        )
        assert profile.chunk_size == 512
        assert profile.chunk_overlap == 50
    
    def test_overlap_validation(self) -> None:
        """Test overlap must be less than chunk size in profiles."""
        with pytest.raises(ValidationError) as exc_info:
            AdaptiveChunkingProfile(chunk_size=256, chunk_overlap=256)
        assert "chunk_overlap" in str(exc_info.value).lower()


class TestFAISSConfig:
    """Tests for FAISSConfig validation."""
    
    def test_valid_config(self) -> None:
        """Test valid FAISS configuration."""
        config = FAISSConfig(hnsw_m=32, metric="inner_product")
        assert config.hnsw_m == 32
        assert config.metric == "inner_product"
    
    def test_hnsw_m_bounds(self) -> None:
        """Test HNSW M parameter bounds."""
        with pytest.raises(ValidationError) as exc_info:
            FAISSConfig(hnsw_m=2)  # Too small
        assert "hnsw_m" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            FAISSConfig(hnsw_m=200)  # Too large
        assert "hnsw_m" in str(exc_info.value)
    
    def test_metric_validation(self) -> None:
        """Test metric must be valid."""
        with pytest.raises(ValidationError) as exc_info:
            FAISSConfig(metric="cosine")  # type: ignore
        assert "metric" in str(exc_info.value)


class TestRetrieverConfig:
    """Tests for RetrieverConfig validation."""
    
    def test_valid_config(self) -> None:
        """Test valid retriever configuration."""
        config = RetrieverConfig(
            semantic_top_k=20,
            hybrid_alpha=0.5,
            enable_hybrid=True
        )
        assert config.semantic_top_k == 20
        assert config.hybrid_alpha == 0.5
        assert config.enable_hybrid is True
    
    def test_alpha_bounds(self) -> None:
        """Test hybrid alpha must be between 0 and 1."""
        with pytest.raises(ValidationError) as exc_info:
            RetrieverConfig(hybrid_alpha=-0.1)
        assert "hybrid_alpha" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            RetrieverConfig(hybrid_alpha=1.5)
        assert "hybrid_alpha" in str(exc_info.value)


class TestRerankerConfig:
    """Tests for RerankerConfig validation."""
    
    def test_valid_config(self) -> None:
        """Test valid reranker configuration."""
        config = RerankerConfig(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=5,
            device="cpu"
        )
        assert config.top_k == 5
        assert config.device == "cpu"
    
    def test_top_k_bounds(self) -> None:
        """Test top K must be within valid range."""
        with pytest.raises(ValidationError) as exc_info:
            RerankerConfig(top_k=0)
        assert "top_k" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            RerankerConfig(top_k=100)
        assert "top_k" in str(exc_info.value)


class TestRateLimitConfig:
    """Tests for RateLimitConfig validation."""
    
    def test_valid_config(self) -> None:
        """Test valid rate limit configuration."""
        config = RateLimitConfig(
            max_concurrent=2,
            queue_size=10,
            default_timeout=600.0
        )
        assert config.max_concurrent == 2
        assert config.queue_size == 10
        assert config.default_timeout == 600.0
    
    def test_bounds_validation(self) -> None:
        """Test rate limit bounds."""
        with pytest.raises(ValidationError) as exc_info:
            RateLimitConfig(max_concurrent=0)
        assert "max_concurrent" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            RateLimitConfig(queue_size=0)
        assert "queue_size" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            RateLimitConfig(default_timeout=0.5)  # Too short
        assert "default_timeout" in str(exc_info.value)


class TestMCPConfig:
    """Tests for MCPConfig validation."""
    
    def test_valid_config(self) -> None:
        """Test valid MCP configuration."""
        config = MCPConfig(
            host="127.0.0.1",
            port=25191,
            debug=True
        )
        assert config.host == "127.0.0.1"
        assert config.port == 25191
        assert config.debug is True
    
    def test_port_bounds(self) -> None:
        """Test port must be within valid range."""
        with pytest.raises(ValidationError) as exc_info:
            MCPConfig(port=80)  # Below 1024
        assert "port" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            MCPConfig(port=70000)  # Above 65535
        assert "port" in str(exc_info.value)


class TestDuplicateDetectionConfig:
    """Tests for DuplicateDetectionConfig validation."""
    
    def test_valid_config(self) -> None:
        """Test valid duplicate detection configuration."""
        config = DuplicateDetectionConfig(
            similarity_threshold=0.95,
            hash_algorithm="sha256"
        )
        assert config.similarity_threshold == 0.95
        assert config.hash_algorithm == "sha256"
    
    def test_threshold_bounds(self) -> None:
        """Test similarity threshold must be between 0 and 1."""
        with pytest.raises(ValidationError) as exc_info:
            DuplicateDetectionConfig(similarity_threshold=1.5)
        assert "similarity_threshold" in str(exc_info.value)
    
    def test_hash_algorithm_validation(self) -> None:
        """Test hash algorithm must be valid."""
        with pytest.raises(ValidationError) as exc_info:
            DuplicateDetectionConfig(hash_algorithm="invalid")  # type: ignore
        assert "hash_algorithm" in str(exc_info.value)


class TestPaddleOCRConfig:
    """Tests for PaddleOCRConfig validation."""
    
    def test_valid_config(self) -> None:
        """Test valid PaddleOCR configuration."""
        config = PaddleOCRConfig(
            enabled=True,
            lang="en",
            use_gpu=False
        )
        assert config.enabled is True
        assert config.lang == "en"
        assert config.use_gpu is False
    
    def test_cpu_threads_bounds(self) -> None:
        """Test CPU threads must be within valid range."""
        with pytest.raises(ValidationError) as exc_info:
            PaddleOCRConfig(cpu_threads=0)
        assert "cpu_threads" in str(exc_info.value)
        
        with pytest.raises(ValidationError) as exc_info:
            PaddleOCRConfig(cpu_threads=100)
        assert "cpu_threads" in str(exc_info.value)
    
    def test_confidence_bounds(self) -> None:
        """Test min confidence must be between 0 and 1."""
        with pytest.raises(ValidationError) as exc_info:
            PaddleOCRConfig(min_confidence=-0.1)
        assert "min_confidence" in str(exc_info.value)


class TestVXRAGSettings:
    """Tests for complete VXRAGSettings validation."""
    
    def test_valid_minimal_config(self) -> None:
        """Test valid minimal configuration with defaults."""
        config = VXRAGSettings()
        assert config.embedding_model == "all-MiniLM-L6-v2"
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 200
    
    def test_valid_full_config(self) -> None:
        """Test valid full configuration."""
        config = VXRAGSettings(
            data_dir="./data",
            embedding_model="all-MiniLM-L6-v2",
            embedding_device="cpu",
            chunk_size=1024,
            chunk_overlap=200,
            vector_store="faiss"
        )
        assert config.data_dir == "./data"
        assert config.vector_store == "faiss"
    
    def test_chunk_overlap_validation(self) -> None:
        """Test root-level chunk overlap validation."""
        with pytest.raises(ValidationError) as exc_info:
            VXRAGSettings(chunk_size=512, chunk_overlap=512)
        assert "chunk_overlap" in str(exc_info.value).lower()
    
    def test_nested_config_validation(self) -> None:
        """Test nested configuration validation."""
        config = VXRAGSettings(
            faiss=FAISSConfig(hnsw_m=32, metric="inner_product"),
            mcp=MCPConfig(host="127.0.0.1", port=25191)
        )
        assert config.faiss.hnsw_m == 32
        assert config.mcp.port == 25191
    
    def test_invalid_nested_config(self) -> None:
        """Test invalid nested configuration is caught."""
        with pytest.raises(ValidationError) as exc_info:
            VXRAGSettings(
                faiss=FAISSConfig(hnsw_m=2)  # Too small
            )
        assert "hnsw_m" in str(exc_info.value)
    
    def test_extra_fields_allowed(self) -> None:
        """Test extra fields are allowed for forward compatibility."""
        config = VXRAGSettings(
            extra_field="extra_value"  # type: ignore
        )
        # Should not raise error due to extra="allow"
        assert hasattr(config, "extra_field")
