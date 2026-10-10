"""
Unit tests for configuration validation schemas.

Tests Pydantic models that validate configuration loaded from settings.yaml.
"""

import pytest
from pydantic import ValidationError

from src.utils.config_schemas import (
    AdaptiveChunkingProfile,
    ChunkingConfig,
    ContextAssemblerConfig,
    DuplicateDetectionConfig,
    EmbeddingConfig,
    IngestionConfig,
    MCPConfig,
    QdrantConfig,
    RateLimitConfig,
    RerankerConfig,
    RetrieverConfig,
    RouterConfig,
    VXRAGSettings,
)


class TestEmbeddingConfig:
    """Tests for EmbeddingConfig validation."""
    
    def test_valid_config(self) -> None:
        """Test valid embedding configuration."""
        config = EmbeddingConfig(
            embedding_model="BAAI/bge-m3",
            embedding_device="cpu",
            embedding_batch_size=10
        )
        assert config.embedding_model == "BAAI/bge-m3"
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

    def test_embedding_cache_size_config(self) -> None:
        """Test embedding_cache_size configuration in EmbeddingConfig."""
        config = EmbeddingConfig(embedding_cache_size=50000)
        assert config.embedding_cache_size == 50000


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


class TestQdrantConfig:
    """Tests for QdrantConfig validation."""

    def test_valid_default_config(self) -> None:
        """Test default Qdrant configuration."""
        config = QdrantConfig()
        assert config.collection_name == "vx_rag_collection"
        assert config.distance == "Cosine"
        assert config.enable_hybrid is True
        assert config.sparse_model == "BAAI/bge-m3"

    def test_custom_config(self) -> None:
        """Test custom Qdrant configuration."""
        config = QdrantConfig(
            collection_name="custom_col",
            path="./custom/path",
            distance="Dot",
            enable_hybrid=False,
            sparse_model="prithivida/Splade_PP_en_v1",
        )
        assert config.collection_name == "custom_col"
        assert config.path == "./custom/path"
        assert config.distance == "Dot"
        assert config.enable_hybrid is False
        assert config.sparse_model == "prithivida/Splade_PP_en_v1"



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
            model_name="BAAI/bge-reranker-v2-m3",
            top_k=5,
            device="cpu"
        )
        assert config.model_name == "BAAI/bge-reranker-v2-m3"
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


class TestVXRAGSettings:
    """Tests for complete VXRAGSettings validation."""
    
    def test_valid_minimal_config(self) -> None:
        """Test valid minimal configuration with defaults."""
        config = VXRAGSettings()
        assert config.embedding_model == "BAAI/bge-m3"
        assert config.embedding_dimensions == {"BAAI/bge-m3": 1024}
        assert config.reranker.model_name == "BAAI/bge-reranker-v2-m3"
        assert config.chunk_size == 1024
        assert config.chunk_overlap == 200
    
    def test_valid_full_config(self) -> None:
        """Test valid full configuration."""
        config = VXRAGSettings(
            data_dir="./data",
            embedding_model="BAAI/bge-m3",
            embedding_device="cpu",
            chunk_size=1024,
            chunk_overlap=200,
            vector_store="qdrant"
        )
        assert config.data_dir == "./data"
        assert config.embedding_model == "BAAI/bge-m3"
        assert config.vector_store == "qdrant"
    
    def test_chunk_overlap_validation(self) -> None:
        """Test root-level chunk overlap validation."""
        with pytest.raises(ValidationError) as exc_info:
            VXRAGSettings(chunk_size=512, chunk_overlap=512)
        assert "chunk_overlap" in str(exc_info.value).lower()
    
    def test_nested_config_validation(self) -> None:
        """Test nested configuration validation."""
        config = VXRAGSettings(
            qdrant=QdrantConfig(collection_name="test_col"),
            mcp=MCPConfig(host="127.0.0.1", port=25191)
        )
        assert config.qdrant.collection_name == "test_col"
        assert config.mcp.port == 25191
    
    def test_invalid_nested_config(self) -> None:
        """Test invalid nested configuration is caught."""
        with pytest.raises(ValidationError) as exc_info:
            VXRAGSettings(
                retriever=RetrieverConfig(semantic_top_k=0)  # Invalid: ge=1
            )
        assert "semantic_top_k" in str(exc_info.value)
    
    def test_extra_fields_allowed(self) -> None:
        """Test extra fields are allowed for forward compatibility."""
        config = VXRAGSettings(
            extra_field="extra_value"  # type: ignore
        )
        # Should not raise error due to extra="allow"
        assert hasattr(config, "extra_field")

    def test_reranker_threads_inheritance(self) -> None:
        """Test reranker threads inherits from top-level threads when not explicitly provided."""
        config = VXRAGSettings(threads=8)
        assert config.threads == 8
        assert config.reranker.threads == 8

    def test_new_sections_default_initialization(self) -> None:
        """Test context_assembler, router, ingestion are initialized by default."""
        config = VXRAGSettings()
        assert config.context_assembler.token_budget == 4096
        assert config.router.selector_type == "pydantic"
        assert config.ingestion.enable_caching is True


class TestContextAssemblerConfig:
    """Tests for ContextAssemblerConfig validation."""

    def test_default_config(self) -> None:
        """Test default values for ContextAssemblerConfig."""
        config = ContextAssemblerConfig()
        assert config.token_budget == 4096
        assert config.model_name == "gpt-3.5-turbo"
        assert config.max_items is None
        assert config.min_score is None

    def test_invalid_token_budget(self) -> None:
        """Test token_budget must be positive."""
        with pytest.raises(ValidationError):
            ContextAssemblerConfig(token_budget=0)


class TestRouterConfig:
    """Tests for RouterConfig validation."""

    def test_default_config(self) -> None:
        """Test default values for RouterConfig."""
        config = RouterConfig()
        assert config.selector_type == "pydantic"
        assert config.use_multi_select is False
        assert config.verbose is False

    def test_invalid_selector_type(self) -> None:
        """Test selector_type validation."""
        with pytest.raises(ValidationError):
            RouterConfig(selector_type="invalid")  # type: ignore


class TestIngestionConfig:
    """Tests for IngestionConfig validation."""

    def test_default_config(self) -> None:
        """Test default values for IngestionConfig."""
        config = IngestionConfig()
        assert config.default_chunk_size == 1024
        assert config.code_chunk_size == 512
        assert config.chunk_overlap == 128
        assert config.overlap_ratio_min == 0.10
        assert config.overlap_ratio_max == 0.15
        assert config.max_unbroken_string_length == 1000
        assert config.enable_caching is True
        assert config.enable_metadata_extraction is False
        assert config.enable_embedding is True
        assert config.enable_vector_store is True
        assert config.enable_persistence is True

    def test_chunk_size_alias(self) -> None:
        """Test that chunk_size acts as an alias for default_chunk_size."""
        config = IngestionConfig(chunk_size=2048, chunk_overlap=256)  # type: ignore[call-arg]
        assert config.default_chunk_size == 2048
        assert config.chunk_overlap == 256

    def test_invalid_overlap_raises(self) -> None:
        """Test that chunk_overlap >= default_chunk_size raises ValueError."""
        with pytest.raises(ValueError, match="chunk_overlap"):
            IngestionConfig(default_chunk_size=512, chunk_overlap=512)

    def test_invalid_ratio_bounds_raises(self) -> None:
        """Test that overlap_ratio_min > overlap_ratio_max raises ValueError."""
        with pytest.raises(ValueError, match="overlap_ratio_min"):
            IngestionConfig(overlap_ratio_min=0.20, overlap_ratio_max=0.10)

    def test_null_ingestion_input(self) -> None:
        """Test that null ingestion input defaults safely."""
        config = IngestionConfig.model_validate(None)
        assert config.default_chunk_size == 1024
        assert config.chunk_overlap == 128

        settings = VXRAGSettings.model_validate({"ingestion": None})
        assert settings.ingestion.default_chunk_size == 1024
        assert settings.ingestion.chunk_overlap == 128

