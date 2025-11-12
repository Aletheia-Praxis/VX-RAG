"""
Configuration validation schemas using Pydantic.

This module defines Pydantic models for validating configuration loaded from settings.yaml.
All configuration must pass validation before being used by the application.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, model_validator


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model name")
    embedding_device: Literal["cpu", "cuda"] = Field(default="cpu", description="Device for embedding")
    embedding_batch_size: int = Field(default=10, ge=1, le=1000, description="Batch size for embeddings")
    embedding_cache_size: int = Field(default=1000, ge=0, description="Cache size for embeddings")
    embedding_trust_remote_code: bool = Field(default=False, description="Trust remote code in models")


class AdaptiveChunkingProfile(BaseModel):
    """Configuration for an adaptive chunking profile."""
    
    chunk_size: int = Field(ge=128, le=4096, description="Chunk size in tokens")
    chunk_overlap: int = Field(ge=0, description="Overlap size in tokens")
    min_lines: Optional[int] = Field(default=None, ge=1, description="Minimum lines threshold")
    max_lines: Optional[int] = Field(default=None, ge=1, description="Maximum lines threshold")
    preserve_integrity: Optional[bool] = Field(default=None, description="Preserve content integrity")
    preserve_structure: Optional[bool] = Field(default=None, description="Preserve content structure")
    
    @model_validator(mode='after')
    def validate_overlap_size(self) -> 'AdaptiveChunkingProfile':
        """Ensure overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")
        return self


class AdaptiveChunkingConfig(BaseModel):
    """Adaptive chunking configuration."""
    
    enabled: bool = Field(default=True, description="Enable adaptive chunking")
    max_chunk_size: int = Field(default=2000, ge=512, le=8192, description="Maximum chunk size")
    large_code_blocks: Optional[AdaptiveChunkingProfile] = None
    tables: Optional[AdaptiveChunkingProfile] = None
    short_snippets: Optional[AdaptiveChunkingProfile] = None
    hex_dumps: Optional[AdaptiveChunkingProfile] = None


class ChunkingConfig(BaseModel):
    """Chunking configuration."""
    
    chunk_size: int = Field(default=1024, ge=128, le=4096, description="Default chunk size")
    chunk_overlap: int = Field(default=200, ge=0, description="Default chunk overlap")
    adaptive_chunking: AdaptiveChunkingConfig = Field(default_factory=AdaptiveChunkingConfig)
    
    @model_validator(mode='after')
    def validate_overlap_size(self) -> 'ChunkingConfig':
        """Ensure overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})")
        return self


class FAISSConfig(BaseModel):
    """FAISS index configuration."""
    
    hnsw_m: int = Field(default=32, ge=4, le=128, description="HNSW neighbors count")
    metric: Literal["inner_product", "L2"] = Field(default="inner_product", description="Similarity metric")


class BM25Config(BaseModel):
    """BM25 configuration."""
    
    index_dir: str = Field(default="./data/index/bm25", description="BM25 index directory")
    similarity_top_k: int = Field(default=20, ge=1, le=100, description="Top K for retrieval")
    enable_persistence: bool = Field(default=True, description="Enable index persistence")


class RetrieverConfig(BaseModel):
    """Retriever configuration."""
    
    semantic_top_k: int = Field(default=20, ge=1, le=100, description="Semantic search top K")
    hybrid_alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Hybrid search alpha")
    metadata_filters: List[str] = Field(default_factory=list, description="Metadata filters")
    enable_hybrid: bool = Field(default=True, description="Enable hybrid search")


class RerankerConfig(BaseModel):
    """Reranker configuration."""
    
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="Reranker model")
    top_k: int = Field(default=5, ge=1, le=50, description="Top K after reranking")
    device: Literal["cpu", "cuda"] = Field(default="cpu", description="Device for reranking")
    metadata_boost: float = Field(default=0.1, ge=0.0, le=1.0, description="Metadata boost factor")
    enable_metadata_prioritization: bool = Field(default=True, description="Enable metadata prioritization")


class ContextAssemblerConfig(BaseModel):
    """Context assembler configuration."""
    
    token_budget: int = Field(default=2048, ge=512, le=32000, description="Token budget")
    model_name: str = Field(default="gpt-3.5-turbo", description="Model for token counting")
    max_items: Optional[int] = Field(default=None, ge=1, description="Max context items")
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Min relevance score")


class RateLimitConfig(BaseModel):
    """Rate limiting configuration."""
    
    max_concurrent: int = Field(default=2, ge=1, le=100, description="Max concurrent requests")
    queue_size: int = Field(default=10, ge=1, le=1000, description="Queue size")
    default_timeout: float = Field(default=600.0, ge=1.0, le=3600.0, description="Default timeout (seconds)")


class MCPTimeoutsConfig(BaseModel):
    """MCP tool-specific timeouts."""
    
    query_knowledge_base: float = Field(default=600.0, ge=1.0, le=3600.0, description="Query timeout")
    search_documents: float = Field(default=300.0, ge=1.0, le=1800.0, description="Search timeout")
    health_check: float = Field(default=30.0, ge=1.0, le=300.0, description="Health check timeout")
    system_context: float = Field(default=10.0, ge=1.0, le=60.0, description="System context timeout")


class MCPDefaultsConfig(BaseModel):
    """MCP tool default parameters."""
    
    top_k: int = Field(default=5, ge=1, le=50, description="Default top K")
    token_budget: int = Field(default=4000, ge=512, le=32000, description="Default token budget")
    search_top_k: int = Field(default=10, ge=1, le=100, description="Default search top K")
    apply_redaction: bool = Field(default=True, description="Enable redaction")


class MCPConfig(BaseModel):
    """MCP server configuration."""
    
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=25191, ge=1024, le=65535, description="Server port")
    debug: bool = Field(default=True, description="Debug mode")
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    timeouts: MCPTimeoutsConfig = Field(default_factory=MCPTimeoutsConfig)
    defaults: MCPDefaultsConfig = Field(default_factory=MCPDefaultsConfig)


class DuplicateDetectionConfig(BaseModel):
    """Duplicate detection configuration."""
    
    similarity_threshold: float = Field(default=0.95, ge=0.0, le=1.0, description="Similarity threshold")
    hash_algorithm: Literal["sha256", "md5", "sha1"] = Field(default="sha256", description="Hash algorithm")


class PaddleOCRConfig(BaseModel):
    """PaddleOCR configuration."""
    
    enabled: bool = Field(default=True, description="Enable PaddleOCR")
    lang: str = Field(default="en", description="OCR language")
    use_gpu: bool = Field(default=False, description="Use GPU")
    use_angle_cls: bool = Field(default=True, description="Use angle classification")
    show_log: bool = Field(default=False, description="Show logs")
    det_model_dir: Optional[str] = Field(default=None, description="Detection model directory")
    rec_model_dir: Optional[str] = Field(default=None, description="Recognition model directory")
    cls_model_dir: Optional[str] = Field(default=None, description="Classification model directory")
    use_space_char: bool = Field(default=True, description="Recognize spaces")
    enable_mkldnn: bool = Field(default=False, description="Enable MKLDNN")
    cpu_threads: int = Field(default=4, ge=1, le=64, description="CPU threads")
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Min confidence")
    save_extracted_images: bool = Field(default=True, description="Save extracted images")
    extracted_images_dir: str = Field(default="./data/extracted_images", description="Extracted images directory")
    replace_image_placeholders: bool = Field(default=True, description="Replace image placeholders")


class BoilerplatePatterns(BaseModel):
    """Boilerplate removal patterns configuration."""
    
    remove_html_comments: bool = Field(default=True, description="Remove HTML comments")
    remove_blog_metadata: bool = Field(default=True, description="Remove blog metadata")
    remove_footer_timestamps: bool = Field(default=True, description="Remove footer timestamps")
    remove_navigation: bool = Field(default=True, description="Remove navigation")
    remove_social_sharing: bool = Field(default=True, description="Remove social sharing")
    preserve_code_blocks: bool = Field(default=True, description="Preserve code blocks")
    preserve_markdown_structure: bool = Field(default=True, description="Preserve markdown structure")


class BoilerplateRemovalConfig(BaseModel):
    """Boilerplate removal configuration."""
    
    enabled: bool = Field(default=True, description="Enable boilerplate removal")
    aggressive_mode: bool = Field(default=True, description="Use aggressive mode")
    position: Literal["after_ocr", "before_normalization"] = Field(default="after_ocr", description="Application position")
    patterns: BoilerplatePatterns = Field(default_factory=BoilerplatePatterns)


class DataDirectoriesConfig(BaseModel):
    """Data directories configuration."""
    
    data_dir: str = Field(default="./data", description="Root data directory")
    raw_data_dir: str = Field(default="./data/raw", description="Raw data directory")
    processed_data_dir: str = Field(default="./data/processed", description="Processed data directory")
    index_dir: str = Field(default="./data/index", description="Index directory")
    snapshots_dir: str = Field(default="./data/snapshots", description="Snapshots directory")


class LoggingConfig(BaseModel):
    """Logging configuration."""
    
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(default="INFO", description="Log level")
    log_file: str = Field(default="logs/vx_rag.log", description="Log file path")
    log_format: Literal["json", "text"] = Field(default="json", description="Log format")
    log_structured_events: bool = Field(default=True, description="Log structured events")
    log_metrics: bool = Field(default=True, description="Log metrics")


class MetricsConfig(BaseModel):
    """Metrics configuration."""
    
    metrics_enabled: bool = Field(default=True, description="Enable metrics")
    metrics_log_interval: int = Field(default=300, ge=60, le=3600, description="Metrics log interval (seconds)")
    metrics_file: str = Field(default="metrics.log", description="Metrics file")


class VXRAGSettings(BaseModel):
    """Complete VX-RAG configuration settings."""
    
    # Data directories
    data_dir: str = Field(default="./data")
    raw_data_dir: str = Field(default="./data/raw")
    processed_data_dir: str = Field(default="./data/processed")
    index_dir: str = Field(default="./data/index")
    snapshots_dir: str = Field(default="./data/snapshots")
    
    # Embedding
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_device: Literal["cpu", "cuda"] = Field(default="cpu")
    embedding_batch_size: int = Field(default=10, ge=1, le=1000)
    embedding_cache_size: int = Field(default=1000, ge=0)
    embedding_trust_remote_code: bool = Field(default=False)
    
    # Chunking
    chunk_size: int = Field(default=1024, ge=128, le=4096)
    chunk_overlap: int = Field(default=200, ge=0)
    vector_store: str = Field(default="faiss")
    
    # Configuration sections
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    adaptive_chunking: AdaptiveChunkingConfig = Field(default_factory=AdaptiveChunkingConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    bm25: BM25Config = Field(default_factory=BM25Config)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    context_assembler: ContextAssemblerConfig = Field(default_factory=ContextAssemblerConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    duplicate_detection: DuplicateDetectionConfig = Field(default_factory=DuplicateDetectionConfig)
    paddle_ocr: PaddleOCRConfig = Field(default_factory=PaddleOCRConfig)
    boilerplate_removal: BoilerplateRemovalConfig = Field(default_factory=BoilerplateRemovalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Metrics
    metrics_enabled: bool = Field(default=True)
    metrics_log_interval: int = Field(default=300, ge=60, le=3600)
    metrics_file: str = Field(default="metrics.log")
    
    # Query configuration
    similarity_top_k: int = Field(default=5, ge=1, le=100)
    query_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    
    @model_validator(mode='after')
    def validate_chunk_overlap(self) -> 'VXRAGSettings':
        """Ensure chunk overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )
        return self
    
    model_config = {
        "extra": "allow",  # Allow extra fields for forward compatibility
        "validate_assignment": True,  # Validate on assignment
    }
