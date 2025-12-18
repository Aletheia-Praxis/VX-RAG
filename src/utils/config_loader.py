"""
Global configuration loader for VX-RAG.

Provides utilities to load configuration from settings.yaml with Pydantic validation.
This centralizes all configuration loading to avoid hardcoded values throughout the codebase.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import ValidationError

from .config_schemas import VXRAGSettings

logger = logging.getLogger(__name__)

# Global validated settings instance
_validated_settings: Optional[VXRAGSettings] = None


def load_settings(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and validate complete settings.yaml configuration using Pydantic.
    
    Args:
        config_path: Path to settings.yaml file. If None, uses default location.
        
    Returns:
        Dictionary with complete validated configuration from settings.yaml
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid or fails validation
    """
    global _validated_settings
    
    config_file_path: Path
    if config_path is None:
        # Default path relative to project root
        project_root = Path(__file__).parent.parent.parent
        config_file_path = project_root / "config" / "settings.yaml"
    else:
        config_file_path = Path(config_path)
    
    if not config_file_path.exists():
        logger.error(f"Config file not found: {config_file_path}")
        raise FileNotFoundError(f"Configuration file not found: {config_file_path}")
    
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file: {e}")
        raise ValueError(f"Invalid YAML in config file: {e}")
    
    if not isinstance(config, dict):
        logger.error("Config root must be a dictionary")
        raise ValueError("Invalid configuration: root must be a dictionary")
    
    # Validate configuration using Pydantic
    try:
        _validated_settings = VXRAGSettings(**config)
        logger.info(f"Configuration validated successfully from {config_file_path}")
        return _validated_settings.model_dump()
    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise ValueError(f"Configuration validation errors:\n{e}")


def get_validated_settings() -> VXRAGSettings:
    """
    Get the current validated settings instance.
    
    Returns:
        Validated VXRAGSettings instance
        
    Raises:
        RuntimeError: If settings have not been loaded yet
    """
    global _validated_settings
    
    if _validated_settings is None:
        # Auto-load settings on first access
        load_settings()
    
    if _validated_settings is None:
        raise RuntimeError("Settings not loaded. Call load_settings() first.")
    
    return _validated_settings


def load_chunking_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load chunking configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file. If None, uses default location.
        
    Returns:
        Dictionary with chunking configuration including:
        - chunk_size: Default chunk size in tokens
        - chunk_overlap: Default overlap in tokens
        - adaptive_chunking: Adaptive configuration dict
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid
    """
    config = load_settings(config_path)
    
    # Extract chunking-related settings
    chunk_size = config.get('chunk_size')
    chunk_overlap = config.get('chunk_overlap')
    adaptive_chunking = config.get('adaptive_chunking', {})
    
    if chunk_size is None:
        logger.warning("chunk_size not found in config, using default 1024")
        chunk_size = 1024
    
    if chunk_overlap is None:
        logger.warning("chunk_overlap not found in config, using default 200")
        chunk_overlap = 200
    
    # Validate adaptive_chunking structure
    if not isinstance(adaptive_chunking, dict):
        logger.warning("adaptive_chunking is not a dict, using empty config")
        adaptive_chunking = {'enabled': False}
    
    chunking_config = {
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap,
        'adaptive_chunking': adaptive_chunking
    }
    
    logger.info(
        f"Loaded chunking config: chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}, "
        f"adaptive_enabled={adaptive_chunking.get('enabled', False)}"
    )
    
    return chunking_config


def get_default_chunker_params(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default parameters for Chunker initialization from config.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with Chunker initialization parameters:
        - chunk_size
        - chunk_overlap
        - adaptive_config
        - use_semantic_chunking (default: True)
        - use_hierarchical_chunking (default: False)
    """
    config = load_chunking_config(config_path)
    
    return {
        'chunk_size': config['chunk_size'],
        'chunk_overlap': config['chunk_overlap'],
        'adaptive_config': config['adaptive_chunking'],
        'use_semantic_chunking': True,      # Default behavior
        'use_hierarchical_chunking': False  # Default behavior
    }


def get_chunking_metadata(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get chunking metadata for index snapshots and storage.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with chunking metadata for snapshots:
        - adaptive_chunking: Whether adaptive chunking is enabled
        - default_chunk_size: Default chunk size
        - chunk_overlap: Default overlap size
        - adaptive_profiles: Names of available adaptive profiles (if enabled)
    """
    config = load_chunking_config(config_path)
    
    adaptive_config = config['adaptive_chunking']
    metadata = {
        'adaptive_chunking': adaptive_config.get('enabled', False),
        'default_chunk_size': config['chunk_size'],
        'chunk_overlap': config['chunk_overlap']
    }
    
    # Add profile names if adaptive chunking is enabled
    if metadata['adaptive_chunking']:
        profiles = []
        for key in adaptive_config.keys():
            if key != 'enabled' and isinstance(adaptive_config[key], dict):
                profiles.append(key)
        if profiles:
            metadata['adaptive_profiles'] = profiles
    
    return metadata


def get_embedding_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get embedding model configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with embedding configuration:
        - embedding_model: Model name/identifier
        - embedding_device: Device to use (cpu/cuda)
        - embedding_batch_size: Batch size for embeddings
        - embedding_cache_size: Cache size for embeddings
        - embedding_trust_remote_code: Whether to trust remote code
    """
    config = load_settings(config_path)
    
    embedding_model = config.get('embedding_model', 'all-MiniLM-L6-v2')
    embedding_device = config.get('embedding_device', 'cpu')
    embedding_batch_size = config.get('embedding_batch_size', 10)
    embedding_cache_size = config.get('embedding_cache_size', 1000)
    embedding_trust_remote_code = config.get('embedding_trust_remote_code', False)
    
    embedding_config = {
        'embedding_model': embedding_model,
        'embedding_device': embedding_device,
        'embedding_batch_size': embedding_batch_size,
        'embedding_cache_size': embedding_cache_size,
        'embedding_trust_remote_code': embedding_trust_remote_code
    }
    
    logger.info(
        f"Loaded embedding config: model={embedding_model}, "
        f"device={embedding_device}, batch_size={embedding_batch_size}"
    )
    return embedding_config


def get_embedding_model_name(config_path: Optional[str] = None) -> str:
    """
    Get embedding model name from settings.yaml.
    
    Simple helper to get just the model name string.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Embedding model name string
    """
    try:
        config = load_settings(config_path)
        return str(config.get('embedding_model', 'all-MiniLM-L6-v2'))
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Failed to load config, using default: {e}")
        return 'all-MiniLM-L6-v2'


def get_vector_store_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get vector store configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with vector store configuration:
        - vector_store: Store type (e.g., 'faiss', 'chroma')
        - Additional vector store settings
    """
    config = load_settings(config_path)
    
    vector_store = config.get('vector_store', 'faiss')
    
    vector_config = {
        'vector_store': vector_store
    }
    
    logger.info(f"Loaded vector store config: type={vector_store}")
    return vector_config


def get_vector_store_type(config_path: Optional[str] = None) -> str:
    """
    Get vector store type from settings.yaml.
    
    Simple helper to get just the store type string.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Vector store type string (e.g., 'faiss', 'chroma')
    """
    try:
        config = load_settings(config_path)
        return str(config.get('vector_store', 'faiss'))
    except (FileNotFoundError, ValueError) as e:
        logger.warning(f"Failed to load config, using default: {e}")
        return 'faiss'


def get_reranker_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get reranker configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with reranker configuration:
        - model_name: Cross-encoder model name
        - top_k: Number of documents after reranking
        - device: Device to use (cpu/cuda)
        - metadata_boost: Boost factor for metadata matches
        - enable_metadata_prioritization: Whether to enable metadata prioritization
    """
    config = load_settings(config_path)
    
    reranker_section = config.get('reranker', {})
    
    reranker_config = {
        'model_name': reranker_section.get('model_name', 'cross-encoder/ms-marco-MiniLM-L-6-v2'),
        'top_k': reranker_section.get('top_k', 5),
        'device': reranker_section.get('device', 'cpu'),
        'metadata_boost': reranker_section.get('metadata_boost', 0.1),
        'enable_metadata_prioritization': reranker_section.get('enable_metadata_prioritization', True)
    }
    
    logger.info(f"Loaded reranker config: model={reranker_config['model_name']}, top_k={reranker_config['top_k']}")
    return reranker_config


def get_context_assembler_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get context assembler configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with context assembler configuration:
        - token_budget: Default token budget for context
        - model_name: Model name for token counting
        - max_items: Maximum number of context items
        - min_score: Minimum relevance score
    """
    config = load_settings(config_path)
    
    assembler_section = config.get('context_assembler', {})
    
    assembler_config = {
        'token_budget': assembler_section.get('token_budget', 2048),
        'model_name': assembler_section.get('model_name', 'gpt-3.5-turbo'),
        'max_items': assembler_section.get('max_items'),
        'min_score': assembler_section.get('min_score')
    }
    
    logger.info(
        f"Loaded context assembler config: token_budget={assembler_config['token_budget']}, "
        f"model={assembler_config['model_name']}"
    )
    return assembler_config


def get_retriever_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get retriever configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with retriever configuration:
        - semantic_top_k: Number of documents to retrieve semantically
        - hybrid_alpha: Weight for hybrid search
        - metadata_filters: List of metadata filters
        - enable_hybrid: Whether to enable hybrid search
    """
    config = load_settings(config_path)
    
    retriever_section = config.get('retriever', {})
    
    retriever_config = {
        'semantic_top_k': retriever_section.get('semantic_top_k', 20),
        'hybrid_alpha': retriever_section.get('hybrid_alpha', 0.5),
        'metadata_filters': retriever_section.get('metadata_filters', []),
        'enable_hybrid': retriever_section.get('enable_hybrid', True)
    }
    
    logger.info(
        f"Loaded retriever config: semantic_top_k={retriever_config['semantic_top_k']}, "
        f"enable_hybrid={retriever_config['enable_hybrid']}"
    )
    return retriever_config


def get_bm25_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get BM25 configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with BM25 configuration:
        - index_dir: Directory for BM25 index
        - similarity_top_k: Default for initial retrieval
        - enable_persistence: Whether to enable persistence
    """
    config = load_settings(config_path)
    
    bm25_section = config.get('bm25', {})
    
    bm25_config = {
        'index_dir': bm25_section.get('index_dir', './data/index/bm25'),
        'similarity_top_k': bm25_section.get('similarity_top_k', 20),
        'enable_persistence': bm25_section.get('enable_persistence', True)
    }
    
    logger.info(f"Loaded BM25 config: index_dir={bm25_config['index_dir']}, similarity_top_k={bm25_config['similarity_top_k']}")
    return bm25_config


def get_data_directories(config_path: Optional[str] = None) -> Dict[str, str]:
    """
    Get data directory paths from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with data directory paths:
        - data_dir: Root data directory
        - raw_data_dir: Raw data directory
        - processed_data_dir: Processed data directory
        - index_dir: Index directory
        - snapshots_dir: Snapshots directory
    """
    config = load_settings(config_path)
    
    data_dirs = {
        'data_dir': config.get('data_dir', './data'),
        'raw_data_dir': config.get('raw_data_dir', './data/raw'),
        'processed_data_dir': config.get('processed_data_dir', './data/processed'),
        'index_dir': config.get('index_dir', './data/index'),
        'snapshots_dir': config.get('snapshots_dir', './data/snapshots')
    }
    
    logger.info(f"Loaded data directories: index_dir={data_dirs['index_dir']}")
    return data_dirs


def get_mcp_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get complete MCP server configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with MCP configuration:
        - host: Server host address
        - port: Server port number
        - debug: Debug mode flag
        - rate_limit: Rate limiting configuration
        - timeouts: Tool-specific timeout configuration
        - defaults: Default parameters for tools
    """
    config = load_settings(config_path)
    
    mcp_section = config.get('mcp', {})
    
    mcp_config = {
        'host': mcp_section.get('host', '127.0.0.1'),
        'port': mcp_section.get('port', 25191),
        'debug': mcp_section.get('debug', True),
        'rate_limit': mcp_section.get('rate_limit', {
            'max_concurrent': 2,
            'queue_size': 10,
            'default_timeout': 600.0
        }),
        'timeouts': mcp_section.get('timeouts', {
            'query_knowledge_base': 600.0,
            'search_documents': 300.0,
            'health_check': 30.0,
            'system_context': 10.0
        }),
        'defaults': mcp_section.get('defaults', {
            'top_k': 5,
            'token_budget': 4000,
            'search_top_k': 10,
            'apply_redaction': True
        })
    }
    
    logger.info(
        f"Loaded MCP config: host={mcp_config['host']}, "
        f"port={mcp_config['port']}, "
        f"max_concurrent={mcp_config['rate_limit']['max_concurrent']}"
    )
    return mcp_config


def get_mcp_rate_limit_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get MCP rate limiting configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with rate limiting configuration:
        - max_concurrent: Maximum concurrent requests
        - queue_size: Queue size for pending requests
        - default_timeout: Default timeout in seconds
    """
    mcp_config = get_mcp_config(config_path)
    rate_limit: Dict[str, Any] = mcp_config['rate_limit']
    return rate_limit


def get_mcp_timeouts(config_path: Optional[str] = None) -> Dict[str, float]:
    """
    Get MCP tool-specific timeouts from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary mapping tool names to timeout values (seconds):
        - query_knowledge_base: Timeout for knowledge base queries
        - search_documents: Timeout for document searches
        - health_check: Timeout for health checks
        - system_context: Timeout for system context requests
    """
    mcp_config = get_mcp_config(config_path)
    timeouts: Dict[str, float] = mcp_config['timeouts']
    return timeouts


def get_mcp_defaults(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get MCP tool default parameters from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with default parameters:
        - top_k: Default number of results for knowledge base queries
        - token_budget: Default token budget for context assembly
        - search_top_k: Default number of results for document search
        - apply_redaction: Enable sensitive data redaction in responses
    """
    mcp_config = get_mcp_config(config_path)
    defaults: Dict[str, Any] = mcp_config['defaults']
    return defaults


def get_faiss_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get FAISS index configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with FAISS configuration:
        - hnsw_m: Number of neighbors for HNSW graph
        - metric: Similarity metric (inner_product, L2)
    """
    config = load_settings(config_path)
    
    faiss_section = config.get('faiss', {})
    
    faiss_config = {
        'hnsw_m': faiss_section.get('hnsw_m', 32),
        'metric': faiss_section.get('metric', 'inner_product')
    }
    
    logger.info(f"Loaded FAISS config: hnsw_m={faiss_config['hnsw_m']}, metric={faiss_config['metric']}")
    return faiss_config


def get_hierarchical_chunker_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get hierarchical chunker configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with hierarchical chunker configuration:
        - max_chunk_size: Maximum chunk size
        - preserve_code_blocks: Whether to preserve code blocks
        - preserve_tables: Whether to preserve tables
    """
    config = load_settings(config_path)
    
    adaptive_section = config.get('adaptive_chunking', {})
    
    chunker_config = {
        'max_chunk_size': adaptive_section.get('max_chunk_size', 2000),
        'preserve_code_blocks': adaptive_section.get('large_code_blocks', {}).get('preserve_integrity', True),
        'preserve_tables': adaptive_section.get('tables', {}).get('preserve_integrity', True)
    }
    
    logger.info(f"Loaded hierarchical chunker config: max_chunk_size={chunker_config['max_chunk_size']}")
    return chunker_config


def get_duplicate_detection_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get duplicate detection configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with duplicate detection configuration:
        - similarity_threshold: Threshold for near-duplicate detection (0.0-1.0)
        - hash_algorithm: Hash algorithm for content hashing
    """
    config = load_settings(config_path)
    
    duplicate_section = config.get('duplicate_detection', {})
    
    duplicate_config = {
        'similarity_threshold': duplicate_section.get('similarity_threshold', 0.95),
        'hash_algorithm': duplicate_section.get('hash_algorithm', 'sha256')
    }
    
    logger.info(f"Loaded duplicate detection config: similarity_threshold={duplicate_config['similarity_threshold']}, hash_algorithm={duplicate_config['hash_algorithm']}")
    return duplicate_config


def get_ingestion_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get ingestion pipeline configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with ingestion configuration:
        - chunk_size: Default chunk size for text splitting
        - chunk_overlap: Default chunk overlap
        - enable_caching: Enable pipeline caching
        - enable_metadata_extraction: Enable metadata extraction
        - enable_persistence: Enable pipeline persistence
    """
    config = load_settings(config_path)
    
    ingestion_section = config.get('ingestion', {})
    
    ingestion_config = {
        'chunk_size': config.get('chunk_size', 1024),
        'chunk_overlap': config.get('chunk_overlap', 200),
        'enable_caching': ingestion_section.get('enable_caching', True),
        'enable_metadata_extraction': ingestion_section.get('enable_metadata_extraction', False),
        'enable_persistence': ingestion_section.get('enable_persistence', True),
        'enable_embedding': ingestion_section.get('enable_embedding', False),
        'enable_vector_store': ingestion_section.get('enable_vector_store', False),
        'embedding_model': config.get('embedding_model', 'all-MiniLM-L6-v2'),
        'faiss_index': ingestion_section.get('faiss_index'),
        'vector_store_kwargs': ingestion_section.get('vector_store_kwargs', {})
    }
    
    logger.info(f"Loaded ingestion config: chunk_size={ingestion_config['chunk_size']}, enable_caching={ingestion_config['enable_caching']}")
    return ingestion_config


def get_router_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get router query engine configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with router configuration:
        - selector_type: Selector type ('pydantic' or 'llm')
        - use_multi_select: Whether to allow multiple tool selection
        - verbose: Enable verbose logging for routing decisions
    """
    config = load_settings(config_path)
    
    router_section = config.get('router', {})
    
    router_config = {
        'selector_type': router_section.get('selector_type', 'pydantic'),
        'use_multi_select': router_section.get('use_multi_select', False),
        'verbose': router_section.get('verbose', False)
    }
    
    logger.info(f"Loaded router config: selector_type={router_config['selector_type']}, verbose={router_config['verbose']}")
    return router_config


def get_boilerplate_removal_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get boilerplate removal configuration from settings.yaml.
    
    Args:
        config_path: Path to settings.yaml file
        
    Returns:
        Dictionary with boilerplate removal configuration:
        - enabled: Enable boilerplate removal
        - aggressive_mode: Use aggressive mode
        - position: Application position ('after_ocr' or 'before_normalization')
        - patterns: Dictionary with pattern settings
    """
    config = load_settings(config_path)
    
    boilerplate_section = config.get('boilerplate_removal', {})
    
    boilerplate_config = {
        'enabled': boilerplate_section.get('enabled', True),
        'aggressive_mode': boilerplate_section.get('aggressive_mode', True),
        'position': boilerplate_section.get('position', 'after_ocr'),
        'patterns': boilerplate_section.get('patterns', {
            'remove_html_comments': True,
            'remove_blog_metadata': True,
            'remove_footer_timestamps': True,
            'remove_navigation': True,
            'remove_social_sharing': True,
            'preserve_code_blocks': True,
            'preserve_markdown_structure': True
        })
    }
    
    logger.info(f"Loaded boilerplate removal config: enabled={boilerplate_config['enabled']}, aggressive_mode={boilerplate_config['aggressive_mode']}")
    return boilerplate_config
