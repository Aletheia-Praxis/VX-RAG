"""
Global configuration loader for VX-RAG.

Provides utilities to load configuration from settings.yaml.
This centralizes all configuration loading to avoid hardcoded values throughout the codebase.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_settings(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load complete settings.yaml configuration.
    
    Args:
        config_path: Path to settings.yaml file. If None, uses default location.
        
    Returns:
        Dictionary with complete configuration from settings.yaml
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid
    """
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
    
    logger.info(f"Loaded configuration from {config_file_path}")
    return config


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
        return config.get('embedding_model', 'all-MiniLM-L6-v2')
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
        return config.get('vector_store', 'faiss')
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
    """
    config = load_settings(config_path)
    
    data_dirs = {
        'data_dir': config.get('data_dir', './data'),
        'raw_data_dir': config.get('raw_data_dir', './data/raw'),
        'processed_data_dir': config.get('processed_data_dir', './data/processed'),
        'index_dir': config.get('index_dir', './data/index')
    }
    
    logger.info(f"Loaded data directories: index_dir={data_dirs['index_dir']}")
    return data_dirs
