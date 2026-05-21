"""
LlamaIndex integration utilities for VX-RAG.

Provides adapters and helpers for seamless integration between
VX-RAG components and LlamaIndex ecosystem.
"""

from src.utils.logging_config import get_logger
from typing import List, Dict, Any, Optional
from llama_index.core.schema import NodeWithScore, BaseNode, Document as LlamaDocument
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
import tiktoken

logger = get_logger(__name__)


class VXRAGLlamaIndexAdapter:
    """
    Adapter for integrating VX-RAG with LlamaIndex global infrastructure.
    
    DEPRECATED: Use standalone functions instead.
    """
    pass


# Convenience functions for common operations
def setup_vxrag_llamaindex_integration(
    model_name: str = "gpt-3.5-turbo",
    verbose: bool = False
) -> TokenCountingHandler:
    """
    One-line setup for VX-RAG + LlamaIndex integration.
    
    Configures global token counting and returns handler for direct access.
    
    Args:
        model_name: Model name for tokenizer
        verbose: If True, prints token usage to console
        
    Returns:
        TokenCountingHandler instance
    """
    # Ensure global counter is registered and return the handler
    return ensure_global_token_counter(model_name=model_name, verbose=verbose)


def get_token_stats() -> Dict[str, Any]:
    """
    Get global token statistics from LlamaIndex CallbackManager.
    
    Returns:
        Dictionary with token counts, or empty dict if no token counter found
    """
    if Settings.callback_manager is None:
        logger.warning("No global CallbackManager configured")
        return {}
    
    # Find TokenCountingHandler in handlers
    for handler in Settings.callback_manager.handlers:
        if isinstance(handler, TokenCountingHandler):
            return {
                'total_embedding_tokens': handler.total_embedding_token_count,
                'prompt_llm_tokens': handler.prompt_llm_token_count,
                'completion_llm_tokens': handler.completion_llm_token_count,
                'total_llm_tokens': handler.total_llm_token_count,
                'llm_event_count': len(handler.llm_token_counts),
                'embedding_event_count': len(handler.embedding_token_counts)
            }
    
    logger.warning("No TokenCountingHandler found in CallbackManager")
    return {}


def reset_token_counts() -> None:
    """Reset all global token counters in LlamaIndex CallbackManager."""
    if Settings.callback_manager is None:
        logger.warning("No global CallbackManager configured")
        return
    
    for handler in Settings.callback_manager.handlers:
        if isinstance(handler, TokenCountingHandler):
            handler.reset_counts()
            logger.debug("Reset global LlamaIndex token counts")
            return
    
    logger.warning("No TokenCountingHandler found to reset")


def get_global_token_counter() -> Optional[TokenCountingHandler]:
    """
    Return the TokenCountingHandler currently registered in LlamaIndex Settings.callback_manager.

    Returns:
        TokenCountingHandler | None
    """
    if Settings.callback_manager is None:
        return None

    for handler in Settings.callback_manager.handlers:
        if isinstance(handler, TokenCountingHandler):
            return handler
    return None


def ensure_global_token_counter(model_name: str = "gpt-3.5-turbo", verbose: bool = False) -> TokenCountingHandler:
    """
    Ensure that a TokenCountingHandler is registered in the global LlamaIndex CallbackManager.
    Creates and registers a TokenCountingHandler if none exists and returns it.

    Args:
        model_name: Model name for tokenizer
        verbose: If True, prints token usage to console

    Returns:
        Configured TokenCountingHandler
    """
    token_counter = get_global_token_counter()
    if token_counter is not None:
        return token_counter

    try:
        tokenizer_fn = tiktoken.encoding_for_model(model_name).encode
    except KeyError:
        tokenizer_fn = tiktoken.get_encoding("cl100k_base").encode
        logger.warning(f"Unknown model {model_name}, using cl100k_base encoding")

    token_counter = TokenCountingHandler(
        tokenizer=tokenizer_fn,
        verbose=verbose
    )

    if Settings.callback_manager is None:
        Settings.callback_manager = CallbackManager([token_counter])
    else:
        Settings.callback_manager.handlers.append(token_counter)

    logger.info(f"Global TokenCountingHandler registered: model={model_name}, verbose={verbose}")
    return token_counter
