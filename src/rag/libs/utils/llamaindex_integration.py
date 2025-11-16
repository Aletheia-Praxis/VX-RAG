"""
LlamaIndex integration utilities for VX-RAG.

Provides adapters and helpers for seamless integration between
VX-RAG components and LlamaIndex ecosystem.
"""

import logging
from typing import List, Dict, Any, Optional
from llama_index.core.schema import NodeWithScore, BaseNode, Document as LlamaDocument
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core import Settings
import tiktoken

logger = logging.getLogger(__name__)


class VXRAGLlamaIndexAdapter:
    """
    Adapter for integrating VX-RAG with LlamaIndex global infrastructure.
    
    Provides conversion between VX-RAG document formats and LlamaIndex Node objects,
    and manages global LlamaIndex settings integration.
    """
    
    @staticmethod
    def setup_global_token_counting(
        model_name: str = "gpt-3.5-turbo",
        verbose: bool = False
    ) -> TokenCountingHandler:
        """
        Configure global LlamaIndex token counting via CallbackManager.
        
        Args:
            model_name: Model name for tokenizer
            verbose: If True, prints token usage to console
            
        Returns:
            Configured TokenCountingHandler
        """
        try:
            tokenizer_fn = tiktoken.encoding_for_model(model_name).encode
        except KeyError:
            tokenizer_fn = tiktoken.get_encoding("cl100k_base").encode
            logger.warning(f"Unknown model {model_name}, using cl100k_base encoding")
        
        token_counter = TokenCountingHandler(
            tokenizer=tokenizer_fn,
            verbose=verbose
        )
        
        # Register with global CallbackManager
        if Settings.callback_manager is None:
            Settings.callback_manager = CallbackManager([token_counter])
        else:
            # Add to existing callback manager
            Settings.callback_manager.handlers.append(token_counter)
        
        logger.info(
            f"Global LlamaIndex token counting configured: model={model_name}, "
            f"verbose={verbose}"
        )
        
        return token_counter
    
    @staticmethod
    def vxrag_doc_to_llama_node(document: Dict[str, Any]) -> NodeWithScore:
        """
        Convert VX-RAG document format to LlamaIndex NodeWithScore.
        
        Args:
            document: VX-RAG document dict with 'id', 'text', 'score', 'metadata'
            
        Returns:
            LlamaIndex NodeWithScore object
        """
        text = document.get('text', '')
        metadata = document.get('metadata', {})
        score = document.get('score', 0.0)
        node_id = document.get('id', document.get('node_id', ''))
        
        # Create LlamaDocument (inherits from BaseNode)
        node = LlamaDocument(
            text=text,
            metadata=metadata,
            id_=node_id
        )
        
        # Wrap in NodeWithScore
        return NodeWithScore(node=node, score=score)
    
    @staticmethod
    def llama_node_to_vxrag_doc(node: NodeWithScore) -> Dict[str, Any]:
        """
        Convert LlamaIndex NodeWithScore to VX-RAG document format.
        
        Args:
            node: LlamaIndex NodeWithScore object
            
        Returns:
            VX-RAG document dictionary
        """
        node_obj = node.node if isinstance(node, NodeWithScore) else node
        
        return {
            'id': node_obj.node_id if hasattr(node_obj, 'node_id') else node_obj.id_,
            'text': node_obj.get_content(),
            'score': node.score if isinstance(node, NodeWithScore) else None,
            'metadata': node_obj.metadata if hasattr(node_obj, 'metadata') else {}
        }
    
    @staticmethod
    def batch_vxrag_to_llama(documents: List[Dict[str, Any]]) -> List[NodeWithScore]:
        """
        Batch convert VX-RAG documents to LlamaIndex nodes.
        
        Args:
            documents: List of VX-RAG document dictionaries
            
        Returns:
            List of LlamaIndex NodeWithScore objects
        """
        return [
            VXRAGLlamaIndexAdapter.vxrag_doc_to_llama_node(doc) 
            for doc in documents
        ]
    
    @staticmethod
    def batch_llama_to_vxrag(nodes: List[NodeWithScore]) -> List[Dict[str, Any]]:
        """
        Batch convert LlamaIndex nodes to VX-RAG documents.
        
        Args:
            nodes: List of LlamaIndex NodeWithScore objects
            
        Returns:
            List of VX-RAG document dictionaries
        """
        return [
            VXRAGLlamaIndexAdapter.llama_node_to_vxrag_doc(node) 
            for node in nodes
        ]
    
    @staticmethod
    def get_global_token_stats() -> Dict[str, Any]:
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
    
    @staticmethod
    def reset_global_token_counts() -> None:
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
    return VXRAGLlamaIndexAdapter.setup_global_token_counting(model_name, verbose)


def get_token_stats() -> Dict[str, Any]:
    """Get global token statistics shorthand."""
    return VXRAGLlamaIndexAdapter.get_global_token_stats()


def reset_token_counts() -> None:
    """Reset global token counts shorthand."""
    VXRAGLlamaIndexAdapter.reset_global_token_counts()
