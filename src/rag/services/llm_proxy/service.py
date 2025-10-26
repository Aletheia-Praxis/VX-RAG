"""
LLM Proxy Service implementation.

Provides classes for LLM interaction and response generation.
"""

from typing import List, Dict, Any, Optional
import logging

from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)

class LLMProxy:
    """Service for LLM interaction and response generation."""
    
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.llm = Ollama(model=self.model_name)
        logger.info(f"Initialized LLM proxy with model: {self.model_name}")
    
    def generate(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None, **kwargs: Any) -> str:
        """
        Generate response using LLM with optional context.
        
        Args:
            prompt: The query prompt
            context: List of context documents
            **kwargs: Additional parameters for LLM
            
        Returns:
            Generated response string
        """
        try:
            if context:
                # Build context string from retrieved documents
                context_str = "\n\n".join([
                    f"Document {i+1}:\n{doc.get('text', '')}"
                    for i, doc in enumerate(context)
                ])
                full_prompt = f"Context:\n{context_str}\n\nQuestion: {prompt}\n\nAnswer:"
            else:
                full_prompt = prompt
            
            # Generate response
            response = self.llm.complete(full_prompt)
            generated_text = str(response)
            
            logger.info(f"Generated response using {self.model_name}")
            return generated_text
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"Error generating response: {e}"
    
    def generate_with_sources(self, prompt: str, context: List[Dict[str, Any]], **kwargs: Any) -> Dict[str, Any]:
        """
        Generate response with source information.
        
        Args:
            prompt: The query prompt
            context: List of context documents
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with response and sources
        """
        response = self.generate(prompt, context, **kwargs)
        
        sources = []
        for doc in context:
            source = {
                'text': doc.get('text', '')[:500] + "..." if len(doc.get('text', '')) > 500 else doc.get('text', ''),
                'score': doc.get('score', 0.0),
                'metadata': doc.get('metadata', {})
            }
            sources.append(source)
        
        return {
            'response': response,
            'sources': sources
        }

# TODO: Add streaming, conversation history, model switching
