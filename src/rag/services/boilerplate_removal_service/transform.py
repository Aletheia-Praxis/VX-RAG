"""
Boilerplate removal transformation for LlamaIndex pipeline.

This module implements a LlamaIndex TransformComponent that removes boilerplate
content from nodes using the logic defined in BoilerplateRemovalService.
"""

import re
from typing import List, Any
from llama_index.core.schema import TransformComponent, BaseNode, TextNode
from .service import BoilerplateRemovalService

class BoilerplateCleaner(TransformComponent):
    """
    LlamaIndex transformation that removes boilerplate from node text.
    
    Wraps the BoilerplateRemovalService to be used in an IngestionPipeline.
    """
    
    def __init__(self, aggressive_mode: bool = True, **kwargs: Any) -> None:
        """
        Initialize the cleaner.
        
        Args:
            aggressive_mode: Whether to use aggressive cleaning patterns.
        """
        super().__init__(**kwargs)
        self.service = BoilerplateRemovalService(aggressive_mode=aggressive_mode)
        
    def __call__(self, nodes: List[BaseNode], **kwargs: Any) -> List[BaseNode]:
        """
        Apply cleaning to a list of nodes.
        
        Args:
            nodes: List of nodes to clean.
            
        Returns:
            List of cleaned nodes.
        """
        for node in nodes:
            # We assume the node has text content
            if isinstance(node, TextNode) and node.text:
                node.text = self.service.remove_boilerplate(node.text)
                
        return nodes
