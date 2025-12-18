from typing import List, Sequence, Optional
from llama_index.core.schema import BaseNode, TransformComponent
from src.rag.services.duplicate_detection_service.service import DuplicateDetector

class DeduplicationTransform(TransformComponent):
    """
    Transform component that removes duplicate nodes using DuplicateDetector.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__()
        self.detector = DuplicateDetector(config_path=config_path)

    def __call__(self, nodes: Sequence[BaseNode], **kwargs) -> Sequence[BaseNode]:
        """
        Apply deduplication to the list of nodes.
        """
        # Convert Sequence to List for DuplicateDetector
        nodes_list = list(nodes)
        unique_nodes = self.detector.remove_duplicates(nodes_list)
        return unique_nodes
