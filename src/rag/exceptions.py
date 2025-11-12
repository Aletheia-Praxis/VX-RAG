"""
Custom exceptions for RAG system.

This module defines specific exception types for different error scenarios
in the RAG pipeline, enabling more precise error handling and debugging.
"""


class RAGException(Exception):
    """Base exception for all RAG-related errors."""
    pass


class ServiceInitializationError(RAGException):
    """Raised when a RAG service fails to initialize."""
    
    def __init__(self, service_name: str, reason: str) -> None:
        """
        Initialize service initialization error.
        
        Args:
            service_name: Name of the service that failed to initialize
            reason: Reason for initialization failure
        """
        self.service_name = service_name
        self.reason = reason
        super().__init__(f"Failed to initialize {service_name}: {reason}")


class IndexNotFoundError(RAGException):
    """Raised when a required index is not found."""
    
    def __init__(self, index_type: str, index_path: str) -> None:
        """
        Initialize index not found error.
        
        Args:
            index_type: Type of index (e.g., "FAISS", "BM25")
            index_path: Path where index was expected
        """
        self.index_type = index_type
        self.index_path = index_path
        super().__init__(f"{index_type} index not found at {index_path}")


class IndexLoadError(RAGException):
    """Raised when an index fails to load."""
    
    def __init__(self, index_type: str, reason: str) -> None:
        """
        Initialize index load error.
        
        Args:
            index_type: Type of index (e.g., "FAISS", "BM25")
            reason: Reason for load failure
        """
        self.index_type = index_type
        self.reason = reason
        super().__init__(f"Failed to load {index_type} index: {reason}")


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""
    
    def __init__(self, query: str, reason: str) -> None:
        """
        Initialize retrieval error.
        
        Args:
            query: The query that failed
            reason: Reason for retrieval failure
        """
        self.query = query
        self.reason = reason
        super().__init__(f"Retrieval failed for query '{query}': {reason}")


class RerankingError(RAGException):
    """Raised when document reranking fails."""
    
    def __init__(self, reason: str, document_count: int = 0) -> None:
        """
        Initialize reranking error.
        
        Args:
            reason: Reason for reranking failure
            document_count: Number of documents that failed to rerank
        """
        self.reason = reason
        self.document_count = document_count
        super().__init__(f"Reranking failed: {reason} (docs: {document_count})")


class EmbeddingError(RAGException):
    """Raised when text embedding generation fails."""
    
    def __init__(self, text_preview: str, reason: str) -> None:
        """
        Initialize embedding error.
        
        Args:
            text_preview: Preview of text that failed to embed
            reason: Reason for embedding failure
        """
        self.text_preview = text_preview[:100]  # Keep first 100 chars
        self.reason = reason
        super().__init__(f"Embedding failed: {reason}")


class ContextAssemblyError(RAGException):
    """Raised when context assembly fails."""
    
    def __init__(self, reason: str, documents_available: int = 0) -> None:
        """
        Initialize context assembly error.
        
        Args:
            reason: Reason for assembly failure
            documents_available: Number of documents available for assembly
        """
        self.reason = reason
        self.documents_available = documents_available
        super().__init__(
            f"Context assembly failed: {reason} (docs available: {documents_available})"
        )


class ConfigurationError(RAGException):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, config_key: str, reason: str) -> None:
        """
        Initialize configuration error.
        
        Args:
            config_key: Configuration key that is invalid/missing
            reason: Reason for configuration error
        """
        self.config_key = config_key
        self.reason = reason
        super().__init__(f"Configuration error for '{config_key}': {reason}")


class DocumentParsingError(RAGException):
    """Raised when document parsing fails."""
    
    def __init__(self, file_path: str, reason: str) -> None:
        """
        Initialize document parsing error.
        
        Args:
            file_path: Path to document that failed to parse
            reason: Reason for parsing failure
        """
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to parse document '{file_path}': {reason}")


class VectorStoreError(RAGException):
    """Raised when vector store operations fail."""
    
    def __init__(self, operation: str, reason: str) -> None:
        """
        Initialize vector store error.
        
        Args:
            operation: Operation that failed (e.g., "add", "search", "save")
            reason: Reason for operation failure
        """
        self.operation = operation
        self.reason = reason
        super().__init__(f"Vector store {operation} failed: {reason}")


class QueryValidationError(RAGException):
    """Raised when query validation fails."""
    
    def __init__(self, query: str, reason: str) -> None:
        """
        Initialize query validation error.
        
        Args:
            query: Invalid query string
            reason: Reason for validation failure
        """
        self.query = query
        self.reason = reason
        super().__init__(f"Invalid query '{query}': {reason}")
