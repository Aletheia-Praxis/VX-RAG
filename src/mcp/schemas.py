"""
MCP Schemas - Pydantic models for MCP request/response structures.

Defines clean, typed interfaces for MCP tools exposed to LLMs.
These models ensure type safety and validation for all MCP interactions.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryKnowledgeBaseRequest(BaseModel):
    """
    Parameters for querying the knowledge base.
    
    This is the primary tool for LLMs to retrieve contextual information
    from the document corpus.
    """
    query: str = Field(
        ...,
        description="The search query or question to find relevant documents for",
        min_length=1,
        max_length=1000
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of most relevant documents to return (1-20)"
    )
    search_type: str = Field(
        "hybrid",
        description="Search strategy: 'semantic' (vector), 'keyword' (BM25), or 'hybrid' (both)"
    )
    token_budget: int = Field(
        4000,
        ge=500,
        le=16000,
        description="Maximum token budget for assembled context (500-16000)"
    )


class SearchDocumentsRequest(BaseModel):
    """
    Parameters for simple document search without context assembly.
    
    Returns raw search results without additional processing or formatting.
    Useful for exploration and discovery.
    """
    query: str = Field(
        ...,
        description="Search query string",
        min_length=1,
        max_length=1000
    )
    top_k: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of results to return (1-50)"
    )
    search_type: str = Field(
        "semantic",
        description="Search strategy: 'semantic', 'keyword', or 'hybrid'"
    )


class GetDocumentRequest(BaseModel):
    """
    Parameters for retrieving a specific document by ID.
    
    Used to fetch full details of a document referenced in previous results.
    """
    document_id: str = Field(
        ...,
        description="Unique identifier of the document to retrieve",
        min_length=1
    )


from src.rag.libs.schemas.mcp_schemas import ContextItem

# Alias SourceDocument to ContextItem for schema unification
# Both represent a single source document in query results.
SourceDocument = ContextItem


class RetrievalStats(BaseModel):
    """
    Performance and execution statistics for query operations.
    
    Provides transparency into the retrieval pipeline performance.
    """
    retrieve_duration_ms: float = Field(
        ...,
        description="Time spent on initial retrieval (milliseconds)"
    )
    rerank_duration_ms: float = Field(
        ...,
        description="Time spent on reranking (milliseconds)"
    )
    assemble_duration_ms: float = Field(
        ...,
        description="Time spent on context assembly (milliseconds)"
    )
    total_duration_ms: float = Field(
        ...,
        description="Total query processing time (milliseconds)"
    )
    candidates_retrieved: int = Field(
        ...,
        description="Number of candidates retrieved initially"
    )
    results_reranked: int = Field(
        ...,
        description="Number of results after reranking"
    )
    search_type: str = Field(
        ...,
        description="Search strategy used (semantic/keyword/hybrid)"
    )


class QueryKnowledgeBaseResponse(BaseModel):
    """
    Response from knowledge base query.
    
    Contains assembled context optimized for LLM consumption, plus source references.
    """
    query: str = Field(
        ...,
        description="Original query string"
    )
    context: List[SourceDocument] = Field(
        default_factory=list,
        description="List of relevant documents with text and metadata"
    )
    total_tokens_estimate: int = Field(
        ...,
        description="Estimated total tokens in the context"
    )
    sources_count: int = Field(
        ...,
        description="Number of source documents included"
    )
    retrieval_stats: RetrievalStats = Field(
        ...,
        description="Performance statistics for the query"
    )


class SearchDocumentsResponse(BaseModel):
    """
    Response from document search.
    
    Simple list of matching documents without additional processing.
    """
    query: str = Field(
        ...,
        description="Original search query"
    )
    results: List[SourceDocument] = Field(
        default_factory=list,
        description="List of matching documents"
    )
    results_count: int = Field(
        ...,
        description="Number of results returned"
    )
    search_type: str = Field(
        ...,
        description="Search strategy used"
    )


class DocumentDetailResponse(BaseModel):
    """
    Response with full document details.
    
    Provides complete information about a specific document.
    """
    document_id: str = Field(
        ...,
        description="Document unique identifier"
    )
    text: str = Field(
        ...,
        description="Full document text"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Complete document metadata"
    )
    found: bool = Field(
        ...,
        description="Whether the document was found"
    )


class HealthStatusResponse(BaseModel):
    """
    System health status response.
    
    Provides visibility into system readiness and service availability.
    """
    overall_status: str = Field(
        ...,
        description="Overall system status: 'healthy', 'degraded', or 'unavailable'"
    )
    initialized: bool = Field(
        ...,
        description="Whether RAG services are initialized"
    )
    indexes_loaded: bool = Field(
        ...,
        description="Whether document indexes are loaded"
    )
    services: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Status of individual services"
    )
    message: Optional[str] = Field(
        None,
        description="Additional status information or error message"
    )


class ErrorResponse(BaseModel):
    """
    Error response for failed operations.
    
    Provides structured error information for debugging and user feedback.
    """
    error: str = Field(
        ...,
        description="Error message describing what went wrong"
    )
    error_type: str = Field(
        "internal_error",
        description="Type of error (validation_error, not_found, internal_error, etc.)"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional error details for debugging"
    )


class SystemCapabilities(BaseModel):
    """
    Description of system capabilities.
    
    Informs LLMs about what the system can do.
    """
    description: str = Field(
        ...,
        description="High-level description of the system"
    )
    supported_formats: List[str] = Field(
        default_factory=list,
        description="Document formats supported (PDF, TXT, MD, etc.)"
    )
    search_types: List[str] = Field(
        default_factory=list,
        description="Available search strategies"
    )
    max_results: int = Field(
        ...,
        description="Maximum number of results that can be returned"
    )
    max_token_budget: int = Field(
        ...,
        description="Maximum token budget for context assembly"
    )


class SystemContextResponse(BaseModel):
    """
    System context and capabilities response.
    
    Provides LLMs with information about system features and limitations.
    """
    capabilities: SystemCapabilities = Field(
        ...,
        description="System capabilities and features"
    )
    version: str = Field(
        ...,
        description="System version"
    )
    corpus_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Information about the document corpus"
    )
