"""
MCP Schemas - Pydantic models for MCP request/response structures.

Defines clean, typed interfaces for MCP tools exposed to LLMs.
These models ensure type safety and validation for all MCP interactions.
"""

from __future__ import annotations

from typing import Any

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
        max_length=1000,
    )
    top_k: int = Field(
        5,
        ge=1,
        le=20,
        description="Number of most relevant documents to return (1-20)",
    )
    search_type: str = Field(
        "hybrid",
        description="Search strategy: 'semantic' (vector), 'keyword' (BM25), or 'hybrid' (both)",
    )
    token_budget: int = Field(
        4000,
        ge=500,
        le=16000,
        description="Maximum token budget for assembled context (500-16000)",
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
        max_length=1000,
    )
    top_k: int = Field(
        10,
        ge=1,
        le=50,
        description="Number of results to return (1-50)",
    )
    search_type: str = Field(
        "semantic",
        description="Search strategy: 'semantic', 'keyword', or 'hybrid'",
    )


class GetDocumentRequest(BaseModel):
    """
    Parameters for retrieving a specific document by ID.

    Used to fetch full details of a document referenced in previous results.
    """

    document_id: str = Field(
        ...,
        description="Unique identifier of the document to retrieve",
        min_length=1,
    )


class SourceDocument(BaseModel):
    """
    Source document representation for MCP responses.

    Contains snippet text, relevance score, and file metadata citations.
    """

    id: str = Field(
        ...,
        description="Unique document or node identifier",
    )
    text: str = Field(
        ...,
        description="Document text content or snippet",
    )
    score: float | None = Field(
        None,
        description="Relevance or reranking score",
    )
    file_name: str | None = Field(
        None,
        description="Source file name",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata dictionary",
    )
    meta: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata dictionary alias for backward compatibility",
    )


class RetrievalStats(BaseModel):
    """
    Performance and execution statistics for query operations.

    Provides transparency into the retrieval pipeline performance.
    """

    retrieve_duration_ms: float = Field(
        0.0,
        description="Time spent on initial retrieval (milliseconds)",
    )
    rerank_duration_ms: float = Field(
        0.0,
        description="Time spent on reranking (milliseconds)",
    )
    assemble_duration_ms: float = Field(
        0.0,
        description="Time spent on context assembly (milliseconds)",
    )
    total_duration_ms: float = Field(
        0.0,
        description="Total query processing time (milliseconds)",
    )
    candidates_retrieved: int = Field(
        0,
        description="Number of candidates retrieved initially",
    )
    results_reranked: int = Field(
        0,
        description="Number of results after reranking",
    )
    search_type: str = Field(
        "unknown",
        description="Search strategy used (semantic/keyword/hybrid)",
    )
    total_results: int = Field(
        0,
        description="Total number of results returned",
    )
    token_budget: int = Field(
        4000,
        description="Maximum token budget configured for retrieval",
    )
    total_tokens: int = Field(
        0,
        description="Estimated total tokens used in context assembly",
    )


class QueryKnowledgeBaseResponse(BaseModel):
    """
    Response from knowledge base query.

    Contains assembled context optimized for LLM consumption, plus source references.
    """

    query: str = Field(
        ...,
        description="Original query string",
    )
    context: str = Field(
        ...,
        description="Assembled context text or fallback message",
    )
    sources: list[SourceDocument] = Field(
        default_factory=list,
        description="List of relevant documents with text and metadata citations",
    )
    stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Performance statistics for the query",
    )
    total_tokens_estimate: int | None = Field(
        None,
        description="Estimated total tokens in the context",
    )
    sources_count: int | None = Field(
        None,
        description="Number of source documents included",
    )
    retrieval_stats: RetrievalStats | None = Field(
        None,
        description="Typed retrieval statistics object",
    )


class SearchDocumentsResponse(BaseModel):
    """
    Response from document search.

    Simple list of matching documents without additional processing.
    """

    query: str = Field(
        ...,
        description="Original search query",
    )
    documents: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of matching documents with full un-truncated text",
    )
    total_count: int = Field(
        0,
        description="Total number of results returned",
    )
    results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Alias for documents list for backward compatibility",
    )
    results_count: int = Field(
        0,
        description="Alias for total_count for backward compatibility",
    )
    search_type: str = Field(
        "semantic",
        description="Search strategy used",
    )


class DocumentDetailResponse(BaseModel):
    """
    Response with full document details.

    Provides complete information about a specific document.
    """

    document_id: str = Field(
        ...,
        description="Document unique identifier",
    )
    text: str = Field(
        ...,
        description="Full document text",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Complete document metadata",
    )
    found: bool = Field(
        ...,
        description="Whether the document was found",
    )


class HealthStatusResponse(BaseModel):
    """
    System health status response.

    Provides visibility into system readiness and service availability.
    """

    overall_status: str = Field(
        ...,
        description="Overall system status: 'healthy', 'degraded', or 'unavailable'",
    )
    initialized: bool = Field(
        ...,
        description="Whether RAG services are initialized",
    )
    indexes_loaded: bool = Field(
        ...,
        description="Whether document indexes are loaded",
    )
    services: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Status of individual services",
    )
    message: str | None = Field(
        None,
        description="Additional status information or error message",
    )


class ErrorResponse(BaseModel):
    """
    Error response for failed operations.

    Provides structured error information for debugging and user feedback.
    """

    error: str = Field(
        ...,
        description="Error message describing what went wrong",
    )
    error_type: str = Field(
        "internal_error",
        description="Type of error (validation_error, not_found, internal_error, etc.)",
    )
    details: dict[str, Any] | None = Field(
        None,
        description="Additional error details for debugging",
    )


class SystemCapabilities(BaseModel):
    """
    Description of system capabilities.

    Informs LLMs about what the system can do.
    """

    description: str = Field(
        ...,
        description="High-level description of the system",
    )
    supported_document_types: list[str] = Field(
        default_factory=lambda: ["PDF", "TXT", "MD"],
        description="Document formats supported (PDF, TXT, MD, etc.)",
    )
    supported_formats: list[str] = Field(
        default_factory=lambda: ["PDF", "TXT", "MD"],
        description="Alias for supported_document_types",
    )
    search_types: list[str] = Field(
        default_factory=lambda: ["semantic", "keyword", "hybrid"],
        description="Available search strategies",
    )
    max_results: int = Field(
        20,
        description="Maximum number of results that can be returned",
    )
    max_token_budget: int = Field(
        16000,
        description="Maximum token budget for context assembly",
    )


class SystemContextResponse(BaseModel):
    """
    System context and capabilities response.

    Provides LLMs with information about system features and limitations.
    """

    system_name: str = Field(
        "VX-RAG",
        description="System name identifier",
    )
    capabilities: SystemCapabilities = Field(
        ...,
        description="System capabilities and features",
    )
    version: str = Field(
        "1.0.0",
        description="System version",
    )
    corpus_info: dict[str, Any] = Field(
        default_factory=dict,
        description="Information about the document corpus",
    )
