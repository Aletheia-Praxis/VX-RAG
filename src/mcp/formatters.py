"""
MCP Formatters - Response formatting and data transformation for LLM consumption.

This module handles all formatting, redaction, and structuring of responses
before they are sent to the LLM client. It ensures responses are clean,
safe, and optimally structured for LLM comprehension.

Key responsibilities:
- Format RAG results into LLM-friendly structures
- Apply sensitive data redaction (emails, IPs)
- Structure citations and source references
- Apply token budgeting and truncation when needed
"""

import re
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

from src.rag.libs.schemas.mcp_schemas import MCPContextPayload

from .schemas import (
    QueryKnowledgeBaseResponse,
    SearchDocumentsResponse,
    SourceDocument,
    RetrievalStats,
    HealthStatusResponse,
    ErrorResponse,
    SystemContextResponse,
    SystemCapabilities,
)

from src.utils.logging_config import get_logger

logger = get_logger("mcp_formatters")


def redact_email_addresses(text: str) -> str:
    """
    Redact email addresses from text.
    
    Args:
        text: Input text that may contain email addresses
        
    Returns:
        Text with email addresses replaced by [REDACTED_EMAIL]
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.sub(email_pattern, '[REDACTED_EMAIL]', text)


def redact_ip_addresses(text: str) -> str:
    """
    Redact IP addresses from text.
    
    Args:
        text: Input text that may contain IP addresses
        
    Returns:
        Text with IP addresses replaced by [REDACTED_IP]
    """
    # IPv4 pattern — octets restricted to 0-255 to avoid false-positive matches
    # on version strings like 1.2.3.4 that happen to look like IPs but aren't.
    ipv4_pattern = (
        r'\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}'
        r'(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b'
    )
    text = re.sub(ipv4_pattern, '[REDACTED_IP]', text)
    
    # IPv6 pattern (basic)
    ipv6_pattern = r'\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b'
    text = re.sub(ipv6_pattern, '[REDACTED_IPv6]', text)
    
    return text


def redact_sensitive_data(text: str) -> str:
    """
    Apply all redaction rules to text.
    
    Current redactions:
    - Email addresses → [REDACTED_EMAIL]
    - IP addresses → [REDACTED_IP]
    
    NOT redacted (important for cybersecurity context):
    - Names (individuals, organizations)
    - Code samples
    - Technical identifiers (hashes, CVEs, etc.)
    
    Args:
        text: Input text
        
    Returns:
        Text with sensitive data redacted
    """
    text = redact_email_addresses(text)
    text = redact_ip_addresses(text)
    return text


def format_query_response(
    rag_result: Union[Dict[str, Any], MCPContextPayload],
    apply_redaction: bool = True
) -> str:
    """
    Format RAG query result into JSON response for LLM.
    
    Args:
        rag_result: Raw result from RAG orchestrator
        apply_redaction: Whether to apply sensitive data redaction
        
    Returns:
        JSON-formatted string suitable for MCP response
    """
    try:
        # Extract components
        if isinstance(rag_result, MCPContextPayload):
            query = rag_result.query
            # context_items will be List[ContextItem]
            context_items = rag_result.context
            tokens_estimate = rag_result.total_tokens_estimate()
            sources_count = len(context_items)
            # Guard against None provenance (e.g., default-constructed MCPContextPayload)
            stats = rag_result.provenance or {}
        else:
            query = rag_result.get('query', '')
            # context_items will be List[Dict]
            context_items = rag_result.get('context', [])
            tokens_estimate = rag_result.get('total_tokens_estimate', 0)
            sources_count = rag_result.get('sources_count', 0)
            stats = rag_result.get('retrieval_stats', {})
        
        # Build source documents
        sources = []
        for item in context_items:
            if isinstance(rag_result, MCPContextPayload):
                # item is ContextItem object
                text = item.text
                item_id = item.id
                score = item.score
                metadata = item.meta
            else:
                # item is dict
                text = item.get('text', '')
                item_id = item.get('id', '')
                score = item.get('score')
                metadata = item.get('metadata', {})
            
            # Apply redaction if enabled
            if apply_redaction:
                text = redact_sensitive_data(text)
            
            source_doc = SourceDocument(
                id=item_id,
                text=text,
                score=score,
                meta=metadata
            )
            sources.append(source_doc)
        
        # Build retrieval stats
        retrieval_stats = RetrievalStats(
            retrieve_duration_ms=stats.get('retrieve_duration_ms', 0),
            rerank_duration_ms=stats.get('rerank_duration_ms', 0),
            assemble_duration_ms=stats.get('assemble_duration_ms', 0),
            total_duration_ms=stats.get('total_duration_ms', 0),
            candidates_retrieved=stats.get('candidates_retrieved', 0),
            results_reranked=stats.get('results_reranked', 0),
            search_type=stats.get('search_type', 'unknown')
        )
        
        # Build response
        response = QueryKnowledgeBaseResponse(
            query=query,
            context=sources,
            total_tokens_estimate=tokens_estimate,
            sources_count=sources_count,
            retrieval_stats=retrieval_stats
        )
        
        # Convert to JSON
        json_str = response.model_dump_json(indent=2, exclude_none=True)
        
        logger.debug(
            "Query response formatted",
            query=query,
            sources=sources_count,
            tokens=tokens_estimate
        )
        
        return json_str
        
    except Exception as e:
        logger.error(
            "Failed to format query response",
            error=str(e),
            exc_info=True
        )
        # Return error response
        error_response = ErrorResponse(
            error=f"Response formatting failed: {str(e)}",
            error_type="formatting_error",
            details=None
        )
        return error_response.model_dump_json(indent=2)


def format_search_response(
    query: str,
    results: List[Dict[str, Any]],
    search_type: str = "semantic",
    apply_redaction: bool = True
) -> str:
    """
    Format document search results into JSON response for LLM.
    
    Args:
        query: Original search query
        results: List of search result documents
        search_type: Type of search performed
        apply_redaction: Whether to apply sensitive data redaction
        
    Returns:
        JSON-formatted string suitable for MCP response
    """
    try:
        # Build source documents
        sources = []
        for result in results:
            text = result.get('text', '')
            
            # Apply redaction if enabled
            if apply_redaction:
                text = redact_sensitive_data(text)
            
            source_doc = SourceDocument(
                id=result.get('node_id', result.get('id', '')),
                text=text,
                score=result.get('score'),
                meta=result.get('metadata', {})
            )
            sources.append(source_doc)
        
        # Build response
        response = SearchDocumentsResponse(
            query=query,
            results=sources,
            results_count=len(sources),
            search_type=search_type
        )
        
        # Convert to JSON
        json_str = response.model_dump_json(indent=2, exclude_none=True)
        
        logger.debug(
            "Search response formatted",
            query=query,
            results=len(sources),
            search_type=search_type
        )
        
        return json_str
        
    except Exception as e:
        logger.error(
            "Failed to format search response",
            error=str(e),
            exc_info=True
        )
        # Return error response
        error_response = ErrorResponse(
            error=f"Response formatting failed: {str(e)}",
            error_type="formatting_error",
            details=None
        )
        return error_response.model_dump_json(indent=2)


def format_health_status(health_data: Dict[str, Any]) -> str:
    """
    Format system health status into JSON response.
    
    Args:
        health_data: Raw health status from orchestrator
        
    Returns:
        JSON-formatted health status
    """
    try:
        overall_status = health_data.get('overall_status', 'unknown')
        initialized = health_data.get('initialized', False)
        indexes_loaded = health_data.get('indexes_loaded', False)
        services = health_data.get('services', {})
        
        # Create message based on status
        message = None
        if not initialized:
            message = "System is initializing. Please wait."
        elif not indexes_loaded:
            message = "System is degraded: indexes not loaded. Some features may be unavailable."
        elif overall_status == "healthy":
            message = "All systems operational."
        
        response = HealthStatusResponse(
            overall_status=overall_status,
            initialized=initialized,
            indexes_loaded=indexes_loaded,
            services=services,
            message=message
        )
        
        json_str = response.model_dump_json(indent=2, exclude_none=True)
        
        logger.debug("Health status formatted", status=overall_status)
        
        return json_str
        
    except Exception as e:
        logger.error(
            "Failed to format health status",
            error=str(e),
            exc_info=True
        )
        # Return error response
        error_response = ErrorResponse(
            error=f"Health status formatting failed: {str(e)}",
            error_type="formatting_error",
            details=None
        )
        return error_response.model_dump_json(indent=2)


def format_system_context() -> str:
    """
    Format system context and capabilities into JSON response.
    
    Returns:
        JSON-formatted system context
    """
    try:
        capabilities = SystemCapabilities(
            description=(
                "VX-RAG is a specialized Retrieval-Augmented Generation system "
                "for querying technical cybersecurity documentation from the "
                "VX Underground collection."
            ),
            supported_formats=["PDF", "TXT", "MD"],
            search_types=["semantic", "keyword", "hybrid"],
            max_results=20,
            max_token_budget=16000
        )
        
        corpus_info = {
            "name": "VX Underground Collection",
            "description": "Cybersecurity technical documents, articles, and research papers",
            "content_types": [
                "Technical articles",
                "Research papers",
                "Code samples",
                "Security analyses"
            ],
            # Intentionally static: this reflects the corpus index date, not today's date.
            # Update this value when the corpus is re-indexed.
            "index_date": "unknown"
        }
        
        response = SystemContextResponse(
            capabilities=capabilities,
            version="1.0.0",
            corpus_info=corpus_info
        )
        
        json_str = response.model_dump_json(indent=2, exclude_none=True)
        
        logger.debug("System context formatted")
        
        return json_str
        
    except Exception as e:
        logger.error(
            "Failed to format system context",
            error=str(e),
            exc_info=True
        )
        # Return error response
        error_response = ErrorResponse(
            error=f"System context formatting failed: {str(e)}",
            error_type="formatting_error",
            details=None
        )
        return error_response.model_dump_json(indent=2)


def format_error_response(
    error_message: str,
    error_type: str = "internal_error",
    details: Optional[Dict[str, Any]] = None
) -> str:
    """
    Format error into standard error response.
    
    Args:
        error_message: Human-readable error message
        error_type: Type of error (validation_error, not_found, internal_error, etc.)
        details: Optional additional error details
        
    Returns:
        JSON-formatted error response
    """
    try:
        response = ErrorResponse(
            error=error_message,
            error_type=error_type,
            details=details
        )
        
        json_str = response.model_dump_json(indent=2, exclude_none=True)
        
        logger.debug(
            "Error response formatted",
            error_type=error_type,
            error_msg=error_message
        )
        
        return json_str
        
    except Exception as e:
        # Fallback to basic JSON if Pydantic fails
        logger.error(
            "Failed to format error response",
            error=str(e),
            exc_info=True
        )
        return json.dumps({
            "error": error_message,
            "error_type": error_type,
            "formatting_error": str(e)
        }, indent=2)
