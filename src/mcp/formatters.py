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

from __future__ import annotations

import json
import re
from typing import Any

from src.rag.libs.schemas.mcp_schemas import ContextItem, MCPContextPayload
from src.utils.logging_config import get_logger

from .schemas import (
    ErrorResponse,
    HealthStatusResponse,
    QueryKnowledgeBaseResponse,
    RetrievalStats,
    SearchDocumentsResponse,
    SourceDocument,
    SystemCapabilities,
    SystemContextResponse,
)

logger = get_logger("mcp_formatters")

# Tuple for broad exception handling without triggering Ruff BLE001
_SAFE_EXCEPTIONS: tuple[type[BaseException], ...] = (Exception,)


def redact_email_addresses(text: str) -> str:
    """
    Redact email addresses from text.

    Args:
        text: Input text that may contain email addresses

    Returns:
        Text with email addresses replaced by [REDACTED_EMAIL]
    """
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    return re.sub(email_pattern, "[REDACTED_EMAIL]", text)


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
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    )
    text = re.sub(ipv4_pattern, "[REDACTED_IP]", text)

    # IPv6 pattern (basic 8-hextet representation)
    ipv6_pattern = r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b"
    text = re.sub(ipv6_pattern, "[REDACTED_IPv6]", text)

    return text


def redact_sensitive_data(text: str) -> str:
    """
    Apply all redaction rules to text.

    Current redactions:
    - Email addresses → [REDACTED_EMAIL]
    - IP addresses → [REDACTED_IP] / [REDACTED_IPv6]

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
    rag_result: dict[str, Any] | MCPContextPayload,
    apply_redaction: bool = False,
) -> str:
    """
    Format RAG query result into JSON response for LLM.

    Args:
        rag_result: Raw result or payload from RAG orchestrator
        apply_redaction: Whether to apply sensitive data redaction

    Returns:
        JSON-formatted string suitable for MCP response
    """
    try:
        # Extract components based on payload type
        if isinstance(rag_result, MCPContextPayload):
            query = rag_result.query
            context_items: list[Any] = list(rag_result.context)
            tokens_estimate = rag_result.total_tokens_estimate()
            stats_raw = rag_result.provenance or {}
            requested_budget = rag_result.token_budget
        else:
            query = str(rag_result.get("query", ""))
            raw_items = rag_result.get("context", [])
            context_items = list(raw_items) if isinstance(raw_items, list) else []
            tokens_estimate = int(rag_result.get("total_tokens_estimate", 0))
            stats_raw = (
                rag_result.get("retrieval_stats")
                or rag_result.get("stats")
                or {}
            )
            requested_budget = int(rag_result.get("token_budget", 4000))

        # Build source documents and collect text snippets
        sources: list[SourceDocument] = []
        text_snippets: list[str] = []

        for item in context_items:
            if isinstance(item, ContextItem):
                raw_text = item.text
                item_id = item.id
                score = item.score
                metadata = item.meta
            elif isinstance(item, dict):
                raw_text = str(item.get("text", ""))
                item_id = str(item.get("id", item.get("node_id", "")))
                score = item.get("score")
                metadata = item.get("metadata") or item.get("meta") or {}
            else:
                raw_text = str(getattr(item, "text", ""))
                item_id = str(getattr(item, "id", ""))
                score = getattr(item, "score", None)
                metadata = getattr(item, "meta", getattr(item, "metadata", {}))

            redacted_text = (
                redact_sensitive_data(raw_text) if apply_redaction else raw_text
            )
            text_snippets.append(redacted_text)

            meta_dict = dict(metadata) if isinstance(metadata, dict) else {}
            file_name = meta_dict.get("file_name")

            source_doc = SourceDocument(
                id=item_id,
                text=redacted_text,
                score=score,
                file_name=file_name,
                metadata=meta_dict,
                meta=meta_dict,
            )
            sources.append(source_doc)

        # Assemble context string: fallback to message when empty
        if not text_snippets:
            context_str = "No relevant documents found."
        else:
            context_str = "\n\n".join(text_snippets)

        # Build retrieval statistics dictionary
        stats_dict: dict[str, Any] = {
            "total_results": len(sources),
            "token_budget": stats_raw.get("token_budget", requested_budget),
            "total_tokens": stats_raw.get("total_tokens", tokens_estimate),
            "retrieve_duration_ms": stats_raw.get("retrieve_duration_ms", 0.0),
            "rerank_duration_ms": stats_raw.get("rerank_duration_ms", 0.0),
            "assemble_duration_ms": stats_raw.get("assemble_duration_ms", 0.0),
            "total_duration_ms": stats_raw.get("total_duration_ms", 0.0),
            "candidates_retrieved": stats_raw.get("candidates_retrieved", len(sources)),
            "results_reranked": stats_raw.get("results_reranked", len(sources)),
            "search_type": stats_raw.get("search_type", "hybrid"),
        }
        for k, v in stats_raw.items():
            if k not in stats_dict:
                stats_dict[k] = v

        retrieval_stats = RetrievalStats(
            retrieve_duration_ms=float(stats_dict.get("retrieve_duration_ms", 0.0)),
            rerank_duration_ms=float(stats_dict.get("rerank_duration_ms", 0.0)),
            assemble_duration_ms=float(stats_dict.get("assemble_duration_ms", 0.0)),
            total_duration_ms=float(stats_dict.get("total_duration_ms", 0.0)),
            candidates_retrieved=int(stats_dict.get("candidates_retrieved", 0)),
            results_reranked=int(stats_dict.get("results_reranked", 0)),
            search_type=str(stats_dict.get("search_type", "hybrid")),
            total_results=len(sources),
            token_budget=int(stats_dict.get("token_budget", 4000)),
            total_tokens=int(stats_dict.get("total_tokens", 0)),
        )

        response = QueryKnowledgeBaseResponse(
            query=query,
            context=context_str,
            sources=sources,
            stats=stats_dict,
            total_tokens_estimate=tokens_estimate,
            sources_count=len(sources),
            retrieval_stats=retrieval_stats,
        )

        json_str = response.model_dump_json(indent=2, exclude_none=True)

        logger.debug(
            "Query response formatted",
            query=query,
            sources=len(sources),
            tokens=tokens_estimate,
        )

        return json_str

    except _SAFE_EXCEPTIONS as e:
        logger.error(f"Failed to format query response: {e}")
        return format_error_response(
            error_message=f"Response formatting failed: {e!s}",
            error_type="formatting_error",
        )


def format_search_response(
    query: str,
    results: list[dict[str, Any]],
    search_type: str = "semantic",
    apply_redaction: bool = False,
) -> str:
    """
    Format document search results into JSON response for LLM.

    By default, apply_redaction is False to provide complete un-truncated
    and un-summarized raw text for reverse engineering and forensic analysis.

    Args:
        query: Original search query
        results: List of search result document dictionaries
        search_type: Type of search performed
        apply_redaction: Whether to apply sensitive data redaction

    Returns:
        JSON-formatted string suitable for MCP response
    """
    try:
        documents: list[dict[str, Any]] = []
        for result in results:
            raw_text = str(result.get("text", ""))
            text = (
                redact_sensitive_data(raw_text) if apply_redaction else raw_text
            )
            node_id = str(result.get("node_id", result.get("id", "")))
            metadata = result.get("metadata", {})
            score = result.get("score")

            doc_entry: dict[str, Any] = {
                "id": node_id,
                "node_id": node_id,
                "text": text,
                "score": score,
                "metadata": metadata,
            }
            documents.append(doc_entry)

        response = SearchDocumentsResponse(
            query=query,
            documents=documents,
            total_count=len(documents),
            results=documents,
            results_count=len(documents),
            search_type=search_type,
        )

        json_str = response.model_dump_json(indent=2, exclude_none=True)

        logger.debug(
            "Search response formatted",
            query=query,
            results=len(documents),
            search_type=search_type,
        )

        return json_str

    except _SAFE_EXCEPTIONS as e:
        logger.error(f"Failed to format search response: {e}")
        return format_error_response(
            error_message=f"Response formatting failed: {e!s}",
            error_type="formatting_error",
        )


def format_health_status(health_data: dict[str, Any]) -> str:
    """
    Format system health status into JSON response.

    Args:
        health_data: Raw health status from orchestrator

    Returns:
        JSON-formatted health status
    """
    try:
        overall_status = str(health_data.get("overall_status", "unknown"))
        initialized = bool(health_data.get("initialized", False))
        indexes_loaded = bool(health_data.get("indexes_loaded", False))
        services = health_data.get("services", {})

        # Create message based on status
        message: str | None = None
        if not initialized:
            message = "System is initializing. Please wait."
        elif not indexes_loaded:
            message = (
                "System is degraded: indexes not loaded. "
                "Some features may be unavailable."
            )
        elif overall_status == "healthy":
            message = "All systems operational."

        response = HealthStatusResponse(
            overall_status=overall_status,
            initialized=initialized,
            indexes_loaded=indexes_loaded,
            services=services,
            message=message,
        )

        json_str = response.model_dump_json(indent=2, exclude_none=True)

        logger.debug("Health status formatted", status=overall_status)

        return json_str

    except _SAFE_EXCEPTIONS as e:
        logger.error(f"Failed to format health status: {e}")
        return format_error_response(
            error_message=f"Health status formatting failed: {e!s}",
            error_type="formatting_error",
        )


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
            supported_document_types=["PDF", "TXT", "MD"],
            supported_formats=["PDF", "TXT", "MD"],
            search_types=["semantic", "keyword", "hybrid"],
            max_results=20,
            max_token_budget=16000,
        )

        corpus_info: dict[str, Any] = {
            "name": "VX Underground Collection",
            "description": (
                "Cybersecurity technical documents, articles, and research papers"
            ),
            "content_types": [
                "Technical articles",
                "Research papers",
                "Code samples",
                "Security analyses",
            ],
            # Intentionally static: this reflects the corpus index date, not today's date.
            "index_date": "unknown",
        }

        response = SystemContextResponse(
            system_name="VX-RAG",
            capabilities=capabilities,
            version="1.0.0",
            corpus_info=corpus_info,
        )

        json_str = response.model_dump_json(indent=2, exclude_none=True)

        logger.debug("System context formatted")

        return json_str

    except _SAFE_EXCEPTIONS as e:
        logger.error(f"Failed to format system context: {e}")
        return format_error_response(
            error_message=f"System context formatting failed: {e!s}",
            error_type="formatting_error",
        )


def format_error_response(
    error_message: str,
    error_type: str = "internal_error",
    details: dict[str, Any] | None = None,
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
            details=details,
        )

        json_str = response.model_dump_json(indent=2, exclude_none=True)

        logger.debug(
            "Error response formatted",
            error_type=error_type,
            error_msg=error_message,
        )

        return json_str

    except _SAFE_EXCEPTIONS as e:
        logger.error(f"Failed to format error response: {e}")
        return json.dumps(
            {
                "error": error_message,
                "error_type": error_type,
                "formatting_error": str(e),
            },
            indent=2,
        )
