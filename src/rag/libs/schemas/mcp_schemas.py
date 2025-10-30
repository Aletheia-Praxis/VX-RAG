"""
MCP (Model Context Protocol) schemas for context assembly.

Defines Pydantic models for MCP-compatible payloads used in RAG system.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime


class ContextItem(BaseModel):
    """Individual context item in MCP payload."""
    
    id: str = Field(..., description="Unique identifier for the context item")
    text: str = Field(..., description="Text content of the context item")
    score: Optional[float] = Field(None, description="Relevance score (0.0 to 1.0)")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata for the context item")
    
    @field_validator('score')
    @classmethod
    def validate_score(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError('Score must be between 0.0 and 1.0')
        return v


class MCPContextPayload(BaseModel):
    """MCP-compatible context payload for LLM queries."""
    
    schema_version: str = Field("1.0", description="MCP schema version")
    context: List[ContextItem] = Field(..., description="List of context items")
    query: str = Field(..., description="Original user query")
    token_budget: int = Field(2048, description="Maximum token budget for context")
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Payload creation timestamp")
    provenance: Dict[str, Any] = Field(default_factory=dict, description="Provenance information")
    
    @field_validator('schema_version')
    @classmethod
    def validate_schema_version(cls, v: str) -> str:
        if v != "1.0":
            raise ValueError('Only schema version 1.0 is supported')
        return v
    
    @field_validator('token_budget')
    @classmethod
    def validate_token_budget(cls, v: int) -> int:
        if v <= 0:
            raise ValueError('Token budget must be positive')
        return v
    
    def total_tokens_estimate(self) -> int:
        """Estimate total tokens in context (rough approximation)."""
        # Rough approximation: 1 token ≈ 4 characters for English text
        total_chars = sum(len(item.text) for item in self.context)
        return total_chars // 4
    
    def is_within_budget(self) -> bool:
        """Check if payload is within token budget."""
        return self.total_tokens_estimate() <= self.token_budget


class ContextAssemblyRequest(BaseModel):
    """Request model for context assembly."""
    
    query: str = Field(..., description="User query")
    documents: List[Dict[str, Any]] = Field(..., description="Retrieved documents with metadata")
    token_budget: Optional[int] = Field(2048, description="Token budget")
    max_items: Optional[int] = Field(10, description="Maximum number of context items")
    min_score: Optional[float] = Field(0.0, description="Minimum relevance score to include")


class ContextAssemblyResponse(BaseModel):
    """Response model for context assembly."""
    
    payload: MCPContextPayload
    selected_count: int = Field(..., description="Number of selected context items")
    total_tokens: int = Field(..., description="Estimated total tokens")
    within_budget: bool = Field(..., description="Whether within token budget")
