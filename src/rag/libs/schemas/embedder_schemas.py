"""Embedder schemas for the RAG system.

This module defines Pydantic models for embedding-related data structures,
ensuring type safety and validation for the embedder service.
"""

from typing import List
from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    """Request model for embedding generation.

    Attributes:
        texts: List of text strings to embed.
    """
    texts: List[str] = Field(..., description="List of text strings to generate embeddings for")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation.

    Attributes:
        embeddings: List of embedding vectors, each as a list of floats.
        model_name: Name of the embedding model used.
        processing_time: Time taken to generate embeddings in seconds.
    """
    embeddings: List[List[float]] = Field(..., description="List of embedding vectors")
    model_name: str = Field(..., description="Name of the embedding model used")
    processing_time: float = Field(..., description="Processing time in seconds")


class EmbeddingVector(BaseModel):
    """Model for a single embedding vector.

    Attributes:
        vector: The embedding vector as a list of floats.
        text: Original text that was embedded.
    """
    vector: List[float] = Field(..., description="Embedding vector")
    text: str = Field(..., description="Original text")
