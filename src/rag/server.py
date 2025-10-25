"""
FastAPI server module for VX-RAG system.

Provides REST API endpoints for querying the RAG system.
"""

import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .query import DocumentQuery

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global query handler
query_handler: Optional[DocumentQuery] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""
    global query_handler
    logger.info("Starting VX-RAG API server")
    
    # Initialize query handler
    query_handler = DocumentQuery()
    if not query_handler.load_index():
        logger.warning("Failed to load index on startup. Queries may not work.")
    
    yield
    
    logger.info("Shutting down VX-RAG API server")


# Create FastAPI app
app = FastAPI(
    title="VX-RAG API",
    description="Retrieval-Augmented Generation API for VX Underground documents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for document queries."""
    query: str = Field(..., description="The search query")
    top_k: int = Field(3, ge=1, le=10, description="Number of top results to return")


class QueryResult(BaseModel):
    """Model for individual query result."""
    text: str = Field(..., description="Extracted text snippet")
    score: float = Field(..., description="Similarity score")
    metadata: dict = Field(..., description="Document metadata")


class QueryResponse(BaseModel):
    """Response model for document queries."""
    query: str = Field(..., description="Original query")
    response: str = Field(..., description="Generated response from LLM")
    results: List[QueryResult] = Field(..., description="Top-k similar documents")
    total_results: int = Field(..., description="Total number of results returned")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    index_loaded: bool = Field(..., description="Whether the FAISS index is loaded")
    version: str = Field(..., description="API version")


class ContextResponse(BaseModel):
    """Response model for system context."""
    description: str = Field(..., description="System description")
    capabilities: List[str] = Field(..., description="System capabilities")
    supported_formats: List[str] = Field(..., description="Supported document formats")


@app.get("/", summary="Root endpoint")
async def root():
    """Get basic API information."""
    return {
        "message": "VX-RAG API - Retrieval-Augmented Generation for VX Underground",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check():
    """Check the health status of the RAG system."""
    global query_handler
    
    index_loaded = query_handler is not None and query_handler.index is not None
    
    status = "healthy" if index_loaded else "degraded"
    
    return HealthResponse(
        status=status,
        index_loaded=index_loaded,
        version="1.0.0"
    )


@app.post("/query", response_model=QueryResponse, summary="Query documents")
async def query_documents(request: QueryRequest):
    """
    Query the RAG system for relevant documents and generate a response.
    
    Performs semantic search on the document index and uses LLM to generate
    a contextual response based on the most relevant documents.
    """
    global query_handler
    
    if query_handler is None:
        raise HTTPException(status_code=503, detail="Query service not initialized")
    
    try:
        logger.info(f"Processing query: {request.query} (top_k={request.top_k})")
        
        # Perform query
        response, results = query_handler.query_documents(request.query, request.top_k)
        
        if not response and not results:
            raise HTTPException(status_code=500, detail="Query execution failed")
        
        # Convert results to response model
        query_results = [
            QueryResult(text=text, score=score, metadata=metadata)
            for text, score, metadata in results
        ]
        
        logger.info(f"Query completed: {len(query_results)} results returned")
        
        return QueryResponse(
            query=request.query,
            response=response,
            results=query_results,
            total_results=len(query_results)
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/context", response_model=ContextResponse, summary="Get system context")
async def get_context():
    """Get information about the RAG system's capabilities and context."""
    return ContextResponse(
        description="VX-RAG is a Retrieval-Augmented Generation system specialized in VX Underground technical documents.",
        capabilities=[
            "Semantic document search using FAISS vector database",
            "Text extraction from PDF documents",
            "LLM-powered response generation",
            "RESTful API for external integrations"
        ],
        supported_formats=["PDF"]
    )


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting VX-RAG API server on http://0.0.0.0:8000")
    logger.info("API documentation available at http://0.0.0.0:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)