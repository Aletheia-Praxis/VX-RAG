"""
FastMCP server module for VX-RAG system.

Provides MCP (Model Context Protocol) interface for querying the RAG system.
Delegates all RAG operations to the MCP bridge for clean separation of concerns.
"""

import json
import time
import asyncio
import uuid
import datetime
from typing import Dict, Any, Optional
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from .bridge import get_mcp_bridge

# Import structured logging and metrics
from src.utils.logging_config import get_logger, log_query_event, log_service_health
from src.utils.metrics import get_metrics
from src.utils.config_loader import get_mcp_config
from src.utils.rate_limiter import get_rate_limiter
from src.utils.task_queue import get_task_queue, TaskPriority

# Get structured logger and metrics
logger = get_logger("mcp_server")
metrics = get_metrics()

# Global MCP bridge instance
mcp_bridge = get_mcp_bridge()

# Initialize rate limiter for concurrent request control
rate_limiter = get_rate_limiter(
    max_concurrent=2,       # Max 2 concurrent queries
    queue_size=10,          # Queue up to 10 requests
    default_timeout=600.0,  # 10 minutes default timeout
)

# Initialize task queue for background operations
task_queue = get_task_queue(
    max_workers=4,              # Max 4 ThreadPoolExecutor workers
    max_concurrent_tasks=2,     # Max 2 concurrent tasks
    max_completed_tasks=100,    # Keep last 100 completed tasks
    max_failed_tasks=50,        # Keep last 50 failed tasks
)


# Create FastMCP server
mcp = FastMCP(
    name="VX-RAG MCP Server"
)


# Pydantic models for tool parameters
class QueryParams(BaseModel):
    """Parameters for document query tool."""
    query: str = Field(..., description="The search query")
    top_k: int = Field(3, ge=1, le=10, description="Number of top results to return")


class IngestParams(BaseModel):
    """Parameters for document ingestion tool."""
    data_dir: str = Field("data/raw", description="Directory containing documents to ingest")
    config_path: str = Field("config/settings.yaml", description="Path to configuration file")
    background: bool = Field(True, description="Run ingestion in background")


class TaskStatusParams(BaseModel):
    """Parameters for task status check."""
    task_id: str = Field(..., description="Unique task identifier")


class TaskCancelParams(BaseModel):
    """Parameters for task cancellation."""
    task_id: str = Field(..., description="Unique task identifier to cancel")


class TaskListParams(BaseModel):
    """Parameters for listing tasks."""
    filter: str = Field(
        "all",
        description="Filter tasks by status: all, pending, running, completed, failed, cancelled"
    )
    limit: int = Field(50, ge=1, le=500, description="Maximum number of tasks to return")


class SnapshotCreateParams(BaseModel):
    """Parameters for creating index snapshot."""
    persist_dir: str = Field("data/index", description="Directory containing the index to snapshot")
    snapshot_name: Optional[str] = Field(None, description="Optional custom name for snapshot (default: timestamp)")
    config_path: str = Field("config/settings.yaml", description="Path to configuration file")


class SnapshotVerifyParams(BaseModel):
    """Parameters for verifying snapshot integrity."""
    persist_dir: str = Field("data/index", description="Directory containing the index snapshots")
    snapshot_name: str = Field(..., description="Name of the snapshot to verify")


# Helper function for background ingestion
def run_ingestion_pipeline(data_dir: str, config_path: str) -> Dict[str, Any]:
    """
    Run the full ingestion pipeline (synchronous, for TaskQueue).
    
    This is a blocking function that will be executed in ThreadPoolExecutor.
    
    Args:
        data_dir: Directory containing raw documents
        config_path: Path to configuration file
        
    Returns:
        Dictionary with ingestion results
    """
    from pathlib import Path
    from src.rag.services.ingest_service.service import (
        PDFIngestAdapter, 
        TXTIngestAdapter, 
        MDIngestAdapter,
        save_processed_text
    )
    from src.rag.services.duplicate_detection_service.service import DuplicateDetector
    from src.rag.services.chunker_service.service import Chunker
    import json
    
    start_time = time.time()
    data_path = Path(data_dir)
    processed_dir = data_path / "processed"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting background ingestion", data_dir=str(data_path))
    
    # Step 1: Parse documents
    pdf_adapter = PDFIngestAdapter()
    txt_adapter = TXTIngestAdapter()
    md_adapter = MDIngestAdapter()
    
    pdf_docs = pdf_adapter.load_data(str(data_path / "pdf"))
    txt_docs = txt_adapter.load_data(str(data_path / "txt"))
    md_docs = md_adapter.load_data(str(data_path / "md"))
    
    all_docs = pdf_docs + txt_docs + md_docs
    
    # Step 2: Remove duplicates
    detector = DuplicateDetector(config_path=config_path)
    unique_docs = detector.remove_duplicates(all_docs)
    
    # Step 3: Save processed documents
    saved_count = save_processed_text(unique_docs, processed_dir)
    
    # Step 4: Chunk documents
    chunker = Chunker(config_path=config_path)
    chunks = chunker.chunk_documents(unique_docs)
    
    # Step 5: Save chunks
    def make_serializable(obj: Any) -> Any:
        """Recursively remove non-serializable objects."""
        if isinstance(obj, dict):
            return {
                k: make_serializable(v) 
                for k, v in obj.items() 
                if k not in ['node_info', 'relationships', 'excluded_llm_metadata_keys', 
                             'excluded_embed_metadata_keys', 'metadata_seperator', 
                             'metadata_template', 'text_template']
            }
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    chunks_file = data_path / "processed" / "chunks.json"
    serializable_chunks = make_serializable(chunks)
    
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_chunks, f, ensure_ascii=False, indent=2)
    
    duration = time.time() - start_time
    
    logger.info(
        "Background ingestion completed", 
        parsed=len(all_docs),
        duplicates_removed=len(all_docs) - len(unique_docs),
        unique=len(unique_docs),
        saved=saved_count,
        chunks=len(chunks),
        duration_sec=round(duration, 2)
    )
    
    return {
        'status': 'completed',
        'documents_parsed': len(all_docs),
        'duplicates_removed': len(all_docs) - len(unique_docs),
        'unique_documents': len(unique_docs),
        'documents_saved': saved_count,
        'chunks_created': len(chunks),
        'chunks_file': str(chunks_file),
        'duration_seconds': round(duration, 2)
    }


@mcp.tool
async def query_documents(params: QueryParams) -> str:
    """
    Query the RAG system for relevant documents and generate a response.
    
    Performs semantic search on the document index and uses LLM to generate
    a contextual response based on the most relevant documents.
    
    Rate limited to max 2 concurrent queries with queue for additional requests.
    
    Args:
        params: Query parameters including query text and number of results
        
    Returns:
        JSON-formatted response with query results, LLM-generated answer, and sources
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Load timeout from config
    mcp_config = get_mcp_config()
    query_timeout = mcp_config.get('query_timeout', 600.0)
    
    try:
        logger.info(
            "Processing MCP query with rate limiting",
            request_id=request_id,
            query=params.query,
            top_k=params.top_k,
        )
        
        # Define query handler
        async def _execute_query() -> Dict[str, Any]:
            return mcp_bridge.query_documents(params.query, params.top_k)
        
        # Execute with rate limiting and timeout
        response = await rate_limiter.execute(
            request_id=request_id,
            handler=_execute_query,
            timeout=query_timeout,
        )
        
        # Convert to JSON string for MCP response
        json_response = json.dumps(response, indent=2, ensure_ascii=False)
        
        duration = time.time() - start_time
        results_count = len(response.get('sources', []))
        
        # Log structured event and metrics
        log_query_event(params.query, params.top_k, results_count, duration)
        
        logger.info(
            "MCP query completed successfully", 
            request_id=request_id,
            query=params.query, 
            results_count=results_count, 
            duration_ms=duration * 1000,
        )
        
        return json_response
        
    except asyncio.TimeoutError:
        duration = time.time() - start_time
        logger.error(
            f"MCP query timed out after {query_timeout}s",
            request_id=request_id,
            query=params.query, 
            timeout=query_timeout,
            duration_ms=duration * 1000,
        )
        error_response = {
            "error": f"Query processing timed out after {query_timeout}s",
            "query": params.query,
            "sources": []
        }
        return json.dumps(error_response, indent=2, ensure_ascii=False)
    
    except RuntimeError as e:
        # Queue full error
        duration = time.time() - start_time
        logger.error(
            "MCP query queue full",
            request_id=request_id,
            query=params.query,
            error=str(e),
            duration_ms=duration * 1000,
        )
        error_response = {
            "error": "Server is busy, request queue is full. Please try again later.",
            "query": params.query,
            "sources": []
        }
        return json.dumps(error_response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "MCP query failed",
            request_id=request_id,
            query=params.query, 
            duration_ms=duration * 1000, 
            error=str(e),
            exc_info=True,
        )
        error_response = {
            "error": f"Query processing failed: {str(e)}",
            "query": params.query,
            "sources": []
        }
        return json.dumps(error_response, indent=2, ensure_ascii=False)


@mcp.tool
async def ingest_documents(params: IngestParams) -> str:
    """
    Ingest documents into the RAG system.
    
    Runs the full ingestion pipeline: parsing, deduplication, saving, and chunking.
    Can run in background mode for large datasets.
    
    Args:
        params: Ingestion parameters including data directory and configuration
        
    Returns:
        JSON-formatted response with task ID (if background) or ingestion results
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        logger.info(
            "Processing ingestion request",
            request_id=request_id,
            data_dir=params.data_dir,
            background=params.background,
        )
        
        if params.background:
            # Submit to task queue for background processing
            task_id = await task_queue.submit_task(
                name="ingest_documents",
                func=run_ingestion_pipeline,
                args=(params.data_dir, params.config_path),
                priority=TaskPriority.HIGH,
                max_retries=1,  # Heavy operation, limit retries
            )
            
            response = {
                "status": "submitted",
                "task_id": task_id,
                "message": "Ingestion started in background. Use vxrag_task_status to check progress.",
                "data_dir": params.data_dir,
            }
            
            logger.info(
                "Ingestion task submitted",
                request_id=request_id,
                task_id=task_id,
            )
            
        else:
            # Run synchronously (blocking)
            logger.warning(
                "Running ingestion synchronously (blocking)",
                request_id=request_id,
            )
            
            result = run_ingestion_pipeline(params.data_dir, params.config_path)
            result['task_id'] = request_id
            response = result
            
            logger.info(
                "Ingestion completed synchronously",
                request_id=request_id,
                duration_sec=result['duration_seconds'],
            )
        
        duration = time.time() - start_time
        metrics.increment("mcp_ingest_requests_total")
        metrics.histogram("mcp_ingest_request_duration_ms", duration * 1000)
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Ingestion request failed",
            request_id=request_id,
            data_dir=params.data_dir,
            error=str(e),
            duration_ms=duration * 1000,
            exc_info=True,
        )
        
        error_response = {
            "error": f"Ingestion failed: {str(e)}",
            "data_dir": params.data_dir,
            "request_id": request_id,
        }
        
        return json.dumps(error_response, indent=2, ensure_ascii=False)


@mcp.tool
async def get_task_status(params: TaskStatusParams) -> str:
    """
    Get the status of a background task.
    
    Returns detailed information about task progress, result, or error.
    
    Args:
        params: Task status parameters including task ID
        
    Returns:
        JSON-formatted task status information
    """
    try:
        logger.info("Checking task status", task_id=params.task_id)
        
        task_status = await task_queue.get_task_status(params.task_id)
        
        if task_status is None:
            response = {
                "error": "Task not found",
                "task_id": params.task_id,
            }
        else:
            response = {
                "task_id": params.task_id,
                "status": task_status['status'],
                "name": task_status['name'],
                "created_at": task_status['created_at'],
                "started_at": task_status['started_at'],
                "completed_at": task_status['completed_at'],
                "result": task_status['result'],
                "error": task_status['error'],
            }
            
            # Calculate duration if available
            if task_status['started_at'] and task_status['completed_at']:
                duration = task_status['completed_at'] - task_status['started_at']
                response['duration_seconds'] = round(duration, 2)
        
        logger.info("Task status retrieved", task_id=params.task_id, status=task_status['status'] if task_status else 'not_found')
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error("Failed to get task status", task_id=params.task_id, error=str(e))
        
        error_response = {
            "error": f"Failed to get task status: {str(e)}",
            "task_id": params.task_id,
        }
        
        return json.dumps(error_response, indent=2, ensure_ascii=False)


@mcp.tool
async def cancel_task(params: TaskCancelParams) -> str:
    """
    Cancel a pending or running background task.
    
    Attempts to cancel the specified task. Tasks that are already completed,
    failed, or cancelled cannot be cancelled.
    
    Args:
        params: Task cancellation parameters including task ID
        
    Returns:
        JSON-formatted cancellation result
    """
    try:
        logger.info("Attempting to cancel task", task_id=params.task_id)
        
        success = await task_queue.cancel_task(params.task_id)
        
        if success:
            response = {
                "success": True,
                "task_id": params.task_id,
                "message": "Task cancelled successfully",
            }
            logger.info("Task cancelled successfully", task_id=params.task_id)
        else:
            # Task not found or already in terminal state
            task_status = await task_queue.get_task_status(params.task_id)
            
            if task_status is None:
                response = {
                    "success": False,
                    "task_id": params.task_id,
                    "error": "Task not found",
                }
            else:
                response = {
                    "success": False,
                    "task_id": params.task_id,
                    "error": f"Cannot cancel task in '{task_status['status']}' state",
                    "current_status": task_status['status'],
                }
            
            logger.warning(
                "Failed to cancel task",
                task_id=params.task_id,
                reason=response.get('error', 'unknown'),
            )
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error("Error cancelling task", task_id=params.task_id, error=str(e))
        
        error_response = {
            "success": False,
            "task_id": params.task_id,
            "error": f"Failed to cancel task: {str(e)}",
        }
        
        return json.dumps(error_response, indent=2, ensure_ascii=False)


@mcp.tool
async def list_tasks(params: TaskListParams) -> str:
    """
    List all tasks in the queue with optional filtering.
    
    Returns a list of tasks matching the specified filter criteria.
    Tasks are sorted by creation time (newest first).
    
    Args:
        params: Task list parameters including filter and limit
        
    Returns:
        JSON-formatted list of tasks
    """
    try:
        logger.info("Listing tasks", filter=params.filter, limit=params.limit)
        
        # Get queue statistics
        stats = task_queue.get_queue_stats()
        
        # Get all tasks from the internal registry
        all_tasks = []
        for task_id, task_data in task_queue._tasks.items():
            task_dict = task_data.to_dict()
            all_tasks.append(task_dict)
        
        # Filter tasks by status
        if params.filter != "all":
            filtered_tasks = [
                task for task in all_tasks 
                if task['status'] == params.filter
            ]
        else:
            filtered_tasks = all_tasks
        
        # Sort by creation time (newest first)
        filtered_tasks.sort(key=lambda t: t['created_at'] or 0, reverse=True)
        
        # Apply limit
        limited_tasks = filtered_tasks[:params.limit]
        
        response = {
            "tasks": limited_tasks,
            "total_matching": len(filtered_tasks),
            "returned": len(limited_tasks),
            "filter": params.filter,
            "queue_stats": stats,
        }
        
        logger.info(
            "Tasks listed",
            filter=params.filter,
            total_matching=len(filtered_tasks),
            returned=len(limited_tasks),
        )
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        logger.error("Error listing tasks", filter=params.filter, error=str(e))
        
        error_response = {
            "error": f"Failed to list tasks: {str(e)}",
            "tasks": [],
            "total_matching": 0,
            "returned": 0,
        }
        
        return json.dumps(error_response, indent=2, ensure_ascii=False)


@mcp.tool
async def create_snapshot(params: SnapshotCreateParams) -> str:
    """
    Create a versioned snapshot of the vector index.
    
    Creates a timestamped backup of the FAISS index with a manifest file
    containing metadata and checksums for integrity verification.
    
    Args:
        params: Snapshot creation parameters including persist directory and optional name
        
    Returns:
        JSON-formatted response with snapshot path and details
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        from pathlib import Path
        from src.rag.services.vectordb_service.service import VectorStoreClient
        from src.rag.services.embedder_service.service import EmbeddingService
        
        logger.info(
            "Creating snapshot",
            request_id=request_id,
            persist_dir=params.persist_dir,
            snapshot_name=params.snapshot_name,
        )
        
        persist_dir = Path(params.persist_dir)
        faiss_index_path = persist_dir / "faiss_index"
        
        if not faiss_index_path.exists():
            error_response = {
                "error": f"Index not found at {faiss_index_path}",
                "persist_dir": params.persist_dir,
            }
            logger.error("Snapshot creation failed: index not found", persist_dir=params.persist_dir)
            return json.dumps(error_response, indent=2, ensure_ascii=False)
        
        # Load embedder for model info
        embedder = EmbeddingService(config_path=params.config_path)
        embed_model_info = embedder.get_model_info()
        
        # Load vector store
        vector_client = VectorStoreClient(
            store_type="faiss",
            config={'index_dir': str(faiss_index_path)}
        )
        vector_client.load_index(embed_model=embedder.embed_model)
        
        if not vector_client.index:
            error_response = {
                "error": "Failed to load index for snapshot creation",
                "persist_dir": params.persist_dir,
            }
            logger.error("Failed to load index", persist_dir=params.persist_dir)
            return json.dumps(error_response, indent=2, ensure_ascii=False)
        
        # Create snapshot (returns bool)
        success = vector_client.create_snapshot(
            snapshot_name=params.snapshot_name,
            embed_model_info=embed_model_info
        )
        
        if not success:
            error_response = {
                "error": "Failed to create snapshot",
                "persist_dir": params.persist_dir,
            }
            logger.error("Snapshot creation failed", request_id=request_id, persist_dir=params.persist_dir)
            return json.dumps(error_response, indent=2, ensure_ascii=False)
        
        # Construct snapshot path (method returns bool, not Path)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        snapshot_name_final = params.snapshot_name or f"snapshot_{timestamp}"
        snapshot_dir = Path(params.persist_dir).parent / "snapshots" / snapshot_name_final
        
        duration = time.time() - start_time
        
        response = {
            "success": True,
            "snapshot_path": str(snapshot_dir),
            "snapshot_name": snapshot_name_final,
            "persist_dir": params.persist_dir,
            "duration_seconds": round(duration, 2),
        }
        
        logger.info(
            "Snapshot created successfully",
            request_id=request_id,
            snapshot_path=str(snapshot_dir),
            snapshot_name=snapshot_name_final,
            duration_ms=duration * 1000,
        )
        
        metrics.increment("mcp_snapshots_created_total")
        metrics.histogram("mcp_snapshot_creation_duration_ms", duration * 1000)
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Snapshot creation failed",
            request_id=request_id,
            persist_dir=params.persist_dir,
            error=str(e),
            duration_ms=duration * 1000,
            exc_info=True,
        )
        
        error_response = {
            "error": f"Snapshot creation failed: {str(e)}",
            "persist_dir": params.persist_dir,
        }
        
        return json.dumps(error_response, indent=2, ensure_ascii=False)


@mcp.tool
async def verify_snapshot(params: SnapshotVerifyParams) -> str:
    """
    Verify the integrity of a snapshot.
    
    Checks snapshot files against stored checksums in the manifest
    to ensure data integrity and completeness.
    
    Args:
        params: Snapshot verification parameters including persist directory and snapshot name
        
    Returns:
        JSON-formatted verification result with status and details
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        from pathlib import Path
        from src.rag.services.vectordb_service.service import VectorStoreClient
        
        logger.info(
            "Verifying snapshot",
            request_id=request_id,
            snapshot_name=params.snapshot_name,
            persist_dir=params.persist_dir,
        )
        
        persist_dir = Path(params.persist_dir)
        faiss_index_path = persist_dir / "faiss_index"
        
        if not faiss_index_path.exists():
            error_response = {
                "error": f"Index directory not found at {faiss_index_path}",
                "persist_dir": params.persist_dir,
            }
            logger.error("Verification failed: index directory not found", persist_dir=params.persist_dir)
            return json.dumps(error_response, indent=2, ensure_ascii=False)
        
        # Initialize vector store client
        vector_client = VectorStoreClient(
            store_type="faiss",
            config={'index_dir': str(faiss_index_path)}
        )
        
        # Verify snapshot
        result = vector_client.verify_snapshot_integrity(params.snapshot_name)
        
        duration = time.time() - start_time
        
        response = {
            "snapshot_name": params.snapshot_name,
            "valid": result['valid'],
            "errors": result.get('errors', []),
            "manifest": result.get('manifest', {}),
            "duration_seconds": round(duration, 2),
        }
        
        logger.info(
            "Snapshot verification completed",
            request_id=request_id,
            snapshot_name=params.snapshot_name,
            valid=result['valid'],
            duration_ms=duration * 1000,
        )
        
        metrics.increment("mcp_snapshot_verifications_total")
        
        return json.dumps(response, indent=2, ensure_ascii=False)
        
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            "Snapshot verification failed",
            request_id=request_id,
            snapshot_name=params.snapshot_name,
            error=str(e),
            duration_ms=duration * 1000,
            exc_info=True,
        )
        
        error_response = {
            "error": f"Snapshot verification failed: {str(e)}",
            "snapshot_name": params.snapshot_name,
            "valid": "false",
        }
        
        return json.dumps(error_response, indent=2, ensure_ascii=False)


@mcp.resource("health://status")
def get_health_status() -> str:
    """
    Get the health status of the RAG system.
    
    Returns:
        JSON-formatted health status including index load state, rate limiter, and task queue stats
    """
    try:
        # Delegate to MCP bridge
        health_data = mcp_bridge.get_health_status()
        
        # Parse health data
        health_dict = json.loads(health_data)
        
        # Add rate limiter statistics
        health_dict['rate_limiter'] = rate_limiter.get_stats()
        
        # Add task queue statistics
        health_dict['task_queue'] = task_queue.get_queue_stats()
        
        # Log service health
        log_service_health("mcp_server", "healthy")
        
        return json.dumps(health_dict, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        log_service_health("mcp_server", "error", error=str(e))
        return json.dumps({"status": "error", "error": str(e)})


@mcp.resource("context://system")
def get_system_context() -> str:
    """
    Get information about the RAG system's capabilities and context.
    
    Returns:
        JSON-formatted system information
    """
    try:
        # Delegate to MCP bridge
        return mcp_bridge.get_system_context()
    except Exception as e:
        logger.error("System context retrieval failed", error=str(e))
        return json.dumps({"error": f"Failed to retrieve system context: {str(e)}"})


async def start_server() -> None:
    """Start MCP server with task queue."""
    logger.info("Starting VX-RAG MCP server")
    log_service_health("mcp_server", "starting")
    
    # Start task queue
    await task_queue.start()
    logger.info("Task queue started")
    
    # Run MCP server (blocking)
    try:
        mcp.run()
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down MCP server")
        await task_queue.stop()
        logger.info("Task queue stopped")
        log_service_health("mcp_server", "stopped")


if __name__ == "__main__":
    asyncio.run(start_server())
