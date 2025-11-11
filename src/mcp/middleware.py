"""
MCP Middleware - Rate limiting, logging, and metrics for MCP operations.

This module provides cross-cutting concerns for MCP tools:
- Rate limiting to prevent overload
- Request/response logging
- Performance metrics collection
- Error tracking

All MCP tool invocations pass through this middleware layer.
"""

import time
import asyncio
import uuid
from typing import Callable, Any, Dict, Optional
from functools import wraps

from src.utils.logging_config import get_logger
from src.utils.metrics import get_metrics
from src.utils.rate_limiter import get_rate_limiter
from src.utils.config_loader import get_mcp_rate_limit_config

logger = get_logger("mcp_middleware")
metrics = get_metrics()


# Load rate limiting configuration from settings.yaml
rate_limit_config = get_mcp_rate_limit_config()

# Initialize rate limiter for MCP operations
# Single-user system, but we still want to prevent runaway queries
mcp_rate_limiter = get_rate_limiter(
    max_concurrent=rate_limit_config['max_concurrent'],
    queue_size=rate_limit_config['queue_size'],
    default_timeout=rate_limit_config['default_timeout'],
)


def with_rate_limit(timeout: Optional[float] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to apply rate limiting to async functions.
    
    Args:
        timeout: Optional custom timeout for this operation (seconds)
        
    Usage:
        @with_rate_limit(timeout=300.0)
        async def my_handler(params):
            ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request_id = str(uuid.uuid4())
            
            try:
                # Execute with rate limiting
                result = await mcp_rate_limiter.execute(
                    request_id=request_id,
                    handler=lambda: func(*args, **kwargs),
                    timeout=timeout or mcp_rate_limiter.default_timeout,
                )
                return result
                
            except asyncio.TimeoutError:
                logger.error(
                    "Request timed out",
                    request_id=request_id,
                    function=func.__name__,
                    timeout=timeout or mcp_rate_limiter.default_timeout
                )
                metrics.increment("mcp_timeouts_total")
                raise
                
            except RuntimeError as e:
                # Queue full
                logger.error(
                    "Request queue full",
                    request_id=request_id,
                    function=func.__name__,
                    error=str(e)
                )
                metrics.increment("mcp_queue_full_total")
                raise
                
        return wrapper
    return decorator


def with_logging(tool_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to add structured logging to tool handlers.
    
    Logs:
    - Tool invocation (with parameters)
    - Execution duration
    - Success/failure status
    - Error details (if any)
    
    Args:
        tool_name: Name of the MCP tool
        
    Usage:
        @with_logging("query_knowledge_base")
        async def handle_query(...):
            ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            # Extract parameters for logging (first arg is usually params object)
            params_repr = None
            if args and hasattr(args[0], 'model_dump'):
                try:
                    params_repr = args[0].model_dump()
                except Exception:
                    params_repr = str(args[0])
            
            logger.info(
                f"MCP tool invoked: {tool_name}",
                request_id=request_id,
                tool=tool_name,
                params=params_repr
            )
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                logger.info(
                    f"MCP tool completed: {tool_name}",
                    request_id=request_id,
                    tool=tool_name,
                    duration_ms=duration * 1000,
                    status="success"
                )
                
                # Record success metric
                metrics.increment(f"mcp_tool_{tool_name}_success_total")
                metrics.histogram(f"mcp_tool_{tool_name}_duration_ms", duration * 1000)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                logger.error(
                    f"MCP tool failed: {tool_name}",
                    request_id=request_id,
                    tool=tool_name,
                    error=str(e),
                    duration_ms=duration * 1000,
                    status="error",
                    exc_info=True
                )
                
                # Record error metric
                metrics.increment(f"mcp_tool_{tool_name}_error_total")
                
                raise
                
        return wrapper
    return decorator


def with_metrics(metric_prefix: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator to collect detailed performance metrics.
    
    Records:
    - Invocation count
    - Duration histogram
    - Error rate
    
    Args:
        metric_prefix: Prefix for metric names
        
    Usage:
        @with_metrics("query")
        async def handle_query(...):
            ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            
            # Increment invocation counter
            metrics.increment(f"{metric_prefix}_invocations_total")
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record successful execution
                metrics.histogram(f"{metric_prefix}_duration_ms", duration * 1000)
                metrics.increment(f"{metric_prefix}_success_total")
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failed execution
                metrics.histogram(f"{metric_prefix}_duration_ms", duration * 1000)
                metrics.increment(f"{metric_prefix}_errors_total")
                
                raise
                
        return wrapper
    return decorator


def with_mcp_middleware(tool_name: str, timeout: Optional[float] = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Apply all middleware to an MCP tool handler.
    
    Combines:
    - Rate limiting
    - Structured logging
    - Performance metrics
    
    Args:
        tool_name: Name of the MCP tool
        timeout: Optional custom timeout (seconds)
        
    Usage:
        @with_mcp_middleware("query_knowledge_base", timeout=300.0)
        async def handle_query(params):
            ...
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        # Apply decorators in reverse order (innermost first)
        decorated: Callable[..., Any] = func
        decorated = with_metrics(f"mcp_{tool_name}")(decorated)
        decorated = with_logging(tool_name)(decorated)
        decorated = with_rate_limit(timeout)(decorated)
        return decorated
    return decorator


def get_rate_limiter_stats() -> Dict[str, Any]:
    """
    Get current rate limiter statistics.
    
    Returns:
        Dictionary with rate limiter stats:
        - current_load: Number of requests currently processing
        - queue_size: Number of requests in queue
        - max_concurrent: Maximum concurrent requests allowed
        - total_requests: Total requests processed
        - total_timeouts: Total requests that timed out
        - total_queue_full: Total requests rejected due to full queue
    """
    return mcp_rate_limiter.get_stats()


def reset_rate_limiter_stats() -> None:
    """Reset rate limiter statistics."""
    # Note: This would require adding a reset method to RateLimiter
    # For now, just log that stats were requested to be reset
    logger.info("Rate limiter stats reset requested")
    metrics.increment("mcp_rate_limiter_resets_total")
