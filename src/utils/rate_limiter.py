"""
Rate Limiter for VX-RAG MCP Server.

Implements simple queue-based rate limiting for incoming requests
as specified in section 7.1 of the technical standard.

For single-user local system, provides:
- Request queueing when concurrent limit is reached
- Configurable concurrent request limit
- Request timeout handling
- Queue statistics and monitoring
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Optional, Dict, Tuple
from collections import deque

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QueuedRequest:
    """Represents a queued request."""
    request_id: str
    handler: Callable[..., Coroutine[Any, Any, Any]]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    queued_at: float
    timeout: float
    
    def is_expired(self) -> bool:
        """Check if request has exceeded timeout."""
        return (time.time() - self.queued_at) > self.timeout


class RateLimiter:
    """
    Simple queue-based rate limiter for MCP server.
    
    Implements specification requirement 7.1:
    "A simple queueing mechanism will be implemented. If the (limited) 
    concurrent request limit is exceeded, new requests will be queued 
    rather than dropped."
    
    Features:
    - FIFO queue for incoming requests
    - Configurable concurrent request limit
    - Request timeout handling
    - Queue statistics
    
    Example:
        >>> limiter = RateLimiter(max_concurrent=2)
        >>> 
        >>> async def handle_query(query: str):
        ...     # Process query
        ...     return {"result": "..."}
        >>> 
        >>> result = await limiter.execute(
        ...     "req_123",
        ...     handle_query,
        ...     args=("what is malware?",),
        ...     timeout=60.0
        ... )
    """
    
    def __init__(
        self,
        max_concurrent: int = 2,
        queue_size: int = 10,
        default_timeout: float = 600.0,
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_concurrent: Maximum number of concurrent requests
            queue_size: Maximum queue size (additional requests are rejected)
            default_timeout: Default timeout for requests in seconds
        """
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        self.default_timeout = default_timeout
        
        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)
        
        # Request queue
        self._queue: deque[QueuedRequest] = deque(maxlen=queue_size)
        
        # Statistics
        self._total_requests = 0
        self._queued_requests = 0
        self._rejected_requests = 0
        self._timed_out_requests = 0
        self._completed_requests = 0
        
        # Lock for queue operations
        self._queue_lock = asyncio.Lock()
        
        logger.info(
            "RateLimiter initialized",
            max_concurrent=max_concurrent,
            queue_size=queue_size,
            default_timeout=default_timeout,
        )
    
    async def execute(
        self,
        request_id: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Any:
        """
        Execute request with rate limiting.
        
        If concurrent limit is reached, request is queued.
        If queue is full, request is rejected.
        
        Args:
            request_id: Unique request identifier
            handler: Async callable to execute
            args: Positional arguments for handler
            kwargs: Keyword arguments for handler
            timeout: Request timeout (uses default if None)
            
        Returns:
            Result from handler
            
        Raises:
            asyncio.TimeoutError: If request times out
            RuntimeError: If queue is full
        """
        kwargs = kwargs or {}
        timeout = timeout or self.default_timeout
        
        self._total_requests += 1
        
        # Try to acquire semaphore immediately
        if self._semaphore.locked():
            # Concurrent limit reached, queue the request
            logger.info(
                f"Concurrent limit reached, queueing request",
                request_id=request_id,
                queue_size=len(self._queue),
            )
            
            async with self._queue_lock:
                if len(self._queue) >= self.queue_size:
                    self._rejected_requests += 1
                    logger.warning(
                        f"Queue full, rejecting request",
                        request_id=request_id,
                        queue_size=len(self._queue),
                    )
                    raise RuntimeError(f"Request queue is full (max: {self.queue_size})")
                
                request = QueuedRequest(
                    request_id=request_id,
                    handler=handler,
                    args=args,
                    kwargs=kwargs,
                    queued_at=time.time(),
                    timeout=timeout,
                )
                self._queue.append(request)
                self._queued_requests += 1
            
            # Wait for slot to become available
            await self._wait_for_slot(request)
        
        # Execute request with semaphore
        async with self._semaphore:
            try:
                logger.info(f"Executing request", request_id=request_id)
                
                result = await asyncio.wait_for(
                    handler(*args, **kwargs),
                    timeout=timeout,
                )
                
                self._completed_requests += 1
                logger.info(f"Request completed", request_id=request_id)
                
                return result
                
            except asyncio.TimeoutError:
                self._timed_out_requests += 1
                logger.error(
                    f"Request timed out",
                    request_id=request_id,
                    timeout=timeout,
                )
                raise
            
            except Exception as e:
                logger.error(
                    f"Request failed",
                    request_id=request_id,
                    error=str(e),
                    exc_info=True,
                )
                raise
    
    async def _wait_for_slot(self, request: QueuedRequest) -> None:
        """
        Wait for a processing slot to become available.
        
        Args:
            request: Queued request
            
        Raises:
            asyncio.TimeoutError: If request times out while waiting
        """
        wait_start = time.time()
        
        while True:
            # Check if request has expired
            if request.is_expired():
                self._timed_out_requests += 1
                async with self._queue_lock:
                    if request in self._queue:
                        self._queue.remove(request)
                
                logger.error(
                    f"Request timed out while waiting in queue",
                    request_id=request.request_id,
                    wait_time=time.time() - wait_start,
                )
                raise asyncio.TimeoutError("Request timed out in queue")
            
            # Check if slot is available
            if not self._semaphore.locked():
                async with self._queue_lock:
                    if request in self._queue:
                        self._queue.remove(request)
                break
            
            # Wait a bit before checking again
            await asyncio.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiter statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'max_concurrent': self.max_concurrent,
            'queue_size': self.queue_size,
            'current_queue_length': len(self._queue),
            'total_requests': self._total_requests,
            'queued_requests': self._queued_requests,
            'rejected_requests': self._rejected_requests,
            'timed_out_requests': self._timed_out_requests,
            'completed_requests': self._completed_requests,
            'current_concurrent': self.max_concurrent - self._semaphore._value,
        }
    
    async def clear_expired_requests(self) -> int:
        """
        Clear expired requests from queue.
        
        Returns:
            Number of cleared requests
        """
        async with self._queue_lock:
            expired = [req for req in self._queue if req.is_expired()]
            
            for req in expired:
                self._queue.remove(req)
                self._timed_out_requests += 1
                logger.warning(
                    f"Removed expired request from queue",
                    request_id=req.request_id,
                    queued_time=time.time() - req.queued_at,
                )
            
            return len(expired)


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(
    max_concurrent: int = 2,
    queue_size: int = 10,
    default_timeout: float = 600.0,
) -> RateLimiter:
    """
    Get or create the global rate limiter instance.
    
    Args:
        max_concurrent: Maximum concurrent requests
        queue_size: Maximum queue size
        default_timeout: Default request timeout
        
    Returns:
        RateLimiter instance
    """
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            max_concurrent=max_concurrent,
            queue_size=queue_size,
            default_timeout=default_timeout,
        )
    
    return _rate_limiter
