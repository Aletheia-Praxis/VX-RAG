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
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Optional, Dict, Tuple
from collections import deque

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QueuedRequest:
    """
    Represents a queued request waiting for a processing slot.

    Attributes:
        request_id: Unique identifier for the request.
        handler: Async callable that will process the request.
        args: Positional arguments passed to ``handler``.
        kwargs: Keyword arguments passed to ``handler``.
        queued_at: Unix timestamp when the request entered the queue.
        timeout: Maximum number of seconds the request may wait before expiring.
        slot_available: Event signalled when a processing slot opens up.
    """

    request_id: str
    handler: Callable[..., Coroutine[Any, Any, Any]]
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    queued_at: float
    timeout: float
    # Each queued request owns its own Event so it can be woken up individually
    # by the slot-release path, eliminating the busy-wait polling loop.
    slot_available: asyncio.Event = field(default_factory=asyncio.Event)

    def is_expired(self) -> bool:
        """Return ``True`` if the request has exceeded its timeout budget."""
        return (time.time() - self.queued_at) > self.timeout


class RateLimiter:
    """
    Simple queue-based rate limiter for MCP server.

    Implements specification requirement 7.1:
    "A simple queueing mechanism will be implemented. If the (limited)
    concurrent request limit is exceeded, new requests will be queued
    rather than dropped."

    Design notes
    ------------
    * Uses ``asyncio.Semaphore`` for concurrency control.
    * Maintains a separate ``_active_count`` integer so that
      ``get_stats()`` never touches the private ``_semaphore._value``
      attribute (which is not part of asyncio's public API).
    * Queued requests wait on a per-request ``asyncio.Event`` instead of
      polling every 100 ms, so there is no busy-wait overhead on the
      event loop.

    Example
    -------
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
    ) -> None:
        """
        Initialise rate limiter.

        Args:
            max_concurrent: Maximum number of concurrent requests.
            queue_size: Maximum queue size (additional requests are rejected).
            default_timeout: Default timeout for requests in seconds.
        """
        self.max_concurrent = max_concurrent
        self.queue_size = queue_size
        self.default_timeout = default_timeout

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Own counter so we never access the private _semaphore._value (B-10)
        self._active_count: int = 0

        # Request queue
        self._queue: deque[QueuedRequest] = deque(maxlen=queue_size)

        # Statistics
        self._total_requests: int = 0
        self._queued_requests: int = 0
        self._rejected_requests: int = 0
        self._timed_out_requests: int = 0
        self._completed_requests: int = 0

        # Lock for queue and _active_count mutations
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
        Execute a request with rate limiting.

        If the concurrent limit is reached the request is queued.
        If the queue is full the request is rejected immediately.

        The implementation avoids a TOCTOU race (B-08) by always acquiring
        the semaphore directly — queueing only gates on whether other
        callers are already waiting, not on a racy ``semaphore.locked()``
        read.

        Args:
            request_id: Unique request identifier.
            handler: Async callable to execute.
            args: Positional arguments for ``handler``.
            kwargs: Keyword arguments for ``handler``.
            timeout: Request timeout (uses ``default_timeout`` if ``None``).

        Returns:
            Result from ``handler``.

        Raises:
            asyncio.TimeoutError: If the request times out.
            RuntimeError: If the queue is full.
        """
        kwargs = kwargs or {}
        effective_timeout = timeout or self.default_timeout

        self._total_requests += 1

        # If other requests are already queued, join the back of the queue so
        # ordering is preserved (FIFO). We do NOT check semaphore.locked() here
        # because that read is not atomic with the subsequent acquire (B-08).
        should_queue: bool
        async with self._queue_lock:
            should_queue = bool(self._queue)

        if should_queue:
            await self._enqueue_and_wait(
                request_id=request_id,
                handler=handler,
                args=args,
                kwargs=kwargs,
                timeout=effective_timeout,
            )

        # Acquire the semaphore — this is the single authoritative gate
        try:
            acquired = await asyncio.wait_for(
                self._semaphore.acquire(), timeout=effective_timeout
            )
        except asyncio.TimeoutError:
            self._timed_out_requests += 1
            logger.error(
                "Request timed out waiting for semaphore",
                request_id=request_id,
                timeout=effective_timeout,
            )
            raise

        async with self._queue_lock:
            self._active_count += 1

        try:
            logger.info("Executing request", request_id=request_id)

            result = await asyncio.wait_for(
                handler(*args, **kwargs),
                timeout=effective_timeout,
            )

            self._completed_requests += 1
            logger.info("Request completed", request_id=request_id)
            return result

        except asyncio.TimeoutError:
            self._timed_out_requests += 1
            logger.error(
                "Request timed out during execution",
                request_id=request_id,
                timeout=effective_timeout,
            )
            raise

        except Exception as e:
            logger.error(
                "Request failed",
                request_id=request_id,
                error=str(e),
                exc_info=True,
            )
            raise

        finally:
            async with self._queue_lock:
                self._active_count -= 1
                # Wake the next waiting request in FIFO order
                if self._queue:
                    next_request = self._queue[0]
                    next_request.slot_available.set()
            self._semaphore.release()

    async def _enqueue_and_wait(
        self,
        request_id: str,
        handler: Callable[..., Coroutine[Any, Any, Any]],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        timeout: float,
    ) -> None:
        """
        Place this request in the FIFO queue and wait for a slot signal.

        Uses a per-request ``asyncio.Event`` instead of a polling loop so
        that no CPU time is wasted on the event loop while waiting (B-09).

        Args:
            request_id: Unique request identifier.
            handler: Async callable (stored for future reference / stats).
            args: Positional arguments for the handler.
            kwargs: Keyword arguments for the handler.
            timeout: Maximum wait time in seconds.

        Raises:
            RuntimeError: If the queue is already at capacity.
            asyncio.TimeoutError: If the request expires while waiting.
        """
        async with self._queue_lock:
            if len(self._queue) >= self.queue_size:
                self._rejected_requests += 1
                logger.warning(
                    "Queue full, rejecting request",
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

        logger.info(
            "Concurrent limit reached, queueing request",
            request_id=request_id,
            queue_length=len(self._queue),
        )

        remaining_timeout = timeout - (time.time() - request.queued_at)
        try:
            await asyncio.wait_for(
                request.slot_available.wait(), timeout=max(0.0, remaining_timeout)
            )
        except asyncio.TimeoutError:
            self._timed_out_requests += 1
            async with self._queue_lock:
                try:
                    self._queue.remove(request)
                except ValueError:
                    pass  # already removed by the slot-release path
            logger.error(
                "Request timed out while waiting in queue",
                request_id=request_id,
            )
            raise asyncio.TimeoutError("Request timed out in queue")
        finally:
            # Always remove from queue (covers the normal wakeup path too)
            async with self._queue_lock:
                try:
                    self._queue.remove(request)
                except ValueError:
                    pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Return current rate-limiter statistics.

        Uses the internal ``_active_count`` counter instead of the private
        ``asyncio.Semaphore._value`` attribute to stay on the public API.

        Returns:
            Dictionary with runtime statistics.
        """
        return {
            "max_concurrent": self.max_concurrent,
            "queue_size": self.queue_size,
            "current_queue_length": len(self._queue),
            "total_requests": self._total_requests,
            "queued_requests": self._queued_requests,
            "rejected_requests": self._rejected_requests,
            "timed_out_requests": self._timed_out_requests,
            "completed_requests": self._completed_requests,
            "current_concurrent": self._active_count,
        }

    async def clear_expired_requests(self) -> int:
        """
        Remove expired requests from the queue.

        Returns:
            Number of requests that were removed.
        """
        async with self._queue_lock:
            expired = [req for req in self._queue if req.is_expired()]

            for req in expired:
                self._queue.remove(req)
                self._timed_out_requests += 1
                logger.warning(
                    "Removed expired request from queue",
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
        max_concurrent: Maximum concurrent requests.
        queue_size: Maximum queue size.
        default_timeout: Default request timeout.

    Returns:
        The global ``RateLimiter`` instance.
    """
    global _rate_limiter

    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            max_concurrent=max_concurrent,
            queue_size=queue_size,
            default_timeout=default_timeout,
        )

    return _rate_limiter
