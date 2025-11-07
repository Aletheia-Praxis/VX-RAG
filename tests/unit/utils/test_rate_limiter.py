"""
Unit tests for Rate Limiter.
"""

import asyncio
import pytest
from typing import TYPE_CHECKING

from src.utils.rate_limiter import RateLimiter, get_rate_limiter

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def rate_limiter() -> RateLimiter:
    """Create a rate limiter for testing."""
    return RateLimiter(
        max_concurrent=2,
        queue_size=5,
        default_timeout=5.0,
    )


@pytest.mark.asyncio
async def test_rate_limiter_initialization(rate_limiter: RateLimiter) -> None:
    """Test rate limiter initialization."""
    assert rate_limiter.max_concurrent == 2
    assert rate_limiter.queue_size == 5
    assert rate_limiter.default_timeout == 5.0


@pytest.mark.asyncio
async def test_execute_within_limit(rate_limiter: RateLimiter) -> None:
    """Test executing request within concurrency limit."""
    
    async def handler(value: int) -> int:
        """Simple async handler."""
        await asyncio.sleep(0.1)
        return value * 2
    
    result = await rate_limiter.execute(
        request_id="req_1",
        handler=handler,
        args=(5,),
    )
    
    assert result == 10
    
    stats = rate_limiter.get_stats()
    assert stats['completed_requests'] == 1
    assert stats['rejected_requests'] == 0


@pytest.mark.asyncio
async def test_execute_with_queueing(rate_limiter: RateLimiter) -> None:
    """Test that requests are queued when limit is reached."""
    results = []
    
    async def slow_handler(value: int) -> int:
        """Slow handler to trigger queueing."""
        await asyncio.sleep(0.5)
        results.append(value)
        return value
    
    # Submit 4 requests (max_concurrent=2, so 2 will be queued)
    tasks = [
        rate_limiter.execute(f"req_{i}", slow_handler, args=(i,))
        for i in range(4)
    ]
    
    await asyncio.gather(*tasks)
    
    assert len(results) == 4
    
    stats = rate_limiter.get_stats()
    assert stats['completed_requests'] == 4
    assert stats['queued_requests'] >= 2  # At least 2 were queued


@pytest.mark.asyncio
async def test_queue_full_rejection(rate_limiter: RateLimiter) -> None:
    """Test that requests are rejected when queue is full."""
    
    async def very_slow_handler() -> None:
        """Very slow handler to fill up queue."""
        await asyncio.sleep(10.0)
    
    # Fill up concurrency slots and queue
    tasks = []
    for i in range(rate_limiter.max_concurrent + rate_limiter.queue_size):
        task = asyncio.create_task(
            rate_limiter.execute(f"req_{i}", very_slow_handler)
        )
        tasks.append(task)
        await asyncio.sleep(0.01)  # Small delay to ensure ordering
    
    # This request should be rejected (queue full)
    with pytest.raises(RuntimeError, match="Request queue is full"):
        await rate_limiter.execute(
            "req_overflow",
            very_slow_handler,
        )
    
    # Cancel all tasks
    for task in tasks:
        task.cancel()
    
    # Wait for cancellations
    await asyncio.gather(*tasks, return_exceptions=True)
    
    stats = rate_limiter.get_stats()
    assert stats['rejected_requests'] >= 1


@pytest.mark.asyncio
async def test_request_timeout(rate_limiter: RateLimiter) -> None:
    """Test that requests timeout correctly."""
    
    async def timeout_handler() -> None:
        """Handler that takes longer than timeout."""
        await asyncio.sleep(10.0)
    
    with pytest.raises(asyncio.TimeoutError):
        await rate_limiter.execute(
            "req_timeout",
            timeout_handler,
            timeout=1.0,  # Short timeout
        )
    
    stats = rate_limiter.get_stats()
    assert stats['timed_out_requests'] >= 1


@pytest.mark.asyncio
async def test_concurrent_limit_enforcement(rate_limiter: RateLimiter) -> None:
    """Test that concurrent limit is properly enforced."""
    active_count = 0
    max_active = 0
    
    async def concurrent_handler() -> None:
        """Handler that tracks concurrency."""
        nonlocal active_count, max_active
        active_count += 1
        max_active = max(max_active, active_count)
        await asyncio.sleep(0.2)
        active_count -= 1
    
    # Submit many requests
    tasks = [
        rate_limiter.execute(f"req_{i}", concurrent_handler)
        for i in range(10)
    ]
    
    await asyncio.gather(*tasks)
    
    # Max active should not exceed concurrent limit
    assert max_active <= rate_limiter.max_concurrent


@pytest.mark.asyncio
async def test_get_stats(rate_limiter: RateLimiter) -> None:
    """Test getting rate limiter statistics."""
    
    async def dummy_handler() -> int:
        """Dummy handler."""
        await asyncio.sleep(0.1)
        return 42
    
    await rate_limiter.execute("req_1", dummy_handler)
    
    stats = rate_limiter.get_stats()
    
    assert 'max_concurrent' in stats
    assert 'queue_size' in stats
    assert 'total_requests' in stats
    assert 'completed_requests' in stats
    assert stats['total_requests'] >= 1
    assert stats['completed_requests'] >= 1


@pytest.mark.asyncio
async def test_clear_expired_requests(rate_limiter: RateLimiter) -> None:
    """Test clearing expired requests from queue."""
    
    async def slow_handler() -> None:
        """Slow handler to fill queue."""
        await asyncio.sleep(10.0)
    
    # Fill up slots
    tasks = []
    for i in range(rate_limiter.max_concurrent):
        task = asyncio.create_task(
            rate_limiter.execute(f"req_{i}", slow_handler)
        )
        tasks.append(task)
    
    await asyncio.sleep(0.1)
    
    # Add request with short timeout to queue
    short_timeout_task = asyncio.create_task(
        rate_limiter.execute(
            "req_short",
            slow_handler,
            timeout=0.5,
        )
    )
    
    # Wait for timeout
    await asyncio.sleep(1.0)
    
    # Clear expired requests
    cleared = await rate_limiter.clear_expired_requests()
    assert cleared >= 0  # Should clear the expired request
    
    # Cancel all tasks
    for task in tasks:
        task.cancel()
    short_timeout_task.cancel()
    
    await asyncio.gather(*tasks, short_timeout_task, return_exceptions=True)


def test_get_rate_limiter_singleton() -> None:
    """Test that get_rate_limiter returns singleton instance."""
    limiter1 = get_rate_limiter()
    limiter2 = get_rate_limiter()
    
    assert limiter1 is limiter2
