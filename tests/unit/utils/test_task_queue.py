"""
Unit tests for Task Queue system.
"""

import asyncio
import pytest
from pathlib import Path
from typing import TYPE_CHECKING

from src.utils.task_queue import (
    TaskQueue,
    Task,
    TaskStatus,
    TaskPriority,
    get_task_queue,
)

if TYPE_CHECKING:
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def task_queue(tmp_path: Path) -> TaskQueue:
    """Create a task queue for testing."""
    state_file = tmp_path / "test_queue_state.json"
    queue = TaskQueue(
        max_workers=2,
        max_concurrent_tasks=1,
        state_file=state_file,
        enable_persistence=False,  # Disable for faster tests
    )
    return queue


@pytest.mark.asyncio
async def test_task_queue_initialization(task_queue: TaskQueue) -> None:
    """Test task queue initialization."""
    assert task_queue.max_workers == 2
    assert task_queue.max_concurrent_tasks == 1
    assert not task_queue._running


@pytest.mark.asyncio
async def test_task_queue_start_stop(task_queue: TaskQueue) -> None:
    """Test starting and stopping the task queue."""
    await task_queue.start()
    assert task_queue._running
    
    await task_queue.stop()
    assert not task_queue._running


@pytest.mark.asyncio
async def test_submit_and_execute_task(task_queue: TaskQueue) -> None:
    """Test submitting and executing a simple task."""
    
    def simple_task(value: int) -> int:
        """Simple sync task."""
        return value * 2
    
    await task_queue.start()
    
    task_id = await task_queue.submit_task(
        name="simple_task",
        func=simple_task,
        args=(5,),
    )
    
    # Wait for task to complete
    await asyncio.sleep(0.5)
    
    status = await task_queue.get_task_status(task_id)
    assert status is not None
    assert status['status'] == TaskStatus.COMPLETED.value
    assert status['result'] == '10'
    
    await task_queue.stop()


@pytest.mark.asyncio
async def test_submit_async_task(task_queue: TaskQueue) -> None:
    """Test submitting and executing an async task."""
    
    async def async_task(value: int) -> int:
        """Simple async task."""
        await asyncio.sleep(0.1)
        return value * 3
    
    await task_queue.start()
    
    task_id = await task_queue.submit_task(
        name="async_task",
        func=async_task,
        args=(7,),
    )
    
    # Wait for task to complete
    await asyncio.sleep(0.5)
    
    status = await task_queue.get_task_status(task_id)
    assert status is not None
    assert status['status'] == TaskStatus.COMPLETED.value
    assert status['result'] == '21'
    
    await task_queue.stop()


@pytest.mark.asyncio
async def test_task_priority(task_queue: TaskQueue) -> None:
    """Test that high priority tasks execute before low priority tasks."""
    results = []
    
    def tracked_task(value: int) -> int:
        """Task that tracks execution order."""
        results.append(value)
        return value
    
    await task_queue.start()
    
    # Submit tasks with different priorities
    await task_queue.submit_task(
        name="low_priority",
        func=tracked_task,
        args=(1,),
        priority=TaskPriority.LOW,
    )
    
    await task_queue.submit_task(
        name="high_priority",
        func=tracked_task,
        args=(2,),
        priority=TaskPriority.HIGH,
    )
    
    await task_queue.submit_task(
        name="normal_priority",
        func=tracked_task,
        args=(3,),
        priority=TaskPriority.NORMAL,
    )
    
    # Wait for all tasks to complete
    await asyncio.sleep(1.0)
    
    # High priority should execute first
    assert results[0] == 2
    
    await task_queue.stop()


@pytest.mark.asyncio
async def test_task_failure_and_retry(task_queue: TaskQueue) -> None:
    """Test task failure and retry mechanism."""
    attempt_count = 0
    
    def failing_task() -> int:
        """Task that fails first attempt, succeeds on second."""
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 2:
            raise ValueError("Temporary failure")
        return 42
    
    await task_queue.start()
    
    task_id = await task_queue.submit_task(
        name="failing_task",
        func=failing_task,
        max_retries=3,
    )
    
    # Wait for retries - give more time for retry to complete
    await asyncio.sleep(10.0)
    
    status = await task_queue.get_task_status(task_id)
    assert status is not None
    assert status['status'] == TaskStatus.COMPLETED.value
    assert attempt_count == 2  # Failed once, succeeded on retry
    
    await task_queue.stop()


@pytest.mark.asyncio
async def test_task_cancellation(task_queue: TaskQueue) -> None:
    """Test cancelling a pending task."""
    
    async def long_task() -> int:
        """Long-running task."""
        await asyncio.sleep(10.0)
        return 123
    
    await task_queue.start()
    
    task_id = await task_queue.submit_task(
        name="long_task",
        func=long_task,
    )
    
    # Cancel task before it completes
    await asyncio.sleep(0.1)
    cancelled = await task_queue.cancel_task(task_id)
    assert cancelled
    
    status = await task_queue.get_task_status(task_id)
    assert status is not None
    assert status['status'] == TaskStatus.CANCELLED.value
    
    await task_queue.stop()


@pytest.mark.asyncio
async def test_queue_stats(task_queue: TaskQueue) -> None:
    """Test queue statistics."""
    
    def dummy_task() -> None:
        """Dummy task."""
        pass
    
    await task_queue.start()
    
    # Submit some tasks
    await task_queue.submit_task("task1", dummy_task)
    await task_queue.submit_task("task2", dummy_task)
    await task_queue.submit_task("task3", dummy_task)
    
    stats = await task_queue.get_queue_stats()
    
    assert stats['total_tasks'] == 3
    assert stats['max_workers'] == 2
    assert stats['max_concurrent_tasks'] == 1
    
    await task_queue.stop()


@pytest.mark.asyncio
async def test_concurrency_limit(task_queue: TaskQueue) -> None:
    """Test that concurrency limit is enforced."""
    active_tasks = 0
    max_active = 0
    
    async def concurrent_task() -> None:
        """Task that tracks concurrency."""
        nonlocal active_tasks, max_active
        active_tasks += 1
        max_active = max(max_active, active_tasks)
        await asyncio.sleep(0.2)
        active_tasks -= 1
    
    await task_queue.start()
    
    # Submit multiple tasks
    for i in range(5):
        await task_queue.submit_task(
            f"task_{i}",
            concurrent_task,
        )
    
    # Wait for all tasks to complete
    await asyncio.sleep(2.0)
    
    # Max concurrent should not exceed limit
    assert max_active <= task_queue.max_concurrent_tasks
    
    await task_queue.stop()


def test_get_task_queue_singleton() -> None:
    """Test that get_task_queue returns singleton instance."""
    queue1 = get_task_queue()
    queue2 = get_task_queue()
    
    assert queue1 is queue2
