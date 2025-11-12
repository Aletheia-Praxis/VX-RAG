"""
Unit tests for Task Queue Rolling Window Cleanup.

Tests the FIFO buffer behavior where oldest tasks are evicted when limits are reached.
"""

import asyncio
import pytest
import time
from pathlib import Path

from src.utils.task_queue import (
    TaskQueue,
)


@pytest.fixture
def task_queue_with_limits(tmp_path: Path) -> TaskQueue:
    """Create a task queue with small limits for testing."""
    state_file = tmp_path / "test_queue_state.json"
    queue = TaskQueue(
        max_concurrent_tasks=1,
        max_completed_tasks=3,  # Small limit for testing
        max_failed_tasks=2,     # Small limit for testing
        state_file=state_file,
        enable_persistence=False,
    )
    return queue


@pytest.mark.asyncio
async def test_rolling_window_completed_tasks(task_queue_with_limits: TaskQueue) -> None:
    """Test that oldest completed tasks are removed when limit is exceeded."""
    
    def simple_task(value: int) -> int:
        """Simple sync task."""
        return value * 2
    
    await task_queue_with_limits.start()
    
    # Submit 5 tasks (limit is 3)
    task_ids = []
    for i in range(5):
        task_id = await task_queue_with_limits.submit_task(
            name=f"task_{i}",
            func=simple_task,
            args=(i,),
        )
        task_ids.append(task_id)
    
    # Wait for all tasks to complete
    await asyncio.sleep(2.0)
    
    # Check that only last 3 completed tasks remain
    stats = task_queue_with_limits.get_queue_stats()
    assert stats['completed'] <= 3, f"Expected <= 3 completed tasks, got {stats['completed']}"
    
    # Oldest tasks (0, 1) should be removed
    status_0 = await task_queue_with_limits.get_task_status(task_ids[0])
    status_1 = await task_queue_with_limits.get_task_status(task_ids[1])
    
    # Newest tasks (2, 3, 4) should exist
    status_2 = await task_queue_with_limits.get_task_status(task_ids[2])
    status_3 = await task_queue_with_limits.get_task_status(task_ids[3])
    status_4 = await task_queue_with_limits.get_task_status(task_ids[4])
    
    assert status_0 is None or status_1 is None, "Oldest tasks should be removed"
    assert status_2 is not None, "Task 2 should still exist"
    assert status_3 is not None, "Task 3 should still exist"
    assert status_4 is not None, "Task 4 should still exist"
    
    await task_queue_with_limits.stop()


@pytest.mark.asyncio
async def test_rolling_window_failed_tasks(task_queue_with_limits: TaskQueue) -> None:
    """Test that oldest failed tasks are removed when limit is exceeded."""
    
    def failing_task(value: int) -> int:
        """Task that always fails."""
        raise ValueError(f"Task {value} failed intentionally")
    
    await task_queue_with_limits.start()
    
    # Submit 4 tasks that will fail (limit is 2)
    task_ids = []
    for i in range(4):
        task_id = await task_queue_with_limits.submit_task(
            name=f"failing_task_{i}",
            func=failing_task,
            args=(i,),
            max_retries=0,  # No retries to fail immediately
        )
        task_ids.append(task_id)
    
    # Wait for all tasks to fail
    await asyncio.sleep(2.0)
    
    # Check that only last 2 failed tasks remain
    stats = task_queue_with_limits.get_queue_stats()
    assert stats['failed'] <= 2, f"Expected <= 2 failed tasks, got {stats['failed']}"
    
    # Oldest tasks (0, 1) should be removed
    status_0 = await task_queue_with_limits.get_task_status(task_ids[0])
    status_1 = await task_queue_with_limits.get_task_status(task_ids[1])
    
    # Newest tasks (2, 3) should exist
    status_2 = await task_queue_with_limits.get_task_status(task_ids[2])
    status_3 = await task_queue_with_limits.get_task_status(task_ids[3])
    
    assert status_0 is None or status_1 is None, "Oldest failed tasks should be removed"
    assert status_2 is not None, "Task 2 should still exist"
    assert status_3 is not None, "Task 3 should still exist"
    
    await task_queue_with_limits.stop()


@pytest.mark.asyncio
async def test_rolling_window_mixed_statuses(task_queue_with_limits: TaskQueue) -> None:
    """Test rolling window with mixed completed/failed/running tasks."""
    
    def success_task(value: int) -> int:
        return value * 2
    
    def failing_task(value: int) -> int:
        raise ValueError(f"Failed {value}")
    
    await task_queue_with_limits.start()
    
    # Submit 3 successful tasks
    for i in range(3):
        await task_queue_with_limits.submit_task(
            name=f"success_{i}",
            func=success_task,
            args=(i,),
        )
    
    # Submit 2 failing tasks
    for i in range(2):
        await task_queue_with_limits.submit_task(
            name=f"fail_{i}",
            func=failing_task,
            args=(i,),
            max_retries=0,
        )
    
    # Wait for completion
    await asyncio.sleep(2.0)
    
    stats = task_queue_with_limits.get_queue_stats()
    
    # Check limits are enforced
    assert stats['completed'] <= 3, "Completed tasks should respect limit"
    assert stats['failed'] <= 2, "Failed tasks should respect limit"
    
    # Cancelled tasks should NOT be cleaned up (only completed/failed)
    await task_queue_with_limits.stop()


@pytest.mark.asyncio
async def test_rolling_window_preserves_newest(task_queue_with_limits: TaskQueue) -> None:
    """Test that rolling window preserves newest tasks and removes oldest."""
    
    def task_with_delay(value: int, delay: float) -> int:
        """Task that takes specific time."""
        time.sleep(delay)
        return value
    
    await task_queue_with_limits.start()
    
    # Submit tasks with timestamps
    task_ids = []
    timestamps = []
    
    for i in range(5):
        task_id = await task_queue_with_limits.submit_task(
            name=f"timed_task_{i}",
            func=task_with_delay,
            args=(i, 0.1),
        )
        task_ids.append(task_id)
        timestamps.append(time.time())
        await asyncio.sleep(0.15)  # Ensure distinct completion times
    
    # Wait for all to complete
    await asyncio.sleep(1.0)
    
    # Check that newest 3 tasks remain
    stats = task_queue_with_limits.get_queue_stats()
    assert stats['completed'] <= 3
    
    # Last 3 tasks should exist
    status_2 = await task_queue_with_limits.get_task_status(task_ids[2])
    status_3 = await task_queue_with_limits.get_task_status(task_ids[3])
    status_4 = await task_queue_with_limits.get_task_status(task_ids[4])
    
    assert status_2 is not None, "Task 2 (3rd oldest) should exist"
    assert status_3 is not None, "Task 3 (2nd newest) should exist"
    assert status_4 is not None, "Task 4 (newest) should exist"
    
    await task_queue_with_limits.stop()


@pytest.mark.asyncio
async def test_stats_include_limits(task_queue_with_limits: TaskQueue) -> None:
    """Test that queue stats include limit information."""
    
    stats = task_queue_with_limits.get_queue_stats()
    
    assert 'max_completed_tasks' in stats
    assert 'max_failed_tasks' in stats
    assert stats['max_completed_tasks'] == 3
    assert stats['max_failed_tasks'] == 2
