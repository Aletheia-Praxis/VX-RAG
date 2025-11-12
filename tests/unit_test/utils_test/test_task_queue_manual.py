"""
Manual test script for TaskQueue with asyncio.to_thread().
"""

import asyncio
import time
from src.utils.task_queue import TaskQueue, TaskPriority


def sync_task(value: int, delay: float = 0.1) -> int:
    """Synchronous CPU-bound task."""
    print(f"[SYNC] Starting task with value={value}")
    time.sleep(delay)
    result = value * 2
    print(f"[SYNC] Completed task with value={value}, result={result}")
    return result


async def async_task(value: int, delay: float = 0.1) -> int:
    """Asynchronous I/O-bound task."""
    print(f"[ASYNC] Starting task with value={value}")
    await asyncio.sleep(delay)
    result = value * 3
    print(f"[ASYNC] Completed task with value={value}, result={result}")
    return result


async def main():
    """Test TaskQueue with both sync and async tasks."""
    print("=== TaskQueue Manual Test ===\n")
    
    # Create queue
    queue = TaskQueue(max_concurrent_tasks=2)
    await queue.start()
    print("✓ Queue started\n")
    
    # Submit sync tasks
    print("--- Submitting sync tasks ---")
    sync_task_ids = []
    for i in range(3):
        task_id = await queue.submit_task(
            name=f"sync_task_{i}",
            func=sync_task,
            args=(i,),
            priority=TaskPriority.NORMAL,
        )
        sync_task_ids.append(task_id)
        print(f"✓ Submitted sync task {i}: {task_id}")
    
    # Submit async tasks
    print("\n--- Submitting async tasks ---")
    async_task_ids = []
    for i in range(3):
        task_id = await queue.submit_task(
            name=f"async_task_{i}",
            func=async_task,
            args=(i + 10,),
            priority=TaskPriority.HIGH,
        )
        async_task_ids.append(task_id)
        print(f"✓ Submitted async task {i}: {task_id}")
    
    # Wait for completion
    print("\n--- Waiting for tasks to complete ---")
    await asyncio.sleep(3.0)
    
    # Check stats
    print("\n--- Queue Statistics ---")
    stats = queue.get_queue_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Check task statuses
    print("\n--- Task Statuses ---")
    for task_id in sync_task_ids + async_task_ids:
        status = await queue.get_task_status(task_id)
        if status:
            print(f"  {status['name']}: {status['status']} (result: {status['result']})")
    
    # Stop queue
    print("\n--- Stopping queue ---")
    await queue.stop()
    print("✓ Queue stopped")
    
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
