"""
Simple test for MCP task management tools.

Tests the new cancel_task and list_tasks MCP tools.
"""

import asyncio

from src.utils.task_queue import TaskPriority, get_task_queue


def dummy_task() -> str:
    """Dummy task for testing."""
    import time
    time.sleep(2)
    return "Task completed successfully"


async def test_task_management():
    """Test task management tools."""
    print("\n" + "="*60)
    print("TESTING MCP TASK MANAGEMENT")
    print("="*60)
    
    # Get task queue instance
    task_queue = get_task_queue()
    
    # Start task queue
    await task_queue.start()
    print("Task queue started")
    
    # Submit test tasks
    print("\n[1/5] Submitting test tasks...")
    task_ids = []
    
    for i in range(3):
        task_id = await task_queue.submit_task(
            name=f"test_task_{i+1}",
            func=dummy_task,
            priority=TaskPriority.NORMAL,
            max_retries=1,
        )
        task_ids.append(task_id)
        print(f"  Task {i+1} submitted: {task_id[:8]}...")
    
    # Get queue stats
    print("\n[2/5] Queue statistics:")
    stats = task_queue.get_queue_stats()
    print(f"  Total tasks: {stats['total_tasks']}")
    print(f"  Pending: {stats['pending']}")
    print(f"  Running: {stats['running']}")
    print(f"  Completed: {stats['completed']}")
    
    # List all tasks
    print("\n[3/5] Listing all tasks...")
    all_tasks = []
    for task_id, task_data in task_queue._tasks.items():
        all_tasks.append(task_data.to_dict())
    
    print(f"  Found {len(all_tasks)} tasks:")
    for task in all_tasks[:3]:
        print(f"    - {task['name']}: {task['status']}")
    
    # Cancel second task
    print(f"\n[4/5] Cancelling task: {task_ids[1][:8]}...")
    success = await task_queue.cancel_task(task_ids[1])
    print(f"  Cancellation {'successful' if success else 'failed'}")
    
    # Check status of cancelled task
    task_status = await task_queue.get_task_status(task_ids[1])
    if task_status:
        print(f"  Task status: {task_status['status']}")
    
    # Wait a bit for tasks to process
    print("\n[5/5] Waiting for remaining tasks to complete...")
    await asyncio.sleep(5)
    
    # Final stats
    stats = task_queue.get_queue_stats()
    print("\nFinal statistics:")
    print(f"  Completed: {stats['completed']}")
    print(f"  Cancelled: {stats['cancelled']}")
    print(f"  Failed: {stats['failed']}")
    
    # Stop task queue
    await task_queue.stop()
    print("\nTask queue stopped")
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_task_management())
