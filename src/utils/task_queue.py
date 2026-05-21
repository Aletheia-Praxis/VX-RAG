"""
Task Queue System for VX-RAG.

Provides asyncio-based task queue with ThreadPoolExecutor for CPU-bound operations.
Implements background processing, rate limiting, and job management for single-user
local deployment without external dependencies like Celery/Redis.

Design Philosophy:
- Simple, lightweight, and optimized for local single-user scenarios
- No external message brokers (Redis, RabbitMQ)
- Persistent task state via JSON files for crash recovery
- Configurable concurrency limits and rate limiting
- Progress tracking and cancellation support
"""

import asyncio
import functools
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 5
    HIGH = 10
    CRITICAL = 20


@dataclass
class QueueTask:
    """
    Represents a single task in the queue.
    
    Attributes:
        task_id: Unique task identifier
        name: Human-readable task name
        func: Callable to execute (sync or async)
        args: Positional arguments for func
        kwargs: Keyword arguments for func
        priority: Task priority (higher = executes first)
        status: Current task status
        result: Task result (None until completed)
        error: Error message if failed
        created_at: Task creation timestamp
        started_at: Task start timestamp
        completed_at: Task completion timestamp
        retries: Number of retry attempts remaining
        max_retries: Maximum number of retries
    """
    task_id: str
    name: str
    func: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    kwargs: Optional[Dict[str, Any]] = None
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retries: int = 3
    max_retries: int = 3
    
    def __post_init__(self) -> None:
        """Initialize timestamps and kwargs."""
        if self.created_at is None:
            self.created_at = time.time()
        if self.kwargs is None:
            self.kwargs = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert task to dictionary for serialization.
        Excludes func as it's not serializable.
        """
        return {
            'task_id': self.task_id,
            'name': self.name,
            'priority': self.priority.value,
            'status': self.status.value,
            'result': str(self.result) if self.result is not None else None,
            'error': self.error,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'retries': self.retries,
            'max_retries': self.max_retries,
        }
    
    def __lt__(self, other: 'QueueTask') -> bool:
        """Compare tasks by priority for priority queue."""
        # Higher priority first, then older tasks first
        if self.priority != other.priority:
            return self.priority > other.priority
        # Safe comparison with fallback to 0.0 if None
        self_created = self.created_at if self.created_at is not None else 0.0
        other_created = other.created_at if other.created_at is not None else 0.0
        return self_created < other_created


class TaskQueue:
    """
    Asyncio-based task queue with thread pool support for CPU-bound operations.
    
    Features:
    - Priority-based task scheduling
    - Configurable concurrency limits (rate limiting)
    - Background task processing
    - Task persistence for crash recovery
    - Progress tracking and cancellation
    - Retry mechanism with exponential backoff
    - Automatic thread pool handling via asyncio.to_thread()
    
    Note:
        Synchronous functions are executed using asyncio.to_thread(), which uses
        Python's default thread pool. This avoids blocking the event loop while
        keeping the implementation simple and maintainable.
    
    Example:
        >>> queue = TaskQueue(max_concurrent_tasks=2)
        >>> await queue.start()
        >>> 
        >>> task_id = await queue.submit_task(
        ...     "process_pdf",
        ...     process_pdf_func,
        ...     args=("file.pdf",),
        ...     priority=TaskPriority.HIGH
        ... )
        >>> 
        >>> status = await queue.get_task_status(task_id)
        >>> await queue.stop()
    """
    
    def __init__(
        self,
        max_concurrent_tasks: int = 2,
        state_file: Optional[Path] = None,
        enable_persistence: bool = True,
        max_completed_tasks: int = 100,
        max_failed_tasks: int = 50,
    ):
        """
        Initialize task queue.
        
        Args:
            max_concurrent_tasks: Maximum number of tasks running simultaneously (rate limiting)
            state_file: Path to task state persistence file
            enable_persistence: Enable/disable task state persistence
            max_completed_tasks: Maximum number of completed tasks to keep (FIFO rolling window)
            max_failed_tasks: Maximum number of failed tasks to keep (FIFO rolling window)
        """
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_persistence = enable_persistence
        self.max_completed_tasks = max_completed_tasks
        self.max_failed_tasks = max_failed_tasks
        
        # Task queue (priority queue)
        self._queue: asyncio.PriorityQueue[QueueTask] = asyncio.PriorityQueue()
        
        # Task registry (all tasks by ID)
        self._tasks: Dict[str, QueueTask] = {}
        
        # Currently running tasks
        self._running_tasks: Dict[str, asyncio.Task[Any]] = {}
        
        # Control flags
        self._running = False
        self._worker_task: Optional[asyncio.Task[Any]] = None
        
        # Semaphore for rate limiting
        self._concurrency_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # State persistence path — directory is created lazily in start() (B-21)
        self.state_file = state_file or Path("data/task_queue_state.json")
        
        logger.info(
            "TaskQueue initialized",
            max_concurrent_tasks=max_concurrent_tasks,
            persistence=enable_persistence,
        )
    
    async def start(self) -> None:
        """Start the task queue worker."""
        if self._running:
            logger.warning("TaskQueue already running")
            return
        
        self._running = True

        # Create state directory here (not in __init__) to avoid blocking the
        # event loop before it starts (B-21).
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Load persisted tasks
        if self.enable_persistence:
            await self._load_state()

        # Start worker task
        self._worker_task = asyncio.create_task(self._worker_loop())

        logger.info("TaskQueue started")
    
    async def stop(self, timeout: float = 30.0) -> None:
        """
        Stop the task queue worker.
        
        Args:
            timeout: Maximum time to wait for tasks to complete
        """
        if not self._running:
            logger.warning("TaskQueue not running")
            return
        
        self._running = False
        
        # Wait for worker to finish
        if self._worker_task:
            try:
                await asyncio.wait_for(self._worker_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning("TaskQueue worker did not stop gracefully, cancelling")
                self._worker_task.cancel()
        
        # Cancel running tasks
        for task_id, task in self._running_tasks.items():
            logger.info(f"Cancelling running task: {task_id}")
            task.cancel()
        
        # Save state
        if self.enable_persistence:
            await self._save_state()
        
        logger.info("TaskQueue stopped")
    
    async def submit_task(
        self,
        name: str,
        func: Callable[..., Any],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
    ) -> str:
        """
        Submit a task to the queue.
        
        Args:
            name: Human-readable task name
            func: Callable to execute (sync or async)
            args: Positional arguments
            kwargs: Keyword arguments
            priority: Task priority
            max_retries: Maximum retry attempts
            
        Returns:
            task_id: Unique task identifier
        """
        task_id = str(uuid.uuid4())
        
        task = QueueTask(
            task_id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            max_retries=max_retries,
            retries=max_retries,
        )
        
        self._tasks[task_id] = task
        await self._queue.put(task)
        
        logger.info(
            f"Task submitted: {name}",
            task_id=task_id,
            priority=priority.value,
        )
        
        if self.enable_persistence:
            await self._save_state()
        
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get status of a task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status dictionary or None if not found
        """
        task = self._tasks.get(task_id)
        if task is None:
            return None
        
        return task.to_dict()
    
    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            True if cancelled, False if not found or already completed
        """
        task = self._tasks.get(task_id)
        if task is None:
            return False
        
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False
        
        # Cancel running task
        if task_id in self._running_tasks:
            self._running_tasks[task_id].cancel()
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()
        
        logger.info(f"Task cancelled: {task.name}", task_id=task_id)
        
        if self.enable_persistence:
            await self._save_state()
        
        return True
    
    async def _enforce_task_limits(self) -> None:
        """
        Enforce rolling window limits for completed/failed tasks.
        
        When limits are reached, removes oldest tasks (FIFO) to make room for new ones.
        This implements a rolling buffer where oldest entries are automatically evicted.
        """
        # Get completed and failed tasks sorted by completion time
        completed_tasks = [
            t for t in self._tasks.values() 
            if t.status == TaskStatus.COMPLETED
        ]
        failed_tasks = [
            t for t in self._tasks.values() 
            if t.status == TaskStatus.FAILED
        ]
        
        # Sort by completion time (oldest first)
        completed_tasks.sort(key=lambda t: t.completed_at or 0)
        failed_tasks.sort(key=lambda t: t.completed_at or 0)
        
        # Remove oldest completed tasks if over limit
        if len(completed_tasks) > self.max_completed_tasks:
            excess_count = len(completed_tasks) - self.max_completed_tasks
            for task in completed_tasks[:excess_count]:
                logger.info(
                    "Removed oldest completed task (rolling window)",
                    task_id=task.task_id,
                    task_name=task.name,
                    completed_at=task.completed_at
                )
                del self._tasks[task.task_id]
        
        # Remove oldest failed tasks if over limit
        if len(failed_tasks) > self.max_failed_tasks:
            excess_count = len(failed_tasks) - self.max_failed_tasks
            for task in failed_tasks[:excess_count]:
                logger.info(
                    "Removed oldest failed task (rolling window)",
                    task_id=task.task_id,
                    task_name=task.name,
                    completed_at=task.completed_at
                )
                del self._tasks[task.task_id]
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """
        Get queue statistics.
        
        Returns:
            Dictionary with queue statistics
        """
        pending = sum(1 for t in self._tasks.values() if t.status == TaskStatus.PENDING)
        running = sum(1 for t in self._tasks.values() if t.status == TaskStatus.RUNNING)
        completed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self._tasks.values() if t.status == TaskStatus.FAILED)
        cancelled = sum(1 for t in self._tasks.values() if t.status == TaskStatus.CANCELLED)
        
        return {
            'total_tasks': len(self._tasks),
            'pending': pending,
            'running': running,
            'completed': completed,
            'failed': failed,
            'cancelled': cancelled,
            'queue_size': self._queue.qsize(),
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'max_completed_tasks': self.max_completed_tasks,
            'max_failed_tasks': self.max_failed_tasks,
        }
    
    async def _worker_loop(self) -> None:
        """Main worker loop that processes tasks from the queue."""
        logger.info("TaskQueue worker loop started")
        
        while self._running:
            try:
                # Get next task (with timeout to check _running flag)
                try:
                    task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Skip cancelled tasks
                if task.status == TaskStatus.CANCELLED:
                    continue
                
                # Execute task with concurrency control
                asyncio.create_task(self._execute_task(task))
                
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
        
        logger.info("TaskQueue worker loop stopped")
    
    async def _execute_task(self, task: QueueTask) -> None:
        """
        Execute a single task with retry logic.
        
        Args:
            task: Task to execute
        """
        # Acquire semaphore for rate limiting
        async with self._concurrency_semaphore:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            logger.info(
                f"Executing task: {task.name}",
                task_id=task.task_id,
                retries_left=task.retries,
            )
            
            try:
                # Execute function (sync or async)
                kwargs_dict = task.kwargs if task.kwargs is not None else {}
                
                if asyncio.iscoroutinefunction(task.func):
                    # Async function - execute directly
                    result = await task.func(*task.args, **kwargs_dict)
                else:
                    # Sync function - run in thread pool to avoid blocking event loop
                    # Using asyncio.to_thread() (Python 3.9+) for better readability
                    # and modern best practices. It properly handles function arguments
                    # and runs in a separate thread without blocking the event loop.
                    func_with_args = functools.partial(task.func, *task.args, **kwargs_dict)
                    result = await asyncio.to_thread(func_with_args)
                
                # Task completed successfully
                task.status = TaskStatus.COMPLETED
                task.result = result
                task.completed_at = time.time()
                
                duration = task.completed_at - task.started_at
                logger.info(
                    f"Task completed: {task.name}",
                    task_id=task.task_id,
                    duration_sec=round(duration, 2),
                )
                
            except Exception as e:
                logger.error(
                    f"Task failed: {task.name}",
                    task_id=task.task_id,
                    error=str(e),
                    exc_info=True,
                )
                
                # Retry logic
                task.retries -= 1
                if task.retries > 0:
                    logger.info(
                        f"Retrying task: {task.name}",
                        task_id=task.task_id,
                        retries_left=task.retries,
                    )
                    
                    # Exponential backoff
                    delay = 2 ** (task.max_retries - task.retries)
                    await asyncio.sleep(delay)
                    
                    # Resubmit task
                    task.status = TaskStatus.PENDING
                    await self._queue.put(task)
                else:
                    # Max retries exceeded
                    task.status = TaskStatus.FAILED
                    task.error = str(e)
                    task.completed_at = time.time()
            
            finally:
                # Enforce rolling window limits after task completion
                if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                    await self._enforce_task_limits()
                
                if self.enable_persistence:
                    await self._save_state()
    
    async def _save_state(self) -> None:
        """Save task queue state to disk."""
        try:
            state = {
                'tasks': [task.to_dict() for task in self._tasks.values()],
                'saved_at': datetime.now().isoformat(),
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save task queue state: {e}")
    
    async def _load_state(self) -> None:
        """Load task queue state from disk."""
        if not self.state_file.exists():
            logger.info("No task queue state file found, starting fresh")
            return
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)

            # Restore pending tasks only (running tasks are lost on crash).
            # NOTE: The actual callable cannot be restored from disk. These tasks
            # cannot be re-executed without being manually resubmitted by the caller.
            lost_count = 0
            for task_data in state.get('tasks', []):
                if task_data['status'] == TaskStatus.PENDING.value:
                    lost_count += 1

            if lost_count:
                # ERROR-level so operators notice data loss on restart (B-20)
                logger.error(
                    "Pending tasks from previous session cannot be recovered "
                    "(callables are not serialisable). They must be resubmitted.",
                    lost_task_count=lost_count,
                    state_file=str(self.state_file),
                )
            else:
                logger.info(
                    "Loaded task queue state — no pending tasks to recover",
                    state_file=str(self.state_file),
                )

        except Exception as e:
            logger.error(f"Failed to load task queue state: {e}")


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue(
    max_concurrent_tasks: int = 2,
    max_completed_tasks: int = 100,
    max_failed_tasks: int = 50,
) -> TaskQueue:
    """
    Get or create the global task queue instance.
    
    Args:
        max_concurrent_tasks: Maximum concurrent tasks (rate limiting)
        max_completed_tasks: Maximum completed tasks to keep (rolling window)
        max_failed_tasks: Maximum failed tasks to keep (rolling window)
        
    Returns:
        TaskQueue instance
    """
    global _task_queue
    
    if _task_queue is None:
        _task_queue = TaskQueue(
            max_concurrent_tasks=max_concurrent_tasks,
            max_completed_tasks=max_completed_tasks,
            max_failed_tasks=max_failed_tasks,
        )
    
    return _task_queue
