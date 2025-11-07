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
import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Coroutine
from collections import deque

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
class Task:
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
    func: Callable
    args: tuple = ()
    kwargs: Optional[dict] = None
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[str] = None
    created_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retries: int = 3
    max_retries: int = 3
    
    def __post_init__(self):
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
    
    def __lt__(self, other: 'Task') -> bool:
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
    Asyncio-based task queue with ThreadPoolExecutor for CPU-bound operations.
    
    Features:
    - Priority-based task scheduling
    - Configurable concurrency limits (rate limiting)
    - Background task processing
    - Task persistence for crash recovery
    - Progress tracking and cancellation
    - Retry mechanism with exponential backoff
    
    Example:
        >>> queue = TaskQueue(max_workers=4, max_concurrent_tasks=2)
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
        max_workers: int = 4,
        max_concurrent_tasks: int = 2,
        state_file: Optional[Path] = None,
        enable_persistence: bool = True,
    ):
        """
        Initialize task queue.
        
        Args:
            max_workers: Maximum number of ThreadPoolExecutor workers
            max_concurrent_tasks: Maximum number of tasks running simultaneously (rate limiting)
            state_file: Path to task state persistence file
            enable_persistence: Enable/disable task state persistence
        """
        self.max_workers = max_workers
        self.max_concurrent_tasks = max_concurrent_tasks
        self.enable_persistence = enable_persistence
        
        # Task queue (priority queue)
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        
        # Task registry (all tasks by ID)
        self._tasks: Dict[str, Task] = {}
        
        # Currently running tasks
        self._running_tasks: Dict[str, asyncio.Task] = {}
        
        # Thread pool for CPU-bound operations
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Control flags
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
        
        # Semaphore for rate limiting
        self._concurrency_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # State persistence
        self.state_file = state_file or Path("data/task_queue_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            "TaskQueue initialized",
            max_workers=max_workers,
            max_concurrent_tasks=max_concurrent_tasks,
            persistence=enable_persistence,
        )
    
    async def start(self) -> None:
        """Start the task queue worker."""
        if self._running:
            logger.warning("TaskQueue already running")
            return
        
        self._running = True
        
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
        
        # Shutdown executor
        self._executor.shutdown(wait=True, cancel_futures=True)
        
        # Save state
        if self.enable_persistence:
            await self._save_state()
        
        logger.info("TaskQueue stopped")
    
    async def submit_task(
        self,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[dict] = None,
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
        
        task = Task(
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
    
    async def get_queue_stats(self) -> Dict[str, Any]:
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
            'max_workers': self.max_workers,
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
    
    async def _execute_task(self, task: Task) -> None:
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
                    result = await task.func(*task.args, **kwargs_dict)
                else:
                    # Run in thread pool for CPU-bound operations
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self._executor,
                        task.func,
                        *task.args,
                        **kwargs_dict,
                    )
                
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
            
            # Restore pending tasks only (running tasks are lost on crash)
            loaded_count = 0
            for task_data in state.get('tasks', []):
                if task_data['status'] == TaskStatus.PENDING.value:
                    # Note: We cannot restore the actual function, so these tasks
                    # will remain in PENDING state until manually resubmitted
                    # This is a limitation of not using Celery-style serialization
                    logger.warning(
                        f"Found pending task from previous session: {task_data['name']}",
                        task_id=task_data['task_id'],
                    )
                    loaded_count += 1
            
            logger.info(f"Loaded {loaded_count} pending tasks from state file")
            
        except Exception as e:
            logger.error(f"Failed to load task queue state: {e}")


# Global task queue instance
_task_queue: Optional[TaskQueue] = None


def get_task_queue(
    max_workers: int = 4,
    max_concurrent_tasks: int = 2,
) -> TaskQueue:
    """
    Get or create the global task queue instance.
    
    Args:
        max_workers: Maximum ThreadPoolExecutor workers
        max_concurrent_tasks: Maximum concurrent tasks (rate limiting)
        
    Returns:
        TaskQueue instance
    """
    global _task_queue
    
    if _task_queue is None:
        _task_queue = TaskQueue(
            max_workers=max_workers,
            max_concurrent_tasks=max_concurrent_tasks,
        )
    
    return _task_queue
