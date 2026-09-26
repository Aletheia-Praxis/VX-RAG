"""
Adversarial stress harness for Requirements R2 and R3.

Stress tests contextvars isolation, async task concurrency, thread pool propagation,
clean context cleanup, and high-throughput metrics emission with zero in-memory accumulation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextvars
import json
import random
import threading
import time
import tracemalloc
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from src.utils.logging_config import (
    StructuredLogger,
    _global_logging_overrides,
    clear_request_id,
    get_logger,
    get_request_id,
    request_context,
    reset_logging_handlers,
    setup_logging,
)
from src.utils.metrics import (
    MetricsCollector,
    gauge,
    get_metrics,
    histogram,
    increment,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


def _type_check_fixtures(
    _c: CaptureFixture[str] | None = None,
    _r: FixtureRequest | None = None,
    _l: LogCaptureFixture | None = None,
    _m: MonkeyPatch | None = None,
    _mock: MockerFixture | None = None,
) -> None:
    """Type checking helper to validate fixture typing."""


@pytest.fixture(autouse=True)
def clean_logging_environment() -> Any:
    """Reset handlers and metrics before and after each test."""
    from src.utils import metrics

    reset_logging_handlers()
    _global_logging_overrides["log_metrics"] = True
    _global_logging_overrides["metrics_enabled"] = True
    metrics._metrics_instance = None
    yield
    reset_logging_handlers()
    metrics._metrics_instance = None


@pytest.mark.asyncio
async def test_stress_asyncio_concurrency_and_isolation(tmp_path: Path) -> None:
    """
    Stress-test 50 concurrent asyncio tasks with interleaved sleeps.

    Verifies:
    1. Each task maintains strict contextvars isolation during interleaving.
    2. Zero cross-talk occurs between concurrent coroutines.
    3. Exiting request_context cleanly resets get_request_id() to None.
    4. Outside log records do not contain request_id.
    """
    log_file = tmp_path / "async_stress.jsonl"
    setup_logging(log_file=str(log_file), log_level="INFO", log_format="json")
    logger: StructuredLogger = get_logger("async_stress_logger")

    num_tasks = 50
    iterations_per_task = 10
    task_requests: dict[int, str] = {
        i: f"stress-req-{i:03d}-{uuid.uuid4()}" for i in range(num_tasks)
    }

    async def worker(task_idx: int) -> None:
        expected_rid = task_requests[task_idx]
        with request_context(expected_rid) as active_rid:
            assert active_rid == expected_rid
            assert get_request_id() == expected_rid

            for step in range(iterations_per_task):
                logger.info(
                    f"Task {task_idx} at step {step}",
                    task_idx=task_idx,
                    step=step,
                )
                # Interleaved sleep to trigger coroutine suspension and context switches
                await asyncio.sleep(random.uniform(0.001, 0.008))
                # Verify request_id immediately after resuming execution
                assert get_request_id() == expected_rid

        # Clean cleanup check: must be None after context exit
        assert get_request_id() is None

    # Run 50 concurrent tasks
    tasks = [asyncio.create_task(worker(i)) for i in range(num_tasks)]
    await asyncio.gather(*tasks)

    # Outside context verification
    assert get_request_id() is None
    outside_message = "Outside of any request context"
    logger.info(outside_message, outside_flag=True)

    # Explicit clear check
    clear_request_id()
    assert get_request_id() is None

    # Flush file handlers
    for handler in logger.logger.handlers:
        handler.flush()

    # Read and audit generated log entries
    assert log_file.exists()
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= (num_tasks * iterations_per_task) + 1

    task_records: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_tasks)}
    outside_records: list[dict[str, Any]] = []

    for line in lines:
        record = json.loads(line)
        assert isinstance(record, dict)

        if record.get("message") == outside_message:
            outside_records.append(record)
        elif "task_idx" in record:
            idx = int(record["task_idx"])
            task_records[idx].append(record)

    # Verify outside log records contain NO request_id (zero leakage)
    assert len(outside_records) == 1
    assert "request_id" not in outside_records[0]

    # Verify isolation and zero cross-talk across all 50 tasks
    for idx, expected_rid in task_requests.items():
        records = task_records[idx]
        assert len(records) == iterations_per_task, f"Task {idx} missed log records"
        for rec in records:
            assert rec.get("request_id") == expected_rid, (
                f"Cross-talk detected! Task {idx} expected {expected_rid} but got {rec.get('request_id')}"
            )


@pytest.mark.asyncio
async def test_stress_threadpool_propagation_and_isolation(tmp_path: Path) -> None:
    """
    Stress-test context propagation to worker threads via asyncio.to_thread.

    Verifies:
    1. Tasks running in thread pools inherit the active request_id.
    2. Concurrent threads do not cross-contaminate request_id.
    3. Thread context cleanly returns None once execution finishes.
    """
    log_file = tmp_path / "thread_stress.jsonl"
    setup_logging(log_file=str(log_file), log_level="INFO", log_format="json")
    logger: StructuredLogger = get_logger("thread_stress_logger")

    num_threads = 50
    thread_requests: dict[int, str] = {
        i: f"thread-req-{i:03d}-{uuid.uuid4()}" for i in range(num_threads)
    }

    def synchronous_worker(worker_id: int, expected_id: str) -> dict[str, Any]:
        observed_id_start = get_request_id()
        current_thread_name = threading.current_thread().name

        logger.info(
            f"Thread worker {worker_id} started on {current_thread_name}",
            worker_id=worker_id,
            thread_name=current_thread_name,
        )
        time.sleep(random.uniform(0.002, 0.008))
        observed_id_end = get_request_id()

        logger.info(
            f"Thread worker {worker_id} completed on {current_thread_name}",
            worker_id=worker_id,
            thread_name=current_thread_name,
        )

        return {
            "worker_id": worker_id,
            "expected_id": expected_id,
            "observed_start": observed_id_start,
            "observed_end": observed_id_end,
            "thread_name": current_thread_name,
        }

    async def async_caller(task_idx: int) -> dict[str, Any]:
        expected_rid = thread_requests[task_idx]
        with request_context(expected_rid):
            result = await asyncio.to_thread(synchronous_worker, task_idx, expected_rid)
        assert get_request_id() is None
        return result

    tasks = [asyncio.create_task(async_caller(i)) for i in range(num_threads)]
    results = await asyncio.gather(*tasks)

    assert len(results) == num_threads
    for res in results:
        assert res["observed_start"] == res["expected_id"]
        assert res["observed_end"] == res["expected_id"]

    for handler in logger.logger.handlers:
        handler.flush()

    # Audit JSON log records from thread pool
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= num_threads * 2

    for line in lines:
        record = json.loads(line)
        if "worker_id" in record:
            worker_id = int(record["worker_id"])
            expected = thread_requests[worker_id]
            assert record.get("request_id") == expected, (
                f"Thread cross-talk! Worker {worker_id} expected {expected} but found {record.get('request_id')}"
            )


def test_stress_concurrent_metrics_emission_under_heavy_load(tmp_path: Path) -> None:
    """
    Stress-test concurrent emission of 10,000 metrics across multiple threads.

    Verifies:
    1. Zero in-memory accumulation: MetricsCollector stores no metrics state.
    2. Zero background daemon threads spawned.
    3. Strict JSON Lines validity with correct event_type and values.
    """
    log_file = tmp_path / "metrics_stress.jsonl"
    setup_logging(log_file=str(log_file), log_level="INFO", log_format="json")

    # Capture baseline daemon thread count
    threads_before = [t for t in threading.enumerate() if t.is_alive()]
    daemon_threads_before = [t for t in threads_before if t.daemon]

    collector = MetricsCollector(
        config={
            "metrics_enabled": True,
            "log_metrics": True,
            "log_file": str(log_file),
            "log_format": "json",
        }
    )

    total_emissions = 10000
    num_worker_threads = 20
    emissions_per_thread = total_emissions // num_worker_threads

    # Start memory tracing to verify bounded memory footprint
    tracemalloc.start()
    mem_before, _ = tracemalloc.get_traced_memory()

    def metric_worker(worker_id: int) -> None:
        for idx in range(emissions_per_thread):
            op = idx % 3
            if op == 0:
                collector.increment(
                    metric_name="stress_counter",
                    value=1,
                    tags={"worker": worker_id, "idx": idx},
                )
            elif op == 1:
                collector.gauge(
                    metric_name="stress_gauge",
                    value=float(idx * 1.5),
                    tags={"worker": worker_id, "idx": idx},
                )
            else:
                collector.histogram(
                    metric_name="stress_histogram",
                    value=float(idx * 0.25),
                    tags={"worker": worker_id, "idx": idx},
                )

    threads: list[threading.Thread] = []
    for wid in range(num_worker_threads):
        t = threading.Thread(target=metric_worker, args=(wid,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    mem_after, _peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # 1. Verify zero in-memory accumulation
    stats = collector.get_stats()
    assert stats == {}, "MetricsCollector accumulated in-memory state in get_stats()"
    assert not hasattr(collector, "metrics"), "MetricsCollector has legacy self.metrics attribute"

    # Memory growth check: should be small temporary allocations, bounded peak
    net_mem_growth_kb = (mem_after - mem_before) / 1024.0
    # Even under 10,000 emissions, permanent memory growth in collector must be negligible (< 1 MB)
    assert net_mem_growth_kb < 1024.0, f"Unbounded memory growth observed: {net_mem_growth_kb:.2f} KB"

    # 2. Verify zero daemon threads created
    threads_after = [t for t in threading.enumerate() if t.is_alive()]
    daemon_threads_after = [t for t in threads_after if t.daemon]
    new_daemons = [t.name for t in daemon_threads_after if t not in daemon_threads_before]
    assert len(new_daemons) == 0, f"Metrics spawned unexpected background daemon threads: {new_daemons}"

    # Flush file handlers
    for handler in collector.logger.logger.handlers:
        handler.flush()

    # 3. Verify emitted JSONL records
    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == total_emissions, f"Expected {total_emissions} metric lines, got {len(lines)}"

    counter_count = 0
    gauge_count = 0
    histogram_count = 0

    for line in lines:
        record = json.loads(line)
        assert record.get("event_type") == "metric"
        mtype = record.get("metric_type")
        assert mtype in {"counter", "gauge", "histogram"}
        assert "value" in record
        assert "metric_name" in record

        if mtype == "counter":
            counter_count += 1
            assert record["value"] == 1
        elif mtype == "gauge":
            gauge_count += 1
            assert isinstance(record["value"], (int, float))
        elif mtype == "histogram":
            histogram_count += 1
            assert isinstance(record["value"], (int, float))

    assert counter_count > 0
    assert gauge_count > 0
    assert histogram_count > 0
    assert counter_count + gauge_count + histogram_count == total_emissions


def test_stress_module_level_metrics_emission_under_concurrent_load(tmp_path: Path) -> None:
    """
    Stress-test module-level increment, gauge, and histogram under concurrent execution.

    Verifies that get_metrics() singleton and module-level functions operate cleanly
    across multiple threads with valid JSONL output.
    """
    log_file = tmp_path / "module_metrics_stress.jsonl"
    setup_logging(log_file=str(log_file), log_level="INFO", log_format="json")
    _global_logging_overrides["log_metrics"] = True
    _global_logging_overrides["metrics_enabled"] = True
    from src.utils import metrics
    metrics._metrics_instance = None

    total_calls = 3000
    num_threads = 10
    calls_per_thread = total_calls // num_threads

    def worker(tid: int) -> None:
        for j in range(calls_per_thread):
            if j % 3 == 0:
                increment("mod_counter", value=2, tags={"tid": tid})
            elif j % 3 == 1:
                gauge("mod_gauge", value=42.5, tags={"tid": tid})
            else:
                histogram("mod_histogram", value=100.0, tags={"tid": tid})

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()

    singleton = get_metrics()
    assert singleton.get_stats() == {}

    for handler in singleton.logger.logger.handlers:
        handler.flush()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == total_calls
    for line in lines:
        data = json.loads(line)
        assert data.get("event_type") == "metric"


def test_stress_contextvars_deep_nesting_and_exception_unwinding() -> None:
    """
    Stress-test nested request_context scopes and exception unwinding across 50 concurrent tasks.

    Verifies:
    1. Entering inner request_context updates get_request_id().
    2. Exceptions inside inner contexts properly unwind to the enclosing request_id.
    3. Exiting the outermost context cleanly leaves get_request_id() as None.
    """
    num_tasks = 50

    def nested_task_runner(idx: int) -> None:
        outer_id = f"outer-{idx}-{uuid.uuid4()}"
        inner_id = f"inner-{idx}-{uuid.uuid4()}"

        assert get_request_id() is None
        with request_context(outer_id):
            assert get_request_id() == outer_id

            try:
                with request_context(inner_id):
                    assert get_request_id() == inner_id
                    raise RuntimeError("Simulated failure inside inner context")
            except RuntimeError:
                pass

            # Context must have safely restored to outer_id
            assert get_request_id() == outer_id

        # Outer context exited, must be None
        assert get_request_id() is None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(nested_task_runner, i) for i in range(num_tasks)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # Will re-raise any assertion errors


def test_stress_threadpoolexecutor_explicit_context_propagation(tmp_path: Path) -> None:
    """
    Stress-test concurrent thread execution with contextvars.copy_context().

    Verifies that raw ThreadPoolExecutor preserves request_id when context is propagated.
    """
    log_file = tmp_path / "tpe_stress.jsonl"
    setup_logging(log_file=str(log_file), log_level="INFO", log_format="json")
    logger = get_logger("tpe_logger")

    num_threads = 50
    requests: dict[int, str] = {i: f"tpe-req-{i:03d}-{uuid.uuid4()}" for i in range(num_threads)}

    def thread_fn(worker_idx: int, req_id: str) -> None:
        assert get_request_id() == req_id
        logger.info(f"TPE worker {worker_idx} executing", worker_idx=worker_idx)
        time.sleep(random.uniform(0.001, 0.005))
        assert get_request_id() == req_id

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures: list[concurrent.futures.Future[None]] = []
        for i in range(num_threads):
            rid = requests[i]
            # Set request_id inside this specific context
            with request_context(rid):
                ctx = contextvars.copy_context()
            fut = executor.submit(ctx.run, thread_fn, i, rid)
            futures.append(fut)

        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    for handler in logger.logger.handlers:
        handler.flush()

    lines = log_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == num_threads

    for line in lines:
        record = json.loads(line)
        w_idx = int(record["worker_idx"])
        assert record.get("request_id") == requests[w_idx]


def test_stress_concurrent_logging_file_integrity_and_rollover(tmp_path: Path) -> None:
    """
    Stress-test concurrent log writing during file rotation under multi-thread load.

    Verifies:
    1. Zero WinError 32 or file lock contention during concurrent rollover.
    2. Every line across current and backup files is strictly valid JSON Lines.
    3. Zero malformed or interleaved records.
    """
    log_file = tmp_path / "rollover_stress.jsonl"
    max_bytes = 12 * 1024  # 12 KB to force frequent rotations
    backup_count = 5
    setup_logging(
        log_file=str(log_file),
        max_bytes=max_bytes,
        backup_count=backup_count,
        log_level="INFO",
        log_format="json",
    )
    logger = get_logger("rollover_stress_logger")

    num_threads = 20
    messages_per_thread = 50

    def worker(worker_id: int) -> None:
        for m_idx in range(messages_per_thread):
            req_id = f"roll-req-{worker_id:02d}-{m_idx:03d}"
            with request_context(req_id):
                payload_padding = "x" * 200  # Expand line size to ~350 bytes
                logger.info(
                    f"Worker {worker_id} message {m_idx}: {payload_padding}",
                    worker_id=worker_id,
                    m_idx=m_idx,
                )
                time.sleep(random.uniform(0.0005, 0.002))

    threads: list[threading.Thread] = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    for handler in logger.logger.handlers:
        handler.flush()

    # Collect all lines from primary and backup rotated files
    all_log_files = sorted(tmp_path.glob("rollover_stress.jsonl*"))
    assert len(all_log_files) > 1, f"Expected rollover files to be created, found: {all_log_files}"

    all_records: list[dict[str, Any]] = []
    for f in all_log_files:
        content = f.read_text(encoding="utf-8").strip()
        if not content:
            continue
        for line in content.splitlines():
            # Strict JSON parse check: will raise json.JSONDecodeError if any line is corrupted
            rec = json.loads(line)
            assert isinstance(rec, dict)
            assert "timestamp" in rec
            assert "level" in rec
            assert "request_id" in rec
            all_records.append(rec)

    # Verify total records accounted for
    # In rotating handlers with backup_count=5, if total data exceeds max_bytes * (backup_count + 1),
    # oldest files are pruned. Let's verify each preserved record is valid and uncorrupted.
    assert len(all_records) > 0
    # Every preserved record must have matching worker_id and m_idx in request_id
    for rec in all_records:
        if "worker_id" in rec and "m_idx" in rec:
            expected_prefix = f"roll-req-{rec['worker_id']:02d}-{rec['m_idx']:03d}"
            assert rec["request_id"] == expected_prefix

