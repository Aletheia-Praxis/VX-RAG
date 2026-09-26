"""Empirical adversarial concurrency and execution integrity stress test suite for Phase M4.

Authored by Challenger 1 (teamwork_preview_challenger) to rigorously verify:
1. Strict serialization under concurrent asyncio.gather() dispatch for query_knowledge_base
   and search_documents via the FastMCP query_lock.
2. Immediate lock release and deadlock freedom under simulated exceptions, task cancellations,
   and intermittent failures.
3. Direct callability and coroutine contracts of FastMCP tools and resources.
4. Text preservation without truncation and cybersecurity IOC retention in search_documents.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import json
import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp.server import (
    _LoopBoundLock,
    health_status,
    mcp,
    query_knowledge_base,
    query_lock,
    search_documents,
    system_context,
)
from src.rag.libs.schemas.mcp_schemas import ContextItem, MCPContextPayload

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


CONCURRENCY_STRESS_BATCH_SIZE: int = 12
INTERLEAVED_BATCH_SIZE: int = 16
TASK_SIMULATION_SLEEP_SECONDS: float = 0.025
TEST_TIMEOUT_SECONDS: float = 10.0
MOCK_TOKEN_BUDGET: int = 4000
EXPECTED_SYSTEM_NAME: str = "VX-RAG"


@pytest.fixture
def _type_checking_contract(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    capsys: CaptureFixture[str],
    request: FixtureRequest,
    mocker: MockerFixture,
) -> None:
    """Fixture ensuring required typing stubs are properly referenced in test signatures."""
    _ = (monkeypatch, caplog, capsys, request, mocker)


@pytest.fixture(autouse=True)
def _reset_module_query_lock_per_test() -> None:
    """Reset module-level query_lock to a fresh _LoopBoundLock instance per test."""
    import src.mcp.server
    src.mcp.server.query_lock = src.mcp.server._LoopBoundLock()


class ExecutionTracker:
    """Thread-safe and task-safe execution tracker for concurrency monitoring."""

    def __init__(self) -> None:
        """Initialize tracker state."""
        self.active_count: int = 0
        self.max_observed_concurrency: int = 0
        self.intervals: list[tuple[str, float, float]] = []
        self._lock: asyncio.Lock = asyncio.Lock()

    async def enter_task(self, task_id: str) -> float:
        """Record task entry and update concurrency statistics.

        Args:
            task_id: Unique identifier for the task.

        Returns:
            Timestamp of task entry.
        """
        async with self._lock:
            self.active_count += 1
            self.max_observed_concurrency = max(self.max_observed_concurrency, self.active_count)
            start_time = time.perf_counter()
            return start_time

    async def exit_task(self, task_id: str, start_time: float) -> None:
        """Record task exit and store execution interval.

        Args:
            task_id: Unique identifier for the task.
            start_time: Timestamp recorded at entry.
        """
        async with self._lock:
            end_time = time.perf_counter()
            self.active_count -= 1
            self.intervals.append((task_id, start_time, end_time))


class TestEmpiricalConcurrencySerialization:
    """Stress tests verifying strict serialization across concurrent MCP operations."""

    @pytest.mark.asyncio
    async def test_concurrent_gather_knowledge_base_strict_serialization(self) -> None:
        """Verify concurrent query_knowledge_base calls never overlap in time."""
        tracker = ExecutionTracker()

        async def simulated_query_async(
            query: str,
            top_k: int = 5,
            search_type: str = "hybrid",
            token_budget: int = 4000,
        ) -> MCPContextPayload:
            start_time = await tracker.enter_task(query)
            try:
                await asyncio.sleep(TASK_SIMULATION_SLEEP_SECONDS)
                return MCPContextPayload(
                    schema_version="1.0",
                    query=query,
                    token_budget=token_budget,
                    context=[
                        ContextItem(
                            id=f"node_{query}",
                            text=f"Sample result for {query}",
                            score=0.9,
                            meta={"file_name": "test.txt"},
                        )
                    ],
                    provenance={"selected_count": 1, "total_tokens": 10},
                )
            finally:
                await tracker.exit_task(query, start_time)

        mock_orchestrator = MagicMock()
        mock_orchestrator.query_async = AsyncMock(side_effect=simulated_query_async)

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            tasks = [
                query_knowledge_base(f"query_{i}")
                for i in range(CONCURRENCY_STRESS_BATCH_SIZE)
            ]
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=TEST_TIMEOUT_SECONDS,
            )

        assert len(responses) == CONCURRENCY_STRESS_BATCH_SIZE
        for resp in responses:
            assert isinstance(resp, str)
            data = json.loads(resp)
            assert "context" in data
            assert "sources" in data

        assert tracker.max_observed_concurrency == 1
        assert len(tracker.intervals) == CONCURRENCY_STRESS_BATCH_SIZE

        # Verify no two intervals overlap
        sorted_intervals = sorted(tracker.intervals, key=lambda item: item[1])
        for idx in range(len(sorted_intervals) - 1):
            _curr_id, _curr_start, curr_end = sorted_intervals[idx]
            _next_id, next_start, _next_end = sorted_intervals[idx + 1]
            assert curr_end <= next_start, (
                f"Concurrency violation: Interval {sorted_intervals[idx]} "
                f"overlaps with {sorted_intervals[idx + 1]}"
            )

    @pytest.mark.asyncio
    async def test_concurrent_gather_search_documents_strict_serialization(self) -> None:
        """Verify concurrent search_documents calls never overlap in time."""
        tracker = ExecutionTracker()

        def simulated_search_documents(
            query: str,
            top_k: int = 10,
            search_type: str = "semantic",
        ) -> list[dict[str, Any]]:
            start_time = time.perf_counter()
            tracker.active_count += 1
            tracker.max_observed_concurrency = max(tracker.max_observed_concurrency, tracker.active_count)

            time.sleep(TASK_SIMULATION_SLEEP_SECONDS)

            end_time = time.perf_counter()
            tracker.active_count -= 1
            tracker.intervals.append((query, start_time, end_time))

            return [
                {
                    "id": f"doc_{query}",
                    "node_id": f"doc_{query}",
                    "text": f"Raw payload for {query}",
                    "score": 0.88,
                    "metadata": {"file_name": f"{query}.md"},
                }
            ]

        mock_orchestrator = MagicMock()
        mock_orchestrator.search_documents = MagicMock(side_effect=simulated_search_documents)

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            tasks = [
                search_documents(f"search_{i}")
                for i in range(CONCURRENCY_STRESS_BATCH_SIZE)
            ]
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=TEST_TIMEOUT_SECONDS,
            )

        assert len(responses) == CONCURRENCY_STRESS_BATCH_SIZE
        assert tracker.max_observed_concurrency == 1
        assert len(tracker.intervals) == CONCURRENCY_STRESS_BATCH_SIZE

        sorted_intervals = sorted(tracker.intervals, key=lambda item: item[1])
        for idx in range(len(sorted_intervals) - 1):
            _curr_id, _curr_start, curr_end = sorted_intervals[idx]
            _next_id, next_start, _next_end = sorted_intervals[idx + 1]
            assert curr_end <= next_start

        for resp in responses:
            data = json.loads(resp)
            assert data["total_count"] == 1
            assert len(data["documents"]) == 1

    @pytest.mark.asyncio
    async def test_concurrent_interleaved_query_and_search_serialization(self) -> None:
        """Verify query_knowledge_base and search_documents share the same query_lock."""
        tracker = ExecutionTracker()

        async def tracked_query_async(query: str, **kwargs: Any) -> MCPContextPayload:
            start_time = await tracker.enter_task(query)
            try:
                await asyncio.sleep(TASK_SIMULATION_SLEEP_SECONDS)
                return MCPContextPayload(
                    schema_version="1.0",
                    query=query,
                    token_budget=MOCK_TOKEN_BUDGET,
                    context=[],
                    provenance={"selected_count": 0, "total_tokens": 0},
                )
            finally:
                await tracker.exit_task(query, start_time)

        def tracked_search_documents(query: str, **kwargs: Any) -> list[dict[str, Any]]:
            start_time = time.perf_counter()
            tracker.active_count += 1
            tracker.max_observed_concurrency = max(tracker.max_observed_concurrency, tracker.active_count)

            time.sleep(TASK_SIMULATION_SLEEP_SECONDS)

            end_time = time.perf_counter()
            tracker.active_count -= 1
            tracker.intervals.append((query, start_time, end_time))
            return [{"id": query, "text": "sample", "score": 0.8, "metadata": {}}]

        mock_orchestrator = MagicMock()
        mock_orchestrator.query_async = AsyncMock(side_effect=tracked_query_async)
        mock_orchestrator.search_documents = MagicMock(side_effect=tracked_search_documents)

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            tasks: list[Any] = []
            for i in range(INTERLEAVED_BATCH_SIZE):
                if i % 2 == 0:
                    tasks.append(query_knowledge_base(f"mixed_query_{i}"))
                else:
                    tasks.append(search_documents(f"mixed_search_{i}"))

            responses = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=TEST_TIMEOUT_SECONDS,
            )

        assert len(responses) == INTERLEAVED_BATCH_SIZE
        assert tracker.max_observed_concurrency == 1
        assert not query_lock.locked()


class TestEmpiricalExceptionHandlingAndDeadlockFreedom:
    """Stress tests verifying immediate lock release upon exceptions."""

    @pytest.mark.asyncio
    async def test_exception_in_query_knowledge_base_releases_lock(self) -> None:
        """Verify query_lock is released when orchestrator.query_async raises an exception."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.query_async = AsyncMock(
            side_effect=RuntimeError("Simulated neural engine out of memory")
        )

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            error_response = await query_knowledge_base("failing_query")
            error_data = json.loads(error_response)
            assert "error" in error_data or "Simulated neural engine" in error_data.get("message", "")
            assert not query_lock.locked()

        # Immediate follow-up query must succeed without deadlock
        mock_orchestrator.query_async = AsyncMock(
            return_value=MCPContextPayload(
                schema_version="1.0",
                query="recovery_query",
                token_budget=MOCK_TOKEN_BUDGET,
                context=[],
                provenance={"selected_count": 0, "total_tokens": 0},
            )
        )
        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            recovery_response = await query_knowledge_base("recovery_query")
            recovery_data = json.loads(recovery_response)
            assert "context" in recovery_data
            assert not query_lock.locked()

    @pytest.mark.asyncio
    async def test_exception_in_search_documents_releases_lock(self) -> None:
        """Verify query_lock is released when orchestrator.search_documents raises an exception."""
        mock_orchestrator = MagicMock()
        mock_orchestrator.search_documents = MagicMock(
            side_effect=ValueError("Corrupted vector store index")
        )

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            error_response = await search_documents("failing_search")
            error_data = json.loads(error_response)
            assert "error" in error_data or "Corrupted vector store" in error_data.get("message", "")
            assert not query_lock.locked()

        # Follow-up search succeeds
        mock_orchestrator.search_documents = MagicMock(return_value=[])
        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            recovery_response = await search_documents("recovery_search")
            recovery_data = json.loads(recovery_response)
            assert recovery_data["total_count"] == 0
            assert not query_lock.locked()

    @pytest.mark.asyncio
    async def test_mixed_concurrent_gather_with_intermittent_failures(self) -> None:
        """Verify concurrent batch with alternating failures executes without deadlock."""
        call_count = 0

        async def fluctuating_query_async(query: str, **kwargs: Any) -> MCPContextPayload:
            nonlocal call_count
            call_count += 1
            if "fail" in query:
                raise RuntimeError(f"Engine failure on {query}")
            return MCPContextPayload(
                schema_version="1.0",
                query=query,
                token_budget=MOCK_TOKEN_BUDGET,
                context=[],
                provenance={"selected_count": 0, "total_tokens": 0},
            )

        mock_orchestrator = MagicMock()
        mock_orchestrator.query_async = AsyncMock(side_effect=fluctuating_query_async)

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            tasks = [
                query_knowledge_base(f"task_{i}_fail" if i % 2 == 0 else f"task_{i}_ok")
                for i in range(CONCURRENCY_STRESS_BATCH_SIZE)
            ]
            responses = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=TEST_TIMEOUT_SECONDS,
            )

        assert len(responses) == CONCURRENCY_STRESS_BATCH_SIZE
        assert not query_lock.locked()

        for idx, resp in enumerate(responses):
            data = json.loads(resp)
            if idx % 2 == 0:
                assert "error" in data or "Engine failure" in str(data)
            else:
                assert "context" in data

    @pytest.mark.asyncio
    async def test_task_cancellation_releases_lock(self) -> None:
        """Verify cancelling a running task frees query_lock for subsequent callers."""
        started_event = asyncio.Event()

        async def slow_query_async(query: str, **kwargs: Any) -> MCPContextPayload:
            started_event.set()
            await asyncio.sleep(10.0)
            return MCPContextPayload(
                schema_version="1.0",
                query=query,
                token_budget=MOCK_TOKEN_BUDGET,
                context=[],
                provenance={"selected_count": 0, "total_tokens": 0},
            )

        mock_orchestrator = MagicMock()
        mock_orchestrator.query_async = AsyncMock(side_effect=slow_query_async)

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            slow_task = asyncio.create_task(query_knowledge_base("slow_task"))
            await started_event.wait()
            import src.mcp.server
            assert src.mcp.server.query_lock.locked()

            # Cancel the running task
            slow_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await slow_task

            # Invariant: Lock must be liberated despite task cancellation
            assert not src.mcp.server.query_lock.locked()

            # Subsequent task acquires lock without deadlock
            mock_orchestrator.query_async = AsyncMock(
                return_value=MCPContextPayload(
                    schema_version="1.0",
                    query="after_cancel",
                    token_budget=MOCK_TOKEN_BUDGET,
                    context=[],
                    provenance={"selected_count": 0, "total_tokens": 0},
                )
            )
            response = await query_knowledge_base("after_cancel")
            assert "context" in json.loads(response)
            assert not src.mcp.server.query_lock.locked()


class TestDirectCallabilityAndInterfaceContracts:
    """Empirical verification of direct callability and FastMCP registration."""

    def test_direct_callability_properties(self) -> None:
        """Verify tools and resources are directly callable coroutines."""
        assert callable(query_knowledge_base)
        assert callable(search_documents)
        assert callable(health_status)
        assert callable(system_context)

        import inspect
        assert inspect.iscoroutinefunction(query_knowledge_base)
        assert inspect.iscoroutinefunction(search_documents)
        assert inspect.iscoroutinefunction(health_status)
        assert inspect.iscoroutinefunction(system_context)

    def test_fastmcp_server_instance_identity(self) -> None:
        """Verify FastMCP server metadata and name."""
        assert mcp.name == EXPECTED_SYSTEM_NAME


class TestSearchDocumentsTextIntegrityAndCybersecurityIOCs:
    """Stress tests verifying search_documents preserves complete text and cybersecurity IOCs."""

    @pytest.mark.asyncio
    async def test_search_documents_massive_text_untruncated(self) -> None:
        """Verify search_documents never truncates large disassembly/source payloads."""
        line_template = "0x00401000: 48 8b 05 29 25 00 00 mov rax, qword ptr [rip + 0x2529] ; payload line {idx}\n"
        massive_text = "".join(line_template.format(idx=i) for i in range(1000))
        text_length = len(massive_text)
        assert text_length > 50000

        mock_metadata: dict[str, str] = {
            "file_name": "malware_disassembly.asm",
            "file_type": "asm",
            "creation_date": "2026-09-21T12:00:00Z",
            "ingestion_date": "2026-09-21T12:30:00Z",
            "file_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        }
        mock_doc: dict[str, Any] = {
            "id": "massive_payload_node",
            "node_id": "massive_payload_node",
            "text": massive_text,
            "score": 0.99,
            "metadata": mock_metadata,
        }

        mock_orchestrator = MagicMock()
        mock_orchestrator.search_documents = MagicMock(return_value=[mock_doc])

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            response_json = await search_documents("find disassembly", top_k=1)
            data = json.loads(response_json)

            assert data["total_count"] == 1
            returned_doc = data["documents"][0]
            assert returned_doc["text"] == massive_text
            assert len(returned_doc["text"]) == text_length
            assert returned_doc["metadata"]["file_name"] == "malware_disassembly.asm"
            assert returned_doc["metadata"]["file_hash"] == mock_metadata["file_hash"]

    @pytest.mark.asyncio
    async def test_search_documents_preserves_all_cybersecurity_iocs(self) -> None:
        """Verify search_documents preserves IP addresses, emails, hashes, and indicators."""
        cyber_forensic_report = (
            "EMOTET C2 TELEMETRY REPORT:\n"
            "Primary IPv4 C2: 198.51.100.22:8080\n"
            "Secondary IPv4 C2: 203.0.113.45:443\n"
            "Fallback IPv6 C2: 2001:0db8:85a3:0000:0000:8a2e:0370:7334\n"
            "Internal staging IP: 10.200.4.15\n"
            "Exfiltration recipient: exfil-operator@apt29-adversary.ru\n"
            "Dropper MD5: 5d41402abc4b2a76b9719d911017c592\n"
            "Payload SHA256: 2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae\n"
            "Persistence Registry: HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\WinUpdate\n"
            "Vulnerability Exploit: CVE-2023-38831 WinRAR execution chain\n"
            "Command line: powershell.exe -NoP -NonI -W Hidden -Enc SUVYKE5ldy1PYmplY3Q=\n"
        )

        mock_doc = {
            "id": "ioc_doc_01",
            "node_id": "ioc_doc_01",
            "text": cyber_forensic_report,
            "score": 0.96,
            "metadata": {
                "file_name": "emotet_indicators.txt",
                "file_type": "txt",
                "creation_date": "2026-09-21T10:00:00Z",
                "ingestion_date": "2026-09-21T10:15:00Z",
                "file_hash": "a" * 64,
            },
        }

        mock_orchestrator = MagicMock()
        mock_orchestrator.search_documents = MagicMock(return_value=[mock_doc])

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            response_json = await search_documents("emotet c2 indicators")
            data = json.loads(response_json)

            retrieved_text = data["documents"][0]["text"]

            # Assert NO redaction tokens exist
            assert "[REDACTED_IP]" not in retrieved_text
            assert "[REDACTED_EMAIL]" not in retrieved_text

            # Assert all indicators survive verbatim
            assert "198.51.100.22" in retrieved_text
            assert "203.0.113.45" in retrieved_text
            assert "2001:0db8:85a3:0000:0000:8a2e:0370:7334" in retrieved_text
            assert "10.200.4.15" in retrieved_text
            assert "exfil-operator@apt29-adversary.ru" in retrieved_text
            assert "5d41402abc4b2a76b9719d911017c592" in retrieved_text
            assert "2c26b46b68ffc68ff99b453c1d30413413422d706483bfa0f98a5e886266e7ae" in retrieved_text
            assert "HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run\\WinUpdate" in retrieved_text
            assert "CVE-2023-38831" in retrieved_text
            assert "powershell.exe -NoP" in retrieved_text

    @pytest.mark.asyncio
    async def test_contrast_query_kb_redacts_while_search_documents_preserves(self) -> None:
        """Verify query_knowledge_base redacts IPs/emails while search_documents preserves them."""
        sensitive_text = (
            "Contact attacker at hacker@darknet.org or investigate host 192.0.2.1."
        )

        mock_orchestrator = MagicMock()
        mock_orchestrator.query_async = AsyncMock(
            return_value=MCPContextPayload(
                schema_version="1.0",
                query="adversary contact",
                token_budget=MOCK_TOKEN_BUDGET,
                context=[
                    ContextItem(
                        id="node_sens",
                        text=sensitive_text,
                        score=0.91,
                        meta={"file_name": "threat.md"},
                    )
                ],
                provenance={"selected_count": 1, "total_tokens": 12},
            )
        )
        mock_orchestrator.search_documents = MagicMock(
            return_value=[
                {
                    "id": "node_sens",
                    "text": sensitive_text,
                    "score": 0.91,
                    "metadata": {"file_name": "threat.md"},
                }
            ]
        )

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            # Query KB with default redaction enabled
            kb_resp = await query_knowledge_base("adversary contact")
            kb_data = json.loads(kb_resp)
            kb_context = kb_data["context"]

            # Search documents with default raw preservation
            search_resp = await search_documents("adversary contact")
            search_data = json.loads(search_resp)
            search_text = search_data["documents"][0]["text"]

            # Verify contrasting behaviors
            assert "[REDACTED_EMAIL]" in kb_context
            assert "[REDACTED_IP]" in kb_context
            assert "hacker@darknet.org" not in kb_context
            assert "192.0.2.1" not in kb_context

            assert "[REDACTED_EMAIL]" not in search_text
            assert "[REDACTED_IP]" not in search_text
            assert "hacker@darknet.org" in search_text
            assert "192.0.2.1" in search_text


class TestEmpiricalLoopBoundLockMultiLoopAndStress:
    """Rigorous stress and multi-loop adversarial tests for _LoopBoundLock."""

    def test_locked_outside_event_loop_returns_false(self) -> None:
        """Verify query_lock.locked() gracefully returns False without raising RuntimeError when no loop is active."""
        lock_proxy = _LoopBoundLock()
        assert not lock_proxy.locked()

    def test_multi_loop_sequential_execution_no_cross_loop_error(self) -> None:
        """Verify query_lock functions correctly across multiple sequential asyncio.run() loops without RuntimeError."""
        executed_loops: list[int] = []

        async def loop_payload(loop_idx: int) -> None:
            async with query_lock:
                assert query_lock.locked()
                await asyncio.sleep(0.005)
                executed_loops.append(loop_idx)

        for iteration in range(10):
            asyncio.run(loop_payload(iteration))

        assert executed_loops == list(range(10))
        assert not query_lock.locked()

    def test_multithreaded_concurrent_event_loops(self) -> None:
        """Verify _LoopBoundLock functions correctly across multiple concurrent threads with separate event loops."""
        worker_count: int = 8
        batch_per_thread: int = 4

        def thread_worker(worker_id: int) -> list[str]:
            async def task(task_id: int) -> str:
                async with query_lock:
                    assert query_lock.locked()
                    await asyncio.sleep(0.005)
                    return f"w{worker_id}_t{task_id}"

            async def main() -> list[str]:
                tasks = [task(i) for i in range(batch_per_thread)]
                return await asyncio.gather(*tasks)

            return asyncio.run(main())

        with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(thread_worker, i) for i in range(worker_count)]
            results = [f.result() for f in futures]

        assert len(results) == worker_count
        for res in results:
            assert len(res) == batch_per_thread

    @pytest.mark.asyncio
    async def test_stress_exception_storm_under_contention(self) -> None:
        """Verify query_lock recovers and prevents deadlocks under an exception storm with 50% failure rate."""
        total_tasks: int = 24
        completed_success: list[int] = []
        handled_failures: list[int] = []

        async def chaotic_task(task_id: int) -> None:
            if task_id % 2 == 0:
                # Failing branch
                try:
                    async with query_lock:
                        assert query_lock.locked()
                        await asyncio.sleep(0.002)
                        raise ValueError(f"Chaos exception in task {task_id}")
                except ValueError:
                    handled_failures.append(task_id)
            else:
                # Successful branch
                async with query_lock:
                    assert query_lock.locked()
                    await asyncio.sleep(0.002)
                    completed_success.append(task_id)

        tasks = [chaotic_task(i) for i in range(total_tasks)]
        await asyncio.gather(*tasks)

        assert len(completed_success) == total_tasks // 2
        assert len(handled_failures) == total_tasks // 2
        assert not query_lock.locked()

        # Invariant: immediate subsequent task acquires lock with zero latency / zero deadlock
        async with query_lock:
            assert query_lock.locked()
        assert not query_lock.locked()

    def test_weakref_garbage_collection_cleans_up_locks(self) -> None:
        """Verify WeakKeyDictionary in _LoopBoundLock garbage collects entries when event loops terminate."""
        lock_proxy = _LoopBoundLock()

        def create_and_run_loop() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                async def noop() -> None:
                    async with lock_proxy:
                        pass
                loop.run_until_complete(noop())
            finally:
                loop.close()
                asyncio.set_event_loop(None)

        # Before run, locks registry is empty
        assert len(lock_proxy._locks) == 0

        create_and_run_loop()

        # Force garbage collection
        gc.collect()

        # The closed and dereferenced loop should be collected from WeakKeyDictionary
        assert len(lock_proxy._locks) == 0

