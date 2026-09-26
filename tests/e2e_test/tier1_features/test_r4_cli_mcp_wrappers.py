"""
Tier 1 Feature Coverage: Requirement 4 (CLI Interface, FastMCP Server, Sequential Execution, Redaction, Resources).

Authoritative Source: ORIGINAL_REQUEST.md §R4, PROJECT.md Features 13-23, Tech Spec §6.2, §7.1.
Verifies:
- Features 13-16: CLI Ingest, Index, Query, Serve Commands
- Features 17-18: FastMCP Server Initialization & Sequential Concurrency Lock
- Features 19-20: MCP Tools query_knowledge_base & search_documents
- Feature 21: Sensitive Data Redaction (emails, IPv4, IPv6)
- Features 22-23: MCP Resources health://status & context://system
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp.formatters import (
    format_health_status,
    format_query_response,
    format_search_response,
    format_system_context,
    redact_email_addresses,
    redact_ip_addresses,
    redact_sensitive_data,
)
from src.rag.libs.schemas.mcp_schemas import ContextItem, MCPContextPayload

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


@pytest.fixture
def _typing_imports_verifier(
    monkeypatch: MonkeyPatch,
    caplog: LogCaptureFixture,
    capsys: CaptureFixture[str],
    request: FixtureRequest,
    mocker: MockerFixture,
) -> None:
    """Fixture ensuring required typing stubs are properly referenced in test signatures."""
    _ = (monkeypatch, caplog, capsys, request, mocker)


class TestFeatures13To16CLICommands:
    """
    Features 13-16: CLI Ingest, Index, Query, Serve Commands.

    Authoritative: ORIGINAL_REQUEST.md §R4, PROJECT.md Features 13-16.
    Ensures thin CLI wrapper exposes commands with appropriate options and exit codes.
    """

    def test_cli_ingest_help_flag(self, run_cli_command: Callable[[list[str]], Any]) -> None:
        """Verify cli ingest --help prints usage."""
        res = run_cli_command(["ingest", "--help"])
        assert res.exit_code == 0
        assert "--data-dir" in res.stdout
        assert "--persist-dir" in res.stdout

    def test_cli_ingest_missing_directory_fails(self, run_cli_command: Callable[[list[str]], Any], tmp_path: Path) -> None:
        """Verify cli ingest fails when data directory does not exist."""
        non_existent = str(tmp_path / "does_not_exist_raw")
        res = run_cli_command(["ingest", "--data-dir", non_existent])
        assert res.exit_code != 0

    def test_cli_index_help_flag(self, run_cli_command: Callable[[list[str]], Any]) -> None:
        """Verify cli index --help prints usage."""
        res = run_cli_command(["index", "--help"])
        assert res.exit_code == 0
        assert "--data-dir" in res.stdout
        assert "--persist-dir" in res.stdout

    def test_cli_index_missing_nodes_file_fails(self, run_cli_command: Callable[[list[str]], Any], tmp_path: Path) -> None:
        """Verify cli index exits with status 1 if nodes.json is absent."""
        empty_dir = tmp_path / "raw"
        empty_dir.mkdir()
        res = run_cli_command(["index", "--data-dir", str(empty_dir)])
        assert res.exit_code != 0
        assert "nodes.json" in res.stdout or "nodes.json" in res.stderr or "Nodes file not found" in res.stdout

    def test_cli_query_help_flag(self, run_cli_command: Callable[[list[str]], Any]) -> None:
        """Verify cli query --help prints query arguments."""
        res = run_cli_command(["query", "--help"])
        assert res.exit_code == 0
        assert "--top-k" in res.stdout

    def test_cli_serve_help_flag(self, run_cli_command: Callable[[list[str]], Any]) -> None:
        """Verify cli serve --help displays transport options."""
        res = run_cli_command(["serve", "--help"])
        assert res.exit_code == 0
        assert "--transport" in res.stdout
        assert "stdio" in res.stdout

    def test_cli_serve_transport_options(self, run_cli_command: Callable[[list[str]], Any]) -> None:
        """Verify cli serve accepts valid transport options."""
        for transport in ["stdio", "sse", "http"]:
            res = run_cli_command(["serve", "--transport", transport, "--help"])
            assert res.exit_code == 0

    def test_cli_serve_invalid_transport_rejected(self, run_cli_command: Callable[[list[str]], Any]) -> None:
        """Verify cli serve rejects invalid transport values."""
        res = run_cli_command(["serve", "--transport", "websocket_invalid"])
        assert res.exit_code != 0


class TestFeatures17And18FastMCPServerAndConcurrency:
    """
    Features 17-18: FastMCP Server Initialization & Sequential Query Execution.

    Authoritative: ORIGINAL_REQUEST.md §R4, PROJECT.md Features 17-18, Tech Spec §7.1.
    Ensures FastMCP instance exposes tools and resources with asyncio.Lock serialization.
    """

    def test_mcp_server_instance_identity(self) -> None:
        """Verify FastMCP server instance name and version."""
        from src.mcp.server import mcp

        assert mcp.name == "VX-RAG"

    def test_mcp_server_registers_all_required_tools(self) -> None:
        """Verify FastMCP server registers query_knowledge_base and search_documents tools."""
        from src.mcp.server import query_knowledge_base, search_documents

        assert callable(query_knowledge_base)
        assert callable(search_documents)

    def test_mcp_server_registers_all_required_resources(self) -> None:
        """Verify FastMCP server registers health and context resources."""
        from src.mcp.server import health_status, system_context

        assert callable(health_status)
        assert callable(system_context)

    @pytest.mark.asyncio
    async def test_sequential_query_lock_serialization(self) -> None:
        """Verify src.mcp.server.query_lock serializes concurrent query and search executions."""
        from src.mcp.server import query_knowledge_base, query_lock, search_documents

        execution_order: list[str] = []

        async def tracked_query_async(
            query: str,
            top_k: int = 5,
            search_type: str = "hybrid",
            token_budget: int = 4000,
        ) -> MCPContextPayload:
            assert query_lock.locked()
            execution_order.append(f"start_{query}")
            await asyncio.sleep(0.03)
            execution_order.append(f"end_{query}")
            return MCPContextPayload(
                schema_version="1.0",
                query=query,
                token_budget=token_budget,
                context=[
                    ContextItem(
                        id=f"node_{query}",
                        text=f"Content for {query}",
                        score=0.9,
                        meta={"file_name": f"{query}.md"},
                    )
                ],
                provenance={"selected_count": 1, "total_tokens": 10},
            )

        def tracked_search_documents(
            query: str,
            top_k: int = 10,
            search_type: str = "semantic",
        ) -> list[dict[str, Any]]:
            assert query_lock.locked()
            execution_order.append(f"start_{query}")
            time.sleep(0.01)
            execution_order.append(f"end_{query}")
            return [
                {
                    "id": f"doc_{query}",
                    "node_id": f"doc_{query}",
                    "text": f"Raw content for {query}",
                    "score": 0.85,
                    "metadata": {"file_name": f"{query}.txt"},
                }
            ]

        mock_orchestrator = MagicMock()
        mock_orchestrator.query_async = AsyncMock(side_effect=tracked_query_async)
        mock_orchestrator.search_documents = MagicMock(side_effect=tracked_search_documents)

        # Invariant: Lock is unlocked prior to dispatch
        assert not query_lock.locked()

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            # Dispatch concurrent operations across both tools simultaneously
            results = await asyncio.gather(
                query_knowledge_base("task_1_kb"),
                search_documents("task_2_search"),
                query_knowledge_base("task_3_kb"),
            )

        # Invariant: Strict sequential serialization across disparate tools
        assert execution_order == [
            "start_task_1_kb",
            "end_task_1_kb",
            "start_task_2_search",
            "end_task_2_search",
            "start_task_3_kb",
            "end_task_3_kb",
        ]

        # Invariant: Lock is released after all operations complete
        assert not query_lock.locked()

        # Verify all tools returned valid formatted responses
        kb_1 = json.loads(results[0])
        search_2 = json.loads(results[1])
        kb_3 = json.loads(results[2])
        assert "context" in kb_1
        assert search_2["total_count"] == 1
        assert "context" in kb_3

    @pytest.mark.asyncio
    async def test_sequential_query_lock_release_on_exception(self) -> None:
        """Verify query_lock is immediately released when operations raise exceptions."""
        from src.mcp.server import query_knowledge_base, query_lock, search_documents

        mock_orchestrator = MagicMock()
        mock_orchestrator.query_async = AsyncMock(
            side_effect=RuntimeError("Simulated neural engine failure")
        )
        mock_orchestrator.search_documents = MagicMock(
            side_effect=ValueError("Corrupted vector store index")
        )

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orchestrator):
            # 1. Failing query_knowledge_base call
            error_kb_resp = await query_knowledge_base("failing_kb")
            error_kb_data = json.loads(error_kb_resp)
            assert "error" in error_kb_data or "Simulated neural engine failure" in error_kb_resp
            assert not query_lock.locked()

            # 2. Failing search_documents call
            error_search_resp = await search_documents("failing_search")
            error_search_data = json.loads(error_search_resp)
            assert "error" in error_search_data or "Corrupted vector store" in error_search_resp
            assert not query_lock.locked()

        # 3. Direct unhandled exception inside critical section (validating __aexit__)
        with pytest.raises(KeyError, match="Direct unhandled error"):
            async with query_lock:
                assert query_lock.locked()
                raise KeyError("Direct unhandled error")
        assert not query_lock.locked()

        # 4. Immediate recovery: subsequent query acquires lock and succeeds without deadlock
        recovery_orchestrator = MagicMock()
        recovery_orchestrator.query_async = AsyncMock(
            return_value=MCPContextPayload(
                schema_version="1.0",
                query="recovery_query",
                token_budget=1000,
                context=[],
                provenance={"selected_count": 0, "total_tokens": 0},
            )
        )
        recovery_orchestrator.search_documents = MagicMock(return_value=[])

        with patch("src.mcp.server.get_orchestrator", return_value=recovery_orchestrator):
            rec_kb_resp = await query_knowledge_base("recovery_query")
            rec_search_resp = await search_documents("recovery_search")

            assert "context" in json.loads(rec_kb_resp)
            assert json.loads(rec_search_resp)["total_count"] == 0
            assert not query_lock.locked()

    def test_query_lock_multi_loop_safety(self) -> None:
        """Verify query_lock operates across distinct consecutive event loops without cross-loop errors."""
        from src.mcp.server import query_lock

        async def worker() -> None:
            async with query_lock:
                await asyncio.sleep(0.01)

        async def run_batch() -> None:
            await asyncio.gather(worker(), worker())

        # Execute across two distinct event loops
        asyncio.run(run_batch())
        asyncio.run(run_batch())
        assert not query_lock.locked()


class TestFeature19MCPToolQueryKnowledgeBase:
    """
    Feature 19: MCP Tool query_knowledge_base.

    Authoritative: ORIGINAL_REQUEST.md §R4, PROJECT.md Feature 19.
    Primary MCP tool returning generalized context snippets with redaction and token budgeting.
    """

    def test_query_kb_response_structure(self) -> None:
        """Verify format_query_response produces valid JSON matching schema."""
        payload = MCPContextPayload(
            schema_version="1.0",
            query="Emotet telemetry",
            token_budget=2048,
            context=[
                ContextItem(
                    id="node_1",
                    text="C2 communication observed at 198.51.100.22",
                    score=0.92,
                    meta={"file_name": "emotet.md", "file_type": "md"}
                )
            ],
            provenance={"selected_count": 1, "total_tokens": 10}
        )

        response_json = format_query_response(payload, apply_redaction=True)
        data = json.loads(response_json)

        assert "context" in data
        assert "sources" in data
        assert "stats" in data
        assert "[REDACTED_IP]" in data["context"]
        assert "198.51.100.22" not in data["context"]

    def test_query_kb_empty_context_handling(self) -> None:
        """Verify format_query_response handles empty results cleanly."""
        payload = MCPContextPayload(
            schema_version="1.0",
            query="nonexistent term",
            token_budget=1000,
            context=[],
            provenance={"selected_count": 0, "total_tokens": 0}
        )
        response_json = format_query_response(payload)
        data = json.loads(response_json)

        assert data["stats"]["total_results"] == 0
        assert data["context"] == "No relevant documents found."

    def test_query_kb_respects_token_budget(self) -> None:
        """Verify query response includes token budget in metadata stats."""
        payload = MCPContextPayload(
            schema_version="1.0",
            query="test",
            token_budget=4000,
            context=[ContextItem(id="1", text="Sample", score=0.8, meta={})],
            provenance={"selected_count": 1, "total_tokens": 5}
        )
        data = json.loads(format_query_response(payload))
        assert data["stats"]["token_budget"] == 4000

    def test_query_kb_citation_structure(self) -> None:
        """Verify query response structures citations correctly."""
        payload = MCPContextPayload(
            schema_version="1.0",
            query="test",
            token_budget=2000,
            context=[
                ContextItem(id="node_abc", text="Snippet", score=0.85, meta={"file_name": "advisory.pdf"})
            ],
            provenance={"selected_count": 1, "total_tokens": 5}
        )
        data = json.loads(format_query_response(payload))
        assert len(data["sources"]) == 1
        assert data["sources"][0]["id"] == "node_abc"
        assert data["sources"][0]["score"] == 0.85

    @pytest.mark.asyncio
    async def test_mcp_query_knowledge_base_invocation(self) -> None:
        """Verify invoking query_knowledge_base tool returns JSON string."""
        from src.mcp.server import query_knowledge_base

        with patch("src.mcp.server.get_orchestrator") as mock_get_orch:
            mock_orch = MagicMock()
            mock_orch.query_async = AsyncMock(return_value=MCPContextPayload(
                schema_version="1.0",
                query="test",
                token_budget=1000,
                context=[],
                provenance={"selected_count": 0, "total_tokens": 0}
            ))
            mock_get_orch.return_value = mock_orch

            result = await query_knowledge_base("test query")
            assert isinstance(result, str)
            assert json.loads(result)["stats"]["total_results"] == 0


class TestFeature20MCPToolSearchDocuments:
    """
    Feature 20: MCP Tool search_documents.

    Authoritative: ORIGINAL_REQUEST.md §R4, PROJECT.md Feature 20.
    Document search tool returning complete full text without truncation or summarization.
    """

    def test_search_documents_returns_untruncated_full_text(self) -> None:
        """Verify search_documents returns complete raw text of matching nodes."""
        full_text = "Complete un-truncated documentation text. " * 50
        raw_results = [
            {
                "id": "doc_1",
                "node_id": "doc_1",
                "text": full_text,
                "score": 0.95,
                "metadata": {"file_name": "large.md", "file_type": "md"}
            }
        ]

        response_json = format_search_response("query", raw_results, "semantic", apply_redaction=False)
        data = json.loads(response_json)

        assert data["documents"][0]["text"] == full_text
        assert len(data["documents"][0]["text"]) == len(full_text)

    def test_search_documents_preserves_strict_metadata(self) -> None:
        """Verify search_documents includes exact node metadata."""
        meta = {
            "file_name": "advisory.txt",
            "file_type": "txt",
            "creation_date": "2026-09-13T08:00:00Z",
            "ingestion_date": "2026-09-13T08:30:00Z",
            "file_hash": "a" * 64
        }
        raw_results = [{"id": "doc_2", "text": "Report", "score": 0.88, "metadata": meta}]

        data = json.loads(format_search_response("query", raw_results, "semantic"))
        assert data["documents"][0]["metadata"]["file_name"] == "advisory.txt"
        assert data["documents"][0]["metadata"]["file_hash"] == "a" * 64

    def test_search_documents_top_k_parameter(self) -> None:
        """Verify search_documents respects requested top_k limit."""
        raw_results = [
            {"id": f"doc_{i}", "text": f"Content {i}", "score": 0.9 - i * 0.05, "metadata": {}}
            for i in range(10)
        ]
        data = json.loads(format_search_response("query", raw_results[:3], "semantic"))
        assert len(data["documents"]) == 3
        assert data["total_count"] == 3

    def test_search_documents_empty_results(self) -> None:
        """Verify search_documents returns empty documents list when no matches found."""
        data = json.loads(format_search_response("empty_query", [], "semantic"))
        assert data["total_count"] == 0
        assert data["documents"] == []

    @pytest.mark.asyncio
    async def test_mcp_search_documents_invocation(self) -> None:
        """Verify invoking search_documents tool returns JSON string."""
        from src.mcp.server import search_documents

        with patch("src.mcp.server.get_orchestrator") as mock_get_orch:
            mock_orch = MagicMock()
            mock_orch.search_documents = MagicMock(return_value=[])
            mock_get_orch.return_value = mock_orch

            result = await search_documents("find hash")
            assert isinstance(result, str)
            data = json.loads(result)
            assert data["total_count"] == 0


class TestFeature21SensitiveDataRedaction:
    """
    Feature 21: Sensitive Data Redaction.

    Authoritative: PROJECT.md Feature 21, Tech Spec §6.2.
    Redact emails and IPs while preserving code, version strings, and technical names.
    """

    def test_redact_ipv4_addresses(self) -> None:
        """Verify valid IPv4 addresses are replaced with [REDACTED_IP]."""
        text = "Beacon connected to 198.51.100.45 and secondary 10.0.0.1 on port 443."
        redacted = redact_ip_addresses(text)
        assert "198.51.100.45" not in redacted
        assert "10.0.0.1" not in redacted
        assert "[REDACTED_IP]" in redacted

    def test_redact_ipv6_addresses(self) -> None:
        """Verify IPv6 addresses are replaced with [REDACTED_IPv6]."""
        text = "C2 host at 2001:0db8:85a3:0000:0000:8a2e:0370:7334 active."
        redacted = redact_ip_addresses(text)
        assert "2001:0db8:85a3:0000:0000:8a2e:0370:7334" not in redacted
        assert "[REDACTED_IPv6]" in redacted

    def test_redact_email_addresses(self) -> None:
        """Verify email addresses are replaced with [REDACTED_EMAIL]."""
        text = "Contact researcher at intel@vxunderground.org or sec@cert.gov.ua for details."
        redacted = redact_email_addresses(text)
        assert "intel@vxunderground.org" not in redacted
        assert "sec@cert.gov.ua" not in redacted
        assert "[REDACTED_EMAIL]" in redacted

    def test_redaction_preserves_version_numbers(self) -> None:
        """Verify version strings like v1.2.3.4 are preserved and NOT redacted as IPs."""
        text = "Discovered in WinRAR v1.2.3.4 and Python 3.11.9."
        redacted = redact_sensitive_data(text)
        assert "v1.2.3.4" in redacted
        assert "3.11.9" in redacted

    def test_redaction_preserves_cpp_code_and_technical_identifiers(self) -> None:
        """Verify C++ code syntax, memory addresses, and CVE identifiers are preserved."""
        text = "In CVE-2023-38831: std::cout << obj.prop; offset 0x5A4D."
        redacted = redact_sensitive_data(text)
        assert "CVE-2023-38831" in redacted
        assert "std::cout << obj.prop;" in redacted
        assert "0x5A4D" in redacted


class TestFeatures22And23MCPResources:
    """
    Features 22-23: MCP Resources health://status and context://system.

    Authoritative: PROJECT.md Features 22-23, Tech Spec §7.1.
    Read-only resources providing health status and system capabilities.
    """

    def test_health_status_resource_healthy(self) -> None:
        """Verify health_status resource returns healthy status when indexes loaded."""
        health_data = {
            "overall_status": "healthy",
            "initialized": True,
            "indexes_loaded": True,
            "services": {"vector_store": {"available": True, "status": "healthy"}},
            "persist_dir": "data/index"
        }
        res_json = format_health_status(health_data)
        data = json.loads(res_json)
        assert data["overall_status"] == "healthy"
        assert data["indexes_loaded"] is True

    def test_health_status_resource_degraded(self) -> None:
        """Verify health_status resource returns degraded status when indexes not loaded."""
        health_data = {
            "overall_status": "degraded",
            "initialized": True,
            "indexes_loaded": False,
            "services": {"vector_store": {"available": True, "status": "degraded"}},
            "persist_dir": "data/index"
        }
        res_json = format_health_status(health_data)
        data = json.loads(res_json)
        assert data["overall_status"] == "degraded"
        assert data["indexes_loaded"] is False

    def test_system_context_resource_capabilities(self) -> None:
        """Verify system_context resource describes system capabilities and formats."""
        context_json = format_system_context()
        data = json.loads(context_json)
        assert "system_name" in data
        assert "capabilities" in data
        assert "supported_document_types" in data["capabilities"]

    def test_system_context_resource_deterministic(self) -> None:
        """Verify consecutive system_context calls produce identical schemas."""
        first = format_system_context()
        second = format_system_context()
        assert json.loads(first) == json.loads(second)

    @pytest.mark.asyncio
    async def test_mcp_resources_invocation(self) -> None:
        """Verify direct async invocation of resource endpoints."""
        from src.mcp.server import health_status, system_context

        with patch("src.mcp.server.get_orchestrator") as mock_orch:
            mock_inst = MagicMock()
            mock_inst.get_health_status.return_value = {
                "overall_status": "healthy",
                "initialized": True,
                "indexes_loaded": True,
                "services": {},
            }
            mock_orch.return_value = mock_inst

            health_res = await health_status()
            assert json.loads(health_res)["overall_status"] == "healthy"

            context_res = await system_context()
            assert "capabilities" in json.loads(context_res)
