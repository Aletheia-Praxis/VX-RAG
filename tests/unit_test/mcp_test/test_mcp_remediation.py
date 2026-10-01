"""
Unit tests for MCP server and CLI remediation (Defects 5.3, 5.4, and 5.5).

Verifies:
- Defect 5.3: Console StreamHandler redirection from stdout to stderr.
- Defect 5.4: Custom persist_dir propagation to MCP server and orchestrator.
- Defect 5.5: Clean configuration defaults resolution.
"""

from __future__ import annotations

import argparse
import io
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.mcp.server import (
    configure_server,
    health_status,
    query_knowledge_base,
    run_http,
    run_stdio,
    search_documents,
)
from src.utils.logging_config import (
    configure_console_stream,
    get_logger,
    reset_logging_handlers,
)


@pytest.fixture(autouse=True)
def _cleanup_logging_handlers() -> Any:
    """Fixture ensuring logging handlers are reset before and after each test."""
    reset_logging_handlers()
    yield
    reset_logging_handlers()


class TestFastMCPStdioLoggingRedirection:
    """Tests verifying standard output log pollution remediation (Defect 5.3)."""

    def test_configure_console_stream_rebinds_existing_handlers(self) -> None:
        """Verify configure_console_stream redirects active stdout handlers to stderr."""
        test_logger = get_logger("test_rebind_existing")
        stdout_handlers = [
            h
            for h in test_logger.logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stdout_handlers) >= 1
        assert stdout_handlers[0].stream is sys.stdout

        custom_stderr = io.StringIO()
        configure_console_stream(custom_stderr)

        assert stdout_handlers[0].stream is custom_stderr

    def test_configure_console_stream_applies_to_subsequent_loggers(self) -> None:
        """Verify loggers created after configure_console_stream target the designated stream."""
        custom_stream = io.StringIO()
        configure_console_stream(custom_stream)

        new_logger = get_logger("test_subsequent_logger")
        stream_handlers = [
            h
            for h in new_logger.logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1
        assert stream_handlers[0].stream is custom_stream

    def test_reset_logging_handlers_restores_default_stdout(self) -> None:
        """Verify reset_logging_handlers restores sys.stdout as default console stream."""
        custom_stream = io.StringIO()
        configure_console_stream(custom_stream)
        reset_logging_handlers()

        restored_logger = get_logger("test_restored_logger")
        stream_handlers = [
            h
            for h in restored_logger.logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1
        assert stream_handlers[0].stream is sys.stdout

    def test_configure_console_stream_multiple_redirections_and_restoration(self) -> None:
        """Verify configure_console_stream safely handles multiple sequential rebinding calls."""
        test_logger = get_logger("test_multiple_rebind")
        stream_handlers = [
            h
            for h in test_logger.logger.handlers
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1
        assert stream_handlers[0].stream is sys.stdout

        # Rebind to stderr
        configure_console_stream(sys.stderr)
        assert stream_handlers[0].stream is sys.stderr

        # Rebind to custom stream
        custom_buf = io.StringIO()
        configure_console_stream(custom_buf)
        assert stream_handlers[0].stream is custom_buf

        # Rebind back to stdout
        configure_console_stream(sys.stdout)
        assert stream_handlers[0].stream is sys.stdout

    def test_handle_serve_configures_stream_before_server_import(self) -> None:
        """Verify handle_serve invokes configure_console_stream before any server imports."""
        from src.cli import handle_serve

        args = argparse.Namespace(
            transport="stdio",
            host="localhost",
            port=8000,
            config="config/settings.yaml",
            persist_dir="data/index",
        )

        call_order: list[str] = []

        def _mock_cfg_stream(stream: Any) -> None:
            call_order.append("configure_console_stream")

        def _mock_cfg_server(*args: Any, **kwargs: Any) -> Any:
            call_order.append("configure_server")
            return MagicMock()

        def _mock_run_stdio(*args: Any, **kwargs: Any) -> None:
            call_order.append("run_stdio")

        with (
            patch("src.utils.logging_config.configure_console_stream", side_effect=_mock_cfg_stream),
            patch("src.mcp.server.configure_server", side_effect=_mock_cfg_server),
            patch("src.mcp.server.run_stdio", side_effect=_mock_run_stdio),
        ):
            handle_serve(args)

        assert call_order == ["configure_console_stream", "configure_server", "run_stdio"]


class TestMCPServerPersistDirWiring:
    """Tests verifying custom persist-dir propagation in MCP server (Defect 5.4)."""

    def test_configure_server_propagates_custom_paths(self) -> None:
        """Verify configure_server passes custom config_path and persist_dir to orchestrator."""
        mock_orch = MagicMock()
        mock_orch.persist_dir = Path("data/custom_index_1000")
        mock_orch.config_path = Path("config/custom_settings.yaml")

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orch) as mock_get_orch:
            orch = configure_server(
                config_path="config/custom_settings.yaml",
                persist_dir="data/custom_index_1000",
            )
            assert orch is mock_orch
            mock_get_orch.assert_called_once_with(
                config_path="config/custom_settings.yaml",
                persist_dir="data/custom_index_1000",
            )

    @pytest.mark.asyncio
    async def test_mcp_tools_use_configured_persist_dir(self) -> None:
        """Verify MCP tools invoke get_orchestrator with the configured persist_dir."""
        mock_orch = MagicMock()
        mock_orch.query_async = AsyncMock(return_value=MagicMock(query="q", context=[], citations=[]))
        mock_orch.search_documents = MagicMock(return_value=[])
        mock_orch.get_health_status = MagicMock(return_value={"overall_status": "healthy"})

        with patch("src.mcp.server.get_orchestrator", return_value=mock_orch) as mock_get_orch:
            configure_server(
                config_path="config/test_wiring.yaml",
                persist_dir="data/test_custom_dir",
            )

            # Test query tool
            await query_knowledge_base("test query")
            mock_get_orch.assert_called_with(
                config_path="config/test_wiring.yaml",
                persist_dir="data/test_custom_dir",
            )

            # Test search tool
            await search_documents("test search")
            mock_get_orch.assert_called_with(
                config_path="config/test_wiring.yaml",
                persist_dir="data/test_custom_dir",
            )

            # Test health status resource
            await health_status()
            mock_get_orch.assert_called_with(
                config_path="config/test_wiring.yaml",
                persist_dir="data/test_custom_dir",
            )

    def test_run_stdio_initializes_with_persist_dir_and_runs_mcp(self) -> None:
        """Verify run_stdio initializes the orchestrator with persist_dir and invokes mcp.run()."""
        mock_orch = MagicMock()
        mock_orch.get_health_status.return_value = {
            "overall_status": "healthy",
            "initialized": True,
            "indexes_loaded": True,
        }

        with (
            patch("src.mcp.server.configure_server", return_value=mock_orch) as mock_cfg,
            patch("src.mcp.server.mcp.run") as mock_mcp_run,
            patch("src.mcp.server.configure_console_stream") as mock_cfg_stream,
        ):
            run_stdio(
                config_path="config/settings.yaml",
                persist_dir="data/index_test_1000",
            )
            mock_cfg_stream.assert_called_once_with(sys.stderr)
            mock_cfg.assert_called_once_with(
                config_path="config/settings.yaml",
                persist_dir="data/index_test_1000",
            )
            mock_mcp_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_http_initializes_and_serves(self) -> None:
        """Verify run_http initializes the orchestrator with persist_dir and starts uvicorn."""
        mock_server = MagicMock()
        mock_server.serve = AsyncMock()

        with (
            patch("src.mcp.server.start_server", new_callable=AsyncMock) as mock_start,
            patch("uvicorn.Server", return_value=mock_server),
        ):
            await run_http(
                host="127.0.0.1",
                port=8080,
                config_path="config/settings.yaml",
                persist_dir="data/index_test_http",
            )
            mock_start.assert_awaited_once_with(
                config_path="config/settings.yaml",
                persist_dir="data/index_test_http",
            )
            mock_server.serve.assert_awaited_once()

    def test_run_sse_removed_from_mcp_server(self) -> None:
        """Verify run_sse is completely removed from src.mcp.server."""
        import src.mcp.server as mcp_srv

        assert not hasattr(mcp_srv, "run_sse")

    def test_configure_server_updates_defaults_and_timeouts(self) -> None:
        """Verify configure_server reloads mcp_defaults and mcp_timeouts."""
        import src.mcp.server as mcp_srv

        with (
            patch("src.mcp.server.get_orchestrator", return_value=MagicMock()),
            patch("src.mcp.server.get_mcp_defaults", return_value={"test_default": 123}),
            patch("src.mcp.server.get_mcp_timeouts", return_value={"test_timeout": 456.0}),
        ):
            configure_server(config_path="config/settings.yaml", persist_dir="data/index")
            assert mcp_srv.mcp_defaults.get("test_default") == 123
            assert mcp_srv.mcp_timeouts.get("test_timeout") == 456.0

    def test_cli_serve_stdio_subprocess_clean_stdout(self) -> None:
        """Verify cli serve with stdio transport in a fresh subprocess never pollutes stdout."""
        script = (
            "import sys\n"
            "from unittest.mock import patch, MagicMock\n"
            "mock_orch = MagicMock()\n"
            "mock_orch.get_health_status.return_value = {'overall_status': 'healthy', 'initialized': True, 'indexes_loaded': True}\n"
            "with patch('src.mcp.server.mcp.run'), patch('src.mcp.server.configure_server', return_value=mock_orch):\n"
            "    from src.cli import main\n"
            "    sys.argv = ['cli.py', 'serve', '--transport', 'stdio', '--persist-dir', 'data/index']\n"
            "    main()\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0
        assert proc.stdout == ""
        assert "Starting MCP server with transport: stdio" in proc.stderr

    def test_direct_mcp_server_module_clean_stdout(self) -> None:
        """Verify direct module execution (python -m src.mcp.server) never pollutes stdout."""
        script = (
            "import sys\n"
            "from unittest.mock import patch, MagicMock\n"
            "mock_orch = MagicMock()\n"
            "mock_orch.get_health_status.return_value = {'overall_status': 'healthy', 'initialized': True, 'indexes_loaded': True}\n"
            "with patch('src.mcp.server.mcp.run'), patch('src.mcp.server.configure_server', return_value=mock_orch):\n"
            "    import runpy\n"
            "    runpy.run_module('src.mcp.server', run_name='__main__')\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0
        assert proc.stdout == ""
        assert "Starting VX-RAG MCP server in STDIO mode" in proc.stderr

