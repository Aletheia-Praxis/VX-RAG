"""
Pytest configuration and shared fixtures for VX-RAG End-to-End (E2E) tests.

Provides isolated test environments, sample Vx Underground cybersecurity documents,
configuration builders, CLI execution helpers, and MCP protocol helpers.
"""

from __future__ import annotations

import io
import os
import sys
import yaml
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
)

import pytest

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


@pytest.fixture
def sample_cybersecurity_corpus(tmp_path: Path) -> Path:
    """
    Generate a representative multi-document Vx Underground cybersecurity corpus.

    Includes malware analysis reports, reverse engineering snippets, C2 telemetry,
    vulnerability disclosures, and code blocks with realistic indicators.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Path to the directory containing raw sample documents.
    """
    corpus_dir = tmp_path / "raw_corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # Document 1: Emotet Banking Trojan & Modular Loader Analysis
    doc1_content = """# Emotet Modular Banking Trojan Analysis
Author: vx-underground-researcher
Date: 2023-11-14
Source: Vx-Underground Collection

## Executive Summary
Emotet functions primarily as a modular downloader and credential harvester.
The malware establishes persistence via scheduled tasks and service creation.
Command and Control communication is encrypted over HTTP/HTTPS with custom XOR keys.

## Network Telemetry and C2 Infrastructure
Telemetry analysis identified primary active C2 beacons connecting to:
- 198.51.100.45:443 (Active beacon server in AS64496)
- 203.0.113.195:8080 (Secondary failover proxy)
- IPv6 C2 endpoint: 2001:0db8:85a3:0000:0000:8a2e:0370:7334
Report submissions should be sent to intel@vxunderground.org or security-team@cert.gov.ua.

## Loader Deobfuscation Routine
The unpacking routine decodes the payload in memory using the following C++ logic:
```cpp
#include <windows.h>
#include <iostream>

void DeobfuscatePayload(BYTE* pBuffer, DWORD dwSize, BYTE bKey) {
    for (DWORD i = 0; i < dwSize; ++i) {
        pBuffer[i] ^= (bKey + (i % 7));
    }
    std::cout << "[*] Decrypted bytes: " << dwSize << std::endl;
}
```

## Indicators of Compromise (IOCs)
- SHA256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
- File Path: %APPDATA%\\Local\\Microsoft\\Windows\\services32.exe
"""
    (corpus_dir / "emotet_modular_trojan.md").write_text(doc1_content, encoding="utf-8")

    # Document 2: Ransomware Volume Shadow Copy Invalidation
    doc2_content = """# LockBit and Conti Volume Shadow Deletion Mechanics
Classification: T1490 - Inhibit System Recovery
Corpus: Vx Underground Technical Series

## Overview
Modern ransomware families automate the invalidation of Volume Shadow Copies
to prevent local data recovery prior to executing symmetric file encryption.

## Command Execution Sequence
The operator dispatches high-privilege subprocess calls executing:
```cmd
vssadmin.exe delete shadows /all /quiet
wmic shadowcopy delete
bcdedit /set {default} bootstatuspolicy ignoreallfailures
bcdedit /set {default} recoveryenabled no
```

## Cryptographic Key Derivation Routine
Ransomware creates ephemeral session keys using Windows CryptoAPI:
```cpp
BOOL InitializeRansomKey(HCRYPTPROV* hProv, HCRYPTKEY* hKey) {
    if (!CryptAcquireContext(hProv, NULL, MS_ENH_RSA_AES_PROV, PROV_RSA_AES, CRYPT_VERIFYCONTEXT)) {
        return FALSE;
    }
    return CryptGenKey(*hProv, CALG_AES_256, CRYPT_EXPORTABLE, hKey);
}
```
"""
    (corpus_dir / "ransomware_recovery_inhibition.md").write_text(doc2_content, encoding="utf-8")

    # Document 3: CVE-2023 Memory Corruption Advisory
    doc3_content = """VULNERABILITY ADVISORY: CVE-2023-38831 WinRAR Execution Precondition
Published: 2023-08-23
Reporter: security-lead@threatlabz.example.com

Affected Software: WinRAR versions prior to 6.23
Vulnerability Type: Logic flaw / Arbitrary code execution via spoofed file extensions

Root Cause Analysis:
When processing ZIP archives containing specially crafted filenames with identical names
for folders and executable payloads, WinRAR inadvertently dispatches execution to the
contained script file rather than opening the target decoy document.
Target IOC Hash: 404c000e2f99d98bb55f4104d43615ec
Contact: alerts@cert.example.net or phone +1-800-555-0199.
"""
    (corpus_dir / "cve_2023_38831_advisory.txt").write_text(doc3_content, encoding="utf-8")

    # Document 4: Reverse Engineering PE Header Reference
    doc4_content = """# Portable Executable (PE) Format Reference for Analysts
Technical Manual - Reverse Engineering Series

## Section Headers and Permissions
Portable Executable section attributes determine page memory permissions at runtime:

| Section Name | Typical Size | Characteristics Flags | Memory Protection |
|--------------|--------------|-----------------------|-------------------|
| .text        | 45 KB        | IMAGE_SCN_MEM_EXECUTE | PAGE_EXECUTE_READ |
| .data        | 12 KB        | IMAGE_SCN_MEM_WRITE   | PAGE_READWRITE    |
| .rdata       | 28 KB        | IMAGE_SCN_MEM_READ    | PAGE_READONLY     |
| .rsrc        | 8 KB         | IMAGE_SCN_CNT_INITIAL | PAGE_READONLY     |

## Hex Dump Inspection
```hex
4D 5A 90 00 03 00 00 00 04 00 00 00 FF FF 00 00
B8 00 00 00 00 00 00 00 40 00 00 00 00 00 00 00
```
This signature (0x5A4D) indicates the legacy DOS stub header.
"""
    (corpus_dir / "pe_header_reference.md").write_text(doc4_content, encoding="utf-8")

    return corpus_dir


@pytest.fixture
def isolated_e2e_env(tmp_path: Path) -> Dict[str, Any]:
    """
    Provide an isolated filesystem environment and configuration for E2E tests.

    Creates directories for raw documents, processed nodes, FAISS indices,
    and a custom settings.yaml file pointing exclusively to temporary paths.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        Dictionary mapping directory keys to Path objects.
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    index_dir = tmp_path / "index"
    config_dir = tmp_path / "config"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = config_dir / "test_settings.yaml"

    config_data: Dict[str, Any] = {
        "data_dir": str(tmp_path),
        "raw_data_dir": str(raw_dir),
        "processed_data_dir": str(processed_dir),
        "index_dir": str(index_dir),
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "embedding_device": "cpu",
        "embedding_batch_size": 10,
        "embedding_trust_remote_code": False,
        "chunk_size": 1024,
        "chunk_overlap": 150,
        "vector_store": "faiss",
        "faiss": {
            "hnsw_m": 32,
            "metric": "inner_product",
        },
        "retriever": {
            "semantic_top_k": 10,
            "hybrid_alpha": 0.5,
            "enable_hybrid": True,
        },
        "bm25": {
            "index_dir": str(index_dir / "bm25"),
            "similarity_top_k": 10,
            "enable_persistence": True,
        },
        "reranker": {
            "model_name": "BAAI/bge-reranker-base",
            "top_k": 5,
            "device": "cpu",
            "metadata_boost": 0.1,
            "enable_metadata_prioritization": True,
        },
        "mcp": {
            "host": "127.0.0.1",
            "port": 8000,
            "debug": True,
            "query_timeout": 60.0,
            "top_k": 5,
            "search_top_k": 10,
            "token_budget": 4000,
            "apply_redaction": True,
        },
        "duplicate_detection": {
            "similarity_threshold": 0.95,
            "hash_algorithm": "sha256",
        },
        "logging": {
            "log_level": "INFO",
            "log_file": str(tmp_path / "test.log"),
            "log_format": "json",
            "log_structured_events": True,
            "log_metrics": False,
        },
    }

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    return {
        "root": tmp_path,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "index_dir": index_dir,
        "config_path": config_path,
        "config_data": config_data,
    }


class CLIExecutionResult:
    """Encapsulates the standard output, standard error, and exit status of a CLI command."""

    def __init__(self, exit_code: int, stdout: str, stderr: str) -> None:
        """
        Initialize the execution result.

        Args:
            exit_code: Integer exit status code (0 for success).
            stdout: Captured standard output string.
            stderr: Captured standard error string.
        """
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def run_cli_command() -> Generator[Any, None, None]:
    """
    Provide an in-process CLI execution helper that simulates invoking python src/cli.py.

    Captures standard output and error streams while intercepting sys.exit calls.

    Yields:
        Callable runner accepting list of string arguments and returning CLIExecutionResult.
    """
    def _runner(args: List[str]) -> CLIExecutionResult:
        """Run CLI with arguments."""
        from src.cli import main

        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()

        sys.argv = ["src/cli.py"] + args
        sys.stdout = captured_stdout
        sys.stderr = captured_stderr

        exit_code = 0
        try:
            main()
        except SystemExit as exc:
            exit_code = exc.code if isinstance(exc.code, int) else (1 if exc.code else 0)
        except Exception as err:
            captured_stderr.write(str(err))
            exit_code = 1
        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return CLIExecutionResult(
            exit_code=exit_code,
            stdout=captured_stdout.getvalue(),
            stderr=captured_stderr.getvalue(),
        )

    yield _runner
