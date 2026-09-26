"""
Tier 2 Boundary Tests: Sensitive Data Redaction Edge Cases and False Positives.

Authoritative Source: Tech Spec §6.2, Spec Miner Handout Edge Cases #13, #14.
Verifies boundary octet ranges (0-255), version strings, C++ stream operators,
complex email formats, and technical identifier preservation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from src.mcp.formatters import (
    redact_email_addresses,
    redact_ip_addresses,
    redact_sensitive_data,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestBoundaryRedaction:
    """Boundary conditions for IP/email redaction and syntax preservation."""

    def test_redaction_boundary_valid_octets_0_and_255(self) -> None:
        """Verify boundary valid IPv4 addresses (0.0.0.0 and 255.255.255.255) are redacted."""
        text = "Binding on 0.0.0.0 and broadcasting to 255.255.255.255."
        redacted = redact_ip_addresses(text)
        assert "0.0.0.0" not in redacted
        assert "255.255.255.255" not in redacted
        assert redacted == "Binding on [REDACTED_IP] and broadcasting to [REDACTED_IP]."

    def test_redaction_invalid_octets_greater_than_255_ignored(self) -> None:
        """Verify numbers with octets > 255 (e.g. 256.1.2.3 or 999.888.777.666) are NOT redacted."""
        text = "Synthetic numbers 256.1.2.3 and 999.888.777.666."
        redacted = redact_ip_addresses(text)
        assert "256.1.2.3" in redacted
        assert "999.888.777.666" in redacted
        assert "[REDACTED_IP]" not in redacted

    def test_redaction_version_strings_preserved(self) -> None:
        """Verify version strings such as v1.2.3.4 are preserved without false-positive redaction."""
        text = "Deploying agent release v1.2.3.4 with kernel patch 6.1.0.28."
        redacted = redact_sensitive_data(text)
        assert "v1.2.3.4" in redacted

    def test_redaction_cpp_stream_insertion_preserved(self) -> None:
        """Verify C++ left shift / stream operators (<<) are not corrupted by redaction."""
        code = "std::cout << obj.property << 100 << std::endl;"
        redacted = redact_sensitive_data(code)
        assert code == redacted

    def test_redaction_complex_email_addresses(self) -> None:
        """Verify emails with plus addressing, subdomains, and hyphenated domains are redacted."""
        text = "Report to sec.ops+vx@subdomain.threat-intel.example.com or admin_team@cert.gov.ua."
        redacted = redact_email_addresses(text)
        assert "sec.ops+vx@subdomain.threat-intel.example.com" not in redacted
        assert "admin_team@cert.gov.ua" not in redacted
        assert redacted.count("[REDACTED_EMAIL]") == 2

    def test_redaction_multiple_adjacent_ips_on_single_line(self) -> None:
        """Verify multiple comma-separated or adjacent IP addresses are all redacted."""
        text = "Cluster nodes: 192.168.1.1, 10.0.0.2, 172.16.0.3."
        redacted = redact_ip_addresses(text)
        assert "192.168.1.1" not in redacted
        assert "10.0.0.2" not in redacted
        assert "172.16.0.3" not in redacted
        assert redacted.count("[REDACTED_IP]") == 3

    def test_redaction_preserves_sha256_hashes_and_cves(self) -> None:
        """Verify 64-char hexadecimal file hashes and CVE identifiers remain intact."""
        sha256_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        cve_id = "CVE-2023-38831"
        text = f"Sample hash {sha256_hash} linked to advisory {cve_id}."
        redacted = redact_sensitive_data(text)
        assert sha256_hash in redacted
        assert cve_id in redacted
