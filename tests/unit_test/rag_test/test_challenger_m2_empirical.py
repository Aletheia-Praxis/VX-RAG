"""
Empirical adversarial validation suite for Milestone M2.

Authored by challenger_m2_iter3 to stress-test:
1. Complete sentence and paragraph distributions across full token spectra (10 to 300 tokens).
2. Strict overlap invariant: ratio in [10%, 15%] OR exactly 0.0%.
3. Absolute elimination of illegal overlap in (0%, 10%) and >15%.
4. Code fences > 512 tokens partitioned with intact opening/closing fences.
5. Tables (2-50 rows) and hex dumps remaining 100% intact.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import pytest

from src.rag.ingestion import (
    DoclingPipeline,
    count_tokens,
)

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture  # noqa: F401
    from _pytest.fixtures import FixtureRequest  # noqa: F401
    from _pytest.logging import LogCaptureFixture  # noqa: F401
    from _pytest.monkeypatch import MonkeyPatch  # noqa: F401
    from pytest_mock.plugin import MockerFixture  # noqa: F401


def _compute_chunk_overlap(c1: str, c2: str) -> int:
    """
    Compute token overlap between c1 suffix and c2 prefix.

    Args:
        c1: The preceding chunk text.
        c2: The subsequent chunk text.

    Returns:
        Number of overlapping tokens according to count_tokens.
    """
    overlap_str = ""
    for k in range(1, len(c2) + 1):
        prefix = c2[:k]
        if c1.endswith(prefix):
            overlap_str = prefix
    if not overlap_str:
        return 0
    return count_tokens(overlap_str)


class TestChallengerOverlapDistributionSweep:
    """Adversarially sweep sentence and paragraph distributions across token lengths."""

    @pytest.mark.parametrize(
        "target_tokens",
        [15, 30, 45, 60, 75, 80, 85, 90, 95, 100, 105, 110, 120, 140, 160, 200],
    )
    def test_complete_sentence_distribution_sweep(self, target_tokens: int) -> None:
        """
        Verify that for arbitrary sentence lengths, overlap is strictly [10%, 15%] or 0.0%.

        Args:
            target_tokens: Desired token length for individual sentences.
        """
        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)

        base_words = ["telemetry", "forensic", "adversary", "payload", "socket", "beacon"]
        sentence_parts: list[str] = []
        while count_tokens(" ".join(sentence_parts) + ".") < target_tokens:
            sentence_parts.append(base_words[len(sentence_parts) % len(base_words)])

        sentence = " ".join(sentence_parts) + "."
        num_paras = max(40, (2000 // max(1, target_tokens)) + 5)
        paragraphs = [f"Item {idx:03d}: {sentence}" for idx in range(num_paras)]
        doc = "\n\n".join(paragraphs)

        nodes = pipeline.chunk_markdown(doc)
        assert len(nodes) >= 2, f"Expected multiple chunks for tokens={target_tokens}"

        for i in range(len(nodes) - 1):
            c1 = nodes[i].text
            c2 = nodes[i + 1].text
            toks = _compute_chunk_overlap(c1, c2)
            ratio = toks / 1024

            is_valid = (ratio == 0.0) or (0.10 <= ratio <= 0.15)
            assert is_valid, (
                f"Target {target_tokens} tokens: Transition {i}->{i+1} has illegal overlap "
                f"{ratio:.2%} ({toks} tokens). Must be 0.0% or within [10%, 15%]!"
            )
            assert not (0.0 < ratio < 0.10), f"Found under-budget overlap {ratio:.2%}"
            assert ratio <= 0.15, f"Found over-budget overlap {ratio:.2%}"

    def test_randomized_heterogeneous_sentence_distribution(self) -> None:
        """
        Verify heavily randomized sentence distribution eliminates all illegal overlap ratios.
        """
        rng = random.Random(42)
        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)

        paragraphs: list[str] = []
        for p in range(50):
            p_sentences: list[str] = []
            num_sentences = rng.randint(1, 8)
            for s in range(num_sentences):
                sentence_len = rng.randint(5, 70)
                words = [f"term_{rng.randint(100, 999)}" for _ in range(sentence_len)]
                p_sentences.append(" ".join(words) + ".")
            paragraphs.append(" ".join(p_sentences))

        doc = "\n\n".join(paragraphs)
        nodes = pipeline.chunk_markdown(doc)
        assert len(nodes) >= 3

        for i in range(len(nodes) - 1):
            c1 = nodes[i].text
            c2 = nodes[i + 1].text
            toks = _compute_chunk_overlap(c1, c2)
            ratio = toks / 1024

            is_valid = (ratio == 0.0) or (0.10 <= ratio <= 0.15)
            assert is_valid, (
                f"Randomized distribution: Transition {i}->{i+1} has illegal overlap "
                f"{ratio:.2%} ({toks} tokens)."
            )


class TestChallengerCodeFencePreservation:
    """Stress-test code blocks exceeding 512 tokens for fence preservation."""

    @pytest.mark.parametrize("line_count", [40, 80, 120])
    def test_large_code_blocks_over_512_tokens_preserve_fences(self, line_count: int) -> None:
        """
        Verify code blocks > 512 tokens partition into chunks with intact opening and closing fences.

        Args:
            line_count: Number of code lines in block.
        """
        pipeline = DoclingPipeline(default_chunk_size=1024, code_chunk_size=512)

        code_lines: list[str] = [
            f"    uint64_t instruction_addr_{i:04d} = resolve_dynamic_symbol({i});"
            for i in range(line_count)
        ]
        raw_code = "```c\n" + "\n".join(code_lines) + "\n```"
        total_tokens = count_tokens(raw_code)

        doc = f"# Technical Analysis\n\n{raw_code}\n\nConcluded disassembly."
        nodes = pipeline.chunk_markdown(doc)

        # Filter nodes that contain the code lines
        code_nodes = [n for n in nodes if "instruction_addr_" in n.text]
        assert len(code_nodes) >= 1

        for idx, node in enumerate(code_nodes):
            text = node.text
            # Must start and end with code fence
            assert text.startswith("```c\n"), f"Code chunk {idx} missing opening fence"
            assert text.endswith("\n```"), f"Code chunk {idx} missing closing fence"
            assert count_tokens(text) <= 512, (
                f"Code chunk {idx} exceeds code budget: {count_tokens(text)} > 512"
            )

        # Verify all lines are preserved across the code chunks
        reconstructed_lines: list[str] = []
        for node in code_nodes:
            # strip fences
            body = node.text.removeprefix("```c\n").removesuffix("\n```")
            reconstructed_lines.extend(body.split("\n"))

        assert reconstructed_lines == code_lines, "Code lines were dropped or corrupted"


class TestChallengerTableAndHexPreservation:
    """Verify table and hex dump preservation under adversarial conditions."""

    @pytest.mark.parametrize("row_count", [7, 13, 23, 37, 50])
    def test_prime_sized_tables_remain_monolithic(self, row_count: int) -> None:
        """
        Verify markdown tables with prime row counts remain monolithic in their chunks.

        Args:
            row_count: Number of data rows.
        """
        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)

        header = "| ID | Metric Name | Observation Status | Risk Score | Notes |"
        divider = "|---|---|---|---|---|"
        rows = [
            f"| {r:03d} | METRIC_EVASION_{r:02X} | CONFIRMED_THREAT | {r * 2} | Analyzed |"
            for r in range(row_count)
        ]
        table_str = "\n".join([header, divider, *rows])

        # Flank with prose
        prefix = " ".join([f"Context statement alpha {i}." for i in range(80)])
        suffix = " ".join([f"Context statement omega {i}." for i in range(80)])
        doc = f"{prefix}\n\n{table_str}\n\n{suffix}"

        nodes = pipeline.chunk_markdown(doc)
        table_chunks = [n for n in nodes if "| 000 | METRIC_EVASION_00 |" in n.text]
        assert len(table_chunks) == 1, "Table was fragmented across multiple chunks"

        chunk_text = table_chunks[0].text
        assert header in chunk_text
        assert divider in chunk_text
        for r in range(row_count):
            assert f"| {r:03d} |" in chunk_text, f"Row {r} was missing from table chunk"

    def test_hex_dump_with_discontinuous_addresses_intact(self) -> None:
        """Verify hex dumps with addresses and ascii decodes remain intact."""
        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)

        hex_lines = [
            "00401000  48 83 ec 28 e8 1b 00 00  00 48 83 c4 28 c3 cc cc  |H..(.....H..(..|",
            "00401010  48 89 5c 24 08 48 89 6c  24 10 48 89 74 24 18 57  |H.\\$.H.l$.H.t$.W|",
            "00401020  48 83 ec 20 48 8b f9 e8  b3 0f 00 00 48 8b d8 48  |H.. H.......H..H|",
            "00401030  85 c0 74 19 48 8b cb e8  85 02 00 00 48 8b cb e8  |..t.H.......H...|",
        ]
        hex_dump = "\n".join(hex_lines)

        prefix = " ".join([f"Preceding forensic entry {i}." for i in range(80)])
        suffix = " ".join([f"Following forensic entry {i}." for i in range(80)])
        doc = f"{prefix}\n\n{hex_dump}\n\n{suffix}"

        nodes = pipeline.chunk_markdown(doc)
        hex_chunks = [n for n in nodes if "00401000" in n.text]
        assert len(hex_chunks) == 1, "Hex dump was split across chunks"

        for line in hex_lines:
            assert line in hex_chunks[0].text, f"Hex line '{line}' missing"
