"""
Stress test suite for chunk overlap compliance in src/rag/ingestion.py.

Empirically challenges:
1. Multi-sentence prose with paragraphs of 50, 100, 150, 200, and 300 words.
2. Variable sentence lengths (short, medium, long, and mixed sentences).
3. Indivisible blocks: single sentences exceeding overlap budget (yielding 0% overlap).
4. Boundary condition: sentences of ~60-80 words where 1 sentence is <10% and 2 sentences >15%.
5. Strict enforcement that every adjacent chunk transition has overlap ratio in [10%, 15%] OR exactly 0.0%.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from src.rag.ingestion import DoclingPipeline, count_tokens

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
        c1: The preceding chunk string.
        c2: The subsequent chunk string.

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


class TestProseParagraphOverlapStress:
    """Stress tests for chunk overlap across various paragraph and sentence structures."""

    @pytest.mark.parametrize("para_words", [50, 100, 150, 200, 300])
    def test_prose_standard_sentences_overlap_in_range(self, para_words: int) -> None:
        """
        Verify multi-sentence paragraphs (15-25 words per sentence) satisfy [10%, 15%] overlap.

        Args:
            para_words: Target word count per paragraph (50, 100, 150, 200, 300 words).
        """
        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)
        words_per_sentence = 20
        sentences_per_para = max(2, para_words // words_per_sentence)

        paras: list[str] = []
        for p in range(40):
            sentences: list[str] = []
            for s in range(sentences_per_para):
                sentences.append(
                    f"Telemetry report segment {p:02d}_{s:02d} indicates process injection activity "
                    f"originating from remote process with suspicious attributes."
                )
            paras.append(" ".join(sentences))

        document = "\n\n".join(paras)
        nodes = pipeline.chunk_markdown(document)
        assert len(nodes) >= 2, f"Expected multiple chunks for para_words={para_words}"

        for i in range(len(nodes) - 1):
            c1 = nodes[i].text
            c2 = nodes[i + 1].text
            toks_overlap = _compute_chunk_overlap(c1, c2)
            ratio = toks_overlap / 1024

            assert 0.10 <= ratio <= 0.15, (
                f"Para words {para_words}: Transition {i}->{i+1} has overlap {ratio:.2%} "
                f"({toks_overlap} tokens), expected strictly between 10% and 15%."
            )

    def test_150_word_paragraphs_with_longer_sentences(self) -> None:
        """
        Test 150-word paragraphs formed by two 75-word sentences.

        Each sentence is ~95 tokens. Two sentences = ~190 tokens (>15%).
        Checks whether chunker allows an under-budget overlap (~9.3%) instead of strictly [10%, 15%] or 0%.
        """
        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)

        paras: list[str] = []
        for p in range(30):
            s1_words = [f"signal_metric_alpha_{p}_{i}" for i in range(75)]
            s2_words = [f"signal_metric_beta_{p}_{i}" for i in range(75)]
            s1 = " ".join(s1_words) + "."
            s2 = " ".join(s2_words) + "."
            paras.append(f"{s1} {s2}")


        document = "\n\n".join(paras)
        nodes = pipeline.chunk_markdown(document)
        assert len(nodes) >= 2

        for i in range(len(nodes) - 1):
            c1 = nodes[i].text
            c2 = nodes[i + 1].text
            toks_overlap = _compute_chunk_overlap(c1, c2)
            ratio = toks_overlap / 1024

            assert ratio == 0.0 or (0.10 <= ratio <= 0.15), (
                f"150-word para (2x 75-word sentences): Transition {i}->{i+1} has invalid overlap {ratio:.2%} "
                f"({toks_overlap} tokens). Must be 0.0% or in [10%, 15%]!"
            )


    def test_single_sentence_65_word_paragraphs_under_budget(self) -> None:
        """
        Verify that 65-word single-sentence paragraphs (~85 tokens) do not produce illegal under-budget overlap.

        Since 1 paragraph is ~85 tokens (<10%) and 2 paragraphs is ~170 tokens (>15%),
        the overlap cannot satisfy [10%, 15%]. Per the specification, it must drop to 0.0%,
        rather than allowing an illegal ~8.3% overlap.
        """
        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)

        paras: list[str] = []
        for p in range(25):
            words = [f"signal_metric_{p}_{i}" for i in range(65)]
            paras.append(" ".join(words) + ".")

        document = "\n\n".join(paras)
        nodes = pipeline.chunk_markdown(document)
        assert len(nodes) >= 2

        for i in range(len(nodes) - 1):
            c1 = nodes[i].text
            c2 = nodes[i + 1].text
            toks_overlap = _compute_chunk_overlap(c1, c2)
            ratio = toks_overlap / 1024

            assert ratio == 0.0 or (0.10 <= ratio <= 0.15), (
                f"65-word single-sentence paras: Transition {i}->{i+1} has invalid overlap {ratio:.2%} "
                f"({toks_overlap} tokens). Must be 0.0% or in [10%, 15%], but got under-budget overlap!"
            )

    @pytest.mark.parametrize("para_words", [150, 200, 300])
    def test_single_sentence_large_paragraphs_drop_to_zero_overlap(
        self, para_words: int
    ) -> None:


        """
        Verify single-sentence paragraphs exceeding max_overlap (153 tokens) drop to 0% overlap.

        Since a single sentence cannot be divided without breaking sentence integrity,
        and its size exceeds 15% of chunk size, the overlap ratio must be 0%.

        Args:
            para_words: Paragraph word count for monolithic single sentences (150, 200, 300 words).
        """
        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)

        paras: list[str] = []
        for p in range(25):
            # Monolithic sentence with no internal punctuation
            words = [f"signal_metric_{p}_{i}" for i in range(para_words)]
            paras.append(" ".join(words) + ".")

        document = "\n\n".join(paras)
        nodes = pipeline.chunk_markdown(document)
        assert len(nodes) >= 2, f"Expected multiple chunks for monolithic paras of {para_words} words"

        for i in range(len(nodes) - 1):
            c1 = nodes[i].text
            c2 = nodes[i + 1].text
            toks_overlap = _compute_chunk_overlap(c1, c2)
            ratio = toks_overlap / 1024

            assert ratio == 0.0 or (0.10 <= ratio <= 0.15), (
                f"Monolithic para {para_words} words: Transition {i}->{i+1} has invalid overlap {ratio:.2%} "
                f"({toks_overlap} tokens). Must be 0% or strictly between 10% and 15%."
            )

    def test_variable_sentence_lengths_strictly_in_range_or_zero(self) -> None:
        """
        Verify that with highly varied sentence lengths (5 to 80 words),
        all chunk transitions are strictly in [10%, 15%] OR drop to 0%.
        """
        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)

        # Mix of short (5-10 words), medium (20 words), and long (40-60 words) sentences
        sentence_templates = [
            "Network alert triggered.",
            "Host telemetry detected anomalous service installation.",
            "The advanced threat adversary persisted across reboot by installing a malicious service.",
            "During in-depth forensic investigation of memory dump samples, analysts extracted several "
            "obfuscated DLL modules exhibiting dynamic API resolution and anti-debugging capabilities "
            "designed specifically to evade sandbox analysis.",
        ]

        paras: list[str] = []
        for p in range(50):
            p_sentences: list[str] = []
            for s_idx in range(6):
                template = sentence_templates[(p + s_idx) % len(sentence_templates)]
                p_sentences.append(f"Section {p}.{s_idx}: {template}")
            paras.append(" ".join(p_sentences))

        document = "\n\n".join(paras)
        nodes = pipeline.chunk_markdown(document)
        assert len(nodes) >= 3

        for i in range(len(nodes) - 1):
            c1 = nodes[i].text
            c2 = nodes[i + 1].text
            toks_overlap = _compute_chunk_overlap(c1, c2)
            ratio = toks_overlap / 1024

            assert ratio == 0.0 or (0.10 <= ratio <= 0.15), (
                f"Variable sentence test: Transition {i}->{i+1} has invalid overlap {ratio:.2%} "
                f"({toks_overlap} tokens). Must be 0.0% or in [10%, 15%]."
            )

    @pytest.mark.parametrize("exact_tokens", [60, 70, 80, 85, 90, 95])
    def test_sentences_in_awkward_size_window(self, exact_tokens: int) -> None:
        """
        Adversarially challenge sentences where 1 sentence is <10% (102 tokens)
        and 2 sentences exceed 15% (153 tokens).

        Verify whether the chunker strictly respects [10%, 15%] OR drops to 0.0%.
        Under no circumstances should an under-budget overlap (e.g. 7-9%) be returned.
        """
        pipeline = DoclingPipeline(default_chunk_size=1024, overlap=128)

        # Build sentence with exact target tokens
        words = ["the", "system", "administrator", "detected", "an", "incident", "in", "log"]
        sentence_parts: list[str] = []
        while count_tokens(" ".join(sentence_parts) + ".") < exact_tokens:
            sentence_parts.append(words[len(sentence_parts) % len(words)])

        calibrated_sentence = " ".join(sentence_parts) + "."
        actual_tokens = count_tokens(calibrated_sentence)

        paras: list[str] = [f"Record {p:02d}: {calibrated_sentence}" for p in range(30)]


        document = "\n\n".join(paras)
        nodes = pipeline.chunk_markdown(document)
        assert len(nodes) >= 2

        for i in range(len(nodes) - 1):
            c1 = nodes[i].text
            c2 = nodes[i + 1].text
            toks_overlap = _compute_chunk_overlap(c1, c2)
            ratio = toks_overlap / 1024

            assert ratio == 0.0 or (0.10 <= ratio <= 0.15), (
                f"Sentence tokens ~{actual_tokens}: Transition {i}->{i+1} has invalid overlap {ratio:.2%} "
                f"({toks_overlap} tokens). Must be 0.0% or in [10%, 15%], but got an illegal ratio!"
            )

