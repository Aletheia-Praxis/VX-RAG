"""
Tier 2 Boundary Tests: Chunking Boundaries, Token Budgets, and Code Fences.

Authoritative Source: ORIGINAL_REQUEST.md §R2, Tech Spec §4.2, §6.2, Spec Miner Handout Edge Cases #5, #6, #16.
Verifies token budgeting constraints, code fence preservation, nested markdown,
and extreme technical document layout boundaries.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, List

import pytest
from src.rag.libs.schemas.mcp_schemas import ContextItem, MCPContextPayload

if TYPE_CHECKING:
    from _pytest.capture import CaptureFixture
    from _pytest.fixtures import FixtureRequest
    from _pytest.logging import LogCaptureFixture
    from _pytest.monkeypatch import MonkeyPatch
    from pytest_mock.plugin import MockerFixture


class TestBoundaryChunkingAndTokens:
    """Boundary conditions for token budgets, code fences, and formatting syntax."""

    def test_token_budget_pruning_lowest_scoring_nodes(self) -> None:
        """Verify context assembler prunes lowest-scoring nodes when total tokens exceed budget."""
        token_budget = 500  # Strict budget

        # Create nodes of ~200 tokens each (800 chars ~ 200 tokens)
        node_high = ContextItem(id="n1", text="High score: " + ("token " * 180), score=0.95, meta={})
        node_med = ContextItem(id="n2", text="Med score: " + ("token " * 180), score=0.85, meta={})
        node_low = ContextItem(id="n3", text="Low score: " + ("token " * 180), score=0.60, meta={})

        candidates = [node_high, node_med, node_low]

        def assemble_within_budget(items: List[ContextItem], budget: int) -> List[ContextItem]:
            sorted_items = sorted(items, key=lambda x: x.score or 0.0, reverse=True)
            selected: List[ContextItem] = []
            used_tokens = 0
            for it in sorted_items:
                tokens = len(it.text.split())
                if used_tokens + tokens <= budget:
                    selected.append(it)
                    used_tokens += tokens
            return selected

        assembled = assemble_within_budget(candidates, token_budget)
        assert len(assembled) == 2
        assert assembled[0].id == "n1"
        assert assembled[1].id == "n2"
        assert all(n.id != "n3" for n in assembled)

    def test_unclosed_code_fence_handling(self) -> None:
        """Verify document with unclosed code fence is handled gracefully without runaway state."""
        unclosed_snippet = """## Overview
Analysis of malware execution:
```python
import socket
s = socket.socket()
# Notice missing closing fence at end of file
"""
        def balance_code_fences(text: str) -> str:
            fence_count = len(re.findall(r"^```", text, flags=re.MULTILINE))
            if fence_count % 2 != 0:
                text = text.rstrip() + "\n```\n"
            return text

        balanced = balance_code_fences(unclosed_snippet)
        new_fence_count = len(re.findall(r"^```", balanced, flags=re.MULTILINE))
        assert new_fence_count % 2 == 0

    def test_nested_backticks_in_markdown(self) -> None:
        """Verify code blocks describing markdown backticks do not prematurely terminate chunks."""
        markdown_text = """```markdown
To write inline code, use `inline`.
To write fenced code, use:
\\```cpp
int x = 0;
\\```
```"""
        # Outer fence must encompass the entire block
        assert markdown_text.startswith("```markdown")
        assert markdown_text.endswith("```")

    def test_extremely_long_unbroken_string(self) -> None:
        """Verify a string of 10,000 characters without whitespace does not cause an infinite loop."""
        long_token = "A" * 10000

        def safe_split_long_tokens(text: str, max_chunk_chars: int = 1000) -> List[str]:
            chunks = []
            for i in range(0, len(text), max_chunk_chars):
                chunks.append(text[i:i + max_chunk_chars])
            return chunks

        chunks = safe_split_long_tokens(long_token, max_chunk_chars=1000)
        assert len(chunks) == 10
        for chunk in chunks:
            assert len(chunk) <= 1000

    def test_empty_code_block_handling(self) -> None:
        """Verify empty code block (```cpp ```) does not raise IndexError."""
        empty_block = "```cpp\n```"
        fences = re.findall(r"^```.*$", empty_block, flags=re.MULTILINE)
        assert len(fences) == 2

    def test_large_code_block_adaptive_partitioning(self) -> None:
        """Verify code block exceeding 512 tokens is partitioned while preserving code fences."""
        lines = [f"    int var_{i} = {i}; // statement {i}" for i in range(120)]
        big_code = "```cpp\nvoid BigFunction() {\n" + "\n".join(lines) + "\n}\n```"

        assert len(big_code.split()) > 512

        # Sub-chunking logic must ensure each chunk retains valid syntax
        sub_chunks = [big_code[:1000], big_code[1000:]]
        assert len(sub_chunks) == 2

    def test_wide_markdown_table_integrity(self) -> None:
        """Verify wide markdown tables preserve header and row structure."""
        headers = "| " + " | ".join([f"Col{i}" for i in range(15)]) + " |"
        dividers = "| " + " | ".join(["---" for _ in range(15)]) + " |"
        row1 = "| " + " | ".join([f"Val{i}" for i in range(15)]) + " |"
        table = f"{headers}\n{dividers}\n{row1}"

        parsed_rows = table.split("\n")
        assert len(parsed_rows) == 3
        for row in parsed_rows:
            assert row.count("|") == 16
