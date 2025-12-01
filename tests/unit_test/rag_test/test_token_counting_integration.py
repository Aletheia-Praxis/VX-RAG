import pytest
from llama_index.core import Settings

from src.rag.libs.utils.llamaindex_integration import (
    get_global_token_counter,
    ensure_global_token_counter,
)
from src.rag.libs.utils.token_counter import LlamaIndexTokenCounter


@pytest.fixture(autouse=True)
def clear_callback_manager():
    # Ensure a clean global callback manager before each test
    Settings.callback_manager = None
    yield
    Settings.callback_manager = None


def test_get_and_ensure_global_token_counter():
    assert get_global_token_counter() is None
    handler = ensure_global_token_counter(model_name="gpt-3.5-turbo", verbose=False)
    assert handler is not None
    # After ensure, get_global should return same instance
    found = get_global_token_counter()
    assert found is handler


def test_llamaindex_token_counter_uses_global_handler():
    # Create global handler
    global_handler = ensure_global_token_counter(model_name="gpt-3.5-turbo", verbose=False)

    counter = LlamaIndexTokenCounter(model_name="gpt-3.5-turbo", verbose=False)
    assert counter.token_counter is global_handler
    assert hasattr(counter.token_counter, 'total_embedding_token_count')


def test_ensure_global_token_counter_idempotent():
    # Ensure global token counter registers only once
    handler1 = ensure_global_token_counter(model_name="gpt-3.5-turbo", verbose=False)
    # After the first ensure, Settings.callback_manager must exist
    assert handler1 is not None
    assert Settings.callback_manager is not None
    initial_len = len(Settings.callback_manager.handlers)
    handler2 = ensure_global_token_counter(model_name="gpt-3.5-turbo", verbose=False)
    assert handler1 is handler2
    assert len(Settings.callback_manager.handlers) == initial_len


def test_tokenbudgeter_uses_global_counter():
    from src.rag.libs.utils.token_utils import TokenBudgeter

    # Ensure global registered
    global_handler = ensure_global_token_counter(model_name="gpt-3.5-turbo", verbose=False)
    tb = TokenBudgeter(model_name="gpt-3.5-turbo", verbose=False)
    assert hasattr(tb, 'get_stats')
    stats = tb.get_stats()
    assert stats['total_embedding_tokens'] == global_handler.total_embedding_token_count


def test_budget_and_assemble_uses_global_handler():
    """Ensure budget_and_assemble uses the global TokenCountingHandler via TokenBudgeter."""
    from src.rag.libs.utils.token_utils import budget_and_assemble

    # Ensure global registered
    global_handler = ensure_global_token_counter(model_name="gpt-3.5-turbo", verbose=False)

    # Prepare a couple of sample documents with text and scores
    docs = [
        {'id': '1', 'text': 'Hello world', 'score': 0.9, 'metadata': {}},
        {'id': '2', 'text': 'Another document', 'score': 0.8, 'metadata': {}},
    ]

    payload = budget_and_assemble(results=docs, token_budget=2048, model_name="gpt-3.5-turbo")

    # Confirm payload structure and that the global handler is still the same instance
    assert isinstance(payload, dict)
    assert 'context' in payload
    assert get_global_token_counter() is global_handler
