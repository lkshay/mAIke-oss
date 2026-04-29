"""Tests for maike.context.tokenizer — tiktoken-based token counting."""

import pytest

from maike.context.tokenizer import (
    TokenCounter,
    _HAS_TIKTOKEN,
    count_message_tokens,
    count_system_prompt_tokens,
    count_tokens,
    count_tool_schema_tokens,
    estimate_payload_tokens,
)


# ------------------------------------------------------------------ #
# Basic counting
# ------------------------------------------------------------------ #


@pytest.mark.skipif(not _HAS_TIKTOKEN, reason="tiktoken not installed")
def test_tiktoken_is_available():
    """tiktoken should be installed and functional."""
    assert _HAS_TIKTOKEN is True


def test_count_tokens_empty():
    assert count_tokens("") == 0


def test_count_tokens_single_word():
    tokens = count_tokens("hello")
    assert tokens == 1  # "hello" is a single token in cl100k_base


def test_count_tokens_english_sentence():
    tokens = count_tokens("The quick brown fox jumps over the lazy dog.")
    # BPE tokenization: ~10 tokens; char heuristic may yield ~13.
    assert 8 <= tokens <= 14


def test_count_tokens_code_is_more_than_chars_div_4():
    """Code tokens average ~2.5-3.5 chars/token, NOT 4.

    This verifies the key insight: the old ``chars // 4`` heuristic
    underestimates code tokens by 15-40%.
    """
    code = """\
def calculate_statistics(data: list[float]) -> dict[str, float]:
    if not data:
        raise ValueError("Cannot calculate statistics for empty data")
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return {"mean": mean, "variance": variance, "std_dev": variance ** 0.5}
"""
    actual_tokens = count_tokens(code)
    naive_estimate = len(code) // 4

    # The actual token count should be HIGHER than chars//4 for code.
    assert actual_tokens > naive_estimate, (
        f"tiktoken counted {actual_tokens} tokens, naive estimate was {naive_estimate}. "
        f"Code should produce more tokens than chars//4."
    )


def test_count_tokens_natural_text_close_to_4_ratio():
    """For plain English, chars//4 is roughly correct — but code differs."""
    text = "This is a simple English sentence with common words and nothing unusual."
    actual = count_tokens(text)
    naive = len(text) // 4
    # For natural text, the ratio is closer to 4, so the difference is small.
    assert abs(actual - naive) < naive * 0.5  # within 50%


def test_count_tokens_very_long_bypasses_cache():
    """Strings over 100K chars bypass the LRU cache."""
    long_text = "word " * 30_000  # ~150K chars
    tokens = count_tokens(long_text)
    assert tokens > 20_000


# ------------------------------------------------------------------ #
# Message counting
# ------------------------------------------------------------------ #


def test_count_message_tokens_empty():
    assert count_message_tokens([]) == 0


def test_count_message_tokens_single_message():
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    tokens = count_message_tokens(messages)
    # Content tokens + 4 overhead per message.
    assert tokens > 4


def test_count_message_tokens_includes_overhead():
    """Each message adds ~4 tokens of structural overhead."""
    one_msg = [{"role": "user", "content": "hi"}]
    two_msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    diff = count_message_tokens(two_msgs) - count_message_tokens(one_msg)
    # Second message adds its content tokens + 4 overhead.
    assert diff >= 4


def test_count_message_tokens_handles_tool_use():
    messages = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_123", "name": "read_file", "input": {"path": "src/main.py"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "toolu_123", "content": "import sys\nprint('hello')"},
        ]},
    ]
    tokens = count_message_tokens(messages)
    assert tokens > 8  # at least overhead for 2 messages


# ------------------------------------------------------------------ #
# System prompt counting
# ------------------------------------------------------------------ #


def test_system_prompt_empty():
    assert count_system_prompt_tokens("") == 0


def test_system_prompt_counts_tokens_plus_overhead():
    tokens = count_system_prompt_tokens("You are a helpful coding assistant.")
    assert tokens > 4  # content + 4 overhead


# ------------------------------------------------------------------ #
# Tool schema counting
# ------------------------------------------------------------------ #


def test_tool_schema_tokens_empty():
    assert count_tool_schema_tokens(None) == 0
    assert count_tool_schema_tokens([]) == 0


def test_tool_schema_tokens_counts_schemas():
    schemas = [
        {"name": "read_file", "description": "Reads a file from disk.", "input_schema": {"type": "object"}},
    ]
    tokens = count_tool_schema_tokens(schemas)
    assert tokens > 5


# ------------------------------------------------------------------ #
# Full payload estimation
# ------------------------------------------------------------------ #


def test_estimate_payload_combines_all():
    messages = [{"role": "user", "content": "Hello"}]
    schemas = [{"name": "tool", "description": "A tool.", "input_schema": {"type": "object"}}]
    system = "You are an agent."

    total = estimate_payload_tokens(messages, schemas, system)
    msg_only = count_message_tokens(messages)
    sys_only = count_system_prompt_tokens(system)
    tool_only = count_tool_schema_tokens(schemas)

    assert total == msg_only + sys_only + tool_only


# ------------------------------------------------------------------ #
# TokenCounter stateful wrapper
# ------------------------------------------------------------------ #


def test_token_counter_tracks_totals():
    counter = TokenCounter()
    assert counter.total_counted == 0
    assert counter.call_count == 0

    t1 = counter.count("hello world")
    assert counter.total_counted == t1
    assert counter.call_count == 1

    t2 = counter.count("goodbye world")
    assert counter.total_counted == t1 + t2
    assert counter.call_count == 2


@pytest.mark.skipif(not _HAS_TIKTOKEN, reason="tiktoken not installed")
def test_token_counter_has_tiktoken():
    counter = TokenCounter()
    assert counter.has_tiktoken is True


def test_token_counter_cache_info():
    counter = TokenCounter()
    counter.count("cache test string")
    info = counter.cache_info
    assert "hits" in info
    assert "misses" in info
    assert "maxsize" in info
    assert "currsize" in info


def test_token_counter_count_messages():
    counter = TokenCounter()
    messages = [{"role": "user", "content": "Test message"}]
    tokens = counter.count_messages(messages)
    assert tokens > 0
    assert counter.total_counted == tokens


def test_token_counter_count_payload():
    counter = TokenCounter()
    messages = [{"role": "user", "content": "Test"}]
    tokens = counter.count_payload(messages, system_prompt="System")
    assert tokens > 0
    assert counter.total_counted == tokens


# ------------------------------------------------------------------ #
# Accuracy regression: code vs natural text
# ------------------------------------------------------------------ #


def test_accuracy_python_code():
    """Verify tiktoken gives meaningfully different counts than chars//4 for code.

    The chars//4 heuristic is inaccurate in both directions depending on
    content.  For code with many short symbols and operators, chars//4
    *underestimates*.  For code with long identifiers, it *overestimates*.
    The key value of tiktoken is *consistency and accuracy*.
    """
    # Code heavy with symbols, operators, special chars — tiktoken counts more.
    symbol_heavy = """
x = {k: v for k, v in d.items() if k != "x"}
y = [i**2 + j**2 for i, j in zip(a, b)]
z = (lambda f: (lambda x: f(lambda *a: x(x)(*a)))(lambda x: f(lambda *a: x(x)(*a))))
result = {**base, **{f"k_{i}": i * 2 for i in range(100)}}
assert all(isinstance(v, (int, float, str)) for v in values.values())
""" * 5
    tiktoken_count = count_tokens(symbol_heavy)
    naive_count = len(symbol_heavy) // 4
    # With symbol-heavy code, tiktoken should differ noticeably from chars//4.
    assert tiktoken_count != naive_count, "tiktoken should differ from chars//4 for code"

    # The important thing: tiktoken is deterministic and based on actual BPE.
    assert count_tokens(symbol_heavy) == tiktoken_count  # stable result
