"""Tests for maike.context.budget — ContextBudgetManager."""

from maike.constants import (
    CONTEXT_BUDGET_SAFETY_MARGIN,
    DEFAULT_LLM_MAX_TOKENS,
    MODEL_CONTEXT_LIMIT,
    context_limit_for_model,
)
from maike.context.budget import ContextBudgetManager, _strip_nested_descriptions


# ------------------------------------------------------------------ #
# context_limit_for_model
# ------------------------------------------------------------------ #


def test_context_limit_known_model():
    assert context_limit_for_model("gemini-2.5-flash") == 1_048_576


def test_context_limit_unknown_model_falls_back():
    assert context_limit_for_model("unknown-model-xyz") == MODEL_CONTEXT_LIMIT


# ------------------------------------------------------------------ #
# estimate_payload_tokens
# ------------------------------------------------------------------ #


def test_estimate_payload_tokens_empty():
    assert ContextBudgetManager.estimate_payload_tokens([], None, "") == 0


def test_estimate_payload_tokens_counts_messages():
    messages = [{"role": "user", "content": "Hello " * 100}]
    tokens = ContextBudgetManager.estimate_payload_tokens(messages)
    assert tokens > 50  # 500 chars / 4 ≈ 125 + overhead


def test_estimate_payload_tokens_includes_system_prompt():
    without_system = ContextBudgetManager.estimate_payload_tokens([], None, "")
    with_system = ContextBudgetManager.estimate_payload_tokens(
        [], None, "You are a coding agent. " * 50,
    )
    assert with_system > without_system


def test_estimate_payload_tokens_includes_tool_schemas():
    schemas = [
        {"name": "read_file", "description": "Reads a file from disk.", "input_schema": {"type": "object"}},
        {"name": "write_file", "description": "Writes content to a file.", "input_schema": {"type": "object"}},
    ]
    without_tools = ContextBudgetManager.estimate_payload_tokens([])
    with_tools = ContextBudgetManager.estimate_payload_tokens([], schemas)
    assert with_tools > without_tools


# ------------------------------------------------------------------ #
# effective_limit
# ------------------------------------------------------------------ #


def test_effective_limit_default_model():
    limit = ContextBudgetManager.effective_limit("unknown-model")
    expected = int(MODEL_CONTEXT_LIMIT * CONTEXT_BUDGET_SAFETY_MARGIN) - DEFAULT_LLM_MAX_TOKENS
    assert limit == expected


def test_effective_limit_gemini_larger():
    gemini_limit = ContextBudgetManager.effective_limit("gemini-2.5-flash")
    default_limit = ContextBudgetManager.effective_limit("unknown-model")
    assert gemini_limit > default_limit


# ------------------------------------------------------------------ #
# fits_budget
# ------------------------------------------------------------------ #


def test_fits_budget_small_payload():
    messages = [{"role": "user", "content": "hello"}]
    assert ContextBudgetManager.fits_budget(messages, model="gemini-2.5-flash")


def test_fits_budget_huge_payload_fails():
    # Create a payload that's definitely over 200K tokens.
    messages = [{"role": "user", "content": "x " * 500_000}]
    assert not ContextBudgetManager.fits_budget(messages, model="unknown-model")


def test_fits_budget_no_model_uses_default():
    messages = [{"role": "user", "content": "hello"}]
    assert ContextBudgetManager.fits_budget(messages)


# ------------------------------------------------------------------ #
# compress_to_fit — no compression needed
# ------------------------------------------------------------------ #


def test_compress_no_op_when_under_budget():
    messages = [{"role": "user", "content": "hello"}]
    compressed, schemas, levels = ContextBudgetManager.compress_to_fit(
        messages, model="gemini-2.5-flash",
    )
    assert compressed == messages
    assert schemas is None
    assert levels == []


# ------------------------------------------------------------------ #
# compress_to_fit — Level 1: strip tool descriptions
# ------------------------------------------------------------------ #


def test_strip_tool_descriptions():
    schemas = [
        {
            "name": "read_file",
            "description": "A very long description " * 100,
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "The file path to read."},
                },
            },
        },
    ]
    stripped = ContextBudgetManager._strip_tool_descriptions(schemas)
    assert len(stripped) == 1
    assert "description" not in stripped[0]
    assert stripped[0]["name"] == "read_file"
    # Nested descriptions should also be removed.
    props = stripped[0]["input_schema"]["properties"]
    assert "description" not in props["path"]


# ------------------------------------------------------------------ #
# compress_to_fit — Level 2: truncate artifact blocks
# ------------------------------------------------------------------ #


def test_truncate_artifact_blocks():
    body = "\n".join(f"line {i}: some content here" for i in range(200))
    content = (
        "## Artifact: spec.md\n"
        "Artifact kind: stage artifact\n"
        "Source stage: requirements\n"
        "Workspace path: none\n"
        "\n"
        + body
    )
    messages = [{"role": "user", "content": content}]
    result = ContextBudgetManager._truncate_artifact_blocks(messages, fraction=0.5)
    result_content = result[0]["content"]
    # Should be shorter than original.
    assert len(result_content) < len(content)
    # Header should be preserved.
    assert "## Artifact: spec.md" in result_content
    # Omission marker should be present.
    assert "lines omitted" in result_content


def test_truncate_artifact_blocks_leaves_short_artifacts():
    content = (
        "## Artifact: spec.md\n"
        "Short content."
    )
    messages = [{"role": "user", "content": content}]
    result = ContextBudgetManager._truncate_artifact_blocks(messages, fraction=0.5)
    assert result[0]["content"] == content


# ------------------------------------------------------------------ #
# compress_to_fit — Level 3: strip environment blocks
# ------------------------------------------------------------------ #


def test_strip_environment_blocks():
    parts = [
        "## Task\n\nBuild a thing",
        "## Environment\n\nLanguage: Python\nFramework: Flask\nPackage manager: pip\n"
        "Project structure:\n  src/\n    app.py\n    models.py",
    ]
    content = "\n\n---\n\n".join(parts)
    messages = [{"role": "user", "content": content}]
    result = ContextBudgetManager._strip_environment_blocks(messages)
    assert "## Environment" not in result[0]["content"]
    assert "## Task" in result[0]["content"]


def test_strip_environment_blocks_noop_when_absent():
    content = "## Task\n\nBuild a thing"
    messages = [{"role": "user", "content": content}]
    result = ContextBudgetManager._strip_environment_blocks(messages)
    assert result == messages


# ------------------------------------------------------------------ #
# compress_to_fit — full cascade
# ------------------------------------------------------------------ #


def test_compress_cascade_applies_levels_progressively():
    """Build a payload that's over budget for a small context window.

    We use a fake model name that maps to the default 200K limit and create
    content heavy enough to trigger compression levels.  tiktoken counts
    ~1 token per common word, so we need enough unique words to exceed the
    effective limit of ~172K tokens.
    """
    # Generate diverse content to avoid tiktoken's efficient encoding of
    # repeated text.  Each line has unique words → ~25-30 tokens per line.
    artifact_body = "\n".join(
        f"requirement_{i}: The system shall implement feature_{i} with "
        f"parameter_{i}_alpha={i*7} and parameter_{i}_beta={i*13} "
        f"ensuring compliance with standard_{i % 50} section_{i % 20} "
        f"using algorithm_{i % 30} variant_{i % 10} mode_{i % 5}"
        for i in range(8000)
    )
    content = (
        "## Artifact: spec.md\n"
        "Artifact kind: stage artifact\n"
        "Source stage: requirements\n"
        "Workspace path: none\n"
        "\n"
        + artifact_body
        + "\n\n---\n\n"
        + "## Environment\n\nLanguage: Python\n"
        + "Structure:\n" + "  file.py\n" * 200
        + "\n\n---\n\n"
        + "## Task\n\nBuild the app"
    )
    messages = [{"role": "user", "content": content}]
    tool_schemas = [
        {"name": f"tool_{i}", "description": "description " * 200, "input_schema": {"type": "object"}}
        for i in range(20)
    ]

    # Verify the payload is actually over budget before compression.
    estimated = ContextBudgetManager.estimate_payload_tokens(
        messages, tool_schemas, "You are an agent.",
    )
    limit = ContextBudgetManager.effective_limit("unknown-model")
    assert estimated > limit, (
        f"Test payload too small: {estimated:,} tokens vs {limit:,} limit. "
        f"Increase content size."
    )

    compressed_msgs, compressed_schemas, levels = ContextBudgetManager.compress_to_fit(
        messages,
        tool_schemas=tool_schemas,
        system_prompt="You are an agent.",
        model="unknown-model",
    )

    # At least one compression level was applied.
    assert len(levels) >= 1
    # The compressed content is shorter.
    assert len(compressed_msgs[0]["content"]) < len(messages[0]["content"])


# ------------------------------------------------------------------ #
# _strip_nested_descriptions
# ------------------------------------------------------------------ #


def test_strip_nested_descriptions():
    schema = {
        "type": "object",
        "description": "Top-level description",
        "properties": {
            "path": {
                "type": "string",
                "description": "A file path.",
            },
            "options": {
                "type": "object",
                "description": "Options for the tool.",
                "properties": {
                    "verbose": {
                        "type": "boolean",
                        "description": "Enable verbose output.",
                    },
                },
            },
        },
    }
    stripped = _strip_nested_descriptions(schema)
    assert "description" not in stripped
    assert "description" not in stripped["properties"]["path"]
    assert "description" not in stripped["properties"]["options"]
    assert "description" not in stripped["properties"]["options"]["properties"]["verbose"]
    # Non-description keys preserved.
    assert stripped["type"] == "object"
    assert stripped["properties"]["path"]["type"] == "string"
    assert stripped["properties"]["options"]["properties"]["verbose"]["type"] == "boolean"
