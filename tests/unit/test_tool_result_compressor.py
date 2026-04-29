"""Tests for maike.context.compressor — ToolResultCompressor."""

from maike.context.compressor import (
    COMPRESSION_THRESHOLD,
    ToolResultCompressor,
    _ast_compress_python,
    _is_algorithmic_docstring,
    _looks_like_python,
    _signature_compress,
    compress_tool_result,
)


def test_small_output_passes_through():
    result = compress_tool_result("read_file", "short output")
    assert result == "short output"


def test_threshold_boundary():
    just_under = "x" * COMPRESSION_THRESHOLD
    assert compress_tool_result("read_file", just_under) == just_under


# ------------------------------------------------------------------ #
# read_file compression
# ------------------------------------------------------------------ #


def test_read_file_compression_keeps_head_and_tail():
    lines = [f"line {i}: content here" for i in range(300)]
    output = "\n".join(lines)
    result = compress_tool_result("read_file", output)
    assert len(result) < len(output)
    assert "line 0:" in result
    assert "lines omitted" in result


def test_read_file_short_file_uses_head_tail():
    lines = [f"line {i}" for i in range(100)]
    output = "\n".join(lines)
    # 100 lines < 120 threshold, but > COMPRESSION_THRESHOLD chars...
    # Each line is ~7 chars, total ~700 chars — under threshold. So no compression.
    assert compress_tool_result("read_file", output) == output


# ------------------------------------------------------------------ #
# grep compression
# ------------------------------------------------------------------ #


def test_grep_compression_groups_by_file():
    lines = []
    for i in range(200):
        lines.append(f"src/file_{i % 10}.py:{i}:match content {i} with extra padding text here")
    output = "\n".join(lines)
    assert len(output) > COMPRESSION_THRESHOLD
    result = compress_tool_result("grep_codebase", output)
    assert "files" in result
    assert "matches" in result
    assert len(result) < len(output)


def test_grep_limits_matches_per_file():
    # 20 matches from same file — should keep only max_per_file (8).
    lines = [f"src/big.py:{i}:match {i}" for i in range(20)]
    output = "\n".join(lines)
    compressor = ToolResultCompressor()
    result = compressor._compress_grep(output, 8000)
    # Count lines with "src/big.py"
    big_lines = [line for line in result.split("\n") if "src/big.py:" in line]
    assert len(big_lines) <= 8


# ------------------------------------------------------------------ #
# bash/test compression
# ------------------------------------------------------------------ #


def test_bash_compression_keeps_head_and_tail():
    lines = [f"output line {i}: some extra padding content here" for i in range(300)]
    output = "\n".join(lines)
    assert len(output) > COMPRESSION_THRESHOLD
    result = compress_tool_result("execute_bash", output)
    assert len(result) < len(output)
    assert "output line 0:" in result
    assert "lines omitted" in result


def test_bash_preserves_error_heavy_output():
    """If >30% of lines contain error keywords, keep full output (up to max)."""
    lines = []
    for i in range(100):
        if i % 2 == 0:
            lines.append(f"ERROR: something failed at step {i}")
        else:
            lines.append(f"normal output {i}")
    output = "\n".join(lines)
    result = compress_tool_result("execute_bash", output)
    # Error-heavy output is kept in full (up to max_chars).
    assert "ERROR: something failed at step 0" in result
    assert "ERROR: something failed at step 98" in result


# ------------------------------------------------------------------ #
# Unknown tool
# ------------------------------------------------------------------ #


def test_unknown_tool_uses_head_tail():
    output = "data " * 2000  # ~10K chars
    result = compress_tool_result("some_unknown_tool", output)
    assert len(result) < len(output)
    assert "chars omitted" in result


# ------------------------------------------------------------------ #
# Module-level shortcut
# ------------------------------------------------------------------ #


def test_module_shortcut_matches_class():
    compressor = ToolResultCompressor()
    output = "x\n" * 5000
    assert compress_tool_result("read_file", output) == compressor.compress("read_file", output)


# ------------------------------------------------------------------ #
# AST-aware Python compression
# ------------------------------------------------------------------ #


def test_looks_like_python_positive():
    code = """
import os
from pathlib import Path

def main():
    print("hello")

class Foo:
    pass
"""
    assert _looks_like_python(code) is True


def test_looks_like_python_negative():
    text = "This is just a plain English paragraph about nothing in particular."
    assert _looks_like_python(text) is False


def test_ast_compress_python_preserves_imports():
    code = """
import os
import sys
from pathlib import Path
from typing import Any

def helper(x: int) -> str:
    result = str(x)
    return result

class MyClass:
    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}"
"""
    result = _ast_compress_python(code)
    assert result is not None
    assert "import os" in result
    assert "import sys" in result
    assert "from pathlib import Path" in result
    assert "from typing import Any" in result


def test_ast_compress_python_preserves_signatures():
    code = """
import os

def calculate(a: int, b: int) -> int:
    intermediate = a * 2
    result = intermediate + b
    return result

async def fetch_data(url: str) -> dict:
    response = await http.get(url)
    data = response.json()
    return data
"""
    result = _ast_compress_python(code)
    assert result is not None
    assert "def calculate(a: int, b: int) -> int:" in result
    assert "async def fetch_data(url: str) -> dict:" in result


def test_ast_compress_python_drops_bodies():
    code = """
import os

def long_function(data: list) -> list:
    result = []
    for item in data:
        processed = item * 2
        if processed > 10:
            result.append(processed)
        else:
            result.append(0)
    filtered = [x for x in result if x > 0]
    sorted_result = sorted(filtered)
    return sorted_result
"""
    result = _ast_compress_python(code)
    assert result is not None
    # The body lines should NOT appear.
    assert "result = []" not in result
    assert "for item in data:" not in result
    assert "processed = item * 2" not in result
    # But the signature and ellipsis should.
    assert "def long_function(data: list) -> list:" in result
    assert "..." in result


def test_ast_compress_python_preserves_class_structure():
    code = """
from dataclasses import dataclass

@dataclass
class Config:
    \"\"\"Application configuration.\"\"\"
    name: str
    debug: bool = False

class Repository:
    \"\"\"Database repository.\"\"\"

    def __init__(self, db):
        self.db = db
        self.cache = {}

    def get(self, key: str):
        if key in self.cache:
            return self.cache[key]
        result = self.db.query(key)
        self.cache[key] = result
        return result

    def delete(self, key: str) -> bool:
        if key in self.cache:
            del self.cache[key]
        return self.db.delete(key)
"""
    result = _ast_compress_python(code)
    assert result is not None
    assert "class Config:" in result or "class Config" in result
    assert "class Repository:" in result
    assert "def __init__" in result
    assert "def get" in result
    assert "def delete" in result
    # Body should be dropped.
    assert "self.cache = {}" not in result
    assert "self.db.query" not in result


def test_ast_compress_python_preserves_docstrings():
    code = """
def well_documented(x: int) -> int:
    \"\"\"Convert x to a processed value.\"\"\"
    return x * 2 + 1
"""
    result = _ast_compress_python(code)
    assert result is not None
    assert "Convert x to a processed value" in result


def test_ast_compress_python_shows_compression_ratio():
    lines = ["import os", ""] + [
        f"def func_{i}(x):\n    return x + {i}\n" for i in range(50)
    ]
    code = "\n".join(lines)
    result = _ast_compress_python(code)
    assert result is not None
    assert "AST-compressed" in result
    assert "lines ->" in result
    assert "bodies omitted" in result


def test_ast_compress_python_handles_syntax_error():
    bad_code = "def broken(\n    this is not valid python"
    result = _ast_compress_python(bad_code)
    assert result is None


def test_ast_compress_python_preserves_decorators():
    code = """
import functools

@functools.lru_cache(maxsize=128)
def cached_compute(n: int) -> int:
    if n <= 1:
        return n
    return cached_compute(n - 1) + cached_compute(n - 2)
"""
    result = _ast_compress_python(code)
    assert result is not None
    assert "@functools.lru_cache" in result


def test_ast_compress_integrated_with_read_file():
    """AST compression should trigger for large Python files via read_file."""
    # Build a Python file with 150+ lines.
    lines = [
        "import os",
        "import sys",
        "from pathlib import Path",
        "",
    ]
    for i in range(40):
        lines.extend([
            f"def function_{i}(arg1: int, arg2: str) -> bool:",
            f'    """Process item {i}."""',
            "    result = arg1 + len(arg2)",
            f"    if result > {i}:",
            "        return True",
            "    return False",
            "",
        ])
    code = "\n".join(lines)
    assert len(code) > COMPRESSION_THRESHOLD

    result = compress_tool_result("read_file", code)
    assert "AST-compressed" in result
    # Imports preserved.
    assert "import os" in result
    # Signatures preserved.
    assert "def function_0" in result
    assert "def function_39" in result
    # Bodies dropped.
    assert "result = arg1 + len(arg2)" not in result


# ------------------------------------------------------------------ #
# Signature compression (JS/Go/Rust)
# ------------------------------------------------------------------ #


def test_signature_compress_javascript():
    code = """
import { useState } from 'react';
import axios from 'axios';

export default function App() {
    const [data, setData] = useState(null);
    return <div>{data}</div>;
}

export async function fetchUsers(limit) {
    const response = await axios.get('/api/users', { params: { limit } });
    return response.data;
}

export class UserService {
    constructor(api) {
        this.api = api;
    }

    async getUser(id) {
        return this.api.get(`/users/${id}`);
    }
}
"""
    result = _signature_compress(code)
    assert result is not None
    assert "Signature-compressed" in result
    assert "import" in result.lower()


def test_signature_compress_go():
    code = """
package main

import (
    "fmt"
    "net/http"
)

func main() {
    http.HandleFunc("/", handler)
    http.ListenAndServe(":8080", nil)
}

func handler(w http.ResponseWriter, r *http.Request) {
    fmt.Fprintf(w, "Hello, World!")
}

type Server struct {
    port int
    host string
}

func (s *Server) Start() error {
    return http.ListenAndServe(fmt.Sprintf("%s:%d", s.host, s.port), nil)
}
"""
    result = _signature_compress(code)
    assert result is not None
    assert "Signature-compressed" in result


def test_signature_compress_returns_none_for_unknown():
    text = "This is just plain text with no code signatures."
    result = _signature_compress(text)
    assert result is None


# ------------------------------------------------------------------ #
# Algorithmic docstring preservation
# ------------------------------------------------------------------ #


def test_is_algorithmic_docstring_keywords():
    """Verify keyword detection function recognises algorithmic terms."""
    assert _is_algorithmic_docstring("Uses recursion with a base case of n=0")
    assert _is_algorithmic_docstring("O(n log n) complexity due to merge sort")
    assert _is_algorithmic_docstring("Memoization avoids recomputation")
    assert _is_algorithmic_docstring("Backtracking search over the solution space")
    assert _is_algorithmic_docstring("Loop invariant: sum equals prefix total")
    assert _is_algorithmic_docstring("Stack overflow risk at depth > 1000")
    assert _is_algorithmic_docstring("Converge to fixpoint via iteration")
    # Negative cases
    assert not _is_algorithmic_docstring("Return the processed value.")
    assert not _is_algorithmic_docstring("Helper to format user names.")
    assert not _is_algorithmic_docstring("")


def test_algorithmic_docstring_preserved():
    """A function with algorithmic docstring keeps up to 3 lines."""
    code = '''
import os

def solve(n: int) -> int:
    """Compute fibonacci using recursion.
    Base case: n <= 1 returns n.
    Recursive step: solve(n-1) + solve(n-2).
    Time complexity: O(2^n) without memoization.
    """
    if n <= 1:
        return n
    return solve(n - 1) + solve(n - 2)
'''
    result = _ast_compress_python(code)
    assert result is not None
    # First 3 lines of docstring should be preserved.
    assert "Compute fibonacci using recursion" in result
    assert "Base case" in result
    assert "Recursive step" in result
    # Lines beyond the first 3 should be summarised, not included verbatim.
    assert "Time complexity: O(2^n)" not in result
    assert "more lines]" in result


def test_normal_docstring_truncated():
    """A function with a plain docstring keeps only the first line."""
    code = '''
import os

def format_name(first: str, last: str) -> str:
    """Format a full name from components.

    Concatenates first and last name with a space separator.
    Strips leading and trailing whitespace from each part.
    Returns the combined string.
    """
    return f"{first.strip()} {last.strip()}"
'''
    result = _ast_compress_python(code)
    assert result is not None
    # Only the first line should appear.
    assert "Format a full name from components" in result
    # Subsequent lines should NOT appear.
    assert "Concatenates first and last" not in result
    assert "Strips leading" not in result
    assert "more lines]" not in result
