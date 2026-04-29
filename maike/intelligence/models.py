"""Data models for the code intelligence subsystem."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SymbolKind(str, Enum):
    """Kind of code symbol."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    CLASS = "class"
    METHOD = "method"
    ASYNC_METHOD = "async_method"
    VARIABLE = "variable"
    INTERFACE = "interface"
    TYPE_ALIAS = "type_alias"
    STRUCT = "struct"
    TRAIT = "trait"
    ENUM = "enum"
    IMPL = "impl"


class Symbol(BaseModel):
    """A named code symbol (function, class, method, variable, etc.)."""

    name: str
    qualified_name: str
    kind: SymbolKind
    file: str
    line: int
    end_line: int | None = None
    signature: str = ""
    scope: str | None = None
    docstring: str | None = None
    decorators: list[str] = Field(default_factory=list)


class ImportRef(BaseModel):
    """A single import statement in a source file."""

    source_file: str
    imported_name: str
    module_path: str
    is_from_import: bool = False
    alias: str | None = None
    line: int = 0


class FileEntry(BaseModel):
    """Index entry for a single source file."""

    path: str
    language: str
    content_hash: str
    symbols: list[Symbol] = Field(default_factory=list)
    imports: list[ImportRef] = Field(default_factory=list)
    line_count: int = 0


class CodeChunk(BaseModel):
    """A chunk of code suitable for embedding."""

    chunk_id: str
    file: str
    start_line: int
    end_line: int
    content: str
    symbol_name: str | None = None
    kind: str = "function"
    language: str = "python"
    parent_class: str | None = None
    has_overlap: bool = False
    overlap_lines: int = 0


class IndexStats(BaseModel):
    """Statistics about a built index."""

    total_files: int = 0
    total_symbols: int = 0
    total_imports: int = 0
    total_chunks: int = 0
    build_time_ms: int = 0
    languages: dict[str, int] = Field(default_factory=dict)
