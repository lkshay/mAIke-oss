"""Hot context assembler — pre-computes task-relevant code context for agents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from maike.intelligence.models import Symbol

if TYPE_CHECKING:
    from maike.intelligence.code_index import CodeIndex

# Common stopwords to filter from task keywords.
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "and", "or", "but", "not", "no", "nor", "so", "yet",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over",
    "that", "this", "these", "those", "it", "its", "my", "your",
    "we", "they", "he", "she", "you", "i", "me", "us", "them",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "only", "own", "same", "than", "too", "very",
    "just", "also", "then", "now", "here", "there",
    "create", "make", "build", "add", "implement", "write", "update",
    "fix", "change", "modify", "use", "run", "test", "check",
    "file", "code", "project", "app", "application", "program",
})

_IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z_]\w*")
_MAX_SYMBOLS = 10
_MAX_RELATED_FILES = 15


@dataclass
class HotContext:
    """Pre-computed code context for an agent's task."""

    relevant_symbols: list[Symbol] = field(default_factory=list)
    related_files: list[str] = field(default_factory=list)
    focused_map: str = ""
    total_tokens: int = 0


class HotContextAssembler:
    """Pre-computes the most relevant code context for an agent's task."""

    def __init__(self, code_index: CodeIndex) -> None:
        self._index = code_index

    def assemble(
        self,
        task: str,
        role: str,
        *,
        files_in_scope: list[str] | None = None,
    ) -> HotContext:
        """Assemble hot context for a task and role."""
        if not self._index.is_built:
            return HotContext()

        keywords = self._extract_keywords(task)
        if not keywords:
            return HotContext()

        # Find relevant symbols.
        scored_symbols = self._find_and_score_symbols(keywords)
        top_symbols = [sym for sym, _ in scored_symbols[:_MAX_SYMBOLS]]

        # Expand via import graph.
        related_files: set[str] = set()
        for sym in top_symbols:
            related = self._index.find_related_files(sym.file, depth=1)
            related_files.update(related.keys())
        # Add files_in_scope if provided.
        if files_in_scope:
            related_files.update(files_in_scope)
        # Remove files that already have symbols in the top list.
        symbol_files = {sym.file for sym in top_symbols}
        related_files -= symbol_files
        related_list = sorted(related_files)[:_MAX_RELATED_FILES]

        # Build focused map around the top symbol's file.
        focused_map = ""
        if top_symbols:
            focused_map = self._index.smart_repo_map(focus_file=top_symbols[0].file)

        return HotContext(
            relevant_symbols=top_symbols,
            related_files=related_list,
            focused_map=focused_map,
        )

    def format_hot_context(self, ctx: HotContext) -> str:
        """Format the hot context as a markdown context block."""
        if not ctx.relevant_symbols and not ctx.related_files:
            return ""

        parts: list[str] = ["## Code Intelligence (pre-computed for your task)\n"]

        if ctx.relevant_symbols:
            parts.append("### Relevant Symbols")
            for sym in ctx.relevant_symbols:
                scope_str = f" (in {sym.scope})" if sym.scope else ""
                parts.append(
                    f"- `{sym.qualified_name}` ({sym.kind.value}{scope_str}) "
                    f"in `{sym.file}:{sym.line}`"
                )
                if sym.signature:
                    parts.append(f"  `{sym.signature}`")

        if ctx.related_files:
            parts.append("\n### Related Files (by import graph)")
            for f in ctx.related_files:
                parts.append(f"- `{f}`")

        parts.append(
            "\n### Deeper Exploration\n"
            "Use `find_symbol`, `find_references`, `find_related_files`, "
            "`semantic_code_search`, or `grep_codebase` for details not shown above."
        )

        return "\n".join(parts)

    def _extract_keywords(self, task: str) -> list[str]:
        """Extract meaningful keywords from a task description."""
        # Find all identifier-like tokens.
        tokens = _IDENTIFIER_PATTERN.findall(task)
        # Filter stopwords and very short tokens.
        keywords: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            lower = token.lower()
            if lower in _STOPWORDS or len(token) < 3:
                continue
            if lower in seen:
                continue
            seen.add(lower)
            keywords.append(token)
        return keywords

    def _find_and_score_symbols(
        self,
        keywords: list[str],
    ) -> list[tuple[Symbol, float]]:
        """Find symbols matching keywords and score them."""
        scored: dict[str, tuple[Symbol, float]] = {}
        for kw in keywords:
            for sym in self._index.find_symbol(kw):
                key = sym.qualified_name
                if key in scored:
                    _, existing_score = scored[key]
                    scored[key] = (sym, existing_score + self._score_match(sym, kw))
                else:
                    scored[key] = (sym, self._score_match(sym, kw))
        # Sort by score descending.
        results = sorted(scored.values(), key=lambda t: -t[1])
        return results

    def _score_match(self, sym: Symbol, keyword: str) -> float:
        """Score how well a symbol matches a keyword."""
        kw_lower = keyword.lower()
        name_lower = sym.name.lower()
        if name_lower == kw_lower:
            return 10.0  # Exact name match.
        if name_lower.startswith(kw_lower):
            return 5.0  # Prefix match.
        if kw_lower in (sym.scope or "").lower():
            return 3.0  # Keyword in scope.
        if kw_lower in sym.file.lower():
            return 2.0  # Keyword in file path.
        return 1.0  # Substring match (already filtered by find_symbol).
