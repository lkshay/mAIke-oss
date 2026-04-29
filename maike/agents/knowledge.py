"""Knowledge module loader — progressive disclosure of domain-specific guidance.

Modules live as `.md` files in ``prompts/knowledge/``. Each has YAML frontmatter
with ``name``, ``description``, ``triggers``, and ``auto_inject``. The system
prompt always gets a compact catalog (one line per module). Full module content
is injected into the initial user message only when the task matches triggers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_KNOWLEDGE_DIR = Path(__file__).parent / "prompts" / "knowledge"

# Reuse the same identifier pattern and stopwords as hot_context.
_IDENTIFIER_PATTERN = re.compile(r"[a-zA-Z_]\w*")
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "and", "or", "but", "not", "no", "nor", "so", "yet",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "about", "into", "through", "during", "before", "after",
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


@dataclass(frozen=True)
class KnowledgeModule:
    name: str
    description: str
    triggers: list[str]
    auto_inject: bool
    content: str


class KnowledgeLoader:
    """Load, catalog, and select knowledge modules."""

    def __init__(self, knowledge_dir: Path | None = None) -> None:
        self._dir = knowledge_dir or _KNOWLEDGE_DIR

    def load_all(self) -> list[KnowledgeModule]:
        """Load all ``.md`` files from the knowledge directory.

        Also scans immediate subdirectories for ``SKILL.md`` files so that
        directory-based skills (e.g. ``test-methodology/SKILL.md``) are
        discovered alongside flat ``.md`` modules.

        When both a flat file and a directory-based ``SKILL.md`` share the
        same ``name``, the directory version wins (later overwrites earlier).
        """
        if not self._dir.is_dir():
            return []
        by_name: dict[str, KnowledgeModule] = {}
        for path in sorted(self._dir.glob("*.md")):
            module = self._parse_module(path)
            if module is not None:
                by_name[module.name] = module

        # Directory-based skills (subdir/SKILL.md) — override flat files.
        for subdir in sorted(self._dir.iterdir()):
            if not subdir.is_dir():
                continue
            skill_md = subdir / "SKILL.md"
            if skill_md.is_file():
                module = self._parse_module(skill_md)
                if module is not None:
                    by_name[module.name] = module

        return sorted(by_name.values(), key=lambda m: m.name)

    def build_catalog(self, modules: list[KnowledgeModule]) -> str:
        """Build a compact catalog for the system prompt (one line per module)."""
        if not modules:
            return "(none available)"
        return "\n".join(
            f"- **{m.name}**: {m.description}" for m in modules
        )

    def select_for_task(
        self,
        task: str,
        modules: list[KnowledgeModule],
    ) -> list[KnowledgeModule]:
        """Select modules whose triggers match task keywords."""
        task_lower = task.lower()
        task_keywords = self._extract_keywords(task)
        selected: list[KnowledgeModule] = []
        for module in modules:
            if module.auto_inject:
                selected.append(module)
                continue
            for trigger in module.triggers:
                trigger_lower = trigger.lower()
                # Direct substring match in the task text.
                if trigger_lower in task_lower:
                    selected.append(module)
                    break
                # Keyword-level match.
                if any(trigger_lower == kw for kw in task_keywords):
                    selected.append(module)
                    break
        return selected

    @staticmethod
    def _extract_keywords(task: str) -> list[str]:
        """Extract meaningful keywords from a task description."""
        tokens = _IDENTIFIER_PATTERN.findall(task)
        keywords: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            lower = token.lower()
            if lower in _STOPWORDS or len(token) < 3:
                continue
            if lower in seen:
                continue
            seen.add(lower)
            keywords.append(lower)
        return keywords

    @staticmethod
    def _parse_module(path: Path) -> KnowledgeModule | None:
        """Parse a ``.md`` file with YAML frontmatter into a KnowledgeModule.

        Supports an optional ``metadata`` map whose ``triggers`` (CSV string)
        and ``auto_inject`` (``"true"``/``"false"``) fields override the
        top-level equivalents for backward compatibility.
        """
        from maike.agents.skill import parse_md_frontmatter, _csv_to_list

        result = parse_md_frontmatter(path)
        if result is None:
            return None
        meta, content = result

        name = str(meta.get("name", path.stem))
        description = str(meta.get("description", ""))

        if not description:
            return None

        # Extract the optional metadata sub-map.
        metadata_raw = meta.get("metadata")
        metadata: dict[str, str] = metadata_raw if isinstance(metadata_raw, dict) else {}

        # Triggers — prefer metadata.triggers (CSV) over top-level list.
        if "triggers" in metadata:
            triggers = _csv_to_list(metadata["triggers"])
        else:
            triggers_raw = meta.get("triggers", [])
            triggers = list(triggers_raw) if isinstance(triggers_raw, list) else []

        # Auto-inject — prefer metadata.auto_inject over top-level.
        if "auto_inject" in metadata:
            auto_inject = metadata["auto_inject"].lower() == "true"
        else:
            auto_inject = bool(meta.get("auto_inject", False))

        return KnowledgeModule(
            name=name,
            description=description,
            triggers=triggers,
            auto_inject=auto_inject,
            content=content,
        )
