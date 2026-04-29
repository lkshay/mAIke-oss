"""Skill system — evolved from knowledge modules with multi-source loading,
path-based activation, tool restrictions, and plugin support.

Skills are loaded from multiple directories with precedence:
    extra_skills (PLUGIN) < BUILTIN < USER < PROJECT

Each skill is a frozen dataclass parsed from ``.md`` files (flat) or
``SKILL.md`` inside a named subdirectory.  Backward-compatible with the
original knowledge-module frontmatter format.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ── Shared text-processing helpers (also re-exported by knowledge.py) ──────

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


def _extract_keywords(text: str) -> list[str]:
    """Extract meaningful lowercase keywords from *text*, filtering stopwords."""
    tokens = _IDENTIFIER_PATTERN.findall(text)
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


def _triggers_match(triggers: list[str], text: str) -> bool:
    """Return ``True`` if any *trigger* matches *text* (substring or keyword)."""
    text_lower = text.lower()
    text_keywords = _extract_keywords(text)
    for trigger in triggers:
        trigger_lower = trigger.lower()
        if trigger_lower in text_lower:
            return True
        if any(trigger_lower == kw for kw in text_keywords):
            return True
    return False


# Minimum word overlap for description-based matching.
_DESCRIPTION_MATCH_MIN_OVERLAP = 2

# Minimal stopwords for description matching — only function words.
# Much lighter than _STOPWORDS to preserve domain terms like "build",
# "page", "design" that _extract_keywords would strip.
_DESC_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "can",
    "and", "or", "but", "not", "no", "nor", "so", "yet",
    "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "up", "into", "through", "during", "before", "after",
    "that", "this", "these", "those", "it", "its", "my", "your",
    "we", "they", "he", "she", "you", "me", "us", "them",
    "what", "which", "who", "whom", "where", "when", "why", "how",
    "if", "then", "than", "also", "just", "very", "too",
    # Meta-language common in skill descriptions — not domain-specific.
    "use", "user", "skill", "ask", "create", "want", "need",
})


def _stem_simple(word: str) -> str:
    """Crude suffix stripping — handles plural/past tense/gerund."""
    if len(word) <= 4:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    # Only strip "es" after sibilants (sh, ch, x, s, z) where it's a real suffix.
    if word.endswith("es") and len(word) > 4:
        if word[-3] in "shxz" or word.endswith("ches") or word.endswith("shes"):
            return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    if word.endswith("ing") and len(word) > 6:
        return word[:-3]
    if word.endswith("ed") and len(word) > 5:
        return word[:-2]
    return word


def _tokenize_light(text: str) -> set[str]:
    """Tokenize with minimal filtering and simple stemming.

    Unlike ``_extract_keywords`` (which strips domain verbs like
    "build", "create", "page"), this keeps them — essential for
    matching skill descriptions against task text.
    """
    words = _IDENTIFIER_PATTERN.findall(text)
    result: set[str] = set()
    for w in words:
        lower = w.lower()
        if lower in _DESC_STOPWORDS or len(lower) < 3:
            continue
        result.add(_stem_simple(lower))
    return result


def _description_matches_task(description: str, task: str) -> bool:
    """Return ``True`` if a skill's *description* shares enough content words with *task*.

    Used as a fallback when a skill has no explicit triggers (common for
    plugin skills).  Uses a light tokenizer with simple stemming so that
    "pages" matches "page" and domain verbs like "build" are preserved.
    """
    desc_tokens = _tokenize_light(description)
    task_tokens = _tokenize_light(task)
    overlap = desc_tokens & task_tokens
    return len(overlap) >= _DESCRIPTION_MATCH_MIN_OVERLAP


# ── Data model ─────────────────────────────────────────────────────────────

class SkillSource(str, Enum):
    """Where a skill was loaded from (determines override precedence)."""
    BUILTIN = "builtin"
    USER = "user"
    PROJECT = "project"
    PLUGIN = "plugin"


@dataclass(frozen=True)
class Skill:
    """A single skill definition — superset of the old KnowledgeModule."""
    name: str
    description: str
    triggers: list[str]
    auto_inject: bool
    content: str
    # New fields (backward-compatible defaults)
    paths: list[str] = field(default_factory=list)
    tools: list[str] | None = None
    disable_model_invocation: bool = False
    user_invocable: bool = True
    source: SkillSource = SkillSource.BUILTIN
    namespace: str | None = None
    skill_dir: Path | None = None
    # Argument system
    arguments: list[str] = field(default_factory=list)
    argument_hint: str = ""
    # Execution mode
    context: str = "inline"  # "inline" or "fork"
    agent_type: str | None = None  # agent type for fork mode
    model_override: str | None = None  # model tier override for fork mode


# ── Frontmatter parser ────────────────────────────────────────────────────

def _parse_frontmatter(text: str) -> tuple[dict[str, str | list[str] | bool | dict[str, str]], str] | None:
    """Parse YAML-ish frontmatter delimited by ``---``.

    Returns ``(meta_dict, body_content)`` or ``None`` if the file has no
    valid frontmatter.  No PyYAML dependency.

    Supports one level of nesting: a key with no inline value followed by
    indented ``key: value`` lines produces a ``dict`` value, while indented
    ``- value`` lines produce a ``list``.
    """
    if not text.startswith("---"):
        return None
    end = text.find("---", 3)
    if end < 0:
        return None
    frontmatter = text[3:end].strip()
    content = text[end + 3:].strip()

    meta: dict[str, str | list[str] | bool | dict[str, str]] = {}
    current_key: str | None = None
    current_list: list[str] | None = None
    current_map: dict[str, str] | None = None
    for line in frontmatter.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Detect indented lines (belonging to a block under current_key).
        is_indented = line.startswith(" ") or line.startswith("\t")
        if is_indented and current_key is not None:
            if stripped.startswith("- ") and current_list is not None:
                value = stripped[2:].strip().strip('"').strip("'")
                current_list.append(value)
                continue
            # Indented key: value → nested map entry.
            if ":" in stripped:
                if current_map is None and current_list is not None and not current_list:
                    # Switch from empty list collector to map collector.
                    current_list = None
                    current_map = {}
                if current_map is not None:
                    k, _, v = stripped.partition(":")
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    current_map[k] = v
                    continue
            continue
        # Flush any pending block.
        if current_key is not None:
            if current_map is not None:
                meta[current_key] = current_map
            elif current_list is not None:
                meta[current_key] = current_list
            current_key = None
            current_list = None
            current_map = None
        if ":" in stripped:
            key, _, val = stripped.partition(":")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not val:
                current_key = key
                current_list = []
            elif val.lower() in ("true", "false"):
                meta[key] = val.lower() == "true"
            else:
                meta[key] = val
    # Flush trailing block.
    if current_key is not None:
        if current_map is not None:
            meta[current_key] = current_map
        elif current_list is not None:
            meta[current_key] = current_list

    return meta, content


def parse_md_frontmatter(path: Path) -> tuple[dict[str, str | list[str] | bool], str] | None:
    """Read a ``.md`` file and parse its ``---`` delimited frontmatter.

    Convenience wrapper around :func:`_parse_frontmatter` for callers
    that start from a file path rather than raw text.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return _parse_frontmatter(text)


def _csv_to_list(value: str) -> list[str]:
    """Split a comma-separated string into a trimmed list of non-empty items."""
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_skill(
    path: Path,
    source: SkillSource,
    skill_dir: Path | None = None,
    namespace: str | None = None,
) -> Skill | None:
    """Parse a ``.md`` file into a :class:`Skill`, or ``None`` on failure.

    Supports an optional ``metadata`` map in the frontmatter whose fields
    (``triggers``, ``paths``, ``auto_inject``) override the top-level
    equivalents.  Metadata values are comma-separated strings rather than
    YAML lists.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    result = _parse_frontmatter(text)
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

    # Paths — prefer metadata.paths (CSV) over top-level list.
    if "paths" in metadata:
        paths = _csv_to_list(metadata["paths"])
    else:
        paths_raw = meta.get("paths", [])
        paths = list(paths_raw) if isinstance(paths_raw, list) else []

    # Auto-inject — prefer metadata.auto_inject (string) over top-level bool.
    if "auto_inject" in metadata:
        auto_inject = metadata["auto_inject"].lower() == "true"
    else:
        auto_inject = bool(meta.get("auto_inject", False))

    tools_raw = meta.get("tools", None)
    tools: list[str] | None = None
    if isinstance(tools_raw, list):
        tools = list(tools_raw)

    disable_model_invocation = bool(meta.get("disable_model_invocation", False))
    user_invocable = bool(meta.get("user_invocable", True))

    # Argument system
    arguments_raw = meta.get("arguments", [])
    arguments: list[str] = list(arguments_raw) if isinstance(arguments_raw, list) else []
    argument_hint = str(meta.get("argument-hint", meta.get("argument_hint", "")))

    # Execution mode
    context_mode = str(meta.get("context", "inline"))
    if context_mode not in ("inline", "fork"):
        context_mode = "inline"
    agent_type_override = meta.get("agent")
    agent_type_override = str(agent_type_override) if agent_type_override else None
    model_override = meta.get("model")
    model_override = str(model_override) if model_override else None

    return Skill(
        name=name,
        description=description,
        triggers=triggers,
        auto_inject=auto_inject,
        content=content,
        paths=paths,
        tools=tools,
        disable_model_invocation=disable_model_invocation,
        user_invocable=user_invocable,
        source=source,
        namespace=namespace,
        skill_dir=skill_dir,
        arguments=arguments,
        argument_hint=argument_hint,
        context=context_mode,
        agent_type=agent_type_override,
        model_override=model_override,
    )


# ── Loader ─────────────────────────────────────────────────────────────────

class SkillLoader:
    """Multi-source skill loader with precedence-based deduplication."""

    def __init__(
        self,
        builtin_dir: Path | None = None,
        user_dir: Path | None = None,
        project_dir: Path | None = None,
        extra_skills: list[Skill] | None = None,
    ) -> None:
        self._builtin_dir = builtin_dir
        self._user_dir = user_dir
        self._project_dir = project_dir
        self._extra_skills = extra_skills or []

    # ── Loading ────────────────────────────────────────────────────────

    def load_all(self) -> list[Skill]:
        """Load skills from all sources.

        Precedence (later wins on name conflicts):
            extra_skills (PLUGIN) < BUILTIN < USER < PROJECT
        """
        by_name: dict[str, Skill] = {}

        # 1. Extra / plugin skills (lowest precedence)
        for skill in self._extra_skills:
            by_name[skill.name] = skill

        # 2. Builtin
        for skill in self._load_dir(self._builtin_dir, SkillSource.BUILTIN):
            by_name[skill.name] = skill

        # 3. User
        for skill in self._load_dir(self._user_dir, SkillSource.USER):
            by_name[skill.name] = skill

        # 4. Project (highest precedence)
        for skill in self._load_dir(self._project_dir, SkillSource.PROJECT):
            by_name[skill.name] = skill

        return sorted(by_name.values(), key=lambda s: s.name)

    def _load_dir(self, directory: Path | None, source: SkillSource) -> list[Skill]:
        """Load skills from a single directory.

        Supports two layouts:
        - Flat ``.md`` files (legacy knowledge modules)
        - Subdirectories containing ``SKILL.md``
        """
        if directory is None or not directory.is_dir():
            return []
        skills: list[Skill] = []

        # Flat .md files
        for path in sorted(directory.glob("*.md")):
            skill = _parse_skill(path, source=source)
            if skill is not None:
                skills.append(skill)

        # Directory-based skills (subdir/SKILL.md)
        for subdir in sorted(directory.iterdir()):
            if not subdir.is_dir():
                continue
            skill_md = subdir / "SKILL.md"
            if skill_md.is_file():
                skill = _parse_skill(
                    skill_md,
                    source=source,
                    skill_dir=subdir,
                )
                if skill is not None:
                    skills.append(skill)

        return skills

    # ── Catalog ────────────────────────────────────────────────────────

    def build_catalog(
        self, skills: list[Skill], budget_chars: int = 8000,
    ) -> str:
        """Build a compact, one-line-per-skill catalog for the system prompt.

        Skills with ``disable_model_invocation=True`` are excluded.
        Budget-constrained: builtin skills keep full descriptions, others
        are truncated proportionally if the total exceeds *budget_chars*.
        """
        _ENTRY_CAP = 250  # per-entry hard cap

        visible = [s for s in skills if not s.disable_model_invocation]
        if not visible:
            return "(none available)"

        def _entry(s: Skill, max_desc: int = _ENTRY_CAP) -> str:
            hint = f" {s.argument_hint}" if s.argument_hint else ""
            desc = s.description[:max_desc]
            if len(s.description) > max_desc:
                desc += "..."
            return f"- **{s.name}**{hint}: {desc}"

        # Try full descriptions first.
        full = "\n".join(_entry(s) for s in visible)
        if len(full) <= budget_chars:
            return full

        # Split: builtin always full, rest get truncated.
        builtin = [s for s in visible if s.source == SkillSource.BUILTIN]
        rest = [s for s in visible if s.source != SkillSource.BUILTIN]
        builtin_text = "\n".join(_entry(s) for s in builtin)
        remaining = max(budget_chars - len(builtin_text) - 1, 0)

        if not rest or remaining < 20 * len(rest):
            # Not enough room for descriptions — names only for non-builtin.
            rest_text = "\n".join(f"- **{s.name}**" for s in rest)
        else:
            max_desc = max(remaining // len(rest) - 20, 10)  # overhead
            rest_text = "\n".join(_entry(s, max_desc) for s in rest)

        parts = [p for p in (builtin_text, rest_text) if p]
        return "\n".join(parts)

    # ── Selection ──────────────────────────────────────────────────────

    def select_for_task(self, task: str, skills: list[Skill]) -> list[Skill]:
        """Select skills whose triggers match *task* keywords.

        When a skill has no explicit triggers (common for plugin skills),
        keywords are extracted from its description as a fallback.  This
        ensures plugin skills like ``frontend-design`` get matched against
        tasks about building web UIs even without hand-written triggers.
        """
        selected: list[Skill] = []
        for skill in skills:
            if skill.auto_inject:
                selected.append(skill)
            elif skill.triggers and _triggers_match(skill.triggers, task):
                selected.append(skill)
            elif not skill.triggers and _description_matches_task(skill.description, task):
                selected.append(skill)
        return selected

    def select_for_paths(
        self,
        changed_paths: list[str],
        skills: list[Skill],
    ) -> list[Skill]:
        """Select skills whose ``paths`` globs match any of *changed_paths*."""
        selected: list[Skill] = []
        for skill in skills:
            if not skill.paths:
                continue
            matched = False
            for pattern in skill.paths:
                for cp in changed_paths:
                    if fnmatch.fnmatch(cp, pattern):
                        matched = True
                        break
                if matched:
                    break
            if matched:
                selected.append(skill)
        return selected

    # ── Lookup by name ──────────────────────────────────────────────────

    def load_by_name(
        self, name: str, skills: list[Skill] | None = None,
    ) -> Skill | None:
        """Return the skill with the given *name*, or ``None``."""
        if skills is None:
            skills = self.load_all()
        for skill in skills:
            if skill.name == name:
                return skill
        return None

    # ── Supporting content ─────────────────────────────────────────────

    @staticmethod
    def load_supporting_content(skill: Skill) -> str:
        """Load and concatenate extra ``.md`` files in a directory-based skill.

        Returns all ``.md`` content in *skill.skill_dir* (including
        subdirectories like ``references/``) except ``SKILL.md`` itself.
        Returns an empty string when the skill has no directory.
        """
        if skill.skill_dir is None or not skill.skill_dir.is_dir():
            return ""
        parts: list[str] = []
        for md_path in sorted(skill.skill_dir.rglob("*.md")):
            if md_path.name == "SKILL.md":
                continue
            try:
                parts.append(md_path.read_text(encoding="utf-8").strip())
            except OSError:
                continue
        return "\n\n".join(parts)

    # ── Mid-session matching ───────────────────────────────────────────

    def match_tool_output(
        self,
        output: str,
        already_injected: set[str],
    ) -> list[Skill]:
        """Match skill triggers against tool *output* text.

        Filters out skills already in *already_injected* (by name).
        Requires a fresh ``load_all()`` set — call the loader first to get
        the full skill list, then pass it indirectly by using this method
        which loads internally.
        """
        all_skills = self.load_all()
        return self._match_tool_output_from(output, already_injected, all_skills)

    def _match_tool_output_from(
        self,
        output: str,
        already_injected: set[str],
        skills: list[Skill],
    ) -> list[Skill]:
        """Core matching logic — testable without disk I/O."""
        return [
            s for s in skills
            if s.name not in already_injected and _triggers_match(s.triggers, output)
        ]

    # ── Conditional activation ────────────────────────────────────────

    def select_conditional(
        self,
        touched_paths: list[str],
        skills: list[Skill],
        already_injected: set[str],
    ) -> list[Skill]:
        """Select skills with ``paths:`` patterns that match *touched_paths*.

        Filters out skills already in *already_injected* and skills without
        ``paths`` patterns (unconditional skills are handled at startup).
        """
        candidates = [
            s for s in skills
            if s.paths and s.name not in already_injected
        ]
        return self.select_for_paths(touched_paths, candidates)


# ── Variable substitution ──────────────────────────────────────────────────

def expand_skill_content(
    skill: Skill,
    args: str = "",
    session_id: str = "",
    workspace: str = "",
) -> str:
    """Substitute variables in skill content at invocation time.

    Supported variables:
      - ``${ARGUMENTS}``  — raw argument string
      - ``${ARG:name}``   — named argument (positional from args)
      - ``${SKILL_DIR}``  — absolute path to skill's directory
      - ``${SESSION_ID}`` — current session ID
      - ``${WORKSPACE}``  — workspace root path
    """
    content = skill.content
    content = content.replace("${ARGUMENTS}", args)
    content = content.replace("${SKILL_DIR}", str(skill.skill_dir or ""))
    content = content.replace("${SESSION_ID}", session_id)
    content = content.replace("${WORKSPACE}", workspace)

    # Named argument substitution: ${ARG:name} → positional value.
    if skill.arguments and args:
        parts = args.split(None, len(skill.arguments) - 1)
        for i, arg_name in enumerate(skill.arguments):
            val = parts[i] if i < len(parts) else ""
            content = content.replace(f"${{ARG:{arg_name}}}", val)

    return content
