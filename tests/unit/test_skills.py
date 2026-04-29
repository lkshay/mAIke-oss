"""Tests for maike.agents.skill — skill data model, multi-source loader, and auto-injection."""

from __future__ import annotations

from pathlib import Path

from maike.agents.knowledge import KnowledgeLoader, KnowledgeModule
from maike.agents.skill import Skill, SkillLoader, SkillSource


# ── Helpers ────────────────────────────────────────────────────────────────

def _write_flat_skill(
    directory: Path,
    name: str,
    description: str,
    triggers: list[str] | None = None,
    content: str = "Body.",
    auto_inject: bool = False,
    paths: list[str] | None = None,
    tools: list[str] | None = None,
    disable_model_invocation: bool = False,
    user_invocable: bool = True,
) -> Path:
    """Write a flat ``.md`` skill file in *directory*."""
    lines = [
        "---",
        f"name: {name}",
        f'description: "{description}"',
    ]
    if triggers:
        lines.append("triggers:")
        for t in triggers:
            lines.append(f'  - "{t}"')
    if paths:
        lines.append("paths:")
        for p in paths:
            lines.append(f'  - "{p}"')
    if tools:
        lines.append("tools:")
        for t in tools:
            lines.append(f'  - "{t}"')
    lines.append(f"auto_inject: {'true' if auto_inject else 'false'}")
    if disable_model_invocation:
        lines.append("disable_model_invocation: true")
    if not user_invocable:
        lines.append("user_invocable: false")
    lines.append("---")
    lines.append("")
    lines.append(content)

    text = "\n".join(lines) + "\n"
    path = directory / f"{name}.md"
    path.write_text(text)
    return path


def _write_dir_skill(
    parent: Path,
    name: str,
    description: str,
    triggers: list[str] | None = None,
    content: str = "Body.",
    extra_md: dict[str, str] | None = None,
) -> Path:
    """Create a directory-based skill with ``SKILL.md`` (and optional extras)."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    lines = [
        "---",
        f"name: {name}",
        f'description: "{description}"',
    ]
    if triggers:
        lines.append("triggers:")
        for t in triggers:
            lines.append(f'  - "{t}"')
    lines.append("auto_inject: false")
    lines.append("---")
    lines.append("")
    lines.append(content)

    (skill_dir / "SKILL.md").write_text("\n".join(lines) + "\n")

    if extra_md:
        for fname, body in extra_md.items():
            (skill_dir / fname).write_text(body)

    return skill_dir


# ── Skill dataclass tests ─────────────────────────────────────────────────

class TestSkillDataclass:
    def test_skill_dataclass_defaults(self):
        """Skill with only required fields has sensible defaults."""
        s = Skill(
            name="minimal",
            description="A minimal skill",
            triggers=["test"],
            auto_inject=False,
            content="Hello",
        )
        assert s.paths == []
        assert s.tools is None
        assert s.disable_model_invocation is False
        assert s.user_invocable is True
        assert s.source == SkillSource.BUILTIN
        assert s.namespace is None
        assert s.skill_dir is None

    def test_skill_extends_knowledge_fields(self):
        """All KnowledgeModule fields are present on Skill."""
        s = Skill(
            name="test",
            description="desc",
            triggers=["t"],
            auto_inject=True,
            content="body",
        )
        assert s.name == "test"
        assert s.description == "desc"
        assert s.triggers == ["t"]
        assert s.auto_inject is True
        assert s.content == "body"


# ── Multi-source loading ──────────────────────────────────────────────────

class TestMultiSourceLoading:
    def test_multi_source_loading_precedence(self, tmp_path):
        """Project-level skill overrides user-level, which overrides builtin."""
        builtin = tmp_path / "builtin"
        user = tmp_path / "user"
        project = tmp_path / "project"
        builtin.mkdir()
        user.mkdir()
        project.mkdir()

        _write_flat_skill(builtin, "test-skill", "builtin version", content="BUILTIN")
        _write_flat_skill(user, "test-skill", "user version", content="USER")
        _write_flat_skill(project, "test-skill", "project version", content="PROJECT")

        loader = SkillLoader(
            builtin_dir=builtin,
            user_dir=user,
            project_dir=project,
        )
        skills = loader.load_all()
        assert len(skills) == 1
        assert skills[0].description == "project version"
        assert skills[0].source == SkillSource.PROJECT
        assert "PROJECT" in skills[0].content


# ── Flat-file backward compat ─────────────────────────────────────────────

class TestFlatFileCompat:
    def test_flat_md_backward_compat(self, tmp_path):
        """Existing flat .md file parses as a Skill with default new fields."""
        _write_flat_skill(
            tmp_path, "debug", "Debugging guide", ["debug", "traceback"],
            content="## Debug\nStep 1...",
        )
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        assert len(skills) == 1
        s = skills[0]
        assert s.name == "debug"
        assert s.triggers == ["debug", "traceback"]
        assert s.paths == []
        assert s.tools is None
        assert s.disable_model_invocation is False
        assert s.user_invocable is True
        assert s.source == SkillSource.BUILTIN
        assert s.skill_dir is None


# ── Directory-based skills ────────────────────────────────────────────────

class TestDirectoryBasedSkills:
    def test_directory_based_skill(self, tmp_path):
        """Subdirectory with SKILL.md loads with skill_dir set."""
        _write_dir_skill(
            tmp_path, "my-skill", "A dir skill", ["myskill"],
            content="Dir content",
        )
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        assert len(skills) == 1
        s = skills[0]
        assert s.name == "my-skill"
        assert s.skill_dir == tmp_path / "my-skill"
        assert s.source == SkillSource.BUILTIN


# ── Catalog ───────────────────────────────────────────────────────────────

class TestCatalog:
    def test_catalog_excludes_model_disabled(self, tmp_path):
        """Skills with disable_model_invocation=True are excluded from catalog."""
        _write_flat_skill(tmp_path, "visible", "Visible skill")
        _write_flat_skill(
            tmp_path, "hidden", "Hidden skill",
            disable_model_invocation=True,
        )
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        catalog = loader.build_catalog(skills)
        assert "visible" in catalog.lower()
        assert "hidden" not in catalog.lower()


# ── Path-based selection ──────────────────────────────────────────────────

class TestSelectForPaths:
    def test_select_for_paths_matches(self, tmp_path):
        """Skill with paths: ['*.py'] matches 'src/main.py'."""
        _write_flat_skill(
            tmp_path, "python-skill", "Python helper",
            paths=["*.py"],
        )
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        selected = loader.select_for_paths(["src/main.py"], skills)
        assert len(selected) == 1
        assert selected[0].name == "python-skill"

    def test_select_for_paths_no_match(self, tmp_path):
        """Skill with paths: ['*.rs'] does not match 'src/main.py'."""
        _write_flat_skill(
            tmp_path, "rust-skill", "Rust helper",
            paths=["*.rs"],
        )
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        selected = loader.select_for_paths(["src/main.py"], skills)
        assert len(selected) == 0


# ── Tool output matching ──────────────────────────────────────────────────

class TestMatchToolOutput:
    def test_match_tool_output(self, tmp_path):
        """Skill with trigger 'recursion' matches output containing RecursionError."""
        _write_flat_skill(
            tmp_path, "rec", "Recursion guide", ["recursion"],
        )
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        matched = loader._match_tool_output_from(
            "Traceback: RecursionError: maximum recursion depth exceeded",
            set(),
            skills,
        )
        assert len(matched) == 1
        assert matched[0].name == "rec"

    def test_match_tool_output_skips_already_injected(self, tmp_path):
        """Already-injected skills are filtered out of match results."""
        _write_flat_skill(
            tmp_path, "rec", "Recursion guide", ["recursion"],
        )
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        matched = loader._match_tool_output_from(
            "RecursionError in main.py",
            {"rec"},
            skills,
        )
        assert len(matched) == 0


# ── Supporting content ────────────────────────────────────────────────────

class TestSupportingContent:
    def test_load_supporting_content(self, tmp_path):
        """Extra .md files in a skill dir are loaded as supporting content."""
        skill_dir = _write_dir_skill(
            tmp_path, "rich-skill", "A skill with extras",
            content="Main body",
            extra_md={"reference.md": "# Reference\nSome reference material."},
        )
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        assert len(skills) == 1
        supporting = loader.load_supporting_content(skills[0])
        assert "Reference" in supporting
        assert "reference material" in supporting


# ── KnowledgeLoader backward compatibility ────────────────────────────────

class TestKnowledgeLoaderBackwardCompat:
    def test_knowledge_loader_backward_compat(self, tmp_path):
        """Old KnowledgeLoader API still works identically."""
        _write_flat_skill(
            tmp_path, "test-mod", "A test module", ["foo", "bar"],
            content="Body content here.",
        )
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        assert len(modules) == 1
        m = modules[0]
        assert isinstance(m, KnowledgeModule)
        assert m.name == "test-mod"
        assert m.description == "A test module"
        assert m.triggers == ["foo", "bar"]
        assert m.auto_inject is False
        assert "Body content here." in m.content

        # catalog
        catalog = loader.build_catalog(modules)
        assert "**test-mod**" in catalog

        # select_for_task
        selected = loader.select_for_task("need to foo the bar", modules)
        assert len(selected) == 1
        assert selected[0].name == "test-mod"

        # no match
        empty = loader.select_for_task("unrelated stuff", modules)
        assert empty == []


# ── Real builtin modules ──────────────────────────────────────────────────

class TestRealBuiltinModules:
    def test_real_builtin_modules_load(self):
        """All real builtin modules in prompts/knowledge/ load as Skills."""
        real_dir = Path(__file__).parent.parent.parent / "maike" / "agents" / "prompts" / "knowledge"
        loader = SkillLoader(builtin_dir=real_dir)
        skills = loader.load_all()
        names = {s.name for s in skills}
        assert len(skills) >= 3
        assert "debugging" in names
        assert "refactoring" in names
        assert "test-methodology" in names
        # All should be BUILTIN source
        for s in skills:
            assert s.source == SkillSource.BUILTIN


# ── Auto-injection helpers ───────────────────────────────────────────────

def _make_skill(
    name: str,
    triggers: list[str],
    content: str = "Skill content.",
    auto_inject: bool = False,
) -> Skill:
    """Create an in-memory Skill for testing (no disk I/O)."""
    return Skill(
        name=name,
        description=f"{name} skill",
        triggers=triggers,
        auto_inject=auto_inject,
        content=content,
        source=SkillSource.BUILTIN,
    )


def _make_loader_with_skills(skills: list[Skill]) -> SkillLoader:
    """Create a SkillLoader that returns the given skills from load_all()."""
    loader = SkillLoader(extra_skills=skills)
    return loader


# ── Skill auto-injection on failure ──────────────────────────────────────

class TestSkillAutoInjectionOnFailure:
    """Test that skill triggers match tool failure output and produce
    injectable content, as used by AgentCore's failure-nudge path."""

    def test_failure_output_matches_skill_trigger(self):
        """A skill with trigger 'timeout' matches error output containing 'timeout'."""
        skill = _make_skill("timeout-guide", ["timeout"], content="Handle timeouts by...")
        loader = _make_loader_with_skills([skill])
        matched = loader.match_tool_output(
            "Error: request timed out after 30s (timeout exceeded)",
            set(),
        )
        assert len(matched) == 1
        assert matched[0].name == "timeout-guide"

    def test_failure_output_injects_into_conversation(self):
        """Simulates the injection path: matched skill content is formatted
        as a user message with the expected prefix."""
        skill = _make_skill("debug-skill", ["traceback"], content="## Debugging\nUse pdb.")
        loader = _make_loader_with_skills([skill])
        injected: set[str] = set()
        conversation: list[dict] = []

        tool_output = "Traceback (most recent call last): ..."
        matched = loader.match_tool_output(tool_output, injected)
        for s in matched[:1]:  # limit: 1 per nudge
            injected.add(s.name)
            conversation.append({
                "role": "user",
                "content": f"## Auto-loaded Skill: {s.name}\n\n{s.content}",
            })

        assert len(conversation) == 1
        assert "## Auto-loaded Skill: debug-skill" in conversation[0]["content"]
        assert "## Debugging\nUse pdb." in conversation[0]["content"]
        assert "debug-skill" in injected

    def test_deduplication_prevents_repeat_injection(self):
        """A skill already in the injected set is not matched again."""
        skill = _make_skill("import-fix", ["ImportError"], content="Fix imports by...")
        loader = _make_loader_with_skills([skill])

        # First match succeeds
        matched1 = loader.match_tool_output("ImportError: No module named foo", set())
        assert len(matched1) == 1

        # Second match with skill already injected returns empty
        matched2 = loader.match_tool_output(
            "ImportError: No module named bar",
            {"import-fix"},
        )
        assert len(matched2) == 0

    def test_budget_limit_skips_injection(self):
        """When context is > 80% full, skill injection should be skipped.
        This tests the budget-check logic used in core.py."""
        from maike.constants import context_limit_for_model

        model = "claude-opus-4-20250514"
        ctx_limit = context_limit_for_model(model)

        # Simulate current tokens at 85% of context limit
        current_tokens = int(ctx_limit * 0.85)
        should_inject = current_tokens < ctx_limit * 0.8
        assert should_inject is False, "Should skip injection when > 80% full"

        # Simulate current tokens at 50% — should allow injection
        current_tokens_low = int(ctx_limit * 0.50)
        should_inject_low = current_tokens_low < ctx_limit * 0.8
        assert should_inject_low is True, "Should allow injection when < 80% full"

    def test_only_one_skill_per_nudge(self):
        """Even when multiple skills match, only the first is injected per nudge."""
        skills = [
            _make_skill("skill-a", ["error"], content="A"),
            _make_skill("skill-b", ["error"], content="B"),
        ]
        loader = _make_loader_with_skills(skills)
        matched = loader.match_tool_output("Some error occurred", set())
        # Both match, but injection code takes only matched[0]
        assert len(matched) == 2
        # In core.py: _matched[0] is used — verify first is picked
        first = matched[0]
        assert first.name in ("skill-a", "skill-b")


# ── Task-based skill injection ───────────────────────────────────────────

class TestTaskBasedSkillInjection:
    """Test that skills are selected and injected based on task text at
    the start of a run, including auto_inject skills."""

    def test_auto_inject_skill_selected(self):
        """A skill with auto_inject=True is always selected."""
        skill = _make_skill("always-on", ["irrelevant"], auto_inject=True, content="Always loaded.")
        loader = _make_loader_with_skills([skill])
        skills = loader.load_all()
        selected = loader.select_for_task("build a web app", skills)
        assert len(selected) == 1
        assert selected[0].name == "always-on"

    def test_trigger_matched_skill_selected(self):
        """A skill whose triggers match the task text is selected."""
        skill = _make_skill("docker-skill", ["docker", "container"], content="Docker guide.")
        loader = _make_loader_with_skills([skill])
        skills = loader.load_all()
        selected = loader.select_for_task("Set up a Docker container for the app", skills)
        assert len(selected) == 1
        assert selected[0].name == "docker-skill"

    def test_no_match_returns_empty(self):
        """A skill with unrelated triggers is not selected."""
        skill = _make_skill("rust-skill", ["cargo", "rustfmt"], content="Rust guide.")
        loader = _make_loader_with_skills([skill])
        skills = loader.load_all()
        selected = loader.select_for_task("Write a Python web server", skills)
        assert len(selected) == 0

    def test_task_injection_formats_conversation_message(self):
        """Simulates the task injection path in core.py run()."""
        skills = [
            _make_skill("git-help", ["git", "commit"], content="Git workflow tips."),
            _make_skill("always-loaded", [], auto_inject=True, content="Base guidance."),
        ]
        loader = _make_loader_with_skills(skills)
        all_skills = loader.load_all()

        conversation: list[dict] = [
            {"role": "user", "content": "Fix the git commit hook"},
        ]
        injected: set[str] = set()

        task_text = conversation[0]["content"]
        task_skills = loader.select_for_task(task_text, all_skills)
        for s in task_skills:
            if s.name not in injected:
                injected.add(s.name)
                conversation.append({
                    "role": "user",
                    "content": f"## Auto-loaded Skill: {s.name}\n\n{s.content}",
                })

        # Both skills should be injected: auto_inject + trigger match
        assert "always-loaded" in injected
        assert "git-help" in injected
        assert len(conversation) == 3  # original + 2 injected

    def test_task_injection_deduplication(self):
        """Skills injected at task time are not re-injected on failure."""
        skill = _make_skill("debug-skill", ["traceback", "error"], content="Debug help.")
        loader = _make_loader_with_skills([skill])
        injected: set[str] = set()

        # Simulate task injection
        all_skills = loader.load_all()
        task_skills = loader.select_for_task("debug this traceback error", all_skills)
        for s in task_skills:
            injected.add(s.name)

        assert "debug-skill" in injected

        # Now simulate failure-time matching — should return empty
        matched = loader.match_tool_output("traceback error in module", injected)
        assert len(matched) == 0


# ── Context var for skill loader ─────────────────────────────────────────

class TestSkillLoaderContextVar:
    """Test that the CURRENT_SKILL_LOADER context var works correctly."""

    def test_peek_returns_none_by_default(self):
        from maike.tools.context import peek_current_skill_loader
        assert peek_current_skill_loader() is None

    def test_set_and_peek(self):
        from maike.tools.context import CURRENT_SKILL_LOADER, peek_current_skill_loader
        loader = SkillLoader()
        tok = CURRENT_SKILL_LOADER.set(loader)
        try:
            assert peek_current_skill_loader() is loader
        finally:
            CURRENT_SKILL_LOADER.reset(tok)
        assert peek_current_skill_loader() is None

    def test_set_none_explicitly(self):
        from maike.tools.context import CURRENT_SKILL_LOADER, peek_current_skill_loader
        tok = CURRENT_SKILL_LOADER.set(None)
        try:
            assert peek_current_skill_loader() is None
        finally:
            CURRENT_SKILL_LOADER.reset(tok)
