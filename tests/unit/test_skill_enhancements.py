"""Tests for skill enhancements: variable substitution, arguments, fork mode,
budget-constrained catalog, and conditional activation."""

from pathlib import Path

from maike.agents.skill import (
    Skill,
    SkillLoader,
    SkillSource,
    expand_skill_content,
)


# ── Variable substitution ──────────────────────────────────────────────────


class TestExpandSkillContent:
    def test_replaces_arguments(self):
        skill = Skill(
            name="test", description="test", triggers=[], auto_inject=False,
            content="Review PR ${ARGUMENTS}",
        )
        result = expand_skill_content(skill, args="123")
        assert result == "Review PR 123"

    def test_replaces_skill_dir(self):
        skill = Skill(
            name="test", description="test", triggers=[], auto_inject=False,
            content="Read ${SKILL_DIR}/template.json",
            skill_dir=Path("/tmp/my-skill"),
        )
        result = expand_skill_content(skill)
        assert result == "Read /tmp/my-skill/template.json"

    def test_replaces_session_and_workspace(self):
        skill = Skill(
            name="test", description="test", triggers=[], auto_inject=False,
            content="Session: ${SESSION_ID}, Workspace: ${WORKSPACE}",
        )
        result = expand_skill_content(skill, session_id="s-123", workspace="/home/user/proj")
        assert "s-123" in result
        assert "/home/user/proj" in result

    def test_named_args(self):
        skill = Skill(
            name="test", description="test", triggers=[], auto_inject=False,
            content="File: ${ARG:file}, Mode: ${ARG:mode}",
            arguments=["file", "mode"],
        )
        result = expand_skill_content(skill, args="main.py strict")
        assert result == "File: main.py, Mode: strict"

    def test_missing_named_args_replaced_with_empty(self):
        skill = Skill(
            name="test", description="test", triggers=[], auto_inject=False,
            content="File: ${ARG:file}, Mode: ${ARG:mode}",
            arguments=["file", "mode"],
        )
        result = expand_skill_content(skill, args="main.py")
        assert result == "File: main.py, Mode: "

    def test_no_variables_unchanged(self):
        skill = Skill(
            name="test", description="test", triggers=[], auto_inject=False,
            content="No variables here",
        )
        result = expand_skill_content(skill)
        assert result == "No variables here"

    def test_empty_args_clears_placeholder(self):
        skill = Skill(
            name="test", description="test", triggers=[], auto_inject=False,
            content="Args: ${ARGUMENTS}",
        )
        result = expand_skill_content(skill, args="")
        assert result == "Args: "

    def test_missing_skill_dir_uses_empty(self):
        skill = Skill(
            name="test", description="test", triggers=[], auto_inject=False,
            content="Dir: ${SKILL_DIR}",
            skill_dir=None,
        )
        result = expand_skill_content(skill)
        assert result == "Dir: "


# ── Argument parsing from frontmatter ──────────────────────────────────────


def _write_skill(tmp_path: Path, name: str, body: str, **extra) -> Path:
    lines = ["---", f"name: {name}", f"description: A {name} skill"]
    for key, value in extra.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f'  - "{item}"')
        else:
            lines.append(f"{key}: {value}")
    lines += ["---", "", body]
    subdir = tmp_path / name
    subdir.mkdir(exist_ok=True)
    path = subdir / "SKILL.md"
    path.write_text("\n".join(lines))
    return path


class TestSkillFrontmatterNewFields:
    def test_parses_arguments(self, tmp_path):
        _write_skill(tmp_path, "review", "Review PR ${ARG:pr}", arguments=["pr", "mode"])
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        assert len(skills) == 1
        assert skills[0].arguments == ["pr", "mode"]

    def test_parses_argument_hint(self, tmp_path):
        lines = [
            "---",
            "name: review",
            "description: Review a PR",
            "argument-hint: <pr-number>",
            "---",
            "",
            "Review body",
        ]
        (tmp_path / "review.md").write_text("\n".join(lines))
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        assert skills[0].argument_hint == "<pr-number>"

    def test_parses_context_fork(self, tmp_path):
        _write_skill(tmp_path, "commit", "Commit changes", context="fork")
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        assert skills[0].context == "fork"

    def test_parses_agent_type(self, tmp_path):
        _write_skill(tmp_path, "deploy", "Deploy", context="fork", agent="implement")
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        assert skills[0].agent_type == "implement"

    def test_parses_model_override(self, tmp_path):
        _write_skill(tmp_path, "analyze", "Analyze", model="strong")
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        assert skills[0].model_override == "strong"

    def test_invalid_context_defaults_to_inline(self, tmp_path):
        _write_skill(tmp_path, "bad", "Bad context", context="invalid")
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        assert skills[0].context == "inline"

    def test_defaults_for_new_fields(self, tmp_path):
        _write_skill(tmp_path, "basic", "Basic skill")
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        s = skills[0]
        assert s.arguments == []
        assert s.argument_hint == ""
        assert s.context == "inline"
        assert s.agent_type is None
        assert s.model_override is None


# ── Budget-constrained catalog ─────────────────────────────────────────────


class TestBudgetConstrainedCatalog:
    def _make_skills(self, count: int, desc_len: int = 50, source: SkillSource = SkillSource.USER) -> list[Skill]:
        return [
            Skill(
                name=f"skill-{i}",
                description="x" * desc_len,
                triggers=[], auto_inject=False,
                content="body",
                source=source,
            )
            for i in range(count)
        ]

    def test_under_budget_full_descriptions(self):
        loader = SkillLoader()
        skills = self._make_skills(3, desc_len=20)
        catalog = loader.build_catalog(skills, budget_chars=5000)
        for s in skills:
            assert s.name in catalog
            assert s.description in catalog

    def test_over_budget_truncates_non_builtin(self):
        loader = SkillLoader()
        skills = self._make_skills(20, desc_len=200)
        catalog = loader.build_catalog(skills, budget_chars=500)
        # Should still contain all names
        for s in skills:
            assert s.name in catalog
        # Descriptions should be truncated (not full 200 chars each)
        assert len(catalog) <= 1500  # generous upper bound

    def test_builtin_never_truncated(self):
        loader = SkillLoader()
        builtin = self._make_skills(2, desc_len=100, source=SkillSource.BUILTIN)
        user = self._make_skills(5, desc_len=200, source=SkillSource.USER)
        # Give each user skill a unique name
        user = [Skill(name=f"user-{i}", description=s.description, triggers=[], auto_inject=False, content="", source=SkillSource.USER) for i, s in enumerate(user)]
        catalog = loader.build_catalog(builtin + user, budget_chars=800)
        # Builtin descriptions should be complete
        for s in builtin:
            assert s.description in catalog

    def test_empty_catalog(self):
        loader = SkillLoader()
        catalog = loader.build_catalog([])
        assert catalog == "(none available)"

    def test_argument_hint_in_catalog(self):
        loader = SkillLoader()
        skill = Skill(
            name="review-pr",
            description="Review a PR",
            triggers=[], auto_inject=False,
            content="body",
            argument_hint="<pr-number>",
        )
        catalog = loader.build_catalog([skill])
        assert "<pr-number>" in catalog
        assert "review-pr" in catalog

    def test_model_disabled_excluded(self):
        loader = SkillLoader()
        skills = [
            Skill(name="visible", description="Visible", triggers=[], auto_inject=False, content=""),
            Skill(name="hidden", description="Hidden", triggers=[], auto_inject=False, content="", disable_model_invocation=True),
        ]
        catalog = loader.build_catalog(skills)
        assert "visible" in catalog
        assert "hidden" not in catalog


# ── Conditional activation ─────────────────────────────────────────────────


class TestConditionalSkillActivation:
    def test_select_conditional_matches_path(self):
        loader = SkillLoader()
        skill = Skill(
            name="api-helper", description="API helper", triggers=[],
            auto_inject=False, content="API guidance",
            paths=["src/api/*"],
        )
        result = loader.select_conditional(
            ["src/api/users.py"], [skill], already_injected=set(),
        )
        assert len(result) == 1
        assert result[0].name == "api-helper"

    def test_select_conditional_skips_injected(self):
        loader = SkillLoader()
        skill = Skill(
            name="api-helper", description="API helper", triggers=[],
            auto_inject=False, content="API guidance",
            paths=["src/api/*"],
        )
        result = loader.select_conditional(
            ["src/api/users.py"], [skill], already_injected={"api-helper"},
        )
        assert result == []

    def test_select_conditional_skips_no_paths(self):
        loader = SkillLoader()
        skill = Skill(
            name="general", description="General", triggers=["debug"],
            auto_inject=False, content="Debug help",
        )
        result = loader.select_conditional(
            ["src/api/users.py"], [skill], already_injected=set(),
        )
        assert result == []

    def test_select_conditional_no_match(self):
        loader = SkillLoader()
        skill = Skill(
            name="api-helper", description="API helper", triggers=[],
            auto_inject=False, content="API guidance",
            paths=["src/api/*"],
        )
        result = loader.select_conditional(
            ["src/models/user.py"], [skill], already_injected=set(),
        )
        assert result == []
