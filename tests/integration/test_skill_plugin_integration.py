"""Integration tests for the skill -> plugin -> context pipeline.

Covers end-to-end flows: skill loading from disk, plugin discovery with
namespaced skills, catalog generation, trigger-based selection, background
process lifecycle, and async delegate manager lifecycle.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from maike.agents.knowledge import KnowledgeLoader
from maike.agents.skill import Skill, SkillLoader, SkillSource
from maike.atoms.agent import AgentResult
from maike.atoms.context import AgentProgress
from maike.orchestrator.orchestrator import AsyncDelegateManager
from maike.plugins.discovery import PluginDiscovery
from maike.plugins.loader import PluginLoader
from maike.runtime.background import BackgroundProcessManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine to completion."""
    return asyncio.run(coro)


def _write_skill_md(
    directory: Path,
    name: str,
    description: str,
    triggers: list[str],
    content: str,
    *,
    auto_inject: bool = False,
    paths: list[str] | None = None,
    tools: list[str] | None = None,
) -> Path:
    """Write a skill .md file with full frontmatter."""
    trigger_lines = "\n".join(f'  - "{t}"' for t in triggers)
    parts = [
        "---",
        f"name: {name}",
        f'description: "{description}"',
        f"triggers:\n{trigger_lines}",
        f"auto_inject: {'true' if auto_inject else 'false'}",
    ]
    if paths:
        path_lines = "\n".join(f'  - "{p}"' for p in paths)
        parts.append(f"paths:\n{path_lines}")
    if tools:
        tool_lines = "\n".join(f'  - "{t}"' for t in tools)
        parts.append(f"tools:\n{tool_lines}")
    parts.append("---")
    parts.append("")
    parts.append(content)

    text = "\n".join(parts) + "\n"
    path = directory / f"{name}.md"
    path.write_text(text, encoding="utf-8")
    return path


def _write_dir_skill(
    parent: Path,
    name: str,
    description: str,
    triggers: list[str],
    content: str,
    **kwargs,
) -> Path:
    """Write a directory-based skill (subdir/SKILL.md)."""
    skill_dir = parent / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    trigger_lines = "\n".join(f'  - "{t}"' for t in triggers)
    parts = [
        "---",
        f"name: {name}",
        f'description: "{description}"',
        f"triggers:\n{trigger_lines}",
        f"auto_inject: {'true' if kwargs.get('auto_inject', False) else 'false'}",
        "---",
        "",
        content,
    ]
    text = "\n".join(parts) + "\n"
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(text, encoding="utf-8")
    return skill_dir


def _create_plugin_structure(
    plugin_root: Path,
    plugin_name: str,
    version: str,
    skills: list[dict],
) -> Path:
    """Create a full plugin directory structure with .maike-plugin/plugin.json
    and skills/<name>/SKILL.md files.
    """
    plugin_dir = plugin_root / plugin_name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    # .maike-plugin/plugin.json
    maike_dir = plugin_dir / ".maike-plugin"
    maike_dir.mkdir(exist_ok=True)
    manifest = {
        "name": plugin_name,
        "version": version,
        "description": f"Test plugin: {plugin_name}",
        "author": "test-author",
    }
    (maike_dir / "plugin.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    # skills/<name>/SKILL.md
    skills_dir = plugin_dir / "skills"
    skills_dir.mkdir(exist_ok=True)
    for skill_def in skills:
        _write_dir_skill(
            skills_dir,
            skill_def["name"],
            skill_def["description"],
            skill_def.get("triggers", []),
            skill_def.get("content", "Skill content."),
        )

    return plugin_dir


def _make_agent_result(**overrides):
    defaults = dict(
        agent_id="test",
        role="delegate",
        stage_name="test",
        output="Done",
        messages=[],
        cost_usd=0.1,
        tokens_used=100,
        success=True,
    )
    defaults.update(overrides)
    return AgentResult(**defaults)


# ---------------------------------------------------------------------------
# a) SkillLoader with real .md files
# ---------------------------------------------------------------------------


class TestSkillLoaderWithRealModules:
    """Load skills from temp .md files and verify the Skill dataclass."""

    def test_flat_md_file_parsed(self, tmp_path):
        _write_skill_md(
            tmp_path,
            "docker-compose",
            "Docker Compose orchestration",
            ["docker", "compose", "container"],
            "Use `docker compose up` for multi-container applications.",
            paths=["docker-compose*.yml"],
            tools=["Bash", "Read"],
        )
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()

        assert len(skills) == 1
        skill = skills[0]
        assert skill.name == "docker-compose"
        assert skill.description == "Docker Compose orchestration"
        assert skill.triggers == ["docker", "compose", "container"]
        assert skill.auto_inject is False
        assert skill.paths == ["docker-compose*.yml"]
        assert skill.tools == ["Bash", "Read"]
        assert skill.source == SkillSource.BUILTIN
        assert "docker compose up" in skill.content

    def test_directory_based_skill_parsed(self, tmp_path):
        _write_dir_skill(
            tmp_path,
            "k8s-deploy",
            "Kubernetes deployment guidance",
            ["kubernetes", "kubectl", "helm"],
            "Deploy with `kubectl apply -f`.",
        )
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()

        assert len(skills) == 1
        skill = skills[0]
        assert skill.name == "k8s-deploy"
        assert skill.description == "Kubernetes deployment guidance"
        assert skill.triggers == ["kubernetes", "kubectl", "helm"]
        assert skill.skill_dir == tmp_path / "k8s-deploy"

    def test_auto_inject_skill(self, tmp_path):
        _write_skill_md(
            tmp_path,
            "always-on",
            "Always injected skill",
            [],
            "This skill is always active.",
            auto_inject=True,
        )
        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        assert skills[0].auto_inject is True

    def test_precedence_project_overrides_builtin(self, tmp_path):
        builtin = tmp_path / "builtin"
        builtin.mkdir()
        project = tmp_path / "project"
        project.mkdir()

        _write_skill_md(builtin, "git-flow", "Builtin git", ["git"], "Builtin version.")
        _write_skill_md(project, "git-flow", "Project git", ["git"], "Project override.")

        loader = SkillLoader(builtin_dir=builtin, project_dir=project)
        skills = loader.load_all()

        assert len(skills) == 1
        assert skills[0].description == "Project git"
        assert skills[0].source == SkillSource.PROJECT


# ---------------------------------------------------------------------------
# b) Plugin discovery with skills
# ---------------------------------------------------------------------------


class TestPluginDiscoveryWithSkills:
    """Discover plugins and load namespaced skills."""

    def test_discover_and_load_skills(self, tmp_path):
        _create_plugin_structure(
            tmp_path,
            "acme-tools",
            "1.0.0",
            [
                {
                    "name": "lint-config",
                    "description": "Linting configuration",
                    "triggers": ["lint", "eslint", "ruff"],
                    "content": "Configure linters for your project.",
                },
                {
                    "name": "ci-pipeline",
                    "description": "CI/CD pipeline setup",
                    "triggers": ["ci", "github-actions", "pipeline"],
                    "content": "Set up CI/CD pipelines.",
                },
            ],
        )

        manifests = PluginDiscovery.discover([tmp_path])
        assert len(manifests) == 1
        assert manifests[0].name == "acme-tools"
        assert manifests[0].version == "1.0.0"

        skills = PluginLoader.load_skills(manifests[0])
        assert len(skills) == 2

        names = {s.name for s in skills}
        assert "acme-tools:lint-config" in names
        assert "acme-tools:ci-pipeline" in names

        for skill in skills:
            assert skill.source == SkillSource.PLUGIN
            assert skill.namespace == "acme-tools"

    def test_multiple_plugins(self, tmp_path):
        _create_plugin_structure(
            tmp_path,
            "plugin-a",
            "0.1.0",
            [{"name": "skill-a", "description": "Skill A", "triggers": ["alpha"]}],
        )
        _create_plugin_structure(
            tmp_path,
            "plugin-b",
            "0.2.0",
            [{"name": "skill-b", "description": "Skill B", "triggers": ["beta"]}],
        )

        manifests = PluginDiscovery.discover([tmp_path])
        assert len(manifests) == 2

        all_skills = PluginLoader.load_all_plugin_skills(manifests)
        names = {s.name for s in all_skills}
        assert "plugin-a:skill-a" in names
        assert "plugin-b:skill-b" in names

    def test_plugin_without_skills_dir(self, tmp_path):
        plugin_dir = tmp_path / "empty-plugin"
        plugin_dir.mkdir()
        maike_dir = plugin_dir / ".maike-plugin"
        maike_dir.mkdir()
        (maike_dir / "plugin.json").write_text(
            json.dumps({"name": "empty-plugin", "version": "1.0.0"}),
            encoding="utf-8",
        )

        manifests = PluginDiscovery.discover([tmp_path])
        assert len(manifests) == 1
        skills = PluginLoader.load_skills(manifests[0])
        assert skills == []


# ---------------------------------------------------------------------------
# c) Skill catalog in system prompt
# ---------------------------------------------------------------------------


class TestSkillCatalogInSystemPrompt:
    """Verify the catalog string contains all skill names and descriptions."""

    def test_catalog_contains_all_skills(self, tmp_path):
        _write_skill_md(tmp_path, "docker", "Docker container management", ["docker"], "D")
        _write_skill_md(tmp_path, "terraform", "Infrastructure as code", ["terraform"], "T")
        _write_skill_md(tmp_path, "graphql", "GraphQL API design", ["graphql"], "G")

        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        catalog = loader.build_catalog(skills)

        assert "**docker**" in catalog
        assert "Docker container management" in catalog
        assert "**terraform**" in catalog
        assert "Infrastructure as code" in catalog
        assert "**graphql**" in catalog
        assert "GraphQL API design" in catalog

    def test_catalog_excludes_disabled_model_invocation(self, tmp_path):
        # Write a skill with disable_model_invocation=true
        text = (
            "---\n"
            "name: internal-only\n"
            'description: "Internal skill"\n'
            "triggers:\n"
            '  - "internal"\n'
            "auto_inject: false\n"
            "disable_model_invocation: true\n"
            "---\n\n"
            "Internal content.\n"
        )
        (tmp_path / "internal-only.md").write_text(text, encoding="utf-8")
        _write_skill_md(tmp_path, "visible", "Visible skill", ["vis"], "V")

        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()
        catalog = loader.build_catalog(skills)

        assert "internal-only" not in catalog
        assert "**visible**" in catalog

    def test_empty_skills_catalog(self):
        loader = SkillLoader()
        catalog = loader.build_catalog([])
        assert catalog == "(none available)"


# ---------------------------------------------------------------------------
# d) Skill selection by task
# ---------------------------------------------------------------------------


class TestSkillSelectionByTask:
    """Verify select_for_task returns only matching skills."""

    def test_trigger_match(self, tmp_path):
        _write_skill_md(tmp_path, "react-spa", "React SPA guide", ["react", "jsx", "hooks"], "R")
        _write_skill_md(tmp_path, "django-orm", "Django ORM guide", ["django", "orm", "queryset"], "D")
        _write_skill_md(tmp_path, "fastapi", "FastAPI guide", ["fastapi", "pydantic"], "F")

        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()

        selected = loader.select_for_task("build a react component with hooks", skills)
        names = {s.name for s in selected}
        assert "react-spa" in names
        assert "django-orm" not in names
        assert "fastapi" not in names

    def test_no_match_returns_empty(self, tmp_path):
        _write_skill_md(tmp_path, "rust-ffi", "Rust FFI guide", ["rust", "ffi", "unsafe"], "R")

        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()

        selected = loader.select_for_task("create a python flask web server", skills)
        assert selected == []

    def test_auto_inject_always_selected(self, tmp_path):
        _write_skill_md(
            tmp_path,
            "conventions",
            "Project conventions",
            [],
            "Always follow these rules.",
            auto_inject=True,
        )
        _write_skill_md(tmp_path, "rust-ffi", "Rust FFI", ["rust"], "R")

        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()

        selected = loader.select_for_task("write a python script", skills)
        assert len(selected) == 1
        assert selected[0].name == "conventions"

    def test_multiple_triggers_match(self, tmp_path):
        _write_skill_md(tmp_path, "testing", "Test patterns", ["pytest", "unittest", "mock"], "T")
        _write_skill_md(tmp_path, "debug", "Debugging", ["debug", "traceback", "pdb"], "D")

        loader = SkillLoader(builtin_dir=tmp_path)
        skills = loader.load_all()

        selected = loader.select_for_task("debug failing pytest tests", skills)
        names = {s.name for s in selected}
        assert "testing" in names
        assert "debug" in names


# ---------------------------------------------------------------------------
# e) Background process lifecycle
# ---------------------------------------------------------------------------


class TestBackgroundProcessLifecycle:
    """Start a background process, check status, stop, verify output."""

    def test_echo_lifecycle(self, tmp_path):
        async def _test():
            mgr = BackgroundProcessManager()
            try:
                bp = await mgr.start("echo hello-integration", cwd=tmp_path)
                assert bp.handle == "bg-001"
                assert bp.status == "running"

                # Wait for the short-lived echo to finish
                await asyncio.sleep(0.5)

                info = await mgr.check(bp.handle)
                assert info["status"] == "exited"
                assert info["exit_code"] == 0
                assert "hello-integration" in info["recent_output"]
            finally:
                await mgr.cleanup()

        _run(_test())

    def test_long_running_stop(self, tmp_path):
        async def _test():
            mgr = BackgroundProcessManager()
            try:
                bp = await mgr.start("sleep 60", cwd=tmp_path)
                # Process should be running
                info = await mgr.check(bp.handle)
                assert info["status"] == "running"

                # Stop it
                result = await asyncio.wait_for(mgr.stop(bp.handle), timeout=15)
                assert result["status"] == "exited"
                assert result["exit_code"] is not None
            finally:
                await mgr.cleanup()

        _run(_test())

    def test_output_captured(self, tmp_path):
        async def _test():
            mgr = BackgroundProcessManager()
            try:
                script = (
                    "import sys\n"
                    "for i in range(5):\n"
                    "    print(f'line-{i}', flush=True)\n"
                )
                bp = await mgr.start(
                    f'{sys.executable} -c "{script}"',
                    cwd=tmp_path,
                )
                # Wait for the process to finish
                await asyncio.sleep(1.5)

                info = await mgr.check(bp.handle)
                output = info["recent_output"]
                assert "line-0" in output
                assert "line-4" in output
            finally:
                await mgr.cleanup()

        _run(_test())


# ---------------------------------------------------------------------------
# f) AsyncDelegateManager lifecycle
# ---------------------------------------------------------------------------


class TestAsyncDelegateManagerLifecycle:
    """Create an AsyncDelegateManager, start coroutines, check transitions."""

    def test_immediate_completion(self):
        async def _test():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _quick():
                return _make_agent_result(output="Task completed successfully")

            delegate = await mgr.start(_quick(), ctx, "quick integration task")
            assert delegate.handle == "delegate-001"
            assert delegate.status == "running"

            # Wait for the task to finish
            await delegate.task_future
            status = mgr.check(delegate.handle)
            assert status["status"] == "completed"
            assert status["output"] == "Task completed successfully"

        _run(_test())

    def test_running_to_completed_transition(self):
        async def _test():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())
            event = asyncio.Event()

            async def _blocking():
                await event.wait()
                return _make_agent_result(output="Unblocked and done")

            delegate = await mgr.start(_blocking(), ctx, "blocking task")

            # Should be running initially
            status = mgr.check(delegate.handle)
            assert status["status"] == "running"

            # Unblock the coroutine
            event.set()
            await delegate.task_future

            # Should now be completed
            status = mgr.check(delegate.handle)
            assert status["status"] == "completed"
            assert status["output"] == "Unblocked and done"

        _run(_test())

    def test_failure_transition(self):
        async def _test():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

            async def _failing():
                raise ValueError("integration test error")

            delegate = await mgr.start(_failing(), ctx, "failing task")
            try:
                await delegate.task_future
            except ValueError:
                pass

            status = mgr.check(delegate.handle)
            assert status["status"] == "failed"
            assert "integration test error" in status["error"]

        _run(_test())

    def test_multiple_delegates_concurrent(self):
        async def _test():
            mgr = AsyncDelegateManager()
            delegates = []

            for i in range(3):
                ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())

                async def _task(idx=i):
                    await asyncio.sleep(0.05)
                    return _make_agent_result(output=f"Result {idx}")

                d = await mgr.start(_task(), ctx, f"task-{i}")
                delegates.append(d)

            # Wait for all to complete
            await asyncio.gather(*(d.task_future for d in delegates))

            for i, d in enumerate(delegates):
                status = mgr.check(d.handle)
                assert status["status"] == "completed"
                assert f"Result {i}" in status["output"]

        _run(_test())

    def test_cleanup_cancels_running(self):
        async def _test():
            mgr = AsyncDelegateManager()
            ctx = SimpleNamespace(cost_used_usd=0.0, tokens_used=0, progress=AgentProgress())
            event = asyncio.Event()

            async def _slow():
                await event.wait()
                return _make_agent_result()

            delegate = await mgr.start(_slow(), ctx, "slow cleanup task")
            assert mgr.running_count == 1

            await mgr.cleanup(grace_period=0.1)

            assert delegate.status == "failed"
            assert "Cancelled on session cleanup" in delegate.error

        _run(_test())
