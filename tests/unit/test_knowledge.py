"""Tests for maike.agents.knowledge — knowledge module loader and selector."""

from pathlib import Path

from maike.agents.knowledge import KnowledgeLoader
from maike.agents.skill import SkillLoader


def _write_module(tmp_path: Path, name: str, description: str, triggers: list[str], content: str, auto_inject: bool = False) -> Path:
    """Helper: write a knowledge module .md file."""
    trigger_lines = "\n".join(f'  - "{t}"' for t in triggers)
    text = (
        f"---\n"
        f"name: {name}\n"
        f'description: "{description}"\n'
        f"triggers:\n{trigger_lines}\n"
        f"auto_inject: {'true' if auto_inject else 'false'}\n"
        f"---\n\n"
        f"{content}\n"
    )
    path = tmp_path / f"{name}.md"
    path.write_text(text)
    return path


class TestKnowledgeLoaderParsing:
    def test_parses_frontmatter(self, tmp_path):
        _write_module(tmp_path, "test-mod", "A test module", ["foo", "bar"], "Body content here.")
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        assert len(modules) == 1
        assert modules[0].name == "test-mod"
        assert modules[0].description == "A test module"
        assert modules[0].triggers == ["foo", "bar"]
        assert modules[0].auto_inject is False
        assert "Body content here." in modules[0].content

    def test_parses_auto_inject(self, tmp_path):
        _write_module(tmp_path, "auto", "Always loaded", [], "Always.", auto_inject=True)
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        assert modules[0].auto_inject is True

    def test_skips_files_without_frontmatter(self, tmp_path):
        (tmp_path / "no-front.md").write_text("Just plain markdown")
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        assert len(modules) == 0

    def test_skips_files_without_description(self, tmp_path):
        (tmp_path / "bad.md").write_text("---\nname: bad\n---\nContent")
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        assert len(modules) == 0

    def test_empty_dir_returns_empty(self, tmp_path):
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        assert modules == []

    def test_nonexistent_dir_returns_empty(self, tmp_path):
        loader = KnowledgeLoader(tmp_path / "nonexistent")
        modules = loader.load_all()
        assert modules == []


class TestBuildCatalog:
    def test_compact_format(self, tmp_path):
        _write_module(tmp_path, "recursion", "Recursive algorithms", ["recursive"], "R body")
        _write_module(tmp_path, "debugging", "Systematic debugging", ["debug"], "D body")
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        catalog = loader.build_catalog(modules)
        assert "- **debugging**: Systematic debugging" in catalog
        assert "- **recursion**: Recursive algorithms" in catalog
        assert catalog.count("\n") == 1  # exactly 2 lines

    def test_empty_catalog(self):
        loader = KnowledgeLoader()
        catalog = loader.build_catalog([])
        assert catalog == "(none available)"


class TestSelectForTask:
    def test_matches_trigger_substring(self, tmp_path):
        _write_module(tmp_path, "rec", "Recursion guide", ["recursive", "parser"], "R")
        _write_module(tmp_path, "api", "API guide", ["HTTP", "REST"], "A")
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        selected = loader.select_for_task("build a recursive parser", modules)
        assert len(selected) == 1
        assert selected[0].name == "rec"

    def test_no_match_returns_empty(self, tmp_path):
        _write_module(tmp_path, "rec", "Recursion guide", ["recursive"], "R")
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        selected = loader.select_for_task("create a web server with flask", modules)
        assert selected == []

    def test_auto_inject_always_selected(self, tmp_path):
        _write_module(tmp_path, "always", "Always loaded", [], "Always.", auto_inject=True)
        _write_module(tmp_path, "rec", "Recursion guide", ["recursive"], "R")
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        selected = loader.select_for_task("create a web server", modules)
        assert len(selected) == 1
        assert selected[0].name == "always"

    def test_multiple_matches(self, tmp_path):
        _write_module(tmp_path, "rec", "Recursion guide", ["recursive", "parser"], "R")
        _write_module(tmp_path, "debug", "Debug guide", ["debug", "traceback"], "D")
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        selected = loader.select_for_task("debug the recursive parser", modules)
        names = {m.name for m in selected}
        assert "rec" in names
        assert "debug" in names

    def test_case_insensitive_matching(self, tmp_path):
        _write_module(tmp_path, "api", "API guide", ["REST", "HTTP"], "A")
        loader = KnowledgeLoader(tmp_path)
        modules = loader.load_all()
        selected = loader.select_for_task("Build a rest API endpoint", modules)
        assert len(selected) == 1
        assert selected[0].name == "api"


class TestRealKnowledgeModules:
    """Test that the actual knowledge modules in prompts/knowledge/ load correctly."""

    def test_real_modules_load(self):
        loader = KnowledgeLoader()
        modules = loader.load_all()
        assert len(modules) >= 3  # debugging, refactoring, test-methodology
        names = {m.name for m in modules}
        assert "debugging" in names
        assert "refactoring" in names
        assert "test-methodology" in names

    def test_real_debugging_module_triggered_by_debug_task(self):
        loader = KnowledgeLoader()
        modules = loader.load_all()
        selected = loader.select_for_task("debug the traceback in the parser", modules)
        names = {m.name for m in selected}
        assert "debugging" in names

    def test_real_catalog_is_compact(self):
        loader = KnowledgeLoader()
        modules = loader.load_all()
        catalog = loader.build_catalog(modules)
        # Each module is one line; total should be reasonable.
        lines = catalog.strip().split("\n")
        assert len(lines) == len(modules)
        for line in lines:
            assert line.startswith("- **")


class TestKnowledgeSkillCompatibility:
    """Verify KnowledgeLoader and SkillLoader produce compatible results
    for the built-in knowledge modules."""

    def test_both_loaders_find_same_modules(self):
        """Both loaders should find the same set of modules from the built-in
        knowledge directory."""
        knowledge_dir = Path(__file__).parent.parent.parent / "maike" / "agents" / "prompts" / "knowledge"

        knowledge_loader = KnowledgeLoader(knowledge_dir)
        skill_loader = SkillLoader(builtin_dir=knowledge_dir)

        modules = knowledge_loader.load_all()
        skills = skill_loader.load_all()

        module_names = {m.name for m in modules}
        skill_names = {s.name for s in skills}
        assert module_names == skill_names, (
            f"Name mismatch: knowledge={module_names}, skills={skill_names}"
        )

    def test_both_loaders_agree_on_descriptions(self):
        """Descriptions should match between KnowledgeModule and Skill."""
        knowledge_dir = Path(__file__).parent.parent.parent / "maike" / "agents" / "prompts" / "knowledge"

        knowledge_loader = KnowledgeLoader(knowledge_dir)
        skill_loader = SkillLoader(builtin_dir=knowledge_dir)

        modules = {m.name: m for m in knowledge_loader.load_all()}
        skills = {s.name: s for s in skill_loader.load_all()}

        for name in modules:
            assert name in skills, f"Skill loader missing module: {name}"
            assert modules[name].description == skills[name].description, (
                f"Description mismatch for '{name}': "
                f"knowledge={modules[name].description!r}, "
                f"skill={skills[name].description!r}"
            )

    def test_both_loaders_agree_on_triggers(self):
        """Triggers should match between KnowledgeModule and Skill."""
        knowledge_dir = Path(__file__).parent.parent.parent / "maike" / "agents" / "prompts" / "knowledge"

        knowledge_loader = KnowledgeLoader(knowledge_dir)
        skill_loader = SkillLoader(builtin_dir=knowledge_dir)

        modules = {m.name: m for m in knowledge_loader.load_all()}
        skills = {s.name: s for s in skill_loader.load_all()}

        for name in modules:
            assert modules[name].triggers == skills[name].triggers, (
                f"Trigger mismatch for '{name}': "
                f"knowledge={modules[name].triggers}, "
                f"skill={skills[name].triggers}"
            )

    def test_both_loaders_agree_on_content(self):
        """Content body should match between KnowledgeModule and Skill."""
        knowledge_dir = Path(__file__).parent.parent.parent / "maike" / "agents" / "prompts" / "knowledge"

        knowledge_loader = KnowledgeLoader(knowledge_dir)
        skill_loader = SkillLoader(builtin_dir=knowledge_dir)

        modules = {m.name: m for m in knowledge_loader.load_all()}
        skills = {s.name: s for s in skill_loader.load_all()}

        for name in modules:
            assert modules[name].content == skills[name].content, (
                f"Content mismatch for '{name}'"
            )

    def test_catalogs_match(self):
        """build_catalog should produce the same output from both loaders."""
        knowledge_dir = Path(__file__).parent.parent.parent / "maike" / "agents" / "prompts" / "knowledge"

        knowledge_loader = KnowledgeLoader(knowledge_dir)
        skill_loader = SkillLoader(builtin_dir=knowledge_dir)

        modules = knowledge_loader.load_all()
        skills = skill_loader.load_all()

        knowledge_catalog = knowledge_loader.build_catalog(modules)
        skill_catalog = skill_loader.build_catalog(skills)

        assert knowledge_catalog == skill_catalog
