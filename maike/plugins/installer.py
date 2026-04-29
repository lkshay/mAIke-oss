"""Plugin installer — install from local directory or git repository."""
from __future__ import annotations

import logging
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from maike.plugins.manifest import PluginManifest, parse_plugin_manifest
from maike.plugins.settings import InstalledPluginRecord, load_settings, save_settings

logger = logging.getLogger(__name__)


class PluginInstallError(RuntimeError):
    """Raised when a plugin install fails."""


_GIT_URL_PATTERN = re.compile(
    r"^(https?://|git@|ssh://|git://)"
    r"|\.git$"
)


def _is_git_url(source: str) -> bool:
    """Detect whether *source* looks like a git URL."""
    return bool(_GIT_URL_PATTERN.search(source))


def resolve_install_dir(dir_preset: str | None, scope: str) -> Path:
    """Resolve the target directory for plugin installation.

    Presets:
    - ``None`` + ``"user"``     → ``~/.config/maike/plugins/``
    - ``None`` + ``"project"``  → ``.maike/plugins/``
    - any other string          → ``Path(dir_preset)``
    """
    from maike.constants import PLUGIN_USER_DIR

    if dir_preset is None:
        if scope == "project":
            from maike.constants import PLUGIN_PROJECT_SUBDIR
            return Path.cwd() / PLUGIN_PROJECT_SUBDIR
        return PLUGIN_USER_DIR

    return Path(dir_preset)


def install_plugin(
    source: str,
    target_dir: Path | None = None,
    scope: str = "user",
    force: bool = False,
) -> PluginManifest:
    """Install a plugin from a local directory or git repository.

    Returns the parsed manifest of the installed plugin.
    Raises ``PluginInstallError`` on failure.
    """
    dest_parent = target_dir or resolve_install_dir(None, scope)
    dest_parent.mkdir(parents=True, exist_ok=True)

    if _is_git_url(source):
        return _install_from_git(source, dest_parent, force=force)
    else:
        return _install_from_local(source, dest_parent, force=force)


def _install_from_local(source: str, dest_parent: Path, force: bool = False) -> PluginManifest:
    """Copy a local plugin directory to the target."""
    src = Path(source).resolve()
    if not src.is_dir():
        raise PluginInstallError(f"Source is not a directory: {source}")

    manifest = parse_plugin_manifest(src)
    if manifest is None:
        raise PluginInstallError(
            f"No valid .maike-plugin/plugin.json found in {source}"
        )

    dest = dest_parent / manifest.name
    if dest.exists():
        if not force:
            raise PluginInstallError(
                f"Plugin '{manifest.name}' already exists at {dest}. Use --force to overwrite."
            )
        shutil.rmtree(dest)

    shutil.copytree(src, dest)

    # Re-parse from installed location
    installed_manifest = parse_plugin_manifest(dest)
    if installed_manifest is None:
        raise PluginInstallError(f"Installed plugin is invalid at {dest}")

    _record_install(installed_manifest, source)
    return installed_manifest


def _install_from_git(url: str, dest_parent: Path, force: bool = False) -> PluginManifest:
    """Clone a git repository as a plugin."""
    if shutil.which("git") is None:
        raise PluginInstallError("git is not installed. Install git and try again.")

    # Clone to a temp name first, then rename after parsing manifest
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_clone = Path(tmpdir) / "plugin"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(tmp_clone)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise PluginInstallError(
                f"git clone failed: {exc.stderr.strip() or exc.stdout.strip()}"
            ) from exc

        manifest = parse_plugin_manifest(tmp_clone)
        if manifest is None:
            raise PluginInstallError(
                f"Cloned repository has no valid .maike-plugin/plugin.json"
            )

        dest = dest_parent / manifest.name
        if dest.exists():
            if not force:
                raise PluginInstallError(
                    f"Plugin '{manifest.name}' already exists at {dest}. Use --force to overwrite."
                )
            shutil.rmtree(dest)

        shutil.copytree(tmp_clone, dest)

    installed_manifest = parse_plugin_manifest(dest)
    if installed_manifest is None:
        raise PluginInstallError(f"Installed plugin is invalid at {dest}")

    _record_install(installed_manifest, url)
    return installed_manifest


def _record_install(manifest: PluginManifest, source: str) -> None:
    """Record the installation in settings.json."""
    settings = load_settings()
    settings.installed[manifest.name] = InstalledPluginRecord(
        source=source,
        version=manifest.version,
        installed_at=str(manifest.path),
        installed_on=datetime.now(timezone.utc).isoformat(),
    )
    # Ensure newly installed plugin is enabled
    settings.disabled.discard(manifest.name)
    save_settings(settings)


def update_plugin(name: str) -> PluginManifest | None:
    """Update a git-installed plugin by pulling latest changes.

    Returns the updated manifest, or ``None`` if the plugin is not
    git-installed.
    """
    settings = load_settings()
    record = settings.installed.get(name)
    if record is None:
        raise PluginInstallError(f"Plugin '{name}' is not recorded as installed")

    if not _is_git_url(record.source):
        return None  # local install — can't auto-update

    plugin_path = Path(record.installed_at)
    if not plugin_path.is_dir():
        raise PluginInstallError(f"Plugin directory not found: {plugin_path}")

    if shutil.which("git") is None:
        raise PluginInstallError("git is not installed")

    try:
        subprocess.run(
            ["git", "-C", str(plugin_path), "pull"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise PluginInstallError(f"git pull failed: {exc.stderr.strip()}") from exc

    manifest = parse_plugin_manifest(plugin_path)
    if manifest is None:
        raise PluginInstallError(f"Plugin invalid after update at {plugin_path}")

    # Update version in settings
    record = settings.installed[name]
    settings.installed[name] = InstalledPluginRecord(
        source=record.source,
        version=manifest.version,
        installed_at=record.installed_at,
        installed_on=datetime.now(timezone.utc).isoformat(),
    )
    save_settings(settings)
    return manifest


# ── Uninstall ──────────────────────────────────────────────────────────────


def uninstall_plugin(name: str, remove_data: bool = False) -> None:
    """Remove an installed plugin.

    Deletes the plugin directory, removes it from settings, and
    optionally removes its persistent data directory.
    """
    settings = load_settings()
    record = settings.installed.get(name)
    if record is None:
        raise PluginInstallError(f"Plugin '{name}' is not installed")

    plugin_path = Path(record.installed_at)
    if plugin_path.is_dir():
        shutil.rmtree(plugin_path)
        logger.info("Removed plugin directory: %s", plugin_path)

    del settings.installed[name]
    settings.disabled.discard(name)
    settings.config.pop(name, None)
    save_settings(settings)

    if remove_data:
        data_dir = Path.home() / ".config" / "maike" / "plugins" / "data" / name
        if data_dir.is_dir():
            shutil.rmtree(data_dir)
            logger.info("Removed plugin data: %s", data_dir)


# ── Standalone skill install ───────────────────────────────────────────────


def install_skill(
    source: str,
    scope: str = "user",
    force: bool = False,
) -> list[str]:
    """Install standalone skill(s) from a local directory or git repo.

    Supported source formats:
    - Local directory with ``SKILL.md`` at root → single skill
    - Local directory with ``skills/`` subdirectory → multiple skills
    - Git URL → clone, then treat as local

    Returns a list of installed skill names.
    """
    from maike.constants import SKILL_PROJECT_SUBDIR, SKILL_USER_DIR

    target = SKILL_USER_DIR if scope == "user" else Path.cwd() / SKILL_PROJECT_SUBDIR
    target.mkdir(parents=True, exist_ok=True)

    if _is_git_url(source):
        return _install_skill_from_git(source, target, force)
    return _install_skill_from_local(source, target, force)


def _install_skill_from_local(
    source: str, target: Path, force: bool,
) -> list[str]:
    """Install skill(s) from a local directory."""
    src = Path(source).resolve()
    if not src.is_dir():
        raise PluginInstallError(f"Source is not a directory: {source}")

    installed: list[str] = []

    # Case 1: SKILL.md at root → single skill
    skill_md = src / "SKILL.md"
    if skill_md.is_file():
        name = _skill_name_from_file(skill_md, src.name)
        _copy_skill_dir(src, target / name, force)
        installed.append(name)
        return installed

    # Case 2: skills/ subdirectory → multiple skills
    skills_dir = src / "skills"
    if skills_dir.is_dir():
        for subdir in sorted(skills_dir.iterdir()):
            if not subdir.is_dir():
                continue
            if (subdir / "SKILL.md").is_file():
                name = _skill_name_from_file(subdir / "SKILL.md", subdir.name)
                _copy_skill_dir(subdir, target / name, force)
                installed.append(name)

    if not installed:
        raise PluginInstallError(
            f"No SKILL.md found at root or in skills/ subdirectory of {source}"
        )
    return installed


def _install_skill_from_git(
    url: str, target: Path, force: bool,
) -> list[str]:
    """Clone a git repo and install skill(s) from it."""
    if shutil.which("git") is None:
        raise PluginInstallError("git is not installed. Install git and try again.")

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_clone = Path(tmpdir) / "skill-repo"
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(tmp_clone)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise PluginInstallError(
                f"git clone failed: {exc.stderr.strip() or exc.stdout.strip()}"
            ) from exc

        return _install_skill_from_local(str(tmp_clone), target, force)


def _skill_name_from_file(skill_md: Path, fallback: str) -> str:
    """Extract the skill name from SKILL.md frontmatter, or use the fallback."""
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return fallback

    import re
    match = re.match(r"^---\s*\n(.*?)\n---", text, re.DOTALL)
    if not match:
        return fallback

    for line in match.group(1).splitlines():
        line = line.strip()
        if line.startswith("name:"):
            name = line[5:].strip().strip('"').strip("'")
            if name:
                return name
    return fallback


def _copy_skill_dir(src: Path, dest: Path, force: bool) -> None:
    """Copy a skill directory to the target location."""
    if dest.exists():
        if not force:
            raise PluginInstallError(
                f"Skill directory already exists: {dest}. Use --force to overwrite."
            )
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


# ── Install feedback ───────────────────────────────────────────────────────


def describe_plugin(manifest: PluginManifest) -> str:
    """Return a human-readable summary of plugin components."""
    parts: list[str] = []

    skills_dir = manifest.skills_dir
    if skills_dir.is_dir():
        skill_count = sum(1 for _ in skills_dir.rglob("SKILL.md"))
        flat_count = sum(1 for f in skills_dir.glob("*.md") if f.name != "SKILL.md")
        total = skill_count + flat_count
        if total:
            parts.append(f"  Skills: {total}")

    agents_dir = manifest.agents_dir
    if agents_dir.is_dir():
        agent_count = sum(1 for _ in agents_dir.glob("*.md"))
        if agent_count:
            parts.append(f"  Agents: {agent_count}")

    if manifest.hooks_file.is_file():
        parts.append("  Hooks: yes")

    if manifest.mcp_config_file.is_file():
        parts.append("  MCP servers: yes")

    if manifest.lsp_config_file.is_file():
        parts.append("  LSP servers: yes")

    return "\n".join(parts) if parts else "  (no components found)"
