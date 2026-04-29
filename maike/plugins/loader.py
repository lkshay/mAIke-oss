"""Load skills from plugins."""
from __future__ import annotations

import dataclasses
import logging

from maike.agents.skill import Skill, SkillLoader, SkillSource
from maike.plugins.manifest import PluginManifest, _substitute_env_vars

logger = logging.getLogger(__name__)


class PluginLoader:
    """Load skills from discovered plugins."""

    @staticmethod
    def load_skills(manifest: PluginManifest) -> list[Skill]:
        """Load all skills from a plugin, namespaced by plugin name.

        Looks for skills/<name>/SKILL.md files in the plugin directory.
        Each skill is namespaced as "plugin-name:skill-name".
        Substitutes ``${MAIKE_PLUGIN_ROOT}`` in skill content so paths
        to scripts and resources resolve correctly.
        """
        skills_dir = manifest.skills_dir
        if not skills_dir.is_dir():
            return []

        # Use SkillLoader's parsing with the plugin's skills dir
        loader = SkillLoader(builtin_dir=skills_dir)
        raw_skills = loader.load_all()

        env = manifest.env_vars()
        return [
            dataclasses.replace(
                skill,
                name=f"{manifest.name}:{skill.name}",
                content=_substitute_env_vars(skill.content, env),
                source=SkillSource.PLUGIN,
                namespace=manifest.name,
            )
            for skill in raw_skills
        ]

    @staticmethod
    def load_all_plugin_skills(manifests: list[PluginManifest]) -> list[Skill]:
        """Load skills from all plugins."""
        all_skills: list[Skill] = []
        for manifest in manifests:
            try:
                skills = PluginLoader.load_skills(manifest)
                all_skills.extend(skills)
                if skills:
                    logger.info(
                        "Loaded %d skills from plugin '%s'",
                        len(skills),
                        manifest.name,
                    )
            except Exception:
                logger.warning(
                    "Failed to load skills from plugin '%s'",
                    manifest.name,
                    exc_info=True,
                )
        return all_skills
