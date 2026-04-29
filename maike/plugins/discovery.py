"""Plugin directory discovery."""
from __future__ import annotations

import logging
from pathlib import Path

from maike.plugins.manifest import PluginManifest, parse_plugin_manifest

logger = logging.getLogger(__name__)


class PluginDiscovery:
    """Discover plugins from multiple directories."""

    @staticmethod
    def discover(
        dirs: list[Path],
        disabled_names: set[str] | None = None,
    ) -> list[PluginManifest]:
        """Scan directories for plugins.

        Each directory is expected to contain plugin subdirectories,
        each with a .maike-plugin/plugin.json manifest.

        If *disabled_names* is provided, manifests whose name is in the
        set are excluded from the result.

        Returns parsed manifests sorted by name.
        """
        manifests: dict[str, PluginManifest] = {}  # name -> manifest (last wins)
        for search_dir in dirs:
            if not search_dir.is_dir():
                continue
            for candidate in sorted(search_dir.iterdir()):
                if not candidate.is_dir():
                    continue
                manifest = parse_plugin_manifest(candidate)
                if manifest is not None:
                    if manifest.name in manifests:
                        logger.info(
                            "Plugin '%s' from %s overrides previous from %s",
                            manifest.name,
                            candidate,
                            manifests[manifest.name].path,
                        )
                    manifests[manifest.name] = manifest

        if disabled_names:
            manifests = {k: v for k, v in manifests.items() if k not in disabled_names}

        return sorted(manifests.values(), key=lambda m: m.name)

    @staticmethod
    def discover_enabled(dirs: list[Path]) -> list[PluginManifest]:
        """Discover plugins, excluding those disabled in settings."""
        from maike.plugins.settings import load_settings

        settings = load_settings()
        return PluginDiscovery.discover(dirs, disabled_names=settings.disabled)
