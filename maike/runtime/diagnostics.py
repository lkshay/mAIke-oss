"""Environment error diagnostics for tool results.

When execute_bash or run_tests fails, this module pattern-matches the output
against known environment error patterns (ModuleNotFoundError, command not
found, etc.) and appends a human-readable diagnostic block.  The agent sees the
diagnostic in its conversation and can fix the root cause (install the package)
instead of working around the symptom (mock the import).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from maike.atoms.tool import ToolResult
    from maike.runtime.probe import EnvironmentManifest


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorDiagnosis:
    """A single diagnosed environment error."""

    category: str           # "missing_dependency" | "missing_venv" | "missing_command" | "permission_error"
    pattern_matched: str    # The regex that matched
    remediation: str        # Human-readable fix suggestion
    install_command: str | None = None   # Specific command if applicable
    severity: str = "environment"        # "environment" | "code" | "system"


@dataclass(frozen=True)
class _ErrorPattern:
    """An error pattern with its regex and remediation template."""

    regex: re.Pattern[str]
    category: str
    remediation_template: str       # May use {module}, {cmd}, etc.
    severity: str = "environment"


# ---------------------------------------------------------------------------
# Error pattern registries — one list per language, plus universal
# ---------------------------------------------------------------------------

_PYTHON_PATTERNS: list[_ErrorPattern] = [
    _ErrorPattern(
        regex=re.compile(r"ModuleNotFoundError: No module named ['\"]?(\w[\w.]*)"),
        category="missing_dependency",
        remediation_template="Missing Python package '{module}'. Install with: {install_cmd}",
    ),
    _ErrorPattern(
        regex=re.compile(r"ImportError: cannot import name ['\"]?(\w+)['\"]? from ['\"]?(\w[\w.]*)"),
        category="missing_dependency",
        remediation_template="Cannot import '{name}' from '{module}'. The package may need updating: {install_cmd}",
    ),
    _ErrorPattern(
        regex=re.compile(r"No such file or directory.*\.venv"),
        category="missing_venv",
        remediation_template="Virtual environment not found. Create one with: python3 -m venv .venv && .venv/bin/pip install -e .",
    ),
]

_NODE_PATTERNS: list[_ErrorPattern] = [
    _ErrorPattern(
        regex=re.compile(r"Cannot find module ['\"]([^'\"]+)['\"]"),
        category="missing_dependency",
        remediation_template="Missing Node module '{module}'. Run: {install_all_cmd}",
    ),
    _ErrorPattern(
        regex=re.compile(r"ERR_MODULE_NOT_FOUND"),
        category="missing_dependency",
        remediation_template="Module not found. Dependencies may not be installed. Run: {install_all_cmd}",
    ),
]

_RUST_PATTERNS: list[_ErrorPattern] = [
    _ErrorPattern(
        regex=re.compile(r"error\[E0433\]: failed to resolve.*`(\w+)`"),
        category="missing_dependency",
        remediation_template="Unresolved crate '{module}'. Add to Cargo.toml: cargo add {module}",
    ),
]

_GO_PATTERNS: list[_ErrorPattern] = [
    _ErrorPattern(
        regex=re.compile(r'cannot find package "([^"]+)"'),
        category="missing_dependency",
        remediation_template="Missing Go package '{module}'. Run: go get {module}",
    ),
]

_UNIVERSAL_PATTERNS: list[_ErrorPattern] = [
    _ErrorPattern(
        regex=re.compile(r"(?:bash: |zsh: )?(?:command not found|not found): (\S+)"),
        category="missing_command",
        remediation_template="Command '{cmd}' not found. It may need to be installed or is not on PATH.",
        severity="system",
    ),
    _ErrorPattern(
        regex=re.compile(r"Permission denied"),
        category="permission_error",
        remediation_template="Permission denied. The command may need elevated privileges or a different user context.",
        severity="system",
    ),
]

# Registry keyed by language — add new languages here.
_LANGUAGE_PATTERNS: dict[str, list[_ErrorPattern]] = {
    "python": _PYTHON_PATTERNS,
    "node": _NODE_PATTERNS,
    "rust": _RUST_PATTERNS,
    "go": _GO_PATTERNS,
}


# ---------------------------------------------------------------------------
# Main diagnostics class
# ---------------------------------------------------------------------------

class EnvironmentDiagnostics:
    """Pattern-match tool output for environment errors and annotate results."""

    def __init__(self, manifest: EnvironmentManifest) -> None:
        self._manifest = manifest
        self._patterns = self._build_pattern_set()

    def _build_pattern_set(self) -> list[_ErrorPattern]:
        """Combine language-specific and universal patterns."""
        lang = self._manifest.language.lower()
        patterns: list[_ErrorPattern] = []
        if lang in _LANGUAGE_PATTERNS:
            patterns.extend(_LANGUAGE_PATTERNS[lang])
        patterns.extend(_UNIVERSAL_PATTERNS)
        return patterns

    def diagnose(self, output: str) -> list[ErrorDiagnosis]:
        """Scan output text for environment error patterns."""
        if not output:
            return []

        diagnoses: list[ErrorDiagnosis] = []
        seen_categories: set[str] = set()

        for pattern in self._patterns:
            match = pattern.regex.search(output)
            if match and pattern.category not in seen_categories:
                seen_categories.add(pattern.category)
                remediation = self._render_remediation(pattern, match)
                install_cmd = self._get_install_command(pattern, match)
                diagnoses.append(ErrorDiagnosis(
                    category=pattern.category,
                    pattern_matched=pattern.regex.pattern,
                    remediation=remediation,
                    install_command=install_cmd,
                    severity=pattern.severity,
                ))
        return diagnoses

    def annotate_tool_result(self, result: ToolResult) -> ToolResult:
        """If the result is a failure, check for environment errors and append hints.

        Successful results are returned unchanged.  Failed results with no
        matching patterns are also returned unchanged.
        """
        if result.success:
            return result

        # Check both output and raw_output for patterns
        text_to_check = result.raw_output or result.output
        diagnoses = self.diagnose(text_to_check)
        if not diagnoses:
            return result

        hint_block = self._format_hint_block(diagnoses)
        annotated_output = f"{result.output}\n\n{hint_block}"

        metadata = dict(result.metadata)
        metadata["environment_diagnoses"] = [
            {
                "category": d.category,
                "remediation": d.remediation,
                "install_command": d.install_command,
                "severity": d.severity,
            }
            for d in diagnoses
        ]

        return result.model_copy(update={
            "output": annotated_output,
            "metadata": metadata,
        })

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _render_remediation(self, pattern: _ErrorPattern, match: re.Match) -> str:
        """Fill in the remediation template with matched groups and manifest info."""
        groups = match.groups()
        replacements: dict[str, str] = {}

        # Extract named captures from the match
        if groups:
            replacements["module"] = groups[0]
        if len(groups) > 1:
            replacements["name"] = groups[0]
            replacements["module"] = groups[1]

        # Pull install commands from the manifest
        if self._manifest.install_command:
            # install_command is a template like "pip install {package}"
            module = replacements.get("module", "")
            replacements["install_cmd"] = self._manifest.install_command.format(package=module)
        else:
            replacements["install_cmd"] = f"install {replacements.get('module', 'the package')}"

        if self._manifest.install_all_command:
            replacements["install_all_cmd"] = self._manifest.install_all_command
        else:
            replacements["install_all_cmd"] = replacements.get("install_cmd", "install all dependencies")

        # For 'command not found' patterns
        if groups:
            replacements["cmd"] = groups[0]

        try:
            return pattern.remediation_template.format(**replacements)
        except KeyError:
            return pattern.remediation_template

    def _get_install_command(self, pattern: _ErrorPattern, match: re.Match) -> str | None:
        """Extract a specific install command if applicable."""
        if pattern.category != "missing_dependency":
            return None
        groups = match.groups()
        module = groups[0] if groups else None
        if not module or not self._manifest.install_command:
            return self._manifest.install_all_command
        return self._manifest.install_command.format(package=module)

    def _format_hint_block(self, diagnoses: list[ErrorDiagnosis]) -> str:
        """Format diagnoses into a readable block appended to tool output."""
        lines = [
            "\u2500\u2500 ENVIRONMENT DIAGNOSTIC \u2500\u2500",
            "This looks like an environment issue, not a code bug.",
        ]
        for d in diagnoses:
            lines.append(d.remediation)
        if self._manifest.install_all_command:
            lines.append(
                f"Or install all project dependencies: {self._manifest.install_all_command}"
            )
        return "\n".join(lines)
