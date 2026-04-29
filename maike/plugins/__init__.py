"""Plugin system — manifest parsing, discovery, skill/tool/hook/agent/LSP loading.

Follows the component-bundle model: a plugin is a directory containing skills,
MCP servers, hooks, agents, and LSP servers.  Discovery scans user and project
directories; components are loaded and registered at session start.
"""
