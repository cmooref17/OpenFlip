"""Deferred tool loading, Claude Code style (its ToolSearch / `defer_loading`).

Every tool definition is still sent to the API on every request, but tools
not in the always-loaded set carry `defer_loading: true`: the API keeps them
out of the model's context (and out of the cached prefix) until the model
loads them with the `tool_search` tool, which returns `tool_reference`
blocks the API expands into full definitions. Loaded tools stay available
for the rest of the conversation (the API re-expands the references found
in history).

Config (config.json, all optional):
  "tool_deferral": {
    "enabled": true,                 # false = old behavior, every tool fully loaded
    "always_load": [...],            # tool names never deferred (replaces the default list)
    "providers": ["anthropic"]       # providers that support defer_loading
  }
Only the anthropic provider supports it today (tool_reference blocks are an
Anthropic Messages API feature); other providers get the full list as before.
"""
from __future__ import annotations

import re

SEARCH_TOOL_NAME = "tool_search"

# Claude Code keeps its everyday tools resident and defers the rest. These are
# openflip's everyday tools: files, shell, messaging, memory, web, and the
# chain/turn controls the framework relies on. The Anthropic docs recommend
# keeping the 3-5 most-used tools (plus the search tool) non-deferred.
DEFAULT_ALWAYS_LOAD = (
    SEARCH_TOOL_NAME,
    "read_file", "write_file", "edit_file", "list_files",
    "run_command", "web_search", "fetch_url",
    "send_message", "talk_to_agent", "end_chain",
    "save_memory", "search_memory", "read_memory",
)


def _cfg() -> dict:
    try:
        from .config_global import get_config
        c = get_config().get("tool_deferral")
        return c if isinstance(c, dict) else {}
    except Exception:
        return {}


def deferral_enabled(provider: str) -> bool:
    c = _cfg()
    if c.get("enabled") is False:
        return False
    providers = c.get("providers") or ["anthropic"]
    return (provider or "") in providers


def always_load_names() -> set[str]:
    c = _cfg()
    names = c.get("always_load")
    base = set(names) if isinstance(names, list) and names else set(DEFAULT_ALWAYS_LOAD)
    base.add(SEARCH_TOOL_NAME)   # the search tool itself must never be deferred
    return base


def deferred_names(tool_names: list[str]) -> list[str]:
    keep = always_load_names()
    return [n for n in tool_names if n not in keep]


def names_block(tool_names: list[str]) -> str:
    """The system-prompt section listing deferred tools by name (stable bytes:
    sorted, and only changes when the tool list or always-load config does)."""
    deferred = sorted(deferred_names(tool_names))
    if not deferred:
        return ""
    return (
        "# Deferred tools\n\n"
        "These tools are available, but only their names are loaded. Before calling one, "
        f"load its full definition with {SEARCH_TOOL_NAME}(query=\"select:<name>[,<name>...]\"), "
        "batching every tool you expect to need into one call. Once loaded, a tool stays "
        "callable for the rest of the conversation.\n\n" + "\n".join(deferred)
    )


# Session-scoped: set by the provider for the request being built, read by the
# search tool. Kept module-level per process (tools run in the same process).
_current_deferred: list[str] = []
_current_descriptions: dict[str, str] = {}


def set_current_catalog(deferred: list[str], descriptions: dict[str, str]) -> None:
    global _current_deferred, _current_descriptions
    _current_deferred = list(deferred)
    _current_descriptions = dict(descriptions)


def _catalog() -> tuple[list[str], dict[str, str]]:
    if _current_deferred:
        return _current_deferred, _current_descriptions
    # Fallback: the whole registry minus the always-load set.
    from .tools._base import TOOL_REGISTRY
    names = deferred_names(list(TOOL_REGISTRY.keys()))
    return names, {n: TOOL_REGISTRY[n].description for n in names}


def search_deferred(query: str, max_results: int = 5) -> tuple[list[str], str]:
    """Return (matching deferred tool names, note for the model)."""
    names, desc = _catalog()
    q = (query or "").strip()
    if not q:
        return [], "Empty query. Use \"select:<name>\" or keywords."
    if q.lower().startswith("select:"):
        wanted = [w.strip() for w in q[len("select:"):].split(",") if w.strip()]
        found = [w for w in wanted if w in names]
        missing = [w for w in wanted if w not in names]
        note = ""
        if missing:
            keep = always_load_names()
            loaded = [m for m in missing if m in keep]
            unknown = [m for m in missing if m not in keep]
            parts = []
            if loaded:
                parts.append(f"Already loaded (call directly): {', '.join(loaded)}.")
            if unknown:
                parts.append(f"No deferred tool named: {', '.join(unknown)}.")
            note = " ".join(parts)
        return found, note
    max_results = max(1, min(int(max_results or 5), 50))
    words = q.lower().split()
    required = [w[1:] for w in words if w.startswith("+") and len(w) > 1]
    terms = [w.lstrip("+") for w in words if w.lstrip("+")]
    scored = []
    for n in names:
        lname = n.lower()
        if any(r not in lname for r in required):
            continue
        text = f"{lname} {lname.replace('_', ' ')} {(desc.get(n) or '').lower()}"
        score = 0
        for t in terms:
            if t in lname:
                score += 3
            score += len(re.findall(re.escape(t), text))
        if score:
            scored.append((score, n))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, n in scored[:max_results]], ""
