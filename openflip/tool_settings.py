"""Owner-controlled per-tool settings.

Each tool registers a schema at import time describing the parameters the OWNER
can tune via /toolset. Tools read current values via `get(tool, key)`. The AI
never sees these — it sees the values in its system message (so it can answer
"what model are you using?") but cannot change them.

Persisted to data/tool_settings.json. Atomic writes via utils.save_json.

Validators run at /toolset time, never at gen time. A bad value rejects
immediately — the AI is never affected by config errors.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .utils import load_json, save_json, resolve_path
from .config_global import get_config


@dataclass
class SettingSchema:
    key: str
    type: str  # "str" | "int" | "float" | "bool" | "choice"
    default: Any
    description: str
    choices: Optional[list[Any]] = None  # for type="choice"
    min: Optional[float] = None  # for int/float
    max: Optional[float] = None  # for int/float
    validator: Optional[Callable[[Any], Optional[str]]] = None  # returns None if ok, error string if not


@dataclass
class ToolSchema:
    tool_name: str
    settings: dict[str, SettingSchema] = field(default_factory=dict)


_SCHEMAS: dict[str, ToolSchema] = {}
_VALUES: dict[str, dict[str, Any]] = {}
_LOADED = False


def _settings_path() -> str:
    data_dir = get_config().get("data_dir", "./data")
    return os.path.join(resolve_path(data_dir), "tool_settings.json")


def register(tool_name: str, schema_entries: list[SettingSchema]) -> None:
    """Called by tool modules at import time. Idempotent — re-registering replaces."""
    schema = ToolSchema(tool_name=tool_name)
    for s in schema_entries:
        schema.settings[s.key] = s
    _SCHEMAS[tool_name] = schema


def _ensure_loaded() -> None:
    global _LOADED
    if _LOADED:
        return
    raw = load_json(_settings_path(), default={})
    if isinstance(raw, dict):
        for tool_name, kvs in raw.items():
            if isinstance(kvs, dict):
                _VALUES[tool_name] = dict(kvs)
    _LOADED = True


def _persist() -> bool:
    return save_json(_settings_path(), _VALUES)


def get(tool_name: str, key: str, fallback: Any = None) -> Any:
    """Read a setting. Returns the saved value, else the schema default, else fallback."""
    _ensure_loaded()
    if tool_name in _VALUES and key in _VALUES[tool_name]:
        return _VALUES[tool_name][key]
    schema = _SCHEMAS.get(tool_name)
    if schema and key in schema.settings:
        return schema.settings[key].default
    return fallback


def get_all(tool_name: str) -> dict[str, Any]:
    """Return the full effective settings for a tool (saved overlaid on defaults)."""
    _ensure_loaded()
    out: dict[str, Any] = {}
    schema = _SCHEMAS.get(tool_name)
    if schema:
        for key, s in schema.settings.items():
            out[key] = s.default
    if tool_name in _VALUES:
        out.update(_VALUES[tool_name])
    return out


def list_tools() -> list[str]:
    return sorted(_SCHEMAS.keys())


def get_schema(tool_name: str) -> Optional[ToolSchema]:
    return _SCHEMAS.get(tool_name)


def coerce_and_validate(tool_name: str, key: str, raw_value: str) -> tuple[Optional[Any], Optional[str]]:
    """Convert a string from a slash command to the right type and validate.
    Returns (value, None) on success or (None, error_message) on failure."""
    schema = _SCHEMAS.get(tool_name)
    if not schema:
        return None, f"Unknown tool: {tool_name}"
    s = schema.settings.get(key)
    if not s:
        return None, f"Unknown setting '{key}' for {tool_name}"

    value: Any
    try:
        if s.type == "str":
            value = str(raw_value)
        elif s.type == "int":
            value = int(raw_value)
        elif s.type == "float":
            value = float(raw_value)
        elif s.type == "bool":
            lowered = str(raw_value).strip().lower()
            if lowered in ("true", "yes", "y", "1", "on"):
                value = True
            elif lowered in ("false", "no", "n", "0", "off"):
                value = False
            else:
                return None, f"Expected true/false, got '{raw_value}'"
        elif s.type == "choice":
            value = str(raw_value)
            if s.choices and value not in s.choices:
                return None, f"Must be one of: {', '.join(map(str, s.choices))}"
        else:
            return None, f"Unsupported setting type '{s.type}'"
    except (TypeError, ValueError) as e:
        return None, f"Invalid {s.type}: {e}"

    if s.type in ("int", "float"):
        if s.min is not None and value < s.min:
            return None, f"Must be >= {s.min}"
        if s.max is not None and value > s.max:
            return None, f"Must be <= {s.max}"

    if s.validator:
        err = s.validator(value)
        if err:
            return None, err

    return value, None


def set_value(tool_name: str, key: str, raw_value: str) -> tuple[bool, str]:
    """Validate + persist a single setting. Returns (ok, message)."""
    _ensure_loaded()
    value, err = coerce_and_validate(tool_name, key, raw_value)
    if err:
        return False, err
    _VALUES.setdefault(tool_name, {})[key] = value
    if not _persist():
        return False, "Failed to write tool_settings.json"
    return True, f"{tool_name}.{key} = {value}"


def reset_tool(tool_name: str) -> tuple[bool, str]:
    """Drop all overrides for one tool (revert to schema defaults)."""
    _ensure_loaded()
    if tool_name not in _SCHEMAS:
        return False, f"Unknown tool: {tool_name}"
    _VALUES.pop(tool_name, None)
    if not _persist():
        return False, "Failed to write tool_settings.json"
    return True, f"Reset {tool_name} to defaults"


def reset_all() -> tuple[bool, str]:
    _ensure_loaded()
    _VALUES.clear()
    if not _persist():
        return False, "Failed to write tool_settings.json"
    return True, "All tool settings reset to defaults"


def render_summary_for_ai(only: Optional[set[str]] = None) -> str:
    """Per-turn system extension content. Lists current settings for the tools
    available to the speaker, so the AI can answer 'what model are you using?'
    without being able to change anything.

    Args:
        only: If provided, restricts the output to settings for these tool
              names. Pass the speaker's currently-callable tool names so a
              tool-less agent (or a speaker with restricted ACL) doesn't
              receive a wall of irrelevant config in its system message.
              If None, returns settings for every registered tool (CLI / fallback).

    Returns the empty string when there are no relevant tools — caller can
    skip appending it without needing a separate guard.
    """
    _ensure_loaded()
    if not _SCHEMAS:
        return ""
    relevant = sorted(
        name for name in _SCHEMAS.keys()
        if only is None or name in only
    )
    if not relevant:
        return ""
    lines = ["Current tool configuration (set by the owner — you cannot change these):"]
    for tool_name in relevant:
        values = get_all(tool_name)
        if not values:
            continue
        kvs = ", ".join(f"{k}={v}" for k, v in values.items())
        lines.append(f"- {tool_name}: {kvs}")
    lines.append("If a user wants any of these changed, tell them to ask the owner to use /toolset <tool> <key> <value>.")
    return "\n".join(lines)


def render_summary_for_command(tool_name: Optional[str] = None) -> str:
    """Human-readable text for /toolset slash command output."""
    _ensure_loaded()
    if tool_name:
        schema = _SCHEMAS.get(tool_name)
        if not schema:
            return f"Unknown tool: {tool_name}"
        values = get_all(tool_name)
        lines = [f"**{tool_name}**"]
        for key, s in schema.settings.items():
            current = values.get(key, s.default)
            constraint = ""
            if s.type == "choice" and s.choices:
                constraint = f" [{', '.join(map(str, s.choices))}]"
            elif s.type in ("int", "float") and (s.min is not None or s.max is not None):
                lo = s.min if s.min is not None else "-∞"
                hi = s.max if s.max is not None else "∞"
                constraint = f" [{lo}..{hi}]"
            lines.append(f"  • `{key}` = `{current}` ({s.type}{constraint}) — {s.description}")
        return "\n".join(lines)
    out = []
    for name in sorted(_SCHEMAS.keys()):
        out.append(render_summary_for_command(name))
    return "\n\n".join(out) if out else "No tools registered."


# ---------------------------------------------------------------------------
# spawn_subagents (openflip/tools/subagent.py) — owner-controlled worker policy.
#
# Registered here (not in the tool module) because the hard-deny set below is
# shared by the /toolset-time validator AND the tool's call-time grant
# computation, and the tool module imports this one. The settings are the
# CEILING for every worker an orchestrating agent spawns; the tool only ever
# narrows them per task.
# ---------------------------------------------------------------------------

# Tools a worker can NEVER be granted, whatever `allowed_tools` says. Workers
# are ephemeral, unattended and non-owner: nothing that mutates state outside
# its own context, reaches a human/agent, re-enters the agent loop, or spawns
# further work is allowed. Enforced in code at call time (never by prompt).
SUBAGENT_DENIED_TOOLS: frozenset[str] = frozenset({
    # recursion / control plane
    "spawn_subagents", "restart_gateway", "restart_flask_app", "claude_code",
    "force_reply", "resolve_approval", "request_approval",
    # anything that talks to a human or another agent
    "talk_to_agent", "send_message", "send_file", "inject_context",
    "delete_message", "discord_post", "discord_manage", "flip_send", "flip_voice",
    # cron
    "add_cron_job", "cancel_cron_job", "list_cron_jobs",
    # shell + file writes (host and sandbox)
    "run_command", "run_command_sandbox", "write_file", "write_file_sandbox",
    "edit_file", "delete_file", "restore_snapshot",
    # memory WRITES (reads stay allowed)
    "save_memory", "update_core_memory", "delete_memory", "dream", "reindex_memory",
})

SUBAGENT_DEFAULT_TOOLS = "web_search,fetch_url,read_file,list_files,search_memory,read_memory"


def parse_tool_list(raw: Any) -> list[str]:
    """Split a comma/whitespace-separated tool-name string into a deduped list
    (order preserved). Accepts a list too (ingress / tests)."""
    if isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
    else:
        items = str(raw or "").replace(",", " ").split()
    out: list[str] = []
    for name in items:
        name = name.strip()
        if name and name not in out:
            out.append(name)
    return out


def _validate_subagent_tools(value: Any) -> Optional[str]:
    names = parse_tool_list(value)
    if not names:
        return "allowed_tools must name at least one tool"
    denied = [n for n in names if n in SUBAGENT_DENIED_TOOLS]
    if denied:
        return f"always denied for subagent workers: {', '.join(denied)}"
    try:
        from .tools._base import TOOL_REGISTRY
    except Exception:
        return None  # registry unavailable (CLI) — skip the existence check
    unknown = [n for n in names if n not in TOOL_REGISTRY]
    if unknown:
        return f"unknown tool(s): {', '.join(unknown)}"
    return None


def _validate_worker_model(value: Any) -> Optional[str]:
    """Owner-time sanity only: the name must be a known config.json model.
    Provider compatibility with the CALLING agent is re-checked at call time
    by the tool (an ollama agent can't run a claude worker and vice versa)."""
    name = str(value or "").strip()
    if not name:
        return "worker_model must be a non-empty model name"
    bare = name.split("/", 1)[1] if "/" in name else name
    models = get_config().get("models") or {}
    if models and bare not in models:
        return f"`{bare}` is not in config.json `models` (known: {', '.join(sorted(models))})"
    return None


register("spawn_subagents", [
    SettingSchema("worker_model", "str", "claude-sonnet-5-5",
        "Default model for spawned workers (a task may name its own). Must be a "
        "config.json `models` entry on the calling agent's provider.",
        validator=_validate_worker_model),
    SettingSchema("worker_effort", "choice", "medium",
        "Reasoning effort applied to every worker turn (ignored by providers without an effort knob).",
        choices=["minimal", "low", "medium", "high", "xhigh", "max"]),
    SettingSchema("allowed_tools", "str", SUBAGENT_DEFAULT_TOOLS,
        "Comma-separated CEILING of tools a worker may be granted. A task's "
        "requested tools are intersected with this; the built-in deny list "
        "(spawn/restart/messaging/cron/shell/file+memory writes) always applies.",
        validator=_validate_subagent_tools),
    SettingSchema("max_parallel", "int", 4,
        "Maximum tasks in ONE spawn_subagents call (the call is rejected above this).",
        min=1, max=16),
    SettingSchema("max_inflight_per_agent", "int", 6,
        "Maximum workers running at once per agent across concurrent calls; "
        "tasks beyond the free slots fail immediately instead of queueing.",
        min=1, max=32),
    SettingSchema("timeout_s", "int", 300,
        "Per-task wall-clock limit in seconds; a worker still running past it is hard-interrupted.",
        min=10, max=3600),
])
