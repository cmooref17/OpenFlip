"""Data layer: reads openflip's filesystem state. Everything here is
synchronous file I/O — Quart routes wrap these in `asyncio.to_thread`
where contention matters. We don't import any openflip modules to keep
this app cleanly decoupled."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (
    OPENFLIP_AGENTS_DIR,
    OPENFLIP_AGENT_STATE_JSON,
    OPENFLIP_CONFIG_JSON,
    OPENFLIP_TOOL_SETTINGS,
)
from .._constants import DANGEROUS_TOOL_NAMES
# Conversation filename codec — on Windows the on-disk stem encodes ":" as
# "%3A" (NTFS forbids colons); ids keep the colon form in memory and URLs.
# Imported from the framework so the encoding has ONE source of truth.
from .._conversation_io import fs_encode as _fs_encode, fs_decode as _fs_decode


# ---------- agent discovery ----------

def list_agents() -> List[Dict[str, Any]]:
    """Return every agent under agents/<id>/agent.json. Agents whose
    directory name starts with `_` are skipped."""
    out: List[Dict[str, Any]] = []
    if not OPENFLIP_AGENTS_DIR.is_dir():
        return out
    state = _load_agent_state()
    for entry in sorted(OPENFLIP_AGENTS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("_"):
            continue
        cfg_path = entry / "agent.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        agent_id = cfg.get("id") or entry.name
        out.append({
            "id": agent_id,
            "dir": str(entry),
            "display_name": cfg.get("display_name") or agent_id,
            "model": cfg.get("model") or "?",
            "provider": cfg.get("provider") or "ollama",
            "enabled": state.get(agent_id, {}).get("enabled", True),
            "config_mtime": cfg_path.stat().st_mtime,
        })
    return out


def get_agent(agent_id: str) -> Optional[Dict[str, Any]]:
    cfg_path = OPENFLIP_AGENTS_DIR / agent_id / "agent.json"
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception:
        return None
    agent_dir = OPENFLIP_AGENTS_DIR / agent_id
    state = _load_agent_state().get(agent_id, {})
    sys_files = []
    for fname in cfg.get("system_files", []):
        if fname.startswith("_shared/"):
            path = OPENFLIP_AGENTS_DIR / fname
        else:
            path = agent_dir / fname
        sys_files.append({
            "name": fname,
            "path": str(path),
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
        })
    return {
        "id": cfg.get("id") or agent_id,
        "dir": str(agent_dir),
        "config": cfg,
        "config_path": str(cfg_path),
        "config_mtime": cfg_path.stat().st_mtime,
        "system_files": sys_files,
        "enabled": state.get("enabled", True),
        "conversations": _list_conversations(agent_dir),
    }


def _list_conversations(agent_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    conv_dir = agent_dir / "conversations"
    if not conv_dir.is_dir():
        return out
    for f in sorted(conv_dir.iterdir()):
        if not f.is_file():
            continue
        if not f.name.endswith(".jsonl"):
            continue
        if ".bak" in f.name or ".compaction_" in f.name:
            continue
        # Single source of truth: full conversation_id ("discord:<id>"),
        # matching the on-disk filename. Stripping the prefix was the source
        # of repeated display bugs (page header lost it, urls lost it, etc).
        # Use it everywhere; URL path looks like /agents/<id>/conversations/discord:<id>.
        ch_id = _fs_decode(f.stem)  # full id, e.g. "discord:1505657385492943008"
        display = ch_id  # alias for template compatibility
        try:
            with f.open("rb") as fh:
                msg_count = sum(1 for _ in fh)
        except Exception:
            msg_count = 0
        out.append({
            "channel_id": ch_id,
            "display": display,
            "file": str(f),
            "size": f.stat().st_size,
            "mtime": f.stat().st_mtime,
            "msg_count": msg_count,
        })
    out.sort(key=lambda c: c["mtime"], reverse=True)
    return out


def read_conversation(agent_id: str, channel_id: str,
                       limit: Optional[int] = None,
                       offset: int = 0) -> List[Dict[str, Any]]:
    """Read a conversation jsonl. Returns list of {role, content, ts, ...}.
    limit+offset slice from the END (newest).

    Accepts both forms of channel_id: full ("discord:1505...") OR bare
    ("1505..."). Full form is now canonical; bare form preserved for
    backward-compat with any old bookmarks."""
    # `channel_id` is the full transport-prefixed conversation id from the
    # session (e.g. "discord:1234", "imessage:5678"). Use it verbatim as
    # the filename stem. The previous version stripped "discord:" and put
    # it back unconditionally, which broke iMessage / any other transport
    # whose conversations live under a different prefix.
    try:
        f = OPENFLIP_AGENTS_DIR / agent_id / "conversations" / f"{_fs_encode(channel_id)}.jsonl"
    except ValueError:
        # fs_encode fail-closes on empty/traversing/control-char ids — a
        # malformed URL param reads as "no such conversation", never a path.
        return []
    if not f.exists():
        # Back-compat: callers that pass just the bare numeric id (no
        # prefix) get the legacy Discord-only lookup. Drop this branch
        # once all callers are confirmed prefix-aware.
        if ":" not in channel_id:
            alt_discord = OPENFLIP_AGENTS_DIR / agent_id / "conversations" / f"{_fs_encode(f'discord:{channel_id}')}.jsonl"
            if alt_discord.exists():
                f = alt_discord
            else:
                alt_bare = OPENFLIP_AGENTS_DIR / agent_id / "conversations" / f"{channel_id}.jsonl"
                if not alt_bare.exists():
                    return []
                f = alt_bare
        else:
            return []
    rows: List[Dict[str, Any]] = []
    try:
        with f.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        return []
    if limit is None:
        return rows
    end = len(rows) - offset
    start = max(0, end - limit)
    return rows[start:end]


def _resolve_system_file_path(agent_id: str, file_name: str) -> Optional[Path]:
    """Resolve a system-file path with strict containment checks.

    Prevents path-traversal exploits via URL params like `../../config.json`.
    The resolved path MUST live under OPENFLIP_AGENTS_DIR (no parent escapes,
    no symlink jumps out). Returns None on any violation.

    Also rejects suspicious filename components up-front so the audit log
    doesn't have to wade through resolved-path comparisons to spot abuse.
    """
    # Reject obvious traversal attempts in the raw input — fast-fail before
    # touching the filesystem. Backslash is a path separator on Windows (and
    # never legitimate in a system-file name), so reject it outright.
    if not file_name or "\\" in file_name or ".." in file_name.split("/"):
        return None
    if file_name.startswith("_shared/"):
        candidate = OPENFLIP_AGENTS_DIR / file_name
    else:
        # Don't allow agent_id with path separators either.
        if "/" in agent_id or "\\" in agent_id or ".." in agent_id or not agent_id:
            return None
        candidate = OPENFLIP_AGENTS_DIR / agent_id / file_name
    try:
        resolved = candidate.resolve(strict=False)
        agents_resolved = OPENFLIP_AGENTS_DIR.resolve(strict=False)
    except Exception:
        return None
    # Strict containment — resolved path must be inside agents dir, period.
    try:
        resolved.relative_to(agents_resolved)
    except ValueError:
        return None
    return resolved


def read_system_file(agent_id: str, file_name: str) -> Optional[str]:
    path = _resolve_system_file_path(agent_id, file_name)
    if path is None or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def write_system_file(agent_id: str, file_name: str, content: str) -> bool:
    path = _resolve_system_file_path(agent_id, file_name)
    if path is None or not path.parent.exists():
        return False
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


# The web config editor must never be able to ADD a dangerous tool to an agent
# that doesn't already have one. The denylist is the framework-wide canonical
# set (openflip/_constants.py), shared with the trigger endpoint so the two
# untrusted-grant paths can't drift apart. Local alias kept for readability.
_DANGEROUS_TOOL_NAMES = DANGEROUS_TOOL_NAMES


def _tool_names(allowed_tools) -> set[str]:
    """Extract tool names from an allowed_tools list (strings or {name:...})."""
    names: set[str] = set()
    for t in allowed_tools or []:
        if isinstance(t, str):
            names.add(t)
        elif isinstance(t, dict) and isinstance(t.get("name"), str):
            names.add(t["name"])
    return names


def _path_acl_strings(val) -> list:
    """Every path string in a path-ACL field, flattening the transport-keyed
    dict form so a `"*"` buried inside any `users`/`all_users` list is still
    seen by the widen guard. Mirrors how `tools/files.py:_effective_allowed`
    reads the dict: per transport block, the `all_users` list plus every list
    under `users`. A flat list is returned as-is.
    """
    if isinstance(val, list):
        return [p for p in val if isinstance(p, str)]
    out: list = []
    if isinstance(val, dict):
        for block in val.values():
            if not isinstance(block, dict):
                continue
            au = block.get("all_users", [])
            if isinstance(au, (list, tuple)):
                out.extend(p for p in au if isinstance(p, str))
            users = block.get("users", {})
            if isinstance(users, dict):
                for plist in users.values():
                    if isinstance(plist, (list, tuple)):
                        out.extend(p for p in plist if isinstance(p, str))
    return out


def validate_agent_config(agent_id: str, cfg: Dict[str, Any]) -> Optional[str]:
    """Validate an incoming agent.json before it's written via the web editor.

    Returns an error string if the config must be REJECTED, else None.

    Two classes of check:
      1. Structural — must be a dict; if present, `allowed_tools` /
         `denied_paths` / `system_files` must be lists; `channels` a dict;
         `allowed_read_paths` / `allowed_write_paths` may be EITHER a flat list
         OR the transport-keyed dict form agent.py from_file accepts. Each
         `allowed_tools` entry must be an object with a string `name` (mirrors
         agent.py:_parse_tool_entry — bare strings are rejected). A malformed
         shape that would brick the agent on hot-reload is refused here.
      2. Privilege non-escalation — relative to the CURRENT on-disk config, the
         web editor may not widen security-sensitive fields:
           * cannot introduce `"*"` into allowed_read/write_paths unless it was
             already there,
           * cannot ADD a dangerous tool (run_command/claude_code/restart_*)
             that the agent didn't already have.
         These are exactly the mutations that turn an authed-but-not-shell web
         session into RCE / arbitrary-FS-write.
    """
    if not isinstance(cfg, dict):
        return "config must be a JSON object"

    list_fields = (
        "allowed_tools", "denied_paths", "system_files",
    )
    for f in list_fields:
        if f in cfg and not isinstance(cfg[f], list):
            return f"`{f}` must be a list"
    # Path ACLs accept EITHER a flat list (applies to everyone — historical
    # form) OR the transport-keyed dict form (per-user scope) that agent.py
    # from_file stores verbatim. denied_paths stays list-only (flat,
    # unconditional). Mirror from_file: the only structural rule is
    # list-or-dict; per-user resolution is validated at access time.
    for pf in ("allowed_read_paths", "allowed_write_paths"):
        if pf in cfg and not isinstance(cfg[pf], (list, dict)):
            return f"`{pf}` must be a list or a transport-keyed object"
    if "channels" in cfg and not isinstance(cfg["channels"], dict):
        return "`channels` must be an object"

    # Reject EXACTLY what the canonical loader rejects by dry-running its own
    # per-entry parser. agent.py:from_file calls `_parse_tool_entry` on every
    # allowed_tools element and lets any error propagate — bare strings,
    # non-object entries, a missing string `name`, a non-object `auth`,
    # non-string transport keys, a non-object `exclude`, non-int-coercible
    # roles/channels all raise and brick the agent on the next hot-reload.
    # Re-running the real parser (instead of hand-mirroring a subset) keeps
    # the web editor in lock-step with the loader with zero drift: whatever
    # from_file refuses, this refuses, with from_file's own message.
    from ..agent import _parse_tool_entry
    for entry in (cfg.get("allowed_tools") or []):
        try:
            _parse_tool_entry(entry, agent_id=agent_id)
        except Exception as e:
            return f"invalid allowed_tools entry: {e}"

    current = get_agent(agent_id) or {}

    # Anti-escalation: dict-aware so a `"*"` buried inside a transport-keyed
    # path ACL (`{"discord": {"all_users": ["*"]}}`) is caught, not just a
    # top-level `"*"` in a flat list. `"*" in dict` would test KEYS, missing it.
    for pf in ("allowed_read_paths", "allowed_write_paths"):
        new_has_wild = "*" in _path_acl_strings(cfg.get(pf))
        old_has_wild = "*" in _path_acl_strings(current.get(pf))
        if new_has_wild and not old_has_wild:
            return (
                f"refusing to widen `{pf}` to '*' via the web editor "
                f"(edit agent.json directly if this is intended)"
            )

    new_tools = _tool_names(cfg.get("allowed_tools"))
    old_tools = _tool_names(current.get("allowed_tools"))
    added_dangerous = (new_tools - old_tools) & _DANGEROUS_TOOL_NAMES
    if added_dangerous:
        return (
            f"refusing to grant dangerous tool(s) {sorted(added_dangerous)} "
            f"via the web editor (edit agent.json directly if intended)"
        )

    return None


def write_agent_config(agent_id: str, cfg: Dict[str, Any]) -> bool:
    """Atomic agent.json write. openflip's hot-reload picks up the new
    mtime on the next message — no restart needed.

    Callers should run `validate_agent_config` first; this function also
    refuses to write a config that fails validation as a backstop.
    """
    if validate_agent_config(agent_id, cfg) is not None:
        return False
    path = OPENFLIP_AGENTS_DIR / agent_id / "agent.json"
    if not path.parent.exists():
        return False
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


# ---------- global state ----------

def _load_agent_state() -> Dict[str, Any]:
    if not OPENFLIP_AGENT_STATE_JSON.exists():
        return {}
    try:
        return json.loads(OPENFLIP_AGENT_STATE_JSON.read_text())
    except Exception:
        return {}


def load_global_config() -> Dict[str, Any]:
    """Load config.json for display in the web /settings panel.

    SECURITY: redacts any `integrations.<x>.tokens` blocks before returning,
    so even if someone accidentally puts tokens back into config.json they
    won't get rendered to the browser. Tokens belong in api_config.json
    (which this function never reads).
    """
    if not OPENFLIP_CONFIG_JSON.exists():
        return {}
    try:
        cfg = json.loads(OPENFLIP_CONFIG_JSON.read_text())
    except Exception:
        return {}
    # Strip tokens defensively. Walks integrations.<integration>.tokens and
    # replaces values with a redacted placeholder so the page renders a
    # marker (operator sees "something was here, gone now") without leaking.
    try:
        integrations = cfg.get("integrations")
        if isinstance(integrations, dict):
            for name, entry in integrations.items():
                if isinstance(entry, dict) and isinstance(entry.get("tokens"), dict):
                    entry["tokens"] = {k: "[REDACTED — stored in api_config.json]"
                                       for k in entry["tokens"].keys()}
    except Exception:
        pass
    return cfg


def load_tool_settings() -> Dict[str, Any]:
    if not OPENFLIP_TOOL_SETTINGS.exists():
        return {}
    try:
        return json.loads(OPENFLIP_TOOL_SETTINGS.read_text())
    except Exception:
        return {}


def write_tool_settings(data: Dict[str, Any]) -> bool:
    OPENFLIP_TOOL_SETTINGS.parent.mkdir(parents=True, exist_ok=True)
    tmp = OPENFLIP_TOOL_SETTINGS.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, OPENFLIP_TOOL_SETTINGS)
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return False


# ---------- formatting helpers ----------

def fmt_ts(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "?"


def fmt_relative(ts: float) -> str:
    try:
        delta = datetime.now().timestamp() - ts
    except Exception:
        return "?"
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    return f"{int(delta / 86400)}d ago"


# ---------- memory ----------

def get_memory(agent_id: str) -> Dict[str, Any]:
    """Returns the agent's core memory + list of daily-log files.
    Does NOT load all daily contents — that's a separate call per file."""
    agent_dir = OPENFLIP_AGENTS_DIR / agent_id
    if not agent_dir.is_dir():
        return {"core": None, "daily": []}
    core_path = agent_dir / "MEMORY.md"
    core = core_path.read_text(encoding="utf-8") if core_path.exists() else None
    mem_dir = agent_dir / "memory"
    daily: List[Dict[str, Any]] = []
    if mem_dir.is_dir():
        for f in sorted(mem_dir.iterdir(), reverse=True):
            if f.is_file() and f.name.endswith(".md"):
                daily.append({
                    "date": f.stem,
                    "path": str(f),
                    "size": f.stat().st_size,
                    "mtime": f.stat().st_mtime,
                })
    return {
        "core": core,
        "core_path": str(core_path),
        "daily": daily,
    }


def read_daily_log(agent_id: str, date: str) -> Optional[str]:
    """date like 'YYYY-MM-DD'."""
    # Reject any path-traversal-shaped input
    if "/" in date or ".." in date:
        return None
    path = OPENFLIP_AGENTS_DIR / agent_id / "memory" / f"{date}.md"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None
