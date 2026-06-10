"""Persisted state: enabled/disabled flags per agent."""
from __future__ import annotations
import os
from .utils import load_json, save_json, project_root


def _state_path() -> str:
    return os.path.join(project_root(), 'agent_state.json')


def load_state() -> dict:
    return load_json(_state_path(), default={})


def save_state(state: dict) -> bool:
    # 0o600 — agent_state.json sits next to api_config.json and gets rewritten
    # whenever an agent is enabled/disabled. Without an explicit mode the
    # tmp+rename in save_json would re-widen perms to the process umask.
    return save_json(_state_path(), state, mode=0o600)


def is_enabled(agent_id: str, *, default: bool = True) -> bool:
    s = load_state().get(agent_id) or {}
    return bool(s.get("enabled", default))


def set_enabled(agent_id: str, enabled: bool) -> bool:
    s = load_state()
    entry = s.get(agent_id) or {}
    entry["enabled"] = bool(enabled)
    s[agent_id] = entry
    return save_state(s)
