"""Per-agent runtime state, written to `agents/<id>/live.json` on turn boundaries.

Renamed from the previous `state.json` because the project-root file
`agent_state.json` (which tracks enabled/disabled flags) collided with this
filename in conversation. Distinct names: `live.json` for *current activity*,
`agent_state.json` for *enabled flags*. Migration is automatic on first read.

Lightweight observability layer for things like the WebApp openflip tab.
The agent doesn't read this file — only external observers do. State is
advisory; nothing in openflip itself depends on it.

Design:
- Atomic writes via os.replace through utils.save_json.
- One state.json per agent, lives at agents/<id>/state.json.
- Updated on turn start, tool start/end, and turn end.
- Last-known channel is sticky — we don't clear it when the turn ends.

Schema:
{
  "agent_id": "<agent>",
  "status": "idle | active | talking | working | offline",
  "activity": "short human-readable string",
  "current_channel_id": 1234567890,        // last channel the agent worked in
  "current_channel_name": "#general",
  "current_tool": "web_search" | null,    // null when not in a tool call
  "is_in_turn": false,                    // true while a turn is being processed
  "last_active_ms": 1746556800000,        // most recent activity timestamp
  "last_tool_call_ms": 1746556800000,     // most recent tool call timestamp
  "updated_ms": 1746556800000             // when this file was last written
}

Status rules:
- `talking`  — currently in an active turn (is_in_turn=true)
- `working`  — currently executing a tool (current_tool != null AND is_in_turn=true)
- `active`   — last activity within ~5 min, no current turn
- `idle`     — nothing recent
- `offline`  — set externally if needed; the runtime never sets this itself
"""
from __future__ import annotations
import os
import time
from typing import Optional

from .utils import load_json, save_json, project_root, print_ts, COLOR_RED, COLOR_END


def _state_path(agent_id: str) -> str:
    return os.path.join(project_root(), "agents", agent_id, "live.json")


def _legacy_state_path(agent_id: str) -> str:
    """Pre-rename location. Migrated automatically on first read."""
    return os.path.join(project_root(), "agents", agent_id, "state.json")


def _migrate_if_legacy(agent_id: str) -> None:
    """One-time rename from `state.json` to `live.json`.

    Idempotent: skips if the new file already exists or the legacy doesn't.
    """
    legacy = _legacy_state_path(agent_id)
    new = _state_path(agent_id)
    if os.path.isfile(new) or not os.path.isfile(legacy):
        return
    try:
        os.replace(legacy, new)
    except OSError:
        pass


def _now_ms() -> int:
    return int(time.time() * 1000)


# Per-agent state lives in-memory and only flushes to disk on turn boundaries
# and offline transitions. Without this, every turn paid 4+ JSON read+write
# cycles (turn-start, tool-start ×N, tool-end ×N, turn-end) — small files but
# hot-path I/O. Observers that read state.json from disk poll on their own
# schedule; subsecond freshness inside a turn isn't required.
_CACHE: dict[str, dict] = {}


def _read(agent_id: str) -> dict:
    cached = _CACHE.get(agent_id)
    if cached is not None:
        return cached
    _migrate_if_legacy(agent_id)
    state = load_json(_state_path(agent_id), default={"agent_id": agent_id})
    _CACHE[agent_id] = state
    return state


def _flush(agent_id: str) -> None:
    """Write the cached state to disk. Called only on turn/offline boundaries."""
    state = _CACHE.get(agent_id)
    if state is None:
        return
    state["agent_id"] = agent_id
    state["updated_ms"] = _now_ms()
    try:
        save_json(_state_path(agent_id), state)
    except Exception as e:
        print_ts(f"{COLOR_RED}agent_state: failed to write {agent_id}: {e}{COLOR_END}", error=True, agent=agent_id)


def _write(agent_id: str, state: dict) -> None:
    """Update cache only — caller decides when to flush via _flush()."""
    state["agent_id"] = agent_id
    state["updated_ms"] = _now_ms()
    _CACHE[agent_id] = state


def on_turn_start(agent_id: str, channel, *, talking_with: str = "") -> None:
    """Mark the agent as in a turn. Channel is a Discord channel object.

    `talking_with` is the id of the human or other agent driving this turn:
    - "owner" (or another human user id) when the turn was triggered by a
      human Discord message
    - "<agent_id>" when the turn was triggered by another agent via
      talk_to_agent (synthetic turn)
    - "" when the turn was triggered by cron / heartbeat / restart sentinel
    The WebApp openflip tab uses this to render inter-agent chain
    lines on the canvas.
    """
    state = _read(agent_id)
    state["status"] = "talking"
    state["is_in_turn"] = True
    state["current_channel_id"] = int(getattr(channel, "id", 0) or 0)
    name = getattr(channel, "name", None)
    state["current_channel_name"] = f"#{name}" if name else "DM"
    state["current_tool"] = None
    state["last_active_ms"] = _now_ms()
    state["activity"] = f"Active in {state['current_channel_name']}"
    state["talking_with"] = talking_with or ""
    # activity_until_ms — the 30-minute stickiness window from agent_house.md.
    # The activity (current state) persists in observers' eyes until this
    # timestamp, even if the agent has been technically idle since. New
    # turn starts reset the window. Half-hour default matches the design doc.
    state["activity_until_ms"] = _now_ms() + 30 * 60 * 1000
    _write(agent_id, state)
    # Flush to disk on turn start as well as on turn end. Without this,
    # live.json always reflects the *last completed turn's* end state,
    # never the active mid-turn state — which means restart_gateway's
    # preflight check (which reads live.json from disk) can't tell that
    # another agent is currently working and may proceed with a restart
    # that kills the mid-turn agent. Confirmed in prod: a restart hit an
    # agent mid-turn because the live.json on disk still said
    # is_in_turn=false from the prior turn minutes earlier.
    # One extra small JSON write per turn boundary is negligible cost
    # for correctness.
    _flush(agent_id)


def on_tool_start(agent_id: str, tool_name: str) -> None:
    state = _read(agent_id)
    state["current_tool"] = tool_name
    state["status"] = "working"
    state["last_tool_call_ms"] = _now_ms()
    state["last_active_ms"] = state["last_tool_call_ms"]
    state["activity"] = f"Running {tool_name}"
    _write(agent_id, state)


def on_tool_end(agent_id: str) -> None:
    state = _read(agent_id)
    state["current_tool"] = None
    # If still in a turn, go back to talking; otherwise active.
    if state.get("is_in_turn"):
        state["status"] = "talking"
        state["activity"] = f"Active in {state.get('current_channel_name', '')}"
    else:
        state["status"] = "active"
        state["activity"] = "Just finished a task"
    _write(agent_id, state)


def on_turn_end(agent_id: str) -> None:
    state = _read(agent_id)
    state["is_in_turn"] = False
    state["current_tool"] = None
    state["status"] = "active"
    state["last_active_ms"] = _now_ms()
    state["activity"] = "Just finished a turn"
    # Clear talking_with on turn end — they're not actively talking with
    # anyone once the turn closes. The activity stickiness window
    # (activity_until_ms set at on_turn_start) is what observers use to
    # decide whether to still render the "recently active" state.
    state["talking_with"] = ""
    _write(agent_id, state)
    # Turn boundary: now is when external observers (e.g. the WebApp
    # openflip tab) want a fresh snapshot. Mid-turn state changes stayed in
    # memory only.
    _flush(agent_id)


def set_offline(agent_id: str) -> None:
    """Mark agent as offline (e.g. on shutdown)."""
    state = _read(agent_id)
    state["status"] = "offline"
    state["is_in_turn"] = False
    state["current_tool"] = None
    state["activity"] = "Offline"
    _write(agent_id, state)
    _flush(agent_id)
