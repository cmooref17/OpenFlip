"""Framework-wide event ring buffer.

Append-only JSONL log of significant events across all agents, written to
`data/events.jsonl`. Capped size (default 10000 events) — rotation happens
in-place on next write past the cap.

Used by the WebApp openflip tab's Activity feed to surface what the
agent system is doing at the framework level — turn starts, tool calls,
inter-agent dispatches, restarts, memory writes. Things the operator can't see from
inside Discord because they happen between turns or across agents.

Design:
- Append-only JSONL. One event per line. Atomic write per append (small
  enough that the line either lands fully or not at all — Linux page-level
  atomicity for sub-4KB writes).
- Capped at MAX_EVENTS lines. When append would exceed, the file is
  rewritten without the oldest line. This is O(n) but n is small (10k)
  and events are tiny.
- No locking — single-writer-per-process is fine because all writers are
  in the same openflip process. The Flask side only reads.
- Event schema is open-ended. Required fields: ts_ms, agent_id, kind.
  Other fields are kind-specific.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from .utils import project_root, print_ts, COLOR_RED, COLOR_END


MAX_EVENTS = 10_000


def _events_path() -> str:
    return os.path.join(project_root(), "data", "events.jsonl")


def _now_ms() -> int:
    return int(time.time() * 1000)


def log_event(agent_id: str, kind: str, **fields: Any) -> None:
    """Append one event to the ring buffer.

    Args:
        agent_id: Which agent the event is about. "" for framework-wide
            events (restart, gateway online).
        kind: Event kind. Free-form string; common values include
            "turn_start", "turn_end", "tool_call", "tool_end",
            "talk_to_agent", "memory_write", "memory_search",
            "restart", "compaction", "framework_error".
        **fields: Additional kind-specific fields. Will be JSON-serialized;
            non-serializable values are coerced to str.
    """
    event = {
        "ts_ms": _now_ms(),
        "agent_id": agent_id or "",
        "kind": kind,
    }
    # Coerce values for JSON safety. ints/floats/strs/bools/None pass
    # through; anything else gets repr()'d.
    for k, v in fields.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            event[k] = v
        elif isinstance(v, (list, tuple, dict)):
            try:
                json.dumps(v)
                event[k] = v
            except (TypeError, ValueError):
                event[k] = repr(v)
        else:
            event[k] = repr(v)

    path = _events_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"

        # Check current line count. If we're past cap, rotate by rewriting
        # without the oldest line. Cheap check via wc-style line counting.
        line_count = 0
        if os.path.isfile(path):
            try:
                with open(path, "rb") as f:
                    for _ in f:
                        line_count += 1
            except OSError:
                line_count = 0

        if line_count >= MAX_EVENTS:
            # Rotate: keep newest (MAX_EVENTS - 1) lines, then append this one.
            # Reads the whole file into memory — fine at 10k lines.
            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                # Trim down to (MAX_EVENTS - 1) so this append puts us at MAX.
                keep = lines[-(MAX_EVENTS - 1):] if len(lines) > MAX_EVENTS - 1 else lines
                tmp = path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.writelines(keep)
                    f.write(line)
                os.replace(tmp, path)
                return
            except OSError as e:
                print_ts(
                    f"{COLOR_RED}events_log: rotate failed: {e}{COLOR_END}",
                    error=True,
                )
                # Fall through to plain append below.

        # Plain append.
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        # Logging must never break the runtime. Swallow but trace once.
        print_ts(
            f"{COLOR_RED}events_log: append failed for {agent_id}/{kind}: {e}{COLOR_END}",
            error=True,
        )


def tail_events(n: int = 100) -> list[dict]:
    """Return the most recent n events. Used by `/api/openflip/activity`.

    Reads the JSONL file from the end. Tolerates partial/torn lines.
    """
    path = _events_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    tail = lines[-n:] if len(lines) > n else lines
    out: list[dict] = []
    for line in tail:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue
    return out
