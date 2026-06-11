"""Filesystem watcher → websocket broadcaster.

Strategy: poll mtime+size of each open conversation jsonl every 500ms.
When something changes, read only the new bytes since the last seen
position and broadcast each new JSON line to every websocket subscribed
to that (agent_id, channel_id) pair.

Polling at 500ms is fine — these are small JSONL files on a local SSD.
inotify would be marginally cleaner but adds an event-loop bridge for
zero meaningful gain at this scale."""
from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

from .config import OPENFLIP_AGENTS_DIR
from .._conversation_io import fs_encode as _fs_encode
from ..utils import print_ts


# (agent_id, channel_id) -> set of subscriber asyncio.Queue
_subs: Dict[Tuple[str, str], Set["asyncio.Queue[dict]"]] = defaultdict(set)
# (agent_id, channel_id) -> last seen file size
_positions: Dict[Tuple[str, str], int] = {}
# Async lock around mutating _subs / _positions
_lock = asyncio.Lock()
# The single background poll task
_poll_task: asyncio.Task | None = None


def _conv_path(agent_id: str, channel_id: str) -> Path:
    # `channel_id` is the full transport-prefixed conversation id
    # ("discord:1234", "imessage:5678") and is used verbatim as the file
    # stem. A bare id with no transport prefix is a caller bug — we refuse
    # to guess a transport (the old code assumed "discord:" and put it
    # back, which silently broke iMessage and masked routing bugs).
    if ":" not in channel_id:
        raise ValueError(
            f"_conv_path: channel_id {channel_id!r} has no transport prefix "
            f"(expected e.g. 'discord:<id>' / 'imessage:<id>'). Caller must "
            f"pass the full conversation_id from the session, not a bare id."
        )
    # fs_encode: on Windows the on-disk stem encodes ":" as "%3A" (NTFS
    # forbids colons); the conversation_id itself keeps the colon form.
    return OPENFLIP_AGENTS_DIR / agent_id / "conversations" / f"{_fs_encode(channel_id)}.jsonl"


async def subscribe(agent_id: str, channel_id: str) -> "asyncio.Queue[dict]":
    """Create a per-websocket queue subscribed to this conversation.
    The websocket handler drains the queue and sends frames to its
    client. Returns the queue."""
    q: asyncio.Queue[dict] = asyncio.Queue(maxsize=1000)
    key = (agent_id, channel_id)
    async with _lock:
        _subs[key].add(q)
        if key not in _positions:
            path = _conv_path(agent_id, channel_id)
            _positions[key] = path.stat().st_size if path.exists() else 0
    _ensure_poller()
    return q


async def unsubscribe(agent_id: str, channel_id: str,
                      q: "asyncio.Queue[dict]") -> None:
    key = (agent_id, channel_id)
    async with _lock:
        _subs[key].discard(q)
        if not _subs[key]:
            del _subs[key]
            _positions.pop(key, None)


def _ensure_poller() -> None:
    global _poll_task
    if _poll_task is None or _poll_task.done():
        _poll_task = asyncio.create_task(_poll_loop())


async def _poll_loop() -> None:
    """Polls every 500ms. Exits when there's nothing to watch."""
    while True:
        await asyncio.sleep(0.5)
        async with _lock:
            keys = list(_subs.keys())
        if not keys:
            return
        for key in keys:
            try:
                await _check_one(key)
            except Exception as e:
                # Don't kill the poller on a single bad file
                print_ts(f"watcher error on {key}: {e}", error=True)


async def _check_one(key: Tuple[str, str]) -> None:
    agent_id, channel_id = key
    path = _conv_path(agent_id, channel_id)
    if not path.exists():
        return
    try:
        size = path.stat().st_size
    except OSError:
        return
    last_pos = _positions.get(key, 0)
    if size == last_pos:
        return
    # File was truncated or rotated — restart from 0
    if size < last_pos:
        last_pos = 0
    new_lines = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            fh.seek(last_pos)
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    new_lines.append(json.loads(line))
                except Exception:
                    continue
        _positions[key] = size
    except Exception:
        return
    if not new_lines:
        return
    # Broadcast to subscribers
    async with _lock:
        subscribers = list(_subs.get(key, ()))
    for q in subscribers:
        for row in new_lines:
            try:
                q.put_nowait({"type": "message", "data": row})
            except asyncio.QueueFull:
                # Drop oldest queued frame and try again
                try:
                    q.get_nowait()
                    q.put_nowait({"type": "message", "data": row})
                except Exception:
                    pass
