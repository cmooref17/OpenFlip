"""Last-used delivery route for identity-linked conversations.

An identity link (config.json `identity_links`) makes a person's Discord DM
and iMessage 1:1 share ONE conversation, keyed by the PRIMARY conversation_id
(e.g. "imessage:+15551234567"). Live replies are unaffected: they post back
through the channel the message arrived on. But anything that later starts
from just that conversation_id (send_message(session_id=...), a cron job, a
restart continuation) used to read the transport off the id's PREFIX and
deliver to the primary's transport, even when the person had been talking
on the other one.

This module remembers, per linked conversation, the native route the person
most recently used (transport, transport_id, handle, speaker_id). It is
written on every inbound message whose conversation is forwarded, persisted
to agents/<id>/linked_routes.json (atomic), and read by the outbound paths
above so delivery follows the person. Conversation identity (history) is
untouched; only where a message is delivered changes.

Unlinked conversations never get an entry, so they behave exactly as before.
Routing only: nothing here confers privilege (ACL still uses the session's
own native transport + identity).
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass

from .utils import load_json, save_json, print_ts

_FILE = "linked_routes.json"
_lock = threading.Lock()
_cache: dict[str, dict] = {}  # agent_dir -> routes dict (loaded once per process)


@dataclass(frozen=True)
class Route:
    transport: str
    transport_id: str
    handle: str = ""
    speaker_id: int = 0
    is_dm: bool = True


def _path(agent_dir: str) -> str:
    return os.path.join(agent_dir, _FILE)


def _routes(agent_dir: str) -> dict:
    if agent_dir not in _cache:
        data = load_json(_path(agent_dir), default={})
        _cache[agent_dir] = data if isinstance(data, dict) else {}
    return _cache[agent_dir]


def record(agent_dir: str, session) -> bool:
    """Remember the native route of an inbound session IF its conversation is
    forwarded (identity-linked). Returns True when the stored route changed.
    Never raises: a routing hint must never break an inbound message."""
    try:
        from .config_global import is_forwarded_conversation
        conv_id = getattr(session, "conversation_id", "") or ""
        transport = getattr(session, "transport", "") or ""
        tid = str(getattr(session, "transport_id", "") or "")
        if not (conv_id and transport and tid):
            return False
        if not is_forwarded_conversation(conv_id, f"{transport}:{tid}") and not _is_primary_side(agent_dir, conv_id, transport):
            return False
        entry = {
            "transport": transport,
            "transport_id": tid,
            "handle": getattr(session, "handle", "") or "",
            "speaker_id": int(getattr(session, "speaker_id", 0) or 0),
            "is_dm": bool(getattr(session, "is_dm", True)),
            "ts": int(time.time()),
        }
        with _lock:
            routes = _routes(agent_dir)
            old = routes.get(conv_id) or {}
            if all(old.get(k) == entry[k] for k in ("transport", "transport_id", "handle", "speaker_id")):
                return False
            routes[conv_id] = entry
            save_json(_path(agent_dir), routes)
        return True
    except Exception as e:
        print_ts(f"linked_routes.record failed (ignored): {e}", error=True)
        return False


def _is_primary_side(agent_dir: str, conv_id: str, transport: str) -> bool:
    """The PRIMARY identity's own messages key by their native id, so they are
    not 'forwarded' — but once a secondary has used the conversation, a later
    message on the primary side must flip the route back. True when conv_id is
    some identity link's primary and this inbound is on that primary's
    transport."""
    from .config_global import get_identity_links
    if conv_id not in set(get_identity_links().values()):
        return False
    return conv_id.split(":", 1)[0] == transport


def lookup(agent_dir: str, conv_id: str) -> Route | None:
    """The last native route used in a linked conversation, or None (unlinked,
    or nobody has messaged since the link was added)."""
    if not conv_id:
        return None
    with _lock:
        e = _routes(agent_dir).get(conv_id)
    if not isinstance(e, dict) or not e.get("transport") or not e.get("transport_id"):
        return None
    return Route(
        transport=str(e["transport"]),
        transport_id=str(e["transport_id"]),
        handle=str(e.get("handle") or ""),
        speaker_id=int(e.get("speaker_id") or 0),
        is_dm=bool(e.get("is_dm", True)),
    )


def delivery_session(agent_dir: str, conv_id: str, *, tool_grants=None):
    """A synthetic Session that keeps conversation_id = conv_id (shared history)
    but delivers through the last native route. None when there is no route,
    so callers keep their existing behavior."""
    r = lookup(agent_dir, conv_id)
    if r is None:
        return None
    from .session import Session
    return Session(
        transport=r.transport,
        transport_id=r.transport_id,
        conversation_id=conv_id,
        speaker_id=0,
        speaker_role_ids=[],
        is_owner=False,
        is_dm=r.is_dm,
        display_name=r.handle or f"synthetic:{r.transport_id}",
        handle=r.handle,
        tool_grants=list(tool_grants or []),
    )


def _reset_cache_for_tests() -> None:
    with _lock:
        _cache.clear()
