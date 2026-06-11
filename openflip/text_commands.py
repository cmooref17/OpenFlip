"""Cross-transport text-prefix commands.

Discord exposes commands as nextcord slash commands (see `commands.py`).
Other transports (iMessage, future) don't have a slash-command layer, so
operators interact via plain message text. This module mirrors the most
useful slash commands as `/command` text prefixes that work on ANY transport.

Wired in from runtime._handle_message (Discord) and runtime._handle_inbound
(transport-agnostic). The handler is called BEFORE soft-inject/hard-interrupt
logic — if it matches a command, it acts and returns True, and the caller
skips the rest of the inbound pipeline (no enqueue).

`/stop` is NOT handled here — its text-prefix handling stays inline in
runtime because it needs special interrupt+enqueue behavior (cancel the
active task, then enqueue the stop message as a fresh turn).
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
import shutil
import time
from typing import Any

from .acl import is_owner
from .utils import print_ts, save_json, COLOR_YELLOW, COLOR_END


# Commands handled as text prefixes. Match is case-insensitive on the first
# whitespace-delimited token.
_TEXT_COMMANDS = {
    "/reset",
    "/compact",
    "/uncompact",
    "/status",
    "/reload",
    "/restart",
    "/help",
}


def is_text_command(raw_text: str) -> bool:
    """True if raw_text starts with a recognized text command."""
    if not raw_text:
        return False
    head = raw_text.strip().split(None, 1)[0].lower()
    return head in _TEXT_COMMANDS


async def _send(transport, session_id: str, channel, text: str) -> None:
    """Send a reply via transport if available, else channel.send."""
    if transport and session_id:
        try:
            await asyncio.wait_for(transport.send(session_id, text), timeout=30.0)
            return
        except asyncio.TimeoutError:
            print_ts(f"{COLOR_YELLOW}text_commands._send via transport timed out{COLOR_END}")
        except Exception as e:
            print_ts(f"{COLOR_YELLOW}text_commands._send via transport failed: {e}{COLOR_END}")
    if channel is not None:
        try:
            await asyncio.wait_for(channel.send(text), timeout=30.0)
        except Exception:
            pass


async def handle_text_command(
    *,
    runner: Any,
    channel: Any,
    speaker_id: int,
    raw_text: str,
    transport: Any = None,
    session_id: str = "",
    session: Any = None,
) -> bool:
    """Dispatch a text-prefix command. Returns True if handled.

    `session` is the inbound Session (when available). Owner-gated commands
    derive transport+handle from it so iMessage owner checks compare the raw
    handle instead of the per-process-unstable hash. Absent session →
    transport="discord", handle="" → the numeric is_owner path (unchanged).
    """
    if not raw_text:
        return False
    text = raw_text.strip()
    parts = text.split(None, 1)
    head = parts[0].lower()
    if head not in _TEXT_COMMANDS:
        return False
    arg = parts[1].strip() if len(parts) > 1 else ""

    ch_id = int(getattr(channel, "id", 0) or 0)
    # Conversation key for the dicts (conversations/_pending_inject): the
    # native int unless the session is identity-linked, in which case the
    # shared "linked:<canonical>" key. ch_id itself stays native — it backs
    # reply routing (CURRENT_CHANNEL_ID in /restart) and channel-scoped ACL
    # checks (/help), which must never follow the link.
    conv_key = runner.conv_key_for_session(session, ch_id) if session is not None else runner.conv_key(ch_id)

    # Transport-aware owner identity for the gated commands below.
    tname = getattr(session, "transport", "discord") or "discord"
    handle = getattr(session, "handle", "") or ""

    if head == "/reset":
        await _do_reset(runner, conv_key, channel, transport, session_id)
        return True

    if head == "/compact":
        await _do_compact(runner, conv_key, channel, transport, session_id)
        return True

    if head == "/uncompact":
        if not is_owner(speaker_id, transport=tname, handle=handle):
            await _send(transport, session_id, channel, "Owner only.")
            return True
        await _do_uncompact(runner, conv_key, channel, transport, session_id)
        return True

    if head == "/status":
        await _do_status(runner, conv_key, channel, transport, session_id)
        return True

    if head == "/reload":
        if not is_owner(speaker_id, transport=tname, handle=handle):
            await _send(transport, session_id, channel, "Owner only.")
            return True
        await _do_reload(runner, channel, transport, session_id)
        return True

    if head == "/restart":
        if not is_owner(speaker_id, transport=tname, handle=handle):
            await _send(transport, session_id, channel, "Owner only.")
            return True
        reason = arg or "Manual restart from text-prefix /restart."
        await _do_restart(runner, ch_id, speaker_id, reason, channel, transport, session_id)
        return True

    if head == "/help":
        await _do_help(runner, speaker_id, ch_id, channel, transport, session_id,
                       owner_transport=tname, owner_handle=handle)
        return True

    return False


# ---------------- shared command bodies ----------------


def _conversation_id_for_channel(channel: Any, ch_id: int) -> str:
    """Return the canonical conversation id for this channel.

    Pulls from the Session object on the channel — works for any transport
    (Discord, iMessage, future). No hardcoded prefix; if the session can't
    be found, returns empty string and the caller must handle it. NEVER
    fall back to assuming "discord:" — that breaks iMessage and any other
    non-Discord transport (a real bug I shipped on first cut).
    """
    # Linked conversations: the dict key IS the conversation id
    # ("linked:<canonical>") — no session lookup needed.
    if isinstance(ch_id, str) and ch_id.startswith("linked:"):
        return ch_id
    try:
        # TransportChannel/_SessionChannel expose `_session`; accept a plain
        # `session` attribute too for any future channel-likes.
        sess = getattr(channel, "session", None) or getattr(channel, "_session", None)
        if sess is not None:
            cid = getattr(sess, "conversation_id", None)
            if cid:
                return str(cid)
    except Exception:
        pass
    return ""


async def _do_reset(runner, ch_id, channel, transport, session_id) -> None:
    conv = runner.conversations.pop(ch_id, None)
    if conv and hasattr(conv, "clear_history"):
        try:
            conv.clear_history()
        except Exception as e:
            print_ts(f"{COLOR_YELLOW}/reset: clear_history failed: {e}{COLOR_END}",
                     agent=runner.agent.id)
    else:
        agent_dir = os.path.dirname(runner.agent.path)
        conv_id = _conversation_id_for_channel(channel, ch_id)
        if not conv_id:
            await _send(transport, session_id, channel,
                        "⚠️ /reset: could not resolve conversation id for this channel.")
            return
        jsonl_path = os.path.join(agent_dir, "conversations", conv_id + ".jsonl")
        if os.path.exists(jsonl_path):
            try:
                backup = f"{jsonl_path}.pre_reset_{int(time.time())}.bak.jsonl"
                shutil.copy2(jsonl_path, backup)
                print_ts(f"/reset: backed up conversation to {backup}",
                         agent=runner.agent.id)
            except Exception as _bk_err:
                print_ts(
                    f"{COLOR_YELLOW}/reset: pre-reset backup failed for "
                    f"{jsonl_path}: {_bk_err}{COLOR_END}",
                    agent=runner.agent.id,
                )
            try:
                _all_bak = sorted(glob.glob(f"{jsonl_path}.pre_reset_*.bak.jsonl"))
                if len(_all_bak) > 5:
                    for _stale in _all_bak[:-5]:
                        try:
                            os.remove(_stale)
                        except OSError:
                            pass
            except Exception as _retain_e:
                print_ts(
                    f"{COLOR_YELLOW}/reset: backup retention sweep failed: "
                    f"{_retain_e}{COLOR_END}",
                    agent=runner.agent.id,
                )
        for ext in (".jsonl", ".meta.json"):
            target = os.path.join(agent_dir, "conversations", conv_id + ext)
            try:
                if os.path.exists(target):
                    os.remove(target)
            except Exception as _rm_err:
                print_ts(
                    f"{COLOR_YELLOW}/reset: failed to delete {target}: "
                    f"{_rm_err}{COLOR_END}",
                    agent=runner.agent.id,
                )
    try:
        runner._pending_inject.pop(ch_id, None)
    except Exception:
        pass
    await _send(transport, session_id, channel, "Conversation reset.")


async def _do_compact(runner, ch_id, channel, transport, session_id) -> None:
    conv = runner.conversations.get(ch_id)
    if conv is None or not hasattr(conv, "force_compact_next"):
        await _send(
            transport, session_id, channel,
            "`/compact` is Anthropic-only and there's no active conversation in this channel.",
        )
        return
    conv.force_compact_next = True
    await _send(
        transport, session_id, channel,
        "⚙️ Compaction queued — will fire on your next message.",
    )


async def _do_uncompact(runner, ch_id, channel, transport, session_id) -> None:
    if runner.agent.provider != "anthropic":
        await _send(transport, session_id, channel, "`/uncompact` is Anthropic-only.")
        return
    agent_dir = os.path.dirname(runner.agent.path)
    conv_id = _conversation_id_for_channel(channel, ch_id)
    if not conv_id:
        await _send(transport, session_id, channel,
                    "⚠️ /uncompact: could not resolve conversation id for this channel.")
        return
    jsonl_path = os.path.join(agent_dir, "conversations", conv_id + ".jsonl")
    meta_path = os.path.join(agent_dir, "conversations", conv_id + ".meta.json")
    baks = sorted(glob.glob(f"{jsonl_path}.compaction_*.bak.jsonl"))
    if not baks:
        await _send(transport, session_id, channel,
                    "No compaction backup found for this channel.")
        return
    latest = baks[-1]
    try:
        shutil.copy2(latest, jsonl_path)
    except Exception as e:
        await _send(transport, session_id, channel, f"⚠️ Restore failed: {e}")
        return
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta.pop("compaction_block", None)
            save_json(meta_path, meta)
        except Exception as e:
            print_ts(
                f"{COLOR_YELLOW}/uncompact: meta cleanup failed: {e}{COLOR_END}",
                agent=runner.agent.id,
            )
    runner.conversations.pop(ch_id, None)
    await _send(transport, session_id, channel,
                f"↩️ Restored conversation from {os.path.basename(latest)}.")


async def _do_status(runner, ch_id, channel, transport, session_id) -> None:
    from .config_global import get_model_context_window
    agent = runner.agent
    conv = runner.conversations.get(ch_id)
    window = get_model_context_window(agent.model, agent.provider)
    lines = [f"**{agent.display_name}**"]
    lines.append(f"• Model: `{agent.model}`")
    usage = getattr(conv, "last_usage", None) if conv else None
    if usage:
        total = usage["total_input"]
        lines.append(f"• Context: {total:,} / {window:,}")
        if agent.provider in ("anthropic", "openai"):
            lines.append(
                f"• Cache: read {usage['cache_read_input_tokens']:,} • "
                f"create {usage['cache_creation_input_tokens']:,}"
            )
    else:
        lines.append(f"• Context: 0 / {window:,}")
    if conv:
        in_mem = len([m for m in conv.messages if m.get("role") != "system"])
        try:
            from . import _conversation_io as _cio
            conv_id = _conversation_id_for_channel(channel, ch_id)
            if conv_id:
                path = _cio.conversation_path(os.path.dirname(agent.path), conv_id)
                on_disk = sum(1 for _ in open(path)) if os.path.exists(path) else in_mem
            else:
                on_disk = in_mem
        except Exception:
            on_disk = in_mem
        lines.append(f"• Messages: {in_mem} in memory / {on_disk} on disk")
    await _send(transport, session_id, channel, "\n".join(lines))


async def _do_reload(runner, channel, transport, session_id) -> None:
    try:
        changed = runner.reload_agent_config()
    except Exception as e:
        await _send(transport, session_id, channel, f"⚠️ Reload failed: {e}")
        return
    if changed:
        await _send(
            transport, session_id, channel,
            f"♻️ **{runner.agent.display_name}** reloaded — system files re-read, conversations re-applied.",
        )
    else:
        await _send(
            transport, session_id, channel,
            f"No on-disk changes detected for **{runner.agent.display_name}**.",
        )


async def _do_restart(runner, ch_id, speaker_id, reason, channel, transport, session_id) -> None:
    from .tools.restart import restart_gateway
    from .tool_executor import CURRENT_AGENT, CURRENT_CHANNEL_ID, CURRENT_SPEAKER_ID
    CURRENT_AGENT.set(runner.agent)
    CURRENT_CHANNEL_ID.set(ch_id)
    CURRENT_SPEAKER_ID.set(int(speaker_id))
    await _send(transport, session_id, channel, f"♻️ Restart triggered — {reason}. Hold on…")
    try:
        await restart_gateway(reason)
    except Exception as e:
        await _send(transport, session_id, channel, f"⚠️ Restart failed: {e}")


async def _do_help(runner, speaker_id, ch_id, channel, transport, session_id,
                   owner_transport="discord", owner_handle="") -> None:
    from .pipeline import build_visible_tools
    from .tools import TOOL_REGISTRY
    agent = runner.agent
    transport_name = getattr(transport, "name", "discord") or "discord"
    callable_funcs, _ext, _preamble = build_visible_tools(
        agent,
        transport=transport_name,
        speaker_id=speaker_id,
        speaker_role_ids=[],
        channel_id=ch_id,
        owner=is_owner(speaker_id, transport=owner_transport, handle=owner_handle),
        handle=owner_handle,
    )
    if not callable_funcs:
        await _send(
            transport, session_id, channel,
            f"**{agent.display_name}** doesn't have any tools available to you here.",
        )
        return
    lines = [f"**{agent.display_name}** can do the following for you:"]
    for f in callable_funcs:
        t = TOOL_REGISTRY.get(f.__name__)
        if t:
            lines.append(f"• `{t.name}` — {t.description}")
    await _send(transport, session_id, channel, "\n".join(lines))
