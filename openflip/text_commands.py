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
from typing import Any

from .acl import is_owner
from .utils import print_ts, save_json, COLOR_YELLOW, COLOR_END


# Commands handled as text prefixes. Match is case-insensitive on the first
# whitespace-delimited token.
_TEXT_COMMANDS = {
    "/reset",
    "/compact",
    "/uncompact",
    "/effort",
    "/model",
    "/models",
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

    # Preload the target conversation from disk before the _do_* handlers run.
    # They look it up with a raw `runner.conversations.get(conv_key)`, which is
    # None for a conversation that exists on disk but isn't loaded this
    # process-run (e.g. right after a restart, or the first touch of a channel)
    # — surfacing a false "no active conversation in this channel" for /effort,
    # /compact, /uncompact, /status. get_conversation get-or-creates from disk
    # under the SAME key conv_key resolves to, so the lookups below succeed.
    _conv_id = (getattr(session, "conversation_id", "") or "") if session is not None else ""
    if _conv_id:
        try:
            runner.get_conversation(ch_id, _conv_id)
        except Exception as _preload_err:
            print_ts(
                f"{COLOR_YELLOW}text_commands: conversation preload failed "
                f"({_preload_err}); a command may report no active conversation"
                f"{COLOR_END}",
                agent=runner.agent.id,
            )

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

    if head == "/effort":
        # Owner-only: /effort changes the Anthropic request body
        # (output_config.effort), which affects reasoning depth + billing.
        # Same gating as the slash effort_cmd and /uncompact above.
        if not is_owner(speaker_id, transport=tname, handle=handle):
            await _send(transport, session_id, channel, "Owner only.")
            return True
        await _do_effort(runner, conv_key, arg, channel, transport, session_id)
        return True

    if head in ("/model", "/models"):
        # Owner-only: /model changes the agent's model (and provider) and
        # rewrites agent.json — the text mirror of the slash /model panel,
        # which is itself owner-gated. Same gating as /effort and /uncompact.
        # `/models` is an alias — bare /model already lists the choices.
        if not is_owner(speaker_id, transport=tname, handle=handle):
            await _send(transport, session_id, channel, "Owner only.")
            return True
        await _do_model(runner, arg, channel, transport, session_id)
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
    # The full race-safe stop+wipe sequence (epoch bump → hard interrupt →
    # queued-turn drop → clear/delete + pre-reset backup) lives in ONE place:
    # AgentRunner.reset_conversation, shared with the slash /reset so the two
    # paths can't drift again. This path historically LACKED the race guard
    # (only the slash command got the 65b1464/ff782f3 fix) — never reintroduce
    # an inline reset body here.
    conv_id = _conversation_id_for_channel(channel, ch_id)
    if not runner.reset_conversation(ch_id, fallback_conv_id=conv_id):
        await _send(transport, session_id, channel,
                    "⚠️ /reset: could not resolve conversation id for this channel.")
        return
    await _send(transport, session_id, channel, "Conversation reset.")


async def _do_compact(runner, ch_id, channel, transport, session_id) -> None:
    conv = runner.conversations.get(ch_id)
    if conv is None or not hasattr(conv, "force_compact_next"):
        await _send(
            transport, session_id, channel,
            "`/compact` is Anthropic-only and there's no active conversation in this channel.",
        )
        return
    # Both flags, same as the slash compact_cmd: force_compact_next opts the
    # next chat() into server-side compaction; force_compact_trigger_override
    # makes that request send the low _MANUAL_COMPACT_TRIGGER (50k, Anthropic's
    # floor) instead of the real per-model trigger. Without the override a
    # sub-threshold conversation silently never compacts — the exact no-op
    # 65b1464 fixed on the slash path.
    conv.force_compact_next = True
    conv.force_compact_trigger_override = True
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
    from . import _conversation_io as _cio_uc
    jsonl_path = _cio_uc.conversation_path(agent_dir, conv_id)
    meta_path = os.path.join(agent_dir, "conversations", _cio_uc.fs_encode(conv_id) + ".meta.json")
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


# Valid effort levels — mirror the slash effort_cmd SlashOption choices.
_EFFORT_LEVELS = ["default", "low", "medium", "high", "xhigh", "max"]


async def _do_effort(runner, ch_id, arg, channel, transport, session_id) -> None:
    conv = runner.conversations.get(ch_id)
    # Anthropic-only: only AnthropicConversation carries `effort_override`.
    # Same shape as _do_compact's hasattr(force_compact_next) guard.
    if conv is None or not hasattr(conv, "effort_override"):
        await _send(
            transport, session_id, channel,
            "`/effort` is Anthropic-only and there's no active conversation in this channel.",
        )
        return
    # Bare `/effort` — report the current override + usage, change nothing.
    if not arg:
        current = getattr(conv, "effort_override", None)
        current_level = current if current else "default"
        shown = "\n".join(
            f"  • `{lvl}`" + ("  ← current" if lvl == current_level else "")
            for lvl in _EFFORT_LEVELS
        )
        note = "" if current else " (model default — no override)"
        await _send(
            transport, session_id, channel,
            f"Current effort for THIS conversation: `{current_level}`{note}.\n"
            f"Available levels:\n{shown}\n"
            f"Usage: `/effort <level>` to set.",
        )
        return
    level = arg.split(None, 1)[0].lower()
    if level not in _EFFORT_LEVELS:
        await _send(
            transport, session_id, channel,
            f"⚠️ Invalid effort level `{level}`. Valid options: {', '.join(_EFFORT_LEVELS)}.",
        )
        return
    if level == "default":
        conv.effort_override = None
        conv._save_meta()
        await _send(
            transport, session_id, channel,
            "⚙️ Effort override cleared for THIS conversation — falling back to the model default.",
        )
        return
    conv.effort_override = level
    conv._save_meta()
    await _send(
        transport, session_id, channel,
        f"⚙️ Effort for THIS conversation set to `{level}` (overrides the model default). "
        f"Use `/effort default` to clear.",
    )


async def _do_model(runner, arg, channel, transport, session_id) -> None:
    # Text mirror of the slash /model panel (agent_ui.open_model_panel). That
    # panel renders dropdowns/buttons that only exist on Discord; this gives any
    # transport the same set+persist behavior in plain text. The set path mirrors
    # agent_ui._ModelPicker.callback step-for-step.
    agent = runner.agent
    # Bare `/model` — report the current model + the configured choices, change
    # nothing. Deliberately instant: the choices come from config.json's
    # `models` block (local, no blocking Ollama/Anthropic fetch like the slash
    # panel does). config.json is the operator-curated set; any other valid
    # model name still works as an explicit arg.
    if not arg:
        from .config_global import get_config
        # agent.model carries the provider prefix ("anthropic/..."); config.json
        # keys are bare. Compare/display bare so the "← current" marker matches.
        bare_current = agent.model.split("/", 1)[-1] if "/" in agent.model else agent.model
        models = list((get_config().get("models") or {}).keys())
        if models:
            shown = "\n".join(
                f"  • `{m}`" + ("  ← current" if m == bare_current else "")
                for m in models
            )
            model_list = f"Available models:\n{shown}"
        else:
            model_list = "No models configured (any valid model name still works)."
        await _send(
            transport, session_id, channel,
            f"Current model: `{bare_current}` (provider `{agent.provider}`).\n"
            f"{model_list}\n"
            f"Usage: `/model <model-name>` to switch.",
        )
        return
    # First whitespace-delimited token only — same arg discipline as _do_effort.
    new_model = arg.split(None, 1)[0].strip()
    from .agent_ui import _provider_for_model, _propagate_model_to_live_conversations
    if new_model == agent.model:
        await _send(transport, session_id, channel, f"Model already set to `{new_model}`.")
        return
    # Auto-infer the provider from the name and flip it too — exactly as the
    # panel does. When the provider changes, clear this agent's live
    # conversations: Ollama / Anthropic / OpenAI histories aren't compatible.
    old_model = agent.model
    new_provider = _provider_for_model(new_model)
    old_provider = agent.provider
    agent.model = new_model
    agent.provider = new_provider
    if new_provider != old_provider:
        runner.conversations.clear()
    if not agent.save():
        await _send(transport, session_id, channel, "❌ Failed to save agent JSON.")
        return
    # Push the new model into any surviving live Conversation objects so the
    # next turn uses it (panel's final step).
    n = _propagate_model_to_live_conversations(agent.id)
    prov_note = f" (provider → `{new_provider}`)" if new_provider != old_provider else ""
    await _send(
        transport, session_id, channel,
        f"✅ Model: `{old_model}` → `{new_model}`{prov_note} "
        f"(updated {n} live conversation(s)).",
    )


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
