"""Silently inject context into another agent's conversation history.

Unlike `talk_to_agent` (which dispatches a synthetic turn the recipient
processes immediately), this tool just *plants* a message in the target
agent's conversation history for a given channel. Nothing is posted to
Discord and the target does NOT take a turn now — the injected text simply
lands in that agent's history so its NEXT turn in that channel sees it as
context.

This mirrors the owner-only `/inject_context` slash command in commands.py
and reuses its exact, already-verified write path:
  * If the target runner is active, inject into the LIVE in-memory
    conversation (get-or-create) then save() — the in-memory list is the
    source of truth for the next model turn; a bare JSONL append would be
    invisible until reload.
  * If the runner is not active, append straight to the channel's JSONL on
    disk; the agent load()s it fresh on its next inbound for that channel.
"""
from __future__ import annotations

import os

from ._base import tool, ToolResult
from ..utils import print_ts


@tool
async def inject_context(agent_id: str, channel_id: int = 0, text: str = "", session_id: str = "") -> ToolResult:
    """Silently inject context into another agent's conversation history for a
    specific conversation. The text is NOT posted to Discord and the target
    does NOT take a turn now — it simply lands in that agent's history (marked
    `[INJECTED CONTEXT]: ...`) so the agent's NEXT turn in that conversation
    sees it as background context.

    Use this to plant a fact, reminder, or piece of context into another
    agent ahead of its next reply, without triggering or interrupting it.

    Mid-turn caveat: do NOT inject into an agent that is actively mid-reply
    (i.e. between a tool_use and its tool_result). Appending a user message
    into the history at that point can corrupt the in-flight turn's message
    sequence. Inject when the target is idle between turns.

    Args:
        agent_id: The target agent's id.
        session_id: CANONICAL conversation key — the transport-prefixed
            conversation id (e.g. "discord:12345", "imessage:1",
            "internal:email-support"). When provided it is used DIRECTLY as
            the conversation key: no int() coercion, no transport-prefix
            guessing. Prefer this on multi-transport agents — a bare channel
            id is ambiguous and can land in the wrong conversation.
        channel_id: DEPRECATED bare-int Discord channel id. Used only as a
            fallback when session_id is empty, in which case the transport
            prefix is inferred from the target runner. Pass session_id
            instead.
        text: The context text to inject.
    """
    from .. import registry
    from ..acl import current_caller_is_owner

    # Owner-only: this plants context into ANOTHER agent's history, the same
    # capability the owner-only /inject_context slash command exposes. Enforce
    # it on the tool itself (defense in depth) so granting the tool to a
    # non-owner can't quietly let them inject into a peer. Fail closed.
    if not current_caller_is_owner():
        return ToolResult.fail("inject_context is owner-only.")

    if not agent_id:
        return ToolResult.fail("agent_id is required")
    if text is not None and not isinstance(text, str):
        text = str(text)
    if not text or not text.strip():
        return ToolResult.fail("text is empty")

    target_agent = registry.ALL_AGENTS.get(agent_id)
    if not target_agent:
        return ToolResult.fail(
            f"Unknown agent: '{agent_id}'. Known agents: {sorted(registry.ALL_AGENTS.keys())}"
        )

    # session_id is the canonical, transport-prefixed conversation key. When
    # the caller supplies it we use it DIRECTLY as the on-disk conversation id
    # — no int() coercion, no transport-prefix inference. channel_id is the
    # DEPRECATED bare-int fallback whose behavior below is byte-for-byte the
    # pre-session_id path.
    from .. import _conversation_io as _cio
    # str()-coerce before .strip() so a non-string arg can't AttributeError.
    session_id = (str(session_id) if session_id else "").strip()

    # channel_id changed from required to default 0; without this guard,
    # omitting BOTH would silently inject into "<transport>:0" instead of
    # erroring. Fail loud.
    if not session_id and not channel_id:
        return ToolResult.fail("session_id or channel_id required")

    target_runner = registry.RUNNERS.get(agent_id)

    # `live_conv` is the in-memory conversation object to append to, or None to
    # fall back to a direct on-disk JSONL append. `conv_id` is always the
    # transport-prefixed id that governs the on-disk filename.
    live_conv = None
    was_loaded = False

    if session_id:
        # Reuse the framework's single filesystem-safety gate (fail-closed on
        # path traversal / control chars) — this id becomes a filename.
        try:
            conv_id = _cio._safe_conversation_id(session_id)
        except ValueError as _e:
            return ToolResult.fail(str(_e))
        conv_label = conv_id
        # CRITICAL: in-memory conversations are keyed by an int derived from the
        # session's transport_id (TransportChannel.id), which does NOT equal the
        # conversation_id suffix for iMessage 1:1 DMs (conversation_id is
        # "imessage:<handle>" while transport_id is the numeric chat_id). So we
        # must NOT re-derive a key from the suffix — that would spawn a
        # duplicate, dead in-memory conversation the agent never reads. Instead
        # find the LIVE conversation by matching conversation_id and reuse its
        # real dict entry.
        if target_runner is not None:
            for _c in target_runner.conversations.values():
                if getattr(_c, "conversation_id", None) == conv_id:
                    live_conv = _c
                    was_loaded = True
                    break
        # If no live conversation matched (agent idle / channel not loaded),
        # live_conv stays None and we append straight to the on-disk JSONL keyed
        # by conv_id — the disk filename is governed by conversation_id, which is
        # correct. The agent load()s it fresh on its next turn in that channel.
    else:
        try:
            ch_id = int(channel_id)
        except (ValueError, TypeError):
            return ToolResult.fail("channel_id must be a numeric channel/session id.")

        # Conversation IDs are transport-prefixed ("discord:1234", "imessage:1",
        # "internal:google"). Resolve the prefix from the target runner's actual
        # transport — never assume Discord. If the target has multiple transports
        # and the conversation isn't already loaded, fall back to the first
        # transport name on the agent (matches how the runner names new sessions).
        _transport_name = "discord"  # safe fallback only if no runner / no transport
        if target_runner is not None:
            # Prefer the runner's current transport name (single-transport case).
            _t = getattr(target_runner, "transport", None)
            if _t is not None and getattr(_t, "name", None):
                _transport_name = _t.name
            else:
                # Multi-transport: take the first declared transport. This matches
                # the convention IMessageTransport/DiscordTransport use when minting
                # their own conversation_ids on inbound message arrival.
                _ts = list(getattr(target_runner.agent, "transports", []) or [])
                if _ts:
                    _transport_name = _ts[0]
        conv_id = f"{_transport_name}:{ch_id}"
        conv_label = ch_id
        # Legacy path: when the runner is active, get-or-create the live
        # conversation (loads from disk on demand) keyed by the bare int — for
        # Discord/internal/imessage-group this int matches TransportChannel.id.
        if target_runner is not None:
            was_loaded = ch_id in target_runner.conversations
            live_conv = target_runner.get_conversation(ch_id, conv_id)

    marked_text = f"[INJECTED CONTEXT]: {text}"

    # The LIVE in-memory conversation list is the source of truth for the
    # target's next model turn — a bare JSONL append is invisible until the
    # conversation reloads. So whenever we have a live conversation object,
    # inject into it (then persist), mirroring the battle-tested
    # _drain_pending_injects path in runtime.py.
    if live_conv is not None:
        # Same ChatMessage construction _drain_pending_injects uses, branched
        # on provider. Append to conv.messages (the live list), then save()
        # — after a fresh load() _persisted_count == len(history), so save()
        # appends ONLY this new message, no duplication.
        if target_agent.provider == "anthropic":
            from ..anthropic_conversation import ChatMessage as _AntMsg
            live_conv.messages.append(_AntMsg("user", marked_text))
        else:
            from ..ollama_api import ChatMessage as _OllamaMsg
            live_conv.messages.append(_OllamaMsg("user", marked_text))
        live_conv.save()
        _how = "in-memory (already loaded)" if was_loaded else "in-memory (loaded on demand)"
        print_ts(f"inject_context tool: injected into {agent_id} conv {conv_label} [{_how}] + disk",
                 agent=agent_id)
    else:
        agent_dir = os.path.dirname(target_agent.path)
        jsonl_path = _cio.conversation_path(agent_dir, conv_id)
        _cio.append_messages(
            jsonl_path,
            [{"role": "user", "content": marked_text}],
            content_extractor=lambda m: m.get("content", ""),
        )
        # No live conversation (runner inactive, or session_id matched no loaded
        # conversation) — disk append is correct; the agent will load() this
        # message fresh on its next inbound for the channel.
        print_ts(f"inject_context tool: appended to {agent_id} conv {conv_label} JSONL (no live conversation)",
                 agent=agent_id)

    return ToolResult(
        model_feedback=f"Injected context into {target_agent.display_name} for conversation {conv_label}. "
        "It will surface as background context on that agent's next turn in that conversation; "
        "nothing was posted to Discord."
    )
