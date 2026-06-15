"""Send a message to a conversation from inside a tool.

Used primarily by agents during synthetic / heartbeat turns, where the
runner does NOT auto-post the final chat text. To say anything to the user
(or post in any conversation) during a heartbeat, an agent must call this
tool.

In a regular message turn, this tool is also available — you can use it to
post in OTHER conversations (cross-posting). Posting in the conversation
that triggered the turn still works through the normal final-text path.

Transport-agnostic addressing: the preferred target is a CANONICAL
transport-prefixed conversation key (`session_id`, e.g. "discord:123",
"imessage:<handle>", "internal:foo"). The transport to route through is
resolved generically from the prefix / the matched live conversation — no
transport name is hardcoded. The legacy bare-int Discord `channel_id` arg
is still honored as a deprecated fallback.

Phase 2 (Discord-decouple): routes through Transport.send/typing so no
nextcord imports live here.

Security: there is no per-conversation ACL on this tool. Anyone who has the
tool in `allowed_tools` can post anywhere the bot has access to. Restrict
at the tool level (`users:` in agent.json) if needed. Cross-target posting
(into a conversation other than the current one) is owner-gated below.
"""
from __future__ import annotations
import asyncio

from ._base import tool, ToolResult
from ..utils import print_ts, COLOR_YELLOW, COLOR_END


def _resolve_transport(runner, prefix: str):
    """Pick the transport to route a send through.

    Multi-transport agents hold several transports (e.g. Discord + iMessage).
    A conversation's transport is identified by its conversation_id prefix
    ("imessage:…" → the transport whose `.name` is "imessage"), mirroring the
    prefix→transport resolution cron.py uses. No transport name is hardcoded.

      * No prefix → primary transport (first declared).
      * Prefix matches a transport name → that transport.
      * Prefix matches nothing → None (fail loud rather than misroute to the
        wrong transport).
    """
    transports = [t for t in (getattr(runner, "_transports", None) or []) if t is not None]
    if not transports:
        return getattr(runner, "transport", None)
    if prefix:
        for t in transports:
            if getattr(t, "name", "") == prefix:
                return t
        return None
    return transports[0]


@tool
async def send_message(text: str, channel_id: int = 0, session_id: str = "") -> ToolResult:
    """Send a message to a conversation.

    Use this during heartbeats or any time you want to push text from inside a
    tool flow. Especially important for synthetic turns (cron, heartbeat) where
    the runtime no longer auto-posts your final reply — you must explicitly call
    this tool to say anything to the user.

    Args:
        text: The message to send. Will be split into multiple sends if it
            exceeds the transport's character limit (Discord's 2000-char cap).
        session_id: CANONICAL conversation key — the transport-prefixed
            conversation id (e.g. "discord:123", "imessage:you@example.com",
            "internal:foo"). When provided it is used DIRECTLY: no int()
            coercion, no transport-prefix guessing. The transport is resolved
            from the prefix and the live conversation, so this works for ANY
            transport. Prefer this — a bare channel id is Discord-only and
            ambiguous on multi-transport agents.
        channel_id: DEPRECATED bare-int Discord channel id. Used only as a
            fallback when session_id is empty. Defaults to the channel that
            triggered the current turn (the one you're "in"); pass an explicit
            id to cross-post somewhere else. Pass session_id instead.
    """
    text = (text or "").strip()
    if not text:
        return ToolResult.fail("`text` is required.")

    from ..tool_executor import CURRENT_AGENT, CURRENT_CHANNEL_ID, CURRENT_SESSION
    try:
        agent = CURRENT_AGENT.get()
    except LookupError:
        return ToolResult.fail("Tool invoked outside an agent context.")

    # Resolve transport from runner registry.
    from ..registry import RUNNERS
    runner = RUNNERS.get(agent.id)
    if not runner:
        return ToolResult.fail(f"No running agent for '{agent.id}'.")

    # str()-coerce before .strip() so a non-string arg can't AttributeError.
    session_id = (str(session_id) if session_id else "").strip()

    # ------------------------------------------------------------------
    # CANONICAL path: an explicit transport-prefixed conversation key.
    # ------------------------------------------------------------------
    if session_id:
        from .. import _conversation_io as _cio
        # Reuse the framework's single filesystem-safety gate (fail-closed on
        # path traversal / control chars). This id is used DIRECTLY as the
        # conversation key — no int() coercion, no transport-prefix guessing,
        # exactly like inject_context.
        try:
            conv_id = _cio._safe_conversation_id(session_id)
        except ValueError as _e:
            return ToolResult.fail(str(_e))

        # Cross-target owner guard (MED-1). Posting into a conversation OTHER
        # than the one the caller is currently in requires owner. current_caller
        # / is_owner are already transport-aware (Discord numeric id vs iMessage
        # handle). Owner-attributed synthetic turns (cron/heartbeat/restart) run
        # is_owner=True so are unaffected.
        current_conv_id = ""
        try:
            _sess = CURRENT_SESSION.get(None)
            if _sess is not None:
                current_conv_id = getattr(_sess, "conversation_id", "") or ""
        except Exception:
            current_conv_id = ""
        if conv_id != current_conv_id:
            from ..acl import is_owner as _is_owner
            from ..tool_executor import current_caller
            speaker_id, _tname, _handle = current_caller()
            if not _is_owner(speaker_id, transport=_tname, handle=_handle):
                return ToolResult.fail(
                    f"send_message: cross-target posting (target {conv_id} ≠ "
                    f"current {current_conv_id or '<none>'}) is restricted to the owner."
                )

        # Transport prefix → which transport to route through.
        prefix = conv_id.split(":", 1)[0] if ":" in conv_id else ""

        # Find the LIVE in-memory conversation by MATCHING conversation_id.
        # CRITICAL: for iMessage 1:1 DMs the conversation_id is
        # "imessage:<handle>" but the live conversation is keyed by the numeric
        # transport_id (chat_id) — which is exactly the transport-native send
        # key (TransportChannel.send passes session.transport_id). So the dict
        # KEY of the matched conversation IS the send key; we must NOT derive a
        # numeric key from the handle suffix. Mirrors inject_context's lookup.
        send_key = None
        for _k, _c in runner.conversations.items():
            if getattr(_c, "conversation_id", None) == conv_id:
                send_key = str(_k)
                break

        transport = _resolve_transport(runner, prefix)
        if transport is None:
            return ToolResult.fail(
                f"send_message: agent '{agent.id}' has no transport matching "
                f"prefix '{prefix}' for session '{conv_id}'."
            )

        if send_key is None:
            # Agent idle / conversation not loaded this process: there's no live
            # conversation to read the transport-native id from. Hand the
            # canonical session id to the transport as-is and let it resolve its
            # own target.
            send_key = conv_id

        target_label = conv_id
    else:
        # --------------------------------------------------------------
        # DEPRECATED legacy path: bare-int Discord channel id. Byte-for-byte
        # the pre-session_id behavior.
        # --------------------------------------------------------------
        explicit_channel = int(channel_id) if channel_id else 0
        target_channel_id = explicit_channel
        # Always resolve current channel too — needed for cross-channel check
        # below regardless of whether an explicit arg was passed.
        current_channel_id = 0
        try:
            session = CURRENT_SESSION.get(None)
            if session is not None and session.transport == "discord":
                current_channel_id = session.channel_id_int
        except Exception:
            pass
        if not current_channel_id:
            try:
                current_channel_id = int(CURRENT_CHANNEL_ID.get())
            except LookupError:
                current_channel_id = 0
        if not target_channel_id:
            target_channel_id = current_channel_id
        if not target_channel_id:
            return ToolResult.fail("No channel_id provided and no current channel in context.")

        # Cross-channel guard (MED-1 from the security audit). If the caller is
        # trying to post into a DIFFERENT channel than the one they're in,
        # require owner privileges. Non-owner users with send_message access
        # could otherwise post anywhere the bot can reach (admin channels they
        # don't have access to, other servers, etc). Owner-attributed turns
        # (cron, heartbeat, restart continuations) all run with is_owner=True
        # so they're unaffected.
        if explicit_channel and explicit_channel != current_channel_id:
            from ..acl import is_owner as _is_owner
            from ..tool_executor import current_caller
            # Transport-aware: Discord → numeric path (unchanged); iMessage →
            # compare the raw handle. Absent session → discord/"" → numeric path.
            speaker_id, _tname, _handle = current_caller()
            if not _is_owner(speaker_id, transport=_tname, handle=_handle):
                return ToolResult.fail(
                    f"send_message: cross-channel posting (target {explicit_channel} ≠ "
                    f"current {current_channel_id}) is restricted to the owner."
                )

        send_key = str(target_channel_id)
        transport = getattr(runner, "transport", None)
        target_label = target_channel_id

    # Split for transport limits (Discord: 2000 chars).
    LIMIT = 1900
    chunks: list[str] = []
    remaining = text
    while remaining:
        chunks.append(remaining[:LIMIT])
        remaining = remaining[LIMIT:]

    print_ts(f"{COLOR_YELLOW}send_message: {len(chunks)} chunk(s) → session={send_key}{COLOR_END}", agent=agent.id)

    if not transport:
        return ToolResult.fail(
            f"send_message: agent '{agent.id}' has no transport attached. "
            f"Cannot deliver."
        )
    # Transport-uniform path. Every transport implementation (Discord, iMessage,
    # Null) satisfies the same Protocol — send(session_id, text) + typing().
    # The transport handles its own resolution and error logging internally.
    try:
        for chunk in chunks:
            async with transport.typing(send_key):
                await asyncio.sleep(0.6)
                await transport.send(send_key, chunk)
    except Exception as e:
        return ToolResult.fail(f"Send failed: {e}")

    return ToolResult(
        model_feedback=f"Sent {len(chunks)} message chunk(s) to {target_label}.",
    )
