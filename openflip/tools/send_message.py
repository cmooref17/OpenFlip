"""Send a message to a channel from inside a tool.

Used primarily by agents during synthetic / heartbeat turns, where the
runner does NOT auto-post the final chat text. To say anything to the user
(or post in any channel) during a heartbeat, an agent must call this tool.

In a regular Discord-message turn, this tool is also available — you can
use it to post in OTHER channels (cross-posting). Posting in the channel
that triggered the turn still works through the normal final-text path.

Phase 2 (Discord-decouple): routes through Transport.send/typing so no
nextcord imports live here. Falls back to legacy runner.bot path when
transport isn't available (compat during transition).

Security: there is no per-channel ACL on this tool. Anyone who has the
tool in `allowed_tools` can post anywhere the bot has access to. Restrict
at the tool level (`users:` in agent.json) if needed.
"""
from __future__ import annotations
import asyncio

from ._base import tool, ToolResult
from ..utils import print_ts, COLOR_YELLOW, COLOR_END


@tool
async def send_message(text: str, channel_id: int = 0) -> ToolResult:
    """Send a message to a Discord channel.

    Use this during heartbeats or any time you want to push text to Discord
    from inside a tool flow. Especially important for synthetic turns (cron,
    heartbeat) where the runtime no longer auto-posts your final reply —
    you must explicitly call this tool to say anything to the user.

    Args:
        text: The message to send. Will be split into multiple sends if it
            exceeds Discord's 2000-char limit.
        channel_id: Optional. The Discord channel ID. Defaults to the channel
            that triggered the current turn (the one you're "in"). Pass an
            explicit ID to cross-post somewhere else.
    """
    text = (text or "").strip()
    if not text:
        return ToolResult.fail("`text` is required.")

    from ..tool_executor import CURRENT_AGENT, CURRENT_CHANNEL_ID, CURRENT_SESSION
    try:
        agent = CURRENT_AGENT.get()
    except LookupError:
        return ToolResult.fail("Tool invoked outside an agent context.")

    # Resolve target session_id: explicit channel_id arg wins, then current
    # session's channel_id, then CURRENT_CHANNEL_ID legacy int contextvar.
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

    session_id = str(target_channel_id)

    # Resolve transport from runner registry.
    from ..registry import RUNNERS
    runner = RUNNERS.get(agent.id)
    if not runner:
        return ToolResult.fail(f"No running agent for '{agent.id}'.")

    transport = getattr(runner, "transport", None)

    # Split for transport limits (Discord: 2000 chars).
    LIMIT = 1900
    chunks: list[str] = []
    remaining = text
    while remaining:
        chunks.append(remaining[:LIMIT])
        remaining = remaining[LIMIT:]

    print_ts(f"{COLOR_YELLOW}send_message: {len(chunks)} chunk(s) → session={session_id}{COLOR_END}", agent=agent.id)

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
            async with transport.typing(session_id):
                await asyncio.sleep(0.6)
                await transport.send(session_id, chunk)
    except Exception as e:
        return ToolResult.fail(f"Send failed: {e}")

    return ToolResult(
        model_feedback=f"Sent {len(chunks)} message chunk(s) to channel {target_channel_id}.",
    )
