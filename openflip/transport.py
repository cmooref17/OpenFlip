"""Transport protocol — Phase 2 of the discord-decouple.

The AgentRunner no longer holds a nextcord.Bot directly. It holds a `Transport`
which abstracts the messaging platform behind a stable interface:

- start/stop lifecycle
- send / send_file outbound text + attachments to a session
- typing indicator context manager
- session resolution (find a session for a given user, e.g. their DM)
- message fetch by id (transport-specific; not all transports support this)

DiscordTransport (in `openflip/transports/discord.py`) wraps a nextcord.Bot
and exposes this interface. Future transports (iMessage, Slack, etc.) plug
in via the same Protocol — the rest of openflip never imports nextcord.

Why this matters: Phase 1 wrapped inbound messages in Session/InboundMessage
at the boundary, but outbound code still called `channel.send(...)` directly
across runtime.py, tools, and the tool executor. Phase 2 closes the outbound
side so the framework is genuinely transport-agnostic.

This file defines only the Protocol + dataclasses. Implementations live under
`openflip/transports/`.
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Protocol, Optional, AsyncContextManager, Any, runtime_checkable

from .session import Session, InboundMessage


@runtime_checkable
class Transport(Protocol):
    """Transport protocol — abstracts the messaging platform.

    Implementations:
      - DiscordTransport (openflip/transports/discord.py) — wraps nextcord.Bot.
      - Future: iMessage, Slack, etc.

    Phase 2 contract: the AgentRunner and all tools talk to messages
    through this protocol only. The runner holds one Transport per agent
    (since each agent has its own Discord bot token / iMessage identity).
    """

    name: str  # transport identifier: "discord", "imessage", etc.

    async def start(self) -> None:
        """Start the transport (connect to Discord, etc.). Returns when ready."""
        ...

    async def stop(self) -> None:
        """Stop the transport. Cleans up connections."""
        ...

    async def send(self, session_id: str, text: str) -> None:
        """Send a text message to a session. Splits long text per transport limits."""
        ...

    async def send_file(self, session_id: str, path: str, content: str = "") -> Optional[str]:
        """Send a file attachment (image/video/audio/etc) to a session.

        Optional accompanying text via `content`.

        Returns the canonical URL/ID of the posted attachment (e.g. a Discord
        CDN URL), or None if the transport doesn't support URL retrieval or
        if the send failed. Callers use this to stash attachment URLs so the
        model can reference prior outputs on follow-up turns.
        """
        ...

    @asynccontextmanager
    async def typing(self, session_id: str) -> AsyncContextManager[Any]:
        """Context manager: show typing indicator while inside the block.

        Implementations should not raise on transport-level rate limits etc.
        — typing is UX polish, not critical path.
        """
        ...

    async def resolve_session_for_user(self, user_id: int) -> Optional[Session]:
        """Find a session for a given user (typically their DM).

        Used by talk_to_agent's 3-priority routing: if the caller's channel
        isn't reachable, fall back to the target's DM with the human.

        Returns None if no session can be resolved (user not found,
        transport doesn't support DMs, etc.).
        """
        ...

    async def fetch_message(self, session_id: str, message_id: str) -> Optional[InboundMessage]:
        """Fetch a specific message by ID, wrapped in InboundMessage.

        Used by fetch_discord_message tool. Not all transports support this
        — returns None for transports that don't.
        """
        ...
