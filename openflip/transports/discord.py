"""DiscordTransport — wraps nextcord.Bot, exposes Transport protocol.

Phase 2 of the discord-decouple. Splits ownership: transport owns the Bot
lifecycle (reconnect backoff, gateway rate limits, slash command
registration). AgentRunner reaches in for nextcord internals via
`runner.transport.bot` as a deliberate Phase-2 escape hatch.

Inbound flow:
  Discord on_message → DiscordTransport._on_message
    → runner._handle_message(message) which builds InboundMessage and dispatches.
"""
from __future__ import annotations
import asyncio
import contextlib
from typing import Optional, TYPE_CHECKING

import nextcord
from nextcord.ext import commands

from ..session import Session, InboundMessage, make_discord_session
from ..utils import print_ts, COLOR_YELLOW, COLOR_RED, COLOR_END

if TYPE_CHECKING:
    from ..runtime import AgentRunner


class DiscordTransport:
    """Discord Transport implementation. Owns the nextcord.Bot."""

    name: str = "discord"

    def __init__(self, token: str):
        self.token = token
        intents = nextcord.Intents.all()
        self.bot = commands.Bot(intents=intents)
        self._runner: Optional["AgentRunner"] = None
        self._task: Optional[asyncio.Task] = None

    def attach_runner(self, runner: "AgentRunner") -> None:
        """Wire the owning AgentRunner. Called by AgentRunner.__init__."""
        self._runner = runner
        self._register_events()

    def _register_events(self):
        @self.bot.event
        async def on_ready():
            if self._runner:
                print_ts(f"Online as {self.bot.user.name} ({self.bot.user.id})", agent=self._runner.agent.id)

        @self.bot.event
        async def on_message(message: nextcord.Message):
            if not self.bot.user or not self._runner:
                return
            from ..pipeline import should_respond, build_inbound_from_discord
            from ..config_global import get_owner_id
            owner_id = get_owner_id("discord")
            inbound = build_inbound_from_discord(message, self.bot.user.id, owner_id)
            if not should_respond(self._runner.agent, inbound, self.bot.user.id):
                return
            # Pass the already-built InboundMessage AND the raw nextcord.Message
            # to the runner. The runner uses inbound for transport-neutral
            # paths and keeps the raw message available for the deliberate
            # bridge code paths (image-bytes extraction, fetch_discord_message
            # tool re-injection) that genuinely need nextcord types.
            await self._runner._handle_message(message, inbound=inbound, transport=self)

        if self._runner is not None:
            from .. import commands as cmds
            cmds.register_commands(self.bot, self._runner)

    @property
    def bot_user_id(self) -> int:
        return int(getattr(self.bot.user, "id", 0) or 0)

    # ---- Transport protocol methods ----

    async def start(self) -> None:
        """Run the bot with reconnect-on-failure logic.

        Backoff: 30s → 60s → 120s → 240s → 480s, capped 10min.
        Rate-limit signals (HTTP 429, code 40062) bump to 5min minimum
        because IDENTIFY rate limits compound under hammering.
        Resets on a connection that stays alive > 60s.
        """
        backoff = 30
        max_backoff = 600
        rate_limit_floor = 300
        agent_id = self._runner.agent.id if self._runner else "unknown"
        while True:
            connect_t = asyncio.get_event_loop().time()
            try:
                if self.bot.is_closed():
                    intents = nextcord.Intents.all()
                    self.bot = commands.Bot(intents=intents)
                    self._register_events()
                self._task = asyncio.create_task(self.bot.start(self.token))
                await self._task
                return
            except asyncio.CancelledError:
                return
            except Exception as e:
                ran_for = asyncio.get_event_loop().time() - connect_t
                status = getattr(e, "status", None)
                code = getattr(e, "code", None)
                rate_limited = status == 429 or code == 40062
                print_ts(
                    f"{COLOR_RED}DiscordTransport for {agent_id} crashed after {ran_for:.0f}s: {e}{COLOR_END}",
                    error=True, agent=agent_id,
                )
                if ran_for > 60:
                    backoff = 30
                wait = max(backoff, rate_limit_floor) if rate_limited else backoff
                tag = " (rate-limited — 5min floor)" if rate_limited else ""
                print_ts(
                    f"{COLOR_YELLOW}reconnecting in {wait}s{tag}…{COLOR_END}",
                    agent=agent_id,
                )
                try:
                    await asyncio.sleep(wait)
                except asyncio.CancelledError:
                    return
                backoff = min(backoff * 2, max_backoff)

    async def stop(self) -> None:
        try:
            await self.bot.close()
        except Exception:
            pass
        if self._task:
            self._task.cancel()

    async def _resolve_channel(self, session_id: str):
        """Resolve a Discord channel from a session_id string. Returns None on failure."""
        try:
            ch_id = int(session_id)
        except (ValueError, TypeError):
            return None
        channel = self.bot.get_channel(ch_id)
        if channel is None:
            try:
                channel = await asyncio.wait_for(
                    self.bot.fetch_channel(ch_id),
                    timeout=15.0,
                )
            except (asyncio.TimeoutError, Exception):
                return None
        return channel

    async def send(self, session_id: str, text: str) -> None:
        """Send text to a Discord channel. Logs on failure, doesn't raise."""
        channel = await self._resolve_channel(session_id)
        if channel is None:
            print_ts(f"{COLOR_RED}DiscordTransport.send: channel {session_id} unresolved{COLOR_END}", error=True)
            return
        try:
            await asyncio.wait_for(channel.send(text), timeout=30.0)
        except asyncio.TimeoutError:
            print_ts(f"channel.send timed out after 30s", error=True)
        except Exception as e:
            print_ts(f"channel.send failed: {e}", error=True)

    async def send_file(self, session_id: str, path: str, content: str = "") -> Optional[str]:
        """Send a file attachment to a Discord channel.

        Returns the Discord CDN URL of the posted attachment, or None on failure.
        The URL is used by tool_executor to tell the model where to find the
        file so it can reference it in follow-up image_url args.
        """
        channel = await self._resolve_channel(session_id)
        if channel is None:
            print_ts(f"{COLOR_RED}DiscordTransport.send_file: channel {session_id} unresolved{COLOR_END}", error=True)
            return None
        try:
            file = nextcord.File(path)
            sent = await asyncio.wait_for(channel.send(content=content or None, file=file), timeout=60.0)
            attachments = getattr(sent, "attachments", None) or []
            return attachments[0].url if attachments else None
        except Exception as e:
            print_ts(f"channel.send_file failed: {e}", error=True)
            return None

    @contextlib.asynccontextmanager
    async def typing(self, session_id: str):
        """Show typing indicator. UX polish — failures swallowed."""
        channel = await self._resolve_channel(session_id)
        if channel is None:
            yield
            return
        try:
            async with channel.typing():
                yield
        except Exception as e:
            print_ts(f"{COLOR_YELLOW}typing indicator failed (continuing): {e}{COLOR_END}")
            yield

    async def resolve_session_for_user(self, user_id: int) -> Optional[Session]:
        """Find this transport's session with a user (their DM channel).

        Used by talk_to_agent's priority-2 resolution: if the caller's
        channel isn't reachable for the recipient bot, fall back to the
        recipient's DM with the originating human.
        Returns None if user unfetchable or DM creation fails.
        """
        if not user_id:
            return None
        try:
            user = self.bot.get_user(int(user_id))
            if user is None:
                user = await asyncio.wait_for(self.bot.fetch_user(int(user_id)), timeout=5)
            if user is None:
                return None
            dm = user.dm_channel
            if dm is None:
                dm = await asyncio.wait_for(user.create_dm(), timeout=5)
            if dm is None:
                return None
            return make_discord_session(
                channel_id=int(dm.id),
                speaker_id=int(user_id),
                speaker_role_ids=[],
                is_owner=False,
                is_dm=True,
                display_name=getattr(user, "display_name", None) or getattr(user, "name", "user"),
            )
        except Exception as e:
            print_ts(f"{COLOR_YELLOW}resolve_session_for_user({user_id}) failed: {e}{COLOR_END}")
            return None

    async def fetch_message(self, session_id: str, message_id: str) -> Optional[InboundMessage]:
        """Fetch a Discord message by id, wrapped in InboundMessage."""
        channel = await self._resolve_channel(session_id)
        if channel is None:
            return None
        try:
            msg = await asyncio.wait_for(channel.fetch_message(int(message_id)), timeout=10)
            from ..pipeline import build_inbound_from_discord
            owner_id = 0
            try:
                from ..config_global import get_owner_id
                owner_id = get_owner_id("discord")
            except Exception:
                pass
            return build_inbound_from_discord(msg, self.bot_user_id, owner_id)
        except Exception as e:
            print_ts(f"{COLOR_YELLOW}fetch_message({session_id}/{message_id}) failed: {e}{COLOR_END}")
            return None
