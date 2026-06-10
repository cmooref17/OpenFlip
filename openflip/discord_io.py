"""Discord-specific I/O helpers extracted from runtime.py.

Pure transport-layer functions. Anything in `turn/` or other modules that
needs to push bytes to Discord should call these — they hold the timeout
discipline, 1900-char chunking, and typing-indicator fallback behavior.

Keeping these out of runtime.py means runtime.py and the turn/ phase
modules can stay transport-agnostic-ish without dragging nextcord imports
through every file.
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Optional

import nextcord

from .utils import print_ts, COLOR_YELLOW, COLOR_END, sanitize_outbound_text


async def safe_channel_send(channel, content=None, *, timeout: float = 30.0, **kwargs):
    """channel.send with a timeout. Logs on timeout, doesn't raise.

    Accepts the same kwargs as nextcord's `channel.send` (embed, files,
    embeds, etc.) so it's a drop-in replacement for raw `await channel.send(...)`.
    The 30s default timeout accommodates Discord's built-in 429 retries
    without false-positive timeouts in normal operation.
    """
    try:
        if content is not None:
            return await asyncio.wait_for(channel.send(content, **kwargs), timeout=timeout)
        return await asyncio.wait_for(channel.send(**kwargs), timeout=timeout)
    except asyncio.TimeoutError:
        print_ts(f"channel.send timed out after {timeout}s", error=True)
    except Exception as e:
        print_ts(f"channel.send failed: {e}", error=True)
    return None


def split_for_discord(text: str, limit: int = 1900) -> list[str]:
    """Chop outbound text into <=limit-char chunks for Discord's 2000-char cap.

    Also strips protocol-tag fragments via sanitize_outbound_text — defense
    against the agent emitting malformed tool-call envelopes (wrong closer,
    format-mixed Claude-Code leak). Broken envelope text never reaches Discord.
    """
    text = sanitize_outbound_text(text)
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    out = []
    while text:
        out.append(text[:limit])
        text = text[limit:]
    return out


@contextlib.asynccontextmanager
async def safe_typing(channel, *, agent_id: str = ""):
    """Show the "typing..." indicator while inside the block.

    Falls back to no-op on Discord 429 or other failures — we never let
    a typing-indicator hiccup break a turn.
    """
    try:
        async with channel.typing():
            yield
    except Exception as _e:
        print_ts(
            f"{COLOR_YELLOW}typing indicator failed (continuing without): {_e}{COLOR_END}",
            agent=agent_id,
        )
        yield


@contextlib.asynccontextmanager
async def silent_typing(bot, speaker_id: int, *, agent_id: str = ""):
    """Inter-agent synthetic turn: no channel posts, but show typing in the
    ORIGINATING human's DM with this agent so the human can tell that
    agent-to-agent back-and-forth is actually happening.

    Without this, inter-agent traffic is completely invisible to the operator
    and they can't tell if agents are working or stalled.

    Resolution: find the originating human (speaker_id) → find this agent's
    DM with them → show typing there. Any failure (no user found, DM not
    creatable, 429 on typing) falls back to no-op so the turn doesn't break
    on UX polish.
    """
    human_dm = None
    try:
        if int(speaker_id or 0):
            user = bot.get_user(int(speaker_id))
            if user is None:
                try:
                    user = await asyncio.wait_for(
                        bot.fetch_user(int(speaker_id)),
                        timeout=5,
                    )
                except Exception:
                    user = None
            if user is not None:
                human_dm = user.dm_channel
                if human_dm is None:
                    try:
                        human_dm = await asyncio.wait_for(
                            user.create_dm(),
                            timeout=5,
                        )
                    except Exception:
                        human_dm = None
    except Exception:
        human_dm = None
    if human_dm is None:
        yield
        return
    try:
        async with human_dm.typing():
            yield
    except Exception as _e:
        print_ts(
            f"{COLOR_YELLOW}inter-agent typing indicator failed (continuing without): {_e}{COLOR_END}",
            agent=agent_id,
        )
        yield
