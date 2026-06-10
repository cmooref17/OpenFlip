"""Transport adapters — Phase 2 of the discord-decouple.

The framework talks to messages through `openflip.transport.Transport`.
Adapters live here, one per messaging platform:
  - discord.py: DiscordTransport (wraps nextcord.Bot)
  - future:     imessage.py, slack.py, etc.
"""
from .discord import DiscordTransport
from .imessage import IMessageTransport
from .null import NullTransport, make_internal_session

__all__ = ["DiscordTransport", "IMessageTransport", "NullTransport", "make_internal_session"]
