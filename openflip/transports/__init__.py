"""Transport adapters — Phase 2 of the discord-decouple.

The framework talks to messages through `openflip.transport.Transport`.
Adapters live here, one per messaging platform:
  - discord.py: DiscordTransport (wraps nextcord.Bot)
  - imessage.py: IMessageTransport (macOS iMessage via imsg CLI)
  - external.py: ExternalTransport (authenticated HTTPS ingress, 0.0.0.0)
  - null.py: NullTransport (headless/internal agents)
"""
from .discord import DiscordTransport
from .imessage import IMessageTransport
from .null import NullTransport, make_internal_session
from .external import ExternalTransport

__all__ = ["DiscordTransport", "IMessageTransport", "NullTransport", "make_internal_session", "ExternalTransport"]
