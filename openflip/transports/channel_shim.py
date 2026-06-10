"""TransportChannel — duck-typed shim around a Transport + Session.

runtime.py and friends were originally written against nextcord.Channel
objects: they call channel.id, channel.send(text, **kwargs), channel.typing(),
and a handful of other nextcord-shaped attrs. Refactoring those ~50 sites
across runtime.py to use self.transport.send(session_id, ...) directly is a
big invasive change. As a pragmatic intermediate step, this shim wraps a
Transport + Session and exposes the nextcord-channel API that runtime.py
actually uses. iMessage (and any future non-Discord transport) builds one
in `_handle_inbound` and hands it to the queue worker as the "channel"
the rest of the code expects.

Limitations:
  * Discord-specific kwargs to .send (embed, view, reference, allowed_mentions,
    tts, etc.) are silently ignored.
  * .send with a `files=`/`file=` kwarg attempts to route through
    `transport.send_file(path)` — only works if the file has a real on-disk
    path. nextcord File objects backed by BytesIO won't transfer.
  * .add_reaction is a no-op on this shim. iMessage doesn't expose tapbacks
    via the imsg CLI's send command yet.
  * .fetch_message delegates to `transport.fetch_message`, which iMessage
    supports.
"""
from __future__ import annotations
import contextlib
from typing import Optional, Any

from ..session import Session


class TransportChannel:
    """Quacks like a nextcord channel for the API surface runtime.py uses."""

    def __init__(self, *, transport, session: Session):
        self._transport = transport
        self._session = session
        # `id` must be an int because runtime.py uses it as a dict key in
        # several places (self._active_turns[ch_id], self._pending_inject[ch_id]).
        # iMessage sessions use the integer chat_id (str-cast into Session).
        try:
            self.id = int(session.transport_id)
        except (TypeError, ValueError):
            # Non-numeric transport_id (future transports): hash to a stable int.
            self.id = abs(hash(session.transport_id)) % (2**31)
        self.name = session.display_name or session.transport_id
        self.type = "dm" if session.is_dm else "channel"
        # nextcord channel has .guild — None for DMs. Mirror that.
        self.guild = None
        # Some code checks `channel.recipient` for DMs — leave as None.
        self.recipient = None

    async def send(self, content: Optional[str] = None, **kwargs) -> Any:
        """Mirror nextcord channel.send. Discord-specific kwargs (embed,
        embeds, view, reference, allowed_mentions, tts, suppress_embeds,
        delete_after) are silently ignored. If `file=` or `files=` is
        passed and the file has a string path, route it through
        transport.send_file."""
        if content is not None:
            try:
                await self._transport.send(self._session.transport_id, str(content))
            except Exception:
                # Don't let send failures crash the caller — match nextcord's
                # behavior where send errors usually propagate but the caller
                # is already wrapping in safe_channel_send for timeout safety.
                raise
        files = list(kwargs.get("files") or [])
        single = kwargs.get("file")
        if single is not None:
            files.append(single)
        for f in files:
            try:
                fpath = None
                if isinstance(f, str):
                    fpath = f
                elif hasattr(f, "fp") and isinstance(f.fp, str):
                    fpath = f.fp
                if fpath:
                    await self._transport.send_file(self._session.transport_id, fpath, "")
            except Exception:
                # Per-file failures don't kill the rest of the send.
                pass
        return None

    @contextlib.asynccontextmanager
    async def typing(self):
        """Forward to transport.typing — IMessageTransport returns a no-op
        async context manager for the typing call."""
        async with self._transport.typing(self._session.transport_id):
            yield

    async def fetch_message(self, message_id):
        """Delegate to transport.fetch_message. Returns an InboundMessage or None."""
        return await self._transport.fetch_message(
            self._session.transport_id, str(message_id),
        )

    # Soft-inject confirmation reaction call site in runtime.py uses
    # `message.add_reaction("👀")` on the inbound nextcord.Message, not on
    # the channel. For non-Discord transports there's no incoming
    # message object, so the runtime.py code path that adds the reaction
    # is skipped (see _handle_inbound). No method needed here.

    def __repr__(self):
        return f"TransportChannel(transport={self._transport.name!r}, session_id={self._session.transport_id!r})"
