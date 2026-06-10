"""NullTransport — a no-op Transport for headless / background-worker agents.

Some agents have no messaging surface of their own: background specialists
that sit idle until another agent hands them work via talk_to_agent. They
still need to (a) land in the RUNNERS registry so talk_to_agent can reach
them, and (b) run turns through the same machinery as every other agent.

The rest of openflip assumes `AgentRunner._transports` is never empty and
`AgentRunner.transport` is never None — the property returns
`self._transports[0]` with no guard, and ~a dozen call sites do
`self.transport.X`. Rather than thread None-checks through all of them, a
headless agent gets exactly one transport: this one, which satisfies the
Transport contract while doing nothing. start/stop idle on an Event;
send/send_file/typing are no-ops; resolve/fetch return None.

Registered under the name "internal" in main._TRANSPORT_BUILDERS. An agent
opts in with `"transports": ["internal"]` in agent.json. Because ACL auth is
keyed by transport name (see acl._check_acl), a headless agent's tool entries
must declare `auth.internal`, e.g.
    {"name": "talk_to_agent", "auth": {"internal": {"all_users": true}}}
A tool with only `auth.discord` is invisible on the internal transport.
"""
from __future__ import annotations

import asyncio
import contextlib
from typing import Optional, AsyncContextManager, Any, TYPE_CHECKING

from ..session import Session, InboundMessage

if TYPE_CHECKING:
    from ..runtime import AgentRunner


def make_internal_session(agent_id: str) -> Session:
    """The single stable conversation context for a headless agent.

    Headless agents run every turn in one internal channel. The
    transport-prefixed `conversation_id` ("internal:<agent_id>") keeps their
    history in agents/<id>/conversations/internal:<agent_id>.jsonl, isolated
    from any real-transport history and from a real Discord channel id.

    The int `transport_id` is a per-process-stable hash of the agent id
    (same pattern as TransportChannel for non-numeric ids — channel_shim.py).
    It's only used as an in-memory dict key (_active_turns / _pending_inject /
    conversations); the on-disk filename is governed by conversation_id, which
    IS stable across restarts.
    """
    cid = abs(hash(f"internal:{agent_id}")) % (2**31)
    return Session(
        transport="internal",
        transport_id=str(cid),
        conversation_id=f"internal:{agent_id}",
        speaker_id=0,
        speaker_role_ids=[],
        is_owner=False,
        is_dm=True,
        display_name=f"{agent_id} (internal)",
    )


class NullTransport:
    """No-op Transport for headless agents. Satisfies the Transport protocol."""

    name: str = "internal"

    def __init__(self):
        self._runner: Optional["AgentRunner"] = None
        # start() idles on this until stop() sets it, mirroring the
        # "runs until stopped" lifecycle of the real transports so the
        # AgentRunner.start() gather behaves identically (stays pending).
        self._stop_event: Optional[asyncio.Event] = None

    def attach_runner(self, runner: "AgentRunner") -> None:
        """Wire the owning AgentRunner. Called by AgentRunner.__init__."""
        self._runner = runner

    async def start(self) -> None:
        """Idle until stop(). A headless agent has no inbound surface to
        connect — turns arrive via run_synthetic_turn off the inbound queue,
        which is started lazily by the runner, not here."""
        self._stop_event = asyncio.Event()
        try:
            await self._stop_event.wait()
        except asyncio.CancelledError:
            return

    async def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    async def send(self, session_id: str, text: str) -> None:
        """No-op. A headless agent has no channel to post to. Its visible
        output reaches humans by auto-routing back to the originating agent
        (see runtime._run_turn's inter-agent auto-route block)."""
        return None

    async def send_file(self, session_id: str, path: str, content: str = "") -> Optional[str]:
        return None

    @contextlib.asynccontextmanager
    async def typing(self, session_id: str) -> AsyncContextManager[Any]:
        """No-op typing indicator — there's no surface to show it on."""
        yield

    async def resolve_session_for_user(self, user_id: int) -> Optional[Session]:
        return None

    async def fetch_message(self, session_id: str, message_id: str) -> Optional[InboundMessage]:
        return None
