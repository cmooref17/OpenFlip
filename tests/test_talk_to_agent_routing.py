"""Pins the talk_to_agent default-dispatch routing split (2026-06-15).

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_talk_to_agent_routing.py

Background
----------
The 2026-06-11 inter-agent leak fix forced EVERY default talk_to_agent dispatch
(no session_id / channel_id) into a private internal:peer-<sender> conversation.
That over-corrected: when the OPERATOR tells an agent to "talk to <peer>", he
wants that message in the recipient's MAIN channel he shares with it, so both
agents share context. This split restores that for owner-rooted chains while
keeping agent/cron/heartbeat-rooted chatter private (the leak stays closed).

This test pins the two pure pieces of the split:

  `_default_dispatch_is_operator_routed` — the discriminator. Owner-routed ONLY
    when the speaker is the owner AND the chain-root visibility is a live human
    channel ("" or "operator_channel"). Cron/heartbeat resolve their attributed
    speaker to owner_id too, so the visibility conjunct is what keeps them
    private — is_owner alone is not enough.

  `_resolve_recipient_operator_channel` — best-effort resolution of the
    recipient's shared channel with the operator (shared guild channel, else the
    recipient's DM with the operator), returning 0 for unreachable / non-Discord
    recipients so the caller falls back to the private peer conversation.
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip.tools.talk_to_agent import (
    _default_dispatch_is_operator_routed,
    _resolve_recipient_operator_channel,
)

FAILURES: list[str] = []


def check(label: str, cond: bool) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        FAILURES.append(label)


# --- discriminator -----------------------------------------------------------

def test_owner_triggered_routes_to_shared_channel():
    print("owner-triggered (human at root) -> shared operator channel")
    check("owner + real-inbound visibility ('') -> operator-routed",
          _default_dispatch_is_operator_routed(speaker_is_owner=True, caller_visibility="") is True)
    check("owner + explicit operator_channel -> operator-routed",
          _default_dispatch_is_operator_routed(speaker_is_owner=True, caller_visibility="operator_channel") is True)


def test_agent_and_cron_triggered_stay_private():
    print("agent/cron/heartbeat-triggered -> private internal:peer")
    # Cron/heartbeat resolve the attributed speaker to owner_id, so speaker_is_owner
    # is True for them — the visibility tag is the ONLY thing keeping them private.
    check("owner-attributed cron -> peer (leak stays closed)",
          _default_dispatch_is_operator_routed(speaker_is_owner=True, caller_visibility="cron") is False)
    check("owner-attributed heartbeat -> peer",
          _default_dispatch_is_operator_routed(speaker_is_owner=True, caller_visibility="heartbeat") is False)
    check("owner-attributed silent_agent_chain -> peer",
          _default_dispatch_is_operator_routed(speaker_is_owner=True, caller_visibility="silent_agent_chain") is False)
    # Non-owner speaker never operator-routes regardless of visibility.
    check("non-owner human ('' visibility) -> peer",
          _default_dispatch_is_operator_routed(speaker_is_owner=False, caller_visibility="") is False)
    check("non-owner operator_channel -> peer",
          _default_dispatch_is_operator_routed(speaker_is_owner=False, caller_visibility="operator_channel") is False)


# --- recipient channel resolution --------------------------------------------

class _FakeDM:
    def __init__(self, dm_id: int):
        self.id = dm_id


class _FakeUser:
    def __init__(self, dm: _FakeDM | None):
        self.dm_channel = dm

    async def create_dm(self):
        self.dm_channel = _FakeDM(999)
        return self.dm_channel


class _FakeGuild:
    def __init__(self, gid: int):
        self.id = gid


class _FakeGuildChannel:
    def __init__(self, cid: int, guild: _FakeGuild):
        self.id = cid
        self.guild = guild


class _FakeBot:
    def __init__(self, *, guilds=None, channels=None, dm_user=None, private_channels=None):
        self._guilds = {g.id: g for g in (guilds or [])}
        self._channels = {c.id: c for c in (channels or [])}
        self._dm_user = dm_user
        self.private_channels = private_channels or []

    def get_channel(self, cid):
        return self._channels.get(cid)

    def get_guild(self, gid):
        return self._guilds.get(gid)

    async def fetch_user(self, uid):
        return self._dm_user


class _FakeTarget:
    """Mimics a runner: `.bot` is a property that raises for non-Discord."""
    def __init__(self, bot=None, has_bot=True):
        self._bot = bot
        self._has_bot = has_bot

    @property
    def bot(self):
        if not self._has_bot:
            raise AttributeError("no bot (headless / non-Discord)")
        return self._bot


class _FakeSession:
    def __init__(self, transport: str, transport_id: str):
        self.transport = transport
        self.transport_id = transport_id

    @property
    def channel_id_int(self) -> int:
        return int(self.transport_id)


def _run(coro):
    return asyncio.run(coro)


def test_resolve_priority1_shared_guild_channel():
    print("resolve: priority 1 — caller's channel in a guild the recipient shares")
    guild = _FakeGuild(700)
    ch = _FakeGuildChannel(123, guild)
    bot = _FakeBot(guilds=[guild], channels=[ch])
    target = _FakeTarget(bot=bot)
    caller_session = _FakeSession("discord", "123")
    out = _run(_resolve_recipient_operator_channel(
        target=target, caller_session=caller_session,
        operator_speaker_id=42, sender_id="rhea", agent_id="miniflip"))
    check("shared guild channel reused (123)", out == 123)


def test_resolve_priority1_unreachable_guild_falls_to_dm():
    print("resolve: caller's guild channel unreachable -> priority 2 recipient DM")
    guild = _FakeGuild(700)
    ch = _FakeGuildChannel(123, guild)
    # Recipient is NOT in guild 700 (empty guilds) -> priority 1 fails.
    bot = _FakeBot(guilds=[], channels=[ch], dm_user=_FakeUser(_FakeDM(555)))
    target = _FakeTarget(bot=bot)
    caller_session = _FakeSession("discord", "123")
    out = _run(_resolve_recipient_operator_channel(
        target=target, caller_session=caller_session,
        operator_speaker_id=42, sender_id="rhea", agent_id="miniflip"))
    check("falls through to recipient's DM with operator (555)", out == 555)


def test_resolve_priority2_dm_when_caller_in_dm():
    print("resolve: priority 2 — caller in a DM, recipient's own DM with operator")
    # Caller's channel is a DM (not in recipient's private_channels) -> priority 1
    # fails; recipient resolves its OWN DM with the operator.
    bot = _FakeBot(dm_user=_FakeUser(_FakeDM(888)))
    target = _FakeTarget(bot=bot)
    caller_session = _FakeSession("discord", "404")  # caller's DM channel id
    out = _run(_resolve_recipient_operator_channel(
        target=target, caller_session=caller_session,
        operator_speaker_id=42, sender_id="rhea", agent_id="miniflip"))
    check("recipient's DM with operator (888)", out == 888)


def test_resolve_non_discord_recipient_returns_zero():
    print("resolve: headless / non-Discord recipient -> 0 (caller falls back to peer)")
    target = _FakeTarget(has_bot=False)
    caller_session = _FakeSession("discord", "123")
    out = _run(_resolve_recipient_operator_channel(
        target=target, caller_session=caller_session,
        operator_speaker_id=42, sender_id="rhea", agent_id="headless_worker"))
    check("no bot -> 0", out == 0)


def test_resolve_no_operator_no_channel_returns_zero():
    print("resolve: nothing to resolve (no caller channel, no operator) -> 0")
    bot = _FakeBot(dm_user=None)
    target = _FakeTarget(bot=bot)
    out = _run(_resolve_recipient_operator_channel(
        target=target, caller_session=None,
        operator_speaker_id=0, sender_id="rhea", agent_id="miniflip"))
    check("unresolvable -> 0", out == 0)


if __name__ == "__main__":
    test_owner_triggered_routes_to_shared_channel()
    test_agent_and_cron_triggered_stay_private()
    test_resolve_priority1_shared_guild_channel()
    test_resolve_priority1_unreachable_guild_falls_to_dm()
    test_resolve_priority2_dm_when_caller_in_dm()
    test_resolve_non_discord_recipient_returns_zero()
    test_resolve_no_operator_no_channel_returns_zero()
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: ALL PASS")
