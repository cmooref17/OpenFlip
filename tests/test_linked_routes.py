"""Linked-conversation delivery routes (openflip/linked_routes.py).

Standalone runnable script (no pytest in this venv):
    .lvenv/bin/python tests/test_linked_routes.py

Covers:
  (a) a forwarded (secondary) inbound records its native route
  (b) a primary-side inbound flips the route back to the primary transport
  (c) unlinked conversations never get an entry
  (d) the route survives a process reload (read back from disk)
  (e) delivery_session keeps conversation_id but delivers via the last route
  (f) identical inbound doesn't rewrite the file
"""
from __future__ import annotations

import os
import sys
import tempfile
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip import linked_routes as lr  # noqa: E402
from openflip.session import Session  # noqa: E402

FAILS: list[str] = []
PRIMARY = "imessage:+15550001111"
LINKS = {"discord:100": PRIMARY}


def check(ok: bool, label: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        FAILS.append(label)


def sess(transport, tid, conv, handle="", speaker=0):
    return Session(transport=transport, transport_id=str(tid), conversation_id=conv,
                   speaker_id=speaker, speaker_role_ids=[], is_owner=False, is_dm=True,
                   display_name="x", handle=handle)


def run() -> None:
    d = tempfile.mkdtemp(prefix="lrtest_")
    with mock.patch("openflip.config_global.get_identity_links", return_value=LINKS):
        lr._reset_cache_for_tests()
        # (a) Discord DM of the secondary identity, forwarded into the primary.
        changed = lr.record(d, sess("discord", 555, PRIMARY, speaker=100))
        r = lr.lookup(d, PRIMARY)
        check(changed and r is not None and r.transport == "discord" and r.transport_id == "555",
              "(a) forwarded inbound records discord route")
        # (f) same inbound again: no rewrite.
        mtime = os.path.getmtime(os.path.join(d, "linked_routes.json"))
        check(lr.record(d, sess("discord", 555, PRIMARY, speaker=100)) is False
              and os.path.getmtime(os.path.join(d, "linked_routes.json")) == mtime,
              "(f) identical inbound leaves the file alone")
        # (b) Primary side (iMessage) talks next: its session keys natively = PRIMARY.
        lr.record(d, sess("imessage", 7, PRIMARY, handle="+15550001111"))
        r = lr.lookup(d, PRIMARY)
        check(r is not None and r.transport == "imessage" and r.handle == "+15550001111",
              "(b) primary-side inbound flips route back to imessage")
        # (c) Unlinked conversation.
        check(lr.record(d, sess("discord", 999, "discord:999", speaker=5)) is False
              and lr.lookup(d, "discord:999") is None, "(c) unlinked conversation not recorded")
        # Back to discord, then (d) reload from disk.
        lr.record(d, sess("discord", 555, PRIMARY, speaker=100))
        lr._reset_cache_for_tests()
        r = lr.lookup(d, PRIMARY)
        check(r is not None and r.transport == "discord", "(d) route survives reload")
        # (e) delivery session.
        s = lr.delivery_session(d, PRIMARY, tool_grants=["send_message"])
        check(s is not None and s.transport == "discord" and s.transport_id == "555"
              and s.conversation_id == PRIMARY and s.tool_grants == ["send_message"],
              "(e) delivery_session: shared history, discord delivery")
        check(lr.delivery_session(d, "imessage:+19999") is None, "(e) no route -> None (caller keeps old behavior)")


def run_outbound() -> None:
    """(g) cron + (h) synthetic-channel transport + (i) imsg addressing."""
    import asyncio
    from types import SimpleNamespace
    from openflip import cron
    from openflip.transports.imessage import IMessageTransport
    d = tempfile.mkdtemp(prefix="lrtest2_")
    with mock.patch("openflip.config_global.get_identity_links", return_value=LINKS):
        lr._reset_cache_for_tests()
        runner = SimpleNamespace(agent=SimpleNamespace(path=os.path.join(d, "agent.json")),
                                 _transports=[])
        job = {"name": "j", "sessionId": PRIMARY, "toolGrants": ["send_message"]}
        # No route yet: old behavior (imessage session from the prefix).
        s0 = cron._resolve_session_target(job, runner)
        check(s0.transport == "imessage" and s0.conversation_id == PRIMARY,
              "(g) cron without a route keeps the prefix transport")
        lr.record(d, sess("discord", 555, PRIMARY, speaker=100))
        s1 = cron._resolve_session_target(job, runner)
        check(s1.transport == "discord" and s1.transport_id == "555"
              and s1.conversation_id == PRIMARY and s1.tool_grants == ["send_message"],
              "(g) cron with a route delivers via discord, shared history, grants kept")

    # (h) transport_named picks the Session's own transport on a multi-transport runner.
    from openflip.runtime import AgentRunner
    imsg_t = SimpleNamespace(name="imessage")
    disc_t = SimpleNamespace(name="discord", bot=object())
    fake = SimpleNamespace(_transports=[imsg_t, disc_t])
    check(AgentRunner.transport_named(fake, "discord") is disc_t
          and AgentRunner.transport_named(fake, "imessage") is imsg_t
          and AgentRunner.transport_named(fake, "email") is None,
          "(h) transport_named finds each transport by name")

    # (i) imsg never receives a prefixed conversation id as an address.
    a = IMessageTransport._addr_args
    check(a("imessage:+15550001111") == ["--to", "+15550001111"]
          and a("imessage:7") == ["--chat-id", "7"] and a("7") == ["--chat-id", "7"]
          and a("") is None, "(i) imsg addressing strips the imessage: prefix")


if __name__ == "__main__":
    print("test_linked_routes")
    run()
    run_outbound()
    print("\nRESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FAIL")
    sys.exit(1 if FAILS else 0)
