"""Peer-triggered turns: plain final text reaches the human when it ran in their conversation.

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_peer_reply_delivery.py

Claude Code parity ("All text you output outside of tool use is displayed to the
user"). On 2026-11-11/12 an agent answered the operator in plain text on a turn
another agent's message had started; the text sat in the operator's own DM
history but never posted, so the operator saw nothing.

`runtime._peer_turn_text_reaches_human` is the pure gate. Pins:

  (1) operator-rooted chain, real DM, plain text, nothing sent yet  -> POSTS
  (2) cron / kairos / heartbeat / silent_agent_chain root           -> SILENT
  (3) internal:peer-* conversation (agent-to-agent only)            -> SILENT
  (4) send_message / end_chain already fired                        -> SILENT (no double-post)
  (5) empty text or STAY_SILENT                                     -> SILENT
  (6) legacy empty visibility is treated as operator-rooted         -> POSTS
  (7) a session with transport "internal" but odd conversation id   -> SILENT
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip.runtime import _peer_turn_text_reaches_human  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILURES.append(label)


def base(**overrides):
    kw = dict(
        chain_root_operator=True,
        final_text="Here's the comparison you asked for.",
        channel_session_transport="discord",
        channel_conversation_id="discord:123456789012345678",
        already_delivered=False,
    )
    kw.update(overrides)
    return _peer_turn_text_reaches_human(**kw)


check("(1) operator DM, plain text -> posts", base() is True)
check("(1b) plain nextcord DM with no session -> posts",
      base(channel_session_transport="", channel_conversation_id="") is True)
check("(2) cron / agent-rooted chain -> silent", base(chain_root_operator=False) is False)
check("(3) internal:peer conversation -> silent",
      base(channel_session_transport="internal",
           channel_conversation_id="internal:peer-agentb") is False)
check("(4) already delivered via send_message -> silent", base(already_delivered=True) is False)
check("(5) empty text -> silent", base(final_text="   ") is False)
check("(5b) STAY_SILENT -> silent", base(final_text="STAY_SILENT") is False)
check("(7) internal transport, other conv id -> silent",
      base(channel_session_transport="internal", channel_conversation_id="x") is False)
check("(7b) internal: prefix on a non-internal transport -> silent",
      base(channel_session_transport="discord", channel_conversation_id="internal:a-b") is False)

print(f"\n{'ALL PASS' if not FAILURES else str(len(FAILURES)) + ' FAILED'}")
sys.exit(1 if FAILURES else 0)
