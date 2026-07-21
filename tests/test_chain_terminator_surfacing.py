"""Pins the operator-rooted chain-terminator surfacing fix (2026-06-15).

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_chain_terminator_surfacing.py

Background
----------
The 2026-06-11 inter-agent leak fix ran EVERY chain-terminator turn (a peer's
reply auto-routing back to the agent that started the chain) silent. That also
swallowed the operator's actual answer when an agent consulted a peer and then
replied to the human in plain text on the return turn — the operator saw
silence (observed on two deployments).

`runtime._terminator_text_surfaces` is the pure discriminator the fix added.
This test pins:

  (1) operator-rooted terminator, plain final text, REAL human channel  -> POSTS
      (root agent == this agent — the agent the human directly messaged)
  (2) agent-rooted background chain (cron/dream/silent_agent_chain)      -> SILENT
  (3) operator-rooted but NESTED (internal:peer-* conversation)          -> SILENT
      (the middle agent the human never addressed — keeps the leak closed)
  (4) operator-rooted but send_message already fired                     -> SILENT (no double-post)
  (5) operator-rooted but already posted a direct reply                  -> SILENT (no double-post)
  (6) operator-rooted, media-only turn that already produced attachment  -> SILENT
  (7) not a chain terminator at all                                      -> n/a (False)
  (8) NESTED terminator dispatched into an explicit REAL operator channel -> SILENT
      (2026-06-15 hardening: root agent != this agent, so even a real/numeric
       channel target cannot leak — the residual explicit-channel_id path)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the pure helper WITHOUT importing the whole runtime (nextcord etc.):
# load runtime.py's source isn't necessary — it imports cleanly under the venv.
from openflip.runtime import _terminator_text_surfaces

FAILURES: list[str] = []


def check(label: str, cond: bool) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        FAILURES.append(label)


def base(**overrides):
    """A canonical operator-rooted, real-channel, text-ending terminator that
    SHOULD surface, with named overrides for each negative case."""
    kw = dict(
        is_chain_terminator=True,
        chain_root_operator=True,
        final_text="oh — yeah, that's agent_b being agent_b... he answered the cheese question",
        channel_session_transport="",        # plain nextcord channel (top-level operator DM)
        channel_conversation_id="",
        reply_equivalent_tool_fired=False,
        already_posted=False,
        media_only=False,
        any_attachments=False,
        human_softinject_drained=False,
        # Genuine top-level repro: the operator messaged 'agent_a' directly; agent_a
        # consulted agent_b and is now relaying the answer. agent_a IS the chain root,
        # so root agent id == the agent running this terminator turn.
        this_agent_id="agent_a",
        chain_root_agent_id="agent_a",
    )
    kw.update(overrides)
    return kw


def test_operator_rooted_real_channel_posts():
    print("operator-rooted terminator, real channel, plain text -> POSTS")
    # The exact repro from log.txt: agent_a relays agent_b's answer to the operator.
    check("posts to the originating channel", _terminator_text_surfaces(**base()) is True)
    # iMessage operator (real human channel, non-internal transport) also posts.
    check("iMessage operator real channel posts",
          _terminator_text_surfaces(**base(
              channel_session_transport="imessage",
              channel_conversation_id="imessage:42")) is True)
    # Empty-string visibility is the conservative operator-rooted default.
    check("empty visibility treated operator-rooted (caller passes True)",
          _terminator_text_surfaces(**base(chain_root_operator=True)) is True)


def test_agent_rooted_background_stays_silent():
    print("agent-rooted background chains -> SILENT (leak fix preserved)")
    # cron / dream / heartbeat / silent_agent_chain all resolve chain_root_operator=False
    # upstream (originator_visibility not in {'', 'operator_channel'}).
    check("cron-rooted stays silent",
          _terminator_text_surfaces(**base(chain_root_operator=False)) is False)


def test_nested_internal_peer_stays_silent():
    print("operator-rooted but NESTED internal:peer terminator -> SILENT")
    # Middle agent (operator -> A -> B -> A) runs in internal:peer-<sender>.
    # chain_root_operator propagates True, but the channel is internal & the
    # human never addressed this agent — must NOT leak.
    check("internal transport stays silent",
          _terminator_text_surfaces(**base(
              channel_session_transport="internal",
              channel_conversation_id="internal:peer-agent_a")) is False)
    check("internal: conversation_id prefix stays silent (transport blank)",
          _terminator_text_surfaces(**base(
              channel_session_transport="",
              channel_conversation_id="internal:peer-agent_b")) is False)


def test_no_double_post():
    print("operator-rooted but already delivered -> SILENT (no double-post)")
    check("send_message/end_chain already fired -> silent",
          _terminator_text_surfaces(**base(reply_equivalent_tool_fired=True)) is False)
    check("already posted a direct reply -> silent",
          _terminator_text_surfaces(**base(already_posted=True)) is False)


def test_media_only_attachment_turn():
    print("operator-rooted media-only turn that produced an attachment -> SILENT")
    check("media-only + attachment, no human inject -> silent",
          _terminator_text_surfaces(**base(media_only=True, any_attachments=True)) is False)
    # ...unless a human soft-injected text into the turn (text back in play).
    check("media-only + attachment + human inject -> posts",
          _terminator_text_surfaces(**base(
              media_only=True, any_attachments=True, human_softinject_drained=True)) is True)


def test_nested_explicit_real_channel_stays_silent():
    print("NESTED terminator dispatched into an explicit REAL operator channel -> SILENT")
    # 2026-06-15 hardening. The residual leak the "is it a real channel" check
    # could NOT close: an agent EXPLICITLY calls
    #   talk_to_agent(..., channel_id=<a real operator channel>)
    # on a nested chain (operator -> A -> B -> C -> B), so B's nested terminator
    # resolves to a REAL/numeric channel (transport blank, conversation_id a
    # plain Discord channel id — NOT internal:). Every other gate passes:
    # operator-rooted, real channel, plain text, no double-post. ONLY the
    # root-agent identity differs — B is running the terminator but the human
    # addressed A — and that alone MUST block the post.
    check("nested terminator, explicit real channel, root != self -> silent",
          _terminator_text_surfaces(**base(
              this_agent_id="bravo",          # the nested middle agent running this turn
              chain_root_agent_id="alpha",    # the agent the human actually messaged
              channel_session_transport="",   # real channel, NOT internal:
              channel_conversation_id="998877665544",  # explicit numeric op channel id
          )) is False)
    # Same nested identity, but routed via a real Discord session prefix —
    # still silent: the discriminator is identity, not channel shape.
    check("nested terminator, discord: real session, root != self -> silent",
          _terminator_text_surfaces(**base(
              this_agent_id="bravo",
              chain_root_agent_id="alpha",
              channel_session_transport="discord",
              channel_conversation_id="discord:998877665544")) is False)
    # Control: the SAME real explicit channel, but THIS agent IS the root
    # (operator messaged it directly) -> POSTS. Confirms the gate keys on
    # identity, not on rejecting explicit channels wholesale.
    check("root == self in a real explicit channel still posts",
          _terminator_text_surfaces(**base(
              this_agent_id="alpha",
              chain_root_agent_id="alpha",
              channel_session_transport="discord",
              channel_conversation_id="discord:998877665544")) is True)


def test_root_agent_gate():
    print("root-agent identity gate (fail-closed)")
    # Missing root identity (legacy / unthreaded) fails CLOSED — never surface
    # without positive proof THIS agent is the human-addressed root.
    check("empty chain_root_agent_id -> silent",
          _terminator_text_surfaces(**base(chain_root_agent_id="")) is False)
    check("empty this_agent_id -> silent",
          _terminator_text_surfaces(**base(this_agent_id="")) is False)
    check("root != self -> silent",
          _terminator_text_surfaces(**base(
              this_agent_id="bravo", chain_root_agent_id="alpha")) is False)


def test_guards():
    print("guard conditions")
    check("not a chain terminator -> False",
          _terminator_text_surfaces(**base(is_chain_terminator=False)) is False)
    check("empty final text -> False",
          _terminator_text_surfaces(**base(final_text="   ")) is False)


if __name__ == "__main__":
    test_operator_rooted_real_channel_posts()
    test_agent_rooted_background_stays_silent()
    test_nested_internal_peer_stays_silent()
    test_nested_explicit_real_channel_stays_silent()
    test_root_agent_gate()
    test_no_double_post()
    test_media_only_attachment_turn()
    test_guards()
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: ALL PASS")
