"""Quiet-turn nudge: counters + gate logic (openflip/runtime.py, QuietTurnTracker).

On a long operator-facing human turn the model rarely writes text beside its
tool calls, so the operator sees only a typing indicator. The runtime now
appends a short [FRAMEWORK] note after a round's tool results once N rounds
or T seconds have passed since the agent last posted text in the turn, and an
acknowledge note right after a human soft-inject is drained mid-turn. These
tests drive the pure helpers in isolation (fake clock, no model, no Discord)
so a regression that makes it fire every round, or leak onto silent /
synthetic / peer turns, is caught.

Run: .lvenv/bin/python -m pytest tests/test_quiet_turn_nudge.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openflip.runtime import (  # noqa: E402
    QUIET_TURN_NUDGE_TEXT,
    SOFT_INJECT_ACK_TEXT,
    QuietTurnTracker,
    get_quiet_turn_nudge_settings,
    should_quiet_turn_nudge,
)
from openflip.turn_retries import operator_facing_turn  # noqa: E402


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, s: float) -> None:
        self.t += s


SETTINGS = {"enabled": True, "rounds": 3, "seconds": 90.0}


def _gate(**kw) -> bool:
    """operator_facing_turn with the defaults of a plain human Discord turn;
    override a flag to model the other turn kinds."""
    base = dict(auto_post_final_text=True, silent=False, is_chain_terminator=False,
                originator_agent_id="", auto_route_from_peer="", log_tag="")
    base.update(kw)
    return operator_facing_turn(**base)


def _tracker(clock: FakeClock | None = None, settings: dict | None = None) -> QuietTurnTracker:
    return QuietTurnTracker(settings or SETTINGS, clock=clock or FakeClock())


# ---------------------------------------------------------------- pure decision

def test_decision_fires_on_rounds_or_seconds():
    common = dict(enabled=True, rounds=3, seconds=90.0, operator_facing=True)
    assert not should_quiet_turn_nudge(rounds_since_post=2, seconds_since_post=10.0, **common)
    assert should_quiet_turn_nudge(rounds_since_post=3, seconds_since_post=10.0, **common)
    assert should_quiet_turn_nudge(rounds_since_post=1, seconds_since_post=90.0, **common)
    assert should_quiet_turn_nudge(rounds_since_post=0, seconds_since_post=500.0, **common)


def test_decision_respects_enable_and_gate():
    assert not should_quiet_turn_nudge(rounds_since_post=10, seconds_since_post=1e6,
                                       enabled=False, rounds=3, seconds=90.0, operator_facing=True)
    assert not should_quiet_turn_nudge(rounds_since_post=10, seconds_since_post=1e6,
                                       enabled=True, rounds=3, seconds=90.0, operator_facing=False)


# ---------------------------------------------------------------- tracker

def test_fires_after_n_rounds_on_human_turn():
    t = _tracker()
    notes = [t.after_tool_round(operator_facing=_gate()) for _ in range(3)]
    assert notes == [None, None, QUIET_TURN_NUDGE_TEXT]


def test_fires_after_t_seconds_on_human_turn():
    clk = FakeClock()
    t = _tracker(clk)
    clk.advance(89.0)
    assert t.after_tool_round(operator_facing=_gate()) is None   # round 1, 89s: not yet
    clk.advance(1.0)
    assert t.after_tool_round(operator_facing=_gate()) == QUIET_TURN_NUDGE_TEXT  # round 2, 90s


def test_turn_start_is_the_time_baseline():
    clk = FakeClock(5000.0)
    t = _tracker(clk)
    assert t.rounds_since_post == 0
    assert t.seconds_since_post == 0.0
    clk.advance(120.0)
    # One slow first tool round past T fires even though rounds < N.
    assert t.after_tool_round(operator_facing=_gate()) == QUIET_TURN_NUDGE_TEXT


def test_never_fires_on_silent_or_synthetic_or_peer_turns():
    excluded = [
        dict(silent=True),                                # silent dispatch
        dict(auto_post_final_text=False),                 # posting off
        dict(log_tag="[synthetic] cron"),                 # cron / heartbeat / trigger
        dict(log_tag="[synthetic] kairos"),
        dict(originator_agent_id="peer_agent"),           # peer-triggered hop
        dict(auto_route_from_peer="peer_agent", is_chain_terminator=True),  # chain terminator
    ]
    for kw in excluded:
        clk = FakeClock()
        t = _tracker(clk)
        for _ in range(10):
            clk.advance(100.0)
            assert t.after_tool_round(operator_facing=_gate(**kw)) is None, kw
            # the drain ack is gated the same way
            assert t.after_tool_round(operator_facing=_gate(**kw), human_drained=True) is None, kw


def test_disabled_by_config_never_fires():
    clk = FakeClock()
    t = _tracker(clk, {"enabled": False, "rounds": 1, "seconds": 1.0})
    for _ in range(5):
        clk.advance(100.0)
        assert t.after_tool_round(operator_facing=_gate()) is None
        assert t.after_tool_round(operator_facing=_gate(), human_drained=True) is None


def test_resets_after_agent_posts_text():
    t = _tracker()
    assert t.after_tool_round(operator_facing=_gate()) is None
    assert t.after_tool_round(operator_facing=_gate()) is None
    t.note_posted()   # model wrote text beside a tool call → posted to channel
    assert t.rounds_since_post == 0
    assert t.after_tool_round(operator_facing=_gate()) is None
    assert t.after_tool_round(operator_facing=_gate()) is None
    assert t.after_tool_round(operator_facing=_gate()) == QUIET_TURN_NUDGE_TEXT


def test_time_resets_after_agent_posts_text():
    clk = FakeClock()
    t = _tracker(clk)
    clk.advance(80.0)
    t.note_posted()
    clk.advance(80.0)   # 160s since turn start, but only 80s since the post
    assert t.after_tool_round(operator_facing=_gate()) is None


def test_fires_at_most_once_per_window_not_every_round():
    t = _tracker()
    notes = [t.after_tool_round(operator_facing=_gate()) for _ in range(9)]
    assert notes == [None, None, QUIET_TURN_NUDGE_TEXT] * 3
    assert t.fired == 3


def test_time_window_does_not_refire_every_round():
    clk = FakeClock()
    t = _tracker(clk)
    clk.advance(100.0)
    assert t.after_tool_round(operator_facing=_gate()) == QUIET_TURN_NUDGE_TEXT
    # Next rounds come fast: neither threshold is crossed again.
    clk.advance(1.0)
    assert t.after_tool_round(operator_facing=_gate()) is None
    clk.advance(1.0)
    assert t.after_tool_round(operator_facing=_gate()) is None


def test_soft_inject_drain_nudge_fires_and_resets():
    clk = FakeClock()
    t = _tracker(clk)
    assert t.after_tool_round(operator_facing=_gate()) is None
    assert t.after_tool_round(operator_facing=_gate()) is None
    clk.advance(85.0)
    # Human message drained this round → ack note right away, window reset.
    assert t.after_tool_round(operator_facing=_gate(), human_drained=True) == SOFT_INJECT_ACK_TEXT
    assert t.rounds_since_post == 0
    assert t.seconds_since_post == 0.0
    # Quiet nudge needs a fresh full window after the ack.
    clk.advance(10.0)
    assert t.after_tool_round(operator_facing=_gate()) is None
    assert t.after_tool_round(operator_facing=_gate()) is None
    assert t.after_tool_round(operator_facing=_gate()) == QUIET_TURN_NUDGE_TEXT


def test_soft_inject_ack_wins_over_quiet_nudge_same_round():
    t = _tracker()
    t.after_tool_round(operator_facing=_gate())
    t.after_tool_round(operator_facing=_gate())
    # Round 3 would fire the quiet nudge; a drained human message takes it.
    assert t.after_tool_round(operator_facing=_gate(), human_drained=True) == SOFT_INJECT_ACK_TEXT


# ---------------------------------------------------------------- config

def test_settings_defaults_and_overrides():
    assert get_quiet_turn_nudge_settings({}) == {"enabled": True, "rounds": 3, "seconds": 90.0}
    assert get_quiet_turn_nudge_settings({"quiet_turn_nudge": {"rounds": 5}}) == \
        {"enabled": True, "rounds": 5, "seconds": 90.0}
    assert get_quiet_turn_nudge_settings({"quiet_turn_nudge": {"enabled": False, "seconds": 30}}) == \
        {"enabled": False, "rounds": 3, "seconds": 30.0}
    assert get_quiet_turn_nudge_settings({"quiet_turn_nudge": False})["enabled"] is False
    # Garbage falls back per-field; non-dict/non-bool values are ignored.
    assert get_quiet_turn_nudge_settings({"quiet_turn_nudge": {"rounds": "x", "seconds": None}}) == \
        {"enabled": True, "rounds": 3, "seconds": 90.0}
    assert get_quiet_turn_nudge_settings({"quiet_turn_nudge": "yes"}) == \
        {"enabled": True, "rounds": 3, "seconds": 90.0}


def test_tracker_reads_settings_dict():
    t = QuietTurnTracker({"enabled": True, "rounds": 2, "seconds": 1e9}, clock=FakeClock())
    assert t.after_tool_round(operator_facing=True) is None
    assert t.after_tool_round(operator_facing=True) == QUIET_TURN_NUDGE_TEXT


if __name__ == "__main__":
    import traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"ok   {fn.__name__}")
        except AssertionError:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
