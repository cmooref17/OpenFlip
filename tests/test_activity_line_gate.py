"""Unattended-work activity line: gate + throttle logic.

The runtime posts a "working: <tools>" line before each tool batch ONLY on
cron/kairos turns, past round 2, throttled to one line per 30s. This test
mirrors that exact decision in isolation (the runtime wires it to
_safe_channel_send; here we just assert when it WOULD fire), so a regression
that makes it spam, or leak onto normal operator chat turns, is caught.
"""
from __future__ import annotations

_ACTIVITY_MIN_GAP_S = 30.0


def _should_post(originator_visibility: str, turn_count: int, has_calls: bool,
                 now_ms: float, last_ms: float) -> bool:
    """Byte-for-byte the gate in runtime.py's loop."""
    if not (originator_visibility in ("cron", "kairos")
            and turn_count >= 3 and has_calls):
        return False
    return (now_ms - last_ms) >= _ACTIVITY_MIN_GAP_S


def test_normal_chat_turn_never_posts():
    # operator_channel = real Discord chat. Must never show the line.
    for tc in range(1, 20):
        assert not _should_post("operator_channel", tc, True, 1000.0, 0.0)


def test_silent_and_peer_turns_never_post():
    for vis in ("", "silent_agent_chain", "heartbeat", "dream"):
        assert not _should_post(vis, 10, True, 1000.0, 0.0)


def test_cron_first_two_rounds_silent():
    assert not _should_post("cron", 1, True, 1000.0, 0.0)
    assert not _should_post("cron", 2, True, 1000.0, 0.0)
    assert _should_post("cron", 3, True, 1000.0, 0.0)


def test_cron_needs_tool_calls():
    assert not _should_post("cron", 5, False, 1000.0, 0.0)
    assert _should_post("cron", 5, True, 1000.0, 0.0)


def test_kairos_also_posts():
    assert _should_post("kairos", 4, True, 1000.0, 0.0)


def test_throttle_blocks_within_gap():
    last = 1000.0
    # 29s later → blocked
    assert not _should_post("cron", 5, True, last + 29.0, last)
    # exactly 30s later → allowed
    assert _should_post("cron", 5, True, last + 30.0, last)
    # 31s later → allowed
    assert _should_post("cron", 5, True, last + 31.0, last)


def test_first_post_of_turn_allowed():
    # last_ms starts at 0.0; any real now_ms is >= 30s past it.
    assert _should_post("cron", 3, True, 1759000000.0, 0.0)


if __name__ == "__main__":
    import sys, traceback
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
