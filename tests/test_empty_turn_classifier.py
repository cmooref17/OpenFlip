"""Pins the terminal-contract empty-turn classifier (2026-07-29 incident fix).

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_empty_turn_classifier.py

Background
----------
During the 2026-07-29 Anthropic incident the API returned HTTP 200 with usage
billed, ZERO content blocks, and NO stop_reason. No error status existed for
the 429/529/5xx retry branches to catch, and the old terminal-contract
classifier ("empty + no framework error = legitimate clean empty, suppress
the warning") converted hundreds of those responses into total dead air in
the channel — the operator could not distinguish "provider outage" from "bot
is dead".

`turn_retries.classify_empty_turn` is the extracted classifier. This test
pins:

  (a) empty + MISSING stop_reason, human turn      -> "provider_anomaly" (LOUD)
  (b) empty + stop_reason=end_turn, cron/synthetic -> "suppress" (still silent)
  (c) empty + stop_reason=end_turn, human DM,
      no attachments, no tool required             -> "notify_minimal"
  plus the surrounding decision table: provider errors, ollama's "stop",
  attachments-already-posted, every silent-by-design dispatch shape, the
  OPENFLIP_DISABLE_NO_FINAL_TEXT_RETRY kill switch (which must NOT disable
  the provider-anomaly path), and the done_reason threading contract the
  providers rely on.

Note: STAY_SILENT / silent-dispatch turns never reach the classifier at all —
runtime's terminal contract is gated on `not _intentionally_silent` upstream.
The `silent=True` cases here pin the classifier's own behavior should that
gate ever be loosened.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip.turn_retries import classify_empty_turn

FAILURES: list[str] = []


def check(label: str, actual, expected) -> None:
    ok = actual == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}  (got {actual!r}, expected {expected!r})")
    if not ok:
        FAILURES.append(label)


def base(**overrides):
    """A canonical HUMAN-initiated, operator-facing turn that ended with an
    empty assistant message and NO stop_reason — the 2026-07-29 incident
    shape. Named overrides build every other case."""
    kw = dict(
        captured_framework_error=None,
        diag="empty_assistant_message",
        last_stop_reason="",
        any_attachments_this_turn=False,
        auto_post_final_text=True,
        silent=False,
        is_chain_terminator=False,
        originator_agent_id="",
        auto_route_from_peer="",
        log_tag="",
    )
    kw.update(overrides)
    return kw


def test_scenario_a_missing_stop_reason_human_turn():
    print("(a) empty + NO stop_reason on a human turn -> LOUD provider anomaly, NOT suppression")
    check("missing stop_reason ('') -> provider_anomaly",
          classify_empty_turn(**base()), "provider_anomaly")
    # The same anomaly on background dispatches is also NOT a clean empty —
    # it classifies as anomaly (whether it posts is gated upstream by the
    # intentionally-silent check, same as provider errors today).
    check("missing stop_reason on synthetic turn -> provider_anomaly",
          classify_empty_turn(**base(log_tag="[synthetic] ")), "provider_anomaly")
    # An unknown stop_reason value is not clean either.
    check("unrecognized stop_reason -> provider_anomaly",
          classify_empty_turn(**base(last_stop_reason="weird_value")),
          "provider_anomaly")
    # The kill switch narrows the notify_minimal re-arm ONLY — it must never
    # re-suppress the provider anomaly.
    os.environ["OPENFLIP_DISABLE_NO_FINAL_TEXT_RETRY"] = "1"
    try:
        check("kill switch does NOT suppress the anomaly",
              classify_empty_turn(**base()), "provider_anomaly")
    finally:
        del os.environ["OPENFLIP_DISABLE_NO_FINAL_TEXT_RETRY"]


def test_scenario_b_clean_empty_background_stays_silent():
    print("(b) empty + stop_reason=end_turn on synthetic/cron/peer turns -> still silent")
    check("synthetic (cron/kairos/dream) log_tag -> suppress",
          classify_empty_turn(**base(last_stop_reason="end_turn",
                                     log_tag="[synthetic] ")), "suppress")
    check("peer turn (originator_agent_id) -> suppress",
          classify_empty_turn(**base(last_stop_reason="end_turn",
                                     originator_agent_id="agent_a")), "suppress")
    check("chain-terminator turn -> suppress",
          classify_empty_turn(**base(last_stop_reason="end_turn",
                                     is_chain_terminator=True)), "suppress")
    check("chain return (auto_route_from_peer) -> suppress",
          classify_empty_turn(**base(last_stop_reason="end_turn",
                                     auto_route_from_peer="agent_b")), "suppress")
    check("silent dispatch -> suppress",
          classify_empty_turn(**base(last_stop_reason="end_turn",
                                     silent=True)), "suppress")
    check("auto_post_final_text=False -> suppress",
          classify_empty_turn(**base(last_stop_reason="end_turn",
                                     auto_post_final_text=False)), "suppress")
    # Attachments already posted (caption / text_then_media): the operator
    # got the attachment; an extra warning would be noise. Stays suppressed
    # even on a human turn.
    check("human turn, attachments already posted -> suppress",
          classify_empty_turn(**base(last_stop_reason="end_turn",
                                     any_attachments_this_turn=True)), "suppress")


def test_scenario_c_clean_empty_human_turn_minimal_notice():
    print("(c) empty + stop_reason=end_turn on a human DM turn, no attachments -> minimal notice")
    check("human turn, clean end_turn, no attachments -> notify_minimal",
          classify_empty_turn(**base(last_stop_reason="end_turn")),
          "notify_minimal")
    # ollama reports done_reason='stop' for a deliberate finish — same rule.
    check("human turn, ollama done_reason='stop' -> notify_minimal",
          classify_empty_turn(**base(last_stop_reason="stop")),
          "notify_minimal")
    # 2026-07-29 amendment: NO any_tool_called requirement — this is the
    # turn-1 death shape; the classifier has no tool input at all, pinning
    # that the decision cannot depend on one.
    # Kill switch restores the pre-fix suppression for the CLEAN shape only.
    os.environ["OPENFLIP_DISABLE_NO_FINAL_TEXT_RETRY"] = "1"
    try:
        check("kill switch set -> clean human empty falls back to suppress",
              classify_empty_turn(**base(last_stop_reason="end_turn")),
              "suppress")
    finally:
        del os.environ["OPENFLIP_DISABLE_NO_FINAL_TEXT_RETRY"]


def test_other_shapes_unchanged():
    print("other no-output shapes keep their existing routing")
    check("captured framework error -> provider_error",
          classify_empty_turn(**base(
              captured_framework_error="⚠️ 529 overloaded")),
          "provider_error")
    check("provider error wins even with clean stop_reason",
          classify_empty_turn(**base(
              captured_framework_error="⚠️ 429",
              last_stop_reason="end_turn")),
          "provider_error")
    check("text_present_but_not_posted -> loud",
          classify_empty_turn(**base(diag="text_present_but_not_posted")),
          "loud")
    check("tools_called_but_no_reply_emitted -> loud",
          classify_empty_turn(**base(diag="tools_called_but_no_reply_emitted")),
          "loud")
    check("no_assistant_message -> loud",
          classify_empty_turn(**base(diag="no_assistant_message")), "loud")


def test_done_reason_threading_contract():
    print("provider done_reason threading (message attribute the runtime reads)")
    # The anthropic/openai wrappers set `done_reason` on the EMPTY message
    # shape; runtime extracts it via getattr(ai_message, 'done_reason', '').
    # ChatMessage is a dict subclass whose __getattr__ raises AttributeError
    # for missing keys — pin that getattr-with-default returns '' (missing)
    # and the set value (present).
    from openflip.anthropic_conversation import AnthropicAIChatMessage
    msg = AnthropicAIChatMessage(content="", tool_calls=[])
    check("done_reason missing -> getattr default ''",
          getattr(msg, "done_reason", "") or "", "")
    msg.done_reason = "end_turn"
    check("done_reason set -> readable via getattr",
          getattr(msg, "done_reason", "") or "", "end_turn")


if __name__ == "__main__":
    test_scenario_a_missing_stop_reason_human_turn()
    test_scenario_b_clean_empty_background_stays_silent()
    test_scenario_c_clean_empty_human_turn_minimal_notice()
    test_other_shapes_unchanged()
    test_done_reason_threading_contract()
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: ALL PASS")
