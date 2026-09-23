"""In-loop turn-retry heuristics for `runtime._run_turn`.

Pure-ish decision helpers extracted verbatim from the model-call loop in
`runtime._run_turn` (audit §4 item 1, full_audit_2026-06-07). These functions
inspect the assistant reply (text / tool_calls) and decide whether a synthetic
retry or [FRAMEWORK] nudge should fire. They DO NOT touch runner state: the
caller in runtime.py still owns the one-shot flags, the `conv.messages` append,
the sticky `tool_choice` override, the `print_ts` logging, and the `continue`.
Each helper takes explicit inputs and returns a plain decision (bool / Optional
nudge text / the detected peer id / the stop-hook result).

The regexes, phrase lists, thresholds, env kill switches, and nudge strings are
preserved EXACTLY as they were inline — this is code motion, not a redesign.
"""
from __future__ import annotations
import os
from typing import Optional


# Nudge injected before retrying an empty (no text, no tool_use) reply so the
# API call has different input. A bare retry would send the same body and get
# the same empty back.
_EMPTY_RETRY_NUDGE = (
    "[FRAMEWORK]: Your previous reply was empty "
    "(no text and no tool calls). The user is "
    "waiting on a response. Reply now with what "
    "you found from the tool results above, in "
    "your normal voice. If you need another tool "
    "call, fire it. Do not stay silent."
)


def empty_retry_nudge(already_used: bool) -> Optional[str]:
    """Empty-reply retry: returns the [FRAMEWORK] nudge text to inject before
    retrying when the model returned no text AND no tool_use. Returns None when
    the retry should not fire (already used this turn, or kill switch set). Cap
    at 1 retry per turn (`already_used`). Kill switch:
    OPENFLIP_DISABLE_EMPTY_RETRY=1 falls through to the normal break path."""
    if os.environ.get("OPENFLIP_DISABLE_EMPTY_RETRY") == "1":
        return None
    if already_used:
        return None
    return _EMPTY_RETRY_NUDGE


# Nudge injected by the final-text guarantee (runtime._run_turn's turn-
# completion gate) when a HUMAN turn that ran tools is about to end with
# nothing operator-visible. Distinct wording from _EMPTY_RETRY_NUDGE so
# history/logs show which mechanism fired.
_NO_FINAL_TEXT_NUDGE = (
    "[FRAMEWORK]: You ran tools this turn but your "
    "last reply was empty — the operator has seen "
    "NOTHING and is waiting. Reply now, in your "
    "normal voice, with what the tool results above "
    "showed. If another tool call is genuinely "
    "needed, fire it. Do not stay silent."
)


def no_final_text_nudge() -> str:
    """The [FRAMEWORK] nudge the final-text guarantee feeds forward with the
    tool results when it forces another model round (see
    no_final_text_guarantee_enabled for the feature description)."""
    return _NO_FINAL_TEXT_NUDGE


def no_final_text_guarantee_enabled() -> bool:
    """Kill switch for the final-text guarantee + its terminal-contract
    un-suppression (2026-07-15 fix for "tool ran, then dead silence" on
    human turns). The guarantee mirrors Claude Code's query loop: on an
    operator-facing human turn that executed at least one tool, the agentic
    loop may not terminate on a textless round — it continues to another
    model round until the turn has produced operator-visible output (text,
    attachment, or reply-equivalent tool), bounded only by MAX_TOOL_TURNS.
    Since 2026-07-29 the terminal-contract re-arm this switch also gates no
    longer requires a tool to have run: any human-facing clean empty
    end_turn with no attachments gets a minimal notice instead of silence.
    OPENFLIP_DISABLE_NO_FINAL_TEXT_RETRY=1 restores the pre-fix behavior
    wholesale: textless exits allowed, clean empty end_turns stay
    warning-suppressed. It does NOT affect the empty/missing-stop_reason
    provider-anomaly path (2026-07-29 incident fix), which always
    surfaces."""
    return os.environ.get("OPENFLIP_DISABLE_NO_FINAL_TEXT_RETRY") != "1"


def operator_facing_turn(
    *,
    auto_post_final_text: bool,
    silent: bool,
    is_chain_terminator: bool,
    originator_agent_id: str,
    auto_route_from_peer: str,
    log_tag: str,
) -> bool:
    """True when THIS turn's output is destined for a human who is actively
    awaiting it: a real inbound message (visible, top-level, not an
    inter-agent hop). Built strictly from existing turn signals — every
    synthetic dispatch (cron / kairos / dream / slash-command / follow-up /
    talk_to_agent) carries the "[synthetic] " log_tag, peer turns carry
    originator_agent_id, and chain-terminator returns carry
    auto_route_from_peer. All of those are legitimately allowed to end
    silent and MUST stay excluded here."""
    return (
        auto_post_final_text
        and not silent
        and not is_chain_terminator
        and not originator_agent_id
        and not auto_route_from_peer
        and not str(log_tag or "").strip().startswith("[synthetic]")
    )


# Provider stop_reason values that mean "the model deliberately finished the
# turn". anthropic reports "end_turn"; openai_conversation normalizes
# finish_reason="stop" to "end_turn"; ollama's done_reason is "stop".
_CLEAN_STOP_REASONS = ("end_turn", "stop")


def classify_empty_turn(
    *,
    captured_framework_error: Optional[str],
    diag: str,
    last_stop_reason: str,
    any_attachments_this_turn: bool,
    auto_post_final_text: bool,
    silent: bool,
    is_chain_terminator: bool,
    originator_agent_id: str,
    auto_route_from_peer: str,
    log_tag: str,
) -> str:
    """Terminal-contract classifier for a turn that ended with nothing
    operator-visible (extracted from runtime._run_turn, 2026-07-29).

    History: the previous inline classifier treated "empty assistant message
    + no framework error" as a legitimate clean empty per Claude Code parity
    and suppressed the operator warning. That reasoning was layer-confused —
    in a terminal, the spinner stopping and the prompt returning make silence
    visible; in Discord the message IS the entire interface, so posting
    nothing is indistinguishable from a dead bot. During the 2026-07-29
    Anthropic incident the API returned HTTP 200 with usage billed, ZERO
    content blocks, and NO stop_reason — no error status for the 429/529/5xx
    retry branches to catch — and the suppression converted hundreds of those
    responses into total dead air. A genuine "model chose to say nothing"
    always carries stop_reason=end_turn ("stop" on ollama); the provider
    layer detected the difference all along ("empty reply with no
    stop_reason" in anthropic_conversation.py / openai_conversation.py) but
    the signal was discarded before classification. `last_stop_reason`
    threads it here.

    Returns one of:
      "provider_error"   — a framework/API error was captured this turn;
                           surface the ACTUAL error text.
      "provider_anomaly" — empty assistant message with a MISSING/blank
                           stop_reason (the 2026-07-29 incident shape);
                           surface loudly as a likely provider outage.
      "notify_minimal"   — genuine empty end_turn, but on a HUMAN-initiated
                           operator-facing turn with no attachments: the
                           operator asked and is owed at least a minimal
                           notice (2026-07-15 re-arm, amended 2026-07-29 to
                           drop the any_tool_called requirement — no-tool
                           turns died on turn 1 throughout the incident and
                           stayed fully suppressed). Honors the
                           OPENFLIP_DISABLE_NO_FINAL_TEXT_RETRY kill switch.
      "suppress"         — genuine empty end_turn on a silent-by-design
                           dispatch (peer/cron/synthetic/chain-terminator —
                           fails operator_facing_turn) or a turn whose
                           attachments already posted (caption /
                           text_then_media shapes; media_only is handled
                           upstream by _media_satisfied). Stay silent.
      "loud"             — every other no-output shape
                           (text_present_but_not_posted,
                           tools_called_but_no_reply_emitted,
                           no_assistant_message, unknown).
    """
    if captured_framework_error:
        return "provider_error"
    if diag != "empty_assistant_message":
        return "loud"
    if last_stop_reason not in _CLEAN_STOP_REASONS:
        return "provider_anomaly"
    # Genuine clean empty end_turn from here on.
    if (not any_attachments_this_turn
            and no_final_text_guarantee_enabled()
            and operator_facing_turn(
                auto_post_final_text=auto_post_final_text,
                silent=silent,
                is_chain_terminator=is_chain_terminator,
                originator_agent_id=originator_agent_id,
                auto_route_from_peer=auto_route_from_peer,
                log_tag=log_tag,
            )):
        return "notify_minimal"
    return "suppress"
