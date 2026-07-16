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
from typing import Callable, Optional


# Phrases that read like an action-commitment ("lemme look", "imma do it",
# etc.). If the assistant emits one of these with NO accompanying tool_use we
# retry the turn forcing tool_choice=any so the model must emit a tool.
_PROMISE_PHRASES = (
    "lemme ", "imma ", "lookin ", "peek at", "peek it",
    "let me ", "i'll do", "i'll look", "one sec",
    "hold on", "checkin ", "workin on",
)


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


def action_promise_should_retry(text: str, already_used: bool) -> bool:
    """Action-promise retry: True when `text` reads like an action-commitment
    ("lemme look", "imma do it", etc.) but no tool_use accompanied it, so the
    turn should retry forcing tool_choice=any. Cap at 1 retry per turn
    (`already_used`). Kill switch: OPENFLIP_DISABLE_ACTION_PROMISE_RETRY=1.
    """
    if os.environ.get("OPENFLIP_DISABLE_ACTION_PROMISE_RETRY") == "1":
        return False
    if already_used:
        return False
    lowered = text.lower()
    return any(p in lowered for p in _PROMISE_PHRASES)


def detect_peer_prose(
    text: str,
    this_agent_id: str,
    is_known_peer: Callable[[str], bool],
    already_used: bool,
) -> Optional[str]:
    """Peer-prose leak detection. Returns the detected peer agent id when the
    model is addressing another agent in prose (line begins with
    "<peer_agent_id>: " / "<peer> ," / "<peer> —") but did NOT fire
    talk_to_agent — without intervention the text auto-routes to whoever
    triggered this turn (often the operator), leaking inter-agent prose into
    the wrong channel. Returns None when nothing detected. Cap at 1 retry per
    turn (`already_used`). Kill switch: OPENFLIP_DISABLE_PEER_PROSE_RETRY=1.

    Whole-message scan, not just first token. Catches three real-world shapes:
      1. "<peer>: hi"            (first-line prefix)
      2. "night Mini. the maintainer agent: g'night you too"
                                 (mid-message line start)
      3. "thanks. \n<peer>: ..."
                                 (later line prefix)

    We look line-by-line for any line whose first whitespace-token, with
    trailing :,—- stripped, is a known peer agent id (not this agent). Lines
    inside fenced code blocks (```...```) are skipped — quoted code that
    happens to name a peer shouldn't trigger the nudge.
    """
    if os.environ.get("OPENFLIP_DISABLE_PEER_PROSE_RETRY") == "1":
        return None
    if already_used:
        return None
    _detected_peer = None
    _in_fence = False
    for _line in text.splitlines():
        _stripped = _line.strip()
        if _stripped.startswith("```"):
            _in_fence = not _in_fence
            continue
        if _in_fence or not _stripped:
            continue
        _tok = _stripped.split(None, 1)[0]
        _bare = _tok.rstrip(":,—-")
        if (_bare and _bare != this_agent_id
                and is_known_peer(_bare)):
            _detected_peer = _bare
            break
    return _detected_peer


def build_peer_prose_nudge(detected_peer: str) -> str:
    """The [FRAMEWORK] nudge text injected when peer-prose is detected: names
    the detected peer and requires the model to either (a) re-emit using
    talk_to_agent, or (b) rewrite the reply for the actual reader."""
    return (
        f"[FRAMEWORK]: Your last reply began with "
        f"'{detected_peer}:' as if addressing peer "
        f"agent '{detected_peer}', but you did not "
        f"call talk_to_agent. Plain prose in this "
        f"channel does NOT route to a peer — it goes "
        f"to whoever this channel belongs to (often the "
        f"operator). If you meant to message "
        f"'{detected_peer}', call talk_to_agent now. "
        f"If the message was actually meant for the "
        f"current channel's reader, rewrite without "
        f"the peer-id prefix."
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
    OPENFLIP_DISABLE_NO_FINAL_TEXT_RETRY=1 restores the pre-fix behavior
    wholesale: textless exits allowed, clean empty end_turns stay
    warning-suppressed."""
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


def run_stop_hooks(
    *,
    agent_id: str,
    channel_id: int,
    assistant_text: str,
    tool_was_called: bool,
    depth: int,
    is_chain_terminator: bool,
    is_synthetic: bool,
    originator_visibility: str,
):
    """Stop-hook invocation. Mirrors Claude Code's `handleStopHooks` pattern: a
    text-only turn (no tool_use) gets one chance to be rewritten/extended if any
    registered hook decides the reply is malformed. The current single hook,
    `promise_without_action`, catches text like "checking…" / "let me look" /
    "on it" that leaves the operator staring at a dangling promise.

    Thin wrapper around `stop_hooks.evaluate_stop_hooks` so the import + call
    live with the other turn-retry heuristics. Exceptions propagate to the
    caller's existing try/except (which sets the result to None and logs).
    """
    from .stop_hooks import evaluate_stop_hooks
    return evaluate_stop_hooks(
        agent_id=agent_id,
        channel_id=channel_id,
        assistant_text=assistant_text,
        tool_was_called=tool_was_called,
        depth=depth,
        is_chain_terminator=is_chain_terminator,
        is_synthetic=is_synthetic,
        originator_visibility=originator_visibility,
    )
