"""Stop hooks — structural backstop for the "promise without action" bug class.

Background
==========
Claude Code's `query()` async generator (see `src/query/stopHooks.ts` in the
2.1.88 source mirror — analysis in `agents/an agent/audits/claude_code_findings_2026-05-19.md`)
runs a registered list of "stop hooks" at end-of-turn. Each hook can BLOCK the
terminal exit and force a follow-up turn with a synthetic user message. That's
the architectural pattern this module mirrors.

The recurring openflip bug we're backstopping
---------------------------------------------
The model emits text like "checking…" / "let me look" / "on it" / "I'll fix it"
with ZERO tool calls in the same turn. The runtime correctly exits on no-tools
(`runtime.py:_run_turn`, see the `needs_follow_up = bool(_tc)` gate around
line 1442). The operator is then left staring at a dangling promise — the
agent SAID it would do something but never fired the tool.

FRAMEWORK.md carries an "Action-promise STOP-TEST" rule at the prompt level
that asks the model to either fire the tool in the same response or delete the
promise phrase. That prompt rule catches most cases. This hook is the
structural backstop for the cases where the rule fails — exactly the same
shape as the "noted with no save_memory" pattern, but for ACTIONS instead of
memory writes.

Design constraints
==================
- ONE hook for now (`promise_without_action`), but the module is built around a
  registry of `HookFn` callables so adding the second hook (e.g. an
  inter-agent-ack-only check) is a one-line registration.
- Hard depth cap — at most ONE retry per turn, enforced by the runtime caller
  via the `depth` argument. We refuse to fire at depth >= 2 because we ARE
  the retry layer; a recursive retry would mask a deeper model failure.
- Chain-terminator turns are exempt — those turns have their own narrow
  routing-tool protocol (see `runtime.py` chain-terminator block around
  line 797) and inject their own diagnostics if no routing tool fires.
- `OPENFLIP_DISABLE_PROMISE_HOOK=1` env var kills the promise hook entirely.
  Lets the operator disable a misbehaving regex live without a code edit.

Historical note
===============
A version of this module shipped 2026-05-21 with three bundled hooks
(`promise_without_action`, `inter_agent_ack_only`, `chain_terminator_no_post`)
and was removed shortly after — see `agents/an agent/TODO.md` "Tier 2.1 Stop hooks"
entry. This is the rebuild, scoped to the single load-bearing hook with the
exclusion-set explicitly designed to keep the documented false positives
("the checking process is complete", "I am not checking that", "did you mean
checking?") OUT, and the documented true positives ("checking shutdown time:",
"let me look:", "on it") IN.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class StopHookResult:
    """Result of a hook firing.

    Fields:
      blocked: True if the hook is forcing a follow-up turn.
      reason: short diagnostic tag for the log line — `<hook_name>: <detail>`.
      suggested_user_message: the synthetic `[FRAMEWORK]` user message the
        caller should append to history before re-running chat(). May be None
        for hooks that block without nudging (none currently exist, but the
        type allows it).
    """
    blocked: bool
    reason: str
    suggested_user_message: Optional[str]


# A hook function takes the kwargs that `evaluate_stop_hooks` receives and
# returns either a blocking StopHookResult or None to pass.
HookFn = Callable[..., Optional[StopHookResult]]


# ---------------------------------------------------------------------------
# Hook: promise_without_action
# ---------------------------------------------------------------------------

# Single-word promise verbs (present-continuous / imperative). Only fire when
# these are at the START of the first sentence — mid-sentence occurrences are
# almost always adjective / participial usage ("the checking process", "after
# reading the file") which is a documented false-positive class.
_BARE_VERBS = (
    "checking", "looking", "running", "verifying", "investigating",
    "digging", "fetching", "grepping", "reading", "writing",
)

# Multi-word action commitments. These can appear ANYWHERE in the first
# sentence because the phrasing itself is specific enough that false-positive
# rate is low (e.g. "let me check" is rarely used as anything but a promise).
_PHRASE_PATTERNS = (
    r"on it\b",
    r"let me (?:check|look|see|run|grep|find)\b",
    r"going to (?:check|look|run|grep|read|fetch|write|fix)\b",
    # Both contracted "I'll" and unapostrophized "Ill" surface in model output.
    r"i'?ll (?:check|look|see|run|grep|fix|fetch|write)\b",
    # Status-report promises about work not yet done.
    r"fixing now\b",
    r"patching now\b",
    r"doing it now\b",
    r"\bshipping\b",
    r"\btrying\b",
)

# Compile once. Both groups are case-insensitive; both anchor on word
# boundaries so "checking" doesn't match "fact-checking" or "rechecking".
_BARE_VERB_RE = re.compile(
    r"\b(?:" + "|".join(_BARE_VERBS) + r")\b",
    re.IGNORECASE,
)
_PHRASE_RE = re.compile(
    "|".join(_PHRASE_PATTERNS),
    re.IGNORECASE,
)

# Negation immediately preceding a verb cancels the promise reading.
# "not checking", "won't check", "can't run", "don't look" etc. Window is
# .{0,5} because real-world negation hugs the verb closely; expanding the
# window invites cross-clause false-positives ("I'm not — well, checking…").
_NEGATION_RE = re.compile(
    r"\b(?:not|won't|can't|cannot|don't|wasn't|isn't|aren't)\b\s{0,5}$",
    re.IGNORECASE,
)

# Max length for the assistant_text to qualify. Long replies that mention a
# promise verb almost always use it descriptively ("after running the migration
# I noticed X" inside a 600-char debrief). This is the simplest false-positive
# guard there is and the previous round of misfires came almost exclusively
# from long messages.
_MAX_TEXT_LEN = 200

# How far into the message we look for a match. The first sentence is where
# real promises live — "checking. Then I'll do X" — anything mid-paragraph is
# descriptive prose with a verb in a different role.
def _first_sentence(text: str) -> str:
    """Return everything before the first `. ` or newline. The trailing `.`
    of a single-sentence message is kept (`re.split` on `. ` requires the
    space, so "checking the file." stays intact)."""
    return re.split(r"\. |\n", text, maxsplit=1)[0]


def _in_code_fence(full_text: str, abs_pos: int) -> bool:
    """True if `abs_pos` falls between an unbalanced pair of triple-backticks
    in `full_text`. We count fences left of the position; odd → inside.

    Rare in a first sentence (the model would have to open a code fence on
    line one) but handled anyway because it's a clean exclusion and the
    cost is one regex scan.
    """
    fences = [m.start() for m in re.finditer(r"```", full_text)]
    open_count = sum(1 for f in fences if f < abs_pos)
    return (open_count % 2) == 1


def _has_negation_before(text: str, match_start: int) -> bool:
    """True if `not|won't|can't|cannot|don't|...` appears within ~5 chars
    before the match. Window matches the spec in the rebuild brief."""
    # Look back up to ~30 chars so the negation word itself (max ~7) plus its
    # whitespace window (5) plus a small buffer fits inside the snippet we
    # hand to the regex.
    window_start = max(0, match_start - 30)
    snippet = text[window_start:match_start]
    return bool(_NEGATION_RE.search(snippet))


# Short filler interjections that often precede a promise verb in casual
# replies ("okay, checking now" / "back! checking real quick" / "right —
# looking it up"). Allowing one of these to precede the verb without
# canceling the position-0 check catches the common real-world shape
# without enabling broad "verb anywhere in the first sentence" matching
# (which would re-introduce "the checking process is complete" misfires).
_LEADING_FILLERS = frozenset({
    "back", "okay", "ok", "right", "alright", "yep", "yeah", "yes",
    "sure", "got it", "hi", "hey", "oh", "well", "fine",
})


def _starts_with_verb(first_sentence: str, verb_match) -> bool:
    """True if the bare-verb match is the first meaningful token of the first
    sentence — either literally first OR preceded only by a short
    interjection filler from `_LEADING_FILLERS`. Catches:
      "checking shutdown time:"   → True   (verb is leading token)
      "Checking — one sec"        → True   (uppercase, em-dash after)
      "back! checking real quick" → True   (filler "back" precedes verb)
      "okay, checking that"       → True   (filler "okay" precedes verb)
      "the checking process..."   → False  ("the" is not a filler)
      "After reading the file"    → False  ("after" is not a filler)
    """
    # Find first non-whitespace, non-punctuation char.
    leading = re.match(r"^[\s\"'`*_>\-]*", first_sentence)
    leading_len = leading.end() if leading else 0
    if verb_match.start() == leading_len:
        return True
    # Tokenize the prefix between leading_len and the verb match. If every
    # alphabetic token there is in _LEADING_FILLERS, treat as still-at-start.
    prefix = first_sentence[leading_len:verb_match.start()]
    # Extract just the word tokens; ignore punctuation/whitespace between.
    tokens = re.findall(r"[A-Za-z']+", prefix.lower())
    if not tokens:
        # Only punctuation/whitespace between leading edge and verb — same
        # as position 0 effectively. Defensive; the position check above
        # should normally cover this.
        return True
    return all(t in _LEADING_FILLERS for t in tokens)


def _hook_promise_without_action(
    *,
    assistant_text: str,
    tool_was_called: bool,
    depth: int,
    is_chain_terminator: bool,
    originator_visibility: str = "",
    **_unused,
) -> Optional[StopHookResult]:
    """The actual hook logic. Returns a blocking StopHookResult if the model
    emitted a promise-shaped reply with no accompanying tool call, else None.
    """

    # ---- Kill switch ----
    # Lets the operator disable this hook live without a code edit when
    # tuning regex. Set in env: OPENFLIP_DISABLE_PROMISE_HOOK=1.
    if os.environ.get("OPENFLIP_DISABLE_PROMISE_HOOK") == "1":
        return None

    # ---- Kairos exemption ----
    # Proactive (kairos) ticks may emit text like "checking…" as part of
    # their assessment before deciding to do nothing — that's not a promise
    # to the operator. The idle-tick pruning in runtime.py handles cleanup.
    if originator_visibility == "kairos":
        return None

    # ---- Dream exemption ----
    # Auto-dream consolidation turns (originator_visibility="dream") emit
    # text like "consolidating…" before the update_core_memory tool call.
    # That's the dream's own internal flow, never a promise to the operator
    # (dream turns are silent and never post to Discord). Exempt them like
    # kairos so the promise hook can't fight the consolidation pass.
    if originator_visibility == "dream":
        return None

    # ---- Trivial bypasses ----
    # Hook ONLY applies to text-only turns. A tool call IS the action; a
    # non-empty reply with a tool call is fine even if it contains promise
    # language ("checking now…" + actual grep call is the bundled response
    # we WANT the model to emit).
    if tool_was_called:
        return None
    if not assistant_text or not assistant_text.strip():
        return None

    # ---- Depth cap ----
    # We're the retry layer. Allowing depth >= 2 would let one misfire
    # cascade into an unbounded loop of nudges. Cap matches the convention
    # in `runtime.py`'s other one-shot retry flags (_empty_retry_used,
    # _force_tool_retry_used, _peer_prose_retry_used) — each at most ONCE
    # per turn.
    if depth >= 2:
        return None

    # ---- Chain-terminator exemption ----
    # Chain-terminator turns have their own narrow protocol — they're
    # expected to dispatch via send_message / talk_to_agent / end_chain.
    # See `runtime.py` chain-terminator block around line 797 and the
    # chain-terminator rescue path. Double-handling here would either
    # double-post text the rescue path already surfaced, or fight the
    # narrowed toolset with a promise nudge it can't satisfy.
    if is_chain_terminator:
        return None

    # ---- Question is not a promise ----
    # "Did you mean checking?" is a question, not a commitment. The `?`
    # check on the FULL text (not just the first sentence) is intentional:
    # a trailing clarifying question after a promise is still inquiry-shaped,
    # not the silent-failure pattern we're protecting against.
    if "?" in assistant_text:
        return None

    # ---- Length guard ----
    # Long replies with a promise verb are overwhelmingly legitimate
    # ("after running the upgrade I noticed three regressions, …"). This
    # is the cheapest false-positive guard we have — past misfires were
    # almost exclusively in messages over 200 chars where the verb was
    # used in a different role.
    stripped = assistant_text.strip()
    if len(stripped) > _MAX_TEXT_LEN:
        return None

    # ---- First sentence only ----
    # Mid-reply mentions of the verbs (in a longer explanation) don't
    # count. By the time we reach this point we know `stripped` is <=
    # _MAX_TEXT_LEN so first_sentence is short and cheap to scan.
    sentence = _first_sentence(stripped)

    matched_phrase: Optional[str] = None

    # ---- Phase 1: multi-word phrase patterns ----
    # These are specific enough that mid-sentence occurrences are still
    # promise-shaped ("got it, on it" — yes, that's a promise).
    for m in _PHRASE_RE.finditer(sentence):
        # Absolute position in the full assistant_text — used for code-fence
        # and negation checks below. The sentence starts at the same
        # position as stripped because we strip BEFORE taking the first
        # sentence; we then offset by however much leading whitespace the
        # original text had to map back.
        abs_pos = assistant_text.find(stripped) + m.start()
        if _in_code_fence(assistant_text, abs_pos):
            continue
        if _has_negation_before(assistant_text, abs_pos):
            continue
        matched_phrase = m.group(0)
        break

    # ---- Phase 2: bare verb (must be at start) ----
    # Only check if no phrase already matched. Bare verbs MUST be the
    # leading token of the first sentence — see _starts_with_verb. This
    # is the central guard against the "the checking process" /
    # "after reading the file" false-positive class.
    if matched_phrase is None:
        bm = _BARE_VERB_RE.search(sentence)
        if bm is not None and _starts_with_verb(sentence, bm):
            abs_pos = assistant_text.find(stripped) + bm.start()
            if not _in_code_fence(assistant_text, abs_pos) \
               and not _has_negation_before(assistant_text, abs_pos):
                matched_phrase = bm.group(0)

    if matched_phrase is None:
        return None

    # ---- Build the nudge ----
    # The nudge is a synthetic user message. It names the matched phrase
    # back to the model (so it can SEE what it just said), explains the
    # operator-visible consequence, and offers the two valid fixes —
    # mirroring the prompt-level rule's structure so the model sees a
    # consistent framing whether the rule self-applies or this hook
    # injects it.
    suggested = (
        f"[FRAMEWORK]: Your last reply said '{matched_phrase}' but you did not "
        f"actually call any tool. The operator is waiting on the action you "
        f"described. Either (a) fire the tool that completes the action you "
        f"promised, or (b) rewrite your reply without that promise phrase. "
        f"Do not emit another text-only reply with action-language; that's "
        f"the exact failure mode this check exists to prevent."
    )

    return StopHookResult(
        blocked=True,
        reason=f"promise_without_action: matched '{matched_phrase}' without tool call",
        suggested_user_message=suggested,
    )


# ---------------------------------------------------------------------------
# Registry + dispatcher
# ---------------------------------------------------------------------------

# Registry of active hooks. Order matters only insofar as the FIRST hook to
# block wins — we don't run later hooks once one fires. Keep this list short
# and curated; every entry adds another regex pass to every text-only turn.
_HOOKS: list[tuple[str, HookFn]] = [
    ("promise_without_action", _hook_promise_without_action),
]


def evaluate_stop_hooks(
    *,
    agent_id: str,
    channel_id: int,
    assistant_text: str,
    tool_was_called: bool,
    depth: int,
    is_chain_terminator: bool,
    is_synthetic: bool,
    originator_visibility: str = "",
) -> Optional[StopHookResult]:
    """Run every registered hook in order, returning the first blocking
    result (or None if all pass).

    Called from `runtime.py:_run_turn` immediately before the
    `needs_follow_up = bool(_tc); if not needs_follow_up: break` exit.
    See "Wire-in" section of this module's docstring for the depth-cap
    contract.

    Note on `is_synthetic`: synthetic turns (restart-continuation,
    chain-terminator dispatch, cron) ARE evaluated. That's deliberate —
    those turns are exactly where the promise-leak shows up most, because
    the agent re-orients on a context it didn't itself produce.
    """
    for name, fn in _HOOKS:
        try:
            result = fn(
                agent_id=agent_id,
                channel_id=channel_id,
                assistant_text=assistant_text,
                tool_was_called=tool_was_called,
                depth=depth,
                is_chain_terminator=is_chain_terminator,
                is_synthetic=is_synthetic,
                originator_visibility=originator_visibility,
            )
        except Exception as e:
            # A buggy hook must not break the turn. Surface to stderr and
            # treat as pass. Hooks that consistently throw should be
            # caught in `python -m openflip.stop_hooks` before deploy.
            from .utils import print_ts
            print_ts(f"[stop_hooks] hook '{name}' raised: {e!r}", error=True)
            continue
        if result is not None and result.blocked:
            return result
    return None


# ---------------------------------------------------------------------------
# Inline tests — `python -m openflip.stop_hooks`
# ---------------------------------------------------------------------------

def _run_tests() -> int:
    """Returns 0 on success, non-zero on failure (count of failed cases)."""

    def _call(text: str, *, tool_was_called=False, depth=0,
              is_chain_terminator=False, is_synthetic=False,
              originator_visibility=""):
        return evaluate_stop_hooks(
            agent_id="test", channel_id=0,
            assistant_text=text, tool_was_called=tool_was_called,
            depth=depth, is_chain_terminator=is_chain_terminator,
            is_synthetic=is_synthetic,
            originator_visibility=originator_visibility,
        )

    cases: list[tuple[str, dict, bool, str]] = [
        # (description, kwargs, expect_blocked, text)
        # ---- Positive: must fire ----
        ("bare-verb at start: 'checking shutdown time:'",
         {}, True, "checking shutdown time:"),
        ("phrase 'let me look:'",
         {}, True, "let me look:"),
        ("phrase 'on it'",
         {}, True, "on it"),
        ("bare verb capitalized: 'Reading.'",
         {}, True, "Reading."),
        ("phrase 'I'll fix it'",
         {}, True, "I'll fix it"),
        ("phrase 'going to grep the logs'",
         {}, True, "going to grep the logs"),
        ("status: 'fixing now'",
         {}, True, "fixing now"),
        ("filler 'back!' before verb: 'back! checking real quick —'",
         {}, True, "back! checking real quick —"),
        ("filler 'okay,' before verb: 'okay, checking that now'",
         {}, True, "okay, checking that now"),
        ("filler 'hi' before verb: 'hi — looking now'",
         {}, True, "hi — looking now"),

        # ---- Negative: must NOT fire ----
        ("verb-as-adjective: 'the checking process is complete.'",
         {}, False, "the checking process is complete."),
        ("negation: 'I am not checking that.'",
         {}, False, "I am not checking that."),
        ("question: 'Did you mean checking?'",
         {}, False, "Did you mean checking?"),
        ("mid-sentence bare verb: 'After reading the file, all good.'",
         {}, False, "After reading the file, all good."),
        ("long reply with verb: " + ("x" * 210),
         {}, False, "running the upgrade succeeded. " + ("x" * 210)),
        ("tool was called this turn",
         {"tool_was_called": True}, False, "checking that now"),
        ("depth cap: depth=2",
         {"depth": 2}, False, "checking that now"),
        ("chain terminator exempt",
         {"is_chain_terminator": True}, False, "checking that now"),
        ("empty text",
         {}, False, ""),
        ("whitespace only",
         {}, False, "   \n  "),
        ("phrase negation: 'I'm not going to fix that'",
         {}, False, "I'm not going to fix that"),
        ("code fence around match",
         {}, False, "```\nchecking the file\n```"),
        ("question with promise verb earlier in long text",
         {}, False, "checking the file? maybe later."),
        ("kairos exemption: promise exempt in proactive tick",
         {"originator_visibility": "kairos"}, False, "checking that now"),
    ]

    failures: list[str] = []
    for desc, kwargs, expect, text in cases:
        got = _call(text, **kwargs)
        got_blocked = got is not None and got.blocked
        if got_blocked != expect:
            failures.append(
                f"  FAIL: {desc}\n"
                f"    text={text!r}\n"
                f"    expected blocked={expect}, got={got_blocked}"
                + (f" (reason: {got.reason})" if got else "")
            )

    # Smoke-test the kill switch.
    os.environ["OPENFLIP_DISABLE_PROMISE_HOOK"] = "1"
    try:
        killed = _call("checking shutdown time:")
        if killed is not None:
            failures.append(
                "  FAIL: kill switch did not disable hook\n"
                f"    got: {killed}"
            )
    finally:
        del os.environ["OPENFLIP_DISABLE_PROMISE_HOOK"]

    if failures:
        print(f"{len(failures)} test(s) failed:")
        for f in failures:
            print(f)
        return len(failures)

    print(f"all {len(cases) + 1} tests passed")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_run_tests())
