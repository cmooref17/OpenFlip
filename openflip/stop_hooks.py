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
# Registry + dispatcher
# ---------------------------------------------------------------------------

# Registry of active hooks. Order matters only insofar as the FIRST hook to
# block wins — we don't run later hooks once one fires. Keep this list short
# and curated; every entry adds another regex pass to every text-only turn.
_HOOKS: list[tuple[str, HookFn]] = []


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
