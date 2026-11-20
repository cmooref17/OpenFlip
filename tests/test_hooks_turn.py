"""Stop + UserPromptSubmit hooks at the turn level (hooks.evaluate_stop, and
the runtime wiring checked by source inspection).

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_hooks_turn.py

evaluate_stop is the exact function runtime._run_turn calls at its exit gate,
driven here with real /bin/sh Stop hooks. Guards: a blocking Stop hook keeps
the turn going with its reason; stop_hook_active is false on the first check
and true on re-entry; the 8-continuation cap lets the turn end; STAY_SILENT
and impossible:true end it; no hooks = None (old behaviour). The wiring tests
pin that runtime runs UserPromptSubmit before the user message is appended and
Stop before the exit-gate break, so a refactor can't silently drop them.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openflip.hooks as hooks  # noqa: E402

hooks._global_hooks = lambda: {}
os.environ.pop("OPENFLIP_DISABLE_HOOKS", None)
_TMP = tempfile.mkdtemp(prefix="openflip_hooks_turn_")


def _agent(cmd: str):
    os.makedirs(os.path.join(_TMP, "a"), exist_ok=True)
    return SimpleNamespace(id="t", path=os.path.join(_TMP, "a", "agent.json"),
                           hooks={"Stop": [{"hooks": [{"type": "command", "command": cmd}]}]})


def _eval(agent, state, text="All done!"):
    return asyncio.run(hooks.evaluate_stop(agent, state, {"hook_event_name": "Stop"}, text))


def test_blocking_stop_hook_keeps_turn_going():
    s = hooks.StopState()
    fb = _eval(_agent("echo 'you never ran the tests' >&2; exit 2"), s)
    assert fb is not None and "you never ran the tests" in fb and fb.startswith("[FRAMEWORK]")
    assert s.continuations == 1


def test_stop_hook_active_flag_and_last_message():
    dump = os.path.join(_TMP, "stop_in.jsonl")
    a = _agent(f"cat >> {dump}; echo >> {dump}; exit 2")
    s = hooks.StopState()
    _eval(a, s, "first try")
    _eval(a, s, "second try")
    rows = [json.loads(l) for l in open(dump) if l.strip()]
    assert [r["stop_hook_active"] for r in rows] == [False, True]
    assert [r["last_assistant_message"] for r in rows] == ["first try", "second try"]


def test_cap_lets_turn_end():
    a = _agent("exit 2")
    s = hooks.StopState()
    results = [_eval(a, s) for _ in range(hooks.STOP_CONTINUATION_CAP + 2)]
    assert all(r is not None for r in results[:hooks.STOP_CONTINUATION_CAP])
    assert results[hooks.STOP_CONTINUATION_CAP] is None, "the 9th stop is allowed"
    assert s.continuations == hooks.STOP_CONTINUATION_CAP


def test_allowing_hook_and_no_hooks_end_turn():
    assert _eval(_agent("exit 0"), hooks.StopState()) is None
    no = SimpleNamespace(id="n", path=os.path.join(_TMP, "a", "agent.json"), hooks={})
    assert _eval(no, hooks.StopState()) is None
    assert asyncio.run(hooks.evaluate_stop(_agent("exit 2"), None, {}, "x")) is None, \
        "worker turns pass state=None"


def test_stay_silent_is_never_blocked():
    marker = os.path.join(_TMP, "silent_marker")
    s = hooks.StopState()
    assert _eval(_agent(f"touch {marker}; exit 2"), s, "STAY_SILENT") is None
    assert not os.path.exists(marker) and s.continuations == 0


def test_impossible_prompt_answer_ends_turn():
    async def fake(model, prompt, timeout):
        return '{"ok": false, "reason": "the file was deleted", "impossible": true}'
    hooks.PROMPT_CALLER = fake
    a = SimpleNamespace(id="p", path=os.path.join(_TMP, "a", "agent.json"),
                        hooks={"Stop": [{"hooks": [{"type": "prompt", "model": "m", "prompt": "done?"}]}]})
    s = hooks.StopState()
    assert _eval(a, s) is None and s.continuations == 0


def test_runtime_wiring_order():
    from openflip.runtime import AgentRunner
    src = inspect.getsource(AgentRunner._run_turn)
    ups = src.index('run_event(self.agent, "UserPromptSubmit"')
    append_user = src.index("conv.messages.append(ChatMessage('user', framed_user))")
    assert ups < append_user, "UserPromptSubmit must run before the user message enters history"
    gate = src.index("needs_follow_up = bool(_tc)")
    stop = src.index("_hooks.evaluate_stop(")
    brk = src.index("# NOTE: no post_drain_retry here.")
    assert gate < stop < brk, "Stop hooks run inside the exit gate, before its break"
    assert "_hooks.StopState() if result_future is None else None" in src, \
        "worker turns get no Stop hooks"
    assert "verbatim_user_message is None and _hooks.has_hooks" in src, "/redo skips UserPromptSubmit"
    human = src.index("CURRENT_TURN_FROM_HUMAN.set(inbound is not None or discord_message is not None)")
    assert human < ups, "turn_kind's human flag is set before the first hook can fire"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"{'FAILED' if failed else 'OK'}: {failed} failed")
    sys.exit(1 if failed else 0)
