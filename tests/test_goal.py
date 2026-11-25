"""/goal (openflip/goal.py, docs/goal_plan_reread.md): Claude Code's model-checked
"keep working until X is true".

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_goal.py

The evaluator model is faked (goal.EVAL_CALLER), everything else is the real
code on a real temp agent dir: the saved goal file, on_stop's three verdicts,
the no-progress pause, background deferral, error clearing, the /goal text
command (set starts a turn, status, clear + aliases), /reset clearing it, and
the runtime wiring pinned by source inspection.
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

import openflip.goal as goal  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="openflip_goal_")
CID = "discord:1"
_calls: list[str] = []


def _fake(verdict: str, reason: str = "r"):
    async def caller(model, prompt, timeout):
        _calls.append(prompt)
        return json.dumps({"verdict": verdict, "reason": reason})
    return caller


goal.evaluator_model = lambda: "fake-small-model"


def _dir(name: str) -> str:
    d = os.path.join(_TMP, name)
    os.makedirs(os.path.join(d, "conversations"), exist_ok=True)
    return d


def _msgs(*texts):
    return [SimpleNamespace(role="assistant" if i % 2 else "user", content=t) for i, t in enumerate(texts)]


def _stop(d, used=True, bg=False, msgs=None):
    return asyncio.run(goal.on_stop(d, CID, msgs or _msgs("do it", "ran tests: 3 failed"),
                                    used_tools_this_turn=used, background_running=bg))


def test_not_met_keeps_going_with_reason():
    d = _dir("notmet")
    goal.set_goal(d, CID, "all tests pass")
    goal.EVAL_CALLER = _fake("not_met", "3 tests still fail")
    dec = _stop(d)
    assert dec.keep_going and "3 tests still fail" in dec.feedback and dec.feedback.startswith("[FRAMEWORK]")
    g = goal.active_goal(d, CID)
    assert g.turns_evaluated == 1 and g.last_reason == "3 tests still fail"


def test_met_records_achieved_and_ends():
    d = _dir("met")
    goal.set_goal(d, CID, "all tests pass")
    goal.EVAL_CALLER = _fake("met", "tests/run.sh shows 0 failed")
    dec = _stop(d)
    assert not dec.keep_going and dec.notice.startswith("Goal achieved: all tests pass")
    assert goal.active_goal(d, CID) is None
    st = goal.status_text(d, CID)
    assert "Last achieved: all tests pass" in st, st


def test_impossible_clears_goal():
    d = _dir("imp")
    goal.set_goal(d, CID, "make 1 equal 2")
    goal.EVAL_CALLER = _fake("impossible", "math")
    dec = _stop(d)
    assert not dec.keep_going and "judged impossible: math" in dec.notice
    assert goal.load_goal(d, CID) is None


def test_no_progress_pauses_after_limit_and_resumes():
    d = _dir("noprog")
    goal.set_goal(d, CID, "x")
    goal.EVAL_CALLER = _fake("not_met", "nothing done")
    for _ in range(goal.NO_PROGRESS_LIMIT - 1):
        assert _stop(d, used=False).keep_going
    dec = _stop(d, used=False)
    assert not dec.keep_going and dec.notice.startswith("Goal paused")
    assert goal.active_goal(d, CID).paused
    # paused: no evaluation at all
    n = len(_calls)
    assert _stop(d).keep_going is False and len(_calls) == n
    goal.unpause_on_operator_message(d, CID)
    assert _stop(d).keep_going


def test_tool_use_resets_no_progress():
    d = _dir("reset_np")
    goal.set_goal(d, CID, "x")
    goal.EVAL_CALLER = _fake("not_met")
    _stop(d, used=False); _stop(d, used=False)
    _stop(d, used=True)
    assert goal.active_goal(d, CID).no_progress == 0


def test_background_work_defers_evaluation():
    d = _dir("bg")
    goal.set_goal(d, CID, "x")
    goal.EVAL_CALLER = _fake("met")
    n = len(_calls)
    dec = _stop(d, bg=True)
    assert not dec.keep_going and not dec.notice and len(_calls) == n
    assert goal.active_goal(d, CID) is not None


def test_evaluator_error_ends_turn_keeps_goal():
    d = _dir("err")
    goal.set_goal(d, CID, "x")

    async def boom(*a):
        raise RuntimeError("network down")
    goal.EVAL_CALLER = boom
    dec = _stop(d)
    assert not dec.keep_going and "Goal check failed" in dec.notice
    assert goal.active_goal(d, CID) is not None


def test_bad_json_is_error_not_met():
    assert goal._parse_verdict("sure, looks done").verdict == "error"
    assert goal._parse_verdict('{"verdict": "Not Met", "reason": "x"}').verdict == "not_met"


def test_no_goal_does_nothing():
    d = _dir("none")
    n = len(_calls)
    dec = _stop(d)
    assert not dec.keep_going and not dec.notice and len(_calls) == n


def test_unrecoverable_error_clears_other_errors_keep():
    d = _dir("fwerr")
    goal.set_goal(d, CID, "x")
    assert goal.on_framework_error(d, CID, "rate_limit") == ""
    assert goal.active_goal(d, CID) is not None
    note = goal.on_framework_error(d, CID, "auth")
    assert note.startswith("Goal cleared after an unrecoverable error") and goal.active_goal(d, CID) is None


def test_evaluator_sees_condition_and_recent_transcript():
    d = _dir("prompt")
    goal.set_goal(d, CID, "CHANGELOG has every PR")
    goal.EVAL_CALLER = _fake("not_met")
    _stop(d, msgs=_msgs("go", "I added 4 of 5 entries"))
    assert "CHANGELOG has every PR" in _calls[-1] and "I added 4 of 5 entries" in _calls[-1]


def test_transcript_tail_keeps_newest():
    msgs = _msgs(*[f"message {i} " + "x" * 100 for i in range(500)])
    t = goal.transcript_tail(msgs, limit=2000)
    assert "message 499" in t and "message 0 " not in t and len(t) <= 2100


def test_parse_command_and_aliases():
    assert goal.parse_command("") == ("status", "")
    for w in ("clear", "STOP", "off", "reset", "none", "cancel"):
        assert goal.parse_command(w)[0] == "clear"
    assert goal.parse_command("tests pass") == ("set", "tests pass")


def test_condition_capped_and_survives_reload():
    d = _dir("cap")
    goal.set_goal(d, CID, "y" * 5000)
    assert len(goal.load_goal(d, CID).condition) == goal.MAX_CONDITION_CHARS


# ----- /goal text command, driven through the real handle_text_command -----

class _Runner:
    def __init__(self, d):
        self.agent = SimpleNamespace(id="t", path=os.path.join(d, "agent.json"))
        self.synthetic: list[tuple] = []

    def conv_key_for_session(self, session, ch_id):
        return ch_id

    def conv_key(self, ch_id):
        return ch_id

    def get_conversation(self, *a, **k):
        return None

    async def run_synthetic_turn(self, target, prompt, **kw):
        self.synthetic.append((target, prompt, kw))


class _Chan:
    def __init__(self):
        self.id = 1
        self.sent: list[str] = []

    async def send(self, text):
        self.sent.append(text)


def _cmd(runner, text):
    import openflip.text_commands as tc
    tc.is_owner = lambda *a, **k: True
    ch = _Chan()
    sess = SimpleNamespace(conversation_id=CID, transport="discord", handle="")
    asyncio.run(tc.handle_text_command(runner=runner, channel=ch, speaker_id=7,
                                       raw_text=text, session=sess))
    return ch.sent


def test_goal_command_set_starts_turn_status_clear():
    d = _dir("cmd")
    r = _Runner(d)
    sent = _cmd(r, "/goal tests/run.sh exits 0")
    assert sent == ["Goal set: tests/run.sh exits 0"], sent
    assert len(r.synthetic) == 1
    _, prompt, kw = r.synthetic[0]
    assert "tests/run.sh exits 0" in prompt and kw["auto_post_final_text"] is True and kw["speaker_id"] == 7
    assert "Goal active: tests/run.sh exits 0" in _cmd(r, "/goal")[0]
    assert _cmd(r, "/goal cancel") == ["Goal cleared: tests/run.sh exits 0"]
    assert _cmd(r, "/goal clear") == ["No goal set"]


def test_goal_command_owner_only():
    import openflip.text_commands as tc
    d = _dir("cmd_owner")
    r = _Runner(d)
    tc.is_owner = lambda *a, **k: False
    ch = _Chan()
    sess = SimpleNamespace(conversation_id=CID, transport="discord", handle="")
    asyncio.run(tc.handle_text_command(runner=r, channel=ch, speaker_id=9, raw_text="/goal x", session=sess))
    assert ch.sent == ["Owner only."] and goal.load_goal(d, CID) is None and not r.synthetic


# ----- runtime wiring (source pins so a refactor can't silently drop it) -----

def test_runtime_wiring():
    import openflip.runtime as rt
    src = inspect.getsource(rt.AgentRunner._run_turn)
    i_goal = src.index("_goal_mod.on_stop(")
    i_break = src.index("# NOTE: no post_drain_retry here.")
    i_stop_hook = src.index("_hooks.evaluate_stop(")
    assert i_stop_hook < i_goal < i_break, "goal check must sit after Stop hooks, before the exit break"
    assert "_goal_eligible = result_future is None and depth == 0" in src
    assert "_goal_mod.on_framework_error(" in src and "_goal_notice_pending" in src
    # The operator-message unpause must run AFTER conv exists (it reads conv).
    assert src.index("conv = self.get_conversation(") < src.index("unpause_on_operator_message(")
    reset_src = inspect.getsource(rt.AgentRunner.reset_conversation)
    assert "_goal_mod.save_goal(" in reset_src


def test_background_owner_tracking():
    import openflip.tools.run_command as rc
    src = inspect.getsource(rc.run_command)
    assert "_BG_OWNERS[job_id]" in src
    loop = asyncio.new_event_loop()
    fut = loop.create_future()
    rc._BG_JOBS["bg-x"] = fut
    rc._BG_OWNERS["bg-x"] = CID
    try:
        assert goal.background_running_for(CID) is True
        assert goal.background_running_for("discord:2") is False
        fut.set_result(None)
        assert goal.background_running_for(CID) is False
    finally:
        rc._BG_JOBS.pop("bg-x", None); rc._BG_OWNERS.pop("bg-x", None)
        loop.close()


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        try:
            f()
            print(f"PASS {n}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {n}: {e!r}")
            traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
