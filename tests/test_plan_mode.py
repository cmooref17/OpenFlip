"""Plan mode (openflip/plan_mode.py + tools/plan.py, docs/goal_plan_reread.md).

Standalone runnable: .lvenv/bin/python tests/test_plan_mode.py

Claude Code's plan mode: research and plan, change nothing, present the plan,
the operator approves or sends it back. These drive the real code on a temp
agent dir: write tools and write-looking shell commands blocked through the
real execute_tool_calls while reads pass, exit_plan_mode saving + showing the
plan, the /plan text command (toggle, request, approve starts a turn with the
plan, reject keeps planning), the per-turn reminder, compaction re-inject, and
/reset clearing it.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openflip.plan_mode as pm  # noqa: E402
import openflip.tool_executor as te  # noqa: E402
from openflip.tools import TOOL_REGISTRY  # noqa: E402,F401
import openflip.tools.files as files  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="openflip_plan_")
CID = "discord:7"
files._check_access = lambda full, mode: None


def _dir(name):
    d = os.path.join(_TMP, name)
    os.makedirs(os.path.join(d, "conversations"), exist_ok=True)
    return d


def _run(d, name, args):
    agent = SimpleNamespace(id="t", path=os.path.join(d, "agent.json"), hooks={},
                            tool_response_mode="", memory_enabled=False)
    tc = SimpleNamespace(function_name=name, args=args, tool_use_id="x")
    t1 = te.CURRENT_SESSION.set(SimpleNamespace(conversation_id=CID, transport="discord", handle=""))
    t2 = te.CURRENT_AGENT.set(agent)
    try:
        return asyncio.run(te.execute_tool_calls(
            agent=agent, conversation=None, ai_message=SimpleNamespace(tool_calls=[tc]),
            callable_tool_names={name}, channel=None, speaker_id=1,
            session_id=CID, silent=True))[0][1]
    finally:
        te.CURRENT_SESSION.reset(t1); te.CURRENT_AGENT.reset(t2)


def test_blocks_writes_allows_reads_through_real_executor():
    d = _dir("exec")
    target = os.path.join(d, "x.txt")
    open(target, "w").write("orig")
    pm.turn_on(d, CID)
    r = _run(d, "write_file", {"path": os.path.join(d, "new.txt"), "content": "hi"})
    assert not r.ok and "Plan mode is on" in r.error and not os.path.exists(os.path.join(d, "new.txt"))
    r = _run(d, "edit_file", {"path": target, "old_string": "orig", "new_string": "changed"})
    assert not r.ok and open(target).read() == "orig"
    r = _run(d, "read_file", {"path": target})
    assert r.ok and "orig" in (r.model_feedback or "")
    pm.turn_off(d, CID)
    r = _run(d, "edit_file", {"path": target, "old_string": "orig", "new_string": "changed"})
    assert r.ok and open(target).read() == "changed"


def test_run_command_write_detection():
    bad = ["rm -rf x", "echo hi > f", "cat a >> b", "git commit -m x", "cd x && git push",
           "sudo pacman -S foo", "systemctl --user restart openflip", "sed -i s/a/b/ f",
           "pip install x", "mv a b", "kill 123", "ls; touch f", "curl -X POST http://x"]
    good = ["ls -la", "git status", "git log --oneline -5", "git diff", "grep -rn x . 2>/dev/null",
            "cat f | head", "sed -n 1,10p f", "find . -name x 2>&1 | head", "systemctl --user status x",
            "python3 -c 'print(1)'", "ps aux | grep x"]
    for c in bad:
        assert pm.blocked_reason("run_command", {"command": c}), c
    for c in good:
        assert not pm.blocked_reason("run_command", {"command": c}), c
    assert pm.blocked_reason("run_command", {"command": "ls", "run_in_background": True})


def test_exit_plan_mode_saves_and_shows():
    from openflip.tools.plan import exit_plan_mode
    d = _dir("exit")
    agent = SimpleNamespace(id="t", path=os.path.join(d, "agent.json"))
    t1 = te.CURRENT_SESSION.set(SimpleNamespace(conversation_id=CID))
    t2 = te.CURRENT_AGENT.set(agent)
    try:
        r = asyncio.run(exit_plan_mode(plan="1. do a\n2. test b"))
        assert not r.ok and "isn't on" in r.error
        pm.turn_on(d, CID)
        r = asyncio.run(exit_plan_mode(plan="1. do a\n2. test b"))
    finally:
        te.CURRENT_SESSION.reset(t1); te.CURRENT_AGENT.reset(t2)
    assert r.ok and "1. do a" in r.text and "/plan approve" in r.text
    assert "Stop here" in r.model_feedback
    assert pm.load_plan(d, CID) == "1. do a\n2. test b"
    assert pm.state(d, CID)["awaiting_approval"] is True


class _Runner:
    def __init__(self, d):
        self.agent = SimpleNamespace(id="t", path=os.path.join(d, "agent.json"))
        self.synthetic = []

    def conv_key_for_session(self, s, ch):
        return ch

    def get_conversation(self, *a, **k):
        return None

    async def run_synthetic_turn(self, target, prompt, **kw):
        self.synthetic.append((prompt, kw))


class _Chan:
    id = 1

    def __init__(self):
        self.sent = []

    async def send(self, t):
        self.sent.append(t)


def _cmd(r, text):
    import openflip.text_commands as tc
    tc.is_owner = lambda *a, **k: True
    ch = _Chan()
    asyncio.run(tc.handle_text_command(runner=r, channel=ch, speaker_id=5, raw_text=text,
                                       session=SimpleNamespace(conversation_id=CID, transport="discord", handle="")))
    return ch.sent


def test_plan_command_flow():
    d = _dir("cmd")
    r = _Runner(d)
    assert _cmd(r, "/plan")[0].startswith("Plan mode on") and pm.is_on(d, CID) and not r.synthetic
    assert _cmd(r, "/plan") == ["Plan mode off."] and not pm.is_on(d, CID)
    # request form: on + a turn
    _cmd(r, "/plan add a dark mode toggle")
    assert pm.is_on(d, CID) and "add a dark mode toggle" in r.synthetic[-1][0]
    # approve with nothing waiting
    assert _cmd(r, "/plan approve") == ["No plan is waiting for approval."]
    pm.save_plan(d, CID, "step 1\nstep 2")
    # reject keeps planning and passes feedback
    _cmd(r, "/plan no use css variables")
    assert pm.is_on(d, CID) and "use css variables" in r.synthetic[-1][0]
    assert not pm.state(d, CID)["awaiting_approval"]
    pm.save_plan(d, CID, "step 1\nstep 2")
    n = len(r.synthetic)
    assert _cmd(r, "/plan approve")[0].startswith("Plan approved")
    assert not pm.is_on(d, CID) and len(r.synthetic) == n + 1
    prompt, kw = r.synthetic[-1]
    assert "step 1\nstep 2" in prompt and kw["auto_post_final_text"] and kw["speaker_id"] == 5


def test_reminder_and_compaction_block():
    d = _dir("remind")
    assert pm.reminder(d, CID) == "" and pm.compaction_block(d, CID) == ""
    pm.turn_on(d, CID)
    assert "Plan mode is ON" in pm.reminder(d, CID)
    pm.save_plan(d, CID, "the plan body")
    assert "waiting on the operator" in pm.reminder(d, CID)
    cb = pm.compaction_block(d, CID)
    assert "the plan body" in cb and "waiting for the operator's approval" in cb
    pm.turn_off(d, CID)
    cb = pm.compaction_block(d, CID)
    assert "the plan body" in cb and "working from" in cb and "Plan mode is ON" not in cb
    pm.clear(d, CID)
    assert pm.compaction_block(d, CID) == "" and not pm.is_on(d, CID)


def test_parse_command():
    assert pm.parse_command("") == ("toggle", "")
    assert pm.parse_command("approve") == ("approve", "")
    assert pm.parse_command("yes") == ("approve", "")
    assert pm.parse_command("no change step 2") == ("reject", "change step 2")
    assert pm.parse_command("off") == ("off", "")
    assert pm.parse_command("yes add a button") == ("plan", "yes add a button")


def test_wiring():
    import openflip.runtime as rt
    import openflip.pipeline as pl
    import openflip.tool_deferral as td
    assert "exit_plan_mode" in pl.FRAMEWORK_TOOL_NAMES and "exit_plan_mode" in td.DEFAULT_ALWAYS_LOAD
    src = inspect.getsource(rt.AgentRunner._run_turn)
    assert "_pm.reminder(" in src
    assert src.index("_pm.reminder(") < src.index("conv.messages.append(ChatMessage('user', framed_user))")
    ex = inspect.getsource(te.execute_tool_calls)
    assert ex.index("_pm.blocked_reason(") < ex.index("_hooks.run_event(agent, \"PreToolUse\"")
    assert "_pm.clear(" in inspect.getsource(rt.AgentRunner.reset_conversation)


if __name__ == "__main__":
    failed = 0
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for n, f in tests:
        try:
            f(); print(f"PASS {n}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"FAIL {n}: {e!r}"); traceback.print_exc()
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
