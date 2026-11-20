"""PreToolUse / PostToolUse hooks wired into tool_executor.execute_tool_calls.

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_hooks_executor.py

Runs a real registered test tool through the real executor with real /bin/sh
hook commands. Guards: (1) exit 2 stops the tool from running and the model
gets the reason as a denied error; (2) updatedInput replaces the args the tool
receives; (3) a matcher for another tool leaves this tool alone; (4) PostToolUse
gets tool_input + tool_response on stdin and its feedback reaches the model via
build_model_feedback; (5) the ACL check still runs BEFORE hooks; (6) no hooks
configured = old behaviour, no hook_feedback.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openflip.hooks as hooks  # noqa: E402
from openflip.session import Session  # noqa: E402
from openflip.tool_executor import (  # noqa: E402
    CURRENT_AGENT, CURRENT_SESSION, CURRENT_TURN_OWNER, execute_tool_calls, build_model_feedback,
)
from openflip.tools._base import tool  # noqa: E402

hooks._global_hooks = lambda: {}
os.environ.pop("OPENFLIP_DISABLE_HOOKS", None)
_TMP = tempfile.mkdtemp(prefix="openflip_hooks_exec_")
CALLS: list[dict] = []


@tool
async def hook_probe(path: str = "") -> str:
    """Test-only tool for the hooks executor test.

    Args:
        path: any string; echoed back
    """
    CALLS.append({"path": path})
    return f"probe saw {path}"


def _agent(hcfg: dict):
    os.makedirs(os.path.join(_TMP, "a"), exist_ok=True)
    return SimpleNamespace(id="hooktest", path=os.path.join(_TMP, "a", "agent.json"), hooks=hcfg,
                           tool_response_mode="caption", memory_enabled=False,
                           provider="anthropic", model="m", display_name="T")


def _exec(agent, args=None, callable_names=frozenset({"hook_probe"})):
    sess = Session(transport="discord", transport_id="1", conversation_id="discord:1",
                   speaker_id=7, speaker_role_ids=[], is_owner=True, is_dm=True, display_name="o")
    msg = SimpleNamespace(tool_calls=[SimpleNamespace(function_name="hook_probe",
                                                      args=args or {"path": "/orig"},
                                                      tool_use_id="toolu_1")])

    async def go():
        ta, ts = CURRENT_AGENT.set(agent), CURRENT_SESSION.set(sess)
        to = CURRENT_TURN_OWNER.set(True)
        try:
            return await execute_tool_calls(agent=agent, conversation=None, ai_message=msg,
                                            callable_tool_names=set(callable_names), channel=None,
                                            speaker_id=7, session_id="discord:1", silent=True)
        finally:
            CURRENT_TURN_OWNER.reset(to)
            CURRENT_AGENT.reset(ta)
            CURRENT_SESSION.reset(ts)
    CALLS.clear()
    return asyncio.run(go())


def _pre(cmd, matcher="hook_probe"):
    return {"PreToolUse": [{"matcher": matcher, "hooks": [{"type": "command", "command": cmd}]}]}


def _post(cmd):
    return {"PostToolUse": [{"matcher": "hook_probe", "hooks": [{"type": "command", "command": cmd}]}]}


def test_pretooluse_exit2_stops_the_tool():
    out = _exec(_agent(_pre("echo 'not that path' >&2; exit 2")))
    assert CALLS == [], "the tool must not run"
    name, res = out[0]
    assert name == "hook_probe" and not res.ok and res.denied
    assert "not that path" in build_model_feedback(name, res)


def test_pretooluse_updated_input_replaces_args():
    upd = json.dumps({"hookSpecificOutput": {"permissionDecision": "allow",
                                             "updatedInput": {"path": "/rewritten"}}})
    out = _exec(_agent(_pre(f"echo '{upd}'")))
    assert CALLS == [{"path": "/rewritten"}]
    assert out[0][1].text == "probe saw /rewritten"


def test_matcher_for_other_tool_does_not_fire():
    _exec(_agent(_pre("exit 2", matcher="edit_file")))
    assert CALLS == [{"path": "/orig"}]


def test_posttooluse_sees_input_and_response_and_feeds_model():
    dump = os.path.join(_TMP, "post.json")
    a = _agent(_post(f"cat > {dump}; echo 'lint: 2 warnings' >&2; exit 2"))
    out = _exec(a)
    assert CALLS == [{"path": "/orig"}], "PostToolUse can't stop a tool that already ran"
    got = json.load(open(dump))
    assert got["hook_event_name"] == "PostToolUse" and got["tool_name"] == "hook_probe"
    assert got["tool_input"] == {"path": "/orig"} and got["tool_use_id"] == "toolu_1"
    assert got["tool_response"]["ok"] is True and got["tool_response"]["text"] == "probe saw /orig"
    assert got["speaker_is_owner"] is True, "CURRENT_TURN_OWNER was set for this turn"
    assert got["speaker_id"] == 7 and got["turn_kind"] == "human"
    assert got["transcript_path"].endswith(os.path.join("conversations", "discord:1.jsonl"))
    name, res = out[0]
    fb = build_model_feedback(name, res)
    assert "probe saw /orig" in fb and "[hook feedback]" in fb and "lint: 2 warnings" in fb
    assert res.text == "probe saw /orig", "user-facing text is untouched"


def test_acl_runs_before_hooks():
    marker = os.path.join(_TMP, "acl_marker")
    if os.path.exists(marker):
        os.remove(marker)
    out = _exec(_agent(_pre(f"touch {marker}")), callable_names=frozenset())
    assert CALLS == [] and out[0][1].denied
    assert not os.path.exists(marker), "a call the ACL refuses never reaches a hook"


def test_no_hooks_is_unchanged():
    out = _exec(_agent({}))
    assert CALLS == [{"path": "/orig"}]
    assert out[0][1].hook_feedback == []
    assert build_model_feedback("hook_probe", out[0][1]) == "probe saw /orig"


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
