"""Hooks core (openflip/hooks.py, design in docs/hooks.md).

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_hooks.py

Every command hook here is a real /bin/sh command run through the real
subprocess path; prompt hooks use a fake PROMPT_CALLER so no network is used.
config.json hooks are switched off by stubbing _global_hooks, so the test never
depends on the live config.
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

hooks._global_hooks = lambda: {}
os.environ.pop("OPENFLIP_DISABLE_HOOKS", None)
_TMP = tempfile.mkdtemp(prefix="openflip_hooks_test_")


def _agent(cfg: dict):
    os.makedirs(os.path.join(_TMP, "agent"), exist_ok=True)
    return SimpleNamespace(id="testagent", path=os.path.join(_TMP, "agent", "agent.json"), hooks=cfg)


def _cmd(event: str, command: str, matcher=None, **extra) -> dict:
    group = {"hooks": [{"type": "command", "command": command, **extra}]}
    if matcher is not None:
        group["matcher"] = matcher
    return {event: [group]}


def _run(agent, event, payload=None, match=""):
    return asyncio.run(hooks.run_event(agent, event, payload or {"hook_event_name": event}, match))


def test_matcher_table():
    m = hooks.matcher_matches
    assert m(None, "edit_file") and m("", "edit_file") and m("*", "x")
    assert m("edit_file", "edit_file") and not m("edit_file", "edit_file_x")
    assert m("edit_file|write_file", "write_file") and m("edit_file, write_file", "write_file")
    assert not m("edit", "edit_file"), "a simple name is exact, not a substring"
    assert m("^mtg_", "mtg_draw") and not m("^mtg_", "x_mtg_draw"), "special chars -> regex"
    assert m("file$", "read_file"), "regex is unanchored re.search"
    assert not m("(", "anything"), "bad regex matches nothing"


def test_matcher_filters_tool_events_only():
    a = _agent({**_cmd("PreToolUse", "exit 2", matcher="edit_file"),
                **_cmd("Stop", "exit 2", matcher="nothing_matches_this")})
    assert hooks.handlers_for(a, "PreToolUse", "read_file") == []
    assert len(hooks.handlers_for(a, "PreToolUse", "edit_file")) == 1
    assert len(hooks.handlers_for(a, "Stop")) == 1, "Stop ignores the matcher"


def test_command_gets_event_json_on_stdin():
    out = os.path.join(_TMP, "stdin.json")
    a = _agent(_cmd("PostToolUse", f"cat > {out}"))
    payload = {"hook_event_name": "PostToolUse", "tool_name": "read_file", "tool_input": {"path": "/x"}}
    r = _run(a, "PostToolUse", payload, "read_file")
    assert not r.blocked and not r.errors
    got = json.load(open(out))
    assert got["tool_name"] == "read_file" and got["tool_input"] == {"path": "/x"}


def test_command_env_and_cwd():
    out = os.path.join(_TMP, "env.txt")
    a = _agent(_cmd("UserPromptSubmit", f'echo "$OPENFLIP_AGENT_ID $OPENFLIP_HOOK_EVENT $(pwd)" > {out}'))
    _run(a, "UserPromptSubmit")
    line = open(out).read().split()
    assert line[0] == "testagent" and line[1] == "UserPromptSubmit"
    assert os.path.realpath(line[2]) == os.path.realpath(os.path.join(_TMP, "agent"))


def test_exit2_blocks_with_stderr_reason():
    a = _agent(_cmd("PreToolUse", "echo 'no deleting prod' >&2; exit 2"))
    r = _run(a, "PreToolUse", match="run_command")
    assert r.blocked and r.reason == "no deleting prod"


def test_exit2_on_posttooluse_is_context_not_block():
    a = _agent(_cmd("PostToolUse", "echo 'tests failed: 3' >&2; exit 2"))
    r = _run(a, "PostToolUse", match="edit_file")
    assert not r.blocked, "PostToolUse can't block, the tool already ran"
    assert r.context == ["tests failed: 3"]


def test_other_exit_code_is_non_blocking_error():
    a = _agent(_cmd("PreToolUse", "echo boom >&2; exit 1"))
    r = _run(a, "PreToolUse", match="x")
    assert not r.blocked and len(r.errors) == 1 and "exited 1" in r.errors[0]


def test_json_deny_and_updated_input():
    deny = json.dumps({"hookSpecificOutput": {"permissionDecision": "deny",
                                              "permissionDecisionReason": "nope"}})
    a = _agent(_cmd("PreToolUse", f"echo '{deny}'"))
    r = _run(a, "PreToolUse", match="x")
    assert r.blocked and r.reason == "nope"
    upd = json.dumps({"hookSpecificOutput": {"permissionDecision": "allow",
                                             "updatedInput": {"path": "/safe"}}})
    a = _agent(_cmd("PreToolUse", f"echo '{upd}'"))
    r = _run(a, "PreToolUse", match="x")
    assert not r.blocked and r.allowed and r.updated_input == {"path": "/safe"}


def test_deny_beats_allow_across_hooks():
    allow = json.dumps({"hookSpecificOutput": {"permissionDecision": "allow",
                                               "updatedInput": {"a": 1}}})
    a = _agent({"PreToolUse": [{"hooks": [
        {"type": "command", "command": f"echo '{allow}'"},
        {"type": "command", "command": "echo denied >&2; exit 2"},
    ]}]})
    r = _run(a, "PreToolUse", match="x")
    assert r.blocked and not r.allowed and r.updated_input is None


def test_stop_decision_block_json():
    blk = json.dumps({"decision": "block", "reason": "run the tests first"})
    a = _agent(_cmd("Stop", f"echo '{blk}'"))
    r = _run(a, "Stop")
    assert r.blocked and r.reason == "run the tests first"
    assert "run the tests first" in hooks.stop_feedback(r)


def test_plain_stdout_is_context_on_userpromptsubmit_only():
    a = _agent({**_cmd("UserPromptSubmit", "echo 'today is payday'"),
                **_cmd("PreToolUse", "echo 'ignored text'")})
    assert _run(a, "UserPromptSubmit").context == ["today is payday"]
    assert _run(a, "PreToolUse", match="x").context == []


def test_invalid_json_is_non_blocking():
    a = _agent(_cmd("Stop", "echo '{not json'"))
    r = _run(a, "Stop")
    assert not r.blocked and r.errors and "invalid JSON" in r.errors[0]


def test_timeout_is_non_blocking():
    a = _agent(_cmd("Stop", "sleep 5; exit 2", timeout=1))
    r = _run(a, "Stop")
    assert not r.blocked and r.errors and "timed out" in r.errors[0]


def test_duplicate_handler_runs_once():
    out = os.path.join(_TMP, "dup.txt")
    if os.path.exists(out):
        os.remove(out)
    cmd = _cmd("Stop", f"echo x >> {out}")
    a = _agent(cmd)
    old = hooks._global_hooks
    hooks._global_hooks = lambda: cmd
    try:
        _run(a, "Stop")
    finally:
        hooks._global_hooks = old
    assert open(out).read().count("x") == 1


def test_kill_switch():
    a = _agent(_cmd("PreToolUse", "exit 2"))
    os.environ["OPENFLIP_DISABLE_HOOKS"] = "1"
    try:
        r = _run(a, "PreToolUse", match="x")
        assert not r.blocked and not hooks.has_hooks(a, "PreToolUse")
    finally:
        os.environ.pop("OPENFLIP_DISABLE_HOOKS", None)
    assert hooks.has_hooks(a, "PreToolUse")


def _with_prompt_answer(answer: str):
    seen = {}

    async def fake(model, prompt, timeout):
        seen["model"], seen["prompt"] = model, prompt
        return answer
    hooks.PROMPT_CALLER = fake
    return seen


def test_prompt_hook_ok_false_blocks_stop_and_fills_arguments():
    seen = _with_prompt_answer('{"ok": false, "reason": "tests not run"}')
    a = _agent({"Stop": [{"hooks": [{"type": "prompt", "model": "anthropic/claude-x",
                                     "prompt": "Is it done? $ARGUMENTS"}]}]})
    r = _run(a, "Stop", {"hook_event_name": "Stop", "last_assistant_message": "all done!"})
    assert r.blocked and r.reason == "tests not run"
    assert seen["model"] == "claude-x", "provider prefix stripped"
    assert '"last_assistant_message": "all done!"' in seen["prompt"]


def test_prompt_hook_ok_true_and_impossible_and_garbage():
    a = _agent({"Stop": [{"hooks": [{"type": "prompt", "model": "m", "prompt": "done?"}]}]})
    _with_prompt_answer('{"ok": true}')
    assert not _run(a, "Stop").blocked
    _with_prompt_answer('Sure! ```json\n{"ok": false, "reason": "never", "impossible": true}\n```')
    r = _run(a, "Stop")
    assert r.blocked and r.impossible
    _with_prompt_answer("I think it's fine")
    r = _run(a, "Stop")
    assert not r.blocked and r.errors


def test_stop_state_cap():
    s = hooks.StopState()
    assert not s.active and not s.capped()
    s.continuations = hooks.STOP_CONTINUATION_CAP
    assert s.active and s.capped()


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
