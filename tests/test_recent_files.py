"""Re-reading recently touched files after compaction (openflip/recent_files.py).

Standalone runnable: .lvenv/bin/python tests/test_recent_files.py

Claude Code (context-window.md): after compaction it re-reads up to five files
read or edited this session, most recently modified first; a file over 5,000
tokens comes back as "Referenced file <path>" without its content. These tests
drive the real module on real temp files, the real tool_executor recording path
(a successful read_file through execute_tool_calls), and pin the runtime wiring.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openflip.recent_files as rf  # noqa: E402

_TMP = tempfile.mkdtemp(prefix="openflip_rf_")
CID = "discord:42"


def _file(name, text, age=0.0):
    p = os.path.join(_TMP, name)
    with open(p, "w") as f:
        f.write(text)
    t = time.time() - age
    os.utime(p, (t, t))
    return p


def test_top_five_by_mtime_newest_first():
    rf.forget(CID)
    paths = [_file(f"f{i}.txt", f"body {i}", age=100 - i) for i in range(7)]  # f6 newest
    for p in paths:
        rf.note_touched(CID, p)
    block = rf.reread_block(CID)
    order = [i for i in range(7) if f"f{i}.txt" in block]
    assert order == [2, 3, 4, 5, 6], order
    assert block.index("f6.txt") < block.index("f2.txt")
    assert "body 6" in block


def test_big_file_is_reference_only():
    rf.forget(CID)
    big = _file("big.txt", "y" * (rf.MAX_TOKENS_PER_FILE * rf.CHARS_PER_TOKEN + 10))
    rf.note_touched(CID, big)
    block = rf.reread_block(CID)
    assert f"Referenced file {big}" in block and "yyyy" not in block


def test_deleted_files_skipped_and_empty_when_none():
    rf.forget(CID)
    assert rf.reread_block(CID) == ""
    p = _file("gone.txt", "x")
    rf.note_touched(CID, p)
    os.remove(p)
    assert rf.reread_block(CID) == ""


def test_conversations_are_separate_and_forget_works():
    rf.forget(CID); rf.forget("other")
    p = _file("mine.txt", "mine")
    rf.note_touched(CID, p)
    assert rf.reread_block("other") == ""
    rf.forget(CID)
    assert rf.reread_block(CID) == ""


def test_retouch_moves_to_end_and_list_is_bounded():
    rf.forget(CID)
    for i in range(rf.MAX_TRACKED + 10):
        rf.note_touched(CID, f"/nope/{i}")
    rf.note_touched(CID, "/nope/15")
    t = rf.touched(CID)
    assert len(t) == rf.MAX_TRACKED and t[-1] == "/nope/15"


def test_executor_records_successful_read_file():
    """Drive the real execute_tool_calls with a real read_file call."""
    from types import SimpleNamespace
    import openflip.tool_executor as te
    from openflip.tools import TOOL_REGISTRY  # noqa: F401  (registers tools)
    rf.forget(CID)
    p = _file("via_executor.txt", "hello executor")
    agent = SimpleNamespace(id="t", path=os.path.join(_TMP, "agent.json"), hooks={}, tool_response_mode="")
    tok = te.CURRENT_SESSION.set(SimpleNamespace(conversation_id=CID, transport="discord", handle=""))
    import openflip.tools.files as files
    files._check_access = lambda full, mode: None   # path ACLs aren't under test here
    tc = SimpleNamespace(function_name="read_file", args={"path": p}, tool_use_id="x")
    ai_message = SimpleNamespace(tool_calls=[tc])
    agent.memory_enabled = False
    try:
        res = asyncio.run(te.execute_tool_calls(
            agent=agent, conversation=None, ai_message=ai_message,
            callable_tool_names={"read_file"}, channel=None, speaker_id=1,
            session_id="discord:42", silent=True))
    finally:
        te.CURRENT_SESSION.reset(tok)
    assert res and res[0][1].ok, res
    assert os.path.realpath(p) in [os.path.realpath(x) for x in rf.touched(CID)], rf.touched(CID)


def test_runtime_wiring():
    import openflip.runtime as rt
    src = inspect.getsource(rt.AgentRunner._run_turn)
    i_note = src.index("'[FRAMEWORK]: Your conversation was just compacted")
    assert src.index("_rf.reread_block(_cc_cid)") < i_note
    assert src.index("_pm.compaction_block(_agent_dir, _cc_cid)") < i_note
    assert "_after" in src[i_note:i_note + 600]
    assert "_rf.forget(_g_cid)" in inspect.getsource(rt.AgentRunner.reset_conversation)


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
