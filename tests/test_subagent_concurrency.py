"""spawn_subagents concurrency safety: per-worker executor lock + per-file
mutation locks in files.py.

Standalone runnable script (no pytest in this venv):

    .lvenv/bin/python tests/test_subagent_concurrency.py

Background: the executor's `_inflight` key is (agent, speaker, tool) and a
held lock is REJECTED ("Already running"), not awaited. A worker turn carries
the OWNER's speaker_id, so two parallel workers calling the same tool used to
collide. Workers now lock under their own session `speaker_id`. Separately,
write_file / edit_file now hold a per-realpath asyncio.Lock across their
check-and-write so concurrent callers serialize instead of clobbering.

What this guards:
  (1) two worker sessions (distinct speaker ids) calling the same tool at
      once BOTH get through the lock — no "Already running"; the turn's
      speaker attribution (CURRENT_SPEAKER_ID) is still the turn speaker;
  (2) two NON-worker calls with the same turn speaker still collide exactly
      as before (second is refused "Already running");
  (3) two concurrent edit_file calls on different lines of one file both
      land — final content carries both edits;
  (4) edit_file actually waits on the per-path lock (held externally → the
      edit is pending and the file untouched; released → it lands), and the
      lock is keyed by realpath (a symlink spelling shares it);
  (5) write_file race on the same new path: one succeeds, the other fails
      cleanly as already-exists, file content is one of the two payloads.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip import tool_executor as te
from openflip.session import Session, make_subagent_session
from openflip.tool_executor import (
    CURRENT_AGENT, CURRENT_SESSION, CURRENT_SPEAKER_ID, execute_tool_calls,
)
from openflip.tools import files as files_mod
from openflip.tools._base import tool, TOOL_REGISTRY
from openflip.tools.files import write_file, edit_file

FAILURES: list[str] = []
OWNER_ID = 999


def check(label: str, cond: bool) -> None:
    print(("  ok    " if cond else "  FAIL  ") + label)
    if not cond:
        FAILURES.append(label)


# --- a slow probe tool: blocks until the test releases it, records the
# speaker attribution it observed ------------------------------------------
_probe_gate: asyncio.Event | None = None
_probe_started = 0
_probe_speakers: list[int] = []


@tool
async def concurrency_probe() -> str:
    """Test-only probe tool: waits on a gate, then returns.

    Args:
    """
    global _probe_started
    _probe_started += 1
    _probe_speakers.append(int(CURRENT_SPEAKER_ID.get(0) or 0))
    assert _probe_gate is not None
    await _probe_gate.wait()
    return "probe done"


def _agent():
    return SimpleNamespace(id="orch", tool_response_mode="caption", memory_enabled=False,
                           provider="anthropic", model="claude-opus-5-5", display_name="Orch")


def _ai_message():
    return SimpleNamespace(tool_calls=[SimpleNamespace(function_name="concurrency_probe", args={})])


def _discord_session() -> Session:
    return Session(
        transport="discord", transport_id="123", conversation_id="discord:123",
        speaker_id=OWNER_ID, speaker_role_ids=[], is_owner=True, is_dm=True, display_name="owner",
    )


async def _exec_as(agent, session):
    """Run one execute_tool_calls under a session, the way a worker/owner turn
    would: speaker_id is the TURN speaker (the owner) in every case."""
    tok_a = CURRENT_AGENT.set(agent)
    tok_s = CURRENT_SESSION.set(session)
    try:
        return await execute_tool_calls(
            agent=agent, conversation=None, ai_message=_ai_message(),
            callable_tool_names={"concurrency_probe"}, channel=None,
            speaker_id=OWNER_ID, session_id=session.conversation_id, silent=True,
        )
    finally:
        CURRENT_AGENT.reset(tok_a)
        CURRENT_SESSION.reset(tok_s)


async def _spin(n: int = 10) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


async def test_worker_lock() -> None:
    global _probe_gate, _probe_started
    print("(1) two workers (distinct speaker ids) call the same tool concurrently")
    agent = _agent()
    w1 = make_subagent_session("orch", "task-aaaa", ["concurrency_probe"])
    w2 = make_subagent_session("orch", "task-bbbb", ["concurrency_probe"])
    check("worker sessions have distinct speaker ids", w1.speaker_id != w2.speaker_id)
    check("worker speaker ids are not the turn speaker",
          w1.speaker_id != OWNER_ID and w2.speaker_id != OWNER_ID)
    _probe_gate = asyncio.Event()
    _probe_started = 0
    _probe_speakers.clear()
    t1 = asyncio.create_task(_exec_as(agent, w1))
    t2 = asyncio.create_task(_exec_as(agent, w2))
    await _spin()
    check("both workers entered the tool before either finished", _probe_started == 2)
    check("worker-keyed locks are held",
          te._inflight[("orch", w1.speaker_id, "concurrency_probe")].locked()
          and te._inflight[("orch", w2.speaker_id, "concurrency_probe")].locked())
    check("owner-keyed lock is NOT held by a worker",
          not te._inflight[("orch", OWNER_ID, "concurrency_probe")].locked())
    _probe_gate.set()
    r1, r2 = await asyncio.gather(t1, t2)
    ok1 = r1 and r1[0][1].ok
    ok2 = r2 and r2[0][1].ok
    check("worker 1 result ok", bool(ok1))
    check("worker 2 result ok (no 'Already running')", bool(ok2))
    check("speaker attribution inside the tool is still the turn speaker",
          _probe_speakers == [OWNER_ID, OWNER_ID])


async def test_non_worker_collision() -> None:
    global _probe_gate, _probe_started
    print("(2) two non-worker calls with the same turn speaker still collide")
    agent = _agent()
    s1 = _discord_session()
    s2 = _discord_session()
    _probe_gate = asyncio.Event()
    _probe_started = 0
    _probe_speakers.clear()
    t1 = asyncio.create_task(_exec_as(agent, s1))
    await _spin()
    check("first call is inside the tool", _probe_started == 1)
    check("owner-keyed lock is held", te._inflight[("orch", OWNER_ID, "concurrency_probe")].locked())
    r2 = await _exec_as(agent, s2)
    check("second call did NOT enter the tool", _probe_started == 1)
    check("second call refused as 'Already running'",
          bool(r2) and not r2[0][1].ok and "Already running" in (r2[0][1].error or ""))
    _probe_gate.set()
    r1 = await t1
    check("first call completed ok", bool(r1) and r1[0][1].ok)


def _files_agent(root: str):
    return SimpleNamespace(
        id="orch", path=os.path.join(root, "agent.json"),
        allowed_read_paths=[root], allowed_write_paths=[root], denied_paths=[],
    )


async def _as_owner(agent, coro_factory):
    tok_a = CURRENT_AGENT.set(agent)
    tok_s = CURRENT_SESSION.set(_discord_session())
    try:
        return await coro_factory()
    finally:
        CURRENT_AGENT.reset(tok_a)
        CURRENT_SESSION.reset(tok_s)


async def test_file_locks(tmp: str) -> None:
    agent = _files_agent(tmp)

    print("(3) two concurrent edit_file calls on different lines both land")
    f = os.path.join(tmp, "both.txt")
    with open(f, "w", encoding="utf-8") as fh:
        fh.write("line one\nline two\nline three\n")
    r1, r2 = await asyncio.gather(
        _as_owner(agent, lambda: edit_file(f, "line one", "LINE ONE")),
        _as_owner(agent, lambda: edit_file(f, "line three", "LINE THREE")),
    )
    check("edit A ok", r1.ok)
    check("edit B ok", r2.ok)
    with open(f, encoding="utf-8") as fh:
        body = fh.read()
    check("final content has both edits", body == "LINE ONE\nline two\nLINE THREE\n")

    print("(4) edit_file waits on the per-path lock; lock is keyed by realpath")
    g = os.path.join(tmp, "gated.txt")
    with open(g, "w", encoding="utf-8") as fh:
        fh.write("alpha\n")
    link = os.path.join(tmp, "gated-link.txt")
    os.symlink(g, link)
    check("symlink spelling shares the lock", files_mod._path_lock(link) is files_mod._path_lock(g))
    lock = files_mod._path_lock(g)
    await lock.acquire()
    t = asyncio.create_task(_as_owner(agent, lambda: edit_file(link, "alpha", "beta")))
    await _spin()
    check("edit is pending while the lock is held", not t.done())
    with open(g, encoding="utf-8") as fh:
        check("file untouched while the lock is held", fh.read() == "alpha\n")
    lock.release()
    r = await asyncio.wait_for(t, timeout=5)
    check("edit lands after release", r.ok)
    with open(g, encoding="utf-8") as fh:
        check("edited content on disk", fh.read() == "beta\n")

    print("(5) write_file race on one new path: one wins, the other fails as already-exists")
    n = os.path.join(tmp, "new.txt")
    nlock = files_mod._path_lock(n)
    await nlock.acquire()
    ta = asyncio.create_task(_as_owner(agent, lambda: write_file(n, "payload A")))
    tb = asyncio.create_task(_as_owner(agent, lambda: write_file(n, "payload B")))
    await _spin()
    check("neither write landed while the lock is held", not os.path.exists(n))
    nlock.release()
    ra, rb = await asyncio.gather(ta, tb)
    oks = [r for r in (ra, rb) if r.ok]
    fails = [r for r in (ra, rb) if not r.ok]
    check("exactly one write succeeded", len(oks) == 1)
    check("the other failed cleanly as already-exists",
          len(fails) == 1 and "already exists" in (fails[0].error or ""))
    with open(n, encoding="utf-8") as fh:
        check("file holds exactly one payload", fh.read() in ("payload A", "payload B"))


async def _run() -> None:
    saved_cfg = te.get_config
    te.get_config = lambda *a, **k: {}
    tmp = os.path.realpath(tempfile.mkdtemp(prefix="of_concurrency_"))
    try:
        await test_worker_lock()
        await test_non_worker_collision()
        await test_file_locks(tmp)
    finally:
        te.get_config = saved_cfg
        TOOL_REGISTRY.pop("concurrency_probe", None)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    asyncio.run(_run())
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
