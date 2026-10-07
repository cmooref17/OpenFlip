"""Subagent foundation (result_future + make_subagent_session).

Standalone runnable script (no pytest in this venv):

    .lvenv/bin/python tests/test_subagent_foundation.py

What this guards:

  (a) `result_future` defaults to None on run_synthetic_turn / _run_turn and
      the existing call shape still enqueues the same dict (plus the new key
      carrying None) — no behavior change for pre-existing callers;
  (b) make_subagent_session: is_owner False, distinct non-owner speaker_ids
      for two task uuids, the `internal:subagent-<uuid>` conversation id,
      grants == allowlist == the caller's list (copied, not aliased);
  (c) a result_future can never leave an awaiter hanging: the settle helper
      resolves once and only once, the `_on_turn_done` backstop settles a
      pending future on a cancelled supervisor (CancelledError) and on a
      normally-finished one (""), and run_synthetic_turn's own early-return
      paths settle "" before the turn is ever queued.
"""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip import runtime as _rt
from openflip.runtime import AgentRunner, _settle_result_future
from openflip.session import Session, make_subagent_session
from openflip import config_global as _cfg

FAILURES: list[str] = []


def check(label: str, cond: bool) -> None:
    print(("  ok    " if cond else "  FAIL  ") + label)
    if not cond:
        FAILURES.append(label)


class _FakeQueue:
    def __init__(self) -> None:
        self.items: list[dict] = []
        self.task_done_calls = 0

    async def put(self, item: dict) -> None:
        self.items.append(item)

    def task_done(self) -> None:
        self.task_done_calls += 1


def _fake_runner(channel) -> SimpleNamespace:
    """Minimal stand-in for an AgentRunner `self` as run_synthetic_turn sees it."""
    async def _resolve(channel_id, *, speaker_id=0, speaker_handle=""):
        return channel

    return SimpleNamespace(
        agent=SimpleNamespace(id="testagent"),
        _inbound_queue=_FakeQueue(),
        _resolve_synthetic_channel=_resolve,
        _ensure_worker_started=lambda: None,
        _active_turns={},
    )


def test_a_default_none() -> None:
    print("(a) result_future default None / existing call shape")
    sig = inspect.signature(AgentRunner.run_synthetic_turn)
    check("run_synthetic_turn has result_future kwarg defaulting to None",
          sig.parameters["result_future"].default is None
          and sig.parameters["result_future"].kind is inspect.Parameter.KEYWORD_ONLY)
    sig2 = inspect.signature(AgentRunner._run_turn)
    check("_run_turn has result_future kwarg defaulting to None",
          sig2.parameters["result_future"].default is None
          and sig2.parameters["result_future"].kind is inspect.Parameter.KEYWORD_ONLY)

    chan = SimpleNamespace(id=4242, name="fake")
    fake = _fake_runner(chan)
    _orig = _cfg.get_owner_id
    _cfg.get_owner_id = lambda transport="discord": 111
    try:
        asyncio.run(AgentRunner.run_synthetic_turn(fake, 4242, "hello", silent=True))
    finally:
        _cfg.get_owner_id = _orig
    check("existing call shape still enqueues exactly one turn", len(fake._inbound_queue.items) == 1)
    item = fake._inbound_queue.items[0] if fake._inbound_queue.items else {}
    check("queued dict carries result_future=None by default",
          "result_future" in item and item["result_future"] is None)
    check("pre-existing queue fields unchanged",
          item.get("channel") is chan and item.get("user_text") == "hello"
          and item.get("silent") is True and item.get("owner") is False
          and item.get("speaker_id") == 111 and item.get("auto_post_final_text") is False)
    # Every key _run_turn accepts (minus the ones the dispatcher adds) is
    # present in the queued dict, so **kwargs dispatch can't TypeError.
    run_turn_params = set(inspect.signature(AgentRunner._run_turn).parameters) - {"self"}
    extra = set(item) - run_turn_params
    check("queued dict keys are all _run_turn parameters", not extra)


def test_b_make_subagent_session() -> None:
    print("(b) make_subagent_session")
    grants = ["read_file", "web_search"]
    s1 = make_subagent_session("rhea", "aaaa-1111", grants)
    s2 = make_subagent_session("rhea", "bbbb-2222", grants)
    check("returns a Session", isinstance(s1, Session))
    check("is_owner is False", s1.is_owner is False and s2.is_owner is False)
    check("transport is internal", s1.transport == "internal")
    check("conversation_id is internal:subagent-<uuid>",
          s1.conversation_id == "internal:subagent-aaaa-1111"
          and s2.conversation_id == "internal:subagent-bbbb-2222")
    check("distinct speaker_ids for two task uuids", s1.speaker_id != s2.speaker_id)
    check("speaker_id is a positive int (non-zero, hashable lock key)",
          isinstance(s1.speaker_id, int) and s1.speaker_id > 0)
    check("same uuid → same speaker_id within a process",
          make_subagent_session("rhea", "aaaa-1111", grants).speaker_id == s1.speaker_id)
    check("grants == allowlist == caller list",
          s1.tool_grants == grants and s1.tool_allowlist == grants)
    check("grants/allowlist are copies, not the caller's list object",
          s1.tool_grants is not grants and s1.tool_allowlist is not grants
          and s1.tool_grants is not s1.tool_allowlist)
    s3 = make_subagent_session("rhea", "cccc-3333", [])
    check("empty grants → empty allowlist (deny-all, not None)",
          s3.tool_grants == [] and s3.tool_allowlist == [] and s3.tool_allowlist is not None)
    check("no handle, no roles", s1.handle == "" and s1.speaker_role_ids == [])


def test_c_never_hangs() -> None:
    print("(c) result_future can never leave an awaiter hanging")

    async def _body() -> None:
        loop = asyncio.get_running_loop()

        # -- settle helper semantics --
        f = loop.create_future()
        _settle_result_future(f, text="hi")
        check("settle sets the text", f.done() and f.result() == "hi")
        _settle_result_future(f, text="later")
        check("second settle is a no-op (first wins)", f.result() == "hi")
        f2 = loop.create_future()
        _settle_result_future(f2, cancelled=True)
        try:
            f2.result()
            check("cancel-settle raises CancelledError to the awaiter", False)
        except asyncio.CancelledError:
            check("cancel-settle raises CancelledError to the awaiter", True)
        f3 = loop.create_future()
        _settle_result_future(f3)
        check("settle with no text resolves to empty string", f3.result() == "")
        _settle_result_future(None, text="x")
        check("settle(None) is a harmless no-op", True)
        f4 = loop.create_future()
        f4.cancel()
        _settle_result_future(f4, text="x")
        check("settling an already-cancelled future doesn't raise", f4.cancelled())

        # -- _on_turn_done backstop: cancelled supervisor --
        fake = SimpleNamespace(_active_turns={}, _inbound_queue=_FakeQueue())
        pending = loop.create_future()

        async def _never():
            await asyncio.sleep(3600)

        t = asyncio.create_task(_never())
        t._channel_id = "internal:subagent-x"
        t._prev_task = None
        t._started_run = False
        t._result_future = pending
        fake._active_turns["internal:subagent-x"] = t
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        AgentRunner._on_turn_done(fake, t)
        check("backstop: task_done balanced", fake._inbound_queue.task_done_calls == 1)
        check("backstop: slot popped", "internal:subagent-x" not in fake._active_turns)
        try:
            await asyncio.wait_for(pending, timeout=1.0)
            check("backstop: cancelled supervisor → awaiter gets CancelledError", False)
        except asyncio.CancelledError:
            check("backstop: cancelled supervisor → awaiter gets CancelledError", True)
        except asyncio.TimeoutError:
            check("backstop: cancelled supervisor → awaiter gets CancelledError (HUNG)", False)

        # -- _on_turn_done backstop: supervisor that returned without settling --
        pending2 = loop.create_future()

        async def _ok():
            return None

        t2 = asyncio.create_task(_ok())
        t2._channel_id = "internal:subagent-y"
        t2._prev_task = None
        t2._started_run = True
        t2._result_future = pending2
        await t2
        AgentRunner._on_turn_done(fake, t2)
        try:
            r = await asyncio.wait_for(pending2, timeout=1.0)
            check("backstop: finished supervisor with unsettled future → \"\"", r == "")
        except asyncio.TimeoutError:
            check("backstop: finished supervisor with unsettled future → \"\" (HUNG)", False)

        # -- _on_turn_done leaves an already-settled future alone --
        settled = loop.create_future()
        settled.set_result("real answer")
        t3 = asyncio.create_task(_ok())
        t3._channel_id = 0
        t3._prev_task = None
        t3._started_run = True
        t3._result_future = settled
        await t3
        AgentRunner._on_turn_done(fake, t3)
        check("backstop doesn't overwrite a real result", settled.result() == "real answer")

        # -- _on_turn_done on a task with no future attached (legacy shape) --
        t4 = asyncio.create_task(_ok())
        t4._channel_id = 0
        t4._prev_task = None
        t4._started_run = True
        await t4
        AgentRunner._on_turn_done(fake, t4)
        check("backstop tolerates tasks without _result_future", True)

        # -- run_synthetic_turn early returns settle "" before queueing --
        _orig = _cfg.get_owner_id
        _cfg.get_owner_id = lambda transport="discord": 0
        fake_r = _fake_runner(SimpleNamespace(id=1, name="c"))
        f5 = loop.create_future()
        try:
            await AgentRunner.run_synthetic_turn(fake_r, 1, "x", result_future=f5)
        finally:
            _cfg.get_owner_id = _orig
        check("no owner_id → future settled \"\", nothing queued",
              f5.done() and f5.result() == "" and not fake_r._inbound_queue.items)

        _cfg.get_owner_id = lambda transport="discord": 111
        fake_r2 = _fake_runner(None)  # channel not found
        f6 = loop.create_future()
        try:
            await AgentRunner.run_synthetic_turn(fake_r2, 1, "x", result_future=f6)
        finally:
            _cfg.get_owner_id = _orig
        check("channel not found → future settled \"\", nothing queued",
              f6.done() and f6.result() == "" and not fake_r2._inbound_queue.items)

        # -- happy path: the future rides the queue dict and the task attr --
        _cfg.get_owner_id = lambda transport="discord": 111
        fake_r3 = _fake_runner(SimpleNamespace(id=7, name="c"))
        f7 = loop.create_future()
        try:
            await AgentRunner.run_synthetic_turn(fake_r3, 7, "x", result_future=f7)
        finally:
            _cfg.get_owner_id = _orig
        item = fake_r3._inbound_queue.items[0]
        check("future carried in the queue dict", item.get("result_future") is f7 and not f7.done())
        check("future-bearing turn is enqueued silent-safe (owner False)", item.get("owner") is False)

    asyncio.run(_body())


def main() -> int:
    test_a_default_none()
    test_b_make_subagent_session()
    test_c_never_hangs()
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
