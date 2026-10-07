"""spawn_subagents tool.

Standalone runnable script (no pytest in this venv):

    .lvenv/bin/python tests/test_subagent_tool.py

The AgentRunner is mocked (no model, no Discord): a FakeRunner records the
sessions/prompts it was asked to run, settles each worker's result_future
after a scripted delay (or never, to exercise the timeout path), and records
`_hard_interrupt` / `reset_conversation` calls. Settings are injected into
tool_settings' in-memory store (never persisted).

What this guards:
  (1) owner gate — a non-owner CURRENT_SESSION is refused;
  (2) grants — denied tools are stripped even when requested and even when
      the owner put them in allowed_tools; requested ∩ ceiling otherwise;
  (3) > max_parallel is rejected outright;
  (4) two tasks run concurrently (their fake turns overlap) and both
      summaries come back in the report;
  (5) timeout → error entry + _hard_interrupt on the worker's conv key;
  (6) cleanup (reset_conversation on the worker's key + conversation id)
      runs on success, timeout, exception and tool cancellation;
  (7) the inflight counter returns to 0 after success, timeout, exception
      and cancellation, and excess tasks fail fast when the cap is hit;
  (8) worker session/override shape: is_owner False, allowlist == grants,
      model/effort/memory overrides applied non-persistently, invalid
      models fail that task cleanly;
  (9) the report is capped.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip import tool_settings as ts
from openflip import config_global as _cfg
from openflip import registry
from openflip.runtime import _settle_result_future
from openflip.session import Session
from openflip.tool_executor import CURRENT_SESSION, CURRENT_AGENT
from openflip.tools import subagent as sa
from openflip.tools.subagent import spawn_subagents, compute_grants, format_results, worker_conv_key
from openflip.tool_settings import SUBAGENT_DENIED_TOOLS

FAILURES: list[str] = []


def check(label: str, cond: bool) -> None:
    print(("  ok    " if cond else "  FAIL  ") + label)
    if not cond:
        FAILURES.append(label)


# ------------------------------------------------------------------ fakes

class FakeConv:
    def __init__(self, conversation_id: str) -> None:
        self.conversation_id = conversation_id
        self.overrides: dict = {}
        self.persist_calls = 0

    def set_override(self, key, raw, *, persist=True):
        if persist:
            self.persist_calls += 1
        if key == "model" and str(raw).startswith("reject-"):
            return False, "rejected by set_override"
        self.overrides[key] = raw
        return True, f"{key}={raw}"


class FakeRunner:
    """Just enough AgentRunner surface for the tool: get_conversation,
    run_synthetic_turn(result_future=...), _hard_interrupt, reset_conversation."""

    def __init__(self, agent, behaviors: dict | None = None, *, raise_on_get_conv: bool = False) -> None:
        self.agent = agent
        self.behaviors = behaviors or {}
        self.raise_on_get_conv = raise_on_get_conv
        self.conversations: dict = {}
        self.turns: list[dict] = []
        self.interrupts: list = []
        self.resets: list[tuple] = []
        self.spans: dict[str, tuple[float, float]] = {}
        self._active: dict = {}

    def get_conversation(self, channel_id, conversation_id, native_key=""):
        if self.raise_on_get_conv:
            raise RuntimeError("boom in get_conversation")
        c = self.conversations.get(channel_id)
        if c is None:
            c = FakeConv(conversation_id)
            self.conversations[channel_id] = c
        return c

    async def run_synthetic_turn(self, channel_id, prompt_text, *, result_future=None, **kw):
        session = channel_id
        label = session.display_name
        self.turns.append({"session": session, "prompt": prompt_text, "kw": kw, "future": result_future})
        key = worker_conv_key(session)
        # Behaviors are keyed by the task prompt's first line after the preamble.
        task_line = prompt_text.split("## Task\n", 1)[-1].splitlines()[0]
        beh = self.behaviors.get(task_line, {})
        delay = float(beh.get("delay", 0.05))
        text = beh.get("text", f"summary for {task_line}")
        hang = bool(beh.get("hang", False))

        async def _turn():
            t0 = time.monotonic()
            try:
                if hang:
                    await asyncio.sleep(3600)
                else:
                    await asyncio.sleep(delay)
                _settle_result_future(result_future, text=text)
            except asyncio.CancelledError:
                _settle_result_future(result_future, cancelled=True)
                raise
            finally:
                self.spans[task_line] = (t0, time.monotonic())

        self._active[key] = asyncio.create_task(_turn())

    def _hard_interrupt(self, channel_id):
        self.interrupts.append(channel_id)
        t = self._active.get(channel_id)
        if t is not None and not t.done():
            t.cancel()
        return 0

    def reset_conversation(self, conv_key, fallback_conv_id=""):
        self.resets.append((conv_key, fallback_conv_id))
        self._hard_interrupt(conv_key)
        self.conversations.pop(conv_key, None)
        return True


def _agent(provider="anthropic"):
    return SimpleNamespace(id="orch", provider=provider, model="claude-opus-5-5", display_name="Orch")


def _session(is_owner: bool) -> Session:
    return Session(
        transport="discord", transport_id="123", conversation_id="discord:123",
        speaker_id=999, speaker_role_ids=[], is_owner=is_owner, is_dm=True, display_name="flip",
    )


_SAVED_VALUES: dict | None = None
_SAVED_GET_CONFIG = _cfg.get_config


def _settings(**over):
    ts._ensure_loaded()
    base = {
        "worker_model": "claude-sonnet-5-5", "worker_effort": "medium",
        "allowed_tools": "web_search,fetch_url,read_file,list_files,search_memory,read_memory",
        "max_parallel": 4, "max_inflight_per_agent": 6, "timeout_s": 300,
    }
    base.update(over)
    ts._VALUES["spawn_subagents"] = base


def _fake_config(*_a, **_k):
    return {"models": {
        "claude-sonnet-5-5": {"provider": "anthropic", "context_window": 1_000_000},
        "claude-opus-5-5": {"provider": "anthropic", "context_window": 1_000_000},
        "qwen3.5:cloud": {"provider": "ollama"},
    }}


async def _call(runner, tasks, *, is_owner=True):
    registry.RUNNERS[runner.agent.id] = runner
    tok_a = CURRENT_AGENT.set(runner.agent)
    tok_s = CURRENT_SESSION.set(_session(is_owner))
    try:
        return await spawn_subagents(tasks)
    finally:
        CURRENT_AGENT.reset(tok_a)
        CURRENT_SESSION.reset(tok_s)
        registry.RUNNERS.pop(runner.agent.id, None)


# ------------------------------------------------------------------ tests

async def test_owner_gate():
    print("(1) owner gate")
    _settings()
    r = FakeRunner(_agent())
    res = await _call(r, [{"prompt": "p"}], is_owner=False)
    check("non-owner session is refused", res.error is not None and "owner" in res.error)
    check("nothing launched", not r.turns and sa.inflight_count("orch") == 0)
    # A worker session itself (is_owner False, internal transport) is refused too.
    registry.RUNNERS["orch"] = r
    from openflip.session import make_subagent_session
    tok_a = CURRENT_AGENT.set(r.agent)
    tok_s = CURRENT_SESSION.set(make_subagent_session("orch", "u-1", ["read_file"]))
    try:
        res2 = await spawn_subagents([{"prompt": "p"}])
    finally:
        CURRENT_AGENT.reset(tok_a); CURRENT_SESSION.reset(tok_s); registry.RUNNERS.pop("orch", None)
    check("a worker session cannot spawn workers (no recursion)", res2.error is not None)
    res3 = await _call(FakeRunner(_agent()), "not json at all")
    check("malformed tasks rejected", res3.error is not None)


async def test_grants():
    print("(2) grants / deny list")
    # Owner (mis)configured the ceiling to include denied tools — they must still be stripped.
    _settings(allowed_tools="read_file,web_search,run_command,save_memory,talk_to_agent")
    r = FakeRunner(_agent())
    res = await _call(r, [{"prompt": "g", "tools": ["run_command", "talk_to_agent", "read_file", "generate_image", "spawn_subagents"]}])
    check("call succeeded", res.error is None)
    sess = r.turns[0]["session"]
    check("denied tools stripped even if requested AND in allowed_tools",
          sess.tool_allowlist == ["read_file"] and sess.tool_grants == ["read_file"])
    check("worker session is non-owner internal", sess.is_owner is False and sess.transport == "internal")
    r2 = FakeRunner(_agent())
    await _call(r2, [{"prompt": "g"}])
    check("no tools requested → ceiling minus denied",
          r2.turns[0]["session"].tool_allowlist == ["read_file", "web_search"])
    check("compute_grants: requested outside ceiling dropped",
          compute_grants(["fetch_url", "read_file"], ["read_file"]) == ["read_file"])
    check("every hard-denied name is excluded by compute_grants",
          compute_grants(None, sorted(SUBAGENT_DENIED_TOOLS) + ["read_memory"]) == ["read_memory"])
    for must in ("spawn_subagents", "restart_gateway", "claude_code", "force_reply", "resolve_approval",
                 "talk_to_agent", "send_message", "send_file", "inject_context", "add_cron_job",
                 "cancel_cron_job", "list_cron_jobs", "run_command", "write_file", "edit_file",
                 "delete_file", "restore_snapshot", "save_memory", "update_core_memory",
                 "delete_memory", "dream", "reindex_memory", "discord_post", "discord_manage",
                 "flip_send", "flip_voice"):
        if must not in SUBAGENT_DENIED_TOOLS:
            check(f"deny list contains {must}", False)
    check("deny list covers the required set", True)
    check("/toolset validator rejects denied tools in allowed_tools",
          ts.coerce_and_validate("spawn_subagents", "allowed_tools", "read_file,run_command")[1] is not None)
    check("/toolset validator accepts the default list",
          ts.coerce_and_validate("spawn_subagents", "allowed_tools", ts.SUBAGENT_DEFAULT_TOOLS)[1] is None)


async def test_max_parallel():
    print("(3) max_parallel")
    _settings(max_parallel=2)
    r = FakeRunner(_agent())
    res = await _call(r, [{"prompt": "a"}, {"prompt": "b"}, {"prompt": "c"}])
    check("> max_parallel rejected", res.error is not None and "max_parallel" in res.error)
    check("nothing launched, inflight 0", not r.turns and sa.inflight_count("orch") == 0)


async def test_concurrent_success():
    print("(4) two tasks run concurrently, both summaries returned")
    _settings()
    r = FakeRunner(_agent(), {"alpha": {"delay": 0.3, "text": "ALPHA RESULT"},
                              "beta": {"delay": 0.3, "text": "BETA RESULT"}})
    t0 = time.monotonic()
    res = await _call(r, [{"prompt": "alpha", "label": "A"}, {"prompt": "beta", "label": "B"}])
    elapsed = time.monotonic() - t0
    check("tool ok", res.error is None)
    a, b = r.spans.get("alpha"), r.spans.get("beta")
    check("both turns ran", a is not None and b is not None)
    check("turns overlapped (parallel, not serialized)",
          a is not None and b is not None and b[0] < a[1] and a[0] < b[1])
    check("wall time ~one delay, not two", elapsed < 0.55)
    text = res.text or ""
    check("report has both ok entries", "### A — ok\nALPHA RESULT" in text and "### B — ok\nBETA RESULT" in text)
    check("model_feedback counts workers", (res.model_feedback or "").startswith("2/2 workers ok"))
    check("prompt framed as worker task + user prompt",
          all(sa.WORKER_PREAMBLE in t["prompt"] and "## Task\n" in t["prompt"] for t in r.turns))
    check("turns are silent with a result_future", all(t["kw"].get("silent") is True and t["future"] is not None for t in r.turns))
    check("distinct worker sessions / conv keys",
          len({worker_conv_key(t["session"]) for t in r.turns}) == 2
          and all(t["session"].conversation_id.startswith("internal:subagent-") for t in r.turns))
    check("cleanup ran for both (key + conversation id)",
          sorted(r.resets) == sorted((worker_conv_key(t["session"]), t["session"].conversation_id) for t in r.turns)
          and not r.conversations)
    check("inflight back to 0 after success", sa.inflight_count("orch") == 0)
    check("tool is silent_to_discord", sa.TOOL_REGISTRY["spawn_subagents"].silent_to_discord is True)


async def test_overrides_and_model_validation():
    print("(8) worker overrides + model validation")
    _settings(worker_model="claude-sonnet-5-5", worker_effort="high")
    r = FakeRunner(_agent())
    convs: list[FakeConv] = []
    _orig = r.get_conversation

    def _gc(*a, **k):
        c = _orig(*a, **k); convs.append(c); return c
    r.get_conversation = _gc
    res = await _call(r, [{"prompt": "o1"}, {"prompt": "o2", "model": "claude-opus-5-5"}])
    check("ok", res.error is None and len(convs) == 2)
    check("model/effort/memory overrides applied, none persisted",
          all(c.overrides.get("effort") == "high" and c.overrides.get("memory") is False and c.persist_calls == 0 for c in convs)
          and convs[0].overrides.get("model") == "claude-sonnet-5-5" and convs[1].overrides.get("model") == "claude-opus-5-5")
    r2 = FakeRunner(_agent())
    res2 = await _call(r2, [{"prompt": "bad", "model": "claude-nope-1", "label": "bad"}, {"prompt": "good", "label": "good"}])
    check("unknown model fails only that task", "### bad — error\ninvalid worker model" in (res2.text or "")
          and "### good — ok" in (res2.text or ""))
    check("failed task never launched; good one did", len(r2.turns) == 1)
    r3 = FakeRunner(_agent(provider="ollama"))
    res3 = await _call(r3, [{"prompt": "x", "label": "x"}])
    check("provider mismatch (ollama agent, claude worker) fails cleanly",
          "### x — error" in (res3.text or "") and "ollama" in (res3.text or "") and not r3.turns)
    r4 = FakeRunner(_agent())
    res4 = await _call(r4, [{"prompt": "y", "label": "y", "model": "reject-claude-sonnet-5-5"}])
    check("set_override rejection fails cleanly", "### y — error" in (res4.text or "") and not r4.turns)
    check("inflight back to 0", sa.inflight_count("orch") == 0)
    check("cleanup ran even for never-launched tasks", len(r2.resets) == 2 and len(r4.resets) == 1)


async def test_timeout():
    print("(5) timeout → error + interrupt + cleanup")
    _settings(timeout_s=0.3)
    r = FakeRunner(_agent(), {"slow": {"hang": True}, "fast": {"delay": 0.01, "text": "FAST"}})
    t0 = time.monotonic()
    res = await _call(r, [{"prompt": "slow", "label": "S"}, {"prompt": "fast", "label": "F"}])
    check("returned promptly", time.monotonic() - t0 < 1.5)
    check("timeout yields an error entry", "### S — error\ntimed out after 0s" in (res.text or "") or "### S — error\ntimed out" in (res.text or ""))
    check("fast task still ok", "### F — ok\nFAST" in (res.text or ""))
    slow_key = next(worker_conv_key(t["session"]) for t in r.turns if "slow" in t["prompt"])
    check("interrupt fired on the slow worker's conv key", slow_key in r.interrupts)
    check("slow worker task actually cancelled", r._active[slow_key].cancelled() or r._active[slow_key].done())
    check("cleanup ran for both", len(r.resets) == 2 and not r.conversations)
    check("inflight back to 0 after timeout", sa.inflight_count("orch") == 0)


async def test_exception():
    print("(6) exception inside a task → error entry, cleanup, inflight 0")
    _settings()
    r = FakeRunner(_agent(), raise_on_get_conv=True)
    res = await _call(r, [{"prompt": "e", "label": "E"}])
    check("tool itself does not raise", res.error is None)
    check("error entry carries the exception", "### E — error\nRuntimeError: boom" in (res.text or ""))
    check("cleanup ran", len(r.resets) == 1)
    check("inflight back to 0 after exception", sa.inflight_count("orch") == 0)


async def test_inflight_cap():
    print("(7) max_inflight_per_agent across concurrent calls")
    _settings(max_inflight_per_agent=3)
    r = FakeRunner(_agent(), {"c1": {"delay": 0.3}, "c2": {"delay": 0.3}, "c3": {"delay": 0.05}, "c4": {"delay": 0.05}})
    first = asyncio.create_task(_call(r, [{"prompt": "c1", "label": "c1"}, {"prompt": "c2", "label": "c2"}]))
    await asyncio.sleep(0.05)
    check("two workers inflight during first call", sa.inflight_count("orch") == 2)
    r2 = FakeRunner(_agent(), r.behaviors)
    res2 = await _call(r2, [{"prompt": "c3", "label": "c3"}, {"prompt": "c4", "label": "c4"}])
    check("second call: one slot free → one launched, the excess fails fast",
          "### c3 — ok" in (res2.text or "") and "### c4 — error\nnot started: max_inflight_per_agent" in (res2.text or "")
          and len(r2.turns) == 1)
    res1 = await first
    check("first call completed normally", "### c1 — ok" in (res1.text or "") and "### c2 — ok" in (res1.text or ""))
    check("inflight back to 0 after both calls", sa.inflight_count("orch") == 0)


async def test_tool_cancelled():
    print("(6b) tool cancelled mid-flight → interrupt + cleanup + inflight 0")
    _settings()
    r = FakeRunner(_agent(), {"h1": {"hang": True}, "h2": {"hang": True}})
    task = asyncio.create_task(_call(r, [{"prompt": "h1"}, {"prompt": "h2"}]))
    await asyncio.sleep(0.1)
    check("two workers running", len(r.turns) == 2 and sa.inflight_count("orch") == 2)
    task.cancel()
    try:
        await task
        check("CancelledError propagates out of the tool", False)
    except asyncio.CancelledError:
        check("CancelledError propagates out of the tool", True)
    keys = {worker_conv_key(t["session"]) for t in r.turns}
    check("both workers interrupted", keys <= set(r.interrupts))
    check("cleanup ran for both", len(r.resets) == 2 and not r.conversations)
    check("inflight back to 0 after cancellation", sa.inflight_count("orch") == 0)


def test_format_cap():
    print("(9) report cap")
    big = [(f"t{i}", True, "x" * 10_000) for i in range(4)]
    out = format_results(big)
    check("total report ≤ 12k", len(out) <= 12_000)
    check("every entry present + truncated marker", all(f"### t{i} — ok" in out for i in range(4)) and "…[truncated]" in out)
    small = format_results([("a", True, "fine"), ("b", False, "nope")])
    check("small report verbatim", small == "### a — ok\nfine\n\n### b — error\nnope")


async def _main_async():
    await test_owner_gate()
    await test_grants()
    await test_max_parallel()
    await test_concurrent_success()
    await test_overrides_and_model_validation()
    await test_timeout()
    await test_exception()
    await test_inflight_cap()
    await test_tool_cancelled()
    test_format_cap()


def main() -> int:
    global _SAVED_VALUES
    ts._ensure_loaded()
    _SAVED_VALUES = dict(ts._VALUES.get("spawn_subagents") or {})
    _cfg.get_config = _fake_config
    try:
        asyncio.run(_main_async())
    finally:
        _cfg.get_config = _SAVED_GET_CONFIG
        if _SAVED_VALUES:
            ts._VALUES["spawn_subagents"] = _SAVED_VALUES
        else:
            ts._VALUES.pop("spawn_subagents", None)
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
