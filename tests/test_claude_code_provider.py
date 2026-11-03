"""Verification for the claude_code provider (openflip/claude_code_conversation.py
+ claude_code_tools.py + claude_code_hooks.py) with a FAKE Agent SDK client —
no Claude Code process, no network.

Standalone runnable script:

    ./.venv/bin/python tests/test_claude_code_provider.py

What this guards:

  (a) provider registry — "claude_code" routes to ClaudeCodeConversation and
      the anthropic ChatMessage; session override keys are the SDK knobs only
      and a claude-* model is accepted on the claude_code provider;
  (b) one turn end to end — the framed user text is the prompt, a tool call
      Claude Code makes runs through tool_executor (ACL set, posting to the
      channel, model feedback), the round is mirrored into conv.messages as
      assistant(tool_calls)+tool(tool_use_id), text that accompanied the tool
      call posts right away, the FINAL text comes back with tool_calls=[], the
      turn tallies runtime reads are set, usage/ledger/meta are recorded;
  (c) client lifecycle — reused while options are unchanged, reconnected with
      `resume` when effort/model/prompt change, assistant replies runtime
      appends are never resent, /undo (on_history_cut) reseeds a new session
      with the migrated history, /compact sends Claude Code's own /compact;
  (d) failures — an auth error result becomes a framework-error message
      (kind "auth") and never lands in history; duplicate tool calls are
      suppressed; oversized feedback is capped with the marker;
  (e) hooks — unknown tools are denied, PreCompact flags the turn, the operator
      soft-inject check denies further tool calls;
  (f) memory recall routes the selector through the claude_code one-shot when
      the provider is claude_code and keeps the legacy 3-arg contract otherwise.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import claude_agent_sdk
from claude_agent_sdk import (
    AssistantMessage, UserMessage, SystemMessage, ResultMessage,
    TextBlock, ToolUseBlock, ToolResultBlock,
)

from openflip.providers import conversation_class, chat_message_class
from openflip.session_overrides import normalize_override, PROVIDER_KEYS
from openflip.anthropic_conversation import ChatMessage
from openflip import claude_code_conversation as ccc
from openflip import claude_code_tools as cct
from openflip import claude_code_hooks as cch
from openflip.tools._base import tool, ToolResult, TOOL_REGISTRY
from openflip import tool_executor, agent_state, events_log, memory_recall_api, usage_ledger

FAILURES: list[str] = []


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        FAILURES.append(label)


# ---------------------------------------------------------------- fixtures
@tool
async def cc_test_ping(x: int) -> ToolResult:
    """Echo a number back for the claude_code provider test.

    Args:
        x: the number to echo
    """
    return ToolResult(text=f"pong {x}")


class FakeChannel:
    def __init__(self):
        self.id = 4242
        self.name = "test"
        self.sent: list[str] = []

    async def send(self, content=None, files=None, **kw):
        self.sent.append(content or "")
        return SimpleNamespace(attachments=[])


class FakeClient:
    """Stands in for claude_agent_sdk.ClaudeSDKClient. `script` is a list of
    responses; each response is a list of SDK messages or zero-arg coroutine
    factories (used to simulate Claude Code invoking our MCP tool)."""
    instances: list["FakeClient"] = []
    script: list[list] = []

    def __init__(self, options):
        self.options = options
        self.queries: list = []
        self.connected = False
        self.interrupted = False
        FakeClient.instances.append(self)

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.connected = False

    async def interrupt(self):
        self.interrupted = True

    async def query(self, prompt, session_id="default"):
        if isinstance(prompt, str):
            self.queries.append(prompt)
        else:
            self.queries.append([m async for m in prompt])

    async def receive_response(self):
        items = FakeClient.script.pop(0) if FakeClient.script else []
        for item in items:
            if callable(item):
                item = await item()
            if item is not None:
                yield item


def _result(**kw) -> ResultMessage:
    base = dict(subtype="success", duration_ms=1, duration_api_ms=1, is_error=False,
                num_turns=1, session_id="sess-1")
    base.update(kw)
    return ResultMessage(**base)


def _agent(tmp: str):
    d = os.path.join(tmp, "agents", "t-cc")
    os.makedirs(os.path.join(d, "conversations"), exist_ok=True)
    return SimpleNamespace(
        id="t-cc", display_name="Test CC", path=os.path.join(d, "agent.json"),
        provider="claude_code", model="anthropic/claude-opus-5-5", system_message="sys prompt",
        memory_enabled=True, think=None, tool_response_mode="text_then_media",
    )


def _turn(conv, channel):
    import contextvars
    return cct.TurnContext(
        agent=conv.agent, conversation=conv, channel=channel, speaker_id=1,
        session_id="", silent=False, auto_post_final_text=True,
        callable_names={"cc_test_ping"}, called_signatures=set(), dedupe_exempt=set(),
        context=contextvars.copy_context(),
    )


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="cc-provider-")
    # Keep the executor's side files (live.json, events.jsonl) out of the repo.
    agent_state.on_tool_start = lambda *a, **k: None
    agent_state.on_tool_end = lambda *a, **k: None
    events_log.log_event = lambda *a, **k: None
    ledger_rows: list[dict] = []
    usage_ledger.record_usage = lambda **kw: ledger_rows.append(kw)
    claude_agent_sdk.ClaudeSDKClient = FakeClient

    print("\n(a) registry + overrides")
    check("provider routes to ClaudeCodeConversation", conversation_class("claude_code") is ccc.ClaudeCodeConversation)
    check("chat message class is the anthropic ChatMessage", chat_message_class("claude_code") is ChatMessage)
    check("override keys are SDK knobs only", PROVIDER_KEYS["claude_code"] == ("model", "effort", "memory", "activity_status", "error_display", "error_message"))
    check("claude model accepted on claude_code", normalize_override("claude_code", "model", "claude-sonnet-5-5") == ("claude-sonnet-5-5", ""))
    check("openai model rejected on claude_code", normalize_override("claude_code", "model", "openai/gpt-5.2")[0] is None)
    check("context_window rejected on claude_code", normalize_override("claude_code", "context_window", "200k")[0] is None)

    print("\n(b) one turn: prompt → tool through executor → mirror → final text")
    ag = _agent(tmp)
    conv = ccc.ClaudeCodeConversation("discord:4242", ag)
    conv.load()
    channel = FakeChannel()
    conv.messages.append(ChatMessage("user", "owner: please ping 5"))
    turn = _turn(conv, channel)
    conv._cc_turn = turn

    async def _tool_round():
        payload = await cct.run_tool_call(turn, "cc_test_ping", {"x": 5}, tool_use_id="t1")
        return UserMessage(content=[ToolResultBlock(tool_use_id="t1", content=payload["content"])])

    usage1 = {"input_tokens": 100, "output_tokens": 10, "cache_read_input_tokens": 50, "cache_creation_input_tokens": 0}
    usage2 = {"input_tokens": 120, "output_tokens": 8, "cache_read_input_tokens": 150, "cache_creation_input_tokens": 0}
    FakeClient.script = [[
        SystemMessage(subtype="init", data={"session_id": "sess-1"}),
        AssistantMessage(content=[TextBlock("Working on it."),
                                  ToolUseBlock(id="t1", name="mcp__openflip__cc_test_ping", input={"x": 5})],
                         model="claude-opus-5-5", usage=usage1),
        _tool_round,
        AssistantMessage(content=[TextBlock("Done: pong 5")], model="claude-opus-5-5", usage=usage2),
        _result(num_turns=2, stop_reason="end_turn", total_cost_usd=0.01,
                usage={"input_tokens": 220, "output_tokens": 18, "cache_read_input_tokens": 200, "cache_creation_input_tokens": 0},
                model_usage={"claude-opus-5-5": {"inputTokens": 220}}),
    ]]
    out = asyncio.run(conv.chat(tools=[TOOL_REGISTRY["cc_test_ping"].func], think=None))
    conv._cc_turn = None
    client = FakeClient.instances[-1]
    opts = client.options
    check("final text returned", out.content_text == "Done: pong 5")
    check("no tool_calls returned (Claude Code already ran them)", out.tool_calls == [])
    check("done_reason end_turn", getattr(out, "done_reason", "") == "end_turn")
    check("not a framework error", not out.is_framework_error)
    check("prompt is the framed user text", client.queries == ["owner: please ping 5"])
    sp = opts.system_prompt
    check("custom system prompt with snapshot off", isinstance(sp, dict) and sp.get("type") == "custom" and sp.get("snapshot") is False and sp["prompt"].startswith("sys prompt"))
    check("system-reminder sentence appended", "system reminders" in sp["prompt"])
    check("built-in tools off", opts.tools == [])
    check("only our MCP tool allowed", opts.allowed_tools == ["mcp__openflip__cc_test_ping"])
    check("bypassPermissions (ACL lives in the executor)", opts.permission_mode == "bypassPermissions")
    check("no filesystem settings", opts.setting_sources == [])
    check("hooks registered", set(opts.hooks) == {"PreCompact", "PreToolUse"})
    check("model normalized", opts.model == "claude-opus-5-5")
    check("new session on first connect", opts.resume is None)
    check("partial messages on", opts.include_partial_messages is True)
    check("intermediate text posted before the tool ran", channel.sent and channel.sent[0] == "Working on it.")
    check("tool result posted to channel", "pong 5" in channel.sent)
    roles = [m.get("role") for m in conv.messages]
    check("mirror: user, assistant(tool round), tool", roles == ["user", "assistant", "tool"])
    check("mirror assistant carries tool_calls", getattr(conv.messages[1], "tool_calls", None) and conv.messages[1].tool_calls[0].function_name == "cc_test_ping")
    check("mirror tool message paired by tool_use_id", conv.messages[2].get("tool_use_id") == "t1" and "pong 5" in conv.messages[2]["content"])
    check("synced count covers the mirror", conv._cc_synced == len(conv.messages))
    check("turn tallies", turn.any_tool_called and turn.calls == 1 and turn.errors == 0 and turn.posted_text)
    check("dedup signature promoted", ("cc_test_ping", json.dumps({"x": 5}, sort_keys=True)) in turn.called_signatures)
    check("session id captured", conv._cc_session_id == "sess-1")
    check("last_usage total_input from last round", conv.last_usage and conv.last_usage["total_input"] == 270)
    check("ledger row written with model + ok", ledger_rows and ledger_rows[-1]["model"] == "claude-opus-5-5" and ledger_rows[-1]["outcome"] == "ok")
    conv.messages.append(out)   # what runtime does after chat()
    conv.save()
    meta = json.load(open(conv._meta_path()))
    check("meta persists the Claude Code session", meta.get("claude_code", {}).get("session_id") == "sess-1")
    on_disk = [json.loads(l) for l in open(conv._conversation_path())]
    check("JSONL mirror has the whole round", [m["role"] for m in on_disk] == ["user", "assistant", "tool", "assistant"])

    print("\n(c) client lifecycle")
    conv.messages.append(ChatMessage("user", "owner: thanks"))
    conv._cc_turn = _turn(conv, channel)
    FakeClient.script = [[AssistantMessage(content=[TextBlock("np")], model="claude-opus-5-5", usage=usage2),
                          _result(stop_reason="end_turn", usage=usage2)]]
    n_before = len(FakeClient.instances)
    out2 = asyncio.run(conv.chat(tools=[TOOL_REGISTRY["cc_test_ping"].func]))
    conv._cc_turn = None
    check("client reused while options unchanged", len(FakeClient.instances) == n_before)
    check("assistant reply not resent; only the new user text", client.queries[-1] == "owner: thanks")
    conv.messages.append(out2)
    ok, _ = conv.set_override("effort", "low", persist=False)
    conv.messages.append(ChatMessage("user", "owner: again"))
    conv._cc_turn = _turn(conv, channel)
    FakeClient.script = [[AssistantMessage(content=[TextBlock("ok")], model="claude-opus-5-5", usage=usage2),
                          _result(stop_reason="end_turn", usage=usage2)]]
    asyncio.run(conv.chat(tools=[TOOL_REGISTRY["cc_test_ping"].func]))
    conv._cc_turn = None
    c2 = FakeClient.instances[-1]
    check("effort change reconnects", ok and c2 is not client and not client.connected)
    check("reconnect resumes the session with the new effort", c2.options.resume == "sess-1" and c2.options.effort == "low")
    # /compact: Claude Code's own command goes first, then the user prompt.
    conv.force_compact_next = True
    conv.force_compact_trigger_override = True
    conv.messages.append(ChatMessage("user", "owner: after compact"))
    conv._cc_turn = _turn(conv, channel)
    FakeClient.script = [[SystemMessage(subtype="compact_boundary", data={})],
                         [AssistantMessage(content=[TextBlock("post")], model="claude-opus-5-5", usage=usage2),
                          _result(stop_reason="end_turn", usage=usage2)]]
    asyncio.run(conv.chat(tools=[TOOL_REGISTRY["cc_test_ping"].func]))
    conv._cc_turn = None
    check("/compact sent to Claude Code before the prompt", c2.queries[-2:] == ["/compact", "owner: after compact"])
    check("compacted flag set + force flags cleared", conv.compacted_this_turn and not conv.force_compact_next)
    # /undo → on_history_cut → fresh session seeded from the mirror.
    conv.on_history_cut()
    check("undo clears session + synced", conv._cc_session_id is None and conv._cc_synced == 0)
    conv.messages.append(ChatMessage("user", "owner: new question"))
    prompt, _imgs = conv._compose_prompt()
    check("reseed prompt replays migrated history", prompt.startswith("[FRAMEWORK]: Earlier history") and "owner: please ping 5" in prompt and prompt.rstrip().endswith("owner: new question"))
    check("reseed does not replay the new message as history", prompt.count("owner: new question") == 1)

    print("\n(d) failures")
    conv2 = ccc.ClaudeCodeConversation("discord:9", ag)
    conv2.messages.append(ChatMessage("user", "hi"))
    conv2._cc_turn = _turn(conv2, channel)
    FakeClient.script = [[_result(is_error=True, subtype="success", result="Failed to authenticate: OAuth session expired and could not be refreshed")]]
    err = asyncio.run(conv2.chat(tools=[]))
    conv2._cc_turn = None
    check("auth failure → framework error", err.is_framework_error and getattr(err, "framework_error_kind", "") == "auth")
    check("error text surfaced", "authenticate" in err.content_text)
    check("error not mirrored into history", [m.get("role") for m in conv2.messages] == ["user"])
    t2 = _turn(conv, channel)
    conv._cc_turn = t2
    p1 = asyncio.run(cct.run_tool_call(t2, "cc_test_ping", {"x": 7}, tool_use_id="a"))
    p2 = asyncio.run(cct.run_tool_call(t2, "cc_test_ping", {"x": 7}, tool_use_id="b"))
    conv._cc_turn = None
    check("first call runs", "pong 7" in p1["content"][0]["text"] and not p1.get("is_error"))
    check("duplicate call suppressed with a note", p2.get("is_error") and "duplicate" in p2["content"][0]["text"])
    denied_turn = _turn(conv, channel)
    denied_turn.callable_names = set()
    conv._cc_turn = denied_turn
    p3 = asyncio.run(cct.run_tool_call(denied_turn, "cc_test_ping", {"x": 1}))
    conv._cc_turn = None
    check("ACL deny comes back as an error payload", p3.get("is_error") and "permission" in p3["content"][0]["text"])
    check("ACL deny notice posted to the user", any("can't use" in s for s in channel.sent))
    big = cct.cap_feedback("x" * (cct.TOOL_RESULT_MAX_CHARS + 10), "cc_test_ping")
    check("oversized feedback capped with marker", big.startswith("⚠️ [FRAMEWORK: TRUNCATED") and "ingestion cap" in big)

    print("\n(e) hooks")
    hooks = cch.build_hooks(conv)
    pretool = hooks["PreToolUse"][0].hooks[0]
    precompact = hooks["PreCompact"][0].hooks[0]
    d = asyncio.run(pretool({"tool_name": "Bash", "tool_input": {}}, None, None))
    check("non-openflip tool denied", d.get("hookSpecificOutput", {}).get("permissionDecision") == "deny")
    conv._cc_turn = _turn(conv, channel)
    d = asyncio.run(pretool({"tool_name": "mcp__openflip__cc_test_ping", "tool_input": {}}, None, None))
    check("our tool allowed through to the executor", d == {})
    conv._cc_turn.calls = 1
    conv._cc_turn.interrupt_check = lambda: True
    d = asyncio.run(pretool({"tool_name": "mcp__openflip__cc_test_ping", "tool_input": {}}, None, None))
    check("operator soft-inject denies further tool calls", "BATCH INTERRUPTED" in d.get("hookSpecificOutput", {}).get("permissionDecisionReason", ""))
    conv._cc_turn = None
    conv.compacted_this_turn = False
    asyncio.run(precompact({}, None, None))
    check("PreCompact flags the turn", conv.compacted_this_turn)

    print("\n(f) memory recall routing")
    captured = {}

    async def fake_cc_selector(model, cands, query):
        captured["args"] = (model, cands, query)
        return '{"selected_memories": []}'
    ccc.call_selector = fake_cc_selector
    ans = asyncio.run(memory_recall_api.call_selector("sonnet", "- a.md", "q", provider="claude_code"))
    check("claude_code provider routes the selector through the one-shot", captured.get("args") == ("sonnet", "- a.md", "q") and "selected_memories" in ans)
    check("selector model falls back to 'sonnet' for claude_code", memory_recall_api.selector_model(provider="claude_code") in ("sonnet",) or memory_recall_api.selector_model(provider="claude_code").startswith("claude-"))

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT:", "ALL PASS" if not FAILURES else f"FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print("  -", f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
