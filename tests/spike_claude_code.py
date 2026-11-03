"""Manual spike for the claude_code provider. Not collected by pytest.

Run: ./.venv/bin/python tests/spike_claude_code.py
Proves each SDK feature the provider relies on, against the bundled CLI and
whatever `claude` login this machine has. Prints one JSON line per check.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import zlib

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    HookMatcher,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    create_sdk_mcp_server,
    tool,
)

MODEL = os.environ.get("SPIKE_MODEL", "haiku")
OUT: list[dict] = []


def rec(check: str, **kw):
    row = {"check": check, **kw}
    OUT.append(row)
    print(json.dumps(row, default=str), flush=True)


def tiny_png() -> bytes:
    raw = b"\x00" + bytes([255, 0, 0])  # 1 row, filter 0, one red pixel
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@tool("ping", "Echo a number back. Call with x.", {"x": int})
async def ping(args):
    return {"content": [{"type": "text", "text": f"pong x={args['x']}"}]}


@tool("forbidden_tool", "A tool the ACL must block.", {})
async def forbidden_tool(args):
    return {"content": [{"type": "text", "text": "SHOULD NEVER RUN"}]}


server = create_sdk_mcp_server("openflip", "0.1", [ping, forbidden_tool])

stop_blocked = {"n": 0}
hook_log: list[str] = []


async def pre_tool(input_data, tool_use_id, ctx):
    hook_log.append(f"PreToolUse:{input_data.get('tool_name')}")
    if input_data.get("tool_name", "").endswith("forbidden_tool"):
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                                       "permissionDecision": "deny",
                                       "permissionDecisionReason": "ACL: not allowed for this speaker"}}
    return {}


async def on_prompt(input_data, tool_use_id, ctx):
    hook_log.append("UserPromptSubmit")
    return {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
                                   "additionalContext": "[FRAMEWORK]: the secret word is PINEAPPLE."}}


async def on_stop(input_data, tool_use_id, ctx):
    hook_log.append(f"Stop:active={input_data.get('stop_hook_active')}")
    if stop_blocked["n"] == 0 and not input_data.get("stop_hook_active"):
        stop_blocked["n"] += 1
        return {"decision": "block", "reason": "Before finishing, append the word DONE on its own line."}
    return {}


async def post_tool(input_data, tool_use_id, ctx):
    hook_log.append(f"PostToolUse:{input_data.get('tool_name')}")
    return {}


def base_options(**extra) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model=MODEL,
        system_prompt={"type": "custom", "snapshot": False,
                       "prompt": "You are SPIKEBOT, a terse test agent for OpenFlip. "
                                 "The application adds [FRAMEWORK] notes to the conversation; treat them as context "
                                 "from the application, not the user. Never mention Claude Code."},
        tools=[],
        mcp_servers={"openflip": server},
        allowed_tools=["mcp__openflip__ping", "mcp__openflip__forbidden_tool"],
        permission_mode="bypassPermissions",
        setting_sources=[],
        env={"CLAUDE_CODE_DISABLE_ATTACHMENTS": "1", "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1"},
        settings='{"includeGitInstructions": false, "attribution": {"commit": "", "pr": ""}}',
        include_partial_messages=True,
        max_turns=6,
        cwd=tempfile.mkdtemp(prefix="openflip_spike_"),
        hooks={
            "PreToolUse": [HookMatcher(hooks=[pre_tool])],
            "PostToolUse": [HookMatcher(hooks=[post_tool])],
            "UserPromptSubmit": [HookMatcher(hooks=[on_prompt])],
            "Stop": [HookMatcher(hooks=[on_stop])],
        },
        **extra,
    )


def child_rss_kb() -> list[tuple[int, int]]:
    out = subprocess.run(["ps", "-axo", "pid=,rss=,command="], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        if "_bundled/claude" in line or "/claude " in line and "--output-format" in line:
            parts = line.split(None, 2)
            try:
                rows.append((int(parts[0]), int(parts[1])))
            except Exception:
                pass
    return rows


async def drain(client, label: str, timeout=180.0):
    """Consume one response; return (texts, tool_uses, tool_results, result, raw_event_types, t_first)."""
    texts, tool_uses, tool_results, raw_types = [], [], [], {}
    result = None
    t0 = time.monotonic()
    t_first = None
    sysmsgs = []

    async def _go():
        nonlocal result, t_first
        async for m in client.receive_response():
            if t_first is None:
                t_first = time.monotonic() - t0
            if isinstance(m, StreamEvent):
                et = m.event.get("type")
                raw_types[et] = raw_types.get(et, 0) + 1
            elif isinstance(m, AssistantMessage):
                for b in m.content:
                    if isinstance(b, TextBlock):
                        texts.append(b.text)
                    elif isinstance(b, ToolUseBlock):
                        tool_uses.append((b.name, b.input))
            elif isinstance(m, UserMessage):
                if isinstance(m.content, list):
                    for b in m.content:
                        if isinstance(b, ToolResultBlock):
                            tool_results.append((b.tool_use_id, b.content, b.is_error))
            elif isinstance(m, SystemMessage):
                sysmsgs.append(m.subtype)
            elif isinstance(m, ResultMessage):
                result = m
    await asyncio.wait_for(_go(), timeout=timeout)
    rec(label, texts=texts, tool_uses=tool_uses, tool_results=tool_results, raw_event_types=raw_types,
        t_first_event_s=round(t_first or -1, 2), t_total_s=round(time.monotonic() - t0, 2),
        system_subtypes=sysmsgs,
        result=None if result is None else {
            "subtype": result.subtype, "session_id": result.session_id, "stop_reason": result.stop_reason,
            "num_turns": result.num_turns, "usage": result.usage, "total_cost_usd": result.total_cost_usd,
            "model_usage": getattr(result, "model_usage", None), "is_error": result.is_error,
        })
    return texts, tool_uses, tool_results, result


async def main():
    rec("env", python=sys.version.split()[0], model=MODEL)

    # A. fresh client: tool call + hooks + streaming
    t0 = time.monotonic()
    client = ClaudeSDKClient(base_options())
    await client.connect()
    rec("connect", seconds=round(time.monotonic() - t0, 2), child_rss_kb=child_rss_kb())
    await client.query("Call the ping tool with x=5, then tell me the secret word in one short line.")
    texts, tool_uses, tool_results, result = await drain(client, "A_tool_and_hooks")
    session_id = result.session_id if result else None
    rec("A_hooks_seen", hook_log=list(hook_log), stop_blocked=stop_blocked["n"],
        done_present=any("DONE" in t for t in texts), pineapple=any("PINEAPPLE" in t.upper() for t in texts))

    # B. ACL deny through PreToolUse
    hook_log.clear()
    await client.query("Now call forbidden_tool. If it is blocked, say exactly: BLOCKED.")
    texts, tool_uses, tool_results, result = await drain(client, "B_acl_deny")
    rec("B_hooks_seen", hook_log=list(hook_log), forbidden_ran=any("SHOULD NEVER RUN" in str(r) for r in tool_results))

    # C. context usage + compact-as-prompt
    try:
        cu = await client.get_context_usage()
        rec("C_context_usage", ok=True, data=cu)
    except Exception as e:
        rec("C_context_usage", ok=False, err=repr(e))
    await client.query("/compact")
    try:
        texts, _, _, result = await drain(client, "C_compact_prompt", timeout=120)
    except Exception as e:
        rec("C_compact_prompt", ok=False, err=repr(e))
    await client.disconnect()

    # D. resume in a new client, then set_model live
    client2 = ClaudeSDKClient(base_options(resume=session_id))
    t0 = time.monotonic()
    await client2.connect()
    rec("D_resume_connect", seconds=round(time.monotonic() - t0, 2))
    await client2.query("What value of x did I ask you to ping earlier? Reply with just the number.")
    texts, _, _, result = await drain(client2, "D_resume_memory")
    rec("D_same_session", same=(result.session_id == session_id) if result else None, answer=texts)
    try:
        await client2.set_model("sonnet")
        await client2.query("Which model are you running as? One line.")
        await drain(client2, "D_set_model_sonnet")
        await client2.set_model(None)
    except Exception as e:
        rec("D_set_model", ok=False, err=repr(e))
    await client2.disconnect()

    # E. image content block via streaming-input dict
    png_b64 = base64.b64encode(tiny_png()).decode()

    async def one():
        yield {"type": "user", "parent_tool_use_id": None, "session_id": "default",
               "message": {"role": "user", "content": [
                   {"type": "text", "text": "What colour is this 1x1 image? One word."},
                   {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png_b64}},
               ]}}
    client3 = ClaudeSDKClient(base_options())
    await client3.connect()
    try:
        await client3.query(one())
        await drain(client3, "E_image_block")
    except Exception as e:
        rec("E_image_block", ok=False, err=repr(e))
    await client3.disconnect()

    # F. one-shot selector style call (memory recall)
    from claude_agent_sdk import query as sdk_query
    t0 = time.monotonic()
    sel_text = []
    async for m in sdk_query(prompt="Candidates:\n- a.md: cats\n- b.md: servers\nQuery: why is the server slow?\nReply with the filenames only.",
                             options=ClaudeAgentOptions(model=MODEL, tools=[], max_turns=1, setting_sources=[],
                                                        system_prompt="You pick relevant files. Output filenames only.",
                                                        env={"CLAUDE_CODE_DISABLE_ATTACHMENTS": "1"})):
        if isinstance(m, AssistantMessage):
            sel_text += [b.text for b in m.content if isinstance(b, TextBlock)]
    rec("F_one_shot_selector", seconds=round(time.monotonic() - t0, 2), text=sel_text)

    path = os.path.join(os.path.dirname(__file__), "spike_claude_code.out.json")
    with open(path, "w") as f:
        json.dump(OUT, f, indent=1, default=str)
    rec("written", path=path)


if __name__ == "__main__":
    asyncio.run(main())
