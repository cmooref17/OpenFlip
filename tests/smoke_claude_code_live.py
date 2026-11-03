"""LIVE smoke test for the claude_code provider — drives a real Claude Code
session through ClaudeCodeConversation (no Discord, no runtime). Needs a
`claude` login on this machine. Not collected by pytest; run by hand:

    ./.venv/bin/python tests/smoke_claude_code_live.py [model]

Checks, in order: a tool turn (tool runs through tool_executor and its output
is "posted" to a fake channel), a follow-up on the resumed session, /compact,
an image turn, and the one-shot memory-recall selector. Prints a JSON line
per step; exits non-zero on the first failed expectation.
"""
from __future__ import annotations

import asyncio
import base64
import contextvars
import json
import os
import struct
import sys
import tempfile
import time
import zlib
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip import claude_code_conversation as ccc
from openflip import claude_code_tools as cct
from openflip.anthropic_conversation import ChatMessage
from openflip.tools._base import tool, ToolResult, TOOL_REGISTRY
from openflip import agent_state, events_log

MODEL = sys.argv[1] if len(sys.argv) > 1 else "anthropic/claude-haiku-5-5"


@tool
async def smoke_ping(x: int) -> ToolResult:
    """Echo a number back (smoke test tool).

    Args:
        x: the number to echo
    """
    return ToolResult(text=f"pong {x}")


class FakeChannel:
    id = 777
    name = "smoke"

    def __init__(self):
        self.sent = []

    async def send(self, content=None, files=None, **kw):
        self.sent.append(content or "")
        return SimpleNamespace(attachments=[])


def tiny_png(size: int = 32) -> bytes:
    """Solid pure-blue square. (A 1x1 pixel is too small for a model to name
    the colour reliably.)"""
    row = b"\x00" + bytes([0, 0, 255]) * size
    raw = row * size
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def say(step, **kw):
    print(json.dumps({"step": step, **kw}, default=str), flush=True)


def expect(cond, what):
    if not cond:
        say("FAIL", what=what)
        sys.exit(1)


async def main():
    agent_state.on_tool_start = lambda *a, **k: None
    agent_state.on_tool_end = lambda *a, **k: None
    events_log.log_event = lambda *a, **k: None
    tmp = tempfile.mkdtemp(prefix="cc-smoke-")
    d = os.path.join(tmp, "agents", "smoke")
    os.makedirs(os.path.join(d, "conversations"))
    agent = SimpleNamespace(
        id="smoke", display_name="Smoke", path=os.path.join(d, "agent.json"), provider="claude_code",
        model=MODEL, memory_enabled=False, think=None, tool_response_mode="text_then_media",
        system_message=("You are SMOKEBOT, a terse OpenFlip test agent. Use the smoke_ping tool when asked to ping. "
                        "Never mention Claude Code."),
    )
    conv = ccc.ClaudeCodeConversation("internal:smoke", agent)
    conv.load()
    channel = FakeChannel()

    def turn():
        return cct.TurnContext(agent=agent, conversation=conv, channel=channel, speaker_id=1,
                               callable_names={"smoke_ping"}, called_signatures=set(), dedupe_exempt=set(),
                               context=contextvars.copy_context())
    tools = [TOOL_REGISTRY["smoke_ping"].func]

    # 1. tool turn
    conv.messages.append(ChatMessage("user", "Ping 5 with the tool, then reply with the tool's answer in one line."))
    conv._cc_turn = t = turn()
    t0 = time.monotonic()
    out = await conv.chat(tools=tools)
    conv._cc_turn = None
    say("tool_turn", seconds=round(time.monotonic() - t0, 2), text=out.content_text, error=out.is_framework_error,
        posted=channel.sent, roles=[m.get("role") for m in conv.messages], session=(conv._cc_session_id or "")[:8],
        usage=conv.last_usage)
    expect(not out.is_framework_error, f"framework error: {out.content_text}")
    expect(t.any_tool_called and "pong 5" in "".join(channel.sent), "tool did not run / post")
    expect([m.get("role") for m in conv.messages][-2:] == ["assistant", "tool"], "round not mirrored")
    conv.messages.append(out)
    conv.save()

    # 2. follow-up on the resumed session
    conv.messages.append(ChatMessage("user", "What number did I ask you to ping? Digits only."))
    conv._cc_turn = turn()
    out = await conv.chat(tools=tools)
    conv._cc_turn = None
    say("follow_up", text=out.content_text)
    expect("5" in out.content_text, "resumed session lost the earlier turn")
    conv.messages.append(out)

    # 3. /compact then a question
    conv.force_compact_next = True
    conv.messages.append(ChatMessage("user", "Say OK."))
    conv._cc_turn = turn()
    t0 = time.monotonic()
    out = await conv.chat(tools=tools)
    conv._cc_turn = None
    say("compact", seconds=round(time.monotonic() - t0, 2), compacted=conv.compacted_this_turn, text=out.content_text)
    expect(not out.is_framework_error, "compact turn failed")
    conv.messages.append(out)

    # 4. image turn
    p = os.path.join(tmp, "dot.png")
    open(p, "wb").write(tiny_png())
    conv._pending_image_attachments = [{"path": p, "content_type": "image/png", "filename": "dot.png"}]
    conv.messages.append(ChatMessage("user", "What colour is this solid square? One word."))
    conv._cc_turn = turn()
    out = await conv.chat(tools=tools)
    conv._cc_turn = None
    say("image", text=out.content_text)
    expect("blue" in out.content_text.lower(), "image not seen")
    conv.messages.append(out)

    # 5. recall selector one-shot
    t0 = time.monotonic()
    ans = await ccc.call_selector("haiku", "- servers.md (today): server notes\n- cats.md (today): cat facts",
                                  "why is the web server slow?")
    say("selector", seconds=round(time.monotonic() - t0, 2), answer=ans[:200])
    expect("servers.md" in ans, "selector did not pick the obvious file")

    await conv.aclose()
    say("DONE", meta=json.load(open(conv._meta_path())))


if __name__ == "__main__":
    asyncio.run(main())
