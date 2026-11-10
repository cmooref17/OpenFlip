"""Expose OpenFlip's @tool registry to the claude_code provider as an in-process
MCP server.

The Agent SDK runs Claude Code's own tool loop, so the model calls tools
through MCP instead of returning tool_use blocks for runtime._run_turn to
dispatch. Each MCP handler here routes the call through the SAME path the
other providers use — tool_executor.execute_tool_calls — so ACL checks, the
per-user tool lock, dry-run mode, user-facing posting (text + attachments)
and the model feedback string are identical. Nothing about a tool's behaviour
changes with the provider; only who drives the loop.

Per-turn state (channel, speaker, ACL set, dedup signatures, tallies) lives in
a TurnContext that runtime._run_turn attaches to the conversation before
chat() and reads back afterwards. Handlers run inside the SDK's own tasks,
whose contextvars were copied at connect() time, so the handler re-enters the
turn's captured contextvars.Context — tools keep seeing CURRENT_SESSION,
CURRENT_SPEAKER_ID and friends exactly as they do on the other providers.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Optional

from .tools._base import TOOL_REGISTRY
from .utils import print_ts, COLOR_YELLOW, COLOR_END

SERVER_NAME = "openflip"
TOOL_PREFIX = f"mcp__{SERVER_NAME}__"
# Same ingestion cap runtime._run_turn applies to tool results on the other
# providers: one oversized result must never become a single message bigger
# than the context window.
TOOL_RESULT_MAX_CHARS = 100_000
_REPLY_EQUIVALENT_TOOLS = {"send_message", "end_chain"}


@dataclass
class TurnContext:
    """Everything one turn's tool calls need, plus the tallies runtime reads
    back after chat() returns (mirrors the locals _run_turn keeps for the
    in-loop providers)."""
    agent: Any
    conversation: Any
    channel: Any
    speaker_id: int
    session_id: str = ""
    discord_message: Any = None
    silent: bool = False
    auto_post_final_text: bool = True
    interrupt_check: Optional[Callable[[], bool]] = None
    memory_enabled: Optional[bool] = None
    callable_names: set = field(default_factory=set)
    called_signatures: set = field(default_factory=set)
    dedupe_exempt: set = field(default_factory=set)
    context: Optional[contextvars.Context] = None
    # Tallies — read by runtime after the turn.
    calls: int = 0
    errors: int = 0
    denied: int = 0
    any_tool_called: bool = False
    reply_equivalent_fired: bool = False
    any_attachments: bool = False
    posted_text: bool = False
    # (name, args, feedback, ok) per executed call, in order — the
    # conversation mirrors these into history.
    tool_log: list = field(default_factory=list)


def full_tool_name(name: str) -> str:
    return TOOL_PREFIX + name


def bare_tool_name(full: str) -> str:
    return full[len(TOOL_PREFIX):] if full.startswith(TOOL_PREFIX) else full


def call_signature(name: str, args: dict) -> tuple[str, str]:
    try:
        return (name, json.dumps(args or {}, sort_keys=True, default=str))
    except Exception:
        return (name, repr(args or {}))


def cap_feedback(feedback: str, name: str, *, agent_id: str = "", log_tag: str = "") -> str:
    """Truncate an oversized tool result with the same marker text the
    in-loop providers use, so the model learns it saw a prefix only."""
    if len(feedback) <= TOOL_RESULT_MAX_CHARS:
        return feedback
    orig = len(feedback)
    print_ts(
        f"{COLOR_YELLOW}{log_tag}tool result from '{name}' truncated at ingestion: "
        f"{orig:,} chars > {TOOL_RESULT_MAX_CHARS:,} cap{COLOR_END}",
        agent=agent_id,
    )
    return (
        f"⚠️ [FRAMEWORK: TRUNCATED — you are seeing only the first "
        f"{TOOL_RESULT_MAX_CHARS:,} of {orig:,} characters (~{orig // 4:,} tokens). "
        f"The rest is NOT below. Do not assume you read the whole result.]\n\n"
        + feedback[:TOOL_RESULT_MAX_CHARS]
        + f"\n\n[FRAMEWORK: tool result truncated — original was {orig:,} characters "
        f"(~{orig // 4:,} tokens), over the {TOOL_RESULT_MAX_CHARS:,}-char ingestion cap. "
        f"Only the beginning is kept. Narrow the query (limit/filter/fields) or have the "
        f"tool write large output to a file instead.]"
    )


def _payload(text: str, *, is_error: bool = False) -> dict:
    out: dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        out["is_error"] = True
    return out


def _mcp_input_schema(schema: dict) -> dict:
    """The registry's parameters_schema is already a JSON-schema object; the
    SDK accepts that form directly. Copied so the registry entry is never
    mutated by the SDK's validation layer."""
    return json.loads(json.dumps(schema or {"type": "object", "properties": {}}))


async def run_tool_call(turn: TurnContext, name: str, args: dict, tool_use_id: str = "") -> dict:
    """Execute ONE tool call through tool_executor.execute_tool_calls and
    return the MCP result payload the model sees."""
    from .anthropic_conversation import AnthropicToolCall
    from .tool_executor import execute_tool_calls, build_model_feedback

    agent = turn.agent
    args = dict(args or {})
    sig = call_signature(name, args)
    if name not in turn.dedupe_exempt and sig in turn.called_signatures:
        print_ts(
            f"{COLOR_YELLOW}Duplicate tool call suppressed: {name}({args}){COLOR_END}",
            agent=agent.id,
        )
        return _payload(
            "Note: this tool call exactly repeats one that already succeeded this turn, "
            "so it was skipped; its result is above. If the task is not finished, continue "
            "with the next step (change the arguments if you really need to repeat it).",
            is_error=True,
        )

    fake_message = SimpleNamespace(tool_calls=[AnthropicToolCall(name, args, tool_use_id=tool_use_id)])
    results = await execute_tool_calls(
        agent=agent,
        conversation=turn.conversation,
        ai_message=fake_message,
        callable_tool_names=turn.callable_names,
        channel=turn.channel,
        speaker_id=turn.speaker_id,
        session_id=turn.session_id,
        discord_message=turn.discord_message,
        silent=turn.silent,
        interrupt_check=None,
        memory_enabled=turn.memory_enabled,
    )
    if not results:
        return _payload(f"Tool '{name}' was not executed.", is_error=True)
    tname, res = results[0]
    turn.calls += 1
    turn.any_tool_called = True
    if not res.ok:
        turn.errors += 1
    if getattr(res, "denied", False):
        turn.denied += 1
    if res.ok and name not in turn.dedupe_exempt:
        turn.called_signatures.add(sig)
    if name in _REPLY_EQUIVALENT_TOOLS:
        turn.reply_equivalent_fired = True
    if getattr(res, "attachments", None):
        turn.any_attachments = True
    feedback = cap_feedback(build_model_feedback(tname, res), tname, agent_id=agent.id)
    turn.tool_log.append((tname, args, feedback, bool(res.ok)))
    return _payload(feedback, is_error=not res.ok)


def _make_handler(conversation, name: str):
    async def handler(args: dict) -> dict:
        turn: TurnContext | None = getattr(conversation, "_cc_turn", None)
        if turn is None:
            return _payload(f"Tool '{name}' was called outside an active OpenFlip turn.", is_error=True)
        if turn.context is not None:
            # Re-enter the turn's contextvars (CURRENT_SESSION etc.) — the SDK
            # task we're running in was created at connect() time.
            task = asyncio.get_running_loop().create_task(
                run_tool_call(turn, name, args), context=turn.context,
            )
            return await task
        return await run_tool_call(turn, name, args)
    return handler


def build_mcp_server(conversation, tool_names: list[str]):
    """One in-process MCP server exposing `tool_names` (registry names) for a
    conversation. Built per client connect; cheap."""
    from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

    sdk_tools = []
    for name in tool_names:
        t = TOOL_REGISTRY.get(name)
        if t is None:
            continue
        sdk_tools.append(SdkMcpTool(
            name=name,
            description=t.description or name.replace("_", " "),
            input_schema=_mcp_input_schema(t.parameters_schema),
            handler=_make_handler(conversation, name),
        ))
    return create_sdk_mcp_server(SERVER_NAME, version="1.0.0", tools=sdk_tools)
