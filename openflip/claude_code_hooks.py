"""Agent SDK hook callbacks for the claude_code provider.

Two hooks only:

  PreCompact   — Claude Code is about to compact the session. Flag it on the
                 conversation so runtime posts its usual compaction notice and
                 queues the "[FRAMEWORK]: your conversation was just
                 compacted" note for the next turn (same text, same path as
                 the anthropic provider).
  PreToolUse   — operator soft-inject checkpoint. When the operator has typed
                 a new message mid-turn (runtime's interrupt_check), deny the
                 next tool call with a reason so the model wraps up and the
                 queued message is handled on the follow-up turn. Mirrors the
                 batch-interrupt in tool_executor.execute_tool_calls.

ACL, locks, dry-run and dedup are NOT hooks: they run inside the MCP handler
(claude_code_tools.run_tool_call) through execute_tool_calls, so the user
sees the same "(I can't use `x` for you here.)" notice as on every other
provider.
"""
from __future__ import annotations

from typing import Any

from .claude_code_tools import bare_tool_name, TOOL_PREFIX
from .utils import print_ts, COLOR_YELLOW, COLOR_END


def _make_precompact(conversation):
    async def precompact(input_data: dict, tool_use_id: Any, ctx: Any) -> dict:
        conversation.compacted_this_turn = True
        print_ts("claude_code: session compaction starting", agent=conversation.agent.id)
        return {}
    return precompact


def _make_pretool(conversation):
    async def pretool(input_data: dict, tool_use_id: Any, ctx: Any) -> dict:
        turn = getattr(conversation, "_cc_turn", None)
        name = str(input_data.get("tool_name") or "")
        if not name.startswith(TOOL_PREFIX):
            # Only our MCP tools exist (built-ins are off); anything else is a
            # surprise — refuse rather than let an unknown tool run.
            return {"hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Tool '{name}' is not available in this agent.",
            }}
        if turn is not None and turn.interrupt_check is not None and turn.calls > 0:
            try:
                interrupted = bool(turn.interrupt_check())
            except Exception:
                interrupted = False
            if interrupted:
                print_ts(
                    f"{COLOR_YELLOW}BATCH-INTERRUPT: operator message queued, "
                    f"denying further tool call '{bare_tool_name(name)}'{COLOR_END}",
                    agent=conversation.agent.id,
                )
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": (
                        "[BATCH INTERRUPTED] The operator sent a new message. Stop calling "
                        "tools, finish your reply now; you will see their message next turn."
                    ),
                }}
        return {}
    return pretool


def build_hooks(conversation) -> dict:
    from claude_agent_sdk import HookMatcher
    return {
        "PreCompact": [HookMatcher(hooks=[_make_precompact(conversation)])],
        "PreToolUse": [HookMatcher(hooks=[_make_pretool(conversation)])],
    }
