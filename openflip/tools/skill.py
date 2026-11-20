"""use_skill: load a skill's full instructions (Claude Code's Skill tool).

Design: docs/skills.md. The system prompt only lists skill names and
descriptions; this tool returns the full rendered SKILL.md as the tool result,
so it enters the conversation once and stays there (Claude Code's lifecycle).
Framework tool: every agent may call it (pipeline.FRAMEWORK_TOOL_NAMES), it is
never deferred (tool_deferral.DEFAULT_ALWAYS_LOAD).
"""
from __future__ import annotations

import os

from ._base import tool, ToolResult, TOOL_REGISTRY


def _current() -> tuple[object, str, str]:
    """(agent, agent_dir, conversation_id) for the running tool call."""
    from ..tool_executor import CURRENT_AGENT, CURRENT_SESSION, CURRENT_CHANNEL_ID
    agent = CURRENT_AGENT.get(None)
    if agent is None:
        return None, "", ""
    sess = CURRENT_SESSION.get(None)
    conv = str(getattr(sess, "conversation_id", "") or "") if sess is not None else ""
    if not conv:
        try:
            ch = int(CURRENT_CHANNEL_ID.get(0) or 0)
        except (TypeError, ValueError):
            ch = 0
        conv = f"discord:{ch}" if ch else ""
    return agent, os.path.dirname(agent.path), conv


@tool
async def use_skill(skill: str, args: str = "") -> ToolResult:
    """Load a skill's full instructions. When the current task matches a skill in the Skills list of your system prompt, call this BEFORE doing the task and follow what it returns; the skill is the tested way to do it. Only the names and descriptions are in your prompt until you load one. A loaded skill stays in the conversation, so don't load the same one twice. If the operator's message starts with /<skill-name>, that skill is already attached to the message; don't load it again.

    Args:
        skill: The skill name exactly as listed, e.g. "send-file-to-friend" (no leading slash).
        args: Optional arguments for the skill.
    """
    from .. import skills as _sk
    agent, agent_dir, conv = _current()
    if agent is None or not agent_dir:
        return ToolResult.fail("use_skill needs an agent context.")
    name = str(skill or "").strip().lstrip("/")
    if not _sk.skills_enabled(agent):
        return ToolResult.fail("Skills are turned off for this agent.")
    found, _warns = _sk.load_skills(agent_dir, agent)
    s = found.get(name)
    if s is None:
        avail = ", ".join(sorted(n for n, x in found.items() if x.model_invocable)) or "none"
        return ToolResult.fail(f"No skill named {name!r}. Available: {avail}.")
    if not s.model_invocable:
        return ToolResult.fail(
            f"Skill {name!r} can only be run by the operator (it sets disable-model-invocation). "
            f"Ask them to type /{name} if they want it.")
    rendered = _sk.render(s, args, session_id=conv)
    if _sk.note_loaded(conv, name, rendered):
        return ToolResult(model_feedback=_sk.ALREADY_LOADED.format(name=name)
                          + (f" Arguments: {args}" if args else ""))
    return ToolResult(model_feedback=rendered)


if "use_skill" in TOOL_REGISTRY:
    TOOL_REGISTRY["use_skill"].silent_to_discord = True
