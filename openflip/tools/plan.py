"""exit_plan_mode: present a plan for the operator's approval (Claude Code's ExitPlanMode).

Design: docs/goal_plan_reread.md. Framework tool: callable by every agent
(pipeline.FRAMEWORK_TOOL_NAMES), never deferred. Only does anything while the
conversation is in plan mode (/plan).
"""
from __future__ import annotations

import os

from ._base import tool, ToolResult


def _current() -> tuple[str, str]:
    from ..tool_executor import CURRENT_AGENT, CURRENT_SESSION
    agent = CURRENT_AGENT.get(None)
    sess = CURRENT_SESSION.get(None)
    if agent is None:
        return "", ""
    return os.path.dirname(agent.path), str(getattr(sess, "conversation_id", "") or "")


@tool
async def exit_plan_mode(plan: str) -> ToolResult:
    """Present your plan to the operator for approval. Only use this in plan mode (when a reminder says plan mode is on). Write the full plan: what changes, which files, and how you'll test and verify it. The operator then approves it (you'll get a message to carry it out) or sends it back with feedback. After calling this, end your turn and wait.

    Args:
        plan: The complete plan, in markdown.
    """
    from .. import plan_mode as _pm
    agent_dir, cid = _current()
    if not agent_dir or not cid:
        return ToolResult.fail("exit_plan_mode needs a conversation.")
    if not _pm.is_on(agent_dir, cid):
        return ToolResult.fail("Plan mode isn't on, so there's nothing to approve. Just do the work.")
    plan = (plan or "").strip()
    if not plan:
        return ToolResult.fail("The plan is empty. Write the full plan.")
    _pm.save_plan(agent_dir, cid, plan)
    shown = ("**Plan ready for approval**\n\n" + plan +
             "\n\nReply `/plan approve` to carry it out, or `/plan no <what to change>`.")
    return ToolResult(
        text=shown,
        model_feedback=("The plan was shown to the operator and saved. Stop here: end your turn "
                        "and wait for /plan approve or their feedback. Don't start the work."),
    )
