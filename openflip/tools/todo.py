"""todo_write: the session task checklist (Claude Code's TodoWrite).

The agent keeps a short list of the steps in a multi-step task and checks each
off as it goes, so long jobs don't silently drop steps. One list per
conversation, saved next to the conversation file
(conversations/<conversation_id>.todos.json) so it survives restarts.
The runtime reminds the agent of unfinished items at the start of each turn
(see todo_reminder) and /status shows them.
"""
from __future__ import annotations

import json
import os

from ._base import tool, ToolResult, TOOL_REGISTRY

_VALID = ("pending", "in_progress", "completed")
_MAX_ITEMS = 50


def todos_path(agent_dir: str, conversation_id: str) -> str:
    from .._conversation_io import fs_encode
    return os.path.join(agent_dir, "conversations", f"{fs_encode(conversation_id)}.todos.json")


def load_todos(agent_dir: str, conversation_id: str) -> list[dict]:
    if not agent_dir or not conversation_id:
        return []
    try:
        with open(todos_path(agent_dir, conversation_id), "r", encoding="utf-8") as f:
            data = json.load(f)
        return [t for t in data if isinstance(t, dict)] if isinstance(data, list) else []
    except (FileNotFoundError, ValueError, OSError):
        return []


def save_todos(agent_dir: str, conversation_id: str, todos: list[dict]) -> None:
    from ..utils import save_json
    path = todos_path(agent_dir, conversation_id)
    if not todos or all(t.get("status") == "completed" for t in todos):
        # A finished list is cleared (Claude Code empties it once everything is done).
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json(path, todos)


def format_todos(todos: list[dict]) -> str:
    marks = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
    return "\n".join(f"{marks.get(t.get('status'), '[ ]')} {t.get('content', '')}" for t in todos)


def todo_reminder(agent_dir: str, conversation_id: str) -> str:
    """The note the runtime adds to a turn while the list has open items ('' if none)."""
    todos = load_todos(agent_dir, conversation_id)
    if not todos or all(t.get("status") == "completed" for t in todos):
        return ""
    return ("Your task list for this conversation still has open items. Keep it current with "
            "todo_write (mark each item completed as soon as it's done; remove items that no longer "
            "apply), and don't stop while items are open unless you're blocked on the operator:\n"
            + format_todos(todos))


def _current_ids() -> tuple[str, str]:
    from ..tool_executor import CURRENT_AGENT, CURRENT_SESSION
    agent = CURRENT_AGENT.get(None)
    if agent is None:
        return "", ""
    sess = CURRENT_SESSION.get(None)
    conv = str(getattr(sess, "conversation_id", "") or "") if sess is not None else ""
    if not conv:
        from ..tool_executor import CURRENT_CHANNEL_ID
        try:
            ch = int(CURRENT_CHANNEL_ID.get(0) or 0)
        except (TypeError, ValueError):
            ch = 0
        conv = f"discord:{ch}" if ch else ""
    return os.path.dirname(agent.path), conv


@tool
async def todo_write(todos: list[dict]) -> ToolResult:
    """Update the task list for this conversation. Use it proactively for any task with 3 or more distinct steps, when the operator gives several things to do, or right after new instructions: write the steps, mark one in_progress BEFORE starting it, and mark it completed IMMEDIATELY when it's fully done (never batch completions). Exactly one item is in_progress while you work. Only mark an item completed when it's fully done: not while tests fail, the work is partial, or errors are unresolved; if blocked, keep it in_progress and add an item for the blocker. Remove items that no longer apply. Skip the list for a single trivial task or pure conversation. Send the WHOLE list every call; it replaces the previous one.

    Args:
        todos: The full list. Each item is {"content": "imperative step, e.g. Run the tests", "status": "pending" | "in_progress" | "completed"}.
    """
    agent_dir, conv = _current_ids()
    if not agent_dir or not conv:
        return ToolResult.fail("todo_write needs a conversation.")
    if not isinstance(todos, list):
        return ToolResult.fail("todos must be a list of {content, status} items.")
    clean: list[dict] = []
    for i, t in enumerate(todos[:_MAX_ITEMS]):
        if not isinstance(t, dict):
            return ToolResult.fail(f"item {i} is not an object.")
        content = str(t.get("content") or "").strip()
        status = str(t.get("status") or "pending").strip()
        if not content:
            return ToolResult.fail(f"item {i} has no content.")
        if status not in _VALID:
            return ToolResult.fail(f"item {i} status must be one of {', '.join(_VALID)}.")
        clean.append({"content": content[:300], "status": status})
    active = sum(1 for t in clean if t["status"] == "in_progress")
    save_todos(agent_dir, conv, clean)
    done = sum(1 for t in clean if t["status"] == "completed")
    note = ""
    if clean and done < len(clean) and active != 1:
        note = f"\nNote: {active} items are in_progress; keep exactly one in_progress while you work."
    if clean and done == len(clean):
        return ToolResult(model_feedback=f"All {done} items completed; the list is cleared.")
    return ToolResult(model_feedback=f"Task list updated ({done}/{len(clean)} done):\n{format_todos(clean)}{note}")


if "todo_write" in TOOL_REGISTRY:
    TOOL_REGISTRY["todo_write"].silent_to_discord = True
