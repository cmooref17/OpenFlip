"""Plan mode (Claude Code's permission-modes.md "Analyze before you edit").

Design: docs/goal_plan_reread.md. While a conversation is in plan mode the agent
researches and writes a plan but changes nothing: tools that write or change
state are blocked in tool_executor, and run_command is limited to read-only
commands. The agent presents its plan with exit_plan_mode(plan); the operator
approves (`/plan approve`) or sends it back (`/plan no <feedback>`). Approval
turns plan mode off and starts a turn that carries the plan out. The plan file
(conversations/<cid>.plan.md) is re-injected after compaction.

State lives in conversations/<cid>.planmode.json so it survives a restart.
"""
from __future__ import annotations

import json
import os
import re
import time

from .utils import save_json

# Tools that change files, system state, other people's views, or spend money on
# generation. Blocked while planning. Anything not listed (reads, searches,
# memory reads, todo_write, use_skill, tool_search) is allowed.
WRITE_TOOLS = {
    "write_file", "edit_file", "delete_file", "restore_snapshot",
    "write_file_sandbox", "run_command_sandbox",
    "send_file", "discord_post", "discord_manage", "delete_message", "force_reply",
    "restart_gateway", "restart_flask_app", "add_cron_job", "cancel_cron_job",
    "claude_code", "spawn_subagents", "blender_build", "screen", "steam",
    "generate_image", "edit_image", "animate_image", "generate_video", "upscale_image",
    "generate_tts", "make_3d_model", "download_media", "remove_audio_track",
    "extract_audio_track", "update_core_memory", "delete_memory", "save_memory",
    "flip_send", "wow_ah", "inject_context",
}

# run_command while planning: block commands that obviously write or change
# state. Not a sandbox (CC prompts or classifies here; we have no prompt UI),
# just a guard against the common writes; the agent is also told read-only only.
_WRITE_CMD = re.compile(
    r"(^|[;&|`(]\s*|\$\(\s*|\bsudo\s+)"
    r"(rm|rmdir|mv|cp|dd|tee|touch|mkdir|chmod|chown|ln|truncate|shred|install|"
    r"git(\s+-C\s+\S+)?\s+(commit|push|reset|checkout|stash|rebase|merge|add|rm|mv|clean|restore|switch|apply|cherry-pick|tag)|"
    r"pip\d*\s+(install|uninstall)|npm\s+(install|i|uninstall|publish)|pacman|yay|apt(-get)?|"
    r"systemctl(\s+--?[\w-]+)*\s+(start|stop|restart|reload|enable|disable|kill|mask|daemon-reload)|systemd-run|kill|pkill|killall|"
    r"reboot|shutdown|curl\s+.*-X\s*(POST|PUT|DELETE|PATCH)|sed\s+(-[a-zA-Z]*i|--in-place))\b"
)
_REDIRECT = re.compile(r"(?<![0-9&])>{1,2}(?!\s*&?\s*/dev/null)(?!&)")

APPROVE_WORDS = {"approve", "approved", "yes", "y", "go", "ok", "accept", "lgtm"}
REJECT_WORDS = {"no", "n", "reject", "keep", "revise"}
OFF_WORDS = {"off", "exit", "cancel", "stop", "clear"}

REMINDER = (
    "Plan mode is ON (the operator turned it on with /plan). Research and plan only: read "
    "files, search, and run read-only commands, but change nothing. No edits, no writes, no "
    "restarts, no messages to others, no generation; those tools are blocked. When the plan "
    "is ready, call exit_plan_mode with the full plan (what changes, which files, how you'll "
    "test and verify it). The operator approves it or sends it back with feedback."
)


def _paths(agent_dir: str, conversation_id: str) -> tuple[str, str]:
    from ._conversation_io import fs_encode
    base = os.path.join(agent_dir, "conversations", fs_encode(conversation_id))
    return base + ".planmode.json", base + ".plan.md"


def state(agent_dir: str, conversation_id: str) -> dict:
    if not agent_dir or not conversation_id:
        return {}
    try:
        with open(_paths(agent_dir, conversation_id)[0], "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _save(agent_dir: str, conversation_id: str, d: dict) -> None:
    p = _paths(agent_dir, conversation_id)[0]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    save_json(p, d)


def is_on(agent_dir: str, conversation_id: str) -> bool:
    return bool(state(agent_dir, conversation_id).get("on"))


def turn_on(agent_dir: str, conversation_id: str) -> None:
    d = state(agent_dir, conversation_id)
    d.update({"on": True, "since": time.time(), "awaiting_approval": False})
    _save(agent_dir, conversation_id, d)


def turn_off(agent_dir: str, conversation_id: str) -> None:
    d = state(agent_dir, conversation_id)
    d.update({"on": False, "awaiting_approval": False})
    _save(agent_dir, conversation_id, d)


def clear(agent_dir: str, conversation_id: str) -> None:
    """/reset: drop plan mode and the plan file."""
    for p in _paths(agent_dir, conversation_id):
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def save_plan(agent_dir: str, conversation_id: str, plan: str) -> str:
    p = _paths(agent_dir, conversation_id)[1]
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(plan.strip() + "\n")
    d = state(agent_dir, conversation_id)
    d.update({"awaiting_approval": True, "plan_at": time.time()})
    _save(agent_dir, conversation_id, d)
    return p


def load_plan(agent_dir: str, conversation_id: str) -> str:
    try:
        with open(_paths(agent_dir, conversation_id)[1], "r", encoding="utf-8") as f:
            return f.read().strip()
    except (FileNotFoundError, OSError):
        return ""


def blocked_reason(tool_name: str, args: dict) -> str:
    """Why this call is blocked in plan mode ('' = allowed)."""
    if tool_name in WRITE_TOOLS:
        return (f"Plan mode is on: '{tool_name}' changes things, so it's blocked until the "
                f"operator approves a plan. Finish the plan and call exit_plan_mode.")
    if tool_name == "run_command":
        cmd = str((args or {}).get("command") or "")
        if _WRITE_CMD.search(cmd) or _REDIRECT.search(cmd):
            return ("Plan mode is on: that command looks like it writes or changes state "
                    "(rm/mv/cp/git commit/installs/restarts/kill/sed -i/output redirection), "
                    "so it's blocked. Only read-only commands while planning.")
        if (args or {}).get("run_in_background"):
            return "Plan mode is on: background commands are blocked while planning."
    return ""


def parse_command(arg: str) -> tuple[str, str]:
    """('toggle'|'approve'|'reject'|'off'|'plan', text) for `/plan <arg>`."""
    a = (arg or "").strip()
    if not a:
        return "toggle", ""
    head, _, rest = a.partition(" ")
    h = head.lower().rstrip(".,!")
    if h in APPROVE_WORDS and not rest.strip():
        return "approve", ""
    if h in REJECT_WORDS:
        return "reject", rest.strip()
    if h in OFF_WORDS and not rest.strip():
        return "off", ""
    return "plan", a


def reminder(agent_dir: str, conversation_id: str) -> str:
    """Per-turn note while plan mode is on ('' when off)."""
    d = state(agent_dir, conversation_id)
    if not d.get("on"):
        return ""
    if d.get("awaiting_approval"):
        return (REMINDER + " You already presented a plan and it is waiting on the operator; "
                "if they gave feedback, revise it and call exit_plan_mode again.")
    return REMINDER


def compaction_block(agent_dir: str, conversation_id: str) -> str:
    """Re-injected after compaction (CC: the plan comes back from disk)."""
    plan = load_plan(agent_dir, conversation_id)
    d = state(agent_dir, conversation_id)
    if not plan and not d.get("on"):
        return ""
    parts = []
    if d.get("on"):
        parts.append(REMINDER)
    if plan:
        label = ("Your plan, waiting for the operator's approval" if d.get("awaiting_approval")
                 else "The plan you're working from")
        parts.append(f"{label} (re-read from disk):\n<plan>\n{plan}\n</plan>")
    return "\n\n".join(parts)


def approve_prompt(plan: str) -> str:
    return ("[FRAMEWORK]: The operator approved your plan (/plan approve). Plan mode is off. "
            "Carry the plan out now, start to finish, and test it.\n<plan>\n" + plan + "\n</plan>")


def reject_prompt(feedback: str) -> str:
    return ("[FRAMEWORK]: The operator sent your plan back (still in plan mode). Revise it and "
            "call exit_plan_mode again." + (f"\nTheir feedback: {feedback}" if feedback else ""))


def plan_prompt(request: str) -> str:
    return ("[FRAMEWORK]: The operator turned on plan mode with this request. Research it, "
            "change nothing, then present your plan with exit_plan_mode.\nRequest: " + request)
