"""todo_write: the session task checklist (Claude Code's TodoWrite).

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_todo_write.py

Pins:
  (1) a list is saved per conversation and reloads from disk
  (2) the per-turn reminder lists open items, and is empty when none are open
  (3) completing every item clears the list (file removed)
  (4) bad input (no content, unknown status, not a list) is rejected
  (5) more than one in_progress item gets a note back
  (6) todo_write is a framework tool: always callable and never deferred
  (7) separate conversations keep separate lists
"""
from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip.tools import todo as td  # noqa: E402
from openflip.tool_executor import CURRENT_AGENT, CURRENT_SESSION  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILURES.append(label)


class FakeAgent:
    def __init__(self, path):
        self.id = "test-agent"
        self.path = path


class FakeSession:
    def __init__(self, conv):
        self.conversation_id = conv


async def main() -> None:
    tmp = tempfile.mkdtemp(prefix="todo_test_")
    try:
        agent_json = os.path.join(tmp, "agent.json")
        CURRENT_AGENT.set(FakeAgent(agent_json))
        CURRENT_SESSION.set(FakeSession("discord:111"))
        w = td.todo_write

        r = await w([{"content": "Write the test", "status": "in_progress"},
                     {"content": "Run the suite", "status": "pending"}])
        saved = td.load_todos(tmp, "discord:111")
        check("(1) list saved and reloads", len(saved) == 2 and saved[0]["status"] == "in_progress"
              and "0/2 done" in r.model_feedback)

        note = td.todo_reminder(tmp, "discord:111")
        check("(2) reminder lists open items", "[~] Write the test" in note and "[ ] Run the suite" in note)

        CURRENT_SESSION.set(FakeSession("discord:222"))
        await w([{"content": "Other job", "status": "in_progress"}])
        check("(7) conversations keep separate lists",
              len(td.load_todos(tmp, "discord:111")) == 2 and len(td.load_todos(tmp, "discord:222")) == 1)
        CURRENT_SESSION.set(FakeSession("discord:111"))

        r = await w([{"content": "Write the test", "status": "in_progress"},
                     {"content": "Run the suite", "status": "in_progress"}])
        check("(5) two in_progress items get a note", "keep exactly one in_progress" in r.model_feedback)

        r = await w([{"content": "Write the test", "status": "completed"},
                     {"content": "Run the suite", "status": "completed"}])
        check("(3) all completed clears the list",
              "cleared" in r.model_feedback and not os.path.exists(td.todos_path(tmp, "discord:111"))
              and td.todo_reminder(tmp, "discord:111") == "")

        r1 = await w([{"content": "", "status": "pending"}])
        r2 = await w([{"content": "x", "status": "done"}])
        r3 = await w("not a list")
        check("(4) bad input rejected", bool(r1.error) and bool(r2.error) and bool(r3.error))

        from openflip.pipeline import FRAMEWORK_TOOL_NAMES
        from openflip.tool_deferral import always_load_names
        check("(6) todo_write always callable + never deferred",
              "todo_write" in FRAMEWORK_TOOL_NAMES and "todo_write" in always_load_names())
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n{'ALL PASS' if not FAILURES else str(len(FAILURES)) + ' FAILED'}")
    sys.exit(1 if FAILURES else 0)
