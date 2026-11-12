"""run_command: foreground timeout cap + run_in_background delivery.

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_run_command_background.py

Pins:
  (1) foreground: output + exit code returned; timeout is capped at 600s
  (2) foreground: a command over its timeout is killed and reported
  (3) background: returns a job id immediately (doesn't wait for the command)
  (4) background: when the command ends, its output is delivered to the agent
      that started it as a silent synthetic turn in the SAME conversation
  (5) background: a background job's own timeout kills it and still delivers
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip.tools import run_command as rc  # noqa: E402
from openflip import registry  # noqa: E402
from openflip.tool_executor import CURRENT_AGENT, CURRENT_SESSION, CURRENT_CHANNEL_ID  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool) -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILURES.append(label)


class FakeAgent:
    id = "test-agent"


class FakeRunner:
    def __init__(self):
        self.turns: list[tuple] = []

    async def run_synthetic_turn(self, target, prompt, **kw):
        self.turns.append((target, prompt, kw))


async def main() -> None:
    import openflip.acl as acl
    acl.current_caller_is_admin = lambda: True  # the test runs as the operator

    fn = rc.run_command.__wrapped__ if hasattr(rc.run_command, "__wrapped__") else rc.run_command

    r = await fn("echo hello; exit 3")
    check("(1) foreground returns output + exit code", "Exit 3" in r.model_feedback and "hello" in r.model_feedback)

    t0 = time.monotonic()
    r = await fn("sleep 5", timeout=1)
    check("(2) foreground timeout kills the command", bool(r.error) and "timed out after 1s" in r.error
          and time.monotonic() - t0 < 4)

    runner = FakeRunner()
    registry.RUNNERS["test-agent"] = runner
    CURRENT_AGENT.set(FakeAgent())
    CURRENT_SESSION.set("SESSION-X")
    CURRENT_CHANNEL_ID.set(42)

    t0 = time.monotonic()
    r = await fn("sleep 1; echo done-bg", run_in_background=True)
    elapsed = time.monotonic() - t0
    check("(3) background returns immediately with a job id",
          elapsed < 0.5 and "bg-" in (r.model_feedback or ""))
    await asyncio.sleep(2.5)
    ok = (len(runner.turns) == 1 and runner.turns[0][0] == "SESSION-X"
          and "done-bg" in runner.turns[0][1] and "Exit 0" in runner.turns[0][1]
          and runner.turns[0][2].get("silent") is True)
    check("(4) finished background output delivered to the same conversation", ok)

    r = await fn("sleep 10", timeout=1, run_in_background=True)
    await asyncio.sleep(2.5)
    check("(5) background timeout kills it and still delivers",
          len(runner.turns) == 2 and "Timed out after 1s" in runner.turns[1][1])
    check("(1b) timeout is capped at 600s", rc._MAX_TIMEOUT == 600)


if __name__ == "__main__":
    asyncio.run(main())
    print(f"\n{'ALL PASS' if not FAILURES else str(len(FAILURES)) + ' FAILED'}")
    sys.exit(1 if FAILURES else 0)
