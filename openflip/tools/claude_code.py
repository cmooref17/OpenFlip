"""Delegate a coding task to Claude Code as a subprocess.

The agent provides a task description; Claude Code runs in non-interactive
mode (--print) with full filesystem access to the openflip directory, edits
whatever it needs to, and returns its final output as the tool result.

Use this when:
- The task is structural framework work (runtime.py, conversation.py,
  pipeline.py, tools/) where the calling agent shouldn't trust itself to
  ship changes safely.
- The task needs careful tracing across multiple files before editing.
- The calling agent has already proven slip-prone on this kind of change.

Don't use it for:
- Trivial edits the calling agent can do directly (one-line changes,
  config tweaks, memory updates).
- Anything outside the openflip repo — Claude Code is scoped here.

Security: runs with --dangerously-skip-permissions so it can edit/delete
without asking. The cwd is fixed to the openflip repo root. Owner-gated via
the agent.json allowed_tools ACL — only the configured owner can invoke
this tool through an agent.
"""
from __future__ import annotations

import asyncio
import os
import shutil

from ._base import tool, ToolResult
from ..utils import print_ts


# Resolve the openflip repo root from this file's location so the repo can
# live anywhere on disk.
_OPENFLIP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Try to locate the `claude` binary on PATH; fall back to a common install
# location if not found. Users can override via the CLAUDE_BIN env var.
_CLAUDE_BIN = os.environ.get("CLAUDE_BIN") or shutil.which("claude") or "claude"
_DEFAULT_TIMEOUT = 600  # 10 min — claude-code can chew through long sessions


@tool
async def claude_code(task: str, timeout: int = _DEFAULT_TIMEOUT) -> ToolResult:
    """Delegate a task to Claude Code (non-interactive subprocess).

    Claude Code runs from the openflip repo root with full file access
    (--dangerously-skip-permissions). It will read whatever files it needs,
    make edits, and return its final response.

    Args:
        task: A clear description of what to do. Be specific about which
            files to touch, what behavior change you want, and any
            constraints (preflight, no-restart, etc.). Treat this like
            you're handing off to another engineer.
        timeout: Max seconds to wait for Claude Code to finish (default 600,
            i.e. 10 minutes). Capped at 1800 (30 minutes).
    """
    from ..acl import current_caller_is_owner
    if not current_caller_is_owner():
        return ToolResult.fail("claude_code is owner-only.")

    if not task or not task.strip():
        return ToolResult.fail("task is empty")

    if not os.path.isfile(_CLAUDE_BIN):
        return ToolResult.fail(f"claude binary not found at {_CLAUDE_BIN}")

    timeout = max(30, min(int(timeout or _DEFAULT_TIMEOUT), 1800))

    # CLAUDECODE env var causes a "nested session" error if claude is
    # spawned from inside another claude session; strip it from the
    # subprocess env so we can run cleanly.
    env = dict(os.environ)
    env.pop("CLAUDECODE", None)

    print_ts(
        f"claude_code: dispatching task ({len(task)} chars, timeout={timeout}s)",
        agent="claude_code",
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            _CLAUDE_BIN,
            "--print",
            "--dangerously-skip-permissions",
            task,
            cwd=_OPENFLIP_DIR,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except Exception as e:
        return ToolResult.fail(f"failed to spawn claude: {e}")

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
            # Bound the reap — a killed child stuck in uninterruptible state
            # must not hang the per-(agent,user,tool) lock forever.
            await asyncio.wait_for(proc.wait(), timeout=5)
        except Exception:
            pass
        return ToolResult.fail(
            f"claude_code timed out after {timeout}s. Task may have been "
            "partially completed; check files."
        )
    except Exception as e:
        return ToolResult.fail(f"claude_code communicate failed: {e}")

    stdout = (stdout_bytes or b"").decode("utf-8", errors="replace").strip()
    stderr = (stderr_bytes or b"").decode("utf-8", errors="replace").strip()
    rc = proc.returncode

    if rc != 0:
        snippet = (stderr or stdout)[:1500]
        return ToolResult.fail(
            f"claude_code exited {rc}. Output: {snippet}"
        )

    if not stdout:
        return ToolResult(
            text="claude_code returned no output (rc=0). Check whether the "
            "task actually required output, or look at file changes directly.",
        )

    # Truncate very long output — full output goes into the model feedback
    # but Discord-side display caps regardless.
    out = stdout
    if len(out) > 8000:
        out = out[:8000] + f"\n\n[... truncated, total {len(stdout)} chars]"

    return ToolResult(text=out)
