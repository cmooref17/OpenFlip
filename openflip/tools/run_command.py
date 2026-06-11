"""Run a shell command and return its output.

Executes commands via asyncio subprocess. Output is capped and commands
are killed after a timeout. Access is controlled by the normal tool ACL
in agent.json — only users listed under run_command can trigger it.

Path restrictions (allowed_read_paths / allowed_write_paths) do NOT apply
here — this tool can run anything the system user can. The ACL on who can
invoke it is the only gate.
"""
from __future__ import annotations

import asyncio

from ._base import tool, ToolResult, TOOL_REGISTRY
from ..utils import print_ts, COLOR_YELLOW, COLOR_RED, COLOR_END


_DEFAULT_TIMEOUT = 30  # seconds
_MAX_OUTPUT = 50_000   # characters


@tool
async def run_command(command: str, timeout: int = 30) -> ToolResult:
    """Run a shell command and return its output. Use for system tasks, file operations, checking status, running scripts, etc. Commands run as the bot's system user with a timeout.

    Args:
        command: The shell command to run (passed to the system shell —
            /bin/sh -c on Linux/macOS, cmd.exe /c on Windows).
        timeout: Max seconds to wait before killing the process (default 30, max 120).
    """
    from ..acl import current_caller_is_owner
    if not current_caller_is_owner():
        return ToolResult.fail("run_command is owner-only.")

    command = (command or "").strip()
    if not command:
        return ToolResult.fail("No command provided.")

    timeout = max(1, min(int(timeout), 120))

    print_ts(f"{COLOR_YELLOW}run_command: {command!r} (timeout={timeout}s){COLOR_END}")

    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult.fail(f"Command timed out after {timeout}s and was killed.")

        out = stdout.decode("utf-8", errors="replace") if stdout else ""
        err = stderr.decode("utf-8", errors="replace") if stderr else ""

        # Truncate
        if len(out) > _MAX_OUTPUT:
            out = out[:_MAX_OUTPUT] + "\n[stdout truncated]"
        if len(err) > _MAX_OUTPUT:
            err = err[:_MAX_OUTPUT] + "\n[stderr truncated]"

        parts = []
        if out.strip():
            parts.append(out.strip())
        if err.strip():
            parts.append(f"[stderr]\n{err.strip()}")

        code = proc.returncode
        output = "\n\n".join(parts) if parts else "(no output)"

        if code == 0:
            return ToolResult(model_feedback=f"Exit 0:\n{output}")
        else:
            return ToolResult(model_feedback=f"Exit {code}:\n{output}")

    except Exception as e:
        print_ts(f"{COLOR_RED}run_command error: {e}{COLOR_END}", error=True)
        return ToolResult.fail(f"Failed to run command: {e}")


# Don't dump raw command output into Discord
if "run_command" in TOOL_REGISTRY:
    TOOL_REGISTRY["run_command"].silent_to_discord = True
