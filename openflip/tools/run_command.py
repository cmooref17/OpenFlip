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


_DEFAULT_TIMEOUT = 120  # seconds (Claude Code's Bash default is 2 min)
_MAX_TIMEOUT = 600      # seconds (Claude Code's Bash max is 10 min)
_MAX_OUTPUT = 50_000    # characters

# Background jobs (run_in_background=True): job_id -> asyncio.Task. When a job
# ends, its output is delivered to the agent as a new synthetic turn in the
# conversation that started it (Claude Code delivers a finished background
# shell the same way, as a task notification).
_BG_JOBS: dict[str, "asyncio.Task"] = {}


def _format(code: int | None, out: bytes, err: bytes) -> str:
    o = out.decode("utf-8", errors="replace") if out else ""
    e = err.decode("utf-8", errors="replace") if err else ""
    if len(o) > _MAX_OUTPUT:
        o = o[:_MAX_OUTPUT] + "\n[stdout truncated]"
    if len(e) > _MAX_OUTPUT:
        e = e[:_MAX_OUTPUT] + "\n[stderr truncated]"
    parts = []
    if o.strip():
        parts.append(o.strip())
    if e.strip():
        parts.append(f"[stderr]\n{e.strip()}")
    return f"Exit {code}:\n" + ("\n\n".join(parts) if parts else "(no output)")


async def _deliver_bg_result(agent_id: str, session, channel_id: int, job_id: str, command: str, body: str) -> None:
    """Hand a finished background job's output back to the agent as a new turn."""
    from ..registry import RUNNERS
    runner = RUNNERS.get(agent_id)
    if runner is None:
        print_ts(f"{COLOR_YELLOW}run_command bg {job_id}: agent '{agent_id}' not running; result dropped{COLOR_END}")
        return
    prompt = (f"[FRAMEWORK]: Background command {job_id} finished.\n"
              f"Command: {command[:300]}\n{body}")
    target = session if session is not None else channel_id
    if not target:
        print_ts(f"{COLOR_YELLOW}run_command bg {job_id}: no conversation to deliver to; result dropped{COLOR_END}")
        return
    await runner.run_synthetic_turn(target, prompt, auto_post_final_text=False, silent=True,
                                    originator_visibility="cron", log_tag="[bg-command] ")


async def _run_bg(job_id: str, command: str, timeout: int, agent_id: str, session, channel_id: int) -> None:
    try:
        proc = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.PIPE, start_new_session=True)
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            body = _format(proc.returncode, out, err)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            body = f"Timed out after {timeout}s and was killed."
        except asyncio.CancelledError:
            proc.kill()
            raise
    except asyncio.CancelledError:
        return
    except Exception as e:
        body = f"Failed to run: {e}"
    finally:
        _BG_JOBS.pop(job_id, None)
    print_ts(f"run_command bg {job_id} done; delivering result")
    try:
        await _deliver_bg_result(agent_id, session, channel_id, job_id, command, body)
    except Exception as e:
        print_ts(f"{COLOR_RED}run_command bg {job_id}: delivery failed: {e}{COLOR_END}", error=True)


@tool
async def run_command(command: str, timeout: int = 120, run_in_background: bool = False) -> ToolResult:
    """Run a shell command and return its output. Use for system tasks, file operations, checking status, running scripts, etc. Commands run as the bot's system user with a timeout.

    For anything that may take longer than a couple of minutes (builds, test suites, downloads), set run_in_background=true: the command starts, you get a job id right away and keep working, and its output arrives as a new message in this conversation when it finishes. Don't poll or sleep waiting for it.

    Args:
        command: The shell command to run (passed to the system shell —
            /bin/sh -c on Linux/macOS, cmd.exe /c on Windows).
        timeout: Max seconds before the process is killed (default 120, max 600).
        run_in_background: Start it and return immediately; the result is delivered to you as a new message when it ends.
    """
    from ..acl import current_caller_is_admin
    from ..config_global import get_run_command_admin_only
    # Built-in admin gate is ON by default (safe). Config key
    # `run_command_admin_only: false` disables it, leaving run_command
    # governed solely by the per-agent tool ACL in agent.json — for
    # deployments where a non-admin identity (e.g. a handle transport that
    # can't resolve admin) still needs the tool.
    if get_run_command_admin_only() and not current_caller_is_admin():
        return ToolResult.fail("run_command is admin-only.")

    command = (command or "").strip()
    if not command:
        return ToolResult.fail("No command provided.")

    timeout = max(1, min(int(timeout), _MAX_TIMEOUT))

    if run_in_background:
        import uuid
        from ..tool_executor import CURRENT_AGENT, CURRENT_SESSION, CURRENT_CHANNEL_ID
        agent = CURRENT_AGENT.get(None)
        if agent is None:
            return ToolResult.fail("run_in_background needs an agent turn.")
        session = CURRENT_SESSION.get(None)
        try:
            channel_id = int(CURRENT_CHANNEL_ID.get(0) or 0)
        except (TypeError, ValueError):
            channel_id = 0
        job_id = "bg-" + uuid.uuid4().hex[:8]
        print_ts(f"{COLOR_YELLOW}run_command bg {job_id}: {command!r} (timeout={timeout}s){COLOR_END}")
        _BG_JOBS[job_id] = asyncio.create_task(_run_bg(job_id, command, timeout, agent.id, session, channel_id))
        return ToolResult(model_feedback=(
            f"Started in the background as {job_id} (timeout {timeout}s). Its output will arrive as a "
            f"new message in this conversation when it finishes. Keep working; don't poll or sleep for it."))

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
