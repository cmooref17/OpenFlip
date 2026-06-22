"""Cron job management tools for agents.

Lets an agent create, list, and cancel scheduled jobs for itself. Jobs land
in `cron/jobs.json` and are picked up by the scheduler (cron.py) within
~5 seconds.

Job modes:
- "reminder"        — fires a synthetic turn whose final text auto-posts to
                      the configured channel. Default mode for "remind me X
                      on Friday at 9am" style requests.
- "data_collection" — fires a synthetic turn that runs silently (no Discord
                      post). The agent's response lives in conversation
                      history but doesn't reach the user unless the agent
                      explicitly send_message's it.
- "mixed"           — same as data_collection for now; reserved for a
                      future use case where multiple data-collection runs
                      accumulate and a final summary fires as a reminder.

Schedules accept exactly ONE of: a cron expression ("0 9 * * 5" = Fridays
9 AM), an interval in seconds, or an absolute one-shot time (`run_at`). Use
cron expressions for "at a specific recurring time", intervals for "every N
seconds/minutes/hours", and `run_at` for a true run-once reminder that fires
at a single absolute moment and then auto-deletes.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import Optional

from ._base import tool, ToolResult
from ..utils import load_json, save_json, project_root


_VALID_MODES = ("reminder", "data_collection", "mixed")


def _jobs_path() -> str:
    return os.path.join(project_root(), "cron", "jobs.json")


def _load_jobs() -> dict:
    data = load_json(_jobs_path(), default={"version": 1, "jobs": []})
    if not isinstance(data, dict) or "jobs" not in data:
        return {"version": 1, "jobs": []}
    return data


def _save_jobs(data: dict) -> None:
    os.makedirs(os.path.dirname(_jobs_path()), exist_ok=True)
    save_json(_jobs_path(), data)


def _current_agent_id() -> Optional[str]:
    from ..tool_executor import CURRENT_AGENT
    try:
        return CURRENT_AGENT.get().id
    except LookupError:
        return None


def _current_channel_id() -> Optional[int]:
    from ..tool_executor import CURRENT_CHANNEL_ID
    try:
        return int(CURRENT_CHANNEL_ID.get())
    except (LookupError, TypeError, ValueError):
        return None


def _current_session_conversation_id() -> str:
    """Transport-prefixed conversation_id of the session we're currently inside.

    Used to default `session_id` on cron jobs so a job scheduled from an
    iMessage (or any non-default) transport fires back into the SAME
    conversation rather than the agent's default-transport channel. Returns
    "" when no session is in context (legacy synthetic paths).
    """
    from ..tool_executor import CURRENT_SESSION
    sess = CURRENT_SESSION.get(None)
    if sess is None:
        return ""
    return getattr(sess, "conversation_id", "") or ""


def _current_speaker_id() -> Optional[int]:
    """The user_id of whoever's turn we're currently inside.

    Used to attribute cron job creation. Stored as `created_by_speaker_id`
    on the job so when it fires, the synthetic turn runs as the creator
    rather than silently escalating to owner privileges (HIGH-2 from the
    security audit).
    """
    from ..tool_executor import CURRENT_SPEAKER_ID
    try:
        return int(CURRENT_SPEAKER_ID.get())
    except (LookupError, TypeError, ValueError):
        return None


def _validate_cron_expression(expr: str) -> Optional[str]:
    """Return None if valid, error string if invalid."""
    try:
        from croniter import croniter
    except ImportError:
        return "croniter is not installed; cron expressions can't be validated."
    if not croniter.is_valid(expr):
        return f"invalid cron expression: {expr!r}"
    return None


def _parse_run_at(run_at: str, tz_name: str) -> tuple[Optional[int], Optional[str]]:
    """Parse a one-shot `run_at` value into epoch milliseconds.

    Accepts either an ISO 8601 datetime ("2026-06-22T14:30:00", with optional
    timezone offset) OR a unix epoch SECONDS integer-as-string ("1782484200").
    A naive ISO datetime (no tz info) is interpreted in `tz_name` (IANA) when
    provided, else UTC — mirroring how cron expressions treat `timezone`.

    Returns (epoch_ms, None) on success or (None, error_string) on failure.
    """
    import datetime as _dt
    s = (run_at or "").strip()
    if not s:
        return None, "`run_at` is empty."
    # Unix epoch seconds integer-as-string (optional leading sign).
    if s.lstrip("+-").isdigit():
        return int(s) * 1000, None
    # ISO 8601.
    try:
        dt = _dt.datetime.fromisoformat(s)
    except ValueError as e:
        return None, f"`run_at` is not a valid ISO 8601 datetime or epoch seconds: {s!r} ({e})"
    if dt.tzinfo is None:
        tz = _dt.timezone.utc
        if tz_name:
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo(tz_name)
            except Exception as e:
                return None, f"invalid timezone {tz_name!r}: {e}"
        dt = dt.replace(tzinfo=tz)
    return int(dt.timestamp() * 1000), None


@tool
async def add_cron_job(
    name: str,
    prompt: str,
    cron: str = "",
    every_seconds: int = 0,
    run_at: str = "",
    mode: str = "reminder",
    timezone: str = "",
    channel_id: int = 0,
    session_id: str = "",
    tool_grants: list[str] = None,
) -> ToolResult:
    """Schedule a job for yourself (the current agent).

    Use this to create reminders ("remind me on Friday at 9am"), recurring
    research tasks, periodic checks — anything you want to fire on a
    schedule. The job runs as a synthetic turn for YOU, with `prompt` as the
    user message.

    Exactly ONE of `cron`, `every_seconds`, or `run_at` must be set:
      * `cron`: standard cron expression, e.g. "0 9 * * 5" (Fridays 9 AM),
        "*/15 * * * *" (every 15 minutes), "0 6 * * *" (daily 6 AM). Recurring.
      * `every_seconds`: fixed interval in seconds. 3600 = hourly. Recurring.
      * `run_at`: a single absolute time for a TRUE one-shot (run-once)
        reminder. Fires exactly once at that moment, then auto-deletes — no
        self-cancel hack needed. Accepts ISO 8601 ("2026-06-22T14:30:00",
        optional tz offset) or unix epoch seconds as a string ("1782484200").

    Args:
        name: Short human label for the job.
        prompt: What you (the agent) will see as the user message when the
            job fires.
        cron: A cron expression. Mutually exclusive with `every_seconds`/`run_at`.
        every_seconds: Interval in seconds. Mutually exclusive with `cron`/`run_at`.
        run_at: Absolute one-shot time (ISO 8601 or unix epoch seconds string).
            Mutually exclusive with `cron`/`every_seconds`. Must be in the
            future. The job fires once and then auto-deletes from jobs.json.
            A naive ISO datetime (no tz) is interpreted in `timezone` if set,
            else UTC.
        mode: "reminder" (default), "data_collection" (silent), or "mixed".
        timezone: IANA timezone for cron expressions (e.g. "US/Eastern") and
            for naive `run_at` datetimes. Defaults to UTC. Ignored for
            interval schedules.
        channel_id: Discord channel for reminder posts. Defaults to the
            current channel. Required for "reminder" and "mixed"; ignored
            for "data_collection". Legacy bare-int anchor — see `session_id`.
        session_id: Transport-prefixed conversation id (e.g. "imessage:1" or
            "discord:12345"). PREFERRED over `channel_id` on multi-transport
            agents: a bare channel id is ambiguous across transports, so the
            turn can fire into the wrong conversation. When omitted, defaults
            to the current session's prefixed id so the job fires back into
            THIS conversation. The bare `channel_id` is still stored for
            back-compat.
        tool_grants: Optional list of tool names this job's synthetic turn is
            authorized to call REGARDLESS of per-user auth. For trusted
            scheduled work that has no human speaker (speaker_id=0) and would
            otherwise fail per-user ACL — e.g. an email-summary job granted
            ["read_email", "send_message"]. Additive only: it never overrides
            an exclude deny and can't authorize a tool the agent wasn't
            configured with. Stored as `toolGrants` on the job. Empty/None =
            no extra grants (normal per-user ACL applies).

    Returns:
        The job_id of the created job, so you can cancel it later.
    """
    name = (name or "").strip()
    prompt = (prompt or "").strip()
    mode = (mode or "reminder").strip().lower()

    if not name:
        return ToolResult.fail("`name` is required.")
    if not prompt:
        return ToolResult.fail("`prompt` is required.")
    if mode not in _VALID_MODES:
        return ToolResult.fail(f"`mode` must be one of {_VALID_MODES}; got {mode!r}.")

    cron_expr = (cron or "").strip()
    run_at_str = (run_at or "").strip()
    set_count = sum(bool(x) for x in (cron_expr, every_seconds, run_at_str))
    if set_count == 0:
        return ToolResult.fail(
            "Set EXACTLY ONE of `cron` (expression), `every_seconds` (interval), "
            "or `run_at` (one-shot absolute time)."
        )
    if set_count > 1:
        return ToolResult.fail(
            "Set EXACTLY ONE of `cron`, `every_seconds`, or `run_at` — not multiple."
        )

    schedule: dict
    if cron_expr:
        err = _validate_cron_expression(cron_expr)
        if err:
            return ToolResult.fail(err)
        schedule = {"kind": "cron", "expression": cron_expr}
        tz = (timezone or "").strip()
        if tz:
            schedule["timezone"] = tz
    elif run_at_str:
        run_at_ms, err = _parse_run_at(run_at_str, (timezone or "").strip())
        if err:
            return ToolResult.fail(err)
        if run_at_ms <= int(time.time() * 1000):
            return ToolResult.fail("`run_at` must be in the future.")
        schedule = {"kind": "once", "runAtMs": run_at_ms}
    else:
        if every_seconds <= 0:
            return ToolResult.fail("`every_seconds` must be a positive integer.")
        schedule = {"kind": "interval", "seconds": int(every_seconds)}

    agent_id = _current_agent_id()
    if not agent_id:
        return ToolResult.fail("Tool invoked outside an agent context.")

    # Every job needs a channel to anchor its synthetic turn's conversation
    # file — even silent (data_collection) jobs. Mode only controls posting.
    resolved_channel = int(channel_id) if channel_id else (_current_channel_id() or 0)
    if not resolved_channel:
        return ToolResult.fail(
            "`channel_id` required (no current channel to default to)."
        )

    job_id = str(uuid.uuid4())
    # Capture creator's speaker_id so when this job fires, the synthetic
    # turn runs as the creator rather than defaulting to owner privileges.
    # See cron._fire — if `created_by_speaker_id` is absent, the fire-time
    # default is still owner (back-compat for jobs created before this
    # field existed). New jobs always have it stamped.
    creator_id = _current_speaker_id()
    # Transport-prefixed conversation id. Explicit arg wins; otherwise capture
    # the current session's prefixed id so the job fires back into THIS
    # conversation instead of the agent's default-transport channel. channelId
    # stays stored for back-compat (see cron._resolve_session_target).
    resolved_session_id = (session_id or "").strip() or _current_session_conversation_id()
    job = {
        "id": job_id,
        "agentId": agent_id,
        "name": name,
        "enabled": True,
        "mode": mode,
        "schedule": schedule,
        "payload": {"message": prompt},
        "lastRunMs": 0,
        "channelId": resolved_channel,
        "createdBySpeakerId": int(creator_id) if creator_id else 0,
    }
    if resolved_session_id:
        job["sessionId"] = resolved_session_id
    # Per-session tool grants (additive allow-path for this trusted synthetic
    # turn — see Session.tool_grants / cron._build_session). Stored only when
    # non-empty so existing jobs stay byte-identical.
    if tool_grants:
        job["toolGrants"] = [str(g) for g in tool_grants]

    # Validate the job before persisting — catches typos in payload keys,
    # invalid cron expressions, bad timezones, etc. that would otherwise
    # only surface at fire time (or never, for typo'd flags).
    # See cron.validate_job docstring for the full set of checks.
    from ..cron import validate_job
    validation_errors = validate_job(job, ident=name or job_id)
    if validation_errors:
        return ToolResult.fail(
            "Job rejected — validation errors:\n" +
            "\n".join(f"- {err}" for err in validation_errors)
        )

    data = _load_jobs()
    data.setdefault("jobs", []).append(job)
    _save_jobs(data)

    if cron_expr:
        sched_desc = f"cron {cron_expr!r}" + (f" ({timezone})" if timezone else "")
    elif run_at_str:
        sched_desc = f"once at {run_at_str}" + (f" ({timezone})" if timezone else "")
    else:
        sched_desc = f"every {every_seconds}s"
    return ToolResult(
        text=f"Scheduled '{name}' ({sched_desc}, mode={mode}). job_id={job_id}",
        model_feedback=f"Created cron job {job_id} — '{name}' ({sched_desc}, mode={mode}).",
    )


@tool
async def list_cron_jobs(agent_id: str = "", include_all_agents: bool = False) -> ToolResult:
    """List scheduled cron jobs.

    Defaults to YOUR jobs only. Pass `include_all_agents=True` to list
    everyone's jobs, or pass `agent_id` to filter to a specific agent.

    Args:
        agent_id: Filter by agent. Empty = current agent. Ignored if
            `include_all_agents` is True.
        include_all_agents: Show all agents' jobs.

    Returns:
        A formatted list of jobs with their id, name, schedule, mode, and
        enabled state.
    """
    data = _load_jobs()
    jobs = data.get("jobs") or []
    if not jobs:
        return ToolResult(text="No cron jobs scheduled.", model_feedback="No cron jobs.")

    if not include_all_agents:
        target_agent = (agent_id or "").strip() or _current_agent_id()
        if not target_agent:
            return ToolResult.fail("No current agent and no `agent_id` provided.")
        jobs = [j for j in jobs if j.get("agentId") == target_agent]
        if not jobs:
            return ToolResult(
                text=f"No cron jobs for agent '{target_agent}'.",
                model_feedback=f"No cron jobs for {target_agent}.",
            )

    lines = []
    for j in jobs:
        sched = j.get("schedule") or {}
        if sched.get("kind") == "cron":
            sched_desc = f"cron {sched.get('expression')!r}"
            tz = sched.get("timezone")
            if tz:
                sched_desc += f" ({tz})"
        elif sched.get("kind") == "interval":
            sched_desc = f"every {sched.get('seconds')}s"
        elif sched.get("kind") == "once":
            sched_desc = f"once @ {sched.get('runAtMs')}ms (one-shot)"
        else:
            sched_desc = "(unknown schedule)"

        enabled_marker = "" if j.get("enabled", True) else " [DISABLED]"
        mode = j.get("mode") or ("reminder" if j.get("channelId") else "data_collection")
        lines.append(
            f"- {j.get('id')}: {j.get('name')!r}{enabled_marker}\n"
            f"    agent={j.get('agentId')} mode={mode} schedule={sched_desc}"
        )

    body = "\n".join(lines)
    return ToolResult(text=body, model_feedback=body)


@tool
async def cancel_cron_job(job_id: str) -> ToolResult:
    """Cancel (delete) a cron job by its id.

    Args:
        job_id: The job id returned by `add_cron_job` or shown by
            `list_cron_jobs`.

    Returns:
        Success or failure message.
    """
    job_id = (job_id or "").strip()
    if not job_id:
        return ToolResult.fail("`job_id` is required.")

    data = _load_jobs()
    jobs = data.get("jobs") or []
    before = len(jobs)
    remaining = [j for j in jobs if j.get("id") != job_id]
    if len(remaining) == before:
        return ToolResult.fail(f"No cron job with id={job_id!r}.")

    data["jobs"] = remaining
    _save_jobs(data)

    return ToolResult(
        text=f"Canceled job {job_id}.",
        model_feedback=f"Canceled cron job {job_id}.",
    )
