"""Scheduled job runner for openflip — fires synthetic agentTurn messages to AgentRunners.

Reads jobs from `cron/jobs.json`. Each job describes when to wake an agent,
what prompt to send, and which channel to post the response in.

Job schema (cron/jobs.json):
{
  "version": 1,
  "jobs": [
    {
      "id": "uuid-or-stable-string",
      "agentId": "<agent>",
      "name": "Hourly heartbeat",
      "enabled": true,
      "schedule": {
        "kind": "interval",      // "interval" only for now; "cron" planned
        "seconds": 3600           // for kind=interval
      },
      "channelId": 1234567890,    // Discord channel to post results in
      "payload": {
        "message": "Heartbeat. Do something useful.",  // explicit text prompt
        // OR (preferred for heartbeats):
        "heartbeat": true,         // load agents/<id>/HEARTBEAT.md as the prompt
        "timeoutSeconds": 300
      },
      "lastRunMs": 0              // updated by scheduler
    }
  ]
}

Prompt resolution:
- If `payload.heartbeat` is true, the cron reads `agents/<agentId>/HEARTBEAT.md`
  and uses its contents as the prompt. The heartbeat file is the agent's
  own instructions for what to do during a heartbeat — it lives in their
  agent directory but is NOT loaded into their normal system message, so
  Discord conversations stay clean.
- Otherwise, `payload.message` is used as the prompt directly.

Schedule kinds:
- "interval"  — fires every N seconds. Required field: `seconds`.
- "cron"      — standard cron expression. Required field: `expression`
                (e.g. "0 9 * * 5" = Fridays 9 AM). Optional `timezone`
                (IANA name, defaults to UTC). Backed by croniter.

The scheduler is intentionally conservative:
- One pass every 5 seconds, no overlap (job runs serialized per agent).
- Persisted state (`lastRunMs`) survives restarts so jobs don't re-fire on boot.
- A job whose `enabled` is false is skipped without touching state.
"""
from __future__ import annotations
import asyncio
import os
import time
from typing import Optional

from .registry import RUNNERS
from .utils import print_ts, load_json, save_json, project_root, COLOR_YELLOW, COLOR_RED, COLOR_GREEN, COLOR_END


_TICK_SECONDS = 5  # how often the scheduler wakes up to check schedules
_HEARTBEAT_FILE = "HEARTBEAT.md"

# Built-in fallback proactive prompt when an agent has no HEARTBEAT.md.
# Kept minimal — the agent's own HEARTBEAT.md should be the real source.
_DEFAULT_KAIROS_PROMPT = (
    "You are in proactive mode. Look around: check pending tasks, logs, "
    "repo state, anything relevant. If there is genuinely useful work to do, "
    "do it and use send_message to notify the operator. If nothing needs "
    "attention, emit NO text and call NO tools — silence is the correct "
    "response when nothing is actionable. Do NOT invent busywork."
)


def _jobs_path() -> str:
    return os.path.join(project_root(), "cron", "jobs.json")


def _agent_dir(agent_id: str) -> str:
    return os.path.join(project_root(), "agents", agent_id)


def _load_jobs() -> dict:
    data = load_json(_jobs_path(), default={"version": 1, "jobs": []})
    if not isinstance(data, dict) or "jobs" not in data:
        return {"version": 1, "jobs": []}
    return data


# Known top-level keys for a job's `payload` block. Extra keys are logged
# at load time as a "did you mean?" warning — catches typos like `hearbeat`
# that would otherwise silently no-op when the job fires.
_KNOWN_PAYLOAD_KEYS = {"heartbeat", "message", "timeoutSeconds"}


def validate_job(job: dict, ident: str = "(job)") -> list[str]:
    """Validate a single job dict. Returns a list of error messages
    (empty if clean). `ident` is used in error messages for context.

    Catches:
        - Cron expressions that croniter can't parse.
        - Unknown keys in `payload` (typo `hearbeat`, etc.).
        - Interval schedule with seconds <= 0.
        - Missing required fields per schedule kind.

    Does NOT validate referenced agents/channels — those can come and
    go and we don't want to block scheduler startup on a missing agent.

    Defense-in-depth note: `_due()` also catches malformed cron expressions
    at fire time. This function exists for operator visibility at load /
    write time — we want bad jobs surfaced when they enter the system,
    not silently never-fire weeks later.
    """
    errors: list[str] = []
    if not isinstance(job, dict):
        return ["job is not a dict"]

    # Payload key typo check.
    payload = job.get("payload") or {}
    if isinstance(payload, dict):
        extras = set(payload.keys()) - _KNOWN_PAYLOAD_KEYS
        if extras:
            errors.append(
                f"unknown payload key(s) {sorted(extras)} — "
                f"valid keys are {sorted(_KNOWN_PAYLOAD_KEYS)}"
            )

    # Schedule validation per kind.
    schedule = job.get("schedule") or {}
    kind = (schedule.get("kind") or "").strip()
    if kind == "interval":
        seconds = schedule.get("seconds")
        if not isinstance(seconds, (int, float)) or seconds <= 0:
            errors.append(f"interval schedule needs `seconds` > 0 (got {seconds!r})")
    elif kind == "cron":
        expression = (schedule.get("expression") or "").strip()
        if not expression:
            errors.append("cron schedule needs `expression`")
        else:
            try:
                from croniter import croniter
            except ImportError:
                errors.append("croniter not installed; can't validate cron expressions")
            else:
                if not croniter.is_valid(expression):
                    errors.append(f"cron expression {expression!r} is not valid")
        # Optional timezone — validate if present.
        tz_name = (schedule.get("timezone") or "").strip()
        if tz_name:
            try:
                from zoneinfo import ZoneInfo
                ZoneInfo(tz_name)
            except Exception as e:
                errors.append(f"invalid timezone {tz_name!r}: {e}")
    elif kind:
        errors.append(f"unknown schedule kind {kind!r}")
    else:
        errors.append("schedule kind missing")

    return errors


def validate_jobs(data: dict) -> list[tuple[str, str]]:
    """Validate every job in the loaded jobs.json. Returns a list of
    (job_identifier, error_message) for any job with problems. Empty
    list = all clean.

    Thin wrapper around `validate_job` for the per-file case. See
    `validate_job` for the per-job validation details.
    """
    errors: list[tuple[str, str]] = []
    jobs = (data or {}).get("jobs") or []
    if not isinstance(jobs, list):
        return [("(top-level)", "`jobs` field is not a list")]

    for i, job in enumerate(jobs):
        ident = (
            f"job[{i}] {job.get('name') or job.get('id') or '(unnamed)'}"
            if isinstance(job, dict) else f"job[{i}]"
        )
        for msg in validate_job(job, ident):
            errors.append((ident, msg))

    return errors


def _save_jobs(data: dict) -> None:
    os.makedirs(os.path.dirname(_jobs_path()), exist_ok=True)
    save_json(_jobs_path(), data)


def _in_quiet_hours(job: dict, now_ms: int) -> bool:
    """Return True if `now_ms` falls inside the job's quiet_hours window.

    quiet_hours schema on the job dict:
      {"start": "23:00", "end": "08:00", "timezone": "US/Mountain"}
    Missing/empty/malformed → never quiet (returns False).
    """
    qh = job.get("quiet_hours")
    if not qh or not isinstance(qh, dict):
        return False
    start_str = (qh.get("start") or "").strip()
    end_str = (qh.get("end") or "").strip()
    if not start_str or not end_str:
        return False
    try:
        import datetime as _dt
        from zoneinfo import ZoneInfo
        tz_name = (qh.get("timezone") or "").strip()
        tz = ZoneInfo(tz_name) if tz_name else _dt.timezone.utc
        now_dt = _dt.datetime.fromtimestamp(now_ms / 1000.0, tz=tz)
        sh, sm = (int(x) for x in start_str.split(":"))
        eh, em = (int(x) for x in end_str.split(":"))
        start_t = _dt.time(sh, sm)
        end_t = _dt.time(eh, em)
        cur_t = now_dt.time().replace(tzinfo=None)
        if start_t <= end_t:
            # Same-day window (e.g. 09:00–17:00)
            return start_t <= cur_t < end_t
        else:
            # Overnight window (e.g. 23:00–08:00)
            return cur_t >= start_t or cur_t < end_t
    except Exception:
        return False


def _due(job: dict, now_ms: int) -> bool:
    """Return True if this job is due to run right now.

    Two schedule kinds:
        "interval" — fires every N seconds. Field: `seconds`.
        "cron"     — standard cron expression. Field: `expression`
                     (e.g. "0 9 * * *"). Optional `timezone` (IANA name);
                     defaults to UTC. Uses croniter under the hood.
    """
    if not job.get("enabled", True):
        return False

    # ---- Kairos cost-control gates ----
    # Global kill switch: OPENFLIP_DISABLE_KAIROS=1 disables all kairos jobs.
    mode = (job.get("mode") or "").strip().lower()
    if mode == "kairos" and os.environ.get("OPENFLIP_DISABLE_KAIROS") == "1":
        return False

    # Quiet hours: job-level window during which the job is not due.
    if _in_quiet_hours(job, now_ms):
        return False
    schedule = job.get("schedule") or {}
    kind = schedule.get("kind")
    if kind == "interval":
        seconds = int(schedule.get("seconds", 0) or 0)
        if seconds <= 0:
            return False
        last = int(job.get("lastRunMs", 0) or 0)
        return (now_ms - last) >= seconds * 1000
    if kind == "cron":
        expression = (schedule.get("expression") or "").strip()
        if not expression:
            return False
        try:
            from croniter import croniter
        except ImportError:
            print_ts(
                f"{COLOR_YELLOW}cron: croniter not installed; job '{job.get('name')}' "
                f"can't fire. Install with: pip install croniter{COLOR_END}",
            )
            return False
        # Determine the timezone for evaluation. Default UTC.
        tz = None
        tz_name = (schedule.get("timezone") or "").strip()
        if tz_name:
            try:
                from zoneinfo import ZoneInfo
                tz = ZoneInfo(tz_name)
            except Exception as e:
                print_ts(
                    f"{COLOR_YELLOW}cron: job '{job.get('name')}' has invalid "
                    f"timezone {tz_name!r}: {e}; falling back to UTC{COLOR_END}",
                )
                tz = None
        import datetime as _dt
        last = int(job.get("lastRunMs", 0) or 0)
        # First run after install: lastRunMs may be 0. Anchor evaluation at
        # "the start of the scheduler tick window" (one tick ago) so a job
        # with no run history fires its first scheduled occurrence, not a
        # backlog of every occurrence since 1970.
        if last <= 0:
            anchor_ms = now_ms - _TICK_SECONDS * 1000
        else:
            anchor_ms = last
        try:
            anchor_dt = _dt.datetime.fromtimestamp(anchor_ms / 1000.0, tz=tz or _dt.timezone.utc)
            it = croniter(expression, anchor_dt)
            next_dt = it.get_next(_dt.datetime)
            next_ms = int(next_dt.timestamp() * 1000)
        except Exception as e:
            print_ts(
                f"{COLOR_YELLOW}cron: job '{job.get('name')}' expression {expression!r} "
                f"failed to parse: {e}{COLOR_END}",
            )
            return False
        return now_ms >= next_ms
    return False


def _resolve_prompt(agent_id: str, payload: dict) -> Optional[str]:
    """Resolve the prompt text for this job.

    Returns None on resolution failure (missing HEARTBEAT.md, empty message, etc).
    """
    if payload.get("heartbeat"):
        path = os.path.join(_agent_dir(agent_id), _HEARTBEAT_FILE)
        if not os.path.isfile(path):
            print_ts(f"{COLOR_YELLOW}cron: agent '{agent_id}' has no {_HEARTBEAT_FILE}; skipping heartbeat{COLOR_END}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                contents = f.read().strip()
        except OSError as e:
            print_ts(f"{COLOR_RED}cron: failed to read {path}: {e}{COLOR_END}", error=True)
            return None
        if not contents:
            print_ts(f"{COLOR_YELLOW}cron: {path} is empty; skipping{COLOR_END}")
            return None
        return contents
    msg = payload.get("message", "")
    return msg if msg else None


def _build_session(transport_name: str, tid: str, tool_grants: list[str] | None = None):
    """Build a synthetic-turn Session for a transport-prefixed target.

    Mirrors the direct-Session construction run_synthetic_turn uses for its
    non-Discord branch (runtime.py): no real speaker (speaker_id=0), DM-shaped,
    synthetic display name. The conversation_id carries the transport prefix so
    the turn lands in the right conversation file.

    `tool_grants` (from the job's `toolGrants`) makes those tools callable for
    this trusted synthetic session regardless of per-user ACL — additive only,
    see Session.tool_grants.
    """
    from .session import Session
    return Session(
        transport=transport_name,
        transport_id=str(tid),
        conversation_id=f"{transport_name}:{tid}",
        speaker_id=0,
        speaker_role_ids=[],
        is_owner=False,
        is_dm=True,
        display_name=f"synthetic:{tid}",
        handle="",
        tool_grants=list(tool_grants or []),
    )


def _resolve_session_target(job: dict, runner) -> object:
    """Resolve a cron job to a transport-prefixed Session (preferred) or a bare
    int channel id (last-resort fallback) for run_synthetic_turn.

    Preference order:
      1. job["sessionId"] — already transport-prefixed (e.g. "imessage:1").
         Split on ":" and build a Session with that exact conversation_id.
      2. job["channelId"] — legacy bare int. Resolve the prefix from the
         agent's transports: exactly one real transport → use its name; more
         than one → genuinely ambiguous, so log a WARNING naming the job and
         fall back to the first/default transport (no SILENT misroute).
    """
    name = job.get("name")
    # Per-session tool grants carried on the job (additive allow-path for this
    # trusted synthetic turn — see Session.tool_grants). Tolerate non-list junk.
    raw_grants = job.get("toolGrants")
    tool_grants = [str(g) for g in raw_grants] if isinstance(raw_grants, list) else []
    session_id = (job.get("sessionId") or "").strip()
    if session_id:
        transport_name, sep, tid = session_id.partition(":")
        transport_name, tid = transport_name.strip(), tid.strip()
        if sep and transport_name and tid:
            return _build_session(transport_name, tid, tool_grants)
        print_ts(
            f"{COLOR_YELLOW}cron: job '{name}' has malformed sessionId {session_id!r}; "
            f"falling back to channelId{COLOR_END}",
            agent=job.get("agentId"),
        )

    channel_id = job.get("channelId") or 0
    # Ignore headless/no-op transports ("internal") when picking a prefix.
    transports = [
        t for t in (getattr(runner, "_transports", None) or [])
        if getattr(t, "name", "") not in ("", "internal")
    ]
    if len(transports) == 1:
        return _build_session(getattr(transports[0], "name", ""), channel_id, tool_grants)
    if len(transports) > 1:
        first = getattr(transports[0], "name", "")
        names = ", ".join(getattr(t, "name", "?") for t in transports)
        print_ts(
            f"{COLOR_YELLOW}cron: job '{name}' has a legacy bare channelId={channel_id} "
            f"on a multi-transport agent (transports: {names}); cannot disambiguate. "
            f"Routing to first transport '{first}:{channel_id}' — set a prefixed "
            f"sessionId on this job to fix.{COLOR_END}",
            error=True,
            agent=job.get("agentId"),
        )
        return _build_session(first, channel_id, tool_grants)
    # No resolvable transport info (headless / unknown) — hand back the bare int
    # and let run_synthetic_turn apply its own default-transport handling.
    return int(channel_id)


async def _fire(job: dict) -> None:
    """Fire a synthetic turn for this job's agent."""
    agent_id = job.get("agentId")
    runner = RUNNERS.get(agent_id) if agent_id else None
    if not runner:
        print_ts(f"{COLOR_YELLOW}cron: skipping job '{job.get('name')}' — agent '{agent_id}' not running{COLOR_END}")
        return

    # Mode determines posting behavior:
    #   reminder        — needs a channel; final text auto-posts there.
    #   data_collection — no channel needed; runs fully silent. Agent can
    #                     still send_message inside the turn if it chooses.
    #   mixed           — same as data_collection at fire time. Reserved
    #                     for future "accumulate then summarize" pattern.
    #   kairos          — proactive tick. Silent + no auto-post. Agent
    #                     decides whether to act. Tick prompt =
    #                     <tick>HH:MM Weekday</tick> + HEARTBEAT.md.
    # Legacy jobs without `mode`: assume reminder if channelId present,
    # else data_collection.
    mode = (job.get("mode") or "").strip().lower()
    if not mode:
        mode = "reminder" if (job.get("channelId") or job.get("sessionId")) else "data_collection"

    # Every job needs an anchor for its synthetic turn's conversation file —
    # even silent jobs. Prefer the transport-prefixed sessionId; fall back to
    # the legacy bare channelId. Mode only controls posting.
    channel_id = job.get("channelId") or 0
    if not channel_id and not (job.get("sessionId") or "").strip():
        print_ts(f"{COLOR_YELLOW}cron: job '{job.get('name')}' has no channelId/sessionId; skipping{COLOR_END}")
        return

    payload = job.get("payload") or {}
    timeout_s = int(payload.get("timeoutSeconds", 300) or 300)

    # Kairos mode builds its own prompt: <tick>HH:MM Weekday</tick> +
    # HEARTBEAT.md contents (or a built-in default if missing).
    if mode == "kairos":
        import datetime as _dt
        from zoneinfo import ZoneInfo
        qh = job.get("quiet_hours") or {}
        tz_name = (qh.get("timezone") or "").strip() if isinstance(qh, dict) else ""
        try:
            tz = ZoneInfo(tz_name) if tz_name else _dt.timezone.utc
        except Exception:
            tz = _dt.timezone.utc
        now_dt = _dt.datetime.now(tz)
        tick_tag = f"<tick>{now_dt.strftime('%H:%M %A')}</tick>"
        # Load HEARTBEAT.md if present; fall back to built-in default.
        hb_path = os.path.join(_agent_dir(agent_id), _HEARTBEAT_FILE)
        hb_body = ""
        if os.path.isfile(hb_path):
            try:
                with open(hb_path, "r", encoding="utf-8") as f:
                    hb_body = f.read().strip()
            except OSError:
                pass
        if not hb_body:
            hb_body = _DEFAULT_KAIROS_PROMPT
        prompt = f"{tick_tag}\n\n{hb_body}"
    else:
        prompt = _resolve_prompt(agent_id, payload)
        if not prompt:
            return

    # Reminder mode posts the final reply text to the channel; the others
    # run silent (no auto-post). Agent can still call send_message inside
    # the turn if it wants to push something out.
    if mode == "reminder":
        auto_post, silent_flag = True, False
    else:
        # data_collection / mixed / kairos: silent. Agent can still
        # send_message explicitly inside the turn if it wants to push
        # something.
        auto_post, silent_flag = False, True

    # Speaker attribution. SECURITY (HIGH-2 from the audit):
    # synthetic turns previously defaulted to owner_id when no speaker
    # was passed, which meant ANY agent with add_cron_job access could
    # schedule a job that fired with owner privileges later. Now we
    # honor `createdBySpeakerId` stamped on the job at creation time.
    # Legacy jobs (no field) still fall through to owner — operator can
    # delete them or re-create from current code if they want strict
    # attribution backfilled.
    creator = job.get("createdBySpeakerId")
    try:
        creator_id = int(creator) if creator else 0
    except (TypeError, ValueError):
        creator_id = 0

    # Resolve a transport-prefixed Session (or bare-int fallback) BEFORE firing,
    # so the turn lands in the correct conversation file on multi-transport
    # agents instead of defaulting to the first transport's prefix.
    session_target = _resolve_session_target(job, runner)
    from .session import Session as _Session
    route = (
        session_target.conversation_id
        if isinstance(session_target, _Session)
        else f"channel={session_target}"
    )

    print_ts(
        f"{COLOR_GREEN}cron: firing '{job.get('name')}' → agent={agent_id} "
        f"{route} mode={mode} speaker={creator_id or 'owner-default'}{COLOR_END}",
        agent=agent_id,
    )
    try:
        await asyncio.wait_for(
            runner.run_synthetic_turn(
                session_target,
                prompt,
                speaker_id=creator_id,
                auto_post_final_text=auto_post,
                silent=silent_flag,
                # Tag the chain root: cron-fired chains log failures
                # loudly but don't surface them to the operator's channel.
                # The operator didn't ask for this turn — they shouldn't
                # be nagged if the chain went sideways internally.
                # Kairos mode gets its own tag so the runtime can identify
                # proactive ticks for idle-tick pruning and stop-hook
                # exemption.
                originator_visibility="kairos" if mode == "kairos" else "cron",
            ),
            timeout=timeout_s,
        )
    except asyncio.TimeoutError:
        print_ts(f"{COLOR_RED}cron: job '{job.get('name')}' timed out after {timeout_s}s{COLOR_END}", error=True, agent=agent_id)
    except Exception as e:
        print_ts(f"{COLOR_RED}cron: job '{job.get('name')}' crashed: {e}{COLOR_END}", error=True, agent=agent_id)


async def _tick() -> None:
    """One scheduler pass: fire any due jobs, persist updated lastRunMs."""
    data = _load_jobs()
    jobs = data.get("jobs") or []
    if not jobs:
        return
    now_ms = int(time.time() * 1000)
    for job in jobs:
        if _due(job, now_ms):
            # Update timestamp BEFORE running so a long-running job doesn't
            # re-fire if a tick lands mid-run on the next pass.
            job["lastRunMs"] = now_ms
            # Persist the stamp IMMEDIATELY — before awaiting _fire. If _fire
            # (or anything later in the tick) raises, the already-stamped jobs
            # must survive, or every due job re-fires on the next boot/tick
            # (a re-fire storm). Tolerate a save failure: a stamp lost here is
            # the pre-existing behavior, never worse.
            try:
                _save_jobs(data)
            except Exception as _persist_err:
                print_ts(
                    f"{COLOR_YELLOW}cron: failed to persist lastRunMs for "
                    f"{job.get('id', '?')}: {_persist_err}{COLOR_END}",
                    error=True,
                )
            await _fire(job)


async def run_scheduler() -> None:
    """Long-running task: sleep _TICK_SECONDS, run a tick, repeat. Cancellation-safe."""
    # Validate jobs.json once at startup so typos / bad cron expressions
    # surface here, not at fire time. (Audit P0 #7, 2026-05-19.)
    try:
        initial = _load_jobs()
        validation_errors = validate_jobs(initial)
    except Exception as e:
        print_ts(f"{COLOR_YELLOW}cron: jobs.json failed to load: {e}{COLOR_END}", error=True)
        validation_errors = []
    if validation_errors:
        print_ts(f"{COLOR_YELLOW}cron: {len(validation_errors)} job validation issue(s):{COLOR_END}")
        for ident, msg in validation_errors:
            print_ts(f"  {COLOR_YELLOW}- {ident}: {msg}{COLOR_END}")
        print_ts(f"{COLOR_YELLOW}cron: scheduler will still start, but bad jobs won't fire correctly.{COLOR_END}")

    print_ts(f"{COLOR_GREEN}cron: scheduler started (tick={_TICK_SECONDS}s){COLOR_END}")
    try:
        while True:
            try:
                await _tick()
            except Exception as e:
                print_ts(f"{COLOR_RED}cron: tick error: {e}{COLOR_END}", error=True)
            await asyncio.sleep(_TICK_SECONDS)
    except asyncio.CancelledError:
        print_ts("cron: scheduler stopped")
        raise
