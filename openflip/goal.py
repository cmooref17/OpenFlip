"""/goal: keep working until a condition is met (Claude Code's /goal).

Design: docs/goal_plan_reread.md. One goal per conversation, saved next to the
conversation file (conversations/<cid>.goal.json) so it survives a restart.
When the agent tries to end a turn, a small fast model reads the condition and
the recent transcript and answers met / not met / impossible with a reason.
Not met keeps the turn going with the reason as guidance.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict

from .utils import print_ts, save_json, COLOR_YELLOW, COLOR_END

MAX_CONDITION_CHARS = 4000            # Claude Code's limit
NO_PROGRESS_LIMIT = 3                 # continuations with no tool use before the loop pauses
TRANSCRIPT_CHARS = 24_000             # recent transcript the evaluator sees
EVAL_TIMEOUT_S = 60.0
CLEAR_WORDS = {"clear", "stop", "off", "reset", "none", "cancel"}
# FrameworkErrorEvent kinds (anthropic_conversation / _anthropic_stream) that
# won't clear until the operator fixes something: auth (expired/revoked login,
# CC's "authentication failure") and bad_request (a request the API rejects as
# built: too large, unknown model; retrying sends the same thing). Everything
# else (rate_limit, overloaded, transport, timeout, empty_stream, ...) leaves
# the goal set, like CC's retry/pause errors.
UNRECOVERABLE_ERROR_KINDS = {"auth", "bad_request"}

EVALUATOR_SYSTEM = (
    "You judge whether an AI agent has met a goal. You get the goal condition and "
    "the recent conversation (the agent's messages and tool results). Judge ONLY "
    "from what the conversation shows; you can't run anything. If the agent "
    "claims something without the tool output that proves it, it is not met. "
    "Answer with ONE JSON object and nothing else: "
    '{"verdict": "met" | "not_met" | "impossible", "reason": "<one or two '
    'sentences: what is still missing, or why it is met>"}. Use impossible only '
    "when the condition can never be satisfied."
)


@dataclass
class Goal:
    condition: str
    set_at: float = field(default_factory=time.time)
    turns_evaluated: int = 0
    last_reason: str = ""
    no_progress: int = 0          # consecutive continuations without tool use
    paused: bool = False          # no-progress guard tripped; resumes on next operator message
    achieved_at: float = 0.0      # set when met (kept as the "last achieved" record)
    status: str = "active"        # active | achieved


def goal_path(agent_dir: str, conversation_id: str) -> str:
    from ._conversation_io import fs_encode
    return os.path.join(agent_dir, "conversations", f"{fs_encode(conversation_id)}.goal.json")


def load_goal(agent_dir: str, conversation_id: str) -> Goal | None:
    """The saved goal record (active or the last achieved one), or None."""
    if not agent_dir or not conversation_id:
        return None
    try:
        with open(goal_path(agent_dir, conversation_id), "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, ValueError, OSError):
        return None
    if not isinstance(data, dict) or not data.get("condition"):
        return None
    known = {k: data[k] for k in Goal.__dataclass_fields__ if k in data}
    return Goal(**known)


def active_goal(agent_dir: str, conversation_id: str) -> Goal | None:
    g = load_goal(agent_dir, conversation_id)
    return g if g is not None and g.status == "active" else None


def save_goal(agent_dir: str, conversation_id: str, goal: Goal | None) -> None:
    path = goal_path(agent_dir, conversation_id)
    if goal is None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_json(path, asdict(goal))


def set_goal(agent_dir: str, conversation_id: str, condition: str) -> Goal:
    g = Goal(condition=condition.strip()[:MAX_CONDITION_CHARS])
    save_goal(agent_dir, conversation_id, g)
    return g


def clear_goal(agent_dir: str, conversation_id: str) -> Goal | None:
    """Clear an active goal. Returns it, or None if nothing was active."""
    g = active_goal(agent_dir, conversation_id)
    if g is None:
        return None
    save_goal(agent_dir, conversation_id, None)
    return g


def unpause_on_operator_message(agent_dir: str, conversation_id: str) -> None:
    """CC: after the no-progress pause, evaluation resumes on the next prompt."""
    g = active_goal(agent_dir, conversation_id)
    if g is not None and (g.paused or g.no_progress):
        g.paused = False
        g.no_progress = 0
        save_goal(agent_dir, conversation_id, g)


def _fmt_duration(seconds: float) -> str:
    s = int(max(0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h {m}m" if h else (f"{m}m {s}s" if m else f"{s}s")


def status_text(agent_dir: str, conversation_id: str) -> str:
    g = load_goal(agent_dir, conversation_id)
    if g is None:
        return "No goal set. Set one with `/goal <condition>`."
    if g.status == "achieved":
        return (f"No active goal. Last achieved: {g.condition}\n"
                f"Took {_fmt_duration(g.achieved_at - g.set_at)}, {g.turns_evaluated} turns evaluated.")
    lines = [f"Goal active: {g.condition}",
             f"Running {_fmt_duration(time.time() - g.set_at)}"]
    if g.turns_evaluated:
        lines.append(f"Turns evaluated: {g.turns_evaluated}")
        lines.append(f"Last check: {g.last_reason}")
    if g.paused:
        lines.append("Paused: no progress for several turns. Send a message to resume.")
    return "\n".join(lines)


def parse_command(arg: str) -> tuple[str, str]:
    """('status'|'clear'|'set', condition)."""
    a = (arg or "").strip()
    if not a:
        return "status", ""
    if a.lower() in CLEAR_WORDS:
        return "clear", ""
    return "set", a


def kickoff_prompt(condition: str) -> str:
    return (f"[FRAMEWORK]: The operator set a goal with /goal. Work toward it now and keep going "
            f"until it is met; after each turn a separate model checks it, so prove progress "
            f"with real tool output (test results, command output), not claims.\nGoal: {condition}")


# --------------------------------------------------------------------------- evaluation

@dataclass
class Verdict:
    verdict: str          # met | not_met | impossible | error
    reason: str = ""


def _parse_verdict(text: str) -> Verdict:
    t = (text or "").strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return Verdict("error", f"evaluator gave no JSON: {t[:200]!r}")
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return Verdict("error", f"evaluator gave bad JSON: {t[:200]!r}")
    v = str(obj.get("verdict") or "").strip().lower().replace(" ", "_").replace("-", "_")
    if v not in ("met", "not_met", "impossible"):
        return Verdict("error", f"evaluator gave unknown verdict {v!r}")
    return Verdict(v, str(obj.get("reason") or "").strip())


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for b in content:
            if not isinstance(b, dict):
                continue
            t = b.get("type")
            if t == "text":
                out.append(b.get("text", ""))
            elif t == "tool_use":
                out.append(f"[tool call {b.get('name')}: {json.dumps(b.get('input'), default=str)[:1500]}]")
            elif t == "tool_result":
                out.append(f"[tool result: {_content_text(b.get('content'))[:3000]}]")
        return "\n".join(x for x in out if x)
    return str(content or "")


def transcript_tail(messages, limit: int = TRANSCRIPT_CHARS) -> str:
    """Recent conversation as plain text, oldest first, capped at `limit` chars
    (the newest messages are the ones kept)."""
    parts: list[str] = []
    total = 0
    for m in reversed(list(messages or [])):
        if isinstance(m, dict):
            role, content = m.get("role", ""), m.get("content")
        else:
            role, content = getattr(m, "role", ""), getattr(m, "content", None)
        text = _content_text(content).strip()
        if not text:
            continue
        piece = f"{str(role).upper()}: {text}"
        room = limit - total
        if len(piece) > room:
            if room > 200:
                parts.append(piece[-room:])
            break
        parts.append(piece)
        total += len(piece)
    return "\n\n".join(reversed(parts))


async def _default_caller(model: str, prompt: str, timeout: float) -> str:
    """One-shot Anthropic call: same auth + endpoint as prompt hooks."""
    import aiohttp
    from .anthropic_conversation import _load_oauth_access_token, _DEFAULT_API_BASE, build_auth
    from .utils import http_session
    from .config_global import get_config
    cfg = get_config() or {}
    if str(cfg.get("hooks_auth") or "").lower() == "api_key":
        from .config_global import get_anthropic_api_key
        key = get_anthropic_api_key()
        if not key:
            raise RuntimeError("Anthropic API key not configured")
        headers, prefix = build_auth("api_key", api_key=key)
    else:
        token = await _load_oauth_access_token()
        if not token:
            raise RuntimeError("OAuth token unavailable")
        headers, prefix = build_auth("oauth", access_token=token)
    body = {
        "model": model,
        "max_tokens": 1024,
        "system": list(prefix) + [{"type": "text", "text": EVALUATOR_SYSTEM}],
        "messages": [{"role": "user", "content": prompt}],
    }
    session = await http_session()
    async with session.post(f"{_DEFAULT_API_BASE}/v1/messages", json=body, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        data = await resp.json(content_type=None)
        if resp.status != 200:
            raise RuntimeError(f"goal evaluator HTTP {resp.status}: {str(data)[:200]}")
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


# Tests swap this out to avoid network calls.
EVAL_CALLER = _default_caller


def evaluator_model() -> str:
    """hooks_model, else the memory-recall selector model (the small fast model)."""
    from .hooks import _prompt_model
    return _prompt_model({})


async def evaluate(condition: str, messages) -> Verdict:
    model = evaluator_model()
    if not model:
        return Verdict("error", "no evaluator model (set hooks_model in config.json)")
    prompt = (f"GOAL CONDITION:\n{condition}\n\nRECENT CONVERSATION (oldest first):\n"
              f"{transcript_tail(messages)}\n\nHas the goal condition been met?")
    try:
        answer = await asyncio.wait_for(EVAL_CALLER(model, prompt, EVAL_TIMEOUT_S),
                                        timeout=EVAL_TIMEOUT_S + 5)
    except Exception as e:
        return Verdict("error", f"evaluator call failed: {e}")
    return _parse_verdict(answer)


@dataclass
class StopDecision:
    """What the runtime does when the agent tries to end a turn with a goal set."""
    keep_going: bool = False
    feedback: str = ""      # [FRAMEWORK] note for the model when keep_going
    notice: str = ""        # line to post to the operator ('' = nothing)


async def on_stop(agent_dir: str, conversation_id: str, messages, *,
                  used_tools_this_turn: bool, background_running: bool) -> StopDecision:
    """Called when the agent wants to end a turn. Never raises."""
    try:
        g = active_goal(agent_dir, conversation_id)
        if g is None or g.paused:
            return StopDecision()
        if background_running:
            # CC: background work defers evaluation; its result turn evaluates.
            print_ts("goal: background work still running, evaluation deferred")
            return StopDecision()
        v = await evaluate(g.condition, messages)
        # Re-read: the operator may have cleared or replaced it meanwhile.
        g2 = active_goal(agent_dir, conversation_id)
        if g2 is None or g2.set_at != g.set_at:
            return StopDecision()
        g = g2
        if v.verdict == "error":
            print_ts(f"{COLOR_YELLOW}goal: {v.reason}; turn ends, goal stays set{COLOR_END}")
            return StopDecision(notice=f"Goal check failed ({v.reason}). The goal is still set.")
        g.turns_evaluated += 1
        g.last_reason = v.reason
        if v.verdict == "met":
            g.status = "achieved"
            g.achieved_at = time.time()
            save_goal(agent_dir, conversation_id, g)
            return StopDecision(notice=f"Goal achieved: {g.condition}\n{v.reason}")
        if v.verdict == "impossible":
            save_goal(agent_dir, conversation_id, None)
            return StopDecision(notice=f"Goal cleared, judged impossible: {v.reason}")
        g.no_progress = 0 if used_tools_this_turn else g.no_progress + 1
        if g.no_progress >= NO_PROGRESS_LIMIT:
            g.paused = True
            save_goal(agent_dir, conversation_id, g)
            return StopDecision(notice=(
                "Goal paused: no tool use for several turns in a row. The goal is still "
                f"set; send a message to resume.\nLast check: {v.reason}"))
        save_goal(agent_dir, conversation_id, g)
        return StopDecision(keep_going=True, feedback=(
            "[FRAMEWORK]: Goal not met yet, keep working. The goal check said: "
            f"{v.reason}\nGoal: {g.condition}"))
    except Exception as e:
        print_ts(f"{COLOR_YELLOW}goal: on_stop failed: {e}{COLOR_END}")
        return StopDecision()


def on_framework_error(agent_dir: str, conversation_id: str, kind: str) -> str:
    """A turn failed. Unrecoverable kinds clear the goal; returns the notice ('' = none)."""
    g = active_goal(agent_dir, conversation_id)
    if g is None:
        return ""
    if (kind or "").lower() in UNRECOVERABLE_ERROR_KINDS:
        save_goal(agent_dir, conversation_id, None)
        return (f"Goal cleared after an unrecoverable error ({kind}). Fix the cause, then "
                f"run /goal again to continue.")
    return ""


def background_running_for(session_or_channel) -> bool:
    """Any run_command background job still running that was started from this
    conversation? Checked by conversation_id when a Session is given."""
    try:
        from .tools import run_command as _rc
    except Exception:
        return False
    owners = getattr(_rc, "_BG_OWNERS", {})
    want = getattr(session_or_channel, "conversation_id", None) or str(session_or_channel or "")
    for job_id, task in list(getattr(_rc, "_BG_JOBS", {}).items()):
        if task.done():
            continue
        if owners.get(job_id) == want:
            return True
    return False
