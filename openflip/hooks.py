"""Hooks: owner-configured commands / model checks that run on agent events.

Modeled on Claude Code's hooks (code.claude.com/docs/en/hooks). Design and the
event table: docs/hooks.md. Config lives in config.json `hooks` (every agent)
and agent.json `hooks` (one agent), same shape as Claude Code:

    {"<Event>": [{"matcher": "<pattern>", "hooks": [{"type": "command", ...}]}]}

Events in v1: PreToolUse, PostToolUse, UserPromptSubmit, Stop.
Handlers in v1: command (event JSON on stdin; exit 2 = block, stderr = reason)
and prompt (a one-shot model call answering {"ok", "reason", "impossible"}).

Hooks never break a turn: every failure is logged and treated as non-blocking.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .utils import print_ts, COLOR_YELLOW, COLOR_END

EVENTS = ("PreToolUse", "PostToolUse", "UserPromptSubmit", "Stop")
BLOCKING_EVENTS = {"PreToolUse", "UserPromptSubmit", "Stop"}
# Events whose matcher filters on the tool name. The others ignore the matcher.
TOOL_EVENTS = {"PreToolUse", "PostToolUse"}

DEFAULT_COMMAND_TIMEOUT_S = 600.0   # Claude Code's default for command hooks
DEFAULT_PROMPT_TIMEOUT_S = 30.0     # Claude Code's default for prompt hooks
STOP_CONTINUATION_CAP = 8           # Claude Code: 8 consecutive Stop continuations, then stop anyway
TOOL_RESPONSE_CAP = 20_000          # chars of tool output put in PostToolUse input
_OUTPUT_CAP = 10_000                # chars of hook stdout/stderr kept

_SIMPLE_MATCHER = re.compile(r"^[A-Za-z0-9_\- ,|]*$")


# --------------------------------------------------------------------------- config

def hooks_disabled() -> bool:
    """Kill switch: env OPENFLIP_DISABLE_HOOKS=1 or config.json disable_all_hooks."""
    if os.environ.get("OPENFLIP_DISABLE_HOOKS") == "1":
        return True
    try:
        from .config_global import get_config
        return bool((get_config() or {}).get("disable_all_hooks", False))
    except Exception:
        return False


def _global_hooks() -> dict:
    try:
        from .config_global import get_config
        h = (get_config() or {}).get("hooks")
        return h if isinstance(h, dict) else {}
    except Exception:
        return {}


def matcher_matches(matcher: Any, value: str) -> bool:
    """Claude Code's matcher rules. '*', '' or None: everything. Only letters,
    digits, '_', '-', spaces, ',' and '|': exact name(s). Anything else: an
    unanchored regex. A bad regex matches nothing (and is logged)."""
    if matcher is None:
        return True
    m = str(matcher)
    if m in ("", "*"):
        return True
    if _SIMPLE_MATCHER.match(m):
        names = [p.strip() for p in re.split(r"[|,]", m) if p.strip()]
        return value in names
    try:
        return re.search(m, value) is not None
    except re.error as e:
        print_ts(f"{COLOR_YELLOW}hooks: bad matcher regex {m!r}: {e}{COLOR_END}")
        return False


def handlers_for(agent, event: str, match_value: str = "") -> list[dict]:
    """Every handler that should run for `event`: config.json hooks first, then
    the agent's own. Identical handlers (same JSON) run once."""
    if hooks_disabled():
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for source in (_global_hooks(), getattr(agent, "hooks", None) or {}):
        groups = source.get(event) if isinstance(source, dict) else None
        if not isinstance(groups, list):
            continue
        for group in groups:
            if not isinstance(group, dict):
                continue
            if event in TOOL_EVENTS and not matcher_matches(group.get("matcher"), match_value):
                continue
            for h in group.get("hooks") or []:
                if not isinstance(h, dict) or h.get("type") not in ("command", "prompt"):
                    continue
                key = json.dumps(h, sort_keys=True, default=str)
                if key in seen:
                    continue
                seen.add(key)
                out.append(h)
    return out


# --------------------------------------------------------------------------- results

@dataclass
class HookOutcome:
    """Combined result of every handler for one event."""
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)       # why it blocked
    context: list[str] = field(default_factory=list)       # additionalContext / plain stdout
    updated_input: dict | None = None                       # PreToolUse only
    allowed: bool = False                                   # a PreToolUse hook said "allow"
    impossible: bool = False                                # Stop prompt hook: can't be met
    errors: list[str] = field(default_factory=list)         # non-blocking failures

    @property
    def reason(self) -> str:
        return "\n".join(r for r in self.reasons if r) or "Blocked by a hook."


@dataclass
class _HandlerResult:
    block: bool = False
    reason: str = ""
    context: str = ""
    updated_input: dict | None = None
    allow: bool = False
    deny: bool = False
    impossible: bool = False
    error: str = ""


def _apply_json(out: dict, event: str, r: _HandlerResult) -> None:
    """Read Claude Code's JSON output fields from a handler's stdout object."""
    if out.get("decision") == "block":
        r.block = True
        r.reason = str(out.get("reason") or r.reason or "")
    hso = out.get("hookSpecificOutput")
    if isinstance(hso, dict):
        ctx = hso.get("additionalContext")
        if isinstance(ctx, str) and ctx.strip():
            r.context = ctx.strip()
        if event == "PreToolUse":
            pd = hso.get("permissionDecision")
            if pd == "deny":
                r.deny = True
                r.block = True
                r.reason = str(hso.get("permissionDecisionReason") or r.reason or "")
            elif pd == "allow":
                r.allow = True
            ui = hso.get("updatedInput")
            if isinstance(ui, dict):
                r.updated_input = ui


# --------------------------------------------------------------------------- handlers

async def _run_command(handler: dict, event: str, payload: dict, agent) -> _HandlerResult:
    cmd = str(handler.get("command") or "").strip()
    if not cmd:
        return _HandlerResult(error="command hook has no 'command'")
    timeout = float(handler.get("timeout") or DEFAULT_COMMAND_TIMEOUT_S)
    agent_dir = os.path.dirname(getattr(agent, "path", "") or "") or os.getcwd()
    env = dict(os.environ)
    env.update({
        "OPENFLIP_AGENT_ID": str(getattr(agent, "id", "")),
        "OPENFLIP_AGENT_DIR": agent_dir,
        "OPENFLIP_HOOK_EVENT": event,
    })
    data = json.dumps(payload, default=str).encode("utf-8")
    try:
        proc = await asyncio.create_subprocess_exec(
            "/bin/sh", "-c", cmd,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, cwd=agent_dir, env=env,
            start_new_session=True,
        )
    except Exception as e:
        return _HandlerResult(error=f"command hook failed to start: {e}")
    try:
        so, se = await asyncio.wait_for(proc.communicate(data), timeout=timeout)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        return _HandlerResult(error=f"command hook timed out after {timeout:.0f}s: {cmd[:80]}")
    stdout = so.decode("utf-8", "replace")[:_OUTPUT_CAP].strip()
    stderr = se.decode("utf-8", "replace")[:_OUTPUT_CAP].strip()
    r = _HandlerResult()
    parsed = None
    if stdout.startswith("{"):
        try:
            parsed = json.loads(stdout)
        except json.JSONDecodeError:
            parsed = None
    if proc.returncode == 2:
        if event in BLOCKING_EVENTS:
            r.block = True
            r.deny = event == "PreToolUse"
            r.reason = stderr or "Blocked by a hook."
        else:
            # Can't block (the tool already ran): stderr goes to the model.
            r.context = stderr
        if isinstance(parsed, dict):
            _apply_json(parsed, event, r)
            if event in BLOCKING_EVENTS:
                r.block = True  # exit 2 stays a block whatever the JSON says
                r.deny = r.deny or event == "PreToolUse"
                r.allow = False
        return r
    if proc.returncode != 0:
        r.error = f"command hook exited {proc.returncode}: {(stderr or stdout)[:300]}"
        return r
    if isinstance(parsed, dict):
        _apply_json(parsed, event, r)
    elif stdout.startswith("{"):
        r.error = "command hook printed invalid JSON"
    elif stdout and event == "UserPromptSubmit":
        r.context = stdout
    return r


PromptCaller = Callable[[str, str, float], Awaitable[str]]

PROMPT_HOOK_SYSTEM = (
    "You are a hook evaluator for an AI agent framework. You get a condition or "
    "rule and the JSON of an event. Decide whether the event satisfies it. "
    "Answer with ONE JSON object and nothing else: "
    '{"ok": true|false, "reason": "<why, required when ok is false>", '
    '"impossible": true|false}. Set impossible only when the condition can '
    "never be satisfied."
)


async def _default_prompt_caller(model: str, prompt: str, timeout: float) -> str:
    """One-shot Anthropic call, same auth path as the memory-recall selector."""
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
        "system": list(prefix) + [{"type": "text", "text": PROMPT_HOOK_SYSTEM}],
        "messages": [{"role": "user", "content": prompt}],
    }
    session = await http_session()
    async with session.post(f"{_DEFAULT_API_BASE}/v1/messages", json=body, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        data = await resp.json(content_type=None)
        if resp.status != 200:
            raise RuntimeError(f"prompt hook HTTP {resp.status}: {str(data)[:200]}")
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")


# Tests swap this out to avoid network calls.
PROMPT_CALLER: PromptCaller = _default_prompt_caller


def _prompt_model(handler: dict) -> str:
    m = str(handler.get("model") or "").strip()
    if not m:
        try:
            from .config_global import get_config
            m = str((get_config() or {}).get("hooks_model") or "").strip()
        except Exception:
            m = ""
    if not m:
        try:
            from .memory_recall_api import selector_model
            m = selector_model()
        except Exception:
            m = ""
    return m.split("/", 1)[1] if "/" in m else m


def _parse_prompt_answer(text: str) -> dict | None:
    t = (text or "").strip()
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) and "ok" in obj else None


async def _run_prompt(handler: dict, event: str, payload: dict, agent) -> _HandlerResult:
    tmpl = str(handler.get("prompt") or "").strip()
    if not tmpl:
        return _HandlerResult(error="prompt hook has no 'prompt'")
    model = _prompt_model(handler)
    if not model:
        return _HandlerResult(error="prompt hook: no model (set hooks_model in config.json)")
    args = json.dumps(payload, default=str, ensure_ascii=False)
    prompt = tmpl.replace("$ARGUMENTS", args) if "$ARGUMENTS" in tmpl else f"{tmpl}\n\n{args}"
    timeout = float(handler.get("timeout") or DEFAULT_PROMPT_TIMEOUT_S)
    try:
        answer = await asyncio.wait_for(PROMPT_CALLER(model, prompt, timeout), timeout=timeout + 5)
    except Exception as e:
        return _HandlerResult(error=f"prompt hook failed: {e}")
    obj = _parse_prompt_answer(answer)
    if obj is None:
        return _HandlerResult(error=f"prompt hook gave no valid JSON: {str(answer)[:200]!r}")
    if obj.get("ok") is True:
        return _HandlerResult()
    r = _HandlerResult(reason=str(obj.get("reason") or "Condition not met."),
                       impossible=bool(obj.get("impossible", False)))
    if event in BLOCKING_EVENTS:
        r.block = True
        r.deny = event == "PreToolUse"
    else:
        r.context = r.reason
    return r


# --------------------------------------------------------------------------- dispatch

def base_payload(agent, event: str, *, session_id: str = "", speaker_id: int = 0,
                 speaker_is_owner: bool = False, transport: str = "",
                 turn_kind: str = "") -> dict:
    """Common input fields every hook gets (docs/hooks.md "Input JSON")."""
    agent_dir = os.path.dirname(getattr(agent, "path", "") or "")
    transcript = ""
    if session_id and agent_dir:
        transcript = os.path.join(agent_dir, "conversations", f"{session_id}.jsonl")
    return {
        "hook_event_name": event,
        "agent_id": str(getattr(agent, "id", "")),
        "session_id": session_id,
        "transcript_path": transcript,
        "cwd": agent_dir,
        "speaker_id": int(speaker_id or 0),
        "speaker_is_owner": bool(speaker_is_owner),
        "transport": transport,
        "turn_kind": turn_kind,
    }


async def run_event(agent, event: str, payload: dict, match_value: str = "") -> HookOutcome:
    """Run every matching handler for `event` in parallel and merge the results.
    Never raises; an empty HookOutcome means "no hooks / nothing to do"."""
    outcome = HookOutcome()
    try:
        handlers = handlers_for(agent, event, match_value)
    except Exception as e:
        print_ts(f"{COLOR_YELLOW}hooks: loading {event} hooks failed: {e}{COLOR_END}",
                 agent=getattr(agent, "id", ""))
        return outcome
    if not handlers:
        return outcome

    async def _one(h: dict) -> _HandlerResult:
        try:
            if h.get("type") == "command":
                return await _run_command(h, event, payload, agent)
            return await _run_prompt(h, event, payload, agent)
        except Exception as e:  # belt and braces: a hook must never break a turn
            return _HandlerResult(error=f"{h.get('type')} hook crashed: {e}")

    results = await asyncio.gather(*(_one(h) for h in handlers))
    for r in results:
        if r.error:
            outcome.errors.append(r.error)
            print_ts(f"{COLOR_YELLOW}hooks: {event} non-blocking error: {r.error}{COLOR_END}",
                     agent=getattr(agent, "id", ""))
            continue
        if r.context:
            outcome.context.append(r.context)
        if r.updated_input is not None and outcome.updated_input is None:
            outcome.updated_input = r.updated_input
        if r.allow:
            outcome.allowed = True
        if r.impossible:
            outcome.impossible = True
        if r.block:
            outcome.blocked = True
            outcome.reasons.append(r.reason)
    # Precedence across hooks (Claude Code): deny beats allow.
    if outcome.blocked:
        outcome.allowed = False
        outcome.updated_input = None
    print_ts(
        f"hooks: {event}{f' [{match_value}]' if match_value else ''} ran {len(handlers)} "
        f"handler(s): {'BLOCKED' if outcome.blocked else 'ok'}"
        f"{f', {len(outcome.errors)} error(s)' if outcome.errors else ''}",
        agent=getattr(agent, "id", ""),
    )
    return outcome


def has_hooks(agent, event: str) -> bool:
    """Cheap check before building a payload: any handler configured for event?"""
    if hooks_disabled():
        return False
    for source in (_global_hooks(), getattr(agent, "hooks", None) or {}):
        if isinstance(source, dict) and source.get(event):
            return True
    return False


# --------------------------------------------------------------------------- Stop loop guard

@dataclass
class StopState:
    """Per-turn Stop-hook bookkeeping (Claude Code's stop_hook_active + 8 cap)."""
    continuations: int = 0

    @property
    def active(self) -> bool:
        return self.continuations > 0

    def capped(self) -> bool:
        return self.continuations >= STOP_CONTINUATION_CAP


def stop_feedback(outcome: HookOutcome) -> str:
    """The [FRAMEWORK] note fed back to the model when a Stop hook keeps it going."""
    parts = []
    if outcome.blocked:
        parts.append(outcome.reason)
    parts.extend(c for c in outcome.context if c)
    body = "\n".join(p for p in parts if p).strip() or "A Stop hook says the task isn't finished."
    return ("[FRAMEWORK]: A Stop hook did not let this turn end yet. Keep working; "
            f"the hook said:\n{body}")


async def evaluate_stop(agent, state: StopState, payload: dict, last_text: str) -> str | None:
    """Decide whether Stop hooks let the turn end. Returns the [FRAMEWORK]
    feedback to feed the model when a hook keeps the turn going, or None to let
    it end. Bumps state.continuations on a block. A STAY_SILENT reply, the
    continuation cap, an impossible:true answer, or no hooks all return None."""
    if state is None or (last_text or "").strip() == "STAY_SILENT" or not has_hooks(agent, "Stop"):
        return None
    if state.capped():
        print_ts(f"{COLOR_YELLOW}hooks: Stop kept the turn going {state.continuations} times; "
                 f"cap reached, letting it end{COLOR_END}", agent=getattr(agent, "id", ""))
        return None
    payload = dict(payload)
    payload.update({"stop_hook_active": state.active, "last_assistant_message": last_text or ""})
    outcome = await run_event(agent, "Stop", payload)
    if not outcome.blocked or outcome.impossible:
        return None
    state.continuations += 1
    return stop_feedback(outcome)
