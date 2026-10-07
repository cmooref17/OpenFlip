"""spawn_subagents — fan scoped tasks out to parallel, ephemeral worker turns.

Steps 3-4 of the subagent design, on top of the foundation
in runtime.py (`run_synthetic_turn(result_future=...)`) and session.py
(`make_subagent_session`). Mirrors Claude Code's Task tool:

  * one call → N workers run IN PARALLEL, each in its own fresh conversation
    (`internal:subagent-<uuid>`) on the SAME AgentRunner as the caller;
  * a worker gets a restricted tool set, a (cheaper) model and a reasoning
    effort chosen by the owner via /toolset, never by the model;
  * only the worker's final text comes back; its conversation is deleted
    (memory + disk) the moment the task settles, so nothing it did can leak
    into or pollute the orchestrator's context;
  * a worker session is `is_owner=False`, so it can never call this tool
    again (the gate below is on the SESSION's owner flag, not the speaker
    id) — no nested delegation.

Runner access: `registry.RUNNERS[CURRENT_AGENT.id]`, the same lookup
talk_to_agent uses. Conversation keying mirrors `TransportChannel.id`
(transports/channel_shim.py): a non-numeric transport_id hashes to a stable
int, which is what `_active_turns` / `conversations` are keyed on for the
worker, so `_hard_interrupt` and `reset_conversation` hit the right slot.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from ._base import tool, ToolResult, TOOL_REGISTRY
from .. import tool_settings as ts
from .. import config_global as _cfg
from ..tool_settings import SUBAGENT_DENIED_TOOLS, parse_tool_list
from ..session import make_subagent_session
from ..session_overrides import provider_for_model
from ..utils import print_ts, COLOR_YELLOW, COLOR_RED, COLOR_END

SETTINGS = "spawn_subagents"

# Running workers per agent id, across every concurrent spawn_subagents call.
# Reserved synchronously (no await between the read and the add) in the tool
# body, released per task in its `finally`.
_INFLIGHT: dict[str, int] = {}

_TOTAL_OUTPUT_CAP = 12_000
_MIN_PER_TASK_CAP = 600
_LABEL_MAX = 60

WORKER_PREAMBLE = (
    "You are a subagent worker spawned by an orchestrating agent to complete ONE "
    "scoped task. You have a fresh, isolated context: you know nothing about the "
    "conversation that spawned you beyond the task text below, and only your final "
    "reply is returned to the orchestrator. Work autonomously with the tools you "
    "have — you CANNOT ask questions or request clarification (nobody will answer), "
    "so make reasonable assumptions and state them. Do not address a human and do "
    "not try to message anyone. When you are done, reply with a concise, "
    "self-contained summary of your results (findings, facts, excerpts, paths, "
    "URLs — whatever the orchestrator needs to use your work), and nothing else."
)


# ----------------------------------------------------------------- helpers

def inflight_count(agent_id: str) -> int:
    return int(_INFLIGHT.get(agent_id, 0))


def _reserve_inflight(agent_id: str, n: int) -> None:
    _INFLIGHT[agent_id] = inflight_count(agent_id) + n


def _release_inflight(agent_id: str) -> None:
    left = inflight_count(agent_id) - 1
    if left <= 0:
        _INFLIGHT.pop(agent_id, None)
    else:
        _INFLIGHT[agent_id] = left


def worker_conv_key(session) -> int:
    """In-memory conversation / active-turn key for a worker session. Must
    stay identical to how transports/channel_shim.TransportChannel derives
    `.id` from `session.transport_id` (runtime keys `_active_turns` and
    `conversations` on that int for an unlinked session)."""
    try:
        return int(session.transport_id)
    except (TypeError, ValueError):
        return abs(hash(session.transport_id)) % (2**31)


def compute_grants(requested: list[str] | None, ceiling: list[str]) -> list[str]:
    """(requested or ceiling) ∩ ceiling − SUBAGENT_DENIED_TOOLS, order kept."""
    allowed = [t for t in ceiling if t not in SUBAGENT_DENIED_TOOLS]
    if not requested:
        return list(allowed)
    allowed_set = set(allowed)
    out: list[str] = []
    for name in requested:
        if name in allowed_set and name not in out:
            out.append(name)
    return out


def validate_worker_model(model: str, agent_provider: str) -> str:
    """Call-time check: the model must be a config.json `models` entry (when
    that block exists) on the CALLING agent's provider. Returns "" when ok,
    else the reason."""
    name = str(model or "").strip()
    if not name:
        return "model is empty"
    bare = name.split("/", 1)[1] if "/" in name else name
    models = _cfg.get_config().get("models") or {}
    entry = models.get(bare)
    if models and entry is None:
        return f"`{bare}` is not in config.json `models`"
    provider = str((entry or {}).get("provider") or "").strip() or provider_for_model(name)
    if provider != agent_provider:
        return f"`{bare}` is a {provider} model but this agent runs {agent_provider}"
    return ""


def _clean_label(raw: Any, idx: int) -> str:
    label = " ".join(str(raw or "").split())
    if not label:
        label = f"task-{idx + 1}"
    return label[:_LABEL_MAX]


def _normalize_tasks(tasks: Any) -> tuple[list[dict], str]:
    """Coerce the model's `tasks` argument into a list of dicts. Returns
    (tasks, error)."""
    if isinstance(tasks, str):
        try:
            tasks = json.loads(tasks)
        except Exception:
            return [], "tasks must be a JSON array of {prompt, tools?, label?, model?} objects"
    if isinstance(tasks, dict):
        tasks = [tasks]
    if not isinstance(tasks, list) or not tasks:
        return [], "tasks must be a non-empty array of {prompt, tools?, label?, model?} objects"
    out: list[dict] = []
    for i, t in enumerate(tasks):
        if isinstance(t, str):
            t = {"prompt": t}
        if not isinstance(t, dict):
            return [], f"tasks[{i}] must be an object with a `prompt`"
        prompt = t.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return [], f"tasks[{i}] has no non-empty `prompt`"
        tools = t.get("tools")
        if tools is not None and not isinstance(tools, (list, tuple, str)):
            return [], f"tasks[{i}].tools must be a list of tool names"
        out.append({
            "prompt": prompt.strip(),
            "tools": parse_tool_list(tools) if tools else None,
            "label": _clean_label(t.get("label"), i),
            "model": str(t.get("model") or "").strip(),
        })
    return out, ""


def _retrieve_silently(fut: asyncio.Future) -> None:
    """Done-callback: mark a settled exception as retrieved so an abandoned
    worker future (timeout path) never logs 'exception was never retrieved'."""
    if not fut.cancelled():
        try:
            fut.exception()
        except Exception:
            pass


def format_results(results: list[tuple[str, bool, str]], cap: int = _TOTAL_OUTPUT_CAP) -> str:
    """'### <label> — ok|error\\n<body>' per task, bodies truncated so the
    whole report stays within `cap` characters."""
    n = max(1, len(results))
    heads = [f"### {label} — {'ok' if ok else 'error'}\n" for label, ok, _ in results]
    budget = cap - sum(len(h) + 2 for h in heads)
    per = max(_MIN_PER_TASK_CAP, budget // n)
    chunks: list[str] = []
    for head, (_, _, body) in zip(heads, results):
        body = (body or "").strip() or "(no output)"
        if len(body) > per:
            body = body[: per - 16].rstrip() + "\n…[truncated]"
        chunks.append(head + body)
    text = "\n\n".join(chunks)
    if len(text) > cap:
        text = text[: cap - 16].rstrip() + "\n…[truncated]"
    return text


# -------------------------------------------------------------------- tool

@tool
async def spawn_subagents(tasks: list) -> ToolResult:
    """Fan a batch of self-contained tasks out to parallel subagent workers and get back only their summaries. ONE call runs ALL the tasks at once (up to max_parallel), each in a brand-new isolated worker with a fresh context that knows NOTHING about this conversation — not the user, not earlier messages, not files you looked at — so every prompt must be fully self-contained: include every fact, path, URL, constraint and the exact shape of answer you want. Workers run on the configured cheaper worker model with a read-only tool set (see the spawn_subagents settings; tools you request are intersected with that ceiling), cannot ask questions, cannot message anyone, cannot write files or memory, and cannot spawn workers themselves. When all finish you receive one summary per task (ok or error); you then verify and assemble the results yourself. Plan first, delegate the independent read/search chunks, keep the synthesis.

    Args:
        tasks: Array of task objects, one worker each. {"prompt": "<complete standalone instructions>", "label": "<short name, optional>", "tools": ["web_search", ...] (optional subset of the allowed worker tools), "model": "<model id, optional, defaults to the configured worker_model>"}.
    """
    from ..tool_executor import CURRENT_SESSION, CURRENT_AGENT
    from ..registry import RUNNERS

    # 1. Owner gate on the SESSION (not the speaker id): a worker session is
    #    is_owner=False, so a worker calling this is refused → no recursion.
    sess = CURRENT_SESSION.get(None)
    if sess is None or not bool(getattr(sess, "is_owner", False)):
        return ToolResult.fail(
            "spawn_subagents is only available on the owner's own turns "
            "(this session is not an owner session; subagent workers can never spawn workers)."
        )
    agent = CURRENT_AGENT.get(None)
    if agent is None:
        return ToolResult.fail("spawn_subagents: no current agent in this context.")
    runner = RUNNERS.get(agent.id)
    if runner is None:
        return ToolResult.fail(f"spawn_subagents: agent '{agent.id}' has no running AgentRunner.")

    task_list, err = _normalize_tasks(tasks)
    if err:
        return ToolResult.fail(err)

    settings = ts.get_all(SETTINGS)
    max_parallel = int(settings.get("max_parallel") or 4)
    max_inflight = int(settings.get("max_inflight_per_agent") or 6)
    timeout_s = float(settings.get("timeout_s") or 300)
    worker_model = str(settings.get("worker_model") or "").strip()
    worker_effort = str(settings.get("worker_effort") or "").strip()
    ceiling = parse_tool_list(settings.get("allowed_tools"))

    # 3. Caps.
    if len(task_list) > max_parallel:
        return ToolResult.fail(
            f"spawn_subagents: {len(task_list)} tasks requested but max_parallel is {max_parallel}. "
            "Split into several calls (sequential batches) or merge tasks."
        )
    agent_provider = str(getattr(agent, "provider", "") or "ollama")
    results: list[tuple[str, bool, str] | None] = [None] * len(task_list)

    # Inflight reservation — synchronous: nothing awaits between the count
    # read and the add, so two concurrent calls can't both see the same room.
    room = max_inflight - inflight_count(agent.id)
    launch_idx: list[int] = []
    for i in range(len(task_list)):
        if len(launch_idx) < room:
            launch_idx.append(i)
        else:
            results[i] = (
                task_list[i]["label"], False,
                f"not started: max_inflight_per_agent ({max_inflight}) reached — "
                f"{inflight_count(agent.id)} workers already running for this agent; retry later",
            )
    _reserve_inflight(agent.id, len(launch_idx))

    async def _run_one(i: int) -> tuple[str, bool, str]:
        task = task_list[i]
        label = task["label"]
        # 2. Grants: (requested or ceiling) ∩ ceiling − denied.
        grants = compute_grants(task["tools"], ceiling)
        if task["tools"]:
            stripped = [t for t in task["tools"] if t not in grants]
            if stripped:
                print_ts(
                    f"{COLOR_YELLOW}[subagent {label}] stripped tools not in the worker ceiling / denied: "
                    f"{', '.join(stripped)}{COLOR_END}", agent=agent.id,
                )
        model = task["model"] or worker_model
        task_uuid = str(uuid.uuid4())
        session = make_subagent_session(agent.id, task_uuid, grants)
        conv_key = worker_conv_key(session)
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        fut.add_done_callback(_retrieve_silently)
        try:
            # 4. Fresh worker conversation + per-session overrides (never
            #    persisted — the conversation is deleted in `finally`).
            why = validate_worker_model(model, agent_provider)
            if why:
                return label, False, f"invalid worker model: {why}"
            conv = runner.get_conversation(
                conv_key, session.conversation_id, native_key=session.conversation_id,
            )
            ok, msg = conv.set_override("model", model, persist=False)
            if not ok:
                return label, False, f"invalid worker model `{model}`: {msg}"
            if worker_effort:
                ok_e, msg_e = conv.set_override("effort", worker_effort, persist=False)
                if not ok_e:
                    print_ts(f"{COLOR_YELLOW}[subagent {label}] effort not applied: {msg_e}{COLOR_END}", agent=agent.id)
            ok_m, msg_m = conv.set_override("memory", False, persist=False)
            if not ok_m:
                print_ts(f"{COLOR_YELLOW}[subagent {label}] memory off not applied: {msg_m}{COLOR_END}", agent=agent.id)

            prompt = f"{WORKER_PREAMBLE}\n\n## Task\n{task['prompt']}"
            print_ts(
                f"[subagent {label}] launching worker {task_uuid[:8]} model={model} "
                f"tools={','.join(grants) or '-'}", agent=agent.id,
            )
            await runner.run_synthetic_turn(
                session, prompt,
                silent=True,
                log_tag=f"[subagent {label}] ",
                result_future=fut,
            )
            # 5. Wait with a timeout. `asyncio.wait` (not wait_for) never
            #    cancels `fut` itself — same semantics as wait_for(shield(fut))
            #    without shield's "CancelledError in shielded future" log when
            #    the interrupted turn later settles the abandoned future. If
            #    the TOOL is cancelled (/stop), CancelledError propagates here.
            try:
                done, _pending = await asyncio.wait({fut}, timeout=timeout_s)
            except asyncio.CancelledError:
                try:
                    runner._hard_interrupt(conv_key)
                except Exception:
                    pass
                raise
            if not done:
                try:
                    runner._hard_interrupt(conv_key)
                except Exception as _ie:
                    print_ts(f"{COLOR_RED}[subagent {label}] interrupt failed: {_ie}{COLOR_END}", error=True, agent=agent.id)
                return label, False, f"timed out after {int(timeout_s)}s; worker interrupted"
            try:
                text = fut.result()
            except asyncio.CancelledError:
                # The worker's turn was interrupted under us (e.g. /stop or a
                # reset on its conversation) — that is this task's failure,
                # not a cancellation of the orchestrator's tool call.
                return label, False, "worker turn was interrupted before it finished"
            text = (text or "").strip()
            if not text:
                return label, False, "worker finished without a reply"
            return label, True, text
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return label, False, f"{type(e).__name__}: {e}"
        finally:
            # 6. Throw the worker conversation away (interrupt + drop queued +
            #    pop from memory + delete .jsonl/.meta.json) and free the slot.
            try:
                runner.reset_conversation(conv_key, session.conversation_id)
                # reset_conversation keeps a pre_reset .bak.jsonl; a throwaway
                # worker has nothing worth recovering, so sweep it too.
                import glob as _glob, os as _os
                from .._conversation_io import conversation_path as _cp
                _jl = _cp(_os.path.dirname(runner.agent.path), session.conversation_id)
                for _bak in _glob.glob(_glob.escape(_jl) + ".pre_reset_*.bak.jsonl"):
                    _os.remove(_bak)
            except Exception as _ce:
                print_ts(f"{COLOR_RED}[subagent {label}] cleanup failed: {_ce}{COLOR_END}", error=True, agent=agent.id)
            _release_inflight(agent.id)

    if launch_idx:
        gathered = await asyncio.gather(*(_run_one(i) for i in launch_idx), return_exceptions=True)
        for i, r in zip(launch_idx, gathered):
            if isinstance(r, BaseException):
                if isinstance(r, asyncio.CancelledError):
                    raise r
                results[i] = (task_list[i]["label"], False, f"{type(r).__name__}: {r}")
            else:
                results[i] = r

    final = [r for r in results if r is not None]
    n_ok = sum(1 for _, ok, _ in final if ok)
    # 7. Report.
    text = format_results(final)
    return ToolResult(
        text=text,
        model_feedback=f"{n_ok}/{len(final)} workers ok.\n\n{text}",
    )


# Worker summaries are intermediate material for the orchestrator to verify
# and assemble; never post them raw to the channel.
if "spawn_subagents" in TOOL_REGISTRY:
    TOOL_REGISTRY["spawn_subagents"].silent_to_discord = True
