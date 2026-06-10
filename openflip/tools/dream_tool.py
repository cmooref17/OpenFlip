"""Dream — memory consolidation tool.

A "dream" is a periodic consolidation pass over an agent's memory. Over time
the daily logs (agents/<id>/memory/YYYY-MM-DD.md) accumulate raw events and
MEMORY.md (the curated core knowledge) grows unbounded — contradicted facts
are never pruned, relative dates ("yesterday") rot into ambiguity, and the
file drifts past any sane size.

This tool does NOT call an LLM itself. Instead it gathers the full memory
surface (MEMORY.md + every daily log) and hands the agent a 4-phase
consolidation prompt as the tool result. The AGENT then does the actual
distillation reasoning and writes the consolidated result back by calling
`update_core_memory` (see memory.py). This mirrors search_memory's pattern:
the tool gathers + presents, the model reasons.

Phases the agent is asked to perform:
    1. Orient    — read MEMORY.md + all daily logs (provided below).
    2. Consolidate — distill into durable facts; convert relative dates to
                     absolute dates BEFORE writing.
    3. Prune     — DELETE facts that were later contradicted.
    4. Cap       — keep MEMORY.md under max_memory_chars.

Triggering: manual /dream command or a direct dream() tool call. The per-agent
`dream.enabled` flag gates only AUTO-fire (which is wired up separately); it
does NOT gate manual invocation.
"""
from __future__ import annotations

import os
import time

from ._base import tool, ToolResult
from .memory import _memory_md_path, _memory_dir, _maybe_migrate

# Fallback cap when an agent has no dream config / max_memory_chars set.
_DEFAULT_MAX_MEMORY_CHARS = 25000


def _get_agent():
    """Return the current Agent (or None) from the contextvar tool_executor sets."""
    from ..tool_executor import CURRENT_AGENT
    return CURRENT_AGENT.get(None)


def _read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (FileNotFoundError, OSError):
        return ""


def _gather_memory(agent_dir: str) -> tuple[str, list[tuple[str, str]]]:
    """Return (MEMORY.md content, [(date_label, daily_log_content), ...]).

    Daily logs are returned oldest-first so the agent reads memory in
    chronological order — important for the Prune phase, where a later entry
    can contradict an earlier one.
    """
    core = _read_file(_memory_md_path(agent_dir))

    dailies: list[tuple[str, str]] = []
    mem_dir = _memory_dir(agent_dir)
    if os.path.isdir(mem_dir):
        names = sorted(f for f in os.listdir(mem_dir) if f.endswith(".md"))
        for fname in names:
            content = _read_file(os.path.join(mem_dir, fname))
            if content.strip():
                dailies.append((fname.removesuffix(".md"), content))
    return core, dailies


def _build_consolidation_prompt(agent_dir: str, max_memory_chars: int, today: str) -> str:
    """Assemble the full memory dump + 4-phase consolidation instructions."""
    core, dailies = _gather_memory(agent_dir)

    parts: list[str] = []
    parts.append(
        f"DREAM — memory consolidation. Today is {today} (absolute). Use this "
        f"date to resolve any relative references ('yesterday', 'last week', "
        f"'a few days ago') into absolute YYYY-MM-DD dates BEFORE you write."
    )

    parts.append("=== CURRENT CORE MEMORY (MEMORY.md) ===")
    parts.append(core.strip() if core.strip() else "(empty — no core memory yet)")

    if dailies:
        parts.append("=== DAILY LOGS (oldest first) ===")
        for label, content in dailies:
            parts.append(f"--- {label} ---\n{content.strip()}")
    else:
        parts.append("=== DAILY LOGS ===\n(none)")

    parts.append(
        "=== YOUR TASK ===\n"
        "Consolidate the above into a single, clean MEMORY.md. Work through "
        "four phases:\n"
        "  1. ORIENT — you've just read MEMORY.md and every daily log above.\n"
        "  2. CONSOLIDATE — distill the raw events into durable, standalone "
        "facts. Merge duplicates. Convert every relative date to an absolute "
        f"YYYY-MM-DD date (today is {today}).\n"
        "  3. PRUNE — DELETE any fact that a later entry contradicted or "
        "superseded. Keep only what is still true. Do not keep both sides of a "
        "contradiction.\n"
        f"  4. CAP — keep the final MEMORY.md under {max_memory_chars} "
        "characters. If it would exceed that, drop the least important / "
        "least durable details first.\n\n"
        "When you have the consolidated text ready, call update_core_memory() "
        "with the COMPLETE new MEMORY.md content (it overwrites the whole "
        "file). Do not summarize your changes to the user unless asked — just "
        "perform the consolidation and write it back."
    )

    return "\n\n".join(parts)


@tool
async def dream() -> ToolResult:
    """Consolidate your long-term memory. Reviews your core memory (MEMORY.md) and all daily logs, then returns a 4-phase consolidation plan (orient, consolidate, prune contradicted facts, cap size). After calling this, reason over the result and call update_core_memory() with the cleaned-up MEMORY.md. Use this periodically when your memory has grown messy, contradictory, or large.
    """
    agent = _get_agent()
    if not agent:
        return ToolResult.fail("No agent context available for dream().")

    agent_dir = os.path.dirname(agent.path)
    _maybe_migrate(agent_dir)

    dream_cfg = getattr(agent, "dream", None) or {}
    try:
        max_chars = int(dream_cfg.get("max_memory_chars", _DEFAULT_MAX_MEMORY_CHARS))
    except (ValueError, TypeError):
        max_chars = _DEFAULT_MAX_MEMORY_CHARS
    if max_chars <= 0:
        max_chars = _DEFAULT_MAX_MEMORY_CHARS

    today = time.strftime("%Y-%m-%d")
    payload = _build_consolidation_prompt(agent_dir, max_chars, today)

    try:
        from .. import events_log as _events_log
        _events_log.log_event(agent.id, "dream", target="core", max_chars=max_chars)
    except Exception:
        pass

    return ToolResult(model_feedback=payload)
