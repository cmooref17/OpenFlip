"""Dream — memory consolidation tool.

A "dream" is a periodic consolidation pass over an agent's memory. Over time
the daily logs (agents/<id>/memory/YYYY-MM-DD.md) accumulate raw events and
MEMORY.md (the curated core knowledge) grows unbounded — contradicted facts
are never pruned, relative dates ("yesterday") rot into ambiguity, and the
file drifts past any sane size.

This tool does NOT call an LLM itself. Instead it gathers the memory surface
(MEMORY.md in full + every daily log) and hands the agent a 4-phase
consolidation prompt as the tool result. The AGENT then does the actual
distillation reasoning and writes the consolidated result back by calling
`update_core_memory` (see memory.py). This mirrors search_memory's pattern:
the tool gathers + presents, the model reasons.

Payload budget — why a SPILL FILE, not an inline dump:
    MEMORY.md is both the INPUT and the OUTPUT of dream, so an inline-only
    payload degrades monotonically: the more the agent has learned, the larger
    MEMORY.md grows, the less daily-log history fits under a fixed char budget,
    and the worse the Prune phase can see. To remove that ceiling entirely, the
    tool writes the FULL corpus (MEMORY.md + ALL daily logs, nothing dropped)
    to a per-agent spill file in the system temp dir and returns its absolute
    path plus a compact index. The agent reads slices with read_file. A bounded
    inline PREVIEW of the newest logs is still included so the common
    small-memory case needs no extra read_file round-trip; when the preview
    omits or truncates anything, the omission note points at the spill file
    (the data is relocated, never lost).

Phases the agent is asked to perform:
    1. Orient    — read MEMORY.md + the daily logs (inline preview + spill file).
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
import tempfile
import time

from ._base import tool, ToolResult
from ..utils import safe_filename
from .memory import _memory_md_path, _memory_dir, _maybe_migrate

# Fallback cap when an agent has no dream config / max_memory_chars set.
_DEFAULT_MAX_MEMORY_CHARS = 25000

# Ceiling for the assembled tool result. MUST stay under runtime.py's
# _TOOL_RESULT_MAX_CHARS (100_000) ingestion cap. The full corpus no longer
# competes for this budget — it lives in the spill file — so this now bounds
# only MEMORY.md (inlined in full) + the index + instructions + the inline
# daily-log preview.
_MAX_PAYLOAD_CHARS = 90_000

# The inline daily-log preview is deliberately a PREVIEW, not the whole corpus
# (that is in the spill file). Cap it well under _MAX_PAYLOAD_CHARS so a large
# MEMORY.md and the index always have room, while still comfortably inlining the
# newest logs for the common small-memory case.
_MAX_INLINE_DAILY_CHARS = 50_000

# When the newest log that doesn't fit whole still has at least this much inline
# budget left, inline its TAIL (newest entries) with a truncation marker rather
# than dropping it and everything older. Below this, a tail isn't worth it.
_MIN_TAIL_CHARS = 600


class MemoryTooLargeError(Exception):
    """Raised when the mandatory inline parts (MEMORY.md + index + instructions)
    exceed the payload budget. Much harder to hit now that the daily-log corpus
    spills to a file — this only fires for a genuinely absurd MEMORY.md."""


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


def _spill_corpus(agent_id: str, core: str, dailies: list[tuple[str, str]], today: str) -> tuple[str, int]:
    """Write the FULL memory corpus (MEMORY.md + every daily log, nothing
    dropped) to a per-agent spill file and return (absolute_path, total_chars).

    Written into ``tempfile.gettempdir()`` — NOT a hardcoded "/tmp". The file
    tool's universal read fallback keys off ``tempfile.gettempdir()`` (see
    files.py); on macOS that is ``/var/folders/...``, so a hardcoded "/tmp"
    would leave the agent ACL-denied from reading its own spill file. Named
    per-agent so concurrent agents can't collide, and written atomically
    (tempfile + os.replace) per the repo's save_json convention.
    """
    sections: list[str] = [
        f"# DREAM CORPUS — agent {agent_id}",
        (
            f"# Generated {today}. Full memory surface below: MEMORY.md followed "
            f"by every daily log, oldest first. Nothing is dropped — this file is "
            f"the complete corpus the dream index points at."
        ),
        (
            f"=== MEMORY.md ({len(core.strip()):,} chars) ===\n\n"
            + (core.strip() if core.strip() else "(empty — no core memory yet)")
        ),
    ]
    for label, content in dailies:
        c = content.strip()
        sections.append(f"=== DAILY LOG {label} ({len(c):,} chars) ===\n\n{c}")
    body = "\n\n".join(sections)

    tmp_dir = tempfile.gettempdir()
    safe_id = safe_filename(agent_id) or "agent"
    final_path = os.path.join(tmp_dir, f"dream-{safe_id}.md")
    fd, tmp_path = tempfile.mkstemp(prefix=f".dream-{safe_id}.", suffix=".tmp", dir=tmp_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(body)
        os.replace(tmp_path, final_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return final_path, len(body)


def _build_index(core: str, dailies: list[tuple[str, str]], spill_path: str) -> str:
    """Compact index of the spilled corpus: per-file char counts, date range,
    total, and the absolute spill-file path the agent reads slices from."""
    core_chars = len(core.strip())
    lines = [
        "=== MEMORY CORPUS INDEX ===",
        (
            f"The COMPLETE corpus (MEMORY.md + all {len(dailies)} daily log(s), "
            f"nothing dropped) has been written to a single file:"
        ),
        f"    {spill_path}",
        (
            f'Read any part of it with read_file(path="{spill_path}", ...). The '
            f"inline preview further below may omit or truncate older logs to save "
            f"space, but EVERY byte is in that file. Individual daily logs are also "
            f'readable directly via read_memory("YYYY-MM-DD").'
        ),
        "",
        f"MEMORY.md — {core_chars:,} chars",
    ]
    if dailies:
        counts = [(label, len(content.strip())) for label, content in dailies]
        total_daily = sum(c for _, c in counts)
        lines.append(
            f"Daily logs — {len(dailies)} file(s), {counts[0][0]} through "
            f"{counts[-1][0]}, {total_daily:,} chars total:"
        )
        for label, c in counts:
            lines.append(f"    {label}.md — {c:,} chars")
    else:
        lines.append("Daily logs — none.")
    return "\n".join(lines)


def _tail_block(label: str, content: str, budget: int) -> str:
    """Render a tail-truncated inline block for a single daily log: keep the
    NEWEST entries (the tail) that fit `budget`, with an explicit marker that
    earlier entries live in the spill file. Snaps the cut to a line boundary."""
    content = content.strip()
    header = (
        f"--- {label} (TRUNCATED — earlier entries omitted inline; full log in "
        f"the spill file) ---\n"
    )
    marker = "… [earlier entries of this log omitted — read the spill file] …\n"
    avail = budget - len(header) - len(marker) - 2  # -2 for the "\n\n" joiner
    if avail <= 0:
        return (header + marker).rstrip()
    tail = content[-avail:]
    # Snap to the next line boundary so the tail doesn't start mid-line.
    nl = tail.find("\n")
    if nl != -1 and nl < len(tail) - 1:
        tail = tail[nl + 1:]
    return header + marker + tail


def _build_consolidation_prompt(agent_id: str, agent_dir: str, max_memory_chars: int, today: str) -> str:
    """Spill the full corpus to a file, then assemble the index + a bounded
    inline preview + the 4-phase consolidation instructions.

    The full corpus (MEMORY.md + ALL daily logs) is written to a spill file and
    referenced by absolute path, so NOTHING is ever permanently dropped. Inline,
    MEMORY.md and the instructions are always included in full; the newest daily
    logs are inlined as a PREVIEW under a reduced budget. When a single newest
    log overflows the remaining inline budget its TAIL is inlined with a
    truncation marker (never skipping it and everything older); anything not
    inlined is reachable via the spill file, and the omission note says so.

    Raises MemoryTooLargeError if the mandatory inline parts alone exceed the
    budget (the corpus is still spilled first, so the error can point at it).
    """
    core, dailies = _gather_memory(agent_dir)

    # Spill FIRST — even the too-large backstop path can then point at the file.
    spill_path, _corpus_chars = _spill_corpus(agent_id, core, dailies, today)
    index_section = _build_index(core, dailies, spill_path)

    header = (
        f"DREAM — memory consolidation. Today is {today} (absolute). Use this "
        f"date to resolve any relative references ('yesterday', 'last week', "
        f"'a few days ago') into absolute YYYY-MM-DD dates BEFORE you write."
    )
    core_section = (
        "=== CURRENT CORE MEMORY (MEMORY.md) ===\n\n"
        + (core.strip() if core.strip() else "(empty — no core memory yet)")
    )
    task_section = (
        "=== YOUR TASK ===\n"
        "Consolidate the memory surface into a single, clean MEMORY.md. Work "
        "through four phases:\n"
        "  1. ORIENT — read MEMORY.md and the daily logs. The inline preview "
        "below may be partial; the FULL corpus is in the spill file named in the "
        "index above — read it (or the relevant daily logs) before pruning.\n"
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

    # Mandatory inline chars: header + core + index + task, plus slack for the
    # "\n\n" joiners, the daily-logs section header, and the (variable-length)
    # omission note. 1000 chars generously covers all of those.
    fixed_chars = len(header) + len(core_section) + len(index_section) + len(task_section) + 1000
    if fixed_chars > _MAX_PAYLOAD_CHARS:
        raise MemoryTooLargeError(
            f"Core memory is too large to inline in one dream pass: MEMORY.md "
            f"({len(core.strip()):,} chars) plus the index and instructions totals "
            f"~{fixed_chars:,} chars, over the {_MAX_PAYLOAD_CHARS:,}-char payload "
            f"budget. The full corpus was still written to {spill_path} — read it "
            f"and consolidate from there, or trim MEMORY.md before dreaming again."
        )

    # Inline daily-log PREVIEW: walk NEWEST-first, keep a contiguous run of the
    # most recent logs that fits a REDUCED inline budget. included_start is the
    # index (in the oldest-first `dailies` list) of the first log kept. If the
    # first log that doesn't fit whole still has room for a useful tail, inline
    # that tail (truncated) rather than dropping it and everything older.
    blocks = [f"--- {label} ---\n{content.strip()}" for label, content in dailies]
    inline_budget = min(_MAX_PAYLOAD_CHARS - fixed_chars, _MAX_INLINE_DAILY_CHARS)
    if inline_budget < 0:
        inline_budget = 0

    included_start = len(dailies)
    truncated_idx: int | None = None
    truncated_block: str | None = None
    budget = inline_budget
    for i in range(len(blocks) - 1, -1, -1):
        cost = len(blocks[i]) + 2  # +2 for the "\n\n" joiner
        if cost <= budget:
            budget -= cost
            included_start = i
            continue
        # Doesn't fit whole. Inline its TAIL if there's still useful room,
        # instead of skipping it (and, historically, everything older too).
        if budget >= _MIN_TAIL_CHARS:
            truncated_idx = i
            truncated_block = _tail_block(dailies[i][0], dailies[i][1], budget)
            included_start = i
        break

    parts: list[str] = [header, core_section, index_section]

    # Omission note — now the data is RELOCATED (spill file), not lost.
    fully_omitted = dailies[:included_start]
    if dailies and (fully_omitted or truncated_idx is not None):
        if fully_omitted:
            omitted_chars = sum(len(b) for b in blocks[:included_start])
            note = (
                f"!!! PARTIAL INLINE VIEW — {len(fully_omitted)} older daily "
                f"log(s) ({fully_omitted[0][0]} through {fully_omitted[-1][0]}, "
                f"{omitted_chars:,} chars) are NOT shown inline below. They are "
                f"NOT lost — read them from the spill file in the index above "
                f"({spill_path})."
            )
        else:
            note = (
                f"!!! PARTIAL INLINE VIEW — the oldest inlined log was truncated "
                f"to fit; its earlier entries are in the spill file ({spill_path})."
            )
        note += (
            " Be CONSERVATIVE in the Prune phase: only delete a core-memory fact "
            "if a log you have actually read (inline OR from the spill file) "
            "contradicts it — absence from the inline preview is NOT evidence a "
            "fact is stale, since its origin may lie in the omitted range. Read "
            "the spill file before pruning if in doubt."
        )
        parts.append(note)

    if not dailies:
        parts.append("=== DAILY LOGS ===\n(none)")
    elif included_start >= len(dailies):
        parts.append(
            "=== DAILY LOGS ===\n(none fit the inline preview — read them from "
            "the spill file named in the index above)"
        )
    else:
        parts.append("=== DAILY LOGS (inline preview, oldest first) ===")
        for i in range(included_start, len(dailies)):
            parts.append(truncated_block if i == truncated_idx else blocks[i])

    parts.append(task_section)
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
    try:
        payload = _build_consolidation_prompt(agent.id, agent_dir, max_chars, today)
    except MemoryTooLargeError as e:
        return ToolResult.fail(str(e))
    except OSError as e:
        return ToolResult.fail(f"dream() could not write its corpus spill file: {e}")

    try:
        from .. import events_log as _events_log
        _events_log.log_event(agent.id, "dream", target="core", max_chars=max_chars)
    except Exception:
        pass

    return ToolResult(model_feedback=payload)
