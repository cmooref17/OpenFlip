"""Memory recall: Claude-Code-style per-turn selection of relevant topic files.

Mirrors Claude Code 2.1.280's recall supervisor (read from its bundled JS):
candidates are one line per topic file ("- filename (mtime): description",
from frontmatter); a SEPARATE small model call picks up to 5 with CC's
selector prompt; one-word queries, already-surfaced files and an exhausted
per-conversation byte budget are skipped; each chosen file is read capped
with a truncation note and injected under CC's "Retrieved for possible
relevance" prefix. Constants live in memory_recall_prompt.py.

openflip difference: the block rides on the turn's user message so the
cached system prefix stays byte-stable. Recall never breaks a turn: every
failure returns "" and logs.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

from . import memory_index as mi
from .memory_recall_prompt import (
    MAX_FILES, FILE_MAX_LINES, FILE_MAX_BYTES, SESSION_MAX_BYTES,
    SELECTOR_TIMEOUT_S, INJECT_PREFIX,
)
from .utils import print_ts

_FM_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)
_CJK_RE = re.compile(r"[\u3040-\u30FF\u3400-\u4DBF\u4E00-\u9FFF\uAC00-\uD7AF\uF900-\uFAFF]")


def _frontmatter(text: str) -> dict:
    m = _FM_RE.match(text)
    out: dict = {}
    if m:
        for line in m.group(1).splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip().strip('"')
    return out


def list_candidates(agent_dir: str) -> list[dict]:
    """Topic files, newest first: {filename, path, mtime, description}."""
    tdir = mi.topics_dir(agent_dir)
    if not os.path.isdir(tdir):
        return []
    out = []
    for name in os.listdir(tdir):
        if not name.endswith(".md"):
            continue
        path = os.path.join(tdir, name)
        try:
            st = os.stat(path)
            with open(path, "r", encoding="utf-8") as f:
                head = f.read(2048)
        except OSError:
            continue
        fm = _frontmatter(head)
        desc = fm.get("description") or mi.make_hook(mi.topic_body(head))
        mtype = fm.get("type", "") if fm.get("type", "") in mi.MEMORY_TYPES else ""
        out.append({"filename": name, "path": path, "mtime": st.st_mtime, "description": desc, "type": mtype})
    out.sort(key=lambda c: c["mtime"], reverse=True)
    return out


def format_candidates(cands: list[dict]) -> str:
    """CC's `nwt` line shape: "- [type] filename (ISO mtime): description"."""
    lines = []
    for c in cands:
        prefix = f"[{c['type']}] " if c.get("type") else ""
        line = f"- {prefix}{c['filename']} ({time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(c['mtime']))})"
        if c.get("description"):
            line += f": {c['description']}"
        lines.append(line)
    return "\n".join(lines)


def should_skip_query(query: str) -> bool:
    """CC skips empty / single-word queries, except CJK (no word spaces)."""
    q = (query or "").strip()
    if not q:
        return True
    if re.search(r"\s", q):
        return False
    return not _CJK_RE.search(q)


def read_capped(path: str) -> tuple[str, int]:
    """Body capped at FILE_MAX_LINES / FILE_MAX_BYTES with CC's truncation
    note. Returns (content, bytes_used)."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    lines = text.splitlines()
    by_lines = len(lines) > FILE_MAX_LINES
    out = "\n".join(lines[:FILE_MAX_LINES]) if by_lines else text
    raw = out.encode("utf-8")
    by_bytes = len(raw) > FILE_MAX_BYTES
    if by_bytes:
        out = raw[:FILE_MAX_BYTES].decode("utf-8", "ignore")
        cut = out.rfind("\n")
        if cut > 0:
            out = out[:cut]
    if by_lines or by_bytes:
        why = f"{FILE_MAX_BYTES} byte limit" if by_bytes else f"first {FILE_MAX_LINES} lines"
        rel = f"topics/{os.path.basename(path)[:-3]}"
        out += (f"\n> This memory file was truncated ({why}). Use "
                f"read_memory(file=\"{rel}\") to view the complete file.")
    return out, len(out.encode("utf-8"))


def parse_selection(text: str, valid: set[str]) -> list[str]:
    """Selected filenames from the selector's JSON answer (invalid dropped)."""
    data = None
    for cand in (text or "", (re.search(r"\{.*\}", text or "", re.S) or [None])[0]):
        if not cand:
            continue
        try:
            data = json.loads(cand)
            break
        except Exception:
            continue
    picked = data.get("selected_memories") if isinstance(data, dict) else None
    if not isinstance(picked, list):
        return []
    out: list[str] = []
    for p in picked:
        if isinstance(p, str):
            p = re.sub(r"^\[[a-z]+\]\s+", "", p.strip())
            if p in valid and p not in out:
                out.append(p)
    return out[:MAX_FILES]


def build_block(chosen: list[tuple[str, str]]) -> str:
    """[(filename, content)] -> text appended to the turn's user message."""
    if not chosen:
        return ""
    parts = [f"<relevant-memories>\n{INJECT_PREFIX}"]
    for fname, content in chosen:
        parts.append(f"\n## memory/topics/{fname}\n{content.strip()}")
    parts.append("</relevant-memories>")
    return "\n".join(parts)


class RecallState:
    """Per-conversation recall bookkeeping (in memory; resets on restart)."""
    __slots__ = ("surfaced", "bytes_used")

    def __init__(self) -> None:
        self.surfaced: set[str] = set()
        self.bytes_used = 0


async def recall_block(agent_dir: str, query: str, state: RecallState, *, agent_id: str = "") -> str:
    """Pick relevant topic files for `query`; return the block ('' if none).
    Never raises."""
    from .memory_recall_api import recall_enabled, selector_model, call_selector
    try:
        if not recall_enabled() or should_skip_query(query):
            return ""
        if state.bytes_used >= SESSION_MAX_BYTES:
            print_ts("memory recall: session budget exhausted", agent=agent_id)
            return ""
        cands = [c for c in list_candidates(agent_dir) if c["filename"] not in state.surfaced]
        model = selector_model()
        if not cands or not model:
            return ""
        started = time.time()
        answer = await asyncio.wait_for(
            call_selector(model, format_candidates(cands), query[:4000]),
            timeout=SELECTOR_TIMEOUT_S + 5,
        )
        by_name = {c["filename"]: c for c in cands}
        chosen: list[tuple[str, str]] = []
        for fname in parse_selection(answer, set(by_name)):
            if state.bytes_used >= SESSION_MAX_BYTES:
                break
            try:
                content, used = read_capped(by_name[fname]["path"])
            except OSError:
                continue
            state.surfaced.add(fname)
            state.bytes_used += used
            chosen.append((fname, content))
        print_ts(f"memory recall: {len(chosen)}/{len(cands)} via {model} in "
                 f"{time.time() - started:.1f}s {[f for f, _ in chosen]}", agent=agent_id)
        return build_block(chosen)
    except Exception as e:
        print_ts(f"memory recall failed (turn continues without it): {e}", agent=agent_id, error=True)
        return ""
