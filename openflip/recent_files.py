"""Re-read recently touched files after compaction (Claude Code's context-window.md).

Claude Code, right after compaction, "re-reads up to five of the files Claude has
read or edited in the session, choosing the ones modified most recently. A file
over 5,000 tokens comes back as a path reference without its content, shown as
`Referenced file` instead of `Read`."

openflip records every path a successful read_file / edit_file / write_file
touched in this conversation (tool_executor calls note_touched). After a
compaction the runtime calls reread_block, which builds that same block for the
[FRAMEWORK] compaction note. In-memory per process: a restart starts a fresh list,
like a new Claude Code process.
"""
from __future__ import annotations

import os
from collections import OrderedDict

MAX_FILES = 5                    # Claude Code: up to five
MAX_TOKENS_PER_FILE = 5_000      # Claude Code: over this -> path reference only
CHARS_PER_TOKEN = 4              # rough estimate; no tokenizer call on this path
MAX_TRACKED = 50                 # bound the per-conversation list
TRACKED_TOOLS = {"read_file", "edit_file", "write_file"}

_touched: dict[str, "OrderedDict[str, None]"] = {}


def note_touched(conversation_id: str, full_path: str) -> None:
    if not conversation_id or not full_path:
        return
    od = _touched.setdefault(conversation_id, OrderedDict())
    od.pop(full_path, None)
    od[full_path] = None
    while len(od) > MAX_TRACKED:
        od.popitem(last=False)


def touched(conversation_id: str) -> list[str]:
    return list((_touched.get(conversation_id) or {}).keys())


def forget(conversation_id: str) -> None:
    _touched.pop(conversation_id, None)


def _picks(conversation_id: str) -> list[str]:
    """Up to MAX_FILES still-existing touched files, most recently modified first."""
    alive = []
    for p in touched(conversation_id):
        try:
            if os.path.isfile(p):
                alive.append((os.path.getmtime(p), p))
        except OSError:
            continue
    alive.sort(key=lambda t: t[0], reverse=True)
    return [p for _, p in alive[:MAX_FILES]]


def reread_block(conversation_id: str) -> str:
    """The text that re-attaches recent files after compaction ('' if none)."""
    picks = _picks(conversation_id)
    if not picks:
        return ""
    cap = MAX_TOKENS_PER_FILE * CHARS_PER_TOKEN
    parts = ["Files you read or edited recently, re-read now (most recently modified first):"]
    for p in picks:
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read(cap + 1)
        except (UnicodeDecodeError, OSError):
            parts.append(f"Referenced file {p} (not text; read it again if you need it)")
            continue
        if len(text) > cap:
            parts.append(f"Referenced file {p} (over {MAX_TOKENS_PER_FILE} tokens; read it again "
                         f"if you need its contents)")
        else:
            parts.append(f"<file path=\"{p}\">\n{text}\n</file>")
    return "\n\n".join(parts)
