"""Per-session instruction snapshot (Claude Code style prompt-cache discipline).

The system prompt is built ONCE per conversation session from the agent's
system_files plus the memory index, and the exact same bytes are reused on
every turn. Anthropic's prompt cache keys on those bytes, so rebuilding it
after every memory save or AGENT.md edit re-sent the whole conversation at
full price on the next turn.

Claude Code works the same way: instruction files (CLAUDE.md, MEMORY.md) load
at session start and are re-read only on session start, compaction or an
explicit reload. When one changes mid-session, only the changed files are sent,
as a NEW message at the end of the conversation ("These instruction files
changed"), and the earlier copy is never rewritten. This module does exactly
that:

  - `InstructionSnapshot.capture(...)` records each source's text and the
    assembled system prompt for a session.
  - `snapshot.changed_note(current)` compares the files as they are now with
    the snapshot and returns the text of a note naming only the changed files
    (or "" when nothing changed), then updates its own record so the same
    change is announced once.
  - A fresh capture (new session, /reset, compaction, /reload, restart) is the
    only way the system prompt itself changes.

Nothing here is provider- or agent-specific; providers just read
`snapshot.system_prompt` and append the note as a user message.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

# Same wording Claude Code uses for its mid-session instruction-change message.
CHANGED_HEADER = (
    "These instruction files changed - each replaces its earlier copy in your "
    "system prompt. Follow the new versions from now on."
)


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class InstructionSnapshot:
    """The instruction sources a session started with.

    `sources` is an ordered list of (label, text): one entry per system file
    in agent.json order, then the memory index block (label "MEMORY.md")."""
    sources: list[tuple[str, str]] = field(default_factory=list)
    system_prompt: str = ""
    _digests: dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls, sources: list[tuple[str, str]], system_prompt: str) -> "InstructionSnapshot":
        snap = cls(sources=list(sources), system_prompt=system_prompt)
        snap._digests = {label: _digest(text) for label, text in sources}
        return snap

    def changed_note(self, current: list[tuple[str, str]]) -> str:
        """Note text naming only the sources whose text differs from the last
        announced version. Updates the record, so each change is sent once.
        A source that disappeared is announced as removed."""
        changed: list[tuple[str, str]] = []
        seen: set[str] = set()
        for label, text in current:
            seen.add(label)
            d = _digest(text)
            if self._digests.get(label) != d:
                changed.append((label, text))
                self._digests[label] = d
        for label in list(self._digests):
            if label not in seen:
                changed.append((label, ""))
                del self._digests[label]
        if not changed:
            return ""
        parts = [CHANGED_HEADER]
        for label, text in changed:
            body = text.strip() or "(this file is now empty or was removed)"
            parts.append(f"<instruction-file name=\"{label}\">\n{body}\n</instruction-file>")
        return "\n\n".join(parts)
