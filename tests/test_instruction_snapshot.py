"""Per-session instruction snapshot (openflip/instruction_snapshot.py).

The system prompt is captured once per session and must stay byte-identical
across memory saves and self-edits of instruction files; a change is announced
once, as a note naming only the changed files. These tests pin that contract
so a regression that rebuilds the prompt every turn (cache miss on every save)
or re-announces the same change is caught.

Run: .lvenv/bin/python -m pytest tests/test_instruction_snapshot.py -q
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openflip.instruction_snapshot import CHANGED_HEADER, InstructionSnapshot  # noqa: E402
from openflip.agent import _load_system_files, load_system_sources  # noqa: E402

SRC = [("SOUL.md", "soul v1"), ("AGENT.md", "agent v1"), ("MEMORY.md", "index v1")]


def test_no_change_no_note():
    snap = InstructionSnapshot.capture(SRC, "SYS")
    assert snap.changed_note(SRC) == ""


def test_change_names_only_changed_files_and_keeps_prompt():
    snap = InstructionSnapshot.capture(SRC, "SYS")
    note = snap.changed_note([("SOUL.md", "soul v1"), ("AGENT.md", "agent v2"), ("MEMORY.md", "index v1")])
    assert note.startswith(CHANGED_HEADER)
    assert 'name="AGENT.md"' in note and "agent v2" in note
    assert "SOUL.md" not in note and "MEMORY.md" not in note
    assert snap.system_prompt == "SYS"


def test_each_change_announced_once():
    snap = InstructionSnapshot.capture(SRC, "SYS")
    new = [("SOUL.md", "soul v1"), ("AGENT.md", "agent v1"), ("MEMORY.md", "index v2")]
    assert snap.changed_note(new) != ""
    assert snap.changed_note(new) == ""


def test_removed_file_is_announced():
    snap = InstructionSnapshot.capture(SRC, "SYS")
    note = snap.changed_note([("SOUL.md", "soul v1"), ("AGENT.md", "agent v1")])
    assert 'name="MEMORY.md"' in note and "removed" in note


def test_sources_join_to_the_loaded_system_message():
    # load_system_sources must read exactly what _load_system_files reads, so a
    # snapshot taken from it describes the real prompt.
    with tempfile.TemporaryDirectory() as d:
        for name, body in [("SOUL.md", "soul text"), ("AGENT.md", "  agent {agent_id}  "), ("EMPTY.md", "")]:
            with open(os.path.join(d, name), "w", encoding="utf-8") as f:
                f.write(body)
        files = ["SOUL.md", "AGENT.md", "EMPTY.md", "MISSING.md"]
        src = load_system_sources(d, files, agent_id="x", display_name="X")
        assert [label for label, _ in src] == ["SOUL.md", "AGENT.md"]
        assert "\n\n".join(t for _, t in src) == _load_system_files(d, files, agent_id="x", display_name="X")
