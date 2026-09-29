"""Verification for /undo N, the undo preview, and /redo's verbatim re-send.

Standalone runnable script (no pytest in this venv):

    .lvenv/bin/python tests/test_undo_redo.py

What this guards:
  (a) user_facing_text strips the per-speaker preamble, recalled memories,
      time stamp, reply quote and speaker prefix (the old preview showed the
      tool-config block instead of the operator's words);
  (b) find_undo_cut_index_n(n=1) matches find_undo_cut_index, framework
      wrappers are skipped, and n past the start returns -1;
  (c) undo_last_turn(count=N) removes N turns from disk AND memory, keeps
      the head byte-identical, backs up first, and changes nothing when there
      are fewer than N turns;
  (d) AgentRunner.redo_last_turn undoes one turn and re-sends the removed
      turn-starting message verbatim (the prompt-cache contract: the retried
      request's history + user message are identical to the original's).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from openflip import _conversation_io as cio  # noqa: E402

PRE = ("Current tool configuration (set by the owner — you cannot change these):\n"
       "- web_search: count=8\nIf a user wants any of these changed, tell them to ask the owner.")
RECALL = "\n\n<relevant-memories>\nRetrieved for possible relevance — x\n## memory/topics/a\nbody\n</relevant-memories>"


def framed(text: str, stamp: bool = False, recall: bool = False, reply: bool = False) -> str:
    body = f"Op [operator]: {text}"
    if reply:
        body = '[replying to Bot: "earlier"]\n' + body
    if stamp:
        body = "[2026-09-29 09:38 Tuesday]\n" + body
    out = f"{PRE}\n\n---\n\n{body}"
    return out + (RECALL if recall else "")


def history() -> list[dict]:
    return [
        {"role": "user", "content": framed("first", stamp=True), "ts": 1.0},
        {"role": "assistant", "content": "a1", "ts": 2.0},
        {"role": "user", "content": framed("second: with colon", recall=True), "ts": 3.0},
        {"role": "assistant", "content": "", "ts": 4.0},
        {"role": "tool", "content": "tool out", "ts": 5.0},
        {"role": "user", "content": "[FRAMEWORK]: nudge", "ts": 6.0},
        {"role": "assistant", "content": "a2", "ts": 7.0},
        {"role": "user", "content": framed("third\nsecond line", reply=True), "ts": 8.0},
        {"role": "assistant", "content": "a3", "ts": 9.0},
    ]


class FakeConv:
    def __init__(self, path: str, msgs: list[dict]):
        self._path = path
        self.messages = [{"role": "system", "content": "sys"}] + [dict(m) for m in msgs]
        self._persisted_count = len(msgs)

    def _conversation_path(self) -> str:
        return self._path


def write(path: str, msgs: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


fails = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global fails
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f" -- {detail}" if detail and not cond else ""))
    if not cond:
        fails += 1


def test_user_facing_text() -> None:
    print("(a) user_facing_text")
    check("preamble + stamp + prefix", cio.user_facing_text(framed("hi", stamp=True)) == "hi")
    check("recall block stripped, colon kept",
          cio.user_facing_text(framed("a: b", recall=True)) == "a: b")
    check("reply quote stripped, multiline kept",
          cio.user_facing_text(framed("x\ny", reply=True)) == "x\ny")
    check("plain text untouched", cio.user_facing_text("just text") == "just text")
    check("non-operator speaker prefix", cio.user_facing_text("Someone: hello") == "hello")


def test_cut_index() -> None:
    print("(b) find_undo_cut_index_n")
    h = history()
    check("n=1 == old function", cio.find_undo_cut_index_n(h, 1) == cio.find_undo_cut_index(h) == 7)
    check("n=2 skips the [FRAMEWORK] wrapper", cio.find_undo_cut_index_n(h, 2) == 2)
    check("n=3", cio.find_undo_cut_index_n(h, 3) == 0)
    check("n past start = -1", cio.find_undo_cut_index_n(h, 4) == -1)


def test_undo_count() -> None:
    print("(c) undo_last_turn(count=N)")
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "discord:1.jsonl")
        h = history()
        write(p, h)
        conv = FakeConv(p, h)
        r = cio.undo_last_turn(conv, count=2)
        check("returns a result", r is not None)
        removed, previews, backup, cut = r
        on_disk = cio.read_all_messages(p)
        check("removed 7 messages", removed == 7, str(removed))
        check("disk head byte-identical", on_disk == h[:2])
        check("memory mirrors disk", [m["content"] for m in conv.messages[1:]] == [m["content"] for m in h[:2]])
        check("system message kept", conv.messages[0]["role"] == "system")
        check("_persisted_count re-synced", conv._persisted_count == 2)
        check("previews oldest-first, operator words only",
              previews == ["second: with colon", "third\nsecond line"], repr(previews))
        check("cut content is the stored message verbatim", cut == h[2]["content"])
        check("backup written first", os.path.exists(os.path.join(d, backup)))
        before = cio.read_all_messages(p)
        check("too many turns -> None", cio.undo_last_turn(conv, count=5) is None)
        check("too many turns -> nothing changed", cio.read_all_messages(p) == before)


def test_redo() -> None:
    print("(d) AgentRunner.redo_last_turn")
    from openflip.runtime import AgentRunner

    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "discord:1.jsonl")
        h = history()
        write(p, h)
        conv = FakeConv(p, h)

        class _Agent:
            id = "tester"

        r = object.__new__(AgentRunner)
        r.agent = _Agent()
        r._active_turns = {}
        r.conversations = {1: conv}
        sent: dict = {}

        async def fake_synth(target, prompt_text, **kw):
            sent.update(target=target, prompt=prompt_text, **kw)

        r.run_synthetic_turn = fake_synth
        ok, msg = asyncio.run(r.redo_last_turn(1, "discord:1", target=1, speaker_id=42))
        check("ok", ok, msg)
        check("last turn removed from disk", cio.read_all_messages(p) == h[:7])
        check("re-sent message is the stored one, verbatim",
              sent.get("verbatim_user_message") == h[7]["content"] and sent.get("prompt") == h[7]["content"])
        check("runs as an operator turn (not [synthetic])", sent.get("log_tag") == "[redo] "
              and sent.get("auto_post_final_text") is True and sent.get("speaker_id") == 42)
        check("message shows the operator's words", "> third\n> second line" in msg, msg)
        check("message says redoing", msg.startswith("🔁 Redoing the last turn"), msg)

        # in-flight guard: nothing changes, nothing re-sent
        sent.clear()
        before = cio.read_all_messages(p)

        class _Busy:
            def done(self):
                return False

        r._active_turns[1] = _Busy()
        ok2, msg2 = asyncio.run(r.redo_last_turn(1, "discord:1", target=1, speaker_id=42))
        check("refuses mid-turn", not ok2 and "/redo" in msg2, msg2)
        check("mid-turn: history untouched, nothing sent", cio.read_all_messages(p) == before and not sent)


if __name__ == "__main__":
    test_user_facing_text()
    test_cut_index()
    test_undo_count()
    test_redo()
    print("\nALL OK" if not fails else f"\n{fails} FAILED")
    sys.exit(1 if fails else 0)
