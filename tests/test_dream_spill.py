"""Verification for dream()'s corpus-spill payload (openflip/tools/dream_tool.py).

Standalone runnable script (no pytest in this venv):

    .lvenv/bin/python tests/test_dream_spill.py

The bug this guards against: dream() used to inline MEMORY.md + as many daily
logs as fit a fixed char budget, selecting newest-first and BREAKING at the
first log that didn't fit — permanently dropping that log AND every older one
from the payload. Because MEMORY.md is both the input and the output of a dream,
the log budget shrank monotonically as the agent learned more.

The fix spills the FULL corpus (MEMORY.md + every daily log, nothing dropped)
to a per-agent file under tempfile.gettempdir() and returns its absolute path
plus a compact index; a bounded inline preview is still included for the common
small case. These tests prove:

  (a) a corpus far exceeding the inline budget loses ZERO logs — every log is
      reachable via the spill file, and the spill path lives under
      tempfile.gettempdir() (NOT a hardcoded /tmp);
  (b) the inline preview really is partial for that oversized corpus (proving
      the spill is doing the work) and the omission note points at the spill;
  (c) a small corpus fits inline with no omission note, and the spill file is
      still written under tempfile.gettempdir();
  (d) tail-truncation: a single oversized newest log is inlined as a truncated
      tail (not dropped-with-everything-older), and its full copy is in spill;
  (e) MemoryTooLargeError still fires for a genuinely absurd MEMORY.md, and its
      message points at the spill file.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip.tools import dream_tool
from openflip.utils import safe_filename

FAILURES: list[str] = []


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        FAILURES.append(label)


def _mk_agent_dir(agent_id: str, core_chars: int, log_specs: list[tuple[str, int]]) -> str:
    """Create a throwaway agent dir with a MEMORY.md and daily logs.

    log_specs: [(date_label, approx_char_size), ...]. Each log gets a UNIQUE
    marker so we can prove its content survives into the spill file.
    """
    base = tempfile.mkdtemp(prefix=f"dreamtest-{agent_id}-")
    with open(os.path.join(base, "MEMORY.md"), "w", encoding="utf-8") as f:
        f.write("CORE-MARKER\n" + ("core fact line.\n" * (max(core_chars, 20) // 16)))
    mem = os.path.join(base, "memory")
    os.makedirs(mem, exist_ok=True)
    for label, size in log_specs:
        marker = f"UNIQUE-MARKER-{label}"
        filler = (f"event on {label}: something happened here.\n")
        n = max(1, (size - len(marker) - 1) // len(filler))
        with open(os.path.join(mem, f"{label}.md"), "w", encoding="utf-8") as f:
            f.write(marker + "\n" + (filler * n))
    return base


def _spill_path_for(agent_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"dream-{safe_filename(agent_id)}.md")


def test_oversized_corpus_loses_nothing():
    print("\n(a/b) oversized corpus — zero permanently-lost logs; spill under gettempdir; preview partial")
    agent_id = "spilltest_big"
    labels = [f"2026-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}" for i in range(60)]
    specs = [(lbl, 4000) for lbl in labels]  # ~240k of logs — far over the budget
    base = _mk_agent_dir(agent_id, core_chars=25000, log_specs=specs)
    try:
        payload = dream_tool._build_consolidation_prompt(agent_id, base, 25000, "2026-07-29")

        spill_path = _spill_path_for(agent_id)
        check("spill file exists", os.path.isfile(spill_path))
        check("spill path is under tempfile.gettempdir()",
              os.path.dirname(os.path.realpath(spill_path)) == os.path.realpath(tempfile.gettempdir()))
        check("payload references the spill path", spill_path in payload)

        with open(spill_path, "r", encoding="utf-8") as f:
            spill = f.read()

        # ZERO permanently-lost logs: every unique marker is in the spill file.
        missing = [lbl for lbl in labels if f"UNIQUE-MARKER-{lbl}" not in spill]
        check(f"all {len(labels)} logs reachable via spill ({len(missing)} missing)", not missing)
        check("MEMORY.md content is in the spill file", "CORE-MARKER" in spill)

        # The inline preview must genuinely be partial (proving the spill matters).
        inlined = sum(1 for lbl in labels if f"UNIQUE-MARKER-{lbl}" in payload)
        check(f"inline preview is partial ({inlined}/{len(labels)} logs inlined, < total)",
              0 < inlined < len(labels))
        check("omission note fired and points at spill", "PARTIAL INLINE VIEW" in payload and payload.count(spill_path) >= 1)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        try:
            os.unlink(_spill_path_for(agent_id))
        except OSError:
            pass


def test_small_corpus_fits_inline():
    print("\n(c) small corpus — fits inline, no omission note, spill still written under gettempdir")
    agent_id = "spilltest_small"
    labels = ["2026-07-27", "2026-07-28", "2026-07-29"]
    specs = [(lbl, 400) for lbl in labels]
    base = _mk_agent_dir(agent_id, core_chars=800, log_specs=specs)
    try:
        payload = dream_tool._build_consolidation_prompt(agent_id, base, 25000, "2026-07-29")
        spill_path = _spill_path_for(agent_id)
        check("spill file exists (even for small corpus)", os.path.isfile(spill_path))
        check("all small-corpus logs inlined", all(f"UNIQUE-MARKER-{l}" in payload for l in labels))
        check("no omission note for a fully-inlined corpus", "PARTIAL INLINE VIEW" not in payload)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        try:
            os.unlink(_spill_path_for(agent_id))
        except OSError:
            pass


def test_single_oversized_newest_log_truncates():
    print("\n(d) tail-truncation — one oversized newest log is inlined as a truncated tail, full copy in spill")
    agent_id = "spilltest_trunc"
    # One giant newest log bigger than the whole inline budget, plus older ones.
    labels = ["2026-07-01", "2026-07-15", "2026-07-29"]
    specs = [("2026-07-01", 2000), ("2026-07-15", 2000), ("2026-07-29", 70000)]
    base = _mk_agent_dir(agent_id, core_chars=800, log_specs=specs)
    try:
        payload = dream_tool._build_consolidation_prompt(agent_id, base, 25000, "2026-07-29")
        spill_path = _spill_path_for(agent_id)
        with open(spill_path, "r", encoding="utf-8") as f:
            spill = f.read()
        # The giant newest log's full content survives to spill...
        check("oversized newest log fully in spill", "UNIQUE-MARKER-2026-07-29" in spill)
        # ...and instead of being dropped-with-everything-older, its tail is inlined.
        check("oversized newest log inlined as TRUNCATED tail", "TRUNCATED" in payload)
        # The older logs are NOT lost either (regression: old code broke the loop
        # and dropped them). They must at least be reachable via spill.
        check("older logs still reachable via spill",
              "UNIQUE-MARKER-2026-07-01" in spill and "UNIQUE-MARKER-2026-07-15" in spill)
    finally:
        shutil.rmtree(base, ignore_errors=True)
        try:
            os.unlink(_spill_path_for(agent_id))
        except OSError:
            pass


def test_absurd_core_still_raises():
    print("\n(e) genuinely absurd MEMORY.md still raises MemoryTooLargeError, pointing at spill")
    agent_id = "spilltest_absurd"
    base = _mk_agent_dir(agent_id, core_chars=200000, log_specs=[("2026-07-29", 500)])
    try:
        raised = False
        msg = ""
        try:
            dream_tool._build_consolidation_prompt(agent_id, base, 25000, "2026-07-29")
        except dream_tool.MemoryTooLargeError as e:
            raised = True
            msg = str(e)
        check("MemoryTooLargeError raised for absurd core", raised)
        check("error message points at the spill file", _spill_path_for(agent_id) in msg)
        # The corpus was still spilled before the raise.
        check("spill file was still written before raising", os.path.isfile(_spill_path_for(agent_id)))
    finally:
        shutil.rmtree(base, ignore_errors=True)
        try:
            os.unlink(_spill_path_for(agent_id))
        except OSError:
            pass


if __name__ == "__main__":
    test_oversized_corpus_loses_nothing()
    test_small_corpus_fits_inline()
    test_single_oversized_newest_log_truncates()
    test_absurd_core_still_raises()
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: ALL PASS")
