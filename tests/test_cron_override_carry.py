"""Cron fresh-session archiving must carry session OVERRIDES forward while
resetting per-run state.

Bug (2026-10-05): cron.py's per_context_sessions archive step os.replace'd the
whole .meta.json, discarding the "overrides" block (model/effort/etc.) along
with the per-run state (compaction_block, last_usage). A model/effort override
on a cron session only lasted one run.

This test exercises the carry-forward logic in isolation (the exact lines added
to cron.py's archive step) against a real on-disk meta: archive the old meta,
re-emit a settings-only meta, and assert the fresh meta keeps overrides +
legacy effort_override but drops compaction_block and last_usage.
"""
from __future__ import annotations
import json
import os
import tempfile

from openflip.utils import load_json, save_json


def _carry_forward(meta_path: str, ts: int) -> None:
    """Mirror of the cron.py archive carry-forward block. Archives meta_path
    to <meta_path>.archived-<ts>, then re-emits a settings-only meta."""
    archived = f"{meta_path}.archived-{ts}"
    os.replace(meta_path, archived)
    old_meta = load_json(archived, default={})
    if isinstance(old_meta, dict):
        carry: dict = {}
        ovr = old_meta.get("overrides")
        if isinstance(ovr, dict) and ovr:
            carry["overrides"] = ovr
        legacy_eff = old_meta.get("effort_override")
        if isinstance(legacy_eff, str) and legacy_eff:
            carry["effort_override"] = legacy_eff
        if carry:
            save_json(meta_path, carry)


def test_overrides_carry_perrun_resets():
    with tempfile.TemporaryDirectory() as d:
        meta = os.path.join(d, "cron:test.meta.json")
        save_json(meta, {
            "overrides": {"model": "claude-sonnet-5-5", "effort": "high"},
            "compaction_block": {"foo": "bar"},
            "last_usage": {"in": 123, "out": 456},
        })
        _carry_forward(meta, ts=111)

        fresh = json.load(open(meta))
        # settings carried over
        assert fresh.get("overrides") == {"model": "claude-sonnet-5-5", "effort": "high"}, fresh
        # per-run state reset
        assert "compaction_block" not in fresh, fresh
        assert "last_usage" not in fresh, fresh
        # archive still holds the full original
        arch = json.load(open(f"{meta}.archived-111"))
        assert arch.get("compaction_block") == {"foo": "bar"}, arch


def test_legacy_effort_override_carries():
    with tempfile.TemporaryDirectory() as d:
        meta = os.path.join(d, "cron:test.meta.json")
        save_json(meta, {
            "effort_override": "high",
            "compaction_block": {"x": 1},
        })
        _carry_forward(meta, ts=222)
        fresh = json.load(open(meta))
        assert fresh.get("effort_override") == "high", fresh
        assert "compaction_block" not in fresh, fresh


def test_no_overrides_means_no_fresh_meta():
    with tempfile.TemporaryDirectory() as d:
        meta = os.path.join(d, "cron:test.meta.json")
        save_json(meta, {"compaction_block": {"x": 1}, "last_usage": {"in": 1}})
        _carry_forward(meta, ts=333)
        # Nothing worth carrying → no fresh meta re-emitted; the session
        # simply starts with no sidecar (defaults), per-run state gone.
        assert not os.path.isfile(meta), "fresh meta should not exist when nothing carries"
        arch = json.load(open(f"{meta}.archived-333"))
        assert arch.get("compaction_block") == {"x": 1}


if __name__ == "__main__":
    test_overrides_carry_perrun_resets()
    test_legacy_effort_override_carries()
    test_no_overrides_means_no_fresh_meta()
    print("all cron override-carry tests passed")
