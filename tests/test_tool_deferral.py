"""Deferred tool loading (Claude Code style tool search): openflip/tool_deferral.py.

Run: .lvenv/bin/python -m pytest tests/test_tool_deferral.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import openflip.tools  # noqa: E402,F401  (registers tools)
from openflip import tool_deferral as td  # noqa: E402
from openflip.anthropic_conversation import _build_tool_schemas, _openflip_msgs_to_anthropic  # noqa: E402
from openflip.tools._base import TOOL_REGISTRY  # noqa: E402


def _funcs():
    return [t.func for t in TOOL_REGISTRY.values()]


def test_search_tool_never_deferred():
    names = list(TOOL_REGISTRY)
    assert td.SEARCH_TOOL_NAME in names
    assert td.SEARCH_TOOL_NAME not in td.deferred_names(names)


def test_schemas_mark_only_deferred():
    funcs = _funcs()
    deferred = set(td.deferred_names([f.__name__ for f in funcs]))
    schemas = _build_tool_schemas(funcs, deferred)
    assert len(schemas) == len(funcs)  # every definition still sent
    for sc in schemas:
        assert bool(sc.get("defer_loading")) == (sc["name"] in deferred)
    assert not any("defer_loading" in sc for sc in _build_tool_schemas(funcs))


def test_names_block_stable_and_sorted():
    names = list(TOOL_REGISTRY)
    a, b = td.names_block(names), td.names_block(list(reversed(names)))
    assert a == b and a
    listed = a.split("\n\n", 2)[-1].split("\n")
    assert listed == sorted(listed)
    assert "read_file" not in listed


def test_select_and_keyword_search():
    td.set_current_catalog([], {})
    found, note = td.search_deferred("select:generate_image,read_file,nope")
    assert found == ["generate_image"]
    assert "read_file" in note and "nope" in note
    found, _ = td.search_deferred("image upscale")
    assert "upscale_image" in found
    found, _ = td.search_deferred("+mc look")
    assert found and all("mc" in n for n in found)


def test_tool_reference_blocks_round_trip():
    from types import SimpleNamespace as NS
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [NS(tool_use_id="t1", function_name="tool_search", args={"query": "select:generate_image"})]},
        {"role": "tool", "content": "", "tool_use_id": "t1", "tool_references": ["generate_image"]},
    ]
    _, api = _openflip_msgs_to_anthropic(msgs, "sys")
    res = api[-1]["content"][0]
    assert res["type"] == "tool_result"
    assert res["content"] == [{"type": "tool_reference", "tool_name": "generate_image"}]


def test_disable_via_config(monkeypatch=None):
    orig = td._cfg
    try:
        td._cfg = lambda: {"enabled": False}
        assert not td.deferral_enabled("anthropic")
        td._cfg = lambda: {}
        assert td.deferral_enabled("anthropic")
        assert not td.deferral_enabled("claude_code")
        td._cfg = lambda: {"always_load": ["generate_image"]}
        assert "generate_image" not in td.deferred_names(["generate_image", "read_file"])
        assert "read_file" in td.deferred_names(["generate_image", "read_file"])
    finally:
        td._cfg = orig
