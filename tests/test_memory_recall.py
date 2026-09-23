"""Tests for openflip/memory_recall.py (selector mocked; no network).

Run: .lvenv/bin/python tests/test_memory_recall.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openflip import memory_recall as r
from openflip import memory_recall_api as api
from openflip.memory_recall_prompt import FILE_MAX_BYTES, FILE_MAX_LINES, SESSION_MAX_BYTES, INJECT_PREFIX

FAILS: list[str] = []


def check(ok: bool, label: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        FAILS.append(label)


def make_agent(files: dict[str, str]) -> str:
    d = tempfile.mkdtemp(prefix="recalltest_")
    tdir = os.path.join(d, "memory", "topics")
    os.makedirs(tdir)
    for name, body in files.items():
        with open(os.path.join(tdir, name), "w", encoding="utf-8") as f:
            f.write(body)
    return d


def topic(desc: str, body: str) -> str:
    return f"---\nname: x\ndescription: \"{desc}\"\nmodified: 2026-09-22T00:00:00\n---\n\n# T\n\n{body}\n"


def test_candidates() -> None:
    print("test_candidates")
    d = make_agent({"a.md": topic("alpha facts", "A"), "b.md": "# B\n\nno frontmatter body text\n", "c.txt": "x"})
    c = r.list_candidates(d)
    names = sorted(x["filename"] for x in c)
    check(names == ["a.md", "b.md"], f"only .md topic files listed ({names})")
    fmt = r.format_candidates(c)
    check("- a.md (" in fmt and "): alpha facts" in fmt, "nwt line shape with description")
    check("b.md" in fmt and "no frontmatter" in fmt, "falls back to body hook without frontmatter")
    check(r.list_candidates(tempfile.mkdtemp()) == [], "no topics dir -> no candidates")
    typed = "---\nname: x\ndescription: \"typed one\"\ntype: feedback\nmodified: 2026-09-22T00:00:00\n---\n\n# T\n\nbody\n"
    fmt = r.format_candidates(r.list_candidates(make_agent({"t.md": typed})))
    check("- [feedback] t.md (" in fmt and "): typed one" in fmt, "type shown as CC's [type] prefix")


def test_skip_rules() -> None:
    print("test_skip_rules")
    check(r.should_skip_query(""), "empty skipped")
    check(r.should_skip_query("hi"), "single word skipped")
    check(not r.should_skip_query("what serves my sites"), "multi-word kept")
    check(not r.should_skip_query("メモリー"), "CJK single token kept")


def test_read_capped() -> None:
    print("test_read_capped")
    d = make_agent({"long.md": "\n".join(f"line {i}" for i in range(FILE_MAX_LINES + 50)),
                    "wide.md": "x" * (FILE_MAX_BYTES * 2) + "\nend\n", "small.md": "tiny\n"})
    t = os.path.join(d, "memory", "topics")
    c, _ = r.read_capped(os.path.join(t, "long.md"))
    check(f"first {FILE_MAX_LINES} lines" in c and "line 199" in c and "line 200" not in c, "line cap + note")
    c, used = r.read_capped(os.path.join(t, "wide.md"))
    check("byte limit" in c and used < FILE_MAX_BYTES + 300, "byte cap + note")
    c, _ = r.read_capped(os.path.join(t, "small.md"))
    check(c == "tiny\n", "small file untouched")


def test_parse_selection() -> None:
    print("test_parse_selection")
    valid = {"a.md", "b.md"}
    check(r.parse_selection('{"selected_memories": ["a.md"]}', valid) == ["a.md"], "plain JSON")
    check(r.parse_selection('Sure:\n{"selected_memories": ["b.md","zz.md"]}', valid) == ["b.md"], "wrapped JSON, unknown dropped")
    check(r.parse_selection('{"selected_memories": ["[user] a.md"]}', valid) == ["a.md"], "type prefix stripped")
    check(r.parse_selection("garbage", valid) == [], "garbage -> empty")
    many = '{"selected_memories": [' + ",".join(f'"f{i}.md"' for i in range(9)) + "]}"
    check(len(r.parse_selection(many, {f"f{i}.md" for i in range(9)})) == 5, "capped at 5")


def test_recall_flow() -> None:
    print("test_recall_flow")
    d = make_agent({"caddy.md": topic("caddy serves sites", "Caddyfile at /etc/caddy"),
                    "cats.md": topic("cat facts", "cats purr")})
    calls: list[str] = []

    async def fake_selector(model, cands, query):
        calls.append(cands)
        return '{"selected_memories": ["caddy.md"]}'

    orig = (api.call_selector, api.selector_model, api.recall_enabled)
    api.call_selector, api.selector_model, api.recall_enabled = fake_selector, (lambda: "m"), (lambda: True)
    try:
        st = r.RecallState()
        blk = asyncio.run(r.recall_block(d, "what serves my sites", st))
        check(INJECT_PREFIX in blk and "Caddyfile at /etc/caddy" in blk, "block has CC prefix + file body")
        check(st.surfaced == {"caddy.md"} and st.bytes_used > 0, "state records surfaced file + bytes")
        blk2 = asyncio.run(r.recall_block(d, "what serves my sites again", st))
        check(blk2 == "" and "caddy.md" not in calls[-1], "already-surfaced file not offered or re-injected")
        n = len(calls)
        check(asyncio.run(r.recall_block(d, "hi", st)) == "" and len(calls) == n, "one-word query makes no selector call")
        st2 = r.RecallState()
        st2.bytes_used = SESSION_MAX_BYTES
        check(asyncio.run(r.recall_block(d, "what serves my sites", st2)) == "", "budget exhausted -> nothing")

        async def boom(*a):
            raise RuntimeError("network down")
        api.call_selector = boom
        check(asyncio.run(r.recall_block(d, "what serves my sites", r.RecallState())) == "", "selector failure -> '' (no raise)")
        api.recall_enabled = lambda: False
        check(asyncio.run(r.recall_block(d, "what serves my sites", r.RecallState())) == "", "disabled -> ''")
    finally:
        api.call_selector, api.selector_model, api.recall_enabled = orig


if __name__ == "__main__":
    test_candidates()
    test_skip_rules()
    test_read_capped()
    test_parse_selection()
    test_recall_flow()
    print("\nRESULT:", "ALL PASS" if not FAILS else f"{len(FAILS)} FAIL")
    sys.exit(1 if FAILS else 0)
