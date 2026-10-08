"""spawn_subagents — automatic project CLAUDE.md injection for workers.

Standalone runnable script (no pytest in this venv):

    .lvenv/bin/python tests/test_subagent_project_docs.py

What this guards (all in throwaway tmp dirs):
  (1) a task prompt naming a path inside a scope root gets that root's
      CLAUDE.md in a tagged block headed by its source path;
  (2) nested CLAUDE.md files arrive OUTERMOST first;
  (3) the walk never goes above a scope root (a CLAUDE.md in the root's
      parent is not injected) and never scans subtrees;
  (4) no path mentioned → nothing injected; path outside scope → nothing;
      no scope roots → nothing;
  (5) per-file truncation marker (names the file, points at read_file);
  (6) total cap across files; (7) max-files cap; dedupe by realpath;
  (8) "## Task\\n" stays the LAST section of the worker prompt (the fakes and
      other tests split on it);
  (9) content with literal braces / template-looking vars passes through
      unchanged (no substitution);
 (10) end-to-end through spawn_subagents with the FakeRunner: the prompt the
      worker receives carries the block, and the launch log line names it.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from openflip import config_global as _cfg
from openflip import tool_settings as ts
from openflip.tools import subagent as sa
from openflip.tools.subagent import (
    find_mentioned_paths, collect_project_docs, render_project_docs,
    project_docs_block, build_worker_prompt,
)

FAILURES: list[str] = []


def check(label: str, cond: bool) -> None:
    print(("  ok    " if cond else "  FAIL  ") + label)
    if not cond:
        FAILURES.append(label)


def _w(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return os.path.realpath(path)


def _task_section(prompt: str) -> str:
    return prompt.split("## Task\n", 1)[-1]


# ------------------------------------------------------------------ tests

def test_path_extraction():
    print("(0) path extraction")
    got = find_mentioned_paths(
        "Edit `/a/b/c.py` and /a/d.py:12, see https://x.com/foo and/or w/ 24/7 ./rel "
        "(also /tmp/z). Quote '/q/r' end /e/f."
    )
    check("backticked + bare paths found, file:line suffix dropped",
          got[:2] == ["/a/b/c.py", "/a/d.py"])
    check("URL, and/or, w/, 24/7, ./rel ignored",
          not any("x.com" in g or g in ("/or", "/", "/7", "/rel") for g in got))
    check("parenthesis / quotes / trailing period stripped",
          "/tmp/z" in got and "/q/r" in got and "/e/f" in got)
    home = os.path.expanduser("~")
    check("~ expands", find_mentioned_paths("look in ~/notes/x.md") == [os.path.join(home, "notes", "x.md")])
    check("dedupe preserves first-seen order",
          find_mentioned_paths("/x/1 /x/2 /x/1") == ["/x/1", "/x/2"])
    check("no paths → empty", find_mentioned_paths("just words, no slashes here") == [])


def test_inject_and_order():
    print("(1-3) injection, outermost-first, never above root, never down")
    with tempfile.TemporaryDirectory() as td:
        parent = os.path.realpath(td)
        root = os.path.join(parent, "scope")
        _w(os.path.join(parent, "CLAUDE.md"), "PARENT-DOC must never appear")
        root_doc = _w(os.path.join(root, "CLAUDE.md"), "ROOT-DOC conventions")
        sub_doc = _w(os.path.join(root, "pkg", "CLAUDE.md"), "SUB-DOC conventions")
        _w(os.path.join(root, "pkg", "deeper", "CLAUDE.md"), "DEEPER-DOC (below the mention; must not appear)")
        _w(os.path.join(root, "other", "CLAUDE.md"), "SIBLING-DOC (sibling; must not appear)")
        target = os.path.join(root, "pkg", "mod.py")
        _w(target, "x = 1\n")

        paths = collect_project_docs(f"Refactor {target} please.", [root])
        check("mention inside scope → both enclosing CLAUDE.md found", paths == [root_doc, sub_doc])
        check("outermost first", paths[0] == root_doc)
        block, injected = project_docs_block(f"Refactor {target} please.", [root])
        check("block headed by source paths in order",
              block.index(f'<project-doc path="{root_doc}">') < block.index(f'<project-doc path="{sub_doc}">'))
        check("block carries the note", block.startswith(sa.PROJECT_DOCS_NOTE))
        check("file contents present", "ROOT-DOC conventions" in block and "SUB-DOC conventions" in block)
        check("parent-of-root CLAUDE.md NOT injected", "PARENT-DOC" not in block)
        check("sibling / deeper CLAUDE.md NOT injected (no subtree scan)",
              "DEEPER-DOC" not in block and "SIBLING-DOC" not in block)
        check("injected list matches", injected == [root_doc, sub_doc])

        # Mentioning a directory starts the walk AT that directory.
        paths_dir = collect_project_docs(f"Look in {os.path.join(root, 'pkg')}", [root])
        check("directory mention: starts at the directory itself", paths_dir == [root_doc, sub_doc])
        # Mentioning a not-yet-existing file still walks from its parent.
        paths_new = collect_project_docs(f"Create {os.path.join(root, 'pkg', 'new.py')}", [root])
        check("nonexistent file mention: walks from its parent dir", paths_new == [root_doc, sub_doc])
        # Mentioning the root itself.
        check("root mention: root doc only", collect_project_docs(f"See {root}", [root]) == [root_doc])

        # Nested scope roots: walk to the OUTERMOST one.
        inner_root = os.path.join(root, "pkg")
        paths_nested = collect_project_docs(f"Refactor {target}", [inner_root, root])
        check("nested roots: walks up to the outermost root", paths_nested == [root_doc, sub_doc])
        paths_inner_only = collect_project_docs(f"Refactor {target}", [inner_root])
        check("inner root only: stops at inner root", paths_inner_only == [sub_doc])

        # Scope root given via symlink still matches a realpath mention (and vice versa).
        link = os.path.join(parent, "link-to-scope")
        os.symlink(root, link)
        check("symlinked scope root resolves", collect_project_docs(f"Edit {target}", [link]) == [root_doc, sub_doc])
        check("symlinked mention resolves + dedupes by realpath",
              collect_project_docs(f"Edit {os.path.join(link, 'pkg', 'mod.py')} and {target}", [root]) == [root_doc, sub_doc])


def test_nothing_cases():
    print("(4) nothing injected: no mention / outside scope / no roots")
    with tempfile.TemporaryDirectory() as td:
        root = os.path.realpath(os.path.join(td, "scope"))
        _w(os.path.join(root, "CLAUDE.md"), "ROOT-DOC")
        outside = os.path.realpath(os.path.join(td, "elsewhere"))
        _w(os.path.join(outside, "CLAUDE.md"), "OUTSIDE-DOC")
        _w(os.path.join(outside, "f.txt"), "")

        check("no path mentioned → empty", project_docs_block("summarize the news", [root]) == ("", []))
        check("path outside scope → empty",
              project_docs_block(f"read {os.path.join(outside, 'f.txt')}", [root]) == ("", []))
        check("no scope roots → empty",
              project_docs_block(f"read {os.path.join(root, 'x.py')}", []) == ("", []))
        # Prefix trap: /scope2 is not inside /scope.
        trap = os.path.realpath(os.path.join(td, "scope2"))
        _w(os.path.join(trap, "CLAUDE.md"), "TRAP-DOC")
        check("sibling dir sharing the root's name prefix is NOT inside scope",
              project_docs_block(f"read {os.path.join(trap, 'x.py')}", [root]) == ("", []))
        # No CLAUDE.md anywhere on the chain → empty, no error.
        bare = os.path.realpath(os.path.join(td, "bare"))
        _w(os.path.join(bare, "a", "f.py"), "")
        check("scope with no CLAUDE.md → empty", project_docs_block(f"read {bare}/a/f.py", [bare]) == ("", []))


def test_caps():
    print("(5-7) per-file truncation, total cap, max files")
    with tempfile.TemporaryDirectory() as td:
        root = os.path.realpath(os.path.join(td, "scope"))
        big_doc = _w(os.path.join(root, "CLAUDE.md"), "B" * (sa.PROJECT_DOC_FILE_CAP + 5_000))
        target = os.path.join(root, "f.py")
        _w(target, "")
        block, injected = project_docs_block(f"edit {target}", [root])
        body = block.split(f'<project-doc path="{big_doc}">\n', 1)[1].split("\n</project-doc>", 1)[0]
        check("per-file: content cut to PROJECT_DOC_FILE_CAP",
              body.startswith("B" * sa.PROJECT_DOC_FILE_CAP) and "B" * (sa.PROJECT_DOC_FILE_CAP + 1) not in body)
        check("per-file: marker names the file and read_file",
              "[truncated:" in body and big_doc in body and "read_file" in body)

    with tempfile.TemporaryDirectory() as td:
        root = os.path.realpath(os.path.join(td, "scope"))
        n = 9_000  # 3 × 9000 = 27000 > total cap 20000; each < per-file cap
        d1 = _w(os.path.join(root, "CLAUDE.md"), "1" * n)
        d2 = _w(os.path.join(root, "a", "CLAUDE.md"), "2" * n)
        d3 = _w(os.path.join(root, "a", "b", "CLAUDE.md"), "3" * n)
        target = os.path.join(root, "a", "b", "f.py")
        _w(target, "")
        block, injected = project_docs_block(f"edit {target}", [root])
        check("total cap: all three listed", injected == [d1, d2, d3])
        check("total cap: first two files intact", "1" * n in block and "2" * n in block)
        third = block.split(f'<project-doc path="{d3}">\n', 1)[1].split("\n</project-doc>", 1)[0]
        remaining = sa.PROJECT_DOC_TOTAL_CAP - 2 * n
        check("total cap: third file cut to the remaining budget with a marker",
              third.startswith("3" * remaining) and "3" * (remaining + 1) not in third and "[truncated:" in third)
        docs_chars = sum(len(b.split(">\n", 1)[1].split("\n</project-doc>")[0].split("\n\n[truncated:")[0])
                         for b in block.split("<project-doc ")[1:])
        check("total cap: doc content ≤ PROJECT_DOC_TOTAL_CAP", docs_chars <= sa.PROJECT_DOC_TOTAL_CAP)

    with tempfile.TemporaryDirectory() as td:
        root = os.path.realpath(os.path.join(td, "scope"))
        docs = []
        d = root
        for i in range(5):
            docs.append(_w(os.path.join(d, "CLAUDE.md"), f"DOC{i}"))
            d = os.path.join(d, f"l{i}")
        target = os.path.join(d, "f.py")
        _w(target, "")
        paths = collect_project_docs(f"edit {target}", [root])
        check("max files: capped at PROJECT_DOC_MAX_FILES, outermost kept",
              paths == docs[:sa.PROJECT_DOC_MAX_FILES])


def test_prompt_shape_and_braces():
    print("(8-9) ## Task last; braces pass through unchanged")
    with tempfile.TemporaryDirectory() as td:
        root = os.path.realpath(os.path.join(td, "scope"))
        raw = "Use {agent_id} and {display_name}; dict = {'k': {1, 2}}; {{double}} %s %(x)s $HOME"
        _w(os.path.join(root, "CLAUDE.md"), raw)
        target = os.path.join(root, "f.py")
        _w(target, "")
        task = f"Edit {target} carefully.\nSecond line."
        block, _ = project_docs_block(task, [root])
        prompt = build_worker_prompt(task, block)
        check("prompt starts with WORKER_PREAMBLE", prompt.startswith(sa.WORKER_PREAMBLE))
        check("docs block sits between preamble and ## Task",
              prompt.index(sa.WORKER_PREAMBLE) < prompt.index("<project-doc ") < prompt.rindex("## Task\n"))
        check("## Task is the last section; task text intact after it", _task_section(prompt) == task)
        check("exactly one ## Task heading", prompt.count("## Task\n") == 1)
        check("literal braces / template vars untouched", raw in prompt)
        check("no docs → preamble + task only", build_worker_prompt(task, "") == f"{sa.WORKER_PREAMBLE}\n\n## Task\n{task}")


async def test_end_to_end():
    print("(10) end-to-end via spawn_subagents + FakeRunner")
    from test_subagent_tool import FakeRunner, _agent, _call, _settings, _fake_config
    from openflip import utils as _u

    with tempfile.TemporaryDirectory() as td:
        root = os.path.realpath(os.path.join(td, "scope"))
        doc = _w(os.path.join(root, "CLAUDE.md"), "E2E-DOC: 4-space indent {braces}")
        target = os.path.join(root, "f.py")
        _w(target, "")
        logs: list[str] = []
        orig = sa.print_ts

        def _cap(msg, *a, **k):
            logs.append(str(msg))
        sa.print_ts = _cap
        saved_cfg = _cfg.get_config
        _cfg.get_config = _fake_config
        try:
            _settings(allowed_tools="read_file,write_file,edit_file", read_paths=root, write_paths=root)
            r = FakeRunner(_agent())
            res = await _call(r, [
                {"prompt": f"Edit {target} to add a docstring.", "label": "scoped"},
                {"prompt": "Summarize today's weather.", "label": "unscoped"},
            ])
        finally:
            sa.print_ts = orig
            _cfg.get_config = saved_cfg
        check("call succeeded", res.error is None and len(r.turns) == 2)
        prompts = [t["prompt"] for t in r.turns]
        scoped = next(p for p in prompts if _task_section(p).startswith("Edit "))
        unscoped = next(p for p in prompts if _task_section(p).startswith("Summarize "))
        check("scoped worker prompt carries the CLAUDE.md block",
              f'<project-doc path="{doc}">' in scoped and "E2E-DOC: 4-space indent {braces}" in scoped)
        check("scoped worker: ## Task still last with the task text",
              _task_section(scoped) == f"Edit {target} to add a docstring.")
        check("unscoped worker prompt has no block",
              "<project-doc" not in unscoped and sa.PROJECT_DOCS_NOTE not in unscoped)
        launch = [l for l in logs if "launching worker" in l]
        check("launch log names the injected file for the scoped worker",
              any("[subagent scoped]" in l and f"docs={doc}" in l for l in launch))
        check("launch log shows docs=- for the unscoped worker",
              any("[subagent unscoped]" in l and "docs=-" in l for l in launch))


def main() -> int:
    ts._ensure_loaded()
    saved = dict(ts._VALUES.get("spawn_subagents") or {})
    try:
        test_path_extraction()
        test_inject_and_order()
        test_nothing_cases()
        test_caps()
        test_prompt_shape_and_braces()
        asyncio.run(test_end_to_end())
    finally:
        if saved:
            ts._VALUES["spawn_subagents"] = saved
        else:
            ts._VALUES.pop("spawn_subagents", None)
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
