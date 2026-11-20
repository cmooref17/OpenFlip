"""Skills core (openflip/skills.py, design docs/skills.md).

Standalone runnable (no pytest in this venv):

    .lvenv/bin/python tests/test_skills.py

Real folders on disk, real SKILL.md files: discovery, frontmatter, scope
precedence, the listing and its budget, rendering + substitutions, and the
"already loaded" dedupe.
"""
from __future__ import annotations

import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openflip.skills as sk  # noqa: E402

sk._config = lambda: {}          # never read the live config.json
os.environ.pop("OPENFLIP_DISABLE_SKILLS", None)


def _tree():
    root = tempfile.mkdtemp(prefix="openflip_skills_test_")
    agent_dir = os.path.join(root, "agents", "bot")
    os.makedirs(agent_dir)
    return root, agent_dir


def _skill(base, scope, folder, text):
    d = os.path.join(base, "agents", "_shared", "skills", folder) if scope == "shared" \
        else os.path.join(base, "agents", "bot", "skills", folder)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return d


def test_frontmatter_shapes():
    f, b, w = sk.parse_frontmatter("---\nname: x\ndescription: \"Do X\"\n---\nbody\n")
    assert f == {"name": "x", "description": "Do X"} and b == "body\n" and not w
    f, b, w = sk.parse_frontmatter("no frontmatter here")
    assert f == {} and b == "no frontmatter here" and not w
    f, b, w = sk.parse_frontmatter("---\nname: x\nthis line is junk\n---\nbody")
    assert f == {} and b == "body" and "no fields" in w, "malformed = loads with no fields (CC)"
    f, b, w = sk.parse_frontmatter("---\nname: x\nbody without a close")
    assert f == {} and "no closing" in w
    f, _, _ = sk.parse_frontmatter("---\nname: x\nallowed-tools:\n  - Read\n  - Grep\n---\nb")
    assert f["name"] == "x", "list fields (ignored ones) don't break the parse"


def test_discovery_precedence_and_fallbacks():
    root, ad = _tree()
    _skill(root, "shared", "deploy", "---\ndescription: shared deploy\n---\nshared body")
    _skill(root, "agent", "deploy", "---\ndescription: my deploy\n---\nmine")
    _skill(root, "shared", "notes", "# Take notes when asked\n\nsteps")
    _skill(root, "agent", "Bad Name", "---\ndescription: x\n---\nb")
    _skill(root, "agent", "renamed", "---\nname: fancy\ndescription: d\n---\nb")
    os.makedirs(os.path.join(ad, "skills", "empty_dir"))
    found, warns = sk.load_skills(ad)
    assert set(found) == {"deploy", "notes", "fancy"}, sorted(found)
    assert found["deploy"].scope == "agent" and found["deploy"].body == "mine", "agent beats shared"
    assert found["notes"].description == "Take notes when asked", "description falls back to line 1"
    assert any("bad name" in w for w in warns)


def test_disabled_and_kill_switches():
    root, ad = _tree()
    _skill(root, "agent", "a", "---\ndescription: a\n---\nb")
    _skill(root, "agent", "b", "---\ndescription: b\n---\nb")
    found, _ = sk.load_skills(ad, SimpleNamespace(skills={"disabled": ["b"]}))
    assert set(found) == {"a"}
    assert sk.load_skills(ad, SimpleNamespace(skills={"enabled": False}))[0] == {}
    os.environ["OPENFLIP_DISABLE_SKILLS"] = "1"
    try:
        assert sk.load_skills(ad)[0] == {}
    finally:
        os.environ.pop("OPENFLIP_DISABLE_SKILLS", None)


def test_listing_sorted_and_hides_manual_only():
    root, ad = _tree()
    _skill(root, "agent", "zeta", "---\ndescription: last one\nwhen_to_use: say zeta\n---\nb")
    _skill(root, "agent", "alpha", "---\ndescription: first one\n---\nb")
    _skill(root, "agent", "ship", "---\ndescription: ships it\ndisable-model-invocation: YES\n---\nb")
    _skill(root, "agent", "bg", "---\ndescription: background\nuser-invocable: off\n---\nb")
    found, _ = sk.load_skills(ad)
    assert found["ship"].model_invocable is False and found["bg"].user_invocable is False
    text, warns = sk.listing_block(found)
    assert text.startswith("# Skills\n\n" + sk.LISTING_HEADER) and not warns
    assert text.index("- alpha:") < text.index("- bg:") < text.index("- zeta:")
    assert "- zeta: last one say zeta" in text, "when_to_use is appended"
    assert "ship" not in text, "disable-model-invocation skills are not listed (CC)"
    assert sk.listing_block({})[0] == ""


def test_listing_budget_and_cap():
    root, ad = _tree()
    _skill(root, "agent", "big", "---\ndescription: " + "x" * 3000 + "\n---\nb")
    _skill(root, "agent", "small", "---\ndescription: tiny\n---\nb")
    found, _ = sk.load_skills(ad)
    full, _ = sk.listing_block(found, budget=100000)
    assert "x" * sk.DESC_CAP in full and "x" * (sk.DESC_CAP + 1) not in full, "1,536-char cap"
    text, warns = sk.listing_block(found, budget=200)
    assert "- big\n" in text or text.endswith("- big"), "name kept, description dropped"
    assert "- small: tiny" in text, "the short description survives"
    assert warns and "big" in warns[0]


def test_render_substitutions():
    s = sk.Skill(name="t", description="", when_to_use="", scope="agent", dir="/sk/t",
                 body="all=$ARGUMENTS first=$0 second=$ARGUMENTS[1] gone=$5 cost=\\$1 "
                      "dir=${SKILL_DIR} cc=${CLAUDE_SKILL_DIR} sid=${SESSION_ID}")
    out = sk.render(s, 'one "two words"', session_id="discord:9")
    assert out.startswith("Base directory for this skill: /sk/t\n\n")
    assert "all=one \"two words\"" in out and "first=one" in out and "second=two words" in out
    assert "gone=$5" in out, "missing index stays literal (CC)"
    assert "cost=$1" in out, "backslash escapes a placeholder"
    assert "dir=/sk/t" in out and "cc=/sk/t" in out and "sid=discord:9" in out
    assert "ARGUMENTS:" not in out


def test_render_appends_unused_args_and_keeps_injected_dollars_literal():
    s = sk.Skill(name="t", description="", when_to_use="", scope="agent", dir="/d", body="Do the thing.")
    assert sk.render(s, "now please").endswith("Do the thing.\n\nARGUMENTS: now please")
    assert sk.render(s).endswith("Do the thing.")
    s2 = sk.Skill(name="t", description="", when_to_use="", scope="agent", dir="/d", body="Say $0")
    assert sk.render(s2, '"$ARGUMENTS later"').endswith("Say $ARGUMENTS later"), "no re-expansion"


def test_already_loaded_dedupe():
    sk.forget_loaded("c1")
    assert sk.note_loaded("c1", "t", "text A") is False
    assert sk.note_loaded("c1", "t", "text A") is True, "identical re-invoke"
    assert sk.note_loaded("c1", "t", "text B") is False, "changed render sends again"
    assert sk.loaded_names("c1") == ["t"]
    sk.forget_loaded("c1")
    assert sk.loaded_names("c1") == [] and sk.note_loaded("", "t", "x") is False


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    print(f"{'FAILED' if failed else 'OK'}: {failed} failed")
    sys.exit(1 if failed else 0)
