"""Skills wiring (docs/skills.md): agent.json round trip, the use_skill tool
through the real executor, operator /<name>, /skills, and the runtime hooks.

Standalone runnable:  .lvenv/bin/python tests/test_skills_turn.py
"""
from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openflip.skills as sk  # noqa: E402
sk._config = lambda: {}
os.environ.pop("OPENFLIP_DISABLE_SKILLS", None)
from openflip.agent import Agent  # noqa: E402
from openflip.session import Session  # noqa: E402
from openflip.tool_executor import (  # noqa: E402
    CURRENT_AGENT, CURRENT_SESSION, execute_tool_calls, build_model_feedback,
)
from openflip import text_commands  # noqa: E402

_ROOT = tempfile.mkdtemp(prefix="openflip_skills_turn_")
_AD = os.path.join(_ROOT, "agents", "bot")
os.makedirs(_AD)


def _skill(folder, text, scope="agent"):
    d = os.path.join(_ROOT, "agents", "_shared" if scope == "shared" else "bot", "skills", folder)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "SKILL.md"), "w", encoding="utf-8") as f:
        f.write(text)


_skill("greet", "---\ndescription: Greet someone by name\nargument-hint: [who]\n---\nSay hi to $0 from ${SKILL_DIR}.")
_skill("ship", "---\ndescription: Ship it\ndisable-model-invocation: true\n---\nShip steps.")
_skill("lore", "---\ndescription: Background lore\nuser-invocable: false\n---\nLore text.")
_skill("status", "---\ndescription: Shadows a built-in\n---\nnope", scope="shared")


def _agent():
    return SimpleNamespace(id="bot", path=os.path.join(_AD, "agent.json"), tool_response_mode="caption",
                           memory_enabled=False, provider="anthropic", model="m", display_name="Bot",
                           hooks={}, skills={})


def _call(agent, name, args, conv="discord:5"):
    sess = Session(transport="discord", transport_id="5", conversation_id=conv, speaker_id=1,
                   speaker_role_ids=[], is_owner=True, is_dm=True, display_name="o")
    msg = SimpleNamespace(tool_calls=[SimpleNamespace(function_name="use_skill",
                                                      args={"skill": name, "args": args}, tool_use_id="t1")])

    async def go():
        ta, ts = CURRENT_AGENT.set(agent), CURRENT_SESSION.set(sess)
        try:
            return await execute_tool_calls(agent=agent, conversation=None, ai_message=msg,
                                            callable_tool_names={"use_skill"}, channel=None,
                                            speaker_id=1, session_id=conv, silent=True)
        finally:
            CURRENT_AGENT.reset(ta)
            CURRENT_SESSION.reset(ts)
    (name_, res), = asyncio.run(go())
    return res, build_model_feedback(name_, res)


def test_use_skill_renders_then_dedupes():
    sk.forget_loaded("discord:5")
    res, fb = _call(_agent(), "greet", "Flop")
    assert res.ok, res.error
    assert fb.startswith("Base directory for this skill: ") and "Say hi to Flop from " in fb
    assert os.path.join("skills", "greet") in fb, "${SKILL_DIR} expanded to the real folder"
    _, fb2 = _call(_agent(), "greet", "Flop")
    assert fb2.startswith("Skill /greet is already loaded above; instructions unchanged."), fb2
    _, fb3 = _call(_agent(), "greet", "Someone")
    assert "Say hi to Someone" in fb3, "different args = sent again"


def test_use_skill_errors():
    res, fb = _call(_agent(), "nope", "")
    assert not res.ok and "No skill named 'nope'" in res.error and "greet" in res.error
    res, _ = _call(_agent(), "ship", "")
    assert not res.ok and "/ship" in res.error, "manual-only skill refused for the model"
    res, _ = _call(_agent(), "/lore", "")
    assert res.ok, "leading slash tolerated; model-only skill loads for the model"


def test_operator_invocation_rules():
    reserved = set(text_commands._TEXT_COMMANDS) | {"/stop"}
    sk.forget_loaded("c9")
    blk, name = sk.operator_invocation("/greet Bob", _AD, None, is_owner=True,
                                       conversation_id="c9", reserved=reserved)
    assert name == "greet" and blk.startswith('<skill name="greet">') and "Say hi to Bob" in blk
    assert "The operator ran /greet with: Bob" in blk
    assert sk.note_loaded("c9", "greet", sk.render(sk.load_skills(_AD)[0]["greet"], "Bob", "c9")), \
        "operator run counts as loaded"
    assert sk.operator_invocation("/ship", _AD, None, is_owner=True, reserved=reserved)[1] == "ship", \
        "manual-only skills are exactly what /name is for"
    assert sk.operator_invocation("/greet Bob", _AD, None, is_owner=False, reserved=reserved) == ("", ""), \
        "non-owner: plain text"
    assert sk.operator_invocation("/lore", _AD, None, is_owner=True, reserved=reserved) == ("", ""), \
        "user-invocable: false"
    assert sk.operator_invocation("/status", _AD, None, is_owner=True, reserved=reserved) == ("", ""), \
        "built-in command names always win"
    assert sk.operator_invocation("hi /greet Bob", _AD, None, is_owner=True, reserved=reserved) == ("", ""), \
        "only at the start of the message"


def test_skills_report():
    out = text_commands.skills_report(_agent())
    assert "Bot's skills (4)" in out
    assert "- greet (agent; agent, you: /greet [who]): Greet someone by name" in out
    assert "- ship (agent; you: /ship): Ship it" in out and "- lore (agent; agent): Background lore" in out
    assert text_commands.is_text_command("/skills")


def test_agent_json_round_trip_keeps_hooks_and_skills():
    path = os.path.join(_ROOT, "rt", "agent.json")
    os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump({"id": "rt", "display_name": "RT", "model": "m", "provider": "anthropic",
                   "hooks": {"Stop": []}, "skills": {"disabled": ["x"]}}, f)
    a = Agent.from_file(path)
    assert a.skills == {"disabled": ["x"]} and a.hooks == {"Stop": []}
    assert a.save()
    data = json.load(open(path))
    assert data.get("skills") == {"disabled": ["x"]}, "save() must not drop skills"
    assert data.get("hooks") == {"Stop": []}, "save() must not drop hooks"


def test_runtime_wiring():
    from openflip.runtime import AgentRunner
    from openflip.pipeline import FRAMEWORK_TOOL_NAMES
    from openflip.tool_deferral import DEFAULT_ALWAYS_LOAD
    assert "use_skill" in FRAMEWORK_TOOL_NAMES and "use_skill" in DEFAULT_ALWAYS_LOAD
    src = inspect.getsource(AgentRunner._run_turn)
    lst = src.index('_cur_sources.append(("(skills)", _skills_block))')
    cap = src.index("InstructionSnapshot.capture(_cur_sources, _sys)")
    assert lst < cap and '_sys = _sys + "\\n\\n" + _skills_block' in src, "listing rides in the snapshot"
    inv = src.index("_skills_mod.operator_invocation(")
    recall = src.index("from .memory_recall import RecallState, recall_block")
    assert inv < recall and "if not _verbatim and owner and inbound is not None" in src
    assert "_skills_mod.forget_loaded(_sk_cid)" in src, "compaction resets loaded skills"
    assert "_skills_mod.forget_loaded(" in inspect.getsource(AgentRunner.reset_conversation)


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
            except Exception as e:
                failed += 1
                print(f"ERROR {name}: {type(e).__name__}: {e}")
    print(f"{'FAILED' if failed else 'OK'}: {failed} failed")
    sys.exit(1 if failed else 0)
