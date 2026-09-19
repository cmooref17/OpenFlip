"""Verification for the per-conversation session override layer
(openflip/session_overrides.py + the three provider conversation classes).

Standalone runnable script (no pytest in this venv):

    .lvenv/bin/python tests/test_session_overrides.py

What this guards:

  (a) validation — provider-scoped keys, coercion of `200k`-style ints,
      on/off booleans, `k=v` option pairs, and the "can't change provider
      via a session model" rule;
  (b) precedence — per-turn model override > session override > per-model
      config.json entry (looked up with the EFFECTIVE model) > agent.json;
      the 1M beta header, context window, compaction trigger, max_tokens
      and effort ALL follow the effective model (the original bug: a Sonnet
      session on a 1M Opus agent still ran with the 1M header + 1M trigger);
  (c) a session `context_window` on Anthropic derives its own compaction
      trigger (window - reserve) instead of keeping the agent model's;
  (d) persistence — overrides round-trip through the .meta.json sidecar,
      the block is omitted when empty, the legacy top-level `effort_override`
      key is migrated on load and never written back, and the `/effort`
      compatibility property still works on top of the new store;
  (e) the memory switch — effective_memory_enabled() and
      pipeline.strip_memory_tools drop exactly the memory tools;
  (f) ollama — session context_window → num_ctx, session options merge over
      agent.ollama_options, the model resyncs at chat() time;
  (g) the shared /session command body and the ingress bulk-apply.

Uses the real config.json models block for the anthropic lookups (the
claude-opus-4-8-1m → 1M / claude-sonnet-4-6 → 200k entries).
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip.session_overrides import (
    normalize_override, session_command_text, parse_session_args,
    compaction_trigger_for_window, provider_for_model,
)
from openflip.config_global import get_config, get_model_context_window, get_compaction_trigger, get_effort, get_max_tokens
from openflip.anthropic_conversation import AnthropicConversation
from openflip.openai_conversation import OpenAIConversation
from openflip.conversation import DiscordConversation
from openflip.pipeline import strip_memory_tools, MEMORY_TOOL_NAMES
from openflip.tools import TOOL_REGISTRY

FAILURES: list[str] = []


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        FAILURES.append(label)


def _agent(tmp: str, *, provider: str, model: str, memory_enabled: bool = True, ollama_options: dict | None = None):
    d = os.path.join(tmp, "agents", f"t-{provider}")
    os.makedirs(os.path.join(d, "conversations"), exist_ok=True)
    return SimpleNamespace(
        id=f"t-{provider}", display_name=f"Test {provider}", path=os.path.join(d, "agent.json"),
        provider=provider, model=model, system_message="sys", memory_enabled=memory_enabled,
        ollama_options=dict(ollama_options or {}), think=None,
    )


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="sessov-")
    cfg_models = get_config().get("models") or {}
    have_real_models = "claude-opus-4-8-1m" in cfg_models and "claude-sonnet-4-6" in cfg_models
    reserve = get_config().get("compaction_reserve_tokens", 20_000)

    print("\n(a) validation")
    v, e = normalize_override("anthropic", "context_window", "200k")
    check("200k → 200000", v == 200_000 and not e)
    v, e = normalize_override("anthropic", "context_window", "1m")
    check("1m → 1000000", v == 1_000_000)
    v, e = normalize_override("anthropic", "context_window", "12")
    check("tiny window rejected", v is None and "≥" in e)
    v, e = normalize_override("anthropic", "compaction_trigger", 40_000)
    check("trigger below Anthropic floor rejected", v is None)
    v, e = normalize_override("anthropic", "max_tokens", 200_000)
    check("anthropic max_tokens above ceiling rejected", v is None)
    v, e = normalize_override("openai", "max_tokens", 200_000)
    check("openai max_tokens has no anthropic ceiling", v == 200_000)
    v, e = normalize_override("anthropic", "effort", "XHigh")
    check("effort case-folds", v == "xhigh")
    v, e = normalize_override("anthropic", "effort", "minimal")
    check("anthropic rejects openai-only effort", v is None)
    v, e = normalize_override("openai", "effort", "minimal")
    check("openai accepts minimal", v == "minimal")
    v, e = normalize_override("ollama", "effort", "high")
    check("ollama has no effort key", v is None and "not a session setting" in e)
    v, e = normalize_override("anthropic", "memory", "off")
    check("memory off → False", v is False)
    v, e = normalize_override("anthropic", "memory", True)
    check("memory JSON true → True", v is True)
    v, e = normalize_override("anthropic", "memory", "maybe")
    check("memory junk rejected", v is None)
    v, e = normalize_override("ollama", "options", "temperature=0.7 num_predict=512 stop=xyz")
    check("options k=v parse + coercion", v == {"temperature": 0.7, "num_predict": 512, "stop": "xyz"})
    v, e = normalize_override("ollama", "options", {"temperature": 0.2})
    check("options JSON dict accepted", v == {"temperature": 0.2})
    v, e = normalize_override("ollama", "options", "temperature")
    check("options without = rejected", v is None)
    v, e = normalize_override("anthropic", "model", "openai/gpt-5.1")
    check("session model can't change provider", v is None and "can't change provider" in e)
    v, e = normalize_override("anthropic", "model", " anthropic/claude-sonnet-4-6 ")
    check("model trimmed", v == "anthropic/claude-sonnet-4-6")
    v, e = normalize_override("anthropic", "bogus", "1")
    check("unknown key names the valid ones", v is None and "context_window" in e)
    check("provider_for_model bare claude", provider_for_model("claude-sonnet-4-6") == "anthropic")
    check("provider_for_model ollama tag", provider_for_model("qwen3.5:cloud") == "ollama")

    print("\n(b) anthropic precedence — all lookups follow the effective model")
    ag = _agent(tmp, provider="anthropic", model="anthropic/claude-opus-4-8-1m")
    conv = AnthropicConversation("external:test", ag)
    base_window = get_model_context_window(ag.model, "anthropic")
    base_trigger = get_compaction_trigger(ag.model, "anthropic")
    base_effort = get_effort(ag.model, "anthropic")
    check("no overrides: window == config for agent model", conv.effective_context_window() == base_window)
    check("no overrides: 1M header follows agent model suffix", conv._wants_1m_context() is True)
    check("no overrides: trigger == config for agent model", conv.effective_compaction_trigger() == base_trigger)
    check("no overrides: effort == config for agent model", conv._effort_level() == base_effort)
    check("no overrides: max_tokens == config for agent model", conv.effective_max_tokens() == get_max_tokens(ag.model, "anthropic"))
    check("report has one row per anthropic key", [r[0] for r in conv.overrides_report()] == ["model", "context_window", "compaction_trigger", "max_tokens", "effort", "memory"])

    ok, msg = conv.set_override("model", "anthropic/claude-sonnet-4-6")
    check("set session model ok", ok)
    check("session model is effective", conv._effective_raw_model() == "anthropic/claude-sonnet-4-6")
    check("normalized model id drops prefix", conv._normalize_model(conv._effective_raw_model()) == "claude-sonnet-4-6")
    check("1M header OFF for a non-1m session model", conv._wants_1m_context() is False)
    if have_real_models:
        check("window follows session model (200k)", conv.effective_context_window() == cfg_models["claude-sonnet-4-6"]["context_window"] == 200_000)
        check("trigger follows session model", conv.effective_compaction_trigger() == get_compaction_trigger("claude-sonnet-4-6", "anthropic"))
    else:
        print("  [SKIP] real config.json models block not present; window/trigger-by-model checks skipped")
    check("effort follows session model config", conv._effort_level() == get_effort("claude-sonnet-4-6", "anthropic"))

    conv.set_model_override("anthropic/claude-opus-4-8-1m")
    check("per-turn override beats session model", conv._effective_raw_model() == "anthropic/claude-opus-4-8-1m")
    check("per-turn override brings its 1M header back", conv._wants_1m_context() is True)
    check("report labels per-turn source", conv.overrides_report()[0][2] == "per-turn override")
    conv.set_model_override(None)
    check("per-turn override cleared → session model again", conv._effective_raw_model() == "anthropic/claude-sonnet-4-6")

    print("\n(c) session context_window derives its own compaction trigger")
    conv.unset_override("model")
    ok, _ = conv.set_override("context_window", "200k")
    check("set window ok", ok)
    check("window override wins over model config", conv.effective_context_window() == 200_000)
    check("trigger derived from window - reserve", conv.effective_compaction_trigger() == max(200_000 - reserve, 50_000) == compaction_trigger_for_window(200_000))
    check("1M header unchanged by a window override (model still -1m)", conv._wants_1m_context() is True)
    conv.set_override("compaction_trigger", "120k")
    check("explicit trigger beats derived", conv.effective_compaction_trigger() == 120_000)
    conv.set_override("max_tokens", "8000")
    check("max_tokens override", conv.effective_max_tokens() == 8_000)
    conv.set_override("effort", "low")
    check("effort override", conv._effort_level() == "low")
    internal = AnthropicConversation("internal:test", ag)
    internal.set_override("compaction_trigger", "900000", persist=False)
    check("internal sessions still take min(session trigger, internal trigger)",
          internal.effective_compaction_trigger() == min(900_000, get_config().get("internal_compaction_trigger", 150_000) or 900_000))

    print("\n(d) persistence + legacy migration + /effort compat")
    meta_path = conv._meta_path()
    check("meta written on set", os.path.isfile(meta_path))
    raw = json.load(open(meta_path))
    check("meta has overrides block", isinstance(raw.get("overrides"), dict) and raw["overrides"].get("effort") == "low")
    check("legacy effort_override key NOT written", "effort_override" not in raw)
    conv2 = AnthropicConversation("external:test", ag)
    conv2.load()
    check("reload restores overrides", conv2.overrides == conv.overrides)
    check("reload restores effort_override property", conv2.effort_override == "low")
    conv2.effort_override = "max"
    conv2._save_meta()
    conv3 = AnthropicConversation("external:test", ag)
    conv3.load()
    check("/effort property writes through to the store", conv3.overrides.get("effort") == "max")
    conv3.effort_override = None
    check("/effort default clears the key", "effort" not in conv3.overrides)
    n = conv3.clear_overrides()
    check("clear removes all + returns count", n == 3 and conv3.overrides == {})
    raw = json.load(open(meta_path))
    check("meta block omitted when empty", "overrides" not in raw)
    # Legacy file shape: top-level effort_override only.
    with open(meta_path, "w") as f:
        json.dump({"effort_override": "medium", "last_usage": {"total_input": 5}}, f)
    conv4 = AnthropicConversation("external:test", ag)
    conv4.load()
    check("legacy effort_override migrated on load", conv4.overrides == {"effort": "medium"} and conv4.effort_override == "medium")
    check("legacy load keeps last_usage", conv4.last_usage == {"total_input": 5})
    # Junk in a hand-edited meta is dropped, not trusted.
    with open(meta_path, "w") as f:
        json.dump({"overrides": {"context_window": 5, "memory": "off", "nope": 1}}, f)
    conv5 = AnthropicConversation("external:test", ag)
    conv5.load()
    check("stored junk re-validated on load", conv5.overrides == {"memory": False})

    print("\n(e) memory switch")
    check("memory default follows agent", conv5.effective_memory_enabled() is False)
    conv5.unset_override("memory")
    check("memory unset → agent default", conv5.effective_memory_enabled() is True)
    ag_nomem = _agent(tmp, provider="anthropic", model="anthropic/claude-opus-4-8", memory_enabled=False)
    c_nomem = AnthropicConversation("discord:1", ag_nomem)
    c_nomem.set_override("memory", "on", persist=False)
    check("session can turn memory on when agent has it off", c_nomem.effective_memory_enabled() is True)
    all_funcs = [t.func for t in TOOL_REGISTRY.values()]
    stripped = strip_memory_tools(all_funcs)
    mem_in_registry = {n for n in TOOL_REGISTRY if n in MEMORY_TOOL_NAMES}
    check("registry actually has memory tools to strip", len(mem_in_registry) >= 5)
    check("strip_memory_tools removes exactly the memory tools",
          len(all_funcs) - len(stripped) == len(mem_in_registry)
          and not any(f.__name__ in MEMORY_TOOL_NAMES for f in stripped))
    check("strip_memory_tools preserves order of the rest",
          [f.__name__ for f in stripped] == [f.__name__ for f in all_funcs if f.__name__ not in MEMORY_TOOL_NAMES])

    print("\n(f) ollama")
    ag_ol = _agent(tmp, provider="ollama", model="qwen3.5:cloud", ollama_options={"temperature": 0.5})
    oc = DiscordConversation("discord:2", ag_ol)
    check("ollama ctor options from agent", oc.options.get("temperature") == 0.5)
    oc.set_override("context_window", "32k", persist=False)
    oc.set_override("options", "temperature=0.9 num_predict=256", persist=False)
    from openflip.conversation import _ollama_options_with_context
    merged = _ollama_options_with_context(ag_ol, oc)
    check("session context_window → num_ctx", merged.get("num_ctx") == 32_000)
    check("session options merge over agent options", merged.get("temperature") == 0.9 and merged.get("num_predict") == 256)
    check("ollama effective window", oc.effective_context_window() == 32_000)
    ok, msg = oc.set_override("model", "claude-sonnet-4-6", persist=False)
    check("ollama session model rejects a claude id", not ok)
    ok, msg = oc.set_override("model", "llama3.1:8b", persist=False)
    check("ollama session model accepts a tag", ok and oc._effective_raw_model() == "llama3.1:8b")
    oc.reapply_agent()
    check("reapply_agent honors session model + options", oc.model == "llama3.1:8b" and oc.options.get("num_ctx") == 32_000)
    oc.set_override("memory", "off", persist=True)
    check("ollama meta sidecar written for overrides", os.path.isfile(oc._meta_path()))
    oc2 = DiscordConversation("discord:2", ag_ol)
    oc2.load()
    check("ollama overrides reload from meta", oc2.overrides.get("memory") is False and oc2.overrides.get("model") == "llama3.1:8b")
    oc2.clear_history()
    check("ollama clear_history removes the sidecar", not os.path.isfile(oc2._meta_path()))
    check("ollama report keys", [r[0] for r in oc.overrides_report()] == ["model", "context_window", "options", "memory"])

    print("\n(f2) openai")
    ag_oa = _agent(tmp, provider="openai", model="openai/gpt-5.1")
    oa = OpenAIConversation("discord:3", ag_oa)
    oa.set_override("effort", "xhigh", persist=False)
    check("openai chat-completions maps xhigh→high", oa._chat_effort() == "high")
    check("openai codex keeps xhigh", oa._codex_effort_level() == "xhigh")
    oa.set_override("effort", "minimal", persist=False)
    check("openai codex maps minimal→low", oa._codex_effort_level() == "low")
    oa.set_override("max_tokens", "4096", persist=False)
    check("openai max_tokens override", oa._max_output_tokens() == 4096)
    oa.set_override("model", "openai/gpt-5.2", persist=False)
    oa._resync_model()
    check("openai model resync uses session model", oa.model == "gpt-5.2")
    check("openai effort_override property exists (so /effort works)", oa.effort_override == "minimal")
    check("openai report keys", [r[0] for r in oa.overrides_report()] == ["model", "context_window", "max_tokens", "effort", "memory"])

    print("\n(g) /session command body + ingress bulk-apply")
    check("parse: bare", parse_session_args("") == ("show", "", ""))
    check("parse: set with spaced value", parse_session_args("set options temperature=0.7 num_predict=5") == ("set", "options", "temperature=0.7 num_predict=5"))
    check("parse: unset", parse_session_args("unset model") == ("unset", "model", ""))
    t = session_command_text(None, "show")
    check("no conversation → helpful message", "No active conversation" in t)
    t = session_command_text(conv5, "show")
    check("show lists every key + usage", "`model`" in t and "`memory`" in t and "/session set" in t)
    t = session_command_text(conv5, "set", "model", "openai/gpt-5.1")
    check("set rejects cross-provider model with ⚠️", t.startswith("⚠️") and "provider" in t)
    t = session_command_text(conv5, "set", "context_window", "200k")
    check("set window reports derived trigger", "set to 200,000" in t and "Compaction now triggers" in t)
    t = session_command_text(conv5, "set", "memory", "off")
    check("set memory off notes tools hidden", "hidden" in t)
    t = session_command_text(conv5, "unset", "effort")
    check("unset of an unset key says so", "had no session override" in t)
    t = session_command_text(conv5, "unset", "bogus")
    check("unset unknown key rejected", t.startswith("⚠️"))
    t = session_command_text(conv5, "clear")
    check("clear reports count", "Cleared 2" in t and conv5.overrides == {})
    t = session_command_text(conv5, "dance")
    check("unknown action rejected", t.startswith("⚠️"))
    changed, errs = conv5.apply_overrides({"model": "anthropic/claude-sonnet-4-6", "memory": False, "effort": "nope", "max_tokens": 1000})
    check("bulk apply stores valid, reports invalid", changed and conv5.overrides == {"model": "anthropic/claude-sonnet-4-6", "memory": False, "max_tokens": 1000} and len(errs) == 1)
    changed, errs = conv5.apply_overrides({"model": "anthropic/claude-sonnet-4-6", "memory": False, "max_tokens": 1000})
    check("bulk re-apply of same block is a no-op", not changed and not errs)
    changed, errs = conv5.apply_overrides("junk")
    check("bulk apply non-dict rejected", not changed and errs)

    shutil.rmtree(tmp, ignore_errors=True)
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
