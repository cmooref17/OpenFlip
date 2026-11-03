"""Verification for the `anthropic` provider's auth modes
(openflip/anthropic_conversation.build_auth + config_global getters + the
agent.json field). No network.

Standalone runnable script:

    ./.venv/bin/python tests/test_anthropic_auth_mode.py

What this guards:

  (a) "api_key" mode is a plain API request: x-api-key, no Authorization
      bearer, openflip's own user agent, only the public betas the body
      needs, and NO billing block ahead of the system prompt;
  (b) "oauth" mode is unchanged: bearer token, claude-code user agent, the
      two extra betas, and the billing block — exactly the historical bytes;
  (c) precedence: agent.json `anthropic_auth` > config.json
      `integrations.anthropic.auth` > "oauth"; junk values fall through;
  (d) the API key comes from config.json or ANTHROPIC_API_KEY;
  (e) Agent.from_file reads and re-serializes `anthropic_auth`;
  (f) in api_key mode with no key, chat_stream yields an auth
      FrameworkErrorEvent before any request;
  (g) the recall selector routes api_key mode through build_auth("api_key").
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip import config_global
from openflip.anthropic_conversation import build_auth, auth_mode_for, AnthropicConversation, _DEFAULT_USER_AGENT
from openflip._anthropic_stream import FrameworkErrorEvent
from openflip.agent import Agent

FAILURES: list[str] = []


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        FAILURES.append(label)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="anth-auth-")

    print("\n(a) api_key mode request shape")
    h, blocks = build_auth("api_key", api_key="sk-test", wants_1m=True)
    check("x-api-key header", h.get("x-api-key") == "sk-test")
    check("no bearer authorization", "authorization" not in {k.lower() for k in h})
    check("openflip user agent, not claude-code", h["User-Agent"] == "openflip-anthropic" and "claude-code" not in h["User-Agent"])
    betas = set(h["anthropic-beta"].split(","))
    check("only public betas", betas == {"extended-cache-ttl-2025-04-11", "compact-2026-01-12", "context-1m-2025-08-07"})
    check("no claude-code / oauth betas", not ({"claude-code-20250219", "oauth-2025-04-20"} & betas))
    check("no billing block", blocks == [])

    print("\n(b) oauth mode unchanged")
    h2, blocks2 = build_auth("oauth", access_token="tok", wants_1m=False)
    check("bearer token", h2.get("authorization") == "Bearer tok")
    check("claude-code user agent", h2["User-Agent"] == _DEFAULT_USER_AGENT and h2["User-Agent"].startswith("claude-code/"))
    check("historical beta list", h2["anthropic-beta"] == "claude-code-20250219,oauth-2025-04-20,extended-cache-ttl-2025-04-11,compact-2026-01-12")
    check("billing block present", len(blocks2) == 1 and blocks2[0]["text"].startswith("x-anthropic-billing-header: cc_version="))

    print("\n(c) precedence")
    saved_cfg = dict(config_global._config) if config_global._config else None
    config_global._config = {"integrations": {"anthropic": {"auth": "api_key"}}}
    check("config sets api_key", config_global.get_anthropic_auth_mode(None) == "api_key")
    check("agent value wins", config_global.get_anthropic_auth_mode("oauth") == "oauth")
    check("junk agent value falls to config", config_global.get_anthropic_auth_mode("nope") == "api_key")
    config_global._config = {"integrations": {"anthropic": {"auth": "garbage"}}}
    check("junk config falls to oauth default", config_global.get_anthropic_auth_mode(None) == "oauth")
    config_global._config = {}
    check("no config → oauth (unchanged default)", config_global.get_anthropic_auth_mode(None) == "oauth")
    check("auth_mode_for reads agent.anthropic_auth", auth_mode_for(SimpleNamespace(anthropic_auth="api_key")) == "api_key")

    print("\n(d) api key sources")
    os.environ.pop("ANTHROPIC_API_KEY", None)
    config_global._config = {"integrations": {"anthropic": {"api_key": " sk-cfg "}}}
    check("config key (stripped)", config_global.get_anthropic_api_key() == "sk-cfg")
    config_global._config = {}
    os.environ["ANTHROPIC_API_KEY"] = "sk-env"
    check("env key", config_global.get_anthropic_api_key() == "sk-env")
    os.environ.pop("ANTHROPIC_API_KEY", None)
    check("none → empty", config_global.get_anthropic_api_key() == "")

    print("\n(e) agent.json field")
    d = os.path.join(tmp, "agents", "auth-test")
    os.makedirs(d, exist_ok=True)
    for name in ("SOUL.md", "AGENT.md", "TOOLS.md"):
        open(os.path.join(d, name), "w").write("x\n")
    json.dump({"id": "auth-test", "display_name": "Auth Test", "provider": "anthropic",
               "model": "anthropic/claude-sonnet-5-5", "anthropic_auth": "API_KEY",
               "system_files": ["SOUL.md"]}, open(os.path.join(d, "agent.json"), "w"))
    ag = Agent.from_file(os.path.join(d, "agent.json"))
    check("field parsed + normalized", ag.anthropic_auth == "api_key")
    ag.save()
    check("serialized back", json.load(open(ag.path)).get("anthropic_auth") == "api_key")
    json.dump({"id": "auth-test2", "display_name": "x", "provider": "anthropic",
               "model": "anthropic/claude-sonnet-5-5", "system_files": ["SOUL.md"]}, open(os.path.join(d, "agent.json"), "w"))
    ag2 = Agent.from_file(os.path.join(d, "agent.json"))
    ag2.save()
    check("absent → None and not serialized", ag2.anthropic_auth is None and "anthropic_auth" not in json.load(open(ag2.path)))

    print("\n(f) api_key mode without a key fails closed before any request")
    os.makedirs(os.path.join(d, "conversations"), exist_ok=True)
    conv = AnthropicConversation("discord:1", ag)
    config_global._config = {}

    async def first_event():
        async for ev in conv.chat_stream(tools=None, think=None):
            return ev
    ev = asyncio.run(first_event())
    check("auth FrameworkErrorEvent", isinstance(ev, FrameworkErrorEvent) and ev.kind == "auth" and "API key not configured" in ev.message)

    print("\n(g) recall selector in api_key mode")
    from openflip import memory_recall_api, memory_recall
    seen = {}

    class _Resp:
        status = 200
        async def json(self, content_type=None):
            return {"content": [{"type": "text", "text": '{"selected_memories": []}'}], "stop_reason": "end_turn"}
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False

    class _Session:
        def post(self, url, json=None, headers=None, timeout=None):
            seen["headers"] = headers; seen["body"] = json
            return _Resp()

    async def fake_http_session():
        return _Session()
    from openflip import utils as _utils
    orig = _utils.http_session
    _utils.http_session = fake_http_session
    config_global._config = {"integrations": {"anthropic": {"api_key": "sk-sel"}}}
    try:
        out = asyncio.run(memory_recall_api.call_selector("claude-sonnet-5-5", "- a.md", "q", anthropic_auth="api_key"))
    finally:
        _utils.http_session = orig
    check("selector used x-api-key", seen.get("headers", {}).get("x-api-key") == "sk-sel")
    check("selector system has no billing block", seen["body"]["system"][0]["text"].startswith("You") or "billing-header" not in seen["body"]["system"][0]["text"])
    check("selector answer returned", "selected_memories" in out)

    config_global._config = saved_cfg or {}
    shutil.rmtree(tmp, ignore_errors=True)
    print("\nRESULT:", "ALL PASS" if not FAILURES else f"FAILED ({len(FAILURES)}):")
    for f in FAILURES:
        print("  -", f)
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
