"""spawn_subagents worker read-scope (Session.read_scope → files.py ACL).

Standalone runnable script (no pytest in this venv):

    .lvenv/bin/python tests/test_subagent_read_scope.py

A subagent worker runs on the "internal" transport; a discord-only agent's
transport-keyed allowed_read_paths has no block for it, so without a read
scope a worker resolves to [] and can only see the agent-dir + temp fallback.
`make_subagent_session(read_scope=[...])` gives the worker an independent
read scope that OVERRIDES the agent path ACLs.

What this guards:
  (1) a worker with read_scope=[dir] can read a file inside dir;
  (2) it CANNOT read a file outside dir;
  (3) it CANNOT write inside dir (the scope is read-only);
  (4) denied_paths still blocks a path inside the read scope;
  (5) read_scope=[] falls back to the agent-dir + temp default (not the
      agent's discord-only allowed_read_paths, which the worker can't match);
  (6) list_files uses the same read gate;
  (7) a NON-worker session (read_scope=None) is unchanged — it resolves the
      agent's own allowed_read_paths;
  (8) the tool_settings read_paths validator + subagent _parse_read_paths.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip import tool_settings as ts
from openflip.session import Session, make_subagent_session
from openflip.tool_executor import CURRENT_AGENT, CURRENT_SESSION
from openflip.tools.files import read_file, write_file, list_files
from openflip.tools.subagent import _parse_read_paths

FAILURES: list[str] = []


def check(label: str, cond: bool) -> None:
    print(("  ok    " if cond else "  FAIL  ") + label)
    if not cond:
        FAILURES.append(label)


def _agent(agent_dir: str, *, denied=None):
    # Discord-only transport-keyed read ACL: a worker on "internal" matches no
    # block here, which is exactly why read_scope has to override it.
    return SimpleNamespace(
        id="orch",
        path=os.path.join(agent_dir, "agent.json"),
        allowed_read_paths={"discord": {"all_users": [agent_dir]}},
        allowed_write_paths={"discord": {"all_users": [agent_dir]}},
        denied_paths=list(denied or []),
    )


async def _as(agent, session, coro_factory):
    tok_a = CURRENT_AGENT.set(agent)
    tok_s = CURRENT_SESSION.set(session)
    try:
        return await coro_factory()
    finally:
        CURRENT_AGENT.reset(tok_a)
        CURRENT_SESSION.reset(tok_s)


async def _run() -> None:
    # Root the tree in the repo, NOT the system temp dir: the empty-read-scope
    # fallback (test 5) includes the system temp dir, which would otherwise
    # cover the whole tree and mask the fallback boundary.
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tmp = os.path.realpath(tempfile.mkdtemp(prefix="of_readscope_", dir=_repo))
    scope = os.path.join(tmp, "scope")
    outside = os.path.join(tmp, "outside")
    secret = os.path.join(scope, "secret")
    agent_dir = os.path.join(tmp, "agentdir")
    for d in (scope, outside, secret, agent_dir):
        os.makedirs(d, exist_ok=True)
    inside_f = os.path.join(scope, "inside.txt")
    outside_f = os.path.join(outside, "outside.txt")
    secret_f = os.path.join(secret, "secret.txt")
    agent_f = os.path.join(agent_dir, "own.txt")
    for f, body in ((inside_f, "IN"), (outside_f, "OUT"), (secret_f, "SSH"), (agent_f, "OWN")):
        with open(f, "w", encoding="utf-8") as fh:
            fh.write(body)

    agent = _agent(agent_dir)
    worker = make_subagent_session("orch", "task-1", ["read_file", "list_files"], read_scope=[scope])

    print("(1) worker reads inside its read_scope")
    res = await _as(agent, worker, lambda: read_file(inside_f))
    check("read inside scope ok", res.error is None and (res.model_feedback or "") == "IN")

    print("(2) worker cannot read outside its read_scope")
    res = await _as(agent, worker, lambda: read_file(outside_f))
    check("read outside scope denied", res.error is not None and "denied" in res.error.lower())

    print("(3) worker cannot write inside its read_scope (read-only)")
    res = await _as(agent, worker, lambda: write_file(os.path.join(scope, "new.txt"), "x"))
    check("write inside scope denied", res.error is not None and "denied" in res.error.lower())
    check("no file was created", not os.path.exists(os.path.join(scope, "new.txt")))

    print("(4) denied_paths still wins inside the read_scope")
    agent_denied = _agent(agent_dir, denied=[secret])
    worker_denied = make_subagent_session("orch", "task-2", ["read_file"], read_scope=[scope])
    res = await _as(agent_denied, worker_denied, lambda: read_file(secret_f))
    check("denied_paths blocks a file inside read_scope", res.error is not None and "denied" in res.error.lower())
    # control: a non-denied file in the same scope is still readable
    res = await _as(agent_denied, worker_denied, lambda: read_file(inside_f))
    check("non-denied file in scope still readable", res.error is None)

    print("(5) read_scope=[] falls back to agent-dir + temp (not the discord ACL)")
    worker_empty = make_subagent_session("orch", "task-3", ["read_file"], read_scope=[])
    res = await _as(agent, worker_empty, lambda: read_file(agent_f))
    check("empty scope reads agent dir via fallback", res.error is None and (res.model_feedback or "") == "OWN")
    res = await _as(agent, worker_empty, lambda: read_file(inside_f))
    check("empty scope cannot read the scope dir (not in fallback)", res.error is not None)

    print("(6) list_files uses the same read gate")
    res = await _as(agent, worker, lambda: list_files(scope))
    check("list inside scope ok", res.error is None)
    res = await _as(agent, worker, lambda: list_files(outside))
    check("list outside scope denied", res.error is not None and "denied" in res.error.lower())

    print("(7) non-worker session (read_scope=None) unchanged")
    human = Session(
        transport="discord", transport_id="123", conversation_id="discord:123",
        speaker_id=999, speaker_role_ids=[], is_owner=True, is_dm=True, display_name="owner",
    )
    check("human session has read_scope None", human.read_scope is None)
    # Human resolves the agent's OWN allowed_read_paths (discord block → agent_dir).
    res = await _as(agent, human, lambda: read_file(agent_f))
    check("human reads the agent's configured read path", res.error is None and (res.model_feedback or "") == "OWN")
    res = await _as(agent, human, lambda: read_file(inside_f))
    check("human cannot read outside the agent's read path", res.error is not None)

    print("(8) read_paths validator + _parse_read_paths")
    check("validator accepts empty", ts.coerce_and_validate("spawn_subagents", "read_paths", "")[1] is None)
    check("validator accepts an existing absolute dir",
          ts.coerce_and_validate("spawn_subagents", "read_paths", scope)[1] is None)
    check("validator rejects a relative path",
          ts.coerce_and_validate("spawn_subagents", "read_paths", "relative/dir")[1] is not None)
    check("validator rejects a non-existent dir",
          ts.coerce_and_validate("spawn_subagents", "read_paths", os.path.join(tmp, "nope"))[1] is not None)
    check("validator rejects a file (not a dir)",
          ts.coerce_and_validate("spawn_subagents", "read_paths", inside_f)[1] is not None)
    check("_parse_read_paths splits + dedupes, drops blanks",
          _parse_read_paths(f"{scope}, {scope} ,, {outside}") == [scope, outside])
    check("_parse_read_paths empty → []", _parse_read_paths("") == [])

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ts._ensure_loaded()
    asyncio.run(_run())
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
