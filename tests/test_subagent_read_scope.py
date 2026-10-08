"""spawn_subagents worker read/write scope (Session.read_scope /
Session.write_scope → files.py ACL).

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
  (3) with NO write_scope it CANNOT write inside dir (read scope alone
      never grants writes);
  (4) denied_paths still blocks a path inside the read scope;
  (5) read_scope=[] falls back to the agent-dir + temp default (not the
      agent's discord-only allowed_read_paths, which the worker can't match);
  (6) list_files uses the same read gate;
  (7) a NON-worker session (read_scope=None) is unchanged — it resolves the
      agent's own allowed_read_paths;
  (8) the tool_settings read_paths validator + subagent _parse_read_paths;
  (9) write_scope: write_file/edit_file allowed inside it, denied outside,
      denied when write_scope is None or [], denied_paths still wins,
      delete_file/restore_snapshot stay in SUBAGENT_DENIED_TOOLS, and the
      write_paths validator shares read_paths' rules.
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
from openflip.tools.files import read_file, write_file, edit_file, list_files
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

    print("(3) worker with no write_scope cannot write inside its read_scope")
    check("worker write_scope defaults to None", worker.write_scope is None)
    res = await _as(agent, worker, lambda: write_file(os.path.join(scope, "new.txt"), "x"))
    check("write inside read scope denied without write_scope", res.error is not None and "denied" in res.error.lower())
    check("no file was created", not os.path.exists(os.path.join(scope, "new.txt")))
    res = await _as(agent, worker, lambda: edit_file(inside_f, "IN", "EDITED"))
    check("edit inside read scope denied without write_scope", res.error is not None and "denied" in res.error.lower())
    with open(inside_f, encoding="utf-8") as fh:
        check("file untouched by denied edit", fh.read() == "IN")

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

    print("(9) write_scope: writes confined to it; delete/restore stay hard-denied")
    wscope = os.path.join(tmp, "wscope")
    wsecret = os.path.join(wscope, "secret")
    os.makedirs(wsecret, exist_ok=True)
    seed_f = os.path.join(wscope, "seed.txt")
    wsecret_f = os.path.join(wsecret, "s.txt")
    for f in (seed_f, wsecret_f):
        with open(f, "w", encoding="utf-8") as fh:
            fh.write("SEED")
    writer = make_subagent_session("orch", "task-w1", ["read_file", "write_file", "edit_file"],
                                   read_scope=[scope, wscope], write_scope=[wscope])
    check("write_scope stored as independent copy", writer.write_scope == [wscope])
    src_list = [wscope]
    w2 = make_subagent_session("orch", "task-w2", [], read_scope=None, write_scope=src_list)
    src_list.append(outside)
    check("mutating the caller's list does not reach the session", w2.write_scope == [wscope])

    new_f = os.path.join(wscope, "new.txt")
    res = await _as(agent, writer, lambda: write_file(new_f, "hello"))
    check("write_file inside write_scope ok", res.error is None and os.path.exists(new_f))
    res = await _as(agent, writer, lambda: edit_file(seed_f, "SEED", "GROWN"))
    with open(seed_f, encoding="utf-8") as fh:
        body = fh.read()
    check("edit_file inside write_scope ok", res.error is None and body == "GROWN")

    out_new = os.path.join(outside, "w.txt")
    res = await _as(agent, writer, lambda: write_file(out_new, "x"))
    check("write_file outside write_scope denied", res.error is not None and "denied" in res.error.lower())
    check("no file created outside write_scope", not os.path.exists(out_new))
    res = await _as(agent, writer, lambda: edit_file(outside_f, "OUT", "X"))
    check("edit_file outside write_scope denied", res.error is not None and "denied" in res.error.lower())
    # read_scope dir that is NOT in write_scope: readable, not writable
    res = await _as(agent, writer, lambda: write_file(os.path.join(scope, "ro.txt"), "x"))
    check("read-scope-only dir is not writable", res.error is not None and "denied" in res.error.lower())
    res = await _as(agent, writer, lambda: read_file(inside_f))
    check("read-scope-only dir still readable for the writer", res.error is None)
    # write_scope does NOT widen reads: a dir only in write_scope is writable, and
    # readable only because we also listed it in read_scope above.
    reads_none = make_subagent_session("orch", "task-w3", ["write_file"], read_scope=[], write_scope=[wscope])
    res = await _as(agent, reads_none, lambda: read_file(seed_f))
    check("write_scope alone does not grant reads", res.error is not None and "denied" in res.error.lower())
    res = await _as(agent, reads_none, lambda: write_file(os.path.join(wscope, "w3.txt"), "x"))
    check("write_scope with empty read_scope still writes inside scope", res.error is None)

    empty_w = make_subagent_session("orch", "task-w4", ["write_file"], read_scope=[wscope], write_scope=[])
    res = await _as(agent, empty_w, lambda: write_file(os.path.join(wscope, "e.txt"), "x"))
    check("write_scope=[] denies writes", res.error is not None and "denied" in res.error.lower())
    none_w = make_subagent_session("orch", "task-w5", ["write_file"], read_scope=[wscope], write_scope=None)
    res = await _as(agent, none_w, lambda: write_file(os.path.join(wscope, "n.txt"), "x"))
    check("write_scope=None denies writes", res.error is not None and "denied" in res.error.lower())

    agent_wdenied = _agent(agent_dir, denied=[wsecret])
    res = await _as(agent_wdenied, writer, lambda: write_file(os.path.join(wsecret, "x.txt"), "x"))
    check("denied_paths wins over write_scope (write_file)", res.error is not None and "denied" in res.error.lower())
    res = await _as(agent_wdenied, writer, lambda: edit_file(wsecret_f, "SEED", "X"))
    check("denied_paths wins over write_scope (edit_file)", res.error is not None and "denied" in res.error.lower())
    res = await _as(agent_wdenied, writer, lambda: write_file(os.path.join(wscope, "ok.txt"), "x"))
    check("non-denied path in write_scope still writable", res.error is None)

    # non-worker session: both scopes None → agent's own write ACL (discord block)
    check("human session has write_scope None", human.write_scope is None)
    res = await _as(agent, human, lambda: write_file(os.path.join(agent_dir, "h.txt"), "x"))
    check("human writes inside the agent's configured write path", res.error is None)
    res = await _as(agent, human, lambda: write_file(os.path.join(wscope, "h.txt"), "x"))
    check("human cannot write outside the agent's write path", res.error is not None)

    check("write_file no longer hard-denied", "write_file" not in ts.SUBAGENT_DENIED_TOOLS)
    check("edit_file no longer hard-denied", "edit_file" not in ts.SUBAGENT_DENIED_TOOLS)
    check("delete_file still in SUBAGENT_DENIED_TOOLS", "delete_file" in ts.SUBAGENT_DENIED_TOOLS)
    check("restore_snapshot still in SUBAGENT_DENIED_TOOLS", "restore_snapshot" in ts.SUBAGENT_DENIED_TOOLS)
    check("run_command NOT hard-denied (gated by allow_shell + write_paths at call time)",
          "run_command" not in ts.SUBAGENT_DENIED_TOOLS)
    check("run_command_sandbox still in SUBAGENT_DENIED_TOOLS", "run_command_sandbox" in ts.SUBAGENT_DENIED_TOOLS)
    check("write_file_sandbox still in SUBAGENT_DENIED_TOOLS", "write_file_sandbox" in ts.SUBAGENT_DENIED_TOOLS)
    check("write_file not in the default allowed_tools ceiling",
          "write_file" not in [t.strip() for t in ts.SUBAGENT_DEFAULT_TOOLS.split(",")])
    check("edit_file not in the default allowed_tools ceiling",
          "edit_file" not in [t.strip() for t in ts.SUBAGENT_DEFAULT_TOOLS.split(",")])
    check("/toolset allowed_tools validator now accepts write_file,edit_file",
          ts.coerce_and_validate("spawn_subagents", "allowed_tools", "read_file,write_file,edit_file")[1] is None)
    check("/toolset allowed_tools validator still rejects delete_file",
          ts.coerce_and_validate("spawn_subagents", "allowed_tools", "read_file,delete_file")[1] is not None)

    check("write_paths validator accepts empty", ts.coerce_and_validate("spawn_subagents", "write_paths", "")[1] is None)
    check("write_paths validator accepts an existing absolute dir",
          ts.coerce_and_validate("spawn_subagents", "write_paths", wscope)[1] is None)
    check("write_paths validator accepts a comma list",
          ts.coerce_and_validate("spawn_subagents", "write_paths", f"{wscope}, {scope}")[1] is None)
    check("write_paths validator rejects a relative path",
          ts.coerce_and_validate("spawn_subagents", "write_paths", "relative/dir")[1] is not None)
    check("write_paths validator rejects a non-existent dir",
          ts.coerce_and_validate("spawn_subagents", "write_paths", os.path.join(tmp, "nope"))[1] is not None)
    check("write_paths validator rejects a file (not a dir)",
          ts.coerce_and_validate("spawn_subagents", "write_paths", seed_f)[1] is not None)
    check("write_paths default is empty", ts.get_schema("spawn_subagents").settings["write_paths"].default == "")

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
