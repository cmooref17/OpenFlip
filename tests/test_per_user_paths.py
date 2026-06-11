"""Verification for per-user path ACLs (per agents/<id>/specs/per_user_paths_spec.md).

Standalone runnable script (no pytest in this venv):

    .lvenv/bin/python tests/test_per_user_paths.py

The dict form is TRANSPORT-KEYED, structurally identical to a tool's `auth`
block: {"discord": {"users": {...}, "all_users": [...]}, "imessage": {...}}.
The speaker's transport block is selected first (like acl.py's
acl.auth.get(transport)), then `users` / `all_users` resolve inside it. The
owner is just an id under `users`, there is NO magic owner key and NO default
key.

Proves:
  (a) flat-list agents behave BYTE-IDENTICALLY to the pre-change _check_access
      (a reference copy of the old logic is embedded and compared exactly);
  (b) dict form resolves <transport>.users.<id> then falls back to
      <transport>.all_users; missing all_users denies; the owner is matched
      only by its id under `users`;
  (c) denied_paths overrides for ALL users (owner included, even with "*");
  (d) "*" in a per-user list allows-all for THAT user only;
  (e) the /foo vs /foobar separator boundary is unchanged.

Also covers: a speaker not listed under `users` (owner included) falls to
all_users — not to a special owner branch; a Discord id resolves under the
`discord` block and an iMessage handle (normalized) under the `imessage` block;
and a transport with no block at all denies (default-deny, never fail-open).
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openflip.tools.files import _check_access, _is_within
from openflip.utils import safe_path_display
from openflip.config_global import get_owner_id

OWNER_ID = get_owner_id("discord")        # 100000000000000000
NON_OWNER = OWNER_ID + 1                   # any other discord id

FAILURES: list[str] = []


def _set_speaker(speaker_id, transport="discord", handle=""):
    from openflip.tool_executor import CURRENT_SPEAKER_ID, CURRENT_SESSION
    CURRENT_SPEAKER_ID.set(int(speaker_id) if transport == "discord" else 1)
    if transport == "discord":
        CURRENT_SESSION.set(None)
    else:
        CURRENT_SESSION.set(SimpleNamespace(transport=transport, handle=handle))


def _set_agent(agent):
    from openflip.tool_executor import CURRENT_AGENT
    CURRENT_AGENT.set(agent)


def mk_agent(read, write, denied=()):
    # Stand-in: _check_access only reads .path, .allowed_*_paths, .denied_paths.
    return SimpleNamespace(
        path="/srv/agents/testbot/agent.json",
        allowed_read_paths=read,
        allowed_write_paths=write,
        denied_paths=list(denied),
    )


def allowed(agent, path, mode):
    _set_agent(agent)
    return _check_access(os.path.realpath(path), mode) is None


def check(label, cond):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if not cond:
        FAILURES.append(label)


# ---------------------------------------------------------------------------
# Reference: a verbatim copy of _check_access BEFORE the change, reading the
# attributes directly. Used to prove byte-identical back-compat for flat lists.
# ---------------------------------------------------------------------------
def _check_access_OLD(agent, full_path, mode):
    denied = list(getattr(agent, "denied_paths", []) or [])
    for d in denied:
        if _is_within(full_path, d):
            return f"Access denied: {safe_path_display(full_path)}"
    if mode == "read":
        allowed_ = getattr(agent, "allowed_read_paths", []) or []
    else:
        allowed_ = getattr(agent, "allowed_write_paths", []) or []
    if "*" in allowed_:
        return None
    if not allowed_:
        if mode == "read":
            agent_dir = os.path.dirname(agent.path)
            # 2026-06-11 Windows-compat: the live implementation switched the
            # universal read fallback from literal "/tmp" to
            # tempfile.gettempdir() (same dir on POSIX) and reworded the
            # denial message. Mirrored here so the string-equality comparison
            # keeps proving per-user-resolution parity, which is this test's
            # actual subject.
            import tempfile as _tempfile
            for fallback in (agent_dir, _tempfile.gettempdir()):
                if _is_within(full_path, fallback):
                    return None
            return (
                f"Access denied: {safe_path_display(full_path)} (no allowed_read_paths "
                f"configured; default read scope is agent dir + the system temp dir)"
            )
        return (
            f"Access denied: {safe_path_display(full_path)} (no allowed_write_paths configured; "
            f"writes require an explicit allow list)"
        )
    for a in allowed_:
        if _is_within(full_path, a):
            return None
    return f"Access denied: {safe_path_display(full_path)}"


def test_a_flat_byte_identical():
    print("\n(a) flat-list back-compat — new _check_access vs embedded OLD reference (exact string equality)")
    agent_dir = "/srv/agents/testbot"
    flat_agents = [
        mk_agent(["/work/readonly"], ["/work/writable"], ["/work/writable/secret"]),
        mk_agent([], [], []),                       # empty → read fallback / write deny
        mk_agent(["*"], ["*"], ["/work/writable/secret"]),
        mk_agent(["/foo"], ["/foo"], []),           # boundary agent
        mk_agent(None, None, []),                   # None coalesces to []
    ]
    probe_paths = [
        "/work/readonly", "/work/readonly/f", "/work/readonlyEXTRA",
        "/work/writable", "/work/writable/f", "/work/writable/secret",
        "/work/writable/secret/deep", "/foo", "/foo/x", "/foobar",
        "/tmp/x", f"{agent_dir}/inside", "/somewhere/else",
    ]
    mismatches = 0
    total = 0
    # speaker identity is irrelevant for flat lists; set an arbitrary non-owner.
    _set_speaker(NON_OWNER)
    for ag in flat_agents:
        for p in probe_paths:
            rp = os.path.realpath(p)
            for mode in ("read", "write"):
                total += 1
                _set_agent(ag)
                new = _check_access(rp, mode)
                old = _check_access_OLD(ag, rp, mode)
                if new != old:
                    mismatches += 1
                    print(f"    MISMATCH mode={mode} path={p}\n      new={new!r}\n      old={old!r}")
    check(f"{total} (agent,path,mode) combinations are string-identical to OLD ({mismatches} mismatches)", mismatches == 0)


def test_b_dict_resolution():
    print("\n(b) dict form — discord.users.<id> then discord.all_users; owner is just an id; missing all_users denies")
    # Owner is configured exactly like any other user: an id under `users`,
    # inside the speaker's transport block (`discord` here).
    read = {"discord": {"users": {str(OWNER_ID): ["/owner/scope"],
                                  str(NON_OWNER): ["/user/scope"]},
                        "all_users": ["/default/scope"]}}
    write_no_all = {"discord": {"users": {str(NON_OWNER): ["/sandbox"]}}}  # no "all_users"
    ag = mk_agent(read, write_no_all)

    _set_speaker(OWNER_ID)
    check("owner (listed under users) reads own scope", allowed(ag, "/owner/scope/f", "read"))
    check("owner does NOT get all_users scope", not allowed(ag, "/default/scope/f", "read"))
    check("owner does NOT get another user's scope", not allowed(ag, "/user/scope/f", "read"))

    _set_speaker(NON_OWNER)
    check("configured user reads its user-scope", allowed(ag, "/user/scope/f", "read"))
    check("configured user does NOT get owner scope", not allowed(ag, "/owner/scope/f", "read"))
    check("configured user does NOT get all_users scope", not allowed(ag, "/default/scope/f", "read"))

    _set_speaker(OWNER_ID + 777)  # unconfigured discord id
    check("unconfigured user falls to all_users scope", allowed(ag, "/default/scope/f", "read"))
    check("unconfigured user denied outside all_users", not allowed(ag, "/user/scope/f", "read"))

    # missing "all_users" for write → empty fallback → write default-deny for
    # everyone not explicitly listed under users.
    _set_speaker(OWNER_ID + 777)
    check("unconfigured user write denied (no all_users in write dict)", not allowed(ag, "/sandbox/f", "write"))
    _set_speaker(NON_OWNER)
    check("configured user write hits its sandbox", allowed(ag, "/sandbox/f", "write"))

    # The owner has NO special branch: if the owner's id is not under `users`,
    # the owner falls to all_users like everyone else (mirrors tool ACLs, where
    # owner-bypass was removed).
    owner_unlisted = {"discord": {"users": {str(NON_OWNER): ["/u"]}, "all_users": ["/d"]}}
    ag2 = mk_agent(owner_unlisted, ["*"])
    _set_speaker(OWNER_ID)
    check("owner not under users falls to all_users", allowed(ag2, "/d/f", "read"))
    check("owner not under users denied outside all_users", not allowed(ag2, "/u/f", "read"))

    # No all_users and owner not under users → empty list → still gets the
    # universal /tmp read fallback, nothing else.
    owner_unlisted_no_all = {"discord": {"users": {str(NON_OWNER): ["/u"]}}}
    ag3 = mk_agent(owner_unlisted_no_all, ["*"])
    _set_speaker(OWNER_ID)
    check("owner with empty resolution does NOT inherit another user's scope", not allowed(ag3, "/u/f", "read"))
    check("owner with empty resolution still gets the universal /tmp read fallback", allowed(ag3, "/tmp/f", "read"))


def test_c_denied_overrides_all():
    print("\n(c) denied_paths overrides for ALL users (owner included, even with '*')")
    read = {"discord": {"users": {str(OWNER_ID): ["*"], str(NON_OWNER): ["/shared"]},
                        "all_users": ["/shared"]}}
    ag = mk_agent(read, ["*"], denied=["/shared/locked"])
    for who, sid in (("owner", OWNER_ID), ("configured user", NON_OWNER), ("default user", OWNER_ID + 5)):
        _set_speaker(sid)
        check(f"{who} blocked from denied /shared/locked", not allowed(ag, "/shared/locked/f", "read"))
    # owner '*' still works elsewhere
    _set_speaker(OWNER_ID)
    check("owner '*' still reads a non-denied path", allowed(ag, "/anywhere/f", "read"))


def test_d_wildcard_per_user():
    print("\n(d) '*' in a per-user list allows-all for THAT user only")
    read = {"discord": {"users": {str(OWNER_ID): ["*"], str(NON_OWNER): ["/sandbox"]},
                        "all_users": ["/sandbox"]}}
    ag = mk_agent(read, ["*"])
    _set_speaker(OWNER_ID)
    check("owner ('*') reads an arbitrary path", allowed(ag, "/etc/anything", "read"))
    _set_speaker(NON_OWNER)
    check("configured user does NOT inherit owner's '*'", not allowed(ag, "/etc/anything", "read"))
    check("configured user reads only its scope", allowed(ag, "/sandbox/f", "read"))
    _set_speaker(OWNER_ID + 9)
    check("default user does NOT inherit owner's '*'", not allowed(ag, "/etc/anything", "read"))


def test_e_boundary():
    print("\n(e) /foo vs /foobar separator boundary unchanged")
    # flat
    ag = mk_agent(["/foo"], ["/foo"])
    _set_speaker(NON_OWNER)
    check("flat: /foo allows /foo", allowed(ag, "/foo", "read"))
    check("flat: /foo allows /foo/child", allowed(ag, "/foo/child", "read"))
    check("flat: /foo does NOT allow /foobar", not allowed(ag, "/foobar", "read"))
    # dict (boundary must hold through the per-user seam too)
    agd = mk_agent({"discord": {"users": {str(NON_OWNER): ["/foo"]}, "all_users": []}},
                   {"discord": {"users": {str(NON_OWNER): ["/foo"]}}})
    check("dict: user /foo allows /foo/child", allowed(agd, "/foo/child", "read"))
    check("dict: user /foo does NOT allow /foobar", not allowed(agd, "/foobar", "read"))


def test_imessage_handle_norm():
    print("\n(extra) iMessage handle resolves under the `imessage` block; key normalization (.strip().lower())")
    read = {"imessage": {"users": {"alice@example.com": ["/imsg/scope"]}, "all_users": []}}
    ag = mk_agent(read, [])
    _set_speaker(0, transport="imessage", handle="  Alice@Example.COM ")
    check("mixed-case/padded handle matches lowercased users key in imessage block", allowed(ag, "/imsg/scope/f", "read"))
    check("non-matching handle falls to (empty) all_users → deny", not allowed(ag, "/imsg/other", "read"))


def test_transport_keyed_routing():
    print("\n(extra) transport-keyed routing — discord id under `discord`, imessage handle under `imessage`, no block denies")
    # Both transports configured on the SAME field, distinct scopes per block.
    read = {
        "discord":  {"users": {str(NON_OWNER): ["/disc/scope"]}, "all_users": ["/disc/base"]},
        "imessage": {"users": {"bob@example.com": ["/imsg/scope"]}, "all_users": ["/imsg/base"]},
    }
    ag = mk_agent(read, [])

    # Discord speaker resolves ONLY against the discord block.
    _set_speaker(NON_OWNER)
    check("discord user reads its discord-block scope", allowed(ag, "/disc/scope/f", "read"))
    check("discord user does NOT reach the imessage block", not allowed(ag, "/imsg/scope/f", "read"))
    _set_speaker(OWNER_ID + 4242)  # unconfigured discord id
    check("unconfigured discord user falls to discord all_users", allowed(ag, "/disc/base/f", "read"))
    check("unconfigured discord user does NOT reach imessage all_users", not allowed(ag, "/imsg/base/f", "read"))

    # iMessage speaker resolves ONLY against the imessage block.
    _set_speaker(0, transport="imessage", handle="bob@example.com")
    check("imessage handle reads its imessage-block scope", allowed(ag, "/imsg/scope/f", "read"))
    check("imessage handle does NOT reach the discord block", not allowed(ag, "/disc/scope/f", "read"))
    _set_speaker(0, transport="imessage", handle="nobody@example.com")
    check("unconfigured imessage handle falls to imessage all_users", allowed(ag, "/imsg/base/f", "read"))

    # A transport with NO block at all → empty resolution → default-deny
    # (read still gets the universal /tmp + agent-dir fallback, nothing else).
    discord_only = {"discord": {"users": {str(NON_OWNER): ["/disc/scope"]}, "all_users": ["/disc/base"]}}
    ag2 = mk_agent(discord_only, discord_only)
    _set_speaker(0, transport="imessage", handle="bob@example.com")
    check("transport with no block: denied outside fallback (read)", not allowed(ag2, "/disc/scope/f", "read"))
    check("transport with no block: denied at the all_users path too", not allowed(ag2, "/disc/base/f", "read"))
    check("transport with no block: write denied (no block, no fallback)", not allowed(ag2, "/disc/scope/f", "write"))
    check("transport with no block: read still gets universal /tmp fallback", allowed(ag2, "/tmp/f", "read"))


if __name__ == "__main__":
    print(f"owner discord id = {OWNER_ID}")
    test_a_flat_byte_identical()
    test_b_dict_resolution()
    test_c_denied_overrides_all()
    test_d_wildcard_per_user()
    test_e_boundary()
    test_imessage_handle_norm()
    test_transport_keyed_routing()
    print()
    if FAILURES:
        print(f"RESULT: {len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("RESULT: ALL PASS")
