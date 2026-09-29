"""Verification for the external transport's per-token `allowed_ips` lock.

Standalone runnable script (no pytest in this venv):

    .lvenv/bin/python tests/test_external_ip_lock.py

Starts a real ExternalTransport-style aiohttp app on loopback and sends real
HTTP requests through `_authenticate`, so the X-Forwarded-For handling is
exercised the way a reverse proxy on the same machine would hit it.

What this guards:
  (a) a token WITHOUT `allowed_ips` is unaffected (any address);
  (b) a locked token passes from an allowed address arriving via a trusted
      proxy (loopback peer + X-Forwarded-For), including an IPv4-mapped IPv6
      form and a CIDR entry;
  (c) a locked token is refused (403) from any other address;
  (d) a forged X-Forwarded-For is ignored when the peer is NOT a trusted
      proxy (a caller connecting straight to the port can't claim an address);
  (e) only the rightmost hops (the ones our proxy wrote) count: a caller
      can't prepend an allowed address in front of a trusted proxy;
  (f) fail closed: `allowed_ips: []`, a non-list value, and a request with
      no forwarded address at all are all refused;
  (g) the refusal happens before the rate limit, so an outsider can't burn
      the owner's per-minute budget.
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from aiohttp import ClientSession, web  # noqa: E402

from openflip.transports.external import ExternalTransport  # noqa: E402

HOME = "203.0.113.7"
STRANGER = "198.51.100.9"
TOKENS = {
    "open-" + "a" * 40: {"agent": "tester", "session": "open", "default_model": "m"},
    "lock-" + "b" * 40: {"agent": "tester", "session": "lock", "default_model": "m",
                         "allowed_ips": [HOME, "192.0.2.0/24"], "rate_limit": 2},
    "empty-" + "c" * 40: {"agent": "tester", "session": "empty", "default_model": "m",
                          "allowed_ips": []},
    "typo-" + "d" * 40: {"agent": "tester", "session": "typo", "default_model": "m",
                         "allowed_ips": HOME},
}
FAILS: list[str] = []


def check(name: str, got: int, want: int) -> None:
    status = "ok  " if got == want else "FAIL"
    if got != want:
        FAILS.append(name)
    print(f"  {status} {name}: {got} (want {want})")


class _Agent:
    id = "tester"


class _Runner:
    agent = _Agent()


def make_transport(tmp: str, trusted_proxies=None) -> ExternalTransport:
    path = os.path.join(tmp, "tokens.json")
    with open(path, "w") as f:
        json.dump({"tokens": TOKENS}, f)
    t = ExternalTransport(token_path=path, cert_dir=tmp, trusted_proxies=trusted_proxies)
    t.attach_runner(_Runner())
    return t


async def serve(t: ExternalTransport, port: int) -> web.AppRunner:
    async def handler(request: web.Request) -> web.Response:
        entry, err = t._authenticate(request)
        return err if err is not None else web.json_response({"ok": entry["session"]})
    app = web.Application()
    app.router.add_post("/x", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", port).start()
    return runner


async def hit(port: int, token: str, xff: str | None = None) -> int:
    headers = {"Authorization": f"Bearer {token}"}
    if xff is not None:
        headers["X-Forwarded-For"] = xff
    async with ClientSession() as s:
        async with s.post(f"http://127.0.0.1:{port}/x", headers=headers, json={}) as r:
            return r.status


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tok = {v["session"]: k for k, v in TOKENS.items()}

        # Default trust: loopback is the proxy (Caddy on the same machine).
        t = make_transport(tmp)
        runner = await serve(t, 18771)
        print("(a) no allowed_ips = unaffected")
        check("open token, stranger via proxy", await hit(18771, tok["open"], STRANGER), 200)
        check("open token, no header", await hit(18771, tok["open"]), 200)
        print("(b) allowed address via trusted proxy")
        check("home ip", await hit(18771, tok["lock"], HOME), 200)
        check("home ip, ipv4-mapped form", await hit(18771, tok["lock"], f"::ffff:{HOME}"), 200)
        t._rate_hits.clear()
        check("inside CIDR 192.0.2.0/24", await hit(18771, tok["lock"], "192.0.2.44"), 200)
        print("(c) other address refused")
        check("stranger via proxy", await hit(18771, tok["lock"], STRANGER), 403)
        print("(e) only the proxy-written hops count")
        check("stranger prepends home ip", await hit(18771, tok["lock"], f"{HOME}, {STRANGER}"), 403)
        print("(f) fail closed")
        check("allowed_ips: []", await hit(18771, tok["empty"], HOME), 403)
        check("allowed_ips not a list", await hit(18771, tok["typo"], HOME), 403)
        check("locked, proxy sent no header (peer is loopback)", await hit(18771, tok["lock"]), 403)
        check("garbled header", await hit(18771, tok["lock"], "not-an-ip"), 403)
        print("(g) refused before the rate limit")
        t._rate_hits.clear()
        for _ in range(5):
            await hit(18771, tok["lock"], STRANGER)
        check("home still gets through after stranger spam (limit 2/min)", await hit(18771, tok["lock"], HOME), 200)
        await runner.cleanup()

        # (d) Peer is NOT a trusted proxy: the header must be ignored.
        t2 = make_transport(tmp, trusted_proxies=["10.99.99.99/32"])
        runner = await serve(t2, 18772)
        print("(d) forged header from an untrusted peer")
        check("direct caller claims home ip", await hit(18772, tok["lock"], HOME), 403)
        await runner.cleanup()

        # Malformed trusted_proxies → trust nobody → locked token denies.
        t3 = make_transport(tmp, trusted_proxies="127.0.0.1")
        runner = await serve(t3, 18773)
        print("(f) malformed trusted_proxies trusts no proxy")
        check("home ip via 'proxy'", await hit(18773, tok["lock"], HOME), 403)
        await runner.cleanup()

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {', '.join(FAILS)}")
        sys.exit(1)
    print("ALL OK")


if __name__ == "__main__":
    asyncio.run(main())
