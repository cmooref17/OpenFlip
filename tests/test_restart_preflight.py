"""restart_gateway preflight: busy check reads live runner state, not live.json."""
from __future__ import annotations
import asyncio, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openflip import registry
from openflip import agent_state as _as
from openflip.tools.restart import _check_other_agents_busy

FAILS = []
def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond: FAILS.append(name)

class FakeRunner:
    def __init__(self): self._active_turns = {}

async def main():
    saved = dict(registry.RUNNERS)
    registry.RUNNERS.clear()
    try:
        a, b = FakeRunner(), FakeRunner()
        registry.RUNNERS.update({"caller": a, "peer": b})
        check("idle peers -> not busy", _check_other_agents_busy(exclude_agent_id="caller") == [])

        # Long-running peer turn whose cached activity is >60s old (the 2026-10-07 bug).
        _as._CACHE["peer"] = {"activity": "Active in DM", "last_active_ms": int(time.time()*1000) - 30*60*1000}
        t = asyncio.ensure_future(asyncio.sleep(10))
        b._active_turns[123] = t
        busy = _check_other_agents_busy(exclude_agent_id="caller")
        check("30-min-old peer turn still counts as busy", len(busy) == 1 and busy[0]["agent_id"] == "peer")
        check("activity text carried", busy and busy[0]["activity"] == "Active in DM")

        a._active_turns[1] = asyncio.ensure_future(asyncio.sleep(10))
        check("caller's own turns excluded", [x["agent_id"] for x in _check_other_agents_busy(exclude_agent_id="caller")] == ["peer"])

        t.cancel()
        try: await t
        except asyncio.CancelledError: pass
        check("finished task in slot -> not busy", _check_other_agents_busy(exclude_agent_id="caller") == [])
        a._active_turns[1].cancel()
    finally:
        registry.RUNNERS.clear(); registry.RUNNERS.update(saved)
        _as._CACHE.pop("peer", None)

asyncio.run(main())
print("ALL PASS" if not FAILS else f"FAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
