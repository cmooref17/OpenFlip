"""Live check (real API call, not part of tests/run.sh): the /goal evaluator
returns valid verdicts from the real small model. Run:
    .lvenv/bin/python tests/live/live_goal_evaluator_check.py
"""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from openflip import goal
from openflip.anthropic_conversation import ChatMessage

COND = "tests/run.sh prints TOTAL with 0 failed"
met = [ChatMessage("user", "run the tests"),
       ChatMessage("assistant", "Running the full suite."),
       ChatMessage("tool", "Exit 0:\nPASS tests/test_goal.py\nTOTAL: 37 files, 37 passed, 0 failed"),
       ChatMessage("assistant", "37/37 pass.")]
notmet = [ChatMessage("user", "run the tests"),
          ChatMessage("assistant", "Running the full suite."),
          ChatMessage("tool", "Exit 1:\nTOTAL: 37 files, 35 passed, 2 failed ( tests/test_a.py tests/test_b.py )"),
          ChatMessage("assistant", "Two failures.")]

async def main():
    print("model:", goal.evaluator_model())
    a = await goal.evaluate(COND, met); print("met case ->", a.verdict, "|", a.reason)
    b = await goal.evaluate(COND, notmet); print("not_met case ->", b.verdict, "|", b.reason)
    ok = a.verdict == "met" and b.verdict == "not_met"
    print("LIVE OK" if ok else "LIVE FAIL"); sys.exit(0 if ok else 1)
asyncio.run(main())
