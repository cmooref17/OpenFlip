"""Project-wide shared state.

Lives in its own module specifically to dodge the duplicate-module trap that
`python -m openflip.main` creates: the entry point loads `main.py` as
`__main__`, but anything that does `from . import main` gets a SECOND copy
loaded under the name `openflip.main`. Two module objects → two separate
copies of any module-level dict defined in main.py.

`openflip.registry` is never run as __main__ and never imported under two names,
so the dicts here are the single shared source of truth. main.py and
commands.py both reference `registry.ALL_AGENTS` / `registry.RUNNERS` /
`registry.TOKENS` and see the same objects.

Mutate via in-place ops (`.clear()` + `.update()`, or just key assignment) —
NOT rebinding (`registry.ALL_AGENTS = ...`), since rebinding would still work
here but in-place is harder to break.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .agent import Agent
    from .runtime import AgentRunner

RUNNERS: dict[str, "AgentRunner"] = {}
ALL_AGENTS: dict[str, "Agent"] = {}
TOKENS: dict[str, str] = {}
