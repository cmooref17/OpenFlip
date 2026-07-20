"""Tool registry. Importing this package auto-registers every @tool in submodules."""
from ._base import TOOL_REGISTRY, Tool, ToolResult, tool, tool_callable_for_ollama, get_tool

# === Framework-core tools (ship in public repo) ===
from . import web_search  # noqa: F401
from . import memory  # noqa: F401
from . import dream_tool  # noqa: F401
from . import files  # noqa: F401
from . import fetch_url  # noqa: F401
from . import run_command  # noqa: F401
from . import restart  # noqa: F401
from . import send_message  # noqa: F401
from . import talk_to_agent  # noqa: F401
from . import inject_context  # noqa: F401
from . import delete_message  # noqa: F401
from . import fetch_discord_message  # noqa: F401
from . import snapshots  # noqa: F401
from . import claude_code  # noqa: F401
from . import end_chain  # noqa: F401
from . import cron_jobs  # noqa: F401

# === Optional extras ===
# Files at openflip/tools/<name>.py OR package dirs openflip/tools/<name>/ that
# aren't in the framework-core list above can be local (gitignored, see
# .gitignore) personal tool namespaces living alongside the repo — e.g. the
# `myorg/` wrapper package. They use the same relative-import
# convention as the rest of this package. Loaded here with silent skip if
# absent or if a dependency isn't installed.
import os as _os
_tools_dir = _os.path.dirname(__file__)
_core_modules = {
    "_base", "web_search", "memory", "dream_tool", "files", "fetch_url", "run_command",
    "restart", "send_message", "talk_to_agent", "inject_context", "delete_message",
    "fetch_discord_message", "snapshots", "claude_code", "end_chain",
    "cron_jobs",
}
for _fname in sorted(_os.listdir(_tools_dir)):
    if _fname.startswith("__"):
        continue
    _full = _os.path.join(_tools_dir, _fname)
    if _fname.endswith(".py"):
        _stem = _fname[:-3]
    elif _os.path.isdir(_full) and _os.path.exists(_os.path.join(_full, "__init__.py")):
        _stem = _fname  # local extra namespace package (e.g. myorg/)
    else:
        continue
    if _stem in _core_modules:
        continue
    try:
        __import__(f"openflip.tools.{_stem}", fromlist=[_stem])
    except ImportError as _e:
        # Silent skip is correct ONLY when the extras module itself is absent
        # (a gitignored personal namespace that simply isn't present here).
        # A module that IS present but fails to import — missing dependency
        # (torch, demucs, …) or a broken import inside it — must be surfaced:
        # otherwise its tools silently vanish from TOOL_REGISTRY and agents
        # just stop having them with zero log output.
        if isinstance(_e, ModuleNotFoundError) and _e.name == f"openflip.tools.{_stem}":
            continue
        from ..utils import print_ts as _print_ts, COLOR_RED as _RED, COLOR_END as _CEND
        _print_ts(
            f"{_RED}tools: optional module '{_stem}' is present but failed to "
            f"import — its tools are UNAVAILABLE this run: {_e}{_CEND}",
            error=True,
        )
