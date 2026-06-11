import os
import re
import time
import json
import asyncio
from typing import Any

COLOR_END = '\033[0m'
COLOR_RED = '\033[91m'
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_BLUE = '\033[94m'

# Windows consoles need ENABLE_VIRTUAL_TERMINAL_PROCESSING switched on before
# ANSI escapes render — without it, classic cmd.exe shows raw "[92m" noise
# around every log line. Windows Terminal enables it by default; this makes
# cmd/powershell hosts match. No-op anywhere else, never fatal.
if os.name == "nt":
    try:
        import ctypes
        _k32 = ctypes.windll.kernel32
        for _std in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            _h = _k32.GetStdHandle(_std)
            _mode = ctypes.c_uint32()
            if _k32.GetConsoleMode(_h, ctypes.byref(_mode)):
                _k32.SetConsoleMode(_h, _mode.value | 0x0004)
    except Exception:
        pass


def remove_color_tags(raw_text: str) -> str:
    return re.sub(r'\033\[\d+(;\d+)*m', '', raw_text or '')


def red(text):    return f"{COLOR_RED}{remove_color_tags(text)}{COLOR_END}"
def yellow(text): return f"{COLOR_YELLOW}{remove_color_tags(text)}{COLOR_END}"
def green(text):  return f"{COLOR_GREEN}{remove_color_tags(text)}{COLOR_END}"
def blue(text):   return f"{COLOR_BLUE}{remove_color_tags(text)}{COLOR_END}"


def timestamp() -> str:
    ts = time.localtime()
    return f"[{ts.tm_hour:02}:{ts.tm_min:02}:{ts.tm_sec:02}]"


_log_path: str | None = None


def set_log_path(path: str | None):
    global _log_path
    _log_path = path


def _write_log(text: str):
    if not _log_path:
        return
    try:
        with open(_log_path, 'a') as f:
            f.write(remove_color_tags(text) + '\n')
    except Exception:
        pass


_prev_ts = ""


def print_ts(text: str = "", *, agent: str | None = None, error: bool = False, end: str = "\n"):
    global _prev_ts
    ts = timestamp()
    ts_str = ts if ts != _prev_ts else " " * len(ts)
    _prev_ts = ts
    prefix = f"{ts_str} "
    if agent:
        prefix += f"[{agent}] "
    line = prefix + text
    _write_log(line)
    if error:
        line = red(line)
    print(line, end=end, flush=True)


def log_task_exception(task):
    """Done-callback for fire-and-forget tasks: surfaces a swallowed
    exception into log.txt instead of dying silently at GC.

    Bare `asyncio.create_task(coro)` on a long-lived coroutine loses any
    exception the coroutine raises — it only resurfaces as an unretrieved
    "Task exception was never retrieved" warning at GC time, which never
    reaches `log.txt`. Attach this as a done-callback so the traceback lands
    in the log the moment the task dies. This is the "bot looks dead with no
    error" failure mode CLAUDE.md warns against.

    Long-lived `create_task(...)` calls MUST attach this callback.
    """
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    except Exception:
        return
    if exc is not None:
        import traceback
        print_ts(f"background task {task.get_name()!r} died: {exc!r}\n"
                 + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
                 error=True)


def load_json(path: str, default=None):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return default if default is not None else {}
    except Exception as e:
        print_ts(f"Failed to load JSON {path}: {e}", error=True)
        return default if default is not None else {}


def save_json(path: str, data, *, mode: int | None = None) -> bool:
    """Atomically write JSON to `path` via tmpfile + rename.

    `mode` (e.g. 0o600) sets the resulting file's permissions explicitly.
    Without it, the tmp file is created with the process umask (typically 0o644)
    and that's what `os.replace` leaves behind — so a previously-chmod'd
    secrets file would silently widen back to 0o644 on the next save. Pass
    `mode=0o600` for files containing tokens or anything sensitive.
    """
    try:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
        if mode is not None:
            try:
                os.chmod(path, mode)
            except OSError as e:
                print_ts(f"chmod {oct(mode)} on {path} failed: {e}", error=True)
        return True
    except Exception as e:
        print_ts(f"Failed to save JSON {path}: {e}", error=True)
        return False


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def resolve_path(p: str) -> str:
    """Relative paths resolve against project root."""
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(project_root(), p))


def safe_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9._\-]+', '_', name)


# --- Path redaction ----------------------------------------------------------
# Tools must never emit absolute local paths to a model (and therefore to a
# non-owner Discord user): they expose the operator's home dir, OS username,
# directory layout, and worst case a secret embedded in a path or traceback.
# Real incident: an agent revealed an absolute /home/<user>/... path to a user.
# `redact_paths` scrubs operator-identifying paths out of any string before it
# reaches model_feedback / text / an error; `safe_path_display` renders a single
# path for display. Route every leaky site through these — do NOT hand-roll
# os.path.basename() at each call site.


def _sensitive_path_roots() -> list[str]:
    """Absolute path prefixes that identify the operator/host and must never
    reach the model or a non-owner user: the home directory and the project
    root. Together these cover every openflip data/tmp/output/agent/snapshot
    path and the symlinked extras tree under the home dir. Returned longest
    first so a project root nested under home is matched before the home prefix.
    """
    roots: set[str] = set()
    try:
        home = os.path.expanduser("~")
        if home and home not in ("/", ""):
            roots.add(home)
            roots.add(os.path.realpath(home))
    except Exception:
        pass
    try:
        pr = project_root()
        if pr and pr != "/":
            roots.add(pr)
            roots.add(os.path.realpath(pr))
    except Exception:
        pass
    return sorted((r for r in roots if r and r != "/"), key=len, reverse=True)


# Trailing path segments after a sensitive root: zero or more `/segment` (or
# `\segment` — Windows paths use backslashes, and both separators are accepted
# on Windows so a leaked path may carry either).
# The lookahead `(?![\w.\-])` after the root makes the root match a whole path
# component only — so a project root of `.../.openflip` does NOT partial-match
# `.../.openflip-extras/...` (which would leak the `-extras/...` tail). When the
# root pattern declines, the broader home-dir root catches the full path.
_REDACT_TAIL = r'(?![\w.\-])(?:[/\\][\w.\-]+)*'


def redact_paths(text: str) -> str:
    """Strip operator-identifying absolute local paths out of a string before
    it reaches the model or a non-owner user.

    Only paths under a sensitive root (the home dir or the project root) are
    touched: each is replaced with ``<path>`` (or ``<path>/<basename>`` when it
    points at a file), so the filename stays referenceable but the home dir,
    username, and directory layout are gone. Remote content is deliberately left
    intact — HTTP/CDN URLs and unrelated system paths (e.g. `/usr/...` inside a
    fetched page) are NOT operator paths, and over-redacting them would corrupt
    legitimate tool output (fetch_url, web_search, read_file content, etc.).
    """
    if not text:
        return text
    for root in _sensitive_path_roots():
        pattern = re.compile(re.escape(root) + _REDACT_TAIL)

        def _sub(m: "re.Match") -> str:
            s = m.group(0)
            if s == root:
                return "<path>"
            base = re.split(r"[/\\]", s)[-1]
            return f"<path>/{base}"

        text = pattern.sub(_sub, text)
    return text


def safe_path_display(p) -> str:
    """Render a single filesystem path safely for model/user output.

    Returns the path RELATIVE to the project root when it lives inside the repo
    (informative, leaks nothing), otherwise just the basename. Never emits an
    absolute local path. Use this anywhere a tool wants to tell the model which
    file it touched (e.g. "Written: agents/x/notes.md").
    """
    try:
        s = str(p)
    except Exception:
        return "<path>"
    if not s:
        return s
    try:
        root = os.path.realpath(project_root())
        rp = os.path.realpath(s)
        if rp == root or rp.startswith(root + os.sep):
            return os.path.relpath(rp, root)
    except Exception:
        pass
    return os.path.basename(s.rstrip("/\\")) or "<path>"


# --- Shared aiohttp ClientSession --------------------------------------------
# One process-wide session for outbound HTTP. Avoids the per-call connection
# pool churn that came from `async with aiohttp.ClientSession() as session:`
# being repeated in every tool. Sessions must be created inside a running
# event loop, so init is lazy on first use.

_http_session: Any = None
_http_session_lock: asyncio.Lock | None = None


async def http_session():
    """Return the process-wide aiohttp ClientSession, creating it on first use.

    Tools that previously did `async with aiohttp.ClientSession() as s:` should
    now do `s = await http_session()` and use it directly. Do not close it —
    it lives for the process lifetime.
    """
    global _http_session, _http_session_lock
    import aiohttp
    if _http_session_lock is None:
        _http_session_lock = asyncio.Lock()
    async with _http_session_lock:
        if _http_session is None or _http_session.closed:
            _http_session = aiohttp.ClientSession()
        return _http_session


# --- Outbound text sanitizer ------------------------------------------------
# Strips protocol-tag fragments from text we are about to send to Discord.
# Build the literal angle-bracket tag substrings via concatenation so this
# source file itself never contains them as contiguous substrings (the
# A1.1 parser-collision bug documented in TOOLS.md).
_TC_OPEN  = '<'  + 'tool_call'      + '>'
_TC_CLOSE = '</' + 'tool_call'      + '>'
_FC_OPEN  = '<'  + 'function_calls' + '>'
_FC_CLOSE = '</' + 'function_calls' + '>'
# Sub-tag prefixes for Claude Code's structured-call envelope. These openers
# carry attributes (e.g. name="ToolName"), so we only declare the prefix;
# the regex completes it with \b[^>]*>.
_INV_OPEN_PREFIX   = '<'  + 'invoke'
_INV_CLOSE         = '</' + 'invoke'    + '>'
_PARAM_OPEN_PREFIX = '<'  + 'parameter'
_PARAM_CLOSE       = '</' + 'parameter' + '>'

_PAIRED_TC_RE    = re.compile(re.escape(_TC_OPEN) + r'.*?' + re.escape(_TC_CLOSE), re.DOTALL)
_PAIRED_FC_RE    = re.compile(re.escape(_FC_OPEN) + r'.*?' + re.escape(_FC_CLOSE), re.DOTALL)
_PAIRED_INV_RE   = re.compile(re.escape(_INV_OPEN_PREFIX)   + r'\b[^>]*>.*?' + re.escape(_INV_CLOSE),   re.DOTALL)
_PAIRED_PARAM_RE = re.compile(re.escape(_PARAM_OPEN_PREFIX) + r'\b[^>]*>.*?' + re.escape(_PARAM_CLOSE), re.DOTALL)
# Known protocol openers — finding one in the leftover text after paired-strip
# means a broken envelope (wrong closer, format-mixed slip). Strip from the
# match position to end of text.
_ORPHAN_OPENER_RE = re.compile(
    r'<(?:tool_call|function_calls|invoke|parameter)(?:\s[^>]*)?>',
    re.IGNORECASE,
)
_ORPHAN_CLOSER_RE = re.compile(
    r'</(?:tool_call|function_calls|invoke|parameter)>',
    re.IGNORECASE,
)
# Typo'd <tool_*> tags (e.g. <tool_char>, <tool_command>, <tool_array>) — strip
# just the tag, keep surrounding text. The negative lookahead excludes the
# real <tool_call> tag, which is handled by the orphan-opener strip-to-end logic.
_TYPOD_TAG_RE = re.compile(
    r'</?tool_(?!call\b)[a-z_]+(?:\s[^>]*)?>',
    re.IGNORECASE,
)
# Hallucinated conversation-template tags — model sometimes emits these
# when context gets confused (long mixed-type history, session-resume
# oddities). Strip the whole paired block + any orphans.
_CONV_TAGS = ('user', 'human', 'assistant', 'system')
_PAIRED_CONV_RE = re.compile(
    r'<(?:' + '|'.join(_CONV_TAGS) + r')\b[^>]*>.*?</(?:' + '|'.join(_CONV_TAGS) + r')>',
    re.IGNORECASE | re.DOTALL,
)
_ORPHAN_CONV_OPENER_RE = re.compile(
    r'<(?:' + '|'.join(_CONV_TAGS) + r')\b[^>]*>',
    re.IGNORECASE,
)
_ORPHAN_CONV_CLOSER_RE = re.compile(
    r'</(?:' + '|'.join(_CONV_TAGS) + r')>',
    re.IGNORECASE,
)
# Markdown code spans — fenced ```...``` blocks and inline `...` backticks.
# Tag substrings inside these are quoted/example content (e.g. an agent
# describing the `<tool_call>` envelope in chat), not real envelopes.
_CODE_SPAN_RE = re.compile(
    r'```[\s\S]*?```'   # triple-backtick fenced block
    r'|``[^\n]+?``'      # double-backtick inline
    r'|`[^`\n]+?`',      # single-backtick inline
)


def _split_by_code_spans(text: str):
    """Return a list of (segment, is_code) tuples covering `text` end-to-end."""
    out = []
    pos = 0
    for m in _CODE_SPAN_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos:m.start()], False))
        out.append((m.group(), True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False))
    return out


def sanitize_outbound_text(text: str) -> str:
    '''Strip protocol-tag fragments + hallucinated conversation-template tags
    from outbound chat text.

    Two failure classes addressed:
    1. Tool-call envelope leakage (paired `<tool_call>`, `<function_calls>`,
       typo'd variants) — happens when an envelope is malformed and the
       parser doesn't consume it before the send path. Only PAIRED envelopes
       are stripped: a stray opener is preserved as-is because real dispatch
       goes through the structured `tool_calls`/`tool_use` field, and
       truncating at a stray opener drops legitimate message body when the
       model is describing the tool format in prose.
    2. Hallucinated user-prompt-template echoes — the model occasionally
       generates fake `<user>...</user>` / `<human>...</human>` /
       `<assistant>...</assistant>` blocks inside its reply when the
       context gets confused (mixed synthetic + real turns + long
       history). Strip them on the way out so the user never sees the
       hallucinated content.

    Tag substrings inside markdown code spans (fenced ``` blocks or
    `inline backticks`) are protected — those are quoted examples the
    model is showing the user, not real envelopes.
    '''
    if not text:
        return text
    # Strip interrupt markers — belt-and-suspenders in case the model echoes
    # them through any path we missed. Both the legacy 'assistant' role shape
    # and the new 'user' role shape are covered.
    text = text.replace('[Request interrupted by user]', '')
    text = text.replace('[FRAMEWORK]: Previous turn interrupted by new user message.', '')
    out_parts: list[str] = []
    for segment, is_code in _split_by_code_spans(text):
        if is_code:
            out_parts.append(segment)
            continue
        segment = _PAIRED_TC_RE.sub('', segment)
        segment = _PAIRED_FC_RE.sub('', segment)
        segment = _PAIRED_INV_RE.sub('', segment)
        segment = _PAIRED_PARAM_RE.sub('', segment)
        # Strip typo'd <tool_*> tags — just the tag, keep what follows.
        segment = _TYPOD_TAG_RE.sub('', segment)
        # Strip hallucinated conversation-template blocks. These never appear
        # legitimately in agent replies — they're either a hallucinated echo
        # of training-data prompt format, or a leaked synthetic-turn frame.
        segment = _PAIRED_CONV_RE.sub('', segment)
        segment = _ORPHAN_CONV_OPENER_RE.sub('', segment)
        segment = _ORPHAN_CONV_CLOSER_RE.sub('', segment)
        out_parts.append(segment)
    return ''.join(out_parts)
