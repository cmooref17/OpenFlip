"""Skills: on-demand procedures and reference docs (Claude Code's skills).

Design: docs/skills.md. A skill is a folder with a SKILL.md (optional
frontmatter + instructions). Only a one-line listing per skill rides in the
system prompt; the full body loads when the model calls use_skill or the
operator types /<name>. Two scopes: agents/_shared/skills/ (every agent,
tracked, must be generic) and agents/<id>/skills/ (that agent; wins on a
name clash, like Claude Code's personal-over-project).

Nothing here raises into a turn: bad skills are skipped (or loaded with no
frontmatter, Claude Code's behaviour) and reported as warnings.
"""
from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field

# Claude Code's wording ("...for use with the Skill tool:"), openflip's tool name.
LISTING_HEADER = "The following skills are available for use with the use_skill tool:"
# Claude Code's exact header and re-invoke line.
BASE_DIR_HEADER = "Base directory for this skill: "
ALREADY_LOADED = "Skill /{name} is already loaded above; instructions unchanged."
DESC_CAP = 1536                    # Claude Code: description + when_to_use cap in the listing
DEFAULT_LISTING_BUDGET = 8000      # chars; config.json skills.listing_budget_chars
SKILL_FILE = "SKILL.md"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_TRUE = {"true", "yes", "on", "1"}
_FALSE = {"false", "no", "off", "0"}


@dataclass
class Skill:
    name: str
    description: str
    when_to_use: str
    scope: str                      # "shared" or "agent"
    dir: str                        # absolute folder path
    body: str
    model_invocable: bool = True    # False = disable-model-invocation: true
    user_invocable: bool = True     # False = user-invocable: false
    argument_hint: str = ""

    @property
    def path(self) -> str:
        return os.path.join(self.dir, SKILL_FILE)


def _config() -> dict:
    try:
        from .config_global import get_config
        return (get_config() or {}).get("skills") or {}
    except Exception:
        return {}


def skills_enabled(agent=None) -> bool:
    """Kill switches: env OPENFLIP_DISABLE_SKILLS=1, config.json
    skills.enabled=false, agent.json skills.enabled=false. Default on."""
    if os.environ.get("OPENFLIP_DISABLE_SKILLS") == "1":
        return False
    if _config().get("enabled", True) is False:
        return False
    a = getattr(agent, "skills", None) or {}
    return not (isinstance(a, dict) and a.get("enabled", True) is False)


def listing_budget() -> int:
    try:
        v = int(_config().get("listing_budget_chars", DEFAULT_LISTING_BUDGET))
        return v if v > 0 else DEFAULT_LISTING_BUDGET
    except (TypeError, ValueError):
        return DEFAULT_LISTING_BUDGET


def skill_dirs(agent_dir: str) -> list[tuple[str, str]]:
    """(scope, dir) in load order: shared first, the agent's own last (wins)."""
    agents_root = os.path.dirname(os.path.abspath(agent_dir))
    return [
        ("shared", os.path.join(agents_root, "_shared", "skills")),
        ("agent", os.path.join(os.path.abspath(agent_dir), "skills")),
    ]


def _bool(raw, default: bool) -> bool:
    s = str(raw).strip().lower()
    if s in _TRUE:
        return True
    if s in _FALSE:
        return False
    return default


def parse_frontmatter(text: str) -> tuple[dict, str, str]:
    """(fields, body, warning). Frontmatter only counts when '---' is line 1
    (Claude Code). Flat `key: value` lines; indented lines and '- ' items
    (lists/maps, only used by fields openflip ignores) are skipped. A line
    that isn't `key: value` = malformed: the skill loads with NO fields and
    the whole text after the closing --- as body (Claude Code's behaviour)."""
    norm = text.replace("\r\n", "\n")
    if not norm.startswith("---\n"):
        return {}, norm, ""
    end = norm.find("\n---", 3)
    while end != -1 and norm[end + 4:end + 5] not in ("", "\n"):
        end = norm.find("\n---", end + 4)
    if end == -1:
        return {}, norm, "frontmatter has no closing --- line; read the whole file as instructions"
    raw = norm[4:end]
    body = norm[end + 4:].lstrip("\n")
    fields: dict = {}
    for line in raw.split("\n"):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[:1] in (" ", "\t") or line.lstrip().startswith("- "):
            continue
        if ":" not in line:
            return {}, body, f"frontmatter line isn't 'key: value': {line.strip()[:60]!r}; loaded with no fields"
        k, v = line.split(":", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            v = v[1:-1]
        fields[k.strip()] = v
    return fields, body, ""


def _first_line(body: str) -> str:
    for line in body.split("\n"):
        s = line.strip().lstrip("#").strip()
        if s:
            return s
    return ""


def load_skills(agent_dir: str, agent=None) -> tuple[dict[str, Skill], list[str]]:
    """All skills this agent can see, by name, plus warnings. Shared loads
    first, the agent's own folder second and replaces a shared skill of the
    same name. agent.json skills.disabled hides named skills."""
    found: dict[str, Skill] = {}
    warnings: list[str] = []
    if not agent_dir or not skills_enabled(agent):
        return found, warnings
    for scope, root in skill_dirs(agent_dir):
        if not os.path.isdir(root):
            continue
        try:
            entries = sorted(os.listdir(root))
        except OSError as e:
            warnings.append(f"can't list {root}: {e}")
            continue
        for folder in entries:
            sdir = os.path.join(root, folder)
            spath = os.path.join(sdir, SKILL_FILE)
            if folder.startswith(".") or not os.path.isfile(spath):
                continue
            try:
                with open(spath, "r", encoding="utf-8") as f:
                    text = f.read()
            except (OSError, UnicodeDecodeError) as e:
                warnings.append(f"{scope} skill {folder!r}: can't read {SKILL_FILE}: {e}")
                continue
            fields, body, warn = parse_frontmatter(text)
            if warn:
                warnings.append(f"{scope} skill {folder!r}: {warn}")
            name = (fields.get("name") or folder).strip()
            if not _NAME_RE.match(name):
                warnings.append(f"{scope} skill {folder!r}: bad name {name!r} "
                                f"(lowercase letters, digits, - and _ only); skipped")
                continue
            desc = (fields.get("description") or "").strip() or _first_line(body)
            if name in found and found[name].scope == scope:
                warnings.append(f"{scope} skill {folder!r}: name {name!r} already used by "
                                f"{os.path.basename(found[name].dir)!r}; skipped")
                continue
            found[name] = Skill(
                name=name, description=desc,
                when_to_use=(fields.get("when_to_use") or "").strip(),
                scope=scope, dir=os.path.abspath(sdir), body=body.strip(),
                model_invocable=not _bool(fields.get("disable-model-invocation", ""), False),
                user_invocable=_bool(fields.get("user-invocable", ""), True),
                argument_hint=(fields.get("argument-hint") or "").strip(),
            )
    disabled = set()
    a = getattr(agent, "skills", None) or {}
    if isinstance(a, dict) and isinstance(a.get("disabled"), list):
        disabled = {str(x).strip() for x in a["disabled"]}
    for name in disabled:
        found.pop(name, None)
    return found, warnings


def _entry_text(s: Skill) -> str:
    t = " ".join(x for x in (s.description, s.when_to_use) if x).replace("\n", " ").strip()
    return t[:DESC_CAP]


def listing_block(skills: dict[str, Skill], budget: int | None = None) -> tuple[str, list[str]]:
    """The '# Skills' system-prompt section (stable bytes: sorted by name),
    plus warnings. Only model-invocable skills are listed (Claude Code leaves
    disable-model-invocation skills out). Over budget: every name stays, the
    longest descriptions are dropped first until it fits."""
    listed = sorted((s for s in skills.values() if s.model_invocable), key=lambda s: s.name)
    if not listed:
        return "", []
    budget = budget or listing_budget()
    descs = {s.name: _entry_text(s) for s in listed}
    head = f"# Skills\n\n{LISTING_HEADER}\n\n"

    def build() -> str:
        return head + "\n".join(
            f"- {s.name}: {descs[s.name]}" if descs[s.name] else f"- {s.name}" for s in listed)

    text, dropped = build(), []
    for name in sorted(descs, key=lambda n: (-len(descs[n]), n)):
        if len(text) <= budget:
            break
        if descs[name]:
            descs[name] = ""
            dropped.append(name)
            text = build()
    warns = [f"skill listing over its {budget}-char budget: dropped the descriptions of "
             f"{', '.join(sorted(dropped))}"] if dropped else []
    return text, warns


_SUB_RE = re.compile(
    r"(\\?)\$(?:ARGUMENTS\[(\d+)\]|(ARGUMENTS)|(\d+)|\{(?:CLAUDE_)?(SKILL_DIR|SESSION_ID)\})")


def _split_args(args: str) -> list[str]:
    try:
        return shlex.split(args)
    except ValueError:
        return args.split()


def render(skill: Skill, args: str = "", session_id: str = "") -> str:
    """The text the model receives: Claude Code's base-directory header, then
    the body with substitutions done in ONE pass (an argument containing $1
    stays literal). \\$ escapes a placeholder. Args with no placeholder used
    are appended as 'ARGUMENTS: <value>' (Claude Code)."""
    args = (args or "").strip()
    parts = _split_args(args) if args else []
    used = {"args": False}

    def sub(m: re.Match) -> str:
        esc, idx_a, whole, idx, var = m.groups()
        if esc:
            return m.group(0)[1:]
        if whole:
            used["args"] = True
            return args
        if idx_a is not None or idx is not None:
            i = int(idx_a if idx_a is not None else idx)
            if i < len(parts):
                used["args"] = True
                return parts[i]
            return m.group(0)                       # missing index stays literal (CC)
        if var == "SKILL_DIR":
            return skill.dir
        return session_id
    body = _SUB_RE.sub(sub, skill.body)
    if args and not used["args"]:
        body = f"{body}\n\nARGUMENTS: {args}"
    return f"{BASE_DIR_HEADER}{skill.dir}\n\n{body}"


# Per-conversation record of what each skill rendered to, for Claude Code's
# "already loaded" reply. Process memory: after a restart a re-invoke just
# sends the content again (harmless). Cleared on compaction and /reset.
_loaded: dict[str, dict[str, str]] = {}


def note_loaded(conversation_id: str, name: str, rendered: str) -> bool:
    """Record a render; True when the identical text was already loaded in
    this conversation (the caller sends the short line instead)."""
    if not conversation_id:
        return False
    seen = _loaded.setdefault(conversation_id, {})
    if seen.get(name) == rendered:
        return True
    seen[name] = rendered
    return False


def loaded_names(conversation_id: str) -> list[str]:
    return sorted(_loaded.get(conversation_id, {}))


def forget_loaded(conversation_id: str) -> None:
    _loaded.pop(conversation_id, None)


def operator_invocation(raw_text: str, agent_dir: str, agent=None, *,
                        is_owner: bool, conversation_id: str = "",
                        reserved: set[str] | None = None) -> tuple[str, str]:
    """An operator message that starts with /<skill-name> [args] runs that skill
    directly (Claude Code). Returns (block to attach to the user message, skill
    name), or ("", "") when it isn't a skill invocation: not the owner, no
    leading slash, a built-in command name (built-ins always win), an unknown
    skill, or a skill with user-invocable: false. The rendered body is
    recorded as loaded, so a later use_skill for it gets the short line."""
    text = (raw_text or "").strip()
    if not is_owner or not text.startswith("/"):
        return "", ""
    head, _, rest = text.partition(" ")
    name = head[1:].strip().lower()
    if not name or (reserved and f"/{name}" in reserved):
        return "", ""
    found, _ = load_skills(agent_dir, agent)
    s = found.get(name)
    if s is None or not s.user_invocable:
        return "", ""
    rendered = render(s, rest.strip(), session_id=conversation_id)
    note_loaded(conversation_id, name, rendered)
    block = (f"<skill name=\"{name}\">\nThe operator ran /{name}"
             + (f" with: {rest.strip()}" if rest.strip() else "")
             + ". Follow this skill now.\n\n" + rendered + "\n</skill>")
    return block, name
