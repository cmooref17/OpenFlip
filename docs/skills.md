# Skills (design, 2026-11-19)

Claude Code's skills, adapted for openflip's chat agents. Source of truth for the
CC behaviour copied here: code.claude.com/docs/en/skills.md + strings in the CC
2.1.318 bundle (quoted below where the wording is copied).

## Why openflip needs it

Every agent's system prompt carries how-tos it rarely needs (recipes, procedures,
long tool walkthroughs), paid on every request and crowding out the rules that
matter. Memory holds facts and is picked by a selector model; it is not a place
for procedures. A skill is a procedure or reference doc that costs one listing
line until it's needed, then loads in full. Same split as Claude Code:
always-on rules stay in the system files, sometimes-needed procedures become skills.

## What a skill is

A folder with a `SKILL.md`:

```
<skills dir>/<name>/
  SKILL.md          required: frontmatter + instructions
  reference.md ...  optional supporting files, read on demand with read_file
  scripts/...       optional scripts the instructions tell the agent to run
```

`SKILL.md` = optional YAML frontmatter between `---` lines (only when `---` is
line 1), then markdown instructions.

## Where skills live (scopes)

| Scope | Path | Who sees it | Tracked in the public repo? |
|---|---|---|---|
| Shared | `agents/_shared/skills/<name>/` | every agent | yes: must be generic, no personal names/paths |
| Agent | `agents/<id>/skills/<name>/` | that agent only | no (agent dirs are gitignored) |

Same name in both: the agent's own skill wins (CC: personal over project).
Disable per agent with agent.json `skills: {"enabled": false}` or hide named
ones with `skills: {"disabled": ["name", ...]}`. Global kill switch: config.json
`skills.enabled: false` or env `OPENFLIP_DISABLE_SKILLS=1`.

## Frontmatter (v1)

Fields openflip acts on:
- `name`: defaults to the folder name. Lowercase letters, digits, `-`, `_` only; a bad name = skill skipped with a log warning.
- `description`: what it does and when to use it, key use case first. Missing = first non-empty body line.
- `when_to_use`: appended to the description in the listing.
- `disable-model-invocation`: true = not in the listing, the model can't load it; only `/name` from the operator runs it.
- `user-invocable`: false = `/name` doesn't run it; model-only.
- `argument-hint`: shown in `/skills`.

Accepted and ignored (CC fields openflip has no equivalent for yet): `allowed-tools`,
`disallowed-tools`, `model`, `effort`, `context`, `agent`, `background`, `hooks`,
`paths`, `shell`, `metadata`, `license`, `compatibility`, `arguments`. Unknown
fields are ignored (CC behaviour). Malformed YAML = the skill still loads with no
fields (CC behaviour) and a warning is logged.

Parsing: flat `key: value` lines (quotes stripped, booleans accept true/false/
yes/no/on/off/1/0 in any case, CC), the same simple parser memory topic files use
(`memory_recall._frontmatter`). No PyYAML dependency: requirements.txt doesn't list
it and nothing in openflip imports it. Lists/maps in frontmatter are only needed by
the ignored fields, so they're skipped.

## The listing (system prompt)

Built like the deferred-tools block and captured in the per-session instruction
snapshot, so the cached prefix stays byte-stable:

```
# Skills

The following skills are available for use with the use_skill tool:

- <name>: <description> <when_to_use>
```

The header wording is CC's ("The following skills are available for use with the
Skill tool:"), with openflip's tool name. Sorted by name. Each entry's
description+when_to_use is cut at 1,536 chars (CC). Total listing budget:
config.json `skills.listing_budget_chars`, default 8,000 chars; over budget =
keep every name, drop descriptions from the end of the sorted list until it fits
(CC drops least-used first; openflip has no usage counts yet, so it drops the
longest descriptions first and logs a warning). No skills = no block at all.

Mid-session changes (a skill added/edited/removed) arrive through the existing
"These instruction files changed" note, labelled `(skills)`, exactly like an
AGENT.md edit; the system prompt itself only changes on /reset, compaction,
/reload or restart.

## The use_skill tool

Framework tool (in `FRAMEWORK_TOOL_NAMES`, always callable, always loaded, never
deferred, like `todo_write`):

```
use_skill(skill: str, args: str = "")
```

Returns the rendered skill as the tool result:

```
Base directory for this skill: <abs path to the skill folder>

<body with substitutions>
```

(`Base directory for this skill:` is CC's exact header.) Substitutions (CC):
`$ARGUMENTS` = the whole arg string; `$0`, `$1`... = shell-split args (missing
index stays literal); `${SKILL_DIR}` (and CC's `${CLAUDE_SKILL_DIR}`) = the skill
folder; `${SESSION_ID}` (and `${CLAUDE_SESSION_ID}`) = the conversation id.
Args given but no placeholder used = appended as `ARGUMENTS: <value>` (CC).
Errors: unknown name, disabled skill, or `disable-model-invocation` skill called
by the model = a tool error naming the problem.

Re-invoking with identical rendered content returns CC's line instead of a second
copy: `Skill /<name> is already loaded above; instructions unchanged.` (tracked per
conversation, cleared on compaction and /reset).

Dynamic context (`` !`cmd` ``) is NOT in v1: it runs shell commands inside a tool
that every agent holds, which bypasses run_command's admin gate. Skills that need
live data tell the agent to run the command itself.

## Operator invocation: /<name> and /skills

- `/<skill-name> [args]` at the start of an operator message runs the skill
  directly: the turn proceeds with the rendered skill content attached to that
  user message (as a `<skill name="...">` block) and the args as the request.
  Only when the name matches a loaded, user-invocable skill and no built-in text
  command has that name (built-ins always win; a skill named like one is
  skipped with a warning).
- `/skills` (text command): lists this agent's skills with scope, description
  and whether the model / the operator can invoke each.

Owner-only in v1 (like the other operator commands); non-owners' `/name` is just text.

## Compaction

After compaction CC re-attaches each invoked skill's body (first 5,000 tokens
each, 25,000 total). openflip v1: the conversation remembers which skills were
loaded; after a compaction the existing "[FRAMEWORK]: Your conversation was just
compacted" note also names them and says to call use_skill again if still
needed. Cheaper and honest; full re-attach can come later.

## Rules for writing skills (go in FRAMEWORK.md)

- A skill is a procedure or reference that's needed SOMETIMES. Rules that must
  hold every turn stay in the system files; rules that must be ENFORCED go in hooks.
- Description first sentence = when to use it, in the words a user would say.
- Keep SKILL.md short (CC: under 500 lines); push detail into supporting files.
- Shared skills are public: no personal names, paths, IDs.
- When a skill turns out wrong, fix the skill file, not a memory note about it.

## Files touched

- new `openflip/skills.py`: discovery, frontmatter parse, validation, listing, render.
- new `openflip/tools/skill.py`: the `use_skill` tool (+ `.gitignore` allow line).
- `openflip/pipeline.py`: add `use_skill` to `FRAMEWORK_TOOL_NAMES`.
- `openflip/tool_deferral.py`: add `use_skill` to `DEFAULT_ALWAYS_LOAD`.
- `openflip/runtime.py`: listing into `_cur_sources` + snapshot; `/<name>`
  expansion on operator turns; compaction note names loaded skills.
- `openflip/text_commands.py`: `/skills`; `/<name>` routing.
- `openflip/agent.py`: `skills` agent.json field.
- tests: `tests/test_skills.py` (loader, frontmatter, precedence, listing, budget,
  render/substitutions, tool errors, dedupe), `tests/test_skills_turn.py`
  (listing lands in the snapshot, /name expansion, built-in wins, non-owner ignored).
- docs: MANUAL.md section + agent.json/config.json field rows; FRAMEWORK.md rules.

## Verify live

1. A test skill in `agents/<id>/skills/` shows in the listing (in the system prompt / `/skills`).
2. Calling `use_skill` returns the rendered body; a second identical call returns the "already loaded" line.
3. `/skills` lists it; `/<name> hi` runs it on an operator turn.
4. Existing agents with no skills folder: no listing block, prompt unchanged (cache hit in the usage ledger).
