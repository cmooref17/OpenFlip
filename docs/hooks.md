# Hooks (design, v1)

Status: design for v1, being built. Modeled on Claude Code's hooks (code.claude.com/docs/en/hooks).
Why: instructions in SOUL/AGENT/FRAMEWORK are suggestions the model can skip. A hook runs every time,
outside the model: "for anything that has to happen every time, use a hook" (Claude Code docs).

## Config

Same three-level shape as Claude Code: event -> matcher groups -> handlers.

```json
"hooks": {
  "PostToolUse": [
    {"matcher": "edit_file|write_file",
     "hooks": [{"type": "command", "command": "~/bin/check-edit.sh", "timeout": 60}]}
  ],
  "Stop": [
    {"hooks": [{"type": "prompt", "prompt": "Did the agent finish what the user asked? $ARGUMENTS"}]}
  ]
}
```

- Where: `config.json` `hooks` (every agent, read at startup) and `agent.json` `hooks` (that agent,
  hot-reloaded like the rest of agent.json). Both lists run; global first. An identical handler
  in both runs once.
- Both files are owner-controlled. Nothing a chat user sends can add, pick or edit a hook; user
  content only reaches a hook as JSON on stdin, never inside the command string.
- Kill switch: `config.json` `"disable_all_hooks": true` or env `OPENFLIP_DISABLE_HOOKS=1`
  (Claude Code: `disableAllHooks`).

## Matchers (copied from Claude Code)

- `"*"`, `""` or omitted: every occurrence.
- Only letters, digits, `_`, `-`, spaces, `,` and `|`: exact name, or a list split on `|` / `,`.
- Anything else: an unanchored regular expression (`re.search`).
- Tool events match the tool name. `UserPromptSubmit` and `Stop` ignore the matcher.

## Events in v1

| Event | Fires | Can block? | Exit 2 / block does |
| - | - | - | - |
| `PreToolUse` | before a tool runs, after the ACL check | yes | the tool doesn't run; the reason is the tool error the model sees |
| `PostToolUse` | after a tool returns | no | the reason is appended to the tool result the model sees |
| `UserPromptSubmit` | when a turn starts, before the model call | yes | the turn ends without a model call; the reason is posted |
| `Stop` | when the model gives a final answer (no tool calls) | yes | the turn keeps going; the reason becomes the next instruction |

Tool hooks live in `tool_executor.execute_tool_calls`, which every provider uses (the claude_code
provider routes its MCP tool calls through it too), so they fire for every agent.

Stop loop guard (Claude Code parity): `stop_hook_active` is true on re-entry, and after 8
consecutive continuations the stop is allowed anyway. `max_tool_turns` still caps the whole turn.
Stop hooks don't run on a `STAY_SILENT` reply.

## Handlers in v1

**command**: `/bin/sh -c <command>`, cwd = the agent's directory, event JSON on stdin. Env adds
`OPENFLIP_AGENT_ID`, `OPENFLIP_AGENT_DIR`, `OPENFLIP_HOOK_EVENT`. Default timeout 600s (Claude Code's).
- exit 0: success. If stdout is a JSON object, its fields apply. Plain-text stdout on
  `UserPromptSubmit` is added as context.
- exit 2: block (on events that can block); stderr is the reason.
- anything else, timeout or bad JSON: non-blocking error, logged, the action goes ahead.

**prompt**: one-shot model call (config `hooks_model`, else the memory-recall selector model).
`$ARGUMENTS` in the prompt is replaced with the event JSON (appended if absent). The model must
answer `{"ok": bool, "reason": str, "impossible": bool}`. `ok:false` = block with `reason`;
on `Stop`, `impossible:true` lets the turn end. Default timeout 30s.

Handlers for one event run in parallel. Any hook failure is logged and never breaks a turn.

## JSON output fields read

- all events: `decision: "block"` + `reason`; `hookSpecificOutput.additionalContext`.
- `PreToolUse`: `hookSpecificOutput.permissionDecision` (`allow` / `deny`), `permissionDecisionReason`,
  `updatedInput` (replaces the whole args object). Precedence across hooks: deny > allow.

## Input JSON

Common: `hook_event_name`, `session_id` (conversation key), `transcript_path`
(`agents/<id>/conversations/<key>.jsonl`), `cwd`, `agent_id`, `speaker_id`, `speaker_is_owner`,
`transport`, `turn_kind` (`human` / `cron` / `heartbeat` / `peer` / `synthetic`). `human` only when a transport delivered a real message; a restart continuation or other framework-made turn is `synthetic`.
- Tool events: `tool_name`, `tool_input`, `tool_use_id`; `PostToolUse` adds
  `tool_response` (`ok`, `text`/`error`, capped at 20k chars).
- `UserPromptSubmit`: `prompt` (the user's raw text).
- `Stop`: `stop_hook_active`, `last_assistant_message`.

## Not in v1 (planned)

- `/goal <condition>`: a conversation-scoped prompt Stop hook stored in the conversation's meta.
- `/hooks`: list the hooks in effect for this agent.
- `SessionStart` (compact) / `PostCompact`: re-inject context after compaction.
- `http` handlers, the `if` field, `once`, `async`. Claude Code's terminal-only events (worktrees,
  permission prompts, elicitation) don't apply.

## Tests

`tests/test_hooks.py`: matcher table; command hook receives the stdin JSON; PreToolUse exit 2 stops
a real registered tool from running; `updatedInput` replaces args; PostToolUse context reaches the
model feedback; Stop block -> reason returned, 8-continuation cap; prompt hook with a fake model;
timeout and bad JSON are non-blocking; the kill switch.
