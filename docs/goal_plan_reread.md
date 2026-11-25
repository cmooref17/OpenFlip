# /goal, plan mode, and re-reading files after compaction

These copy three Claude Code features (from CC's goal.md, permission-modes.md and context-window.md).

## 1. /goal (openflip/goal.py)
- `/goal <condition>` (owner only, max 4000 chars): sets the conversation's goal and starts a turn right away with
  the condition as the instruction. A new goal replaces the old one.
- `/goal` with no argument: shows the condition, time running, turns evaluated, and the last reason (or the last
  achieved goal).
- `/goal clear|stop|off|reset|none|cancel`: "Goal cleared: <cond>" / "No goal set". `/reset` clears it too.
- Saved in `conversations/<cid>.goal.json`, so it survives a restart (counters reset on load, same as CC resume).
- **Evaluation**: when the model tries to end a turn, a small fast model (`hooks_model` / selector model, the same
  path as prompt hooks) reads the condition plus the recent transcript and returns met / not met / impossible,
  each with a reason.
  - not met: `[FRAMEWORK]` note with the reason, keep going (same mechanics as a blocking Stop hook)
  - met: clear the goal, record it as achieved, post "Goal achieved: <reason>"
  - impossible: clear the goal, post "Goal cleared, judged impossible: <reason>"
- No-progress guard (CC): if 3 continuations in a row make no tool calls, stop the loop and post a warning; the
  goal stays set, and the next operator message resumes evaluation.
- Background work defers (CC): a run_command background job still running from this conversation means no
  evaluation this turn. Its result turn evaluates when it lands.
- Errors: auth, credit and model-not-found errors clear the goal with "Goal cleared after an unrecoverable error";
  other errors leave it set.
- The goal turn cap is the existing max_tool_turns per turn. The goal evaluates on every turn of the
  conversation where the operator is the target (human, bg-result, synthetic), never on peer turns or subagent
  workers.

## 2. Plan mode (openflip/plan_mode.py)
- `/plan` toggles it on/off. `/plan <request>` turns it on and runs that request as a turn.
- While it's on: tools that write or change things are blocked in tool_executor (edit_file, write_file,
  delete_file, restore_snapshot, ...: an explicit WRITE set). run_command is allowed but the agent is told
  "read-only commands only" (CC prompts or classifies them; we have no prompt UI, so it's an instruction plus a
  block on known write verbs).
- The agent writes its plan with the `exit_plan_mode(plan)` tool (CC's ExitPlanMode). That saves
  `conversations/<cid>.plan.md`, posts the plan, and asks the operator to approve.
- Approval: the operator replies `/plan approve` (or yes/approve/go) → mode off, a turn starts: "Plan approved,
  carry it out". `/plan no <feedback>` keeps planning with that feedback.
- A system note is added every turn while plan mode is on (like CC's plan-mode reminder).
- After compaction the plan file is re-injected.

## 3. Re-read after compaction (openflip/recent_files.py)
- read_file / edit_file / write_file record their path in a per-conversation list.
- After a compaction, up to 5 of those files, most recently modified first, are re-read and attached to the
  compaction note. A file over 5000 tokens (~20000 chars) is attached as "Referenced file <path>" with no content.
- The active plan and the active goal go into the same note.
