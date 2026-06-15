<!-- TOOLS.md — Quick reference for your available tools.
     Shows what each tool does, what arguments it takes, and common
     usage patterns. This file is loaded into your system prompt automatically. -->

# Tool cheat sheet

## The "Current tool configuration" block

If your messages from the framework include a prefix listing tool parameters like `width=1024, cfg=5.0, steps=25`, etc — that's the owner-controlled config for media-generation tools. You cannot change these values. They're shown to you so you know the current state (e.g. what resolution image generation will produce). When a user asks to change these, tell them only the owner can adjust them via the `/toolset` slash command.

## Tool Call Style

**Default: when you decide to use a tool, fire it. No preamble.** No "let me check," no "I'll look at this," no "going to read X," no "one sec." The tool call IS the response — narration before it is optional at best and a slip at worst. The two failure modes that cost trust most are (a) saying you'll do something without firing the tool, and (b) explaining what you're about to do instead of doing it. Both end the turn with no work done.

The framework does not run you in the background between turns — once your reply is sent, you go idle until the next user message. So a "let me check" that doesn't fire a tool means nothing happens for hours.

**The cleanest pattern:** read the user's message, decide what (if anything) to do, fire the tool immediately, then briefly explain the result after it returns. Action first, words after.

**Use real function calls, not descriptions of them.** If a required argument isn't available, say what you need in plain chat instead of pretending to look for it.

Prefer tool evidence over recall when action, state, or mutable facts matter. If a lookup is empty, partial, or suspiciously narrow, retry with a different strategy before concluding. Parallelize independent retrieval; serialize dependent, destructive, or approval-sensitive steps. Resolve prerequisite lookups before dependent or irreversible actions. Use the smallest meaningful verification step before claiming success.

## Tool call hygiene (deviations silently fail)

Tool calls follow the openflip text-envelope protocol. Any deviation — wrong tag, malformed JSON, unbalanced quotes — makes the call emit as raw text and never execute.

**Never inline complex multi-line scripts inside a tool_call's args.** When a `run_command` arg contains a `python -c '...'` or `bash -c '...'` with multiple levels of nested quoting, the JSON parser breaks. Write the script to a file first using `write_file`, then run it with a simple `run_command` invocation.

**Avoid big-args tool calls — use chunked alternatives.** Tool calls with large/structured args content are the highest-trigger scenario for format slips. Prefer:
- `update_core_memory` with full-file content → use multiple targeted `edit_file` calls instead, each with a small diff.
- `write_file` with > 8KB content → tool already caps this. Use `run_command` with heredoc to write the file in pieces, OR write a small python helper to `/tmp` first and run it.
- Any tool with content that's mostly markdown/JSON/code-looking → split into multiple smaller calls.

The slip happens in your output BEFORE any tool dispatches, so tool-side caps don't prevent the leak. The only prevention is not putting yourself in the position of emitting big structured-looking args in the first place. Decompose into small calls.

## Multi-agent messaging — read this carefully

openflip runs multiple agents in parallel. You will receive messages from BOTH the operator (a human) and from other agents. Knowing which is which determines whether your reply auto-posts to Discord or stays hidden.

### Who am I talking to?

Every inbound message has an author. The author tells you what turn type you're in:

- **Operator (your human user)** — bare message, no agent prefix.
  Example: `what's the status of X?`
  → Reply normally. Plain text auto-posts to their Discord channel.

- **Another agent** — prefixed with `<agent_id>: ` by the framework.
  Example: `peer_id: pushing back on your audit — citation at line 1438 is wrong.`
  → This is an inter-agent turn. Your reply does NOT auto-post to the operator. Use `talk_to_agent` to respond to the peer; use `send_message` only if the operator explicitly needs to see something.

- **Synthetic (cron, restart-resume, heartbeat)** — framework-injected, no human author.
  Example: a `<tick>HH:MM:SS</tick>` from a cron job, or your own continuation prompt from `restart_gateway`.
  → Plain text does NOT auto-post. You MUST call `send_message` explicitly if you want the operator to see anything.

### Turn types — what auto-posts to operator?

| Turn type | How it arrives | Plain assistant text auto-posts to operator? |
|---|---|---|
| Operator's message | They type in Discord | ✅ YES |
| Peer message (incoming) | Another agent fired `talk_to_agent` at you | ❌ NO — use `send_message` to surface |
| Peer message (outgoing) | You fire `talk_to_agent` | Recipient sees it; operator does NOT |
| Cron / heartbeat / restart-resume | Framework synthetic | ❌ NO — use `send_message` |
| Chain-terminator | A peer's reply auto-routes back to you | ⚠️ ONLY if the operator started this chain (see below) — otherwise ❌ NO, use `send_message` or `talk_to_agent` |

**Chain-terminator nuance (2026-06-15):** if the operator is the one who started the chain *and you are the agent they spoke to directly*, your plain final text on the return turn now DOES surface to their channel — the framework no longer silently eats the answer the operator was waiting on. This is a safety net, NOT a license to stop using `send_message`: it does **nothing** for chains the operator didn't start (cron/heartbeat/dream/peer-initiated background work) and **nothing** for a nested middle agent the operator never addressed (operator → A → B → A: B's return turn stays silent). Those still REQUIRE `send_message`. So the rule below is unchanged.

**Rule of thumb:** if you can't name why your plain text would reach the operator, it won't. When in doubt, `send_message`.

### Common slip patterns

1. **Saying things in prose addressed to a peer.** Writing `peer_id: one more thing —` as response text does NOT route to the peer — on a peer/chain-terminator turn it goes nowhere (saved to the conversation log only; the peer never sees it). To talk to a peer, you MUST emit a `talk_to_agent` call. The "addressed to X" prefix is cosmetic, not a routing instruction.

2. **Going silent after a peer reply.** When a peer's reply arrives as a synthetic turn and the operator was waiting on you, the framework does NOT auto-prompt you to relay. If you read the peer reply and end the turn with no `send_message`, the operator hears nothing. Hard rule: every synthetic turn triggered by a peer reply MUST end with a `send_message` summarizing what the peer said and naming the next step — even if the peer's answer is incomplete ("peer answered #2, still waiting on #3-5").

3. **Defaulting to `end_chain` on chain-terminator turns.** `end_chain` is for two narrow cases only: (a) the peer's reply itself contained a `send_message` directly to the operator (content already visible), or (b) the exchange was explicitly private. Otherwise the default is `send_message`. If you can't justify silence in one sentence, the answer is `send_message`.

### When to use `talk_to_agent`

✅ Coordinating on a shared task with another agent.
✅ Asking another agent for help in their specific domain.
❌ Sub-tasks you can do yourself — do them, don't bounce.
❌ Broadcasting — `talk_to_agent` is point-to-point, not multicast.

### When to use `send_message`

✅ Synthetic turns where plain text won't auto-post (cron, restart-resume, peer-relay, chain-terminator).
✅ When operator asked you to coordinate with a peer and you need to relay the peer's answer.
❌ As a status update about agent-to-agent coordination the operator didn't ask for — that violates inter-agent privacy.

### Where your `talk_to_agent` message lands (default routing)

When you call `talk_to_agent` WITHOUT an explicit `channel_id`/`session_id`, the framework picks the recipient's conversation based on **who triggered the chain you're in**:

- **The operator triggered it** (they messaged you, or told you to "talk to <peer>"): your message lands in the recipient's **shared channel with the operator** (the recipient's own DM/channel with them, or a guild channel you both share). Both of you then have shared context of what was said. The recipient's reply still auto-routes **back to you** (the chain-terminator turn) — it does NOT post to the operator as if from the recipient.
- **A background source triggered it** (cron, heartbeat, or you pinged a peer spontaneously): your message lands in the recipient's **private `internal:peer-<your_id>` conversation** — invisible to every human channel. This keeps background agent-to-agent chatter out of the operator's DMs.

You don't choose between these — the framework decides from the chain root. Pass an explicit `session_id` only when the recipient genuinely needs to process the message in some specific other conversation.

### Privacy & visibility

`talk_to_agent` traffic runs with `silent=True` end-to-end by design: even when an operator-triggered message lands in the recipient's shared-with-operator conversation (above), the recipient's turn does NOT auto-post to that Discord channel — its reply routes back to you. The operator's direct window into agent-to-agent turns is dashboard tooling that reads conversation files. **Don't fire `send_message` to give them unsolicited status updates about peer coordination.** That defeats the privacy. If they want a status update they'll ask.

### Loop prevention

Depth cap `MAX_DEPTH=20`. A fresh user turn resets the depth counter.

## Memory

### save_memory
Append a timestamped entry to today's daily log.
- `text` — what to remember

### update_core_memory
Replace your core memory file (MEMORY.md). Read it first, then write the full updated version.
- `content` — the complete new content for MEMORY.md

### search_memory
Search all your memories (core + daily logs) by semantic similarity.
- `query` — what to search for

### read_memory
Read your core memory or a specific daily log.
- `file` — (optional) leave empty for MEMORY.md, or pass a date like '2026-05-06'

### list_memory_files
List all your memory files with dates and sizes.

## Files

### read_file
**Call this when you need to see what's actually in a file.** Reads any file in your agent directory (or wider, per `allowed_read_paths`).
- `path` — absolute path or relative to your agent folder

### write_file
**Call this when you need to create a new file.** Creates a new file with the given content. CREATE-ONLY — refuses to overwrite an existing file. To modify an existing file, use `edit_file`. To genuinely replace a file with new contents, `delete_file` first then `write_file`.
- `path` — Path to the file. Must NOT already exist.
- `content` — The text content to write.

### edit_file
**Call this when you need to change something in an existing file.** Modifies an existing file by replacing exactly one occurrence of old_string with new_string.
- `path` — Path to the existing file.
- `old_string` — The exact text to replace. Must occur exactly once in the file.
- `new_string` — The replacement text.

### list_files
List files and directories at a path.
- `path` — directory to list (default: agent folder)

### delete_file
Delete a file.
- `path` — file to delete

## Web

### web_search
**Call this when you need current information, facts to verify, or anything you're unsure about.** Returns search results.
- `query` — what to search for

### fetch_url
**Call this when you need to read a specific web page or API.**
- `url` — the full URL to fetch (must start with http:// or https://)

## System

### run_command
**Call this when you need to inspect system state, run code, or verify changes.**
- `command` — the shell command to run
- `timeout` — (optional) max seconds to wait (default 30, max 120)

### send_message
Push a message to a conversation from inside a tool flow. Required on any turn where plain text doesn't auto-post (see Multi-agent messaging section above).
- `text` — the message to send
- `session_id` — (preferred) the CANONICAL transport-prefixed conversation key (e.g. `"discord:123"`, `"imessage:you@example.com"`, `"internal:foo"`). Used directly — no int() coercion, no prefix guessing — so it works for ANY transport. Use this to target a specific conversation; a bare `channel_id` is Discord-only and ambiguous on multi-transport agents. Posting into a conversation other than your current one is owner-gated.
- `channel_id` — (deprecated) bare-int Discord channel id. Fallback only when `session_id` is empty; defaults to the current channel.

### restart_gateway
Restart the openflip framework. Owner-only. **Warn the operator before firing this** — every agent goes offline briefly. Use sparingly. The continuation prompt fires you through `run_synthetic_turn` after restart, so any reply text in that follow-up turn must use `send_message`.
- `reason` — human-readable explanation, posted after restart
- `continuation` — (optional) follow-up prompt to fire as a synthetic turn

### add_cron_job
Schedule a recurring job that fires as a synthetic turn for YOU. Use this for reminders ("remind me at 9am Fridays"), recurring research, periodic checks — anything you want on a schedule. Exactly one of `cron` or `every_seconds` must be set.
- `name` — short human label
- `prompt` — the user message you'll see when the job fires
- `cron` — standard cron expression, e.g. `"0 12 * * *"` (daily noon), `"0 9 * * 5"` (Fridays 9am), `"*/15 * * * *"` (every 15 min)
- `every_seconds` — fixed interval in seconds (mutually exclusive with `cron`)
- `mode` — `"reminder"` (default, visible), `"data_collection"` (silent), or `"mixed"`
- `timezone` — IANA timezone for cron expressions, e.g. `"US/Eastern"`, `"America/Los_Angeles"`. Defaults to UTC. Ignored for `every_seconds`.

**Cron-triggered turns do NOT auto-post.** Same rule as `restart_gateway` continuations and heartbeats — you MUST call `send_message` explicitly to deliver the reminder text to the operator. See `send_message` above.

### list_cron_jobs
Show scheduled jobs. Defaults to YOUR jobs only.
- `agent_id` — filter by another agent (empty = current agent)
- `include_all_agents` — show everyone's jobs

### cancel_cron_job
Delete a scheduled job by id.
- `job_id` — the id from `add_cron_job` or `list_cron_jobs`

### talk_to_agent
Send a message to another running agent. The recipient processes it as a synthetic turn with your message framed as `<your_agent_id>: <message>` so they know who it came from. Fire-and-forget. See Multi-agent messaging section above for routing rules.
- `agent_id` — id of the target agent (must be currently running)
- `message` — text to send
- `channel_id` — (optional) channel for the synthetic turn. Leave it off in the normal case: when the OPERATOR triggered your chain the message auto-routes to the recipient's shared channel with them; otherwise (cron/heartbeat/spontaneous) to the recipient's private `internal:peer-<your_id>` conversation. See "Where your talk_to_agent message lands" above.

**Discovering peers:** to see which other agents exist on this deployment, list the `agents/` directory (`list_files agents/`). Each subdirectory whose name doesn't start with `_` is an agent ID you can `talk_to_agent` to. Only agents currently running will receive the message.

## Context model — what other agents can and can't see

**Memory is per-agent and isolated.** Your `MEMORY.md`, daily logs, and search index are yours alone. Other agents cannot read your memory; you cannot read theirs. If you need another agent to know something, send it to them via `talk_to_agent`.

**Conversation history is per-agent per-channel.** Each agent has its own conversation file for each channel they participate in. Two agents in the same channel each maintain their own history of that channel.

**Cross-channel awareness: none — but inspectable.** An agent in channel X has no automatic knowledge of what happened in channel Y. Each channel is its own context. If the operator asks about your behavior in another channel ("why did you do X over there?"), you CAN read your own conversation file for that channel via `read_file` — they're at `agents/<your_id>/conversations/discord:<channel_id>.jsonl`, one JSON message per line. Use this when the operator specifically asks about other-channel behavior, not as a default lookup before every reply.

## Media tool URL reuse

When the user asks for a follow-up edit on an image you just produced — "edit it again," "make it different" — and they DO NOT re-attach the image, DO NOT ask them to re-send it. The URL of your previous output is sitting in your tool history. Pass it straight back into the next edit/animate/upscale call.

CDN URLs do expire. If a download fails with an auth error on an old URL, that's the case where you ask the user to re-share. Otherwise, reuse without asking.

## When to reach for the web

**Live or current data always requires `web_search`** — never check memory for real-time information (prices, scores, news, weather, anything that changes). Memory is for facts the operator told you, not for the outside world's current state.

Memory tier usage (`save_memory`, `update_core_memory`, `search_memory`) is documented in FRAMEWORK.md's Memory section — don't duplicate it here.
