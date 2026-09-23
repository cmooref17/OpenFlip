"""Constants for memory recall, copied from Claude Code 2.1.280's bundled JS
(2026-09-22). Design choices matching CC, not mirrored external state."""
from __future__ import annotations

# Per-turn and per-session caps (CC: 5 files, M2e=200 lines, Fue=4096 bytes,
# MAX_SESSION_BYTES=61440, g$r=512 selector tokens).
MAX_FILES = 5
FILE_MAX_LINES = 200
FILE_MAX_BYTES = 4096
SESSION_MAX_BYTES = 61_440
SELECTOR_MAX_TOKENS = 512
SELECTOR_TIMEOUT_S = 20

# CC's selector system prompt, verbatim (m$r).
SELECTOR_SYSTEM = (
    "You are selecting memories that will be useful to Claude Code as it processes a user's query. "
    "The first message lists the available memory files with their filenames and descriptions; "
    "subsequent messages each contain one user query.\n"
    "Return a list of filenames for the memories that will clearly be useful to Claude Code as it "
    "processes the user's query (up to 5). Only include memories that you are certain will be "
    "helpful based on their name and description.\n"
    "- If you are unsure if a memory will be useful in processing the user's query, then do not "
    "include it in your list. Be selective and discerning.\n"
    "- If there are no memories in the list that would clearly be useful, feel free to return an "
    "empty list.\n"
    "- Be especially conservative with user-profile and project-overview memories ([user], "
    "[project]). These describe the user's ongoing focus, not what every question is about. A "
    "profile saying \"works on DB performance\" is NOT relevant to a question that merely contains "
    "the word \"performance\" unless the question is actually about that DB work. Match on what the "
    "question IS ABOUT, not on surface keyword overlap with who the user is.\n"
    "- Do not re-select memories you already returned for an earlier query in this conversation.\n"
)

# CC's injection prefix (case "relevant_memories").
INJECT_PREFIX = "Retrieved for possible relevance \u2014 use only if it actually applies to what the user asked."
