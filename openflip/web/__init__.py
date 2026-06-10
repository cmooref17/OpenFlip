# openflip.web — management webapp, mounted IN-PROCESS by openflip.main.
# Lives in the same event loop as the agent runners + cron + heartbeats,
# so it has direct access to RUNNERS, tool_settings._VALUES, conversation
# in-memory state, etc. (no filesystem-bridge cache-staleness).
#
# Single-operator login via argon2-hashed password in data/auth.json.
# Quart + Hypercorn + websockets.
from . import app  # noqa: F401 — expose the app module so main.py can call start_async()
