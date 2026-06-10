"""openflip_web configuration. Paths are resolved relative to the
openflip root (one dir above this package) so the webapp stays bound to
the same data the openflip framework reads/writes."""
from __future__ import annotations

import os
from pathlib import Path

# Repo root: ~/.openflip
# This file lives at openflip/web/config.py, so go up THREE levels:
# config.py → web/ → openflip/ → repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Where openflip's runtime data lives
OPENFLIP_AGENTS_DIR = PROJECT_ROOT / "agents"
OPENFLIP_DATA_DIR = PROJECT_ROOT / "data"
OPENFLIP_LOG = PROJECT_ROOT / "log.txt"
OPENFLIP_CONFIG_JSON = PROJECT_ROOT / "config.json"
OPENFLIP_AGENT_STATE_JSON = PROJECT_ROOT / "agent_state.json"
OPENFLIP_TOOL_SETTINGS = PROJECT_ROOT / "data" / "tool_settings.json"
OPENFLIP_EVENTS_JSONL = PROJECT_ROOT / "data" / "events.jsonl"
OPENFLIP_CRON_DIR = PROJECT_ROOT / "cron"

# Webapp's own state
WEB_DATA_DIR = Path(__file__).resolve().parent / "data"
AUTH_FILE = WEB_DATA_DIR / "auth.json"
SECRET_KEY_FILE = WEB_DATA_DIR / "secret_key"
# Bearer token gating the inbound POST /trigger/<agent_id> API (see
# agents/_shared/MANUAL.md). Single-line plaintext file (machine-to-machine
# credential, read back verbatim and compared constant-time — same trust
# model as secret_key). Created mode 0600 ONCE at webapp startup; the
# request path only ever READS it (never get-or-creates — see auth.py).
TRIGGER_TOKEN_FILE = WEB_DATA_DIR / "trigger_token"
# Per-agent, per-hook /trigger token secret store: {agent_id: {token_name:
# secret}}, mode 0600. Namespaced-by-agent sibling of TRIGGER_TOKEN_FILE; see
# auth.read_trigger_token_secrets.
TRIGGER_TOKENS_FILE = WEB_DATA_DIR / "trigger_token_secrets.json"

# Server bind (LAN-accessible by default — single login protects it)
# Bind to loopback by default — the web app has username/password auth
# but no TLS and no rate-limiting on /login, so exposing it to the LAN
# means anyone on the same network can brute-force the password
# (MED-4 from the security audit). For LAN access, set
# OPENFLIP_WEB_HOST=0.0.0.0 explicitly OR (preferred) front the loopback
# port with nginx/caddy running TLS.
BIND_HOST = os.environ.get("OPENFLIP_WEB_HOST", "127.0.0.1")
BIND_PORT = int(os.environ.get("OPENFLIP_WEB_PORT", "1750"))

# Session cookie name
SESSION_COOKIE = "openflip_web_session"
