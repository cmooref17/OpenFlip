# Running openflip on Windows

openflip was built on Linux (and runs on macOS). As of 2026-06-11 the
framework is Windows-compatible: every POSIX-only code path either has a
Windows implementation or degrades gracefully with a clear message. This
page is the one-time setup, the launch story, and the honest list of what
is and isn't covered.

> **Verification status:** these changes were made and import/byte-compile
> verified on Linux. The Windows-specific branches (`msvcrt` locking, the
> `%3A` filename encoding, `start.bat`, console ANSI enablement) follow
> documented platform behavior but have not been executed on a real
> Windows box yet. First Windows boot should be treated as a smoke test.

## One-time setup

1. **Python 3.11+** (3.12 recommended). Install from python.org; check
   "Add python.exe to PATH".
2. **Clone the repo** somewhere like `C:\Users\<you>\.openflip`.
   - Note: the optional media tools (`image_gen`, `video_gen`, TTS,
     `audio_separate`, …) are local symlinks into a private
     `.openflip-extras` tree on the maintainer's machine and are not in
     the public repo. The tool loader skips missing extras silently.
3. **Virtualenv + deps** (PowerShell or cmd, from the repo root):
   ```bat
   python -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```
   The Linux `.lvenv` symlink does not apply on Windows; `start.bat`
   looks for `.venv\Scripts\python.exe` and falls back to `python` on
   PATH.
4. **Credentials**
   - **Anthropic (Claude subscription OAuth):** install Claude Code on
     the same Windows account and log in once (`claude` → `/login`).
     On Windows, Claude Code stores credentials at
     `%USERPROFILE%\.claude\.credentials.json` (a flat file — same as
     Linux; Windows does **not** use Credential Manager for this, per the
     official Claude Code authentication docs). openflip reads and
     refreshes that file directly. If you set `CLAUDE_CONFIG_DIR`, the
     file lives under that directory and openflip honors it.
   - **OpenAI (ChatGPT/Codex subscription OAuth):** run `codex login`.
     The token file is `%USERPROFILE%\.codex\auth.json` (override dir
     with `CODEX_HOME`). Caveat: newer Codex CLI builds can store creds
     in the OS keyring instead of the file (`cli_auth_credentials_store
     = "auto"`). If `auth.json` is missing after login, set
     `cli_auth_credentials_store = "file"` in `~/.codex/config.toml` and
     log in again — openflip only reads the file.
   - **Discord bot tokens:** same as every platform —
     `integrations.discord.tokens.<agent_id>` in `config.json`.
5. **First run** creates `config.json`, `agents/`, `cron/` etc. exactly
   as on Linux (see SETUP.md).

## Launching

Either run it directly:

```bat
.venv\Scripts\python -m openflip.main
```

or use the supervisor wrapper (recommended):

```bat
start.bat
```

`start.bat` sets `OPENFLIP_SUPERVISED=1` and relaunches openflip in a
loop (with a 2s backoff). That loop is what makes the `restart_gateway`
tool and the `/restart` command work on Windows — "restart" means "exit
cleanly and let the supervisor respawn the process". Ctrl+C (then `Y` at
the terminate prompt) stops it for good.

For an unattended service, use **NSSM** (`nssm install openflip
C:\...\.venv\Scripts\python.exe -m openflip.main`, with restart-on-exit,
and set `OPENFLIP_SUPERVISED=1` in the service environment) or Task
Scheduler with a restart-on-failure trigger. Alternatively set
`OPENFLIP_RESTART_CMD` to a command that restarts the service from the
outside (e.g. `nssm restart openflip`); `restart_gateway` runs that
command instead of exiting.

With neither `OPENFLIP_SUPERVISED` nor `OPENFLIP_RESTART_CMD` set,
`restart_gateway` refuses (with instructions) instead of taking the
framework down with nothing to bring it back.

## Platform behavior differences

- **Conversation filenames.** Conversation ids look like
  `discord:12345`. NTFS forbids `:` in filenames (it's the
  alternate-data-stream separator), so on Windows the on-disk name
  encodes it as `%3A`: `discord%3A12345.jsonl`. Ids keep the colon form
  everywhere else (URLs, config, memory). Implication: a conversations/
  directory copied from a Linux box will not be picked up on Windows
  (and vice versa) without renaming the files accordingly.
- **Cross-process OAuth refresh lock.** POSIX uses `fcntl.flock`;
  Windows uses `msvcrt.locking` (see `openflip/_file_lock.py`). Same
  stale-break and retry policy. Note that on Windows, Claude Code itself
  does not contend on `.oauth_refresh.lock` the same way it does on
  POSIX — the lock still serializes multiple openflip processes.
- **Signals / shutdown.** `loop.add_signal_handler` is unsupported on
  Windows; openflip already guards this. Ctrl+C raises
  KeyboardInterrupt and the process exits; the systemd-style graceful
  drain (close runners, then cancel stragglers) only runs where signal
  handlers are supported.
- **Console colors.** openflip enables ANSI escape processing on
  Windows consoles at import (`utils.py`); Windows Terminal already has
  it on, classic cmd.exe gets it switched on.
- **File permissions.** `os.chmod(..., 0o600)` calls succeed on Windows
  but only toggle the read-only flag — secrets files
  (`.credentials.json`, `auth.json`, web tokens, the sentinel HMAC key)
  inherit your user profile's ACL instead, which restricts them to your
  account by default. Equivalent protection, different mechanism.
- **Event loop.** Windows defaults to the Proactor event loop, which is
  what openflip needs (asyncio subprocesses for `run_command`,
  `claude_code`). Do not switch to `WindowsSelectorEventLoopPolicy` —
  subprocess tools would break. `aiodns` is not in `requirements.txt`;
  if you install it anyway, older aiodns/pycares builds raise "aiodns
  needs a SelectorEventLoop on Windows" (official pycares wheels ≥ 4.7.0
  are thread-safe and fine on the Proactor loop).
- **Web app.** The Quart/hypercorn management webapp mounts in-process
  the same way; hypercorn's asyncio worker runs on the Proactor loop.
  (Quart/hypercorn are optional — if not installed, openflip logs a
  warning and continues without the webapp.)

## Known limitations on Windows

- **iMessage transport** — macOS-only by nature; it raises at init and
  the framework skips it cleanly (`transports/imessage.py` is gated on
  `sys.platform == "darwin"`).
- **`restart_flask_app` tool** — manages a user systemd unit; on Windows
  it fails gracefully with "systemctl not found". (It's an extras tool
  for the maintainer's deployment, not framework-core.)
- **`claude_code` tool** — needs the `claude` CLI on PATH (or
  `CLAUDE_BIN` set to the full path of `claude.exe`). The native Windows
  Claude Code installer provides this.
- **Media/extras tools** (ComfyUI image/video, TTS, Demucs) — not in the
  public repo (see setup note above). ComfyUI/TTS are HTTP services, so
  if you do have the extras they work against any host; Demucs needs a
  Windows-compatible torch install.
- **start.sh / systemd unit** — Linux-only launch path; use `start.bat`
  / NSSM as above.
- **Test suite** — `tests/test_per_user_paths.py` asserts literal
  `/tmp` paths and is POSIX-shaped; run the tests on Linux.
- **Log file** (`log.txt`) is written with `errors`-tolerant text mode;
  on Windows it gets CRLF line endings via universal newlines — harmless.
