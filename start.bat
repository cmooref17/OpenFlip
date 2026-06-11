@echo off
rem openflip launcher for Windows. See docs/WINDOWS.md.
rem
rem The relaunch loop doubles as the restart supervisor: restart_gateway
rem exits the process cleanly and this loop brings it back up.
rem OPENFLIP_SUPERVISED=1 tells restart_gateway that exiting is safe
rem (without it, the tool refuses rather than strand the framework offline).
cd /d "%~dp0"
set OPENFLIP_SUPERVISED=1
:loop
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m openflip.main
) else (
    python -m openflip.main
)
rem Brief pause so a crash-loop can't spin at 100%% CPU. To stop openflip
rem for good: Ctrl+C, then Y at the "Terminate batch job" prompt (or close
rem the window).
timeout /t 2 /nobreak >nul
goto loop
