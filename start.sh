#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ -d ".lvenv" ]; then
    source .lvenv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi
# Cache-prefix diagnostic logging. Set to 1 to enable per-turn hash dumps
# of the Anthropic request prefix (tools/system/messages) for prompt-cache
# debugging. Off by default.
# export OPENFLIP_CACHE_DIAG=1
python -m openflip.main
