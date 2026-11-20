#!/bin/sh
# Run every openflip test file. Non-zero exit if any file fails.
# Usage: tests/run.sh            (all)      tests/run.sh test_hooks.py  (one)
cd "$(dirname "$0")/.." || exit 1
PY=.lvenv/bin/python
if [ -n "$1" ]; then
    exec "$PY" "tests/$1"
fi
total=0; failed=0; failed_names=""
for f in tests/test_*.py; do
    total=$((total + 1))
    if timeout 600 "$PY" "$f" > /tmp/openflip_test_out.$$ 2>&1; then
        echo "PASS $f"
    else
        failed=$((failed + 1)); failed_names="$failed_names $f"
        echo "FAIL $f"; tail -20 /tmp/openflip_test_out.$$ | sed 's/^/    /'
    fi
done
rm -f /tmp/openflip_test_out.$$
echo "TOTAL: $total files, $((total - failed)) passed, $failed failed${failed_names:+ ($failed_names )}"
[ "$failed" -eq 0 ]
