#!/bin/sh
# Docker entrypoint for the AgenticMemoryProject app container.
#
# Uses .venv/bin/python directly (not uv run) so that packages installed
# outside the lockfile (torch CPU, sentence-transformers) are not removed
# by uv's automatic sync on every invocation.
#
# Two usage modes:
#
#   1. Pipeline mode — arguments starting with "-" go to main.py:
#        docker compose run --rm app --variant bm25_baseline --ingest --qa
#        → .venv/bin/python main.py --variant bm25_baseline --ingest --qa
#
#   2. Script mode — run any Python script directly:
#        docker compose run --rm app python scripts/prepare_hotpotqa.py
#        → .venv/bin/python scripts/prepare_hotpotqa.py

set -e

PYTHON=".venv/bin/python"

if [ $# -eq 0 ] || echo "$1" | grep -q "^-"; then
    exec "$PYTHON" main.py "$@"
else
    # Strip leading "python" if provided explicitly
    if [ "$1" = "python" ]; then
        shift
    fi
    exec "$PYTHON" "$@"
fi
