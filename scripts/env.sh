#!/usr/bin/env bash

# Backward-compatible entry point.  Run it; do not source it.
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    printf 'Do not source scripts/env.sh; run scripts/run_in_env.sh <command> instead.\n' >&2
    return 1
fi

exec "$(dirname "$0")/run_in_env.sh" "$@"
