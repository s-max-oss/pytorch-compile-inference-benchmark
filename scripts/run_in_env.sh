#!/usr/bin/env bash

# Run one command with the compiler/runtime required by torch.compile.
# Variables are passed only to the child process and never exported to the
# caller's shell, so other Conda environments remain untouched.

set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "ai-infra-compile" ]]; then
    printf 'Expected Conda environment ai-infra-compile, got %s\n' \
        "${CONDA_DEFAULT_ENV:-<none>}" >&2
    exit 1
fi

if [[ "$#" -eq 0 ]]; then
    printf 'Usage: scripts/run_in_env.sh <command> [args...]\n' >&2
    exit 2
fi

cc="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-gcc"
cxx="${CONDA_PREFIX}/bin/x86_64-conda-linux-gnu-g++"
for compiler in "$cc" "$cxx"; do
    if [[ ! -x "$compiler" ]]; then
        printf 'Compiler is missing or not executable: %s\n' "$compiler" >&2
        exit 1
    fi
done

# Use exactly the library path recorded in results/environment.json.  Do not
# inherit unrelated paths (for example ROS) from the interactive shell: that
# would make benchmark runs depend on how the terminal was started.  `env`
# applies this value only to the child command, so the caller remains intact.
library_path="${CONDA_PREFIX}/lib"

exec env \
    -u TORCHINDUCTOR_CACHE_DIR \
    CC="$cc" \
    CXX="$cxx" \
    LD_LIBRARY_PATH="$library_path" \
    "$@"
