#!/usr/bin/env bash
# POSIX convenience wrapper; the Python installer also supports native Windows.
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
exec "${PYTHON:-python3}" "$script_dir/setup_format.py" "$@"
