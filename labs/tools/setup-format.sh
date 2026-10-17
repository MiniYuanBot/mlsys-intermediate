#!/usr/bin/env bash
# Install the foundations-pinned formatters under labs/.venv-format.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
. tools/formatter-versions.env
set +a

formatter_env=".venv-format"
if [[ ! -x "$formatter_env/bin/python" ]]; then
    python3 -m venv "$formatter_env"
fi
"$formatter_env/bin/python" -m pip install \
    "clang-format==$CLANG_FORMAT" "gersemi==$GERSEMI" "ruff==$RUFF"
npm install --prefix "$formatter_env" --no-audit --no-fund "prettier@$PRETTIER"

case "$(uname -s)" in
    Linux) formatter_os=linux ;;
    Darwin) formatter_os=darwin ;;
    *)
        echo "Use WSL2, Linux or macOS for this setup script." >&2
        exit 1
        ;;
esac
case "$(uname -m)" in
    x86_64 | amd64) formatter_arch=amd64 ;;
    arm64 | aarch64) formatter_arch=arm64 ;;
    *)
        echo "Unsupported architecture for automatic shfmt setup." >&2
        exit 1
        ;;
esac
formatter_url="https://github.com/mvdan/sh/releases/download/v${SHFMT}/shfmt_v${SHFMT}_${formatter_os}_${formatter_arch}"
curl --proto '=https' --tlsv1.2 -fsSL "$formatter_url" -o "$formatter_env/bin/shfmt.tmp"
chmod +x "$formatter_env/bin/shfmt.tmp"
mv "$formatter_env/bin/shfmt.tmp" "$formatter_env/bin/shfmt"
if ! command -v rustc > /dev/null || [[ "$(rustc --version | awk '{print $2}')" != "$RUST" ]]; then
    echo "Use Rust $RUST with rustfmt (same as foundations), then rerun the check." >&2
    exit 1
fi
rustup component add rustfmt
python3 tools/format_check.py
