#!/usr/bin/env bash
# Remote worker execution script for Linear A on Fedora PC worker (pc or pc-remote)
set -euo pipefail

TARGET_HOST="${1:-pc}"
shift || true
COMMAND="${*:-uv run linear-a --help}"

echo "==> Running on ${TARGET_HOST}: ${COMMAND}"
ssh -o BatchMode=yes -o ConnectTimeout=10 "${TARGET_HOST}" "
  export PATH=\$HOME/.local/bin:\$PATH
  cd ~/Developer/linear-a
  ${COMMAND}
"
