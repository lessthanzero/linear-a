#!/usr/bin/env bash
# Synchronize Linear A codebase and experiment artifacts between Mac and Fedora PC worker
set -euo pipefail

TARGET_HOST="${1:-pc}"
DIRECTION="${2:-push}" # push (Mac -> PC) or pull (PC -> Mac)

case "${DIRECTION}" in
  push)
    echo "==> Pushing Linear A workspace to ${TARGET_HOST}:~/Developer/linear-a/"
    ssh "${TARGET_HOST}" "mkdir -p ~/Developer/linear-a"
    rsync -av \
      --exclude '.venv' \
      --exclude '__pycache__' \
      --exclude '.pytest_cache' \
      --exclude '.ruff_cache' \
      --exclude '.git' \
      /Users/sashakatin/developer/linear-a/ \
      "${TARGET_HOST}":~/Developer/linear-a/
    ;;
  pull)
    echo "==> Pulling experiment runs and reports from ${TARGET_HOST}:~/Developer/linear-a/"
    mkdir -p experiments/runs reports
    rsync -av \
      "${TARGET_HOST}":~/Developer/linear-a/experiments/runs/ \
      /Users/sashakatin/developer/linear-a/experiments/runs/ || true
    rsync -av \
      "${TARGET_HOST}":~/Developer/linear-a/reports/ \
      /Users/sashakatin/developer/linear-a/reports/ || true
    ;;
  *)
    echo "Unknown direction: ${DIRECTION}. Use 'push' or 'pull'."
    exit 1
    ;;
esac
