#!/usr/bin/env bash
# Synchronize Linear A codebase and experiment artifacts between local machine and a remote worker.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_HOST="${1:-pc}"
DIRECTION="${2:-push}" # push (local -> remote) or pull (remote -> local)

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
      "${REPO_ROOT}/" \
      "${TARGET_HOST}":~/Developer/linear-a/
    ;;
  pull)
    echo "==> Pulling experiment runs and reports from ${TARGET_HOST}:~/Developer/linear-a/"
    mkdir -p "${REPO_ROOT}/experiments/runs" "${REPO_ROOT}/reports"
    rsync -av \
      "${TARGET_HOST}":~/Developer/linear-a/experiments/runs/ \
      "${REPO_ROOT}/experiments/runs/" || true
    rsync -av \
      "${TARGET_HOST}":~/Developer/linear-a/reports/ \
      "${REPO_ROOT}/reports/" || true
    ;;
  *)
    echo "Unknown direction: ${DIRECTION}. Use 'push' or 'pull'."
    exit 1
    ;;
esac
