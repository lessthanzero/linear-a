#!/usr/bin/env bash
# Push repository directly to self-hosted Forgejo on Fedora PC worker
set -euo pipefail

TARGET_HOST="${1:-pc}"
REPO_NAME="linear-a"

echo "==> Retrieving Forgejo access token from ${TARGET_HOST}..."
TOKEN=$(ssh "${TARGET_HOST}" "cat ~/.config/homelab/forgejo-token 2>/dev/null || true")
if [[ -z "${TOKEN}" ]]; then
  echo "Generating fresh token on ${TARGET_HOST}..."
  TOKEN=$(ssh "${TARGET_HOST}" "podman exec -u 1000 forgejo forgejo admin user generate-access-token --username homelab-admin --token-name cli-push-\$(date +%s) --raw --scopes all")
  ssh "${TARGET_HOST}" "echo '${TOKEN}' > ~/.config/homelab/forgejo-token && chmod 600 ~/.config/homelab/forgejo-token"
fi

LOCAL_PORT=33001
echo "==> Opening secure SSH tunnel to Forgejo on ${TARGET_HOST} (localhost:${LOCAL_PORT} -> 127.0.0.1:3001)..."
ssh -f -N -L ${LOCAL_PORT}:127.0.0.1:3001 "${TARGET_HOST}"
sleep 1

cleanup() {
  pkill -f "ssh -f -N -L ${LOCAL_PORT}:127.0.0.1:3001 ${TARGET_HOST}" 2>/dev/null || true
}
trap cleanup EXIT

FORGEJO_AUTH_URL="http://${TOKEN}@127.0.0.1:${LOCAL_PORT}/homelab-admin/${REPO_NAME}.git"

echo "==> Pushing to Forgejo repository (homelab-admin/${REPO_NAME})..."
git push "${FORGEJO_AUTH_URL}" main --tags

echo "==> Updating working tree on ${TARGET_HOST}:~/Developer/${REPO_NAME}..."
ssh "${TARGET_HOST}" "
  if [[ -d ~/Developer/${REPO_NAME}/.git ]]; then
    cd ~/Developer/${REPO_NAME}
    git remote set-url forgejo 'http://${TOKEN}@127.0.0.1:3001/homelab-admin/${REPO_NAME}.git' 2>/dev/null || git remote add forgejo 'http://${TOKEN}@127.0.0.1:3001/homelab-admin/${REPO_NAME}.git'
    git pull forgejo main || true
  else
    rm -rf ~/Developer/${REPO_NAME}_tmp
    git clone 'http://${TOKEN}@127.0.0.1:3001/homelab-admin/${REPO_NAME}.git' ~/Developer/${REPO_NAME}_tmp
    rm -rf ~/Developer/${REPO_NAME}/.git
    mv ~/Developer/${REPO_NAME}_tmp/.git ~/Developer/${REPO_NAME}/.git
    rm -rf ~/Developer/${REPO_NAME}_tmp
    cd ~/Developer/${REPO_NAME}
    git remote set-url origin 'http://${TOKEN}@127.0.0.1:3001/homelab-admin/${REPO_NAME}.git'
    git reset --hard HEAD
  fi
"

echo ""
echo "✓ Successfully synchronized ${REPO_NAME} to Forgejo on ${TARGET_HOST}!"
echo "  Web UI (Tailscale): https://fedora.tail707210.ts.net/homelab-admin/${REPO_NAME}"
echo "  Local Web UI:       http://localhost:3001/homelab-admin/${REPO_NAME} (via 'ssh -L 3001:127.0.0.1:3001 pc')"
