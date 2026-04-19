#!/usr/bin/env bash
# Run on the GPU VM: fetch + fast-forward pull when origin has new commits, then refresh venv deps.
#
# Requires a normal git clone at SPOTBALLER_HOME (not a tarball-only tree without .git).
#
# Env:
#   SPOTBALLER_HOME       — repo root (default: ~/SpotBaller)
#   SPOTBALLER_GIT_BRANCH — tracked branch (default: main)
#   SPOTBALLER_VM_SYNC_LOG — log file (default: $SPOTBALLER_HOME/runtime/vm_sync.log)
#   SPOTBALLER_VM_POST_PULL — optional command run after a successful pull (e.g. systemctl restart spotballer-api)

set -euo pipefail

SPOTBALLER_HOME="${SPOTBALLER_HOME:-$HOME/SpotBaller}"
BRANCH="${SPOTBALLER_GIT_BRANCH:-main}"
LOG="${SPOTBALLER_VM_SYNC_LOG:-$SPOTBALLER_HOME/runtime/vm_sync.log}"

mkdir -p "$(dirname "$LOG")"
mkdir -p "$SPOTBALLER_HOME"

LOCK="${TMPDIR:-/tmp}/spotballer-vm-sync.lock"
exec 200>"$LOCK"
if ! flock -n 200; then
  exit 0
fi

{
  echo "=== $(date -Iseconds) vm_pull_latest branch=$BRANCH ==="
  cd "$SPOTBALLER_HOME"

  if [[ ! -d .git ]]; then
    echo "ERROR: $SPOTBALLER_HOME is not a git clone (.git missing)."
    echo "Clone once, e.g.:"
    echo "  git clone <YOUR_REPO_URL> $SPOTBALLER_HOME"
    echo "Or set SPOTBALLER_HOME to an existing clone path."
    exit 1
  fi

  git fetch origin

  if ! git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
    echo "ERROR: origin/$BRANCH does not exist. Set SPOTBALLER_GIT_BRANCH or push the branch."
    exit 1
  fi

  OLD=$(git rev-parse HEAD)
  git pull --ff-only origin "$BRANCH"
  NEW=$(git rev-parse HEAD)

  if [[ "$OLD" == "$NEW" ]]; then
    echo "Already up to date with origin/$BRANCH."
    exit 0
  fi

  echo "Updated $(git rev-parse --short "$OLD") -> $(git rev-parse --short "$NEW")"

  if [[ -d .venv ]]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
    echo "==> pip install -r requirements.txt"
    pip install -q -r requirements.txt
    if command -v nvidia-smi >/dev/null 2>&1; then
      PYTORCH_CUDA_URL="${PYTORCH_CUDA_URL:-https://download.pytorch.org/whl/cu124}"
      pip install -q --upgrade torch torchvision --index-url "$PYTORCH_CUDA_URL" || true
    fi
  else
    echo "WARN: no .venv; run infra/gcp/setup_gcp_dev_vm.sh once."
  fi

  if [[ -n "${SPOTBALLER_VM_POST_PULL:-}" ]]; then
    echo "==> SPOTBALLER_VM_POST_PULL: $SPOTBALLER_VM_POST_PULL"
    bash -c "$SPOTBALLER_VM_POST_PULL"
  fi

  echo "Pull complete at $(git rev-parse --short HEAD)."
} >>"$LOG" 2>&1

exit 0
