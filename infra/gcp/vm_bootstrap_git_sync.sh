#!/usr/bin/env bash
# Run once ON THE GCE VM: clone SpotBaller from GitHub (or keep an existing clone),
# run setup_gcp_dev_vm.sh, then install cron so vm_pull_latest.sh keeps the VM in sync.
#
# Required env:
#   SPOTBALLER_REPO  — clone URL, e.g. https://github.com/tungtang/spotBaller.git
#                      or git@github.com:tungtang/spotBaller.git
#
# Optional:
#   SPOTBALLER_HOME         — default ~/SpotBaller
#   SPOTBALLER_GIT_BRANCH   — default main
#   SPOTBALLER_VM_CRON_MINS — passed to install_vm_auto_sync (default 5)

set -euo pipefail

SPOTBALLER_REPO="${SPOTBALLER_REPO:?Set SPOTBALLER_REPO to the git remote URL}"
SPOTBALLER_HOME="${SPOTBALLER_HOME:-$HOME/SpotBaller}"
BRANCH="${SPOTBALLER_GIT_BRANCH:-main}"

echo "==> SpotBaller git bootstrap: SPOTBALLER_HOME=$SPOTBALLER_HOME branch=$BRANCH"

if [[ -d "$SPOTBALLER_HOME/.git" ]]; then
  echo "==> Existing git repo; pulling latest"
  cd "$SPOTBALLER_HOME"
  git fetch origin
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
elif [[ -e "$SPOTBALLER_HOME" ]]; then
  echo "==> $SPOTBALLER_HOME exists without .git (e.g. old tarball deploy); renaming aside"
  mv "$SPOTBALLER_HOME" "${SPOTBALLER_HOME}.pre-git.$(date +%s)"
  git clone "$SPOTBALLER_REPO" "$SPOTBALLER_HOME"
  cd "$SPOTBALLER_HOME"
  git checkout "$BRANCH" 2>/dev/null || true
else
  git clone "$SPOTBALLER_REPO" "$SPOTBALLER_HOME"
  cd "$SPOTBALLER_HOME"
  git checkout "$BRANCH" 2>/dev/null || true
fi

cd "$SPOTBALLER_HOME"
SETUP="$SPOTBALLER_HOME/infra/gcp/setup_gcp_dev_vm.sh"
if [[ ! -f "$SETUP" ]]; then
  echo "ERROR: Missing $SETUP — push infra/gcp from your laptop (git add infra/gcp && git push) then re-run."
  exit 1
fi
chmod +x "$SPOTBALLER_HOME/infra/gcp/vm_pull_latest.sh" "$SPOTBALLER_HOME/infra/gcp/install_vm_auto_sync.sh" 2>/dev/null || true

echo "==> Python / venv / CUDA (idempotent)"
bash "$SETUP"

echo "==> Cron: periodic git pull + pip when origin changes"
bash "$SPOTBALLER_HOME/infra/gcp/install_vm_auto_sync.sh"

echo ""
echo "==> Done. VM will check GitHub every ${SPOTBALLER_VM_CRON_MINS:-5} minutes."
echo "    Log: ${SPOTBALLER_VM_SYNC_LOG:-$SPOTBALLER_HOME/runtime/vm_sync.log}"
echo "    Manual pull: bash $SPOTBALLER_HOME/infra/gcp/vm_pull_latest.sh"
