#!/usr/bin/env bash
# From your laptop (gcloud auth OK): upload vm_bootstrap_git_sync.sh and run it on the VM.
#
# Usage:
#   bash infra/gcp/laptop_vm_clone_and_sync.sh
#
# Env (optional):
#   VM, ZONE, GCP_PROJECT / PROJECT  — same defaults as deploy_to_vm.sh
#   SPOTBALLER_REPO — default https://github.com/tungtang/spotBaller.git
#   SPOTBALLER_GIT_BRANCH — default main

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VM="${VM:-spotballer-vm-2}"
ZONE="${ZONE:-asia-southeast1-a}"
PROJECT="${GCP_PROJECT:-${PROJECT:-datacloudpoc}}"
REPO_URL="${SPOTBALLER_REPO:-https://github.com/tungtang/spotBaller.git}"
BRANCH="${SPOTBALLER_GIT_BRANCH:-main}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found. Install Google Cloud SDK."
  exit 1
fi

BOOT="$ROOT/infra/gcp/vm_bootstrap_git_sync.sh"
if [[ ! -f "$BOOT" ]]; then
  echo "Not found: $BOOT"
  exit 1
fi

SSH_EXTRA=()
if [[ "${VM_SYNC_USE_IAP:-1}" != "0" ]]; then
  SSH_EXTRA=(--tunnel-through-iap)
fi

echo "==> Uploading bootstrap script to $VM ..."
gcloud compute scp "$BOOT" "${VM}:~/vm_bootstrap_git_sync.sh" --zone="$ZONE" --project="$PROJECT"

echo "==> Running bootstrap on VM (clone + setup + cron) ..."
gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" "${SSH_EXTRA[@]}" \
  --command="chmod +x ~/vm_bootstrap_git_sync.sh && \
    export SPOTBALLER_REPO='$REPO_URL' SPOTBALLER_GIT_BRANCH='$BRANCH' && \
    bash ~/vm_bootstrap_git_sync.sh"

echo "==> Finished."
