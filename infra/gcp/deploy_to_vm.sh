#!/usr/bin/env bash
# Upload SpotBaller to a GCE VM and run infra/gcp/setup_gcp_dev_vm.sh on it.
#
# Prerequisites:
#   - gcloud auth login && gcloud config set project YOUR_PROJECT
#   - Google Cloud SDK 565+ needs Python 3.10+. On macOS with only Python 3.9,
#     install Python 3.12 from https://www.python.org/downloads/ or Homebrew, then:
#       export CLOUDSDK_PYTHON="/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"
#     (adjust path). Verify: $(CLOUDSDK_PYTHON) --version
#
# Usage (from repo root):
#   bash infra/gcp/deploy_to_vm.sh
#
# Override defaults:
#   VM=spotballer-vm1 ZONE=asia-southeast1-b PROJECT=datacloudpoc bash infra/gcp/deploy_to_vm.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VM="${VM:-spotballer-vm1}"
ZONE="${ZONE:-asia-southeast1-b}"
PROJECT="${PROJECT:-datacloudpoc}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not in PATH. Install Google Cloud SDK and add to PATH."
  exit 1
fi

PY="${CLOUDSDK_PYTHON:-}"
if [[ -z "$PY" ]]; then
  if command -v python3 >/dev/null 2>&1 && python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PY="$(command -v python3)"
  fi
fi
if [[ -z "$PY" ]] || ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
  echo "gcloud compute ssh/scp requires Python 3.10+ (your gcloud SDK is incompatible with Python 3.9)."
  echo "Install Python 3.12+, then run:"
  echo "  export CLOUDSDK_PYTHON=/path/to/python3.12"
  echo "  bash infra/gcp/deploy_to_vm.sh"
  exit 1
fi
export CLOUDSDK_PYTHON="$PY"

TAR="$(mktemp /tmp/spotballer-deploy-XXXXXX.tgz)"
cleanup() { rm -f "$TAR"; }
trap cleanup EXIT

echo "==> Packing $ROOT -> $TAR"
(
  cd "$ROOT"
  tar czf "$TAR" \
    --exclude='.venv' \
    --exclude='runtime' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.mp4' \
    --exclude='*.mov' \
    .
)

echo "==> Uploading to ${VM}:${ZONE} (project ${PROJECT})..."
gcloud compute scp "$TAR" "${VM}:~/spotballer-deploy.tgz" --zone="$ZONE" --project="$PROJECT"

echo "==> Extracting and running setup_gcp_dev_vm.sh on VM..."
gcloud compute ssh "$VM" --zone="$ZONE" --project="$PROJECT" --command '
set -euo pipefail
mkdir -p "$HOME/SpotBaller"
cd "$HOME/SpotBaller"
tar xzf "$HOME/spotballer-deploy.tgz"
rm -f "$HOME/spotballer-deploy.tgz"
export SPOTBALLER_HOME="$HOME/SpotBaller"
bash infra/gcp/setup_gcp_dev_vm.sh
'

echo "==> Done. SSH: gcloud compute ssh ${VM} --zone=${ZONE} --project=${PROJECT}"
