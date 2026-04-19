#!/usr/bin/env bash
# From your laptop (with gcloud auth): run vm_pull_latest.sh on the VM once.
# Same defaults as deploy_to_vm.sh.
#
# Usage:
#   bash infra/gcp/trigger_vm_pull.sh
#   VM=my-vm ZONE=us-central1-a GCP_PROJECT=myproj bash infra/gcp/trigger_vm_pull.sh
#
# If IAP tunneling is not enabled for your project, remove --tunnel-through-iap below.

set -euo pipefail

VM="${VM:-spotballer-vm1}"
ZONE="${ZONE:-asia-southeast1-b}"
PROJECT="${GCP_PROJECT:-${PROJECT:-datacloudpoc}}"

gcloud compute ssh "$VM" \
  --zone="$ZONE" \
  --project="$PROJECT" \
  --tunnel-through-iap \
  --command='export SPOTBALLER_HOME="${SPOTBALLER_HOME:-$HOME/SpotBaller}"; bash "$SPOTBALLER_HOME/infra/gcp/vm_pull_latest.sh"'
