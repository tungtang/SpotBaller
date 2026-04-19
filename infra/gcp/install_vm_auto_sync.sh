#!/usr/bin/env bash
# Install cron-based polling: every SPOTBALLER_VM_CRON_MINS minutes run vm_pull_latest.sh on this VM.
# Run ON THE VM after the repo is a git clone, e.g.:
#   bash ~/SpotBaller/infra/gcp/install_vm_auto_sync.sh
#
# Env:
#   SPOTBALLER_HOME          — default ~/SpotBaller
#   SPOTBALLER_VM_CRON_MINS  — default 5

set -euo pipefail

SPOTBALLER_HOME="${SPOTBALLER_HOME:-$HOME/SpotBaller}"
MINS="${SPOTBALLER_VM_CRON_MINS:-5}"
BRANCH="${SPOTBALLER_GIT_BRANCH:-main}"
PULL="$SPOTBALLER_HOME/infra/gcp/vm_pull_latest.sh"

if [[ ! -f "$PULL" ]]; then
  echo "Not found: $PULL"
  exit 1
fi
chmod +x "$PULL" 2>/dev/null || true

LINE="*/$MINS * * * * SPOTBALLER_HOME=$SPOTBALLER_HOME SPOTBALLER_GIT_BRANCH=$BRANCH /bin/bash $PULL"

TMP="$(mktemp)"
( crontab -l 2>/dev/null | grep -v 'infra/gcp/vm_pull_latest.sh' || true; echo "$LINE" ) >"$TMP"
crontab "$TMP"
rm -f "$TMP"

echo "Installed crontab entry (every $MINS min):"
echo "  $LINE"
echo "Logs: ${SPOTBALLER_VM_SYNC_LOG:-$SPOTBALLER_HOME/runtime/vm_sync.log}"
crontab -l
