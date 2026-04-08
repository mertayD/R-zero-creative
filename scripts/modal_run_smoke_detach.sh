#!/usr/bin/env bash
# Run R-Zero smoke test on Modal in detached mode so the job keeps running if
# this terminal, SSH session, or VPN drops. Track progress in the Modal dashboard
# or: modal app logs <app-id>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec modal run --detach modal_run.py "$@"
