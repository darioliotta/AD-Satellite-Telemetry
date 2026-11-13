#!/usr/bin/env bash
# Activate ad-satellite-telemetry first. Re-running verifies completed jobs and resumes.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
python -u -B run_campaign.py regularization --check "$@"
exec python -u -B run_campaign.py regularization "$@"
