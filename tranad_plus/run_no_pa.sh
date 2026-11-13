#!/usr/bin/env bash
# Activate the existing ad-satellite-telemetry environment first.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
python -u -B run_campaign.py thresholds --check "$@"
exec python -u -B run_campaign.py thresholds "$@"
