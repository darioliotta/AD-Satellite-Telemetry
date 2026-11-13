#!/usr/bin/env bash
# Activate ad-satellite-telemetry first. Run from any directory.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
# This checks CUDA arithmetic/instructions, all four architectures and parents.
python -u -B run_campaign.py integer --check "$@"
exec python -u -B run_campaign.py integer "$@"
