#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
python -u -B run_campaign.py mechanisms --check "$@"
exec python -u -B run_campaign.py mechanisms "$@"
