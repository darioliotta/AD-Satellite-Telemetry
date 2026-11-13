#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
python -u -B -m tranad_plus.campaigns.temporal.run --check "$@"
exec python -u -B -m tranad_plus.campaigns.temporal.run "$@"
