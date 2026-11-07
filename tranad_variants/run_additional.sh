#!/usr/bin/env bash
# Activate the existing ad-satellite-telemetry environment before running this.
set -euo pipefail
cd "$(dirname "$0")"

python -u run_experiments.py --campaign additional --check "$@"
exec python -u run_experiments.py --campaign additional --stage both "$@"
