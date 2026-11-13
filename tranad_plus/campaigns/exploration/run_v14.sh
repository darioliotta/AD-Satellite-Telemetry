#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
python -u -m tranad_plus.campaigns.exploration.run --check "$@"
exec python -u -m tranad_plus.campaigns.exploration.run "$@"
