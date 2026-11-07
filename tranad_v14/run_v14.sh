#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -u -m tranad_v14.run --check "$@"
exec python -u -m tranad_v14.run "$@"
