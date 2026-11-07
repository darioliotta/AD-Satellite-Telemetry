#!/usr/bin/env bash
# Train the two original WADI replication profiles; activate the GPU venv first.
# Does not run evaluation or change the v12 campaign catalog.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
selection=all
selection_seen=0
dry_run=0
for argument in "$@"; do
    case "$argument" in
        --dry-run) dry_run=1 ;;
        ws20|ws50|all)
            if [[ "$selection_seen" == 1 ]]; then
                printf 'Choose only one of ws20, ws50 or all.\n' >&2
                exit 2
            fi
            selection="$argument"
            selection_seen=1
            ;;
        -h|--help)
            printf 'Usage: bash scripts/train_wadi_original.sh [ws20|ws50|all] [--dry-run]\n'
            printf 'Defaults to all. Use the active GPU Python environment.\n'
            printf 'Optional: WADI_PYTHON, WADI_DEVICE (default cuda:0), WADI_RESULTS_ROOT.\n'
            exit 0
            ;;
        *) printf 'Unknown argument: %s\n' "$argument" >&2; exit 2 ;;
    esac
done

python_bin="${WADI_PYTHON:-python}"
device="${WADI_DEVICE:-cuda:0}"

# Separate namespace: preserve legacy runs and the active v12 study.
export TRANADPLUS_OUTPUT_ROOT="${WADI_RESULTS_ROOT:-$project_dir/results/wadi_original}"
export TRANADPLUS_RESULTS_VERSION=v1
export TRANADPLUS_CAMPAIGN_SUBDIR=''
export TRANADPLUS_DATA_ROOT="${TRANADPLUS_DATA_ROOT:-$project_dir/data}"
export TRANADPLUS_NATIVE_CHECKPOINT_DIR="$TRANADPLUS_OUTPUT_ROOT/v1/WADI"
export TRANADPLUS_NATIVE_PICKLE_DIR="$TRANADPLUS_OUTPUT_ROOT/v1/WADI/_native_pickles"
export TRANADPLUS_CUDA_PRECISION=faithful
export TRANADPLUS_ADAM_IMPL=standard
# Leave GPU memory for the original batch sizes rather than caching the dataset.
export TRANADPLUS_GPU_CACHE=off
unset TRANADPLUS_CAMPAIGN_PLAN

cd -- "$project_dir/tranad_plus"
printf 'Checkpoint directory: %s\n' "$TRANADPLUS_NATIVE_CHECKPOINT_DIR"
printf 'Preprocessed data: %s\n' "$TRANADPLUS_DATA_ROOT/processed/tranad_plus/WADI"
if [[ "$dry_run" == 0 ]]; then
    "$python_bin" -c 'import sys, torch; device=torch.device(sys.argv[1]); assert device.type == "cuda" and torch.cuda.is_available(), "Run this training inside the CUDA environment on the GPU machine"; print("GPU:", torch.cuda.get_device_name(device), "| GiB:", round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 2))' "$device"
    shopt -s nullglob
    tracks=("$TRANADPLUS_DATA_ROOT"/processed/tranad_plus/WADI/Train/Tracks/*.npy)
    if [[ "${#tracks[@]}" == 0 ]]; then
        printf 'Missing WADI preprocessed training tracks.\n' >&2
        exit 1
    fi
    mkdir -p -- "$TRANADPLUS_NATIVE_CHECKPOINT_DIR"
fi

run_one() {
    local window="$1" batch="$2" lr="$3" identifier="$4"
    local command=("$python_bin" -u resumable_training.py
        --model_str TranAD --dataset_str WADI --identifier "$identifier"
        --device "$device" --scaling True
        --epochs 50 --batch_sz "$batch" --window_sz "$window" --lr "$lr"
        --dim_feedforward 2x --num_layers 1 --pot_q 0.001
        --shuffle 0 --padding 1 --num_workers 0 --checkpoint-every 1)
    printf '\n'
    printf '%q ' "${command[@]}"
    printf '\n'
    if [[ "$dry_run" == 0 ]]; then
        # A recovery checkpoint is resumed; an existing final checkpoint is
        # rejected by the runner, rather than silently overwritten.
        "${command[@]}" 2>&1 | tee -a "$TRANADPLUS_NATIVE_CHECKPOINT_DIR/${identifier}_train.log"
    fi
}

if [[ "$selection" == all || "$selection" == ws20 ]]; then
    run_one 20 8192 0.001 wadi_original_ws20_ff2x_bs8192
fi
if [[ "$selection" == all || "$selection" == ws50 ]]; then
    run_one 50 2048 0.0001 wadi_original_ws50_ff2x_bs2048
fi

if [[ "$dry_run" == 1 ]]; then
    printf '\nDry run only: no training or output files created.\n'
fi
