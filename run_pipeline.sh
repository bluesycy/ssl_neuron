#!/usr/bin/env bash
# End-to-end pipeline for the three compartment variants.
# Run from the repository root with the `ssl` conda env active.
#
#   ./run_pipeline.sh preprocess   # SWC -> masked/subsampled SWC -> dataset
#   ./run_pipeline.sh train        # one GraphDINO run per variant, one per GPU
#   ./run_pipeline.sh latents      # export latents for every variant
#   ./run_pipeline.sh all
set -euo pipefail

PY=${PY:-/home/chuyu/miniforge3/envs/ssl/bin/python}
VARIANTS=(full dendrite axon)
LOG_DIR=ssl_neuron/logs
mkdir -p "$LOG_DIR"

preprocess() {
    $PY -m ssl_neuron.preprocessing.subsample --variant all --jobs 16
    $PY -m ssl_neuron.preprocessing.build_dataset --variant all
}

train() {
    local gpu=0
    for v in "${VARIANTS[@]}"; do
        echo "launching $v on GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u -m ssl_neuron.main --variant "$v" \
            > "$LOG_DIR/train_${v}.log" 2>&1 &
        gpu=$((gpu + 1))
    done
    wait
}

latents() {
    $PY -m ssl_neuron.demos.save_latents --variant all
}

case "${1:-all}" in
    preprocess) preprocess ;;
    train)      train ;;
    latents)    latents ;;
    all)        preprocess; train; latents ;;
    *) echo "usage: $0 {preprocess|train|latents|all}" >&2; exit 1 ;;
esac
