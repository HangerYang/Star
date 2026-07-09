#!/usr/bin/env bash
# LEAP: zone-aware MCTS self-speculative decoding.
# Same evaluation pipeline as SWIFT (eval.py / speed.py), different acceleration method.
set -euo pipefail

source /home/hyang/anaconda3/etc/profile.d/conda.sh
conda activate swift

# ─── Shared settings ──────────────────────────────────────────────────────────
TEMP=0.0
TOP_P=0.85
DATA_NUM=100
SEED=2024
GPU_DEVICES=0
MAX_NEW_TOKENS=512
torch_dtype="float16"
TASK_NAME="cnndm"

# ─── LEAP / MCTS hyper-parameters ──────────────────────────────────────────────
MCTS_ITERS=40
MIN_LAYER_RATIO=0.40
MAX_LAYER_RATIO=0.50
TARGET_LAYER_RATIO=0.45
EXPLORATION_WEIGHT=3.0
NUM_POSITIONS=8

MODEL_PATH="meta-llama/Llama-2-13b-hf"
MODEL_ID="llama-2-13b"

echo "=== ${MODEL_ID} baseline (vanilla AR) ==="
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_baseline \
    --model-path  ${MODEL_PATH} \
    --model-id    ${MODEL_ID} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --task-name   ${TASK_NAME} \
    --data-num    ${DATA_NUM} \
    --temperature ${TEMP} \
    --top-p       ${TOP_P} \
    --seed        ${SEED} \
    --dtype       ${torch_dtype}

echo "=== ${MODEL_ID} LEAP (zone-aware MCTS self-speculative decoding) ==="
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_leap \
    --model-path  ${MODEL_PATH} \
    --model-id    ${MODEL_ID} \
    --temperature ${TEMP} \
    --top-p       ${TOP_P} \
    --dtype       ${torch_dtype} \
    --task-name   ${TASK_NAME} \
    --data-num    ${DATA_NUM} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --seed        ${SEED} \
    --mcts-iters  ${MCTS_ITERS} \
    --min-layer-ratio ${MIN_LAYER_RATIO} \
    --max-layer-ratio ${MAX_LAYER_RATIO} \
    --target-layer-ratio ${TARGET_LAYER_RATIO} \
    --exploration-weight ${EXPLORATION_WEIGHT} \
    --num-positions ${NUM_POSITIONS} \
    --optimization

# ─── Derive output file paths (must match inference_*.py naming logic) ──────────
VANILLA_NAME="${MODEL_ID}-vanilla-${torch_dtype}-temp-${TEMP}-top-p-${TOP_P}-seed-${SEED}-max_new_tokens-${MAX_NEW_TOKENS}"
LEAP_NAME="${MODEL_ID}-leap-${torch_dtype}-temp-${TEMP}-top-p-${TOP_P}-seed-${SEED}-max_new_tokens-${MAX_NEW_TOKENS}-mcts_iters-${MCTS_ITERS}-layer_ratio-${TARGET_LAYER_RATIO}"

BASELINE_FILE="test/${TASK_NAME}/${TASK_NAME}_${DATA_NUM}/model_answer/${MODEL_ID}/${VANILLA_NAME}.jsonl"
LEAP_FILE="outputs/${TASK_NAME}/${TASK_NAME}_${DATA_NUM}/model_answer/${MODEL_ID}/${LEAP_NAME}.jsonl"

echo "=== Speed report (LEAP vs vanilla) ==="
python evaluation_llama/speed.py \
    --file-path  "${LEAP_FILE}" \
    --base-path  "${BASELINE_FILE}" \
    || echo "WARNING: speed report failed"

echo "LEAP eval done."
