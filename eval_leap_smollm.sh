#!/usr/bin/env bash
# LEAP (zone-aware MCTS self-speculative decoding) for SmolLM2-135M.
# Same evaluation pipeline as eval_smollm.sh (eval.py / speed.py), LEAP method.
set -euo pipefail

source /home/hyang/anaconda3/etc/profile.d/conda.sh
conda activate swift

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

MODEL_PATH="HuggingFaceTB/SmolLM2-135M"
MODEL_ID="smollm2-135m"

dedupe_jsonl() {
    local file="$1"
    local max_n="${2:-100}"
    python3 - "$file" "$max_n" <<'PY'
import json, sys
path, max_n = sys.argv[1], int(sys.argv[2])
seen = set()
kept = []
for line in open(path):
    obj = json.loads(line)
    if "choices" not in obj:
        continue
    fp = (obj["choices"][0]["turns"], tuple(obj["choices"][0]["new_tokens"]))
    if fp in seen:
        continue
    seen.add(fp)
    kept.append(line.rstrip("\n"))
    if len(kept) >= max_n:
        break
with open(path, "w") as f:
    for row in kept:
        f.write(row + "\n")
print(f"deduped {path}: {len(kept)} samples")
PY
}

# echo "=== SmolLM2-135M baseline (${DATA_NUM} examples) ==="
# CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_baseline \
#     --model-path  ${MODEL_PATH} \
#     --model-id    ${MODEL_ID} \
#     --max-new-tokens ${MAX_NEW_TOKENS} \
#     --task-name   ${TASK_NAME} \
#     --data-num    ${DATA_NUM} \
#     --temperature ${TEMP} \
#     --top-p       ${TOP_P} \
#     --seed        ${SEED} \
#     --dtype       ${torch_dtype}

echo "=== SmolLM2-135M LEAP (${DATA_NUM} examples) ==="
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

VANILLA_NAME="${MODEL_ID}-vanilla-${torch_dtype}-temp-${TEMP}-top-p-${TOP_P}-seed-${SEED}-max_new_tokens-${MAX_NEW_TOKENS}"
LEAP_NAME="${MODEL_ID}-leap-${torch_dtype}-temp-${TEMP}-top-p-${TOP_P}-seed-${SEED}-max_new_tokens-${MAX_NEW_TOKENS}-mcts_iters-${MCTS_ITERS}-layer_ratio-${TARGET_LAYER_RATIO}"

BASELINE_FILE="test/${TASK_NAME}/${TASK_NAME}_${DATA_NUM}/model_answer/${MODEL_ID}/${VANILLA_NAME}.jsonl"
LEAP_FILE="outputs/${TASK_NAME}/${TASK_NAME}_${DATA_NUM}/model_answer/${MODEL_ID}/${LEAP_NAME}.jsonl"

dedupe_jsonl "${BASELINE_FILE}" "${DATA_NUM}"

echo "=== Speed report (LEAP vs vanilla) ==="
python evaluation_llama/speed.py \
    --file-path  "${LEAP_FILE}" \
    --base-path  "${BASELINE_FILE}" \
    || echo "WARNING: speed report failed"

echo "SmolLM2 LEAP eval done."
