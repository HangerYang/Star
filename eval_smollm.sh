#!/usr/bin/env bash
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

OPT_INTERVAL=1
BAYES_INTERVAL=25
MAX_OPT_ITER=1000
MAX_TOLERANCE_ITER=300
MAX_SCORE=0.93
CONTEXT_WINDOW=50
SKIP_RATIO=0.45

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

echo "=== SmolLM2-135M baseline (${DATA_NUM} examples) ==="
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

echo "=== SmolLM2-135M SWIFT (${DATA_NUM} examples) ==="
CUDA_VISIBLE_DEVICES=${GPU_DEVICES} python -m evaluation_llama.inference_swift \
    --model-path  ${MODEL_PATH} \
    --model-id    ${MODEL_ID} \
    --temperature ${TEMP} \
    --top-p       ${TOP_P} \
    --dtype       ${torch_dtype} \
    --task-name   ${TASK_NAME} \
    --data-num    ${DATA_NUM} \
    --max-new-tokens ${MAX_NEW_TOKENS} \
    --seed             ${SEED} \
    --context-window   ${CONTEXT_WINDOW} \
    --opt-interval     ${OPT_INTERVAL} \
    --bayes-interval   ${BAYES_INTERVAL} \
    --max-opt-iter     ${MAX_OPT_ITER} \
    --max-tolerance-iter ${MAX_TOLERANCE_ITER} \
    --max-score        ${MAX_SCORE} \
    --skip-ratio       ${SKIP_RATIO} \
    --optimization --bayes

VANILLA_NAME="${MODEL_ID}-vanilla-${torch_dtype}-temp-${TEMP}-top-p-${TOP_P}-seed-${SEED}-max_new_tokens-${MAX_NEW_TOKENS}"
SWIFT_NAME="${MODEL_ID}-swift-${torch_dtype}-temp-${TEMP}-top-p-${TOP_P}-seed-${SEED}-max_new_tokens-${MAX_NEW_TOKENS}-opt_interval-${OPT_INTERVAL}-bayes_interval-${BAYES_INTERVAL}-max_opt-${MAX_OPT_ITER}-max_tolerance-${MAX_TOLERANCE_ITER}-max_score-${MAX_SCORE}-context_window-${CONTEXT_WINDOW}-skip_ratio-${SKIP_RATIO}"

BASELINE_FILE="test/${TASK_NAME}/${TASK_NAME}_${DATA_NUM}/model_answer/${MODEL_ID}/${VANILLA_NAME}.jsonl"
SWIFT_FILE="outputs/${TASK_NAME}/${TASK_NAME}_${DATA_NUM}/model_answer/${MODEL_ID}/${SWIFT_NAME}.jsonl"

dedupe_jsonl "${BASELINE_FILE}" "${DATA_NUM}"

echo "=== Speed report ==="
python evaluation_llama/speed.py \
    --file-path  "${SWIFT_FILE}" \
    --base-path  "${BASELINE_FILE}"

echo "SmolLM eval done."
