#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CONDA_HOME=${CONDA_HOME:-/home/hyang/anaconda3}
export ANGEL_ENV_NAME=${ANGEL_ENV_NAME:-angel}
export ANGEL_ENV_PREFIX=${ANGEL_ENV_PREFIX:-$CONDA_HOME/envs/$ANGEL_ENV_NAME}
export PYTHON_BIN=${PYTHON_BIN:-$ANGEL_ENV_PREFIX/bin/python}
export TORCHRUN_BIN=${TORCHRUN_BIN:-$ANGEL_ENV_PREFIX/bin/torchrun}

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python interpreter not found: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -x "$TORCHRUN_BIN" ]]; then
    echo "torchrun not found: $TORCHRUN_BIN" >&2
    exit 1
fi

export CONFIG_DIR=angelslim/compressor/speculative/train/configs
export TARGET_MODEL_NAME_OR_PATH=${TARGET_MODEL_NAME_OR_PATH:-HuggingFaceTB/SmolLM2-135M-Instruct}
export DRAFT_MODEL_CONFIG_PATH=${DRAFT_MODEL_CONFIG_PATH:-$CONFIG_DIR/smollm2-135m-instruct-eagle3-skip50.json}
export TRAIN_DATA_PATH=${TRAIN_DATA_PATH:-dataset/train.jsonl}
export EVAL_DATA_PATH=${EVAL_DATA_PATH:-dataset/val.jsonl}
export OUTPUT_DIR=${OUTPUT_DIR:-outputs/smollm2-135m-instruct-eagle3-online-skip50}
export MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
export CHAT_TEMPLATE_TYPE=${CHAT_TEMPLATE_TYPE:-smollm2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export DEEPSPEED_CONFIG_PATH=${DEEPSPEED_CONFIG_PATH:-$CONFIG_DIR/deepspeed_zero3_no_offload.json}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-60}
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export MASTER_PORT=${MASTER_PORT:-$((20000 + RANDOM % 10000))}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
export NPROC_PER_NODE=${NPROC_PER_NODE:-8}
mkdir -p "$TORCH_EXTENSIONS_DIR"

if [[ ! -d "$TARGET_MODEL_NAME_OR_PATH" ]]; then
    export TARGET_MODEL_NAME_OR_PATH=$(
        "$PYTHON_BIN" -c "import os; from huggingface_hub import snapshot_download; print(snapshot_download(repo_id=os.environ['TARGET_MODEL_NAME_OR_PATH'], resume_download=True))"
    )
fi

"$TORCHRUN_BIN" --master_port "$MASTER_PORT" --nproc_per_node="$NPROC_PER_NODE" tools/train_eagle3_online.py \
    --target_model_name_or_path "$TARGET_MODEL_NAME_OR_PATH" \
    --draft_model_config_path "$DRAFT_MODEL_CONFIG_PATH" \
    --train_data_path "$TRAIN_DATA_PATH" \
    --eval_data_path "$EVAL_DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 20 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --weight_decay 0.0 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "constant" \
    --logging_steps 20 \
    --model_max_length "$MODEL_MAX_LENGTH" \
    --deepspeed "$DEEPSPEED_CONFIG_PATH" \
    --chat_template_type "$CHAT_TEMPLATE_TYPE" \
    --report_to none \
    --run_name smollm2-135m-instruct-eagle3-skip50
