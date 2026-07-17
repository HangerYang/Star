DATA_FORMAT="${DATA_FORMAT:-converted}"
DATA_SHARD_SIZE="${DATA_SHARD_SIZE:-50000}"
BASE_PORT="${BASE_PORT:-6000}"
MAX_CLIENTS="${MAX_CLIENTS:-8}"
NUM_THREADS="${NUM_THREADS:-512}"
MODEL_TAG="${MODEL_TAG:-smollm2}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/hyang/Star/AngelSlim/dataset}"
SHAREGPT_INPUT="${SHAREGPT_INPUT:-/home/hyang/Star/processed_data/sharegpt_converted.jsonl}"
ULTRACHAT_INPUT="${ULTRACHAT_INPUT:-/home/hyang/Star/processed_data/ultrachat_converted.jsonl}"

# ShareGPT (already converted)
python3 ./tools/generate_data_for_target_model.py \
    --data_name_or_path ${SHAREGPT_INPUT} \
    --output_dir ${OUTPUT_ROOT}/sharegpt_${MODEL_TAG} \
    --data_format ${DATA_FORMAT} \
    --data_shard_size ${DATA_SHARD_SIZE} \
    --base_port ${BASE_PORT} \
    --max_clients ${MAX_CLIENTS} \
    --num_threads ${NUM_THREADS}

# UltraChat (already converted)
python3 ./tools/generate_data_for_target_model.py \
    --data_name_or_path ${ULTRACHAT_INPUT} \
    --output_dir ${OUTPUT_ROOT}/ultrachat_${MODEL_TAG} \
    --data_format ${DATA_FORMAT} \
    --data_shard_size ${DATA_SHARD_SIZE} \
    --base_port ${BASE_PORT} \
    --max_clients ${MAX_CLIENTS} \
    --num_threads ${NUM_THREADS}
