MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolLM2-135M-Instruct}"
MODEL_LOCAL_PATH="${MODEL_LOCAL_PATH:-${MODEL_NAME}}"
GPU_NUM="${GPU_NUM:-8}"
BASE_PORT="${BASE_PORT:-6000}"
LOG_TAG="${LOG_TAG:-smollm2}"
TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../tools" && pwd)"
export PYTHONPATH="${TOOLS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p ./logs

# Start vLLM server
for i in $(seq 0 $((GPU_NUM-1))); do
    cmd="CUDA_VISIBLE_DEVICES=${i} nohup python3 ./tools/vllm_serve_compat.py ${MODEL_LOCAL_PATH} --port $((BASE_PORT + i)) 2>&1 > ./logs/${LOG_TAG}_${i}.log &"
    echo $cmd
    eval $cmd
done
