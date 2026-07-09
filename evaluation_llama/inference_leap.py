"""LEAP: zone-aware MCTS self-speculative decoding.

Drop-in replacement for `inference_swift.py` that uses the LEAP method (whole-layer
EXECUTE / SKIP / REPEAT actions chosen per-instance by zone-aware MCTS) instead of
SWIFT's attention/MLP skip-set Bayesian search. Uses the same evaluation pipeline
(`run_eval`) and output format, so `evaluation_llama/speed.py` works unchanged.

Usage:
python -m evaluation_llama.inference_leap --model-path <hf_model> --model-id <id> \
    --task-name cnndm --data-num 100 --optimization
"""
import argparse

from fastchat.utils import str_to_torch_dtype
from transformers import AutoTokenizer

from evaluation_llama.eval import run_eval
from model.leap.utils import leap_forward, set_logger, prepare_logits_processor
from model.leap.mcts import MCTSConfig, RewardConfig
from model.leap.layer_contribution import ZoneConfig
from model.leap.modeling_llama import LlamaForCausalLM

import numpy as np
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-gpus-per-model", type=int, default=1)
    parser.add_argument("--num-gpus-total", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.85)
    parser.add_argument("--dtype", type=str, default="float16",
                        choices=["float32", "float64", "float16", "bfloat16"])
    parser.add_argument("--task-name", type=str, required=True)
    parser.add_argument("--data-num", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--optimization", action="store_true", default=False,
                        help="Run per-instance MCTS search (LEAP). If off, uses the full model.")
    # LEAP / MCTS hyper-parameters
    parser.add_argument("--mcts-iters", type=int, default=40, help="Max MCTS iterations per instance.")
    parser.add_argument("--min-layer-ratio", type=float, default=0.40)
    parser.add_argument("--max-layer-ratio", type=float, default=0.50)
    parser.add_argument("--target-layer-ratio", type=float, default=0.45)
    parser.add_argument("--exploration-weight", type=float, default=3.0)
    parser.add_argument("--num-positions", type=int, default=8,
                        help="#last positions used for redundancy estimation in zone init.")

    args = parser.parse_args()

    args.model_name = (args.model_id + "-leap-" + str(args.dtype) + "-temp-" + str(args.temperature)
                       + "-top-p-" + str(args.top_p) + "-seed-" + str(args.seed)
                       + "-max_new_tokens-" + str(args.max_new_tokens)
                       + "-mcts_iters-" + str(args.mcts_iters)
                       + "-layer_ratio-" + str(args.target_layer_ratio))
    answer_file = f"outputs/{args.task_name}/{args.task_name}_{args.data_num}/model_answer/{args.model_id}/{args.model_name}.jsonl"
    set_logger()
    print(f"Output to {answer_file}")

    torch.nn.Linear.reset_parameters = lambda x: None

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=str_to_torch_dtype(args.dtype),
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    if args.temperature > 1e-5:
        logits_processor = prepare_logits_processor(temperature=args.temperature, top_p=args.top_p)
    else:
        logits_processor = None

    mcts_config = MCTSConfig(
        exploration_weight=args.exploration_weight,
        min_layer_ratio=args.min_layer_ratio,
        max_layer_ratio=args.max_layer_ratio,
        target_layer_ratio=args.target_layer_ratio,
        max_iterations=args.mcts_iters,
    )
    reward_config = RewardConfig()
    zone_config = ZoneConfig()

    statistics = {
        "optimization": args.optimization,
        "mcts_config": mcts_config,
        "reward_config": reward_config,
        "zone_config": zone_config,
        "num_positions": args.num_positions,
    }

    run_eval(
        model=model,
        tokenizer=tokenizer,
        forward_func=leap_forward,
        model_id=args.model_id,
        answer_file=answer_file,
        max_new_tokens=args.max_new_tokens,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        task_name=args.task_name,
        data_num=args.data_num,
        seed=args.seed,
        statistics=statistics,
        logits_processor=logits_processor,
    )
