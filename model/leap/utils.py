"""LEAP decoding utilities.

Pipeline for one example (mirrors the LEAP paper):

  1. **Prefill-based initialization** – a single full forward over the prompt yields
     per-layer redundancy signals (relative F-norm ``R_l`` and expected-accept-length gain
     ``ΔEAR_l``). Layers are partitioned into zones (1=protect/EXECUTE, 2=EXECUTE|SKIP,
     3=EXECUTE|REPEAT) and merged into groups. Computed once and cached on the model.
  2. **MCTS-guided acceleration** – for every instance, MCTS explores group-level actions
     and scores each candidate draft configuration with a *real* speculative step
     (draft + verify) on cloned KV caches, using a speedup-based reward. The best-scoring
     ``layer_config`` is fixed for the rest of decoding.
  3. **Speculative decoding** – standard tree verification (reused verbatim from SWIFT,
     which is draft-agnostic); the draft is produced by the LEAP layer configuration.

Only the *drafting configuration* differs from SWIFT; the tree buffers, candidate
generation, verification and cache bookkeeping are shared.
"""
import time
import logging

import numpy as np
import torch

# Reuse SWIFT's (draft-agnostic) speculative-decoding machinery.
from model.swift.utils import (
    set_logger,
    prepare_logits_processor,
    get_choices_list,
    generate_swift_buffers,
    generate_candidates,
    tree_decoding,
    evaluate_posterior,
    update_inference_inputs,
    reset_swift_mode,
    swift_verify,
    swift_draft,
)
from .kv_cache import initialize_past_key_values, clone_past_key_values
from .mcts import MCTS, MCTSConfig, RewardConfig
from .layer_contribution import LayerContributionAnalyzer, build_zone_structure, ZoneConfig
from .modeling_llama import LayerAction


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: prefill-based zone / group initialization (cached on the model)
# ──────────────────────────────────────────────────────────────────────────────
def leap_zone_initialization(model, input_ids, zone_config=None, num_positions=8):
    """Estimate layer redundancy on the current prompt and build LEAP groups.

    Cached on ``model._leap_groups`` so it only runs on the first example.
    """
    if getattr(model, "_leap_groups", None) is not None:
        return model._leap_groups

    zone_config = zone_config or ZoneConfig()
    analyzer = LayerContributionAnalyzer(model, num_positions=num_positions)
    analyzer.register_hooks()
    try:
        rel_f_norms, _, delta_ear = analyzer.analyze_sample({"input_ids": input_ids})
    finally:
        analyzer.remove_hooks()

    rel = rel_f_norms.cpu().numpy()
    dear = delta_ear.cpu().numpy()
    _, _, groups = build_zone_structure(rel, dear, zone_config, use_smart_grouping=True)

    model._leap_groups = groups
    model._leap_num_layers = model.config.num_hidden_layers
    logging.info(
        "LEAP zones: %d groups | zone hist %s",
        len(groups),
        {z: sum(g["zone"] == z for g in groups) for z in (1, 2, 3)},
    )
    return groups


# ──────────────────────────────────────────────────────────────────────────────
# One speculative step on cloned caches -> (accept_length, draft_time, verify_time)
# Used both by MCTS scoring and (with commit=True) by the real decode loop.
# ──────────────────────────────────────────────────────────────────────────────
def _one_spec_step(model, input_ids, sample_token, top1_prob, swift_logits,
                   past_key_values, past_key_values_data, current_length_data,
                   max_new_tokens, new_token_num, logits_processor):
    """Run a single draft(already done)->verify->update cycle. Returns timing + accept."""
    device = model.model.layers[-1].self_attn.q_proj.weight.device
    swift_choices = eval(f"{get_choices_list(top1_prob, logits_processor=logits_processor)}")
    swift_buffers = generate_swift_buffers(swift_choices, device=device)
    model.swift_buffers = swift_buffers
    model.swift_choices = swift_choices
    model.model.swift_mask = swift_buffers["swift_attn_mask"]

    candidates, cart_candidates_prob, tree_candidates = generate_candidates(
        swift_logits, swift_buffers["tree_indices"], swift_buffers["retrieve_indices"],
        sample_token, logits_processor,
    )

    torch.cuda.synchronize()
    t0 = time.time()
    logits, outputs = tree_decoding(
        model, tree_candidates, past_key_values,
        swift_buffers["swift_position_ids"], input_ids, swift_buffers["retrieve_indices"],
    )
    torch.cuda.synchronize()
    verify_time = time.time() - t0

    best_candidate, accept_length, sample_p = evaluate_posterior(
        logits, candidates, logits_processor, cart_candidates_prob, swift_logits[2],
        swift_buffers["p_indices"], tree_candidates, swift_buffers["b_indices"],
    )

    input_ids, new_token_num, sample_token = update_inference_inputs(
        input_ids, candidates, best_candidate, accept_length,
        swift_buffers["retrieve_indices"], logits_processor, new_token_num,
        past_key_values_data, current_length_data, sample_p,
    )
    return input_ids, new_token_num, sample_token, int(accept_length), verify_time


def _draft(model, sample_token, new_token_num, past_key_values_data, current_length_data,
           max_new_tokens, logits_processor):
    torch.cuda.synchronize()
    t0 = time.time()
    swift_logits, top1_prob = swift_draft(
        model, input_ids=sample_token, new_token_num=new_token_num,
        past_key_values_data=past_key_values_data, current_length_data=current_length_data,
        max_new_tokens=max_new_tokens, logits_processor=logits_processor,
    )
    torch.cuda.synchronize()
    return swift_logits, top1_prob, time.time() - t0


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: per-instance MCTS search for the best layer configuration
# ──────────────────────────────────────────────────────────────────────────────
def leap_mcts_search(model, input_ids, sample_token, past_key_values_data, current_length_data,
                     max_new_tokens, logits_processor, groups, baseline_time_per_token,
                     mcts_config=None, reward_config=None):
    """Explore layer configs with MCTS; each candidate is scored by a real draft+verify
    cycle on *cloned* caches (never mutating the committed state). Returns best config."""
    num_layers = model.config.num_hidden_layers
    mcts_config = mcts_config or MCTSConfig()
    reward_config = reward_config or RewardConfig()
    reward_config.baseline_time_per_token = baseline_time_per_token
    mcts = MCTS(groups, num_layers, mcts_config, reward_config)

    input_len = input_ids.shape[1]
    for it in range(mcts_config.max_iterations):
        node, config = mcts.run_one_iteration()
        model.set_layer_config(config)

        # Clone caches so scoring never touches the committed decoding state.
        cur_data = [d.clone() for d in past_key_values_data]
        cur_len = current_length_data.clone()
        clone_pkv = clone_past_key_values(model, cur_data, cur_len)
        model.past_key_values = clone_pkv

        try:
            swift_logits, top1_prob, draft_time = _draft(
                model, sample_token, 0, cur_data, cur_len, max_new_tokens, logits_processor)
            _, _, _, accept_length, verify_time = _one_spec_step(
                model, input_ids.clone(), sample_token, top1_prob, swift_logits,
                clone_pkv, cur_data, cur_len, max_new_tokens, 0, logits_processor)
            reward = mcts.compute_reward(accept_length, draft_time, verify_time, config)
        except Exception as e:  # a pathological config -> strong negative signal
            logging.info("LEAP mcts eval failed (%s); penalizing config", repr(e)[:80])
            reward = -5.0
        mcts.update_with_reward(node, reward, config)

        if it >= mcts_config.min_iterations_before_stop and (mcts.check_convergence() or mcts.check_early_stop()):
            break

    best = mcts.get_best_config()
    stats = mcts.get_statistics()
    logging.info("LEAP MCTS: iters=%d best_reward=%.3f layer_ratio=%.3f",
                 stats["iterations"], stats["best_reward"], stats["layer_ratio"])
    model.model.swift_mask = None  # reset any tree mask left from scoring
    return best


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3: full LEAP generation for one example (drop-in forward_func for run_eval)
# ──────────────────────────────────────────────────────────────────────────────
def leap_forward(input_ids, model, tokenizer, max_new_tokens, statistics=None,
                 logits_processor=None, max_steps=512):
    assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
    statistics = statistics or {}
    input_ids = input_ids.clone()
    accept_length_list = []

    # Prefill-based zone/group initialization (once per model).
    groups = leap_zone_initialization(model, input_ids,
                                      zone_config=statistics.get("zone_config"),
                                      num_positions=statistics.get("num_positions", 8))

    # Initialize KV cache + prefill.
    past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(model.model)
    model.past_key_values = past_key_values
    model.past_key_values_data = past_key_values_data
    model.current_length_data = current_length_data

    input_len = input_ids.shape[1]
    cur_length = input_len
    reset_swift_mode(model)
    model.set_layer_config({})  # full model for the prefill

    outputs, logits = swift_verify(model, input_ids, past_key_values=past_key_values)

    if logits_processor is not None:
        last = logits_processor(None, logits[:, -1])
        probs = torch.nn.functional.softmax(last, dim=1)
        sample_token = torch.multinomial(probs, 1)
    else:
        sample_token = torch.argmax(logits[:, -1])[None, None]

    # Snapshot the prefilled cache for MCTS scoring (kept immutable during search).
    input_past_key_values_data = [d.clone() for d in past_key_values_data]
    input_current_length_data = current_length_data.clone()

    # Baseline = true single-token full-model decode time (on a clone so state is intact).
    _bl_data = [d.clone() for d in past_key_values_data]
    _bl_len = current_length_data.clone()
    _bl_pkv = clone_past_key_values(model, _bl_data, _bl_len)
    model.set_layer_config({})
    torch.cuda.synchronize()
    t0 = time.time()
    swift_verify(model, sample_token, past_key_values=_bl_pkv)
    torch.cuda.synchronize()
    baseline_time_per_token = max(time.time() - t0, 1e-4)

    # MCTS per-instance search -> fixed layer config for the rest of decoding.
    if statistics.get("optimization", True):
        best_config = leap_mcts_search(
            model, input_ids, sample_token,
            input_past_key_values_data, input_current_length_data,
            max_new_tokens, logits_processor, groups, baseline_time_per_token,
            mcts_config=statistics.get("mcts_config"),
            reward_config=statistics.get("reward_config"),
        )
    else:
        best_config = {}
    model.set_layer_config(best_config)
    statistics["best_config"] = best_config

    # First draft under the fixed config.
    model.past_key_values = past_key_values
    swift_logits, top1_prob = swift_draft(
        model, input_ids=sample_token, new_token_num=0,
        past_key_values_data=past_key_values_data, current_length_data=current_length_data,
        max_new_tokens=max_new_tokens, logits_processor=logits_processor,
    )

    new_token_num = 0
    draft_token_num = 0
    total_acc_num = 0
    idx = 0
    for idx in range(max_steps):
        draft_token_num += len(top1_prob)
        input_ids, new_token_num, sample_token, accept_length, _ = _one_spec_step(
            model, input_ids, sample_token, top1_prob, swift_logits,
            past_key_values, past_key_values_data, current_length_data,
            max_new_tokens, new_token_num, logits_processor)

        swift_logits, top1_prob = swift_draft(
            model, input_ids=sample_token, new_token_num=new_token_num,
            past_key_values_data=past_key_values_data, current_length_data=current_length_data,
            max_new_tokens=max_new_tokens, logits_processor=logits_processor,
        )
        accept_length_tree = input_ids.shape[1] - cur_length
        cur_length = accept_length_tree + cur_length
        accept_length_list.append(accept_length_tree)
        total_acc_num += accept_length_tree - 1
        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break
        if new_token_num > max_new_tokens:
            break

    if draft_token_num > 0:
        logging.info("token acceptance rate: {}".format(total_acc_num / draft_token_num))
    return input_ids, new_token_num, idx + 1, accept_length_list, draft_token_num
