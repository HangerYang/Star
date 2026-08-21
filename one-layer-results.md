# One-layer drafts — 8 benchmarks, GPU 0 sequential

Generated 2026-08-21T10:00:40-07:00.

Every 1-layer draft plus the non-speculative target, over 8 benchmarks.
One vLLM job at a time on GPU 0 with nothing else on the box, so tok/s is
comparable across every cell.

| | |
|---|---|
| population | N=80 prompts per benchmark |
| prompt | long prompt (`answer_then_describe`): question unchanged, model answers then justifies |
| temperature | 0 and 1 |
| arms | `no_eagle_baseline` = target-only; every other row = with-draft (EAGLE-3, K=4) |
| decode | `output_len=1024`, `max_num_seqs=1`, `enforce_eager`, gpu-mem 0.8, tp 1 |
| target | HuggingFaceTB/SmolVLM-256M-Instruct |
| drafts | `checkpoint-66466` of each run (1 draft layer) |

Produced by `scripts/speculative/smolvlm/run_atd_acceptance.sh`:

```bash
DATASETS_ATD="Lin-Chen/MMStar MMMU/MMMU opendatalab/OmniDocBench HuggingFaceH4/MATH-500 \
  lmms-lab/textvqa lmms-lab/chartqa ai4math/mathvista lmms-lab/COCO-Caption" \
  TEMP=0 bash scripts/speculative/smolvlm/run_atd_acceptance.sh   # then TEMP=1
python my_angel/make_one_layer_results.py
```

**Engine / how this differs from the default eval.** Same harness every arm uses:
`run_atd_acceptance.sh` -> `scripts/speculative/smolvlm/eval_acceptance_suite_dp.sh`
-> `tools/vllm_offline_eagle3_vlm_batch.py` -> vLLM V1 offline `LLM(...)` from
`third_party/vllm` (v0.25.0) with `speculative_config={method: eagle3, ...}` and
`enforce_eager`. The wrapper overrides exactly three things, none of them the engine:

| | default suite | this sweep |
|---|---|---|
| prompt style | `raw` (`eval_acceptance_suite_dp.sh:114`) | `answer_then_describe` |
| benchmarks | 10 others: ChartQA, VQAv2, GQA, ScienceQA, MME, SEED-Bench, MMVet, MMBench, … (`:38-48`) | the 8 below |
| GPUs | round-robins one job per GPU | pinned `CUDA_VISIBLE_DEVICES=0`, serial, so tok/s is comparable |

`raw` left MMStar / textvqa / mathvista at 9-18 output tokens — too short for
speculation to pay — which is why this sweep, not the default one, is the record.

| run | what it is |
|---|---|
| `no_eagle_baseline` | target only, no speculative decoding |
| `baseline_1layer` | stock EAGLE-3: 3 aux layers, 3H→H fusion FC, 2H layer 0 |
| `banded_mix_fc_3.1` | 9 aux layers → 3 learned band mixes → 3H→H FC, EAGLE 3.1 |
| `banded_mix_wide_3.1` | same 3 band mixes, **no** FC: 4H layer 0 `[emb｜band0｜band1｜seed]` |
| `branch_distill_top1_w01` | `banded_mix_fc_3.1` + branch-aware distillation (draft top-1 vs teacher top-3, w=0.1, 1 step) |
| `vistoken_k1` | `banded_mix_fc_3.1` + learned-query row compression: each tile's 64 visual rows → k=1 summary |

## Prompt

Every benchmark uses the long prompt: the model answers first and then justifies,
so outputs are long enough for speculative decoding to pay. The question text
itself is unchanged. How each benchmark is treated:

| benchmark | treatment |
|---|---|
| MMStar | wrapper |
| MMMU | wrapper |
| OmniDocBench | prompt replaced (OCR) |
| MATH-500 | **unmodified** — reruns the raw prompt |
| textvqa | wrapper |
| chartqa | wrapper |
| mathvista | wrapper |
| COCO-Caption | prompt replaced (caption) |

## Results — temp 0

### Acceptance length

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_1layer` | 2.373 | 2.484 | 2.594 | 3.070 | 2.515 | 2.043 | 2.531 | 2.454 | **2.508** |
| `banded_mix_fc_3.1` | 2.626 | 2.656 | 2.809 | 3.532 | 2.700 | 2.214 | 2.576 | 2.535 | **2.706** |
| `banded_mix_wide_3.1` | 2.525 | 2.637 | 2.813 | 3.379 | 2.752 | 2.176 | 2.742 | 2.536 | **2.695** |
| `branch_distill_top1_w01` | — | — | — | 3.488 | — | — | — | — | **3.488** |
| `vistoken_k1` | — | — | — | — | — | — | — | — | — |

### Throughput (tok/s)

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `no_eagle_baseline` | 180.6 | 199.4 | 207.4 | 207.9 | 186.2 | 185.5 | 180.3 | 205.7 | **194.1** |
| `baseline_1layer` | 254.3 | 324.5 | 361.2 | 424.9 | 292.0 | 246.3 | 283.9 | 334.1 | **315.2** |
| `banded_mix_fc_3.1` | 279.5 | 332.4 | 370.9 | 475.2 | 295.3 | 257.4 | 268.5 | 335.2 | **326.8** |
| `banded_mix_wide_3.1` | 259.5 | 324.9 | 367.4 | 449.1 | 302.4 | 252.1 | 282.3 | 330.2 | **321.0** |
| `branch_distill_top1_w01` | — | — | — | 470.7 | — | — | — | — | **470.7** |
| `vistoken_k1` | — | — | — | — | — | — | — | — | — |

### Speedup vs `no_eagle_baseline`

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_1layer` | 1.408x | 1.628x | 1.742x | 2.044x | 1.568x | 1.328x | 1.575x | 1.624x | **1.615x** |
| `banded_mix_fc_3.1` | 1.548x | 1.667x | 1.789x | 2.286x | 1.586x | 1.388x | 1.489x | 1.629x | **1.673x** |
| `banded_mix_wide_3.1` | 1.437x | 1.630x | 1.772x | 2.160x | 1.624x | 1.359x | 1.566x | 1.605x | **1.644x** |
| `branch_distill_top1_w01` | — | — | — | 2.264x | — | — | — | — | **2.264x** |
| `vistoken_k1` | — | — | — | — | — | — | — | — | — |

> **15 of 48 cells missing** (sweep in flight or failed): `branch_distill_top1_w01 MMStar`, `branch_distill_top1_w01 MMMU`, `branch_distill_top1_w01 OmniDocBench`, `branch_distill_top1_w01 textvqa`, `branch_distill_top1_w01 chartqa`, `branch_distill_top1_w01 mathvista`, `branch_distill_top1_w01 COCO-Caption`, `vistoken_k1 MMStar`, `vistoken_k1 MMMU`, `vistoken_k1 OmniDocBench`, `vistoken_k1 MATH-500`, `vistoken_k1 textvqa` ….

## Results — temp 1

### Acceptance length

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_1layer` | 1.478 | 1.335 | 1.302 | 1.490 | 1.379 | 1.245 | 1.496 | 1.424 | **1.394** |
| `banded_mix_fc_3.1` | 1.553 | 1.368 | 1.296 | 1.630 | 1.430 | 1.245 | 1.481 | 1.433 | **1.429** |
| `banded_mix_wide_3.1` | 1.554 | 1.356 | 1.308 | 1.594 | 1.390 | 1.240 | 1.562 | 1.429 | **1.429** |
| `branch_distill_top1_w01` | — | — | 1.355 | 1.579 | — | — | — | — | **1.467** |
| `vistoken_k1` | — | — | — | — | — | — | — | — | — |

### Throughput (tok/s)

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `no_eagle_baseline` | 171.5 | 197.3 | 200.8 | 208.2 | 195.0 | 189.5 | 172.4 | 205.1 | **192.5** |
| `baseline_1layer` | 163.7 | 178.7 | 176.8 | 209.5 | 174.8 | 160.0 | 179.6 | 196.3 | **179.9** |
| `banded_mix_fc_3.1` | 156.1 | 175.3 | 174.5 | 221.5 | 178.4 | 156.3 | 167.9 | 195.9 | **178.2** |
| `banded_mix_wide_3.1` | 156.6 | 173.8 | 170.5 | 210.4 | 167.2 | 155.2 | 178.3 | 192.1 | **175.5** |
| `branch_distill_top1_w01` | — | — | 179.6 | 216.8 | — | — | — | — | **198.2** |
| `vistoken_k1` | — | — | — | — | — | — | — | — | — |

### Speedup vs `no_eagle_baseline`

| run | MMStar | MMMU | OmniDocBench | MATH-500 | textvqa | chartqa | mathvista | COCO-Caption | mean |
|---|---|---|---|---|---|---|---|---|---|
| `baseline_1layer` | 0.954x | 0.906x | 0.880x | 1.007x | 0.896x | 0.844x | 1.042x | 0.957x | **0.936x** |
| `banded_mix_fc_3.1` | 0.910x | 0.888x | 0.869x | 1.064x | 0.915x | 0.825x | 0.974x | 0.955x | **0.925x** |
| `banded_mix_wide_3.1` | 0.913x | 0.881x | 0.849x | 1.011x | 0.858x | 0.819x | 1.034x | 0.937x | **0.913x** |
| `branch_distill_top1_w01` | — | — | 0.894x | 1.041x | — | — | — | — | **0.968x** |
| `vistoken_k1` | — | — | — | — | — | — | — | — | — |

> **14 of 48 cells missing** (sweep in flight or failed): `branch_distill_top1_w01 MMStar`, `branch_distill_top1_w01 MMMU`, `branch_distill_top1_w01 textvqa`, `branch_distill_top1_w01 chartqa`, `branch_distill_top1_w01 mathvista`, `branch_distill_top1_w01 COCO-Caption`, `vistoken_k1 MMStar`, `vistoken_k1 MMMU`, `vistoken_k1 OmniDocBench`, `vistoken_k1 MATH-500`, `vistoken_k1 textvqa`, `vistoken_k1 chartqa` ….

## Prompt / output sizes

Target-only arm (`no_eagle_baseline`), long prompt, N=80. The with-draft arms
differ slightly on long outputs because the speculative path is not bit-exact
here; do not mix the two.

| dataset | N | temp | avg input tok | avg output tok |
|---|---:|---:|---:|---:|
| MMStar | 80 | 0 | 972.0 | 87.0 |
| MMMU | 80 | 0 | 1033.9 | 297.7 |
| OmniDocBench | 80 | 0 | 903.0 | 592.0 |
| MATH-500 | 80 | 0 | 71.9 | 566.6 |
| textvqa | 80 | 0 | 975.6 | 114.1 |
| chartqa | 80 | 0 | 998.2 | 124.5 |
| mathvista | 80 | 0 | 995.1 | 90.3 |
| COCO-Caption | 80 | 0 | 902.4 | 605.2 |
| MMStar | 80 | 1 | 972.0 | 65.9 |
| MMMU | 80 | 1 | 1033.9 | 171.7 |
| OmniDocBench | 80 | 1 | 903.0 | 281.6 |
| MATH-500 | 80 | 1 | 71.9 | 244.6 |
| textvqa | 80 | 1 | 975.6 | 158.8 |
| chartqa | 80 | 1 | 998.2 | 124.8 |
| mathvista | 80 | 1 | 995.1 | 74.2 |
| COCO-Caption | 80 | 1 | 902.4 | 590.4 |

### Why the long prompt (target-only, temp 0)

The short prompt this sweep replaced left several benchmarks at 9-45 output
tokens — too short for speculation to pay. Superseded numbers, kept as the
reason for the change:

| benchmark | short-prompt out tok | long-prompt out tok | ratio |
|---|---:|---:|---:|
| MMStar | 8.8 | 87.0 | **9.85x** |
| MMMU | 45.0 | 297.7 | **6.61x** |
| OmniDocBench | 416.3 | 592.0 | **1.42x** |
| MATH-500 | 566.6 | 566.6 | **1.00x** |
| textvqa | 17.8 | 114.1 | **6.40x** |
| chartqa | 88.4 | 124.5 | **1.41x** |
| mathvista | 11.4 | 90.3 | **7.92x** |
| COCO-Caption | 484.9 | 605.2 | **1.25x** |
