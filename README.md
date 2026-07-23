# Star

# Speculative Decoding

# Overview

# Standard Speculative Decoding: The Pipeline

**Setup:**
1. **Target prefill** — the target processes the full prompt, producing its KV-cache and the first token.

**Loop (repeats each round):**
2. **Draft prefill/context sync** — the drafter (a genuinely separate model — EAGLE's small head, Medusa's extra heads, MASSV's small sibling LM) builds up its own context. First round: full prefill of the prompt in its own cache. Later rounds: append newly accepted tokens.
3. **Draft** — the drafter proposes k tokens, either sequentially (EAGLE: one small transformer layer autoregressively predicting the next *feature*, then the target's frozen LM head turns it into a token) or in parallel (Medusa: multiple independent heads guessing several future positions at once).
4. **Verify** — the target runs one forward pass over all k drafted tokens (via tree or chain attention), producing its own distribution at each position.
5. **Accept/reject** — rejection sampling: accept the matching prefix, resample a correction token at the first divergence. Lossless.
6. **Cache update** — advance target cache (native) and drafter cache (its own, separate structure) to the accepted length; loop.

Two full sets of weights and two KV-caches exist throughout. The core efficiency lever is **τ (accepted length) vs. c (drafter cost relative to target)** — speedup ≈ τ/(1+k·c).

---

# Self-Speculative Decoding (SSD): The Pipeline

**Setup:**
1. **Target prefill** — same target, but only *one* set of weights exists. Prefill runs the full model once, populating one shared KV-cache for all layers 1…L.

**Loop:**
2. **Draft pass** — run only the first E layers (fixed as in LayerSkip/Kangaroo, or dynamically chosen per-input as in SWIFT/CLaSp/LEAP/CAS-Spec) and apply the model's own LM head (or a tiny adapter, as in Kangaroo) early. No separate prefill step is needed — the drafter's "context" *is* the target's own cache up to layer E, already populated.
3. **Draft** — generate k tokens using only layers 1…E, reusing the same cache the eventual verification will also use.
4. **Verify** — run the *remaining* layers E+1…L over the k drafted tokens, reusing the layers-1…E computation and cache already done in step 2 (not recomputed).
5. **Accept/reject** — same rejection sampling, lossless.
6. **Cache update** — one cache, already correctly positioned; loop, possibly re-selecting E (SWIFT/CLaSp/LEAP re-optimize which layers to skip at this point).

One set of weights, one KV-cache, zero redundant prefill — the structural difference from standard SD that matters most.

---

# Self-Drafting vs. EAGLE-based: Pros and Cons

| | **Self-Drafting (SSD family: LayerSkip, Kangaroo, SWIFT, CLaSp, CAS-Spec, LEAP)** | **EAGLE-based (attached feature-head family: EAGLE, EAGLE-2, DREAM, ViSpec, HiViS, MSD)** |
|---|---|---|
| **Model count / memory** | One model, one weight set. Kangaroo's adapter is the only exception (~67M extra) — negligible next to a full second model. | Two logical components: frozen target + trained draft head. EAGLE's head is tiny (~1 layer, <1B even for 70B targets), so memory overhead is still small, but it's a second artifact to store, version, and load. |
| **KV-cache handling** | **Free, automatic sharing** — layers 1…E computed once, reused by both draft and verify (LayerSkip, Kangaroo). No cache alignment problem exists by construction. | Needs the drafter to build its own cache. EAGLE feeds target features + shifted token embeddings directly (no separate prefill of raw tokens, but still a distinct cache structure); HiViS/ViSpec had to specifically engineer around this (hiding visual tokens, reusing target hidden states) to avoid a redundant/misaligned prefill. |
| **Setup cost** | SWIFT, CLaSp, CAS-Spec, LEAP: **zero training** — pure inference-time search/selection over which layers to skip. LayerSkip needs the target retrained with layer dropout + early-exit loss (heavier). Kangaroo needs only a tiny adapter trained. | Always requires training a dedicated head (few GPU-days on ShareGPT-scale data for EAGLE) — cheap relative to pretraining, but never zero, and needed per target model. |
| **Drafter specialization / acceptance ceiling** | Early/shallow layers are a **compromise representation** — never trained purely to predict the final output (except LayerSkip, which does train for this, at the cost of touching the target). Reported speedups cluster **1.3×–2.3×** (SWIFT ≤1.6×, CLaSp ≤1.67×, Kangaroo ≤2.04×, CAS-Spec ≤2.3× via cascading). | A dedicated head trained specifically to predict the target's next feature reaches **higher acceptance length** — EAGLE 2.7–3.5×, DREAM up to 3.6×, ViSpec up to 3.22×. Specialization buys real headroom the self-drafting family hasn't matched. |
| **Adaptivity to input** | SWIFT/CLaSp/LEAP explicitly re-optimize which layers to skip **per input or per step** (task-specific layer sparsity is their whole premise) — a flexibility EAGLE-style fixed heads don't have built in. | The head is fixed after training; it doesn't adapt its "depth" or which features it reads per input (DREAM's entropy-based intermediate-layer selection is the closest analog, but the head architecture itself is static). |
| **Portability across targets** | SWIFT/CLaSp/LEAP/CAS-Spec are genuinely **plug-and-play** on any off-the-shelf checkpoint, no retraining — huge practical advantage when you don't control the target's training pipeline. LayerSkip/Kangaroo need at least light modification of or addition to the target. | Head must be (re)trained per target model — not portable to a new target without a training run, though the training itself is cheap. |
| **Multimodal / visual grounding** | No published SSD variant handles vision-specific drafting well yet (FastVLM-SSD is the one exception, and it borrows EAGLE-style imitation-network training to get there) — self-drafting's "free" simplicity hasn't been shown to extend cleanly to visual token handling. | This is EAGLE's proven expansion path: MSD, ViSpec, MASSV, DREAM, HiViS, SpecVLM all successfully extend the attached-head idea to VLMs, each solving the visual-token/drafter-alignment problem in a different way. |
| **Systems complexity at scale** | Simple — one model to serve, no tokenizer/family mismatch possible, no separate drafter lifecycle to keep in sync as the target updates. | Extra engineering: tree-attention kernels, cache management for two structures, and (per DSpark's diagnosis) a **separate object to keep aligned** as the target evolves — a real operational cost in production over time. |
| **Failure mode when pushed hard** | Ceiling is architectural: shallow layers have limited capacity no matter how well you select them (LEAP's zone-search and CLaSp's per-token DP are both fighting the same wall). | Failure mode is distributional: a head trained on fixed data plateaus (Draft-OPD's central finding) unless retrained on-policy — a training problem, not an architectural one, and therefore more fixable with better data/objectives. |

## The one-line summary

**Self-drafting** wins on simplicity, deployment cost, and — for the training-free variants — zero setup cost and universal applicability; it loses on raw acceptance-length ceiling because shallow layers are a compromise, not a specialist.

**EAGLE-based** wins on acceptance length and thus raw speedup, and has a clear, repeatedly-proven path to specialization (including into multimodal territory); it loses on deployment simplicity, needing a second trained artifact per target that must be kept in sync.


# Paper 1 — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077, ICML 2024)

**1. Challenges**

Autoregressive decoding generates tokens sequentially, making LLM inference slow and costly. Standard speculative decoding needs a small draft model that mimics the target LLM, but for small models like a 7B there's often no suitable smaller draft model, and using a 7B to draft for a 13B has too much overhead to be worth it. Training a dedicated small draft model from scratch is prohibitively expensive (TinyLLaMA needed ~3,000B tokens). Existing cheap-draft methods (Medusa, Lookahead) have low draft accuracy (~0.6 or less), which limits speedup.

**2. Methods**

Two key insights drive the design:

- **Feature-level autoregression is easier than token-level.** Instead of predicting the next *token*, the draft model autoregressively predicts the next *feature* — the second-to-top-layer hidden state (just before the LM head) — then reuses the target LLM's own LM head to turn features into token distributions. This alone raised speedup from 1.5x → 1.9x.
- **Resolving sampling uncertainty.** The next feature is ambiguous because it depends on which token was actually *sampled* (e.g., after "I", both "am" and "always" branch into different feature trajectories). EAGLE fixes this by also feeding in the **token sequence shifted one step ahead** (i.e., the actual sampled tokens), removing the ambiguity. This pushed speedup from 1.9x → 2.8x.

Architecturally, the draft model is tiny: it reuses the target LLM's frozen Embedding layer and LM Head, and adds only a trainable "Autoregression Head" = one FC layer (to fuse the concatenated feature+embedding, 2×hidden_dim → hidden_dim) + **a single transformer decoder layer** (<1B params even for the 70B target). Drafting builds a **tree-structured draft with tree attention** (e.g., a 10-token tree in 3 forward passes). Verification uses standard speculative sampling acceptance, so the output distribution is **provably lossless** in both greedy and sampling modes, with no fine-tuning of the target LLM.

**3. Loss**

Training combines two objectives:
- **Regression loss:** predicting the next feature is a regression task, trained with Smooth L1 loss between the predicted feature and the target LLM's true feature.
- **Classification loss:** a cross-entropy loss on the token distribution obtained by passing the predicted feature through the (frozen) LM head, to directly optimize the final draft accuracy.
- Combined as L = L_reg + w·L_cls with w = 0.1.

**4. Training data**

No more than ~70k dialogues from the ShareGPT dataset — a fixed dataset (roughly 2–4B tokens total, vs. 3,000B for training TinyLLaMA). Notably, the queries don't need to be generated by the target model itself; an ablation showed using target-LLM-generated responses helps only slightly. Training takes 1–2 days on 4× A100 (40G) for the 70B target, and even a single RTX 3090 node suffices for 7B–33B targets. Same weights are used zero-shot across all evaluation tasks.

**5. Final results**

Evaluated on MT-bench (dialogue), HumanEval (code), GSM8K (math), Alpaca (instruction-following), with Vicuna 7B/13B/33B, LLaMA2-Chat 7B/13B/70B, and Mixtral 8x7B Instruct:

- For LLaMA2-Chat 70B: 2.7x–3.5x latency speedup and roughly doubled throughput, with the output distribution preserved.
- On MT-bench, EAGLE is ~3x faster than vanilla decoding, 2x faster than Lookahead, and 1.6x faster than Medusa.
- Draft accuracy ~0.8 (vs Medusa's ~0.6).


# Paper 2 — MSD: Speculative Decoding Reimagined for Multimodal Large Language Models (arXiv:2505.14260)

**1. Challenges**

Speculative decoding accelerates LLMs without sacrificing accuracy, but current speculative decoding methods for MLLMs fail to achieve the same speedup as they do for LLMs. Concretely: state-of-the-art feature-level (EAGLE-style) speculative decoding reaches an average acceptance length of ~5 in LLMs, but only 3.47 when applied naively to MLLMs like LLaVA-1.5-7B. Two root causes identified: MLLM inputs mix two very different token types (text vs. visual), and MLLM draft models need *both* language modeling and visual perception skills — a drafter trained only on text fails on visually-grounded tokens (e.g., "eating", "giraffe"), while one trained only on vision data struggles with function words like "the" and "to".

**2. Methods**

Builds on EAGLE-style feature-level drafting (token-level ablation: 1.81 acceptance length vs 3.47 feature-level), with two MLLM-specific redesigns:

- **Modality-decoupled drafting.** For text tokens, the drafter concatenates the token's feature with the next token's embedding (the EAGLE recipe — this resolves sampling uncertainty); for visual tokens, their features are fed directly into the draft model without the shifted-token concatenation, since visual tokens aren't generated autoregressively and relate to all other visual tokens rather than just preceding ones.
- **Two-stage training.** Stage 1 trains the draft model on text-only instruction-tuning data to build language modeling ability; Stage 2 gradually introduces multimodal data (progressive data mixing) to add visual perception capability — rather than jumping straight to vision data, the multimodal fraction is ramped up progressively.

Verification is standard lossless speculative sampling by the target MLLM.

**3. Loss**

Same family as EAGLE (it is feature-level SD): a feature regression loss (Smooth L1 between predicted and true target-model features) plus a cross-entropy classification loss through the frozen LM head, applied over the response tokens. The novelty is not in the loss but in the input construction and training curriculum.

**4. Training data**

Stage 1: text-only instruction-tuning dialogues (ShareGPT-style data, as in EAGLE). Stage 2: vision instruction-tuning data (LLaVA-style multimodal instruction data), mixed in with progressively increasing proportion. Targets: LLaVA-1.5-7B and LLaVA-1.5-13B.

**5. Final results**

Evaluated on VQA, document understanding (ChartQA, TextVQA), hallucination and comprehensive benchmarks (MME, etc.): MSD boosts inference speed by up to 2.29× for LLaVA-1.5-7B and up to 2.46× for LLaVA-1.5-13B, with higher acceptance lengths than the naive EAGLE-on-MLLM baseline (3.47), while remaining lossless. Both key modules (decoupling, two-stage training) contribute in ablations.

---

# Paper 3 — ViSpec**(arXiv:2509.15235, NeurIPS 2025, Huawei Noah's Ark Lab)**

**1. Challenges**

Speculative decoding is well-established for LLMs, but on VLMs existing methods managed only modest speedups (<1.5×). Core hypothesis: large VLMs can filter redundant image information layer-by-layer without hurting text comprehension, but a tiny drafter cannot — the hundreds of highly redundant image tokens overwhelm a small draft model, which struggles to extract relevant visual information while keeping textual coherence. A second obstacle: multimodal training datasets rarely contain long assistant responses, which speculative-decoding drafters need to learn from.

**2. Methods**

Built on the EAGLE/EAGLE-2 draft-head framework, with three additions:

- **Vision adaptor (image token compression):** a lightweight module compresses the many image tokens into a small set of compact tokens that are integrated into the draft model's attention layers while preserving the original image's positional information. The drafter attends to a distilled visual summary rather than the full token grid.
- **Global visual feature injection:** a global feature vector is extracted per image and added to every subsequent text token (until the next image), inspired by EAGLE's target-aware feature injection — keeping persistent global visual context in every drafting step.
- **Synthetic long-response dataset + training strategy:** they repurpose existing vision-language datasets and use the *target VLM itself* (with modified prompts) to generate extended responses. Training is designed (with multi-token prediction) to prevent the drafter from shortcut learning — i.e., exploiting its direct access to the target's hidden states instead of genuinely learning to predict.

**3. Loss**

EAGLE-family training objective: feature regression + cross-entropy through the frozen LM head, extended with a multi-token prediction (MTP) objective (the drafter is trained to predict multiple future steps, which combats the hidden-state shortcut problem). Exact weightings follow the EAGLE recipe.

**4. Training data**

A curated synthetic corpus: existing multimodal datasets (e.g., image captioning/VQA sources) whose prompts are modified to elicit long responses, with outputs generated by the target VLM itself. Target models: Qwen2.5-VL-3B/7B-Instruct, LLaVA-v1.6-Vicuna-7B/13B (four popular VLMs).

**5. Final results**

Achieves up to 3.22× speedup — the first substantial speedup in VLM speculative decoding, evaluated on COCO Captions, GQA, MME and other benchmarks, and outperforming EAGLE-2 baselines. Ablations (vs EAGLE-2): image embedding compression adds up to +30% speedup, global feature injection +7%, and the generated long-response dataset another +30% — the three components stack.

---

# Paper 4 — MASSV**(arXiv:2505.10526, EMNLP 2025)**

**1. Challenges**

Two fundamental obstacles to using speculative decoding with VLMs when the drafter is an off-the-shelf small language model: (a) small LMs lack the architectural components to process visual inputs at all, and (b) even if they could see text, their token predictions are misaligned with a VLM target whose predictions are conditioned on visual context — so acceptance rates crater on image-grounded content. Unlike EAGLE-style methods, MASSV addresses the "independent draft model" setting: adapting an existing small LM from the same model family as the target VLM into a multimodal drafter.

**2. Methods**

A two-phase recipe that turns a small same-family LM into a multimodal drafter:

- **Phase 1 — Multimodal adaptation (projector pretraining):** the target VLM's own frozen vision encoder is connected to the small draft LM via a lightweight trainable projector (MLP), LLaVA-style. Only the projector trains in this phase; the vision encoder and draft LM stay frozen. This grafts vision onto the drafter with minimal new parameters and guarantees the drafter sees images through the same "eyes" as the target.
- **Phase 2 — Self-data distillation (self-distilled visual instruction tuning, SDViT):** rather than fine-tuning on ground-truth dataset answers, the *target VLM generates the responses* to visual instructions, and the drafter is fine-tuned on this self-generated data. This directly aligns the drafter's token distribution with the target's actual output distribution — the quantity that matters for acceptance rate in speculative decoding.

At inference, standard draft-then-verify speculative decoding (with greedy verification: draft token accepted iff it matches the target's argmax).

**3. Loss**

Standard language-modeling cross-entropy (next-token prediction), applied in both phases: in Phase 1 on image-caption alignment data (only projector params update), and in Phase 2 on target-generated responses (drafter LM + projector update). The "distillation" is data-level (training on teacher-generated outputs) rather than logit-level KL — hence "self-data distillation."

**4. Training data**

Phase 1: standard vision-language alignment/captioning pairs (LLaVA-style pretraining corpus). Phase 2: visual instruction prompts (from existing visual instruction datasets) whose responses are regenerated by the target VLM itself. Draft models are smaller LMs from the same family as the target VLM (so tokenizer/embedding spaces match); experiments cover targets like the Qwen2.5-VL and Gemma-family VLMs with their small same-family LM counterparts as drafters.

**5. Final results**

MASSV delivers up to ~1.46× end-to-end speedup over text-only speculative decoding baselines (i.e., the same small drafter without vision adaptation/self-distillation), with consistently higher acceptance lengths on visually-grounded tasks. Both phases matter: vision adaptation lets the drafter track image-dependent tokens, and self-data distillation closes the distribution gap. Note the speedup scale is smaller than EAGLE-style methods (Papers 2, 3, 5) — the trade-off is that MASSV is a general recipe requiring no bespoke draft-head design and works with existing small LMs.

# Paper 5 — DREAM: Drafting with Refined Target Features and Entropy-Adaptive Cross-Attention Fusion for Multimodal Speculative Decoding (arXiv:2505.19201, NYU + UPenn + Cerebras)

**1. Challenges**

Speculative decoding is mature for LLMs but underexplored for VLMs, which pose unique problems: heavy visual processing cost in the drafter, and the difficulty of keeping a tiny draft model aligned with a target that builds rich fused visual-textual representations. Prior approaches (EAGLE-style concat of last-layer features) underuse the target model: the target's *intermediate* layers contain highly informative multimodal features that the drafter never sees, and processing all visual tokens in the drafter adds needless latency.

**2. Methods**

Three innovations on top of the draft-and-verify framework:

- **Cross-attention injection of refined target features.** Unlike EAGLE, which concatenates last-layer features with token embeddings, DREAM adds a cross-attention layer inside the draft model: the drafter's generated token embeddings act as queries, and cached intermediate features from the target model (both visual and textual) serve as keys/values. The drafter actively retrieves whatever target knowledge is relevant at each step, instead of passively receiving one feature vector.
- **Entropy-adaptive intermediate feature selection.** Which target layer to draw features from is chosen adaptively using attention entropy — the layer whose representations are most informative (low-entropy, focused attention) is selected to supervise and feed the draft model, rather than hardcoding a fixed layer.
- **Visual token compression.** The draft model's visual input is compressed (important tokens kept, redundant ones dropped), guided by the target model's intermediate features/attention — substantially cutting drafter latency without hurting accuracy.

**3. Loss**

The drafter is trained with the usual speculative-decoding combo — cross-entropy against the target's token distribution plus feature-alignment (regression/distillation) loss — with the distinguishing twist that the feature supervision comes from entropy-selected intermediate target layers rather than only the final layer. So: token-level CE + intermediate-feature distillation, with the entropy criterion deciding which layer supervises.

**4. Training data**

Multimodal instruction-tuning data in the LLaVA style (visual instruction corpora with images + conversations), used to train the lightweight drafter for each target VLM; the target models themselves stay frozen. Targets span four families: LLaVA-v1.6-Vicuna-7B/13B, SmolVLM-2B, Pixtral-12B, and Gemma3 — deliberately diverse in architecture and scale.

**5. Final results**

Up to 3.6× speedup over conventional autoregressive decoding across a broad range of multimodal benchmarks, significantly outperforming prior speculative-decoding baselines (including EAGLE-style adaptations) in both throughput and average acceptance length. Gains hold across all four VLM families, and each of the three components (cross-attention fusion, entropy-adaptive selection, visual compression) contributes in ablations. Output quality is preserved (lossless verification).

**Positioning note:** where ViSpec (Paper 3) compresses image tokens *for* the drafter and injects one global image vector, DREAM goes further — it taps the target's intermediate layers via cross-attention and picks the best layer adaptively. Both attack the same bottleneck: a small drafter can't digest raw visual tokens the way a large target can.

# Paper 6 — HiViS: Hiding Visual Tokens from the Drafter for Speculative Decoding in VLMs (arXiv:2509.23928, CAS)

**1. Challenges it's trying to solve**

Two visual-token problems that cripple VLM drafters: (a) **semantic misalignment / KV-cache bias** — the drafter and target VLM may come from different families, so the target's visual-token representations don't mean the same thing to the drafter; feeding them in poisons the drafter's KV cache at prefill (their ablation shows an EAGLE-style drafter trained *without* visual tokens actually beats one trained *with* them); and (b) **compute burden** — hundreds of visual tokens balloon the drafter's prefill length and slow its self-attention at every drafting step. Meanwhile, visual tokens in large VLMs are known to be highly redundant — most can be dropped without hurting generation.

**2. Methods**

An **explicit-implicit input decomposition**:

- **Hide all visual tokens from the drafter.** The drafter's explicit input is text tokens only, so its prefill sequence length equals the text length — visual processing cost in the drafter drops to zero.
- **Use the target VLM as the semantic fusion model.** Instead of raw visual tokens, the drafter reuses the target's last-layer hidden states for the text positions — since the target already attended over the image, those text-position features implicitly carry the fused visual semantics. The drafter gets vision "for free" through features the target computed anyway.
- **Time-step-aware aligned training with bias-correction residuals.** A problem arises during multi-step independent drafting: the drafter drifts because it no longer receives fresh target features at each speculative step. HiViS trains the drafter with a time-step-aware scheme in which step-dependent bias-correction residuals teach it to autonomously propagate and refine the visual-textual semantics across drafting steps.

**3. Loss**

EAGLE-family training (feature alignment + cross-entropy through the LM head), extended with the time-step-aware alignment objective: supervision is applied per drafting step with step-dependent residual corrections, so the drafter learns to stay aligned with the target over multiple autonomous steps rather than just one.

**4. Training data**

Multimodal instruction data in the standard LLaVA-style setup (they evaluate under ViSpec's exact protocol as well, including its system prompt); drafters trained per target VLM across representative models (LLaVA-family, Qwen2-VL-family targets). The target VLM stays frozen; only the lightweight drafter trains.

**5. Final results**

Across representative VLMs and benchmarks (VQAv2, GQA, TextVQA, ScienceQA, captioning, etc.), HiViS delivers significant improvements in average acceptance length and speedup ratio versus EAGLE-2 and vision-aware baselines (compared head-to-head with ViSpec under identical settings), while keeping generation lossless. The headline efficiency win: drafter prefill length reduced to text-only length (visual tokens contribute zero drafter compute), with no loss of visual grounding.

---

# Paper 7 — Draft-OPD**(arXiv:2605.29343, 2026 — Shanghai AI Lab et al.)**

**1. Challenges it's trying to solve**

Draft models for speculative decoding (EAGLE-3, DFlash style) are built by supervised fine-tuning (SFT) on target-generated trajectories — and this **plateaus**: after a warm-up, more SFT stops improving acceptance length on test data (and can even hurt). Root cause: **offline-to-inference mismatch.** In SFT, every training prefix comes from the target's own trajectories; but at inference, the target verifies blocks proposed *by the drafter*, i.e., the drafter operates on its own (error-containing) states, which it never saw during training. The natural fix — on-policy distillation (OPD) — is itself hard for drafters: a tiny drafter can't reliably roll out complete sequences on its own, and if you let the target "assist" the rollout, the collected sequences collapse back to the target distribution, destroying exactly the on-policy signal you wanted.

**2. Methods**

**Draft-OPD** = on-policy distillation with **error-position replay**:

- **Target-assisted rollout for stable continuations:** sequences are generated with the target in the loop (standard speculative decoding), which keeps trajectories coherent.
- **Replay from verification-exposed error positions:** during verification, the exact positions where the drafter's proposals get *rejected* are recorded; training then replays drafting from those draft-induced error states, so the target supervises the drafter precisely on the states the drafter itself produces — including both accepted and rejected proposals.
- This concentrates training signal on the draft-induced errors that actually limit speculative acceptance, rather than on target-distribution prefixes the drafter already handles.

Infrastructure: built on verl (RL/post-training framework) + an SGLang-based DFlash runtime; used as a post-training stage after standard SFT warm-up.

**3. Loss**

A distillation objective (cross-entropy/KL of the drafter's distribution against the target model's distribution) evaluated **at on-policy, draft-induced states** — i.e., the same KD loss as SFT-based drafter training, but the *states it's computed on* are the drafter's own replayed error positions with target feedback, not offline target trajectories. That state-distribution shift is the whole contribution; the loss functional form stays KD.

**4. Training data**

Prompts from reasoning/coding corpora (GSM8K-style math and code tasks; evaluation JSONLs include AIME24, GSM8K, MATH500, MBPP), with all supervision generated online: rollouts collected via speculative decoding between the drafter and its frozen target model (Qwen-family "thinking"/reasoning models). No fixed response dataset — the data is the drafter's own live proposals plus target verification feedback, after an initial SFT warm-up phase on target trajectories.

**5. Final results**

Over 5× lossless acceleration for thinking (long-reasoning) models across diverse tasks — improving over EAGLE-3 by ~23% and over DFlash by ~13% in acceleration. The paper's diagnostic result is equally important: it demonstrates the SFT plateau empirically and shows the gains come specifically from training on verification-time error states (continuing offline SFT with the same compute doesn't help).

**Positioning note:** this is the same philosophical move as MASSV's self-data distillation (align drafter to target distribution), taken one step further — align the drafter *on its own mistake states*, not just on target outputs. It's modality-agnostic (pure LLM setting), but directly applicable to the VLM drafters in Papers 2–6.

---

# Paper 8 — SpecVLM**(arXiv:2509.11815, AMD + XJTU)**


**1. Challenges it's trying to solve**

Porting speculative decoding to VLMs hits *systems* constraints, not just modeling ones: the prefill stage is dominated by visual tokens whose count scales with image resolution and video length, inflating both compute and memory — especially the KV cache — for target *and* draft models. Also, building drafters usually requires costly offline distillation corpora (generating a big dataset of target outputs before training can even start).

**2. Methods**

Three pieces:

- **EagleVLM baseline:** a carefully engineered EAGLE-2-style draft model for VLMs (lightweight drafter consuming target features + tokens, dynamic draft trees), already giving 1.5–2.3× end-to-end speedups over autoregressive inference.
- **Elastic visual compressor:** instead of committing to one compression scheme, it adaptively selects among four primitives — pruning, pooling, convolution, and resampler — per input, balancing FLOPs/parameters against accuracy. Different images/resolutions/tasks get different compression treatment for the drafter's visual input.
- **Online-logit distillation:** the drafter is trained *on the fly* against the teacher's logits and penultimate features produced during training, eliminating the offline corpus-generation step entirely. This surfaces a training-time scaling effect: longer online training monotonically increases the draft model's average accepted length.

**3. Loss**

A combined online distillation loss: cross-entropy / KL against the teacher's live logits + a regression loss on the teacher's penultimate-layer features (the EAGLE-style feature alignment, but with targets streamed online rather than precomputed). Same functional family as EAGLE's L_reg + L_cls, sourced online.

**4. Training data**

No offline distillation corpus: visual-instruction prompts (LLaVA-style data) are fed through the frozen teacher during training, and its logits/features are the supervision. Training is short — the headline results are reached within 5 epochs. Targets: LLaVA-v1.5-7B/13B and LLaVA-v1.6-7B/13B.

**5. Final results**

2.5–2.9× end-to-end lossless speedups across LLaVA-Bench and MMMU, consistent over image resolutions and task difficulties, beating its own EagleVLM baseline (2.09×/1.91×/2.31×/2.29× on LLaVA-v1.5-7B/v1.6-7B/v1.5-13B/v1.6-13B → SpecVLM 2.20×/2.03×/2.41×/2.38× on LLaVA-Bench-in-the-Wild, measured on an RTX 4090), while preserving the target's output distribution. Latency breakdown shows the visual compressor slashes drafter-side latency (e.g., 65 ms → 46 ms components).

---

# Paper 14 — DSpark**(DeepSeek, June 2026 — released with the open-source DeepSpec training/eval stack)**

**1. Challenges it's trying to solve**

Two production failure modes of speculative decoding at serving scale:

- **Suffix decay in parallel drafters.** Parallel/block drafters (DFlash/MTP-style) propose a whole block in one forward pass, but each position is predicted without seeing its neighbors — so acceptance probability decays rapidly toward the end of the block. Autoregressive drafters (EAGLE-3) don't decay but pay per-token drafting latency.
- **Verification waste under high concurrency.** Verifying long fixed-length draft blocks for every request burns batch capacity on tokens that were likely to be rejected anyway. In a busy serving system this degrades throughput badly — baselines hit an "operational cliff" under strict per-user speed SLAs.

**2. Methods**

- **Semi-autoregressive generation:** a heavy *parallel* drafting backbone coupled with a lightweight *sequential* module (a low-rank "Markov head") that injects intra-block dependencies — later draft tokens get conditioned on earlier ones, mitigating suffix decay at almost no added latency. The best of both drafter families.
- **Confidence-scheduled verification:** a calibrated confidence head estimates each draft prefix's *survival probability*; combined with engine-specific throughput profiles and live GPU load, the scheduler dynamically tailors the verification length per request — short verification for shaky drafts, long for confident ones, and globally load-aware.
- **Zero-Overhead Scheduling (ZOS):** the prefix scheduler runs asynchronously, estimating upcoming verification capacity from confidence outputs two steps earlier so the GPU pipeline never stalls waiting for scheduling decisions.
- **Training-system engineering:** target activations are cached and only pre-LM-head hidden states are communicated (O(d) communication), and *anchor-bounded sequence packing* bundles isolated prediction blocks into dense training batches — decoupling drafter training cost from the target's long contexts.

Verification uses standard rejection sampling, so output distribution is provably identical to vanilla decoding (lossless).

**3. Loss**

Drafter: distillation-style training as in the EAGLE-3/DFlash lineage — cross-entropy of the semi-autoregressive drafter's predictions against the target model's outputs/hidden-state supervision (with the sequential Markov head trained jointly to model intra-block dependencies). Plus a separate **calibrated confidence objective** for the confidence head, trained so its scores track actual prefix acceptance/survival probabilities (calibration is what makes the scheduler's cost-benefit math valid).

**4. Training data**

Target-model-derived training corpora built through the DeepSpec stack (cached target activations over training text; no human-labeled data needed), applied to open targets — three Qwen3 model sizes (also Gemma supported in DeepSpec) — and to DeepSeek-V4 (V4-Flash / V4-Pro) for the production experiments.

**5. Final results**

- **Offline, vs SOTA drafters:** macro-average accepted length improves over EAGLE-3 by 30.9% / 26.7% / 30.0% across the three Qwen3 sizes, while also beating DFlash — i.e., autoregressive-level acceptance at parallel-drafting cost.
- **Production (DeepSeek-V4 serving):** per-user generation speed up 60–85% at matched throughput versus the deployed MTP-1 baseline, shifting the serving Pareto frontier; under strict latency SLAs (120 tok/s/user on V4-Flash, 50 tok/s/user on V4-Pro) aggregate throughput improves by headline figures of ~661% and ~406%, because the baseline collapses near those operating points while DSpark doesn't. (Caveat: production numbers are DeepSeek's own serving stack, not yet independently reproduced.)
- 
---

# Self-Speculative Decoding

# Overview: 

# Paper 9 — LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding (arXiv:2404.16710, Meta — ACL 2024)


**1. Challenges it's trying to solve**

Standard speculative decoding needs a *separate* draft model — extra memory, extra training, tokenizer matching headaches. Early-exit approaches (predict from an intermediate layer) avoid that but suffer poor accuracy: LLMs aren't trained to make their intermediate layers predictive, and each early-exit head needs training. LayerSkip asks: can one model be its own draft *and* verifier, with no extra parameters and minimal accuracy loss?

**2. Methods**

- **Training recipe:** (a) **layer dropout** with rates that increase with depth (early layers rarely dropped, late layers dropped often), making the model robust to skipping later layers; (b) **early exit loss** — the *shared* LM head (no new heads) is applied to intermediate layers during training so earlier layers become genuinely predictive.
- **Inference — self-speculative decoding:** the model drafts tokens by exiting early at layer E, then verifies those drafts by running the remaining layers L−E. Crucially, the draft and verify passes share the same KV cache and compute for layers 1…E (the draft's early-layer computation is reused during verification), so verification only pays for the remaining layers. No auxiliary model, no extra memory beyond the base model.

**3. Loss**

Total loss = a weighted sum of cross-entropy losses at multiple layers: the normal final-layer LM loss plus early-exit cross-entropy at intermediate layers (through the shared LM head), with a curriculum/per-layer weighting scheme (later layers weighted more; early-exit loss scaled so it doesn't degrade final-layer quality). No feature-regression term — it's pure LM loss at multiple depths, combined with stochastic layer dropout during the forward pass.

**4. Training data**

Applied in three regimes, showing the recipe is general: (a) pretraining from scratch (Llama-style models trained on standard large text corpora), (b) continual pretraining of existing checkpoints (e.g., Llama2 7B/13B on a modest token budget), and (c) domain-specific finetuning — code (CodeLlama setting) and semantic parsing (TOPv2). No new datasets are introduced; the contribution is the training recipe.

**5. Final results**

Self-speculative decoding speedups of up to ~2.16× on summarization (CNN/DM), ~1.82× on coding (HumanEval), and ~2.0× on semantic parsing (TOPv2), relative to autoregressive decoding of the same model — with negligible drop in final-layer accuracy from the modified training, and better early-exit accuracy at all depths than baseline models. Memory footprint is essentially that of a single model, unlike two-model speculative decoding.

**Positioning note:** LayerSkip is the "self-drafting" branch of the speculative-decoding family tree — no separate drafter at all — versus EAGLE's "tiny attached head" branch, which the VLM papers (2–6, 8) all descend from.

---

# Inspiration Papers

# Paper 10 — Cache-to-Cache**(arXiv:2510.03215, Tsinghua NICS — ICLR 2026)**

**1. Challenges it's trying to solve**

Multi-LLM systems (ensembles, agent pipelines, big-model-helps-small-model setups) today communicate through **text**: model A must compress its internal understanding into output tokens that model B re-reads. This has two costs — rich internal semantics get lost in the text bottleneck (ambiguous natural-language descriptions of what the model "knows"), and token-by-token generation of the intermediate message adds latency. The question posed: can LLMs communicate directly, beyond text? Oracle experiments show KV-cache is a viable medium: enriching a model's KV-cache semantics improves response quality without increasing cache size.

**2. Methods**

**C2C (Cache-to-Cache)**: a *Sharer* model and a *Receiver* model both prefill the input; then a neural **C2C Fuser** projects the Sharer's KV-cache into the Receiver's KV-cache space and fuses them, so the Receiver decodes conditioned on both models' deep semantics — with no intermediate text ever generated. The Fuser has three modules: (1) a **projection** module (concatenates and maps the two models' KV-caches through projection/feature-fusion layers), (2) **dynamic weighting** (modulating how much Sharer information each position gets), and (3) a **learnable gating** mechanism that selects *which Receiver layers* actually benefit from cache communication (per-layer gates, trained with a differentiable relaxation — Gumbel-style — then thresholded). Both LLMs stay frozen; only the Fuser trains. It works across different model families, sizes, and even different tokenizers (Qwen, Llama, Gemma combinations).

**3. Loss**

Standard next-token-prediction cross-entropy on the Receiver's outputs *given the fused cache* — i.e., the Fuser is trained end-to-end so that the Receiver, decoding with the injected Sharer semantics, better predicts correct responses. No auxiliary reconstruction loss on the cache itself; the supervision flows through the frozen Receiver.

**4. Training data**

General instruction/QA-style corpora used to train only the lightweight Fuser (the paper trains on open SFT-style data and evaluates on held-out benchmarks like MMLU-Redux, OpenBookQA, ARC-C, C-Eval); the Sharer and Receiver LLMs are untouched pretrained checkpoints from Qwen2.5/Qwen3, Llama-3.2, and Gemma families, paired in various size combinations (typically a stronger/specialized Sharer + a smaller Receiver).

**5. Final results**

C2C achieves 8.5–10.5% higher average accuracy than the individual models alone, beats text-based communication between the same model pair by ~3–5% accuracy, and delivers ~2× average latency speedup versus text communication (no intermediate generation). Gains hold across model families and scales; the learned gates show only a subset of layers benefits from fusion.

# Paper 11 — CompoDistill: Attention Distillation for Compositional Reasoning in Multimodal LLMs (arXiv:2510.12184, KAIST)

**1. Challenges it's trying to solve**

To make MLLMs cheap enough for real deployment, people distill a large teacher MLLM into a small student. But existing KD methods (logit matching, last-layer feature alignment) transfer the teacher's *visual recognition* fine ("what is in the image") while failing to transfer its *visual perception / compositional reasoning* abilities ("who is doing what to whom, where") — a gap prior work largely ignored. Their diagnosis: **visual attention misalignment** — for a query like "A woman is on the table," the distilled student attends to irrelevant image regions while the teacher attends to the right ones, even when their output logits look similar.

**2. Methods**

**CompoDistill** = KD framework that explicitly aligns the student's *attention over visual tokens* with the teacher's. Pipeline (per their released code) has three stages: (1) **Distilled Pre-Training** (alignment stage with distillation), (2) **Distilled Fine-Tuning** — the core stage, where the attention distillation is applied: the student's query-conditioned attention maps over image tokens are matched to the teacher's (handling teacher/student layer-count mismatch via layer mapping), alongside response distillation, and (3) standard **SFT**. Teacher/student are same-family LLaVA-style models (e.g., LLaVA-4B teacher → 2B student), so visual token layouts are comparable; the method also transfers to a more advanced backbone, showing generality.

**3. Loss**

A combination of: (a) the standard language-modeling cross-entropy, (b) logit-level KD (KL divergence to the teacher's token distribution), and (c) the novel **attention-distillation loss** — a divergence (KL-style) between teacher and student attention distributions over visual tokens for text queries, which is what forces the student to "look where the teacher looks." (c) is the paper's contribution; (a)+(b) are the inherited recipe.

**4. Training data**

Standard LLaVA training corpora — image-text alignment data for the pretraining stage and visual instruction-tuning data for fine-tuning stages (they also ablate dataset scale, showing distillation gains grow when the training set is doubled). All supervision signals besides ground-truth text come from the frozen teacher (its logits and attention maps).

**5. Final results**

On compositional reasoning benchmarks (the CR suite — SugarCrepe-style perception tests) CompoDistill significantly outperforms prior KD methods distilled from the same teacher, while maintaining strong performance on general VQA (VQAv2, GQA, VizWiz, TextVQA, MME) — i.e., it fixes the perception gap without sacrificing recognition. Sub-2B students get closest-to-teacher performance among compared methods, and attention-similarity analyses confirm the student's visual attention actually aligns after training.

**Positioning note:** complements Paper 7's lesson — *what* you distill matters as much as *how*: Draft-OPD fixed the state distribution; CompoDistill fixes the signal (attention, not just logits).

---

# Token Compression

# Paper 12 — FastVLM**(arXiv:2412.13303, Apple — CVPR 2025)**

**1. Challenges it's trying to solve**

VLM accuracy — especially on text-rich images (documents, charts, UI) — depends heavily on input resolution. But standard ViT encoders scale terribly with resolution: token count explodes quadratically and stacked self-attention makes encoding latency blow up. That hurts twice: slow vision encoding *and* a flood of visual tokens inflating the LLM's prefill. Result: time-to-first-token (TTFT) is dominated by the vision side. Existing fixes (token pruning, merging, tiling schemes like AnyRes) add complexity and often trade away accuracy. The paper asks: what's the *jointly optimal* operating point across image resolution, vision latency, token count, and LLM size?

**2. Methods**

- **A systematic efficiency analysis** of the (resolution, encoder, token count, LLM size) design space, producing Pareto curves of accuracy vs. TTFT. A key finding: at a fixed runtime budget, it's sometimes better to grow the LLM than the resolution.
- **FastViTHD**, a novel hybrid convolutional-transformer vision encoder built for high resolution: a conv stem + three convolutional stages + two transformer-block stages, each stage preceded by patch embedding that halves spatial dimensions, plus multi-scale pooling and extra downsampling. It outputs 4× fewer tokens than FastViT and 16× fewer than ViT-L/14 at 336px, with far lower encoding latency.
- **FastVLM** = FastViTHD + a simple MLP projector + LLM (Qwen2/Vicuna variants). Crucially, no token pruning, merging, or tiling tricks: the token-count/resolution balance is controlled *solely by scaling the input image*, keeping the design simple. Encoder is pretrained CLIP-style on reinforced image-text data (MobileCLIP's DataCompDR recipe), then the VLM is trained LLaVA-style.

**3. Loss**

Two independent stages: (a) the FastViTHD encoder is pretrained with a CLIP-style contrastive image-text objective on reinforced data; (b) VLM training uses the standard autoregressive cross-entropy (next-token prediction) in the usual LLaVA two-stage recipe (projector alignment, then full visual instruction tuning). No new loss is proposed — the contribution is architectural + design-space analysis.

**4. Training data**

Encoder pretraining: large-scale reinforced image-text pairs (DataCompDR-style). VLM training: the LLaVA-1.5 setup (≈558K alignment pairs + 665K visual instruction samples) for controlled comparisons, plus scaled-up instruction mixtures for the stronger released variants (higher-quality multi-source data, text-rich/document-heavy content included). Benchmarked on an M1 MacBook Pro for on-device claims.

**5. Final results**

In the like-for-like LLaVA-1.5 setup: 3.2× faster TTFT at similar benchmark accuracy versus prior encoders. Headline comparison: versus LLaVA-OneVision-0.5B, FastVLM reaches up to 85× faster TTFT with a 3.4× smaller vision encoder at comparable accuracy. Across VLM benchmarks (SeedBench, TextVQA, DocVQA, MMMU, etc.), it sets a state-of-the-art resolution-latency-accuracy trade-off, and the "just scale the image" strategy removes the need for pruning/merging machinery entirely.

**Positioning note:** this is the *orthogonal* attack on the same enemy the speculative-decoding papers face — instead of making the drafter cope with visual-token floods (ViSpec/DREAM/HiViS/SpecVLM), FastVLM shrinks the flood at the source. The two compose: fewer visual tokens also means cheaper drafting and smaller KV caches.

---

# Paper 13 — SmolVLM: Redefining Small and Efficient Multimodal Models (arXiv:2504.05299, Hugging Face) — focused on architecture & vision compression per your request

**1. Challenges it's trying to solve**

Flagship VLM design choices don't transfer to small models. Common practice — bolting a big vision encoder onto a small LM, long context assumptions, CoT-heavy training data — produces small models with disproportionate memory footprints and poor accuracy. The paper systematically studies what *actually* matters for tiny (sub-2.5B) multimodal models targeting on-device inference: encoder/LM parameter balance, context length, visual token compression, tokenization details, and data mixture.

**2. Methods (architecture & vision compression focus)**

- **Backbone pairing:** SmolLM2 language models (135M / 360M / 1.7B) paired with SigLIP vision encoders — with a key finding that **encoder/LM capacity must be balanced**: the smallest LMs pair best with a small encoder (SigLIP-B/16, ~93M), while only the 2.2B model justifies the larger SigLIP-SO400M. A huge encoder on a tiny LM is wasted compute.
- **Aggressive pixel-shuffle vision compression:** visual tokens are compressed via pixel shuffle (space-to-depth) — rearranging spatial patches into channel depth — with a *higher* compression ratio (r=4 for the small models) than larger models like Idefics3 use (r=2). Each 384×384 image tile ends up as just **81 visual tokens**. Small LMs actually *prefer* fewer, denser visual tokens.
- **Extended context:** RoPE base is increased to stretch context to ~16k tokens — necessary because image tiles + text quickly exhaust small default contexts; longer context gave clear gains.
- **Image splitting + learned positional tokens:** high-res images are split into tiles; sub-image positions are marked with **learned positional tokens rather than raw string tokens** — with string tokens, small models suffered badly (notably OCR performance collapse); learned tokens fixed it.
- **Data/curriculum findings:** minimal CoT data (too much reasoning text hurts small models), moderate video sequence lengths, careful media intro/outro token structuring.

**3. Loss**

Standard autoregressive cross-entropy throughout — a vision-language alignment/training stage then instruction tuning (and a video-capable SmolVLM2 variant trained with mixed image/video data). No contrastive or distillation losses; the paper's contributions are architecture, tokenization, and data-mix ablations, all under plain next-token prediction.

**4. Training data**

Open multimodal mixtures: The Cauldron and Docmatix-style document/OCR-heavy corpora (augmented with math-writing/OCR sources) for image training; for the video variant, open video instruction datasets (LLaVA-video-style mixes). All components — data, code, models — are fully open.

**5. Final results**

SmolVLM-2.2B achieves performance competitive with VLMs many times its size while using a fraction of the GPU memory; the tiny SmolVLM-256M **uses <1GB of GPU RAM yet outperforms Idefics-80B** — a model ~300× larger from 18 months earlier. The family (256M/500M/2.2B) delivers strong image and video benchmark scores at unmatched memory efficiency, enabling genuine on-device deployment (demonstrated in browser/phone apps). The ablations are the lasting contribution: balanced encoder/LM sizing, aggressive pixel-shuffle compression, long context, and learned positional tokens are the levers that matter at small scale.

---

# Paper 15 — SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration (arXiv:2410.06916, ICLR 2025)

**1. Challenges it's trying to solve**

Most speculative decoding methods (EAGLE, Medusa, etc.) need extra parameters and extensive training to build an effective draft model, which restricts applicability — you can't just drop them onto an arbitrary LLM/task combo without a training investment first. Fixed layer-skipping self-speculative methods (like LayerSkip) also need dedicated training of the target itself. SWIFT asks: can a *plug-and-play*, training-free layer-skipping drafter work on off-the-shelf models, adapting per input?

**2. Methods**

Key empirical finding motivating the design: LLMs exhibit **layer sparsity** — many intermediate layers can be skipped with little quality loss — and this sparsity is **task-specific** (the best layers to skip differ by input/task), which argues against a single fixed skip configuration and for **on-the-fly adaptation**. SWIFT splits inference into two phases:

- **Context-based layer set optimization:** before/during generation, SWIFT searches for a good skipped-layer set for the *current* input stream, using **random search combined with interval Bayesian optimization** to propose candidate layer sets efficiently. Crucially, it evaluates candidates using the **target LLM's own already-generated tokens as ground truth** — i.e., it checks retrospectively whether a candidate skip-set would have correctly predicted tokens the (unskipped) target actually produced, allowing **parallel candidate evaluation** without extra forward passes dedicated solely to search.
- **Confidence-aware inference acceleration:** once a layer set is chosen, it's used to self-draft (skip those layers) and verify with the full model, as in standard self-speculative decoding — with acceptance handled to preserve the output distribution (supports both greedy and full sampling, losslessly).
- The optimization step is cheap: reported at only ~0.8% of total inference latency, and the maximum optimization iterations / Bayesian interval are tunable to trade search quality for overhead depending on input type.

**3. Loss**

None — this is a **training-free, inference-time-only** method. No parameters are learned; "optimization" refers to a search procedure over which layers to skip, not gradient-based training. This is SWIFT's central selling point versus EAGLE/LayerSkip/Medusa.

**4. Training data**

None required. Evaluated across a range of off-the-shelf models (LLaMA-2/3 series, Yi-34B series, etc.) and downstream tasks with no fine-tuning of the target — the "optimization phase" runs live per input stream, using the model's own generations as its self-supervision.

**5. Final results**

Over 1.6× speedup on average across a wide range of models and tasks while provably preserving the original output distribution (lossless, both greedy and sampled). An observed **scaling law**: speedup and the optimal layer-skip ratio both increase with model size, i.e., larger LLMs have more exploitable layer sparsity (e.g., on Yi-34B with skip ratio r≈0.45, they report acceptance rate α and speedup under greedy FP16 decoding). Optimization overhead is negligible (~0.8% of latency).

**Positioning note:** This is the "no training, no auxiliary model" corner of the self-speculative family — where LayerSkip (Paper 9) pays a training cost to make early layers predictive and Kangaroo adds a small trained adapter, SWIFT pays neither, trading some speedup ceiling for zero setup cost and universal applicability to existing checkpoints.

---

# Paper 16 — CLaSp: In-Context Layer Skip for Self-Speculative Decoding (arXiv:2505.24196)

**1. Challenges it's trying to solve**

Two problems, layered on top of the SWIFT/Self-SD lineage: (a) building compatible draft models for specialized LLMs (no drop-in smaller sibling) remains hard, and existing plug-and-play layer-skip methods (Self-SD, SWIFT) rely on a **costly, largely static, pre-optimized skip-set** (Self-SD: expensive Bayesian optimization over a training corpus to find one fixed set; SWIFT: adapts over the course of many requests but "diminishes when handling sparse or unique task data" — it needs *volume* to adapt well). (b) Neither dynamically adjusts *within* a single generation at the granularity of "what should I skip right now, given exactly what I just generated" — layer importance is context-dependent at a finer grain than either method exploits.

**2. Methods**

- **Per-step dynamic programming for layer selection.** After each verification step, CLaSp uses the **complete hidden states of the last accepted token** (already computed for free during verification) as the ground-truth target, and picks a new skipped-layer set for the *next* drafting round by solving: minimize the cosine distance between the draft model's (skipped) output hidden state and the full model's true hidden state at that position. This is framed as a DP over "which j layers to skip among the first i layers to best approximate the target's hidden-state trajectory," with an approximate transition (not exactly Markov, but empirically close, verified against brute-force search).
- **Sparse Persistence exploitation:** adjacent tokens need highly similar skip sets (Jaccard similarity high locally, decaying with distance) — so instead of recomputing the DP after every single token, CLaSp updates only every few verification steps (a tunable Layer Optimization Interval), trading a small τ reduction for much lower overhead.
- **Sequence-parallel DP implementation:** the DP's O(L·M) double loop is restructured so states at the same "layer index i" but different skip-counts j are computed independently and packed via a custom attention mask (reusing the same KV-cache, no duplication) — cutting DP wall-clock from ~2.5s to ~0.14s on LLaMA3-70B, comparable to a single verification step.
- Standard speculative sampling (lossless) is used for accept/reject, exactly as in prior self-SD work.

**3. Loss**

None — training-free, same as SWIFT. The "objective" being minimized is the DP's cosine-similarity criterion at inference time, not a trained parameter. No gradient-based training occurs anywhere.

**4. Training data**

None required. Evaluated directly on off-the-shelf LLaMA2/LLaMA3 checkpoints (8B, 13B, 70B, 70B-Chat, and 405B via INT8 quantization) across Spec-Bench's six tasks (MT-bench, WMT14, CNN/DM, Natural Questions, GSM8K, DPR).

**5. Final results**

Consistent **1.3×–1.7×** wallclock speedup over vanilla autoregressive decoding, losslessly, and **outperforms both Self-SD and SWIFT** head-to-head on the same hardware/benchmark (e.g., LLaMA3-70B greedy overall: Self-SD 1.47×, SWIFT 1.28×, CLaSp 1.67×). Skipping ~50–60% of layers is the sweet spot. A clear scaling law: bigger models benefit more (LLaMA3-8B → 1.24× vs. LLaMA3.1-405B → 1.73× on MT-bench), consistent with SWIFT's finding that larger models have more exploitable layer redundancy. DP optimization overhead is negligible after the parallel implementation (~4.8% of latency, further reduced by lowering update frequency).

**Positioning note:** CLaSp sits directly between SWIFT and LayerSkip in the design space — training-free like SWIFT, but re-optimizing per *token* (using the hidden-state signal from verification.

---

# Paper 17 — CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs (arXiv:2510.26843, NeurIPS 2025)

**1. Challenges it's trying to solve**

Two separate lines of work each hit a ceiling: on-the-fly self-speculative methods (SWIFT, CLaSp) are training-free and universally applicable, but fall short of the speed gains achieved by trained methods — a single self-drafting stage has limited headroom. **Cascade speculative decoding** (chaining multiple draft models of increasing size/quality before the target verifies) promises more acceleration by inserting several drafting stages instead of one, but classic cascades require **training multiple distinct draft models**, which is prohibitively costly and has limited real-world adoption. Additionally, the standard cascade coordination algorithms (vertical/horizontal cascades, static trees) are designed for *independent trained* draft models and are inefficient/inappropriate when the "draft models" are just configurations of the same self-speculative target.

**2. Methods**

- **Dynamically Switchable Inference Acceleration (DSIA):** rather than training separate draft models, CAS-Spec builds a **hierarchy of draft stages entirely out of the target model itself**, using training-free acceleration knobs that can be toggled per stage — specifically **layer sparsity** (skipping layers, à la Self-SD/SWIFT/CLaSp) and **activation quantization** — stacking multiple such configurations (e.g., aggressive skip+quantize for the cheapest/fastest stage, milder configs for intermediate stages, full model as final verifier) into a multi-level cascade, all embedded in one model's inference process.
- **Dynamic Tree Cascade (DyTC):** because static vertical/horizontal cascade routing doesn't fit this setting well, CAS-Spec introduces an adaptive router that, at runtime, (a) decides which draft stage(s) to route through, (b) constructs the draft tree shape, and (c) controls per-stage draft lengths — driven by heuristics based on **observed token acceptance rates** and **predicted latency** per stage, aiming to maximize throughput rather than follow a fixed schedule.
- Verification remains standard rejection sampling against the full target, preserving losslessness.

**3. Loss**

None — training-free, consistent with the self-speculative lineage (SWIFT, CLaSp). DSIA configurations (which layers to skip, which activations to quantize, at what precision) are selected/tuned via runtime heuristics, not gradient-based training. The "adaptivity" is in the routing algorithm (DyTC), not in learned model parameters.

**4. Training data**

None required for the core method. Evaluated across "various LLMs and datasets" (per the abstract/poster) directly on off-the-shelf checkpoints, following the same evaluation philosophy as SWIFT/CLaSp (Spec-Bench-style multi-task benchmarking).

**5. Final results**

State-of-the-art acceleration among on-the-fly (training-free) speculative decoding methods: average speedup **1.1×–2.3×** over autoregressive decoding across LLMs and datasets — a higher ceiling than single-stage self-SD methods like SWIFT/CLaSp (1.1–1.7× range), attributable to the cascade's multiple drafting stages. The DyTC routing algorithm itself contributes substantially: **+47% average speedup over a cascade baseline** and **+48% over a tree-based baseline**, showing the adaptive routing (not just the cascade structure) is a major source of the gain.

**Positioning note:** CAS-Spec is the "cascade" extension of the SWIFT/CLaSp lineage — instead of picking *one* skip configuration per step (CLaSp) or per request-stream (SWIFT), it maintains a *hierarchy* of configurations and routes dynamically among them, borrowing the "multiple draft models of increasing capability" idea from trained cascade SD but making every stage free (no training) by reusing DSIA knobs on the same target.

---

# Paper 18 — LEAP: Zone-Aware MCTS for LLM Self-Speculative Decoding (ICML 2026)

**1. Challenges it's trying to solve**

Self-speculative decoding's central open problem is **layer configuration strategy** — which subset of the target's layers to use as the draft model. Prior approaches attack this with dynamic programming approximations (CLaSp), Bayesian optimization (Self-SD, SWIFT), or dynamically switchable knobs (CAS-Spec) — all heuristic, greedy, or approximate. LEAP's framing: layer selection is fundamentally a **sequential decision-making problem** (choosing a subset from a huge combinatorial space, e.g., 2^L configurations for L layers), which is exactly the kind of problem Monte Carlo Tree Search is built for — but naively applying MCTS to deep LLMs faces a **prohibitively large search space**.

**2. Methods**

- **MCTS formulation:** frame layer-subset selection as a sequential decision process (include/exclude each layer, or each layer-group, one decision at a time) and search it with MCTS rather than DP approximations or Bayesian optimization — in principle allowing better exploration of good configurations than greedy/local methods.
- **Two structural observations to tame the search space:**
  1. **Prefill-derived redundancy transfers to decoding** — redundancy information computed cheaply during the prompt's prefill pass (which layers are "safe" to skip) remains informative for the subsequent decoding steps, so the expensive redundancy analysis doesn't need to be redone at every token (unlike CLaSp, which re-optimizes per verification step).
  2. **Zone-wise layer redundancy** — layer importance/redundancy isn't uniformly distributed across depth; it clusters into "zones" (contiguous regions of the network with similar redundancy characteristics).
- **Structured search space via zone partitioning + layer grouping:** using observation (2), LEAP partitions the network into zones and groups layers within each zone, turning the raw 2^L configuration space into a much smaller structured space — this grouping is the **inductive bias** that makes MCTS tractable over deep LLMs, since the search now operates over zone/group-level decisions rather than individual layers.
- Plug-and-play: no target retraining, applied directly to existing checkpoints (consistent with the SWIFT/CLaSp/CAS-Spec training-free lineage).

**3. Loss**

None — training-free. MCTS search uses a reward signal (presumably acceptance length or output-fidelity proxy, analogous to CLaSp's cosine-similarity criterion or SWIFT's acceptance-rate objective) to evaluate candidate configurations, but this is a search/planning objective, not a gradient-trained loss.

**4. Training data**

None required for the core method; the prefill-derived redundancy signal is computed from the prompt itself at inference time, not from an offline training set.

**5. Final results**

Speedup of **1.7×–2.0×** for LLM inference — ahead of SWIFT (up to ~1.6×) and comparable to or exceeding CLaSp's best-case numbers (~1.3–1.7×), positioning LEAP as (per its own framing) a stronger search strategy within the same training-free, plug-and-play self-speculative decoding family.

**Positioning note:** LEAP is the natural next point along the SWIFT → CLaSp → CAS-Spec → LEAP progression: each paper keeps the "no training, use the target's own layers" premise fixed and improves the **search/optimization strategy** for deciding what to skip — heuristic search (SWIFT) → DP approximation using verification feedback (CLaSp) → multi-stage cascades with adaptive routing (CAS-Spec) → structured MCTS with a zone-based inductive bias (LEAP). Notably, LEAP's "zone-wise redundancy" finding is close in spirit to your idea 2.1 (prune by criterion rather than arbitrarily) — but LEAP treats zones as a *search-space prior* rather than a pruning/distillation target, so combining LEAP's zone structure with actual weight pruning + healing (rather than pure inference-time skipping) remains open.

---

# Paper 19 — Vegas# Paper 19 — Vegas (a.k.a. SpecAttn): Self-Speculative Decoding with Verification-Guided Sparse Attention (arXiv:2602.07223 / ICML 2026 poster)

*(Note: the ICML poster is titled "Vegas," but its abstract is word-for-word identical to arXiv:2602.07223's "SpecAttn: Co-Designing Sparse Attention with Self-Speculative Decoding" — almost certainly the same paper under a renamed poster title. Treating as one paper.)*

**1. Challenges it's trying to solve**

Long-context LLM inference is severely bottlenecked by KV-cache memory demands. Prior work showed that **sparse-attention self-speculative decoding** — drafting with only a *subset* of the KV cache, then verifying with the *full* KV cache — gives lossless speedups, since a small fraction of KV entries (often ~5%) dominates attention output. But existing sparse-attention drafters rely on **standalone KV-selection algorithms** (e.g., a separate importance-scoring heuristic) to decide which entries to keep for drafting — and this **overlooks that the criticality of every KV entry is already computed, for free, during verification's full-attention pass.** Treating drafting and verification as independent stages wastes that signal.

**2. Methods**

**Co-design drafting and verification around a shared signal:**

- During the **verification** phase (which runs full attention over all KV entries to check draft tokens), the attention weights/logits computed there directly reveal the **exact criticality of every KV entry** — a "free oracle," since this computation happens anyway and previously its byproduct information was thrown away.
- Vegas/SpecAttn captures this: it identifies the critical KV entries **as a byproduct of verification**, and then uses exactly these entries as the sparse KV subset for the **next drafting phase** (rather than an independent, standalone selection heuristic).
- This closes the loop between the two phases: verification not only produces accept/reject decisions but also *tells the drafter which cache entries matter next*, improving draft-token acceptance rate while adding minimal overhead (no separate KV-scoring pass is needed — it's a byproduct, not new computation).

**3. Loss**

None — training-free, purely an inference-time mechanism (KV-entry selection based on attention weights already computed during verification). No model parameters are trained; the "criticality" signal is a direct read-off of existing attention computation.

**4. Training data**

None required. Applied directly to existing long-context LLM checkpoints; evaluated in a long-context inference setting where KV-cache size is the bottleneck (batch inference / long sequence generation).

**5. Final results**

Per the abstract's framing: improves draft token acceptance rate versus prior sparse-attention self-speculative baselines (which used standalone, verification-blind KV selection) while incurring **low KV-selection overhead**, thereby improving overall decoding throughput. (Exact numeric speedup/τ figures aren't in the fetched abstract text — happy to dig further into the PDF body if you want precise numbers.)

**Positioning note:** Vegas/SpecAttn attacks a different axis of self-speculative decoding than SWIFT/CLaSp/CAS-Spec/LEAP — those methods sparsify *layers* (which of the target's depth to use); this paper sparsifies *KV cache/attention* (which of the target's context to attend to), for long-context settings specifically. The two axes (layer sparsity vs. attention/KV sparsity) are orthogonal and, per CAS-Spec's own DSIA framing (which already lists "layer sparsity and activation quantization" as swappable knobs), a natural extension would add "verification-guided KV sparsity" as a third DSIA strategy in a cascade.

---

# Paper 20 — FastVLM (self-speculative decoding for VLMs)# Paper 20 — FastVLM: Self-Speculative Decoding for Fast Vision-Language Model Inference (arXiv:2510.22641, IIT Bombay — IJCNLP-AACL 2025)

*(Important: this is a different paper from Apple's "FastVLM: Efficient Vision Encoding for Vision Language Models" — Paper 12 in our list. Pure name collision; different authors, different problem, different method. I'll call this one "FastVLM-SSD" to avoid confusion going forward if needed.)*

**1. Challenges it's trying to solve**

VLMs suffer high computational cost and inference latency because their decoder (typically an LLM) generates tokens autoregressively, one at a time. The paper targets this specifically through **self-speculative decoding for VLMs** — using part of the VLM's own decoder as the draft, rather than an external drafter — but wants to do better than naive early-exit by giving the draft component genuine access to "deeper" representational knowledge from the full model, not just whatever an early layer happens to compute.

**2. Methods**

A VLM has an encoder (producing multimodal features z = E(I, T₀) from image I and prompt T₀) and a decoder LLM. FastVLM-SSD's draft model is built from an **intermediate decoder layer (the n-th layer)** plus a novel **imitation network (IN)**:

- **Imitation network:** rather than applying the LM head directly to the n-th layer's hidden state (as plain early-exit does), the IN takes the n-th layer's representation and is trained to **mimic the hidden representations of deeper layers** (specifically fusing insights from the last few layers, e.g., L−3 onward — following prior findings that fusing the last ~3 layers' representations works best). This decouples the draft model's task from the final layer's exact behavior, letting a shallow exit still approximate deep-layer knowledge.
- **Training procedure (imitation learning):** the IN's output is passed through the (shared) LM head; training combines **cosine-similarity loss** (matching deeper hidden representations), **knowledge distillation**, and **cross-entropy loss** — the IN learns to produce a representation that "looks like" what a much deeper layer would have produced.
- **Inference:** the n-th decoder layer + IN generate draft tokens autoregressively; the full decoder verifies them non-autoregressively (standard self-speculative verification). Accepted tokens proceed; rejected tokens are corrected by the full model, and this correction signal also **guides further refinement of the draft model** (an online feedback loop between rejection and drafter improvement).
- Crucially, the full model's own weights/performance are preserved throughout — only the lightweight IN is trained, layered onto a frozen decoder-layer subset.

**3. Loss**

A **combined loss**: cosine-similarity loss (IN's output vs. deeper-layer hidden representations) + knowledge distillation loss (matching the full model's output distribution) + standard cross-entropy loss (next-token prediction). This is a genuine trained-component method (unlike SWIFT/CLaSp/LEAP/Vegas, which are training-free) — closer in spirit to LayerSkip/Kangaroo's "train intermediate layers to be predictive," but via an added lightweight network rather than modifying the base layers' training.

**4. Training data**

Standard VLM training/fine-tuning corpora appropriate to the evaluated tasks — image captioning and VQA-style datasets (evaluated on COCO, NoCaps, VisDial per the tables), used to fine-tune the IN (and lightly adapt the backbone) while keeping the full model's core performance intact.

**5. Final results**

Reported speedup of **1.55×–1.85×** over standard autoregressive VLM decoding (per ACL Anthology abstract), while maintaining output quality close to the full model, evaluated on COCO (captioning), NoCaps (zero-shot captioning), and VisDial (dialogue) benchmarks.

**Positioning note:** this is the VLM-specific, *trained* branch of the self-speculative family — contrast with LayerSkip/Kangaroo (LLM-only, train the base model itself) and SWIFT/CLaSp/CAS-Spec/LEAP/Vegas (LLM-only, training-free). FastVLM-SSD sits at the intersection your idea 2.6 (visual grounding for the drafter) and self-speculative decoding (no separate drafter model) — its imitation network is conceptually similar to DREAM's cross-attention-to-intermediate-features (Paper 5), but applied within a single shared model rather than between two separate models.

---

# Paper 21 — Aletheia# Paper 21 — Aletheia: Gradient-Guided Layer Selection for Efficient LoRA Fine-Tuning Across Architectures (arXiv:2604.15351)

*(Note: this paper is about training-time parameter-efficient fine-tuning, not inference-time speculative decoding — a different problem area from Papers 1–20. Flagging clearly since it's a genuine shift in topic.)*

**1. Challenges it's trying to solve**

Standard LoRA practice applies adapters **uniformly to every transformer layer**, regardless of whether that layer is actually relevant to the downstream task. This wastes compute/memory on adapter parameters and forward/backward passes through layers that contribute little to task adaptation. The question: can task-relevant layers be identified cheaply, so LoRA is applied only where it matters?

**2. Methods**

- **Lightweight gradient probe:** run just **5 forward-backward passes** on task data (no adapters yet) and measure **per-layer gradient norms** as a proxy for task relevance — layers with larger gradient signal are presumed more important for adapting to this task.
- **Selective LoRA application with asymmetric rank allocation:** apply LoRA adapters **only to the selected (high-gradient) layers**, skipping low-gradient layers entirely (eliminating their adapter forward/backward compute), with rank allocated asymmetrically across the chosen layers rather than a uniform fixed rank everywhere.
- Key empirical observation supporting the design: for many downstream tasks, **roughly half the layers behave as "pass-through" blocks** that minimally transform their input — precisely the layers gradient-probing identifies as low-relevance and skips.

**3. Loss**

Standard LoRA fine-tuning objective (task cross-entropy / instruction-tuning loss) applied only through the selected layers' adapters; the gradient-probe step itself uses the same task loss's gradients purely as a diagnostic signal (5 batches, no adapter training yet) before committing to a layer subset.

**4. Training data**

Instruction-following fine-tuning data (a single task domain across the paper — the authors flag this as a limitation) applied to 14 successful models across 8 architecture families spanning 0.5B–72B parameters, including dense and Mixture-of-Experts models (e.g., Mixtral 8x variants); evaluated for downstream behavior on MMLU, GSM8K, and HumanEval.

**5. Final results**

**15–28% training speedup** (mean 23.1%, statistically significant p<0.001) across 81 experiment rows, with a **100% per-model speed win rate** in the main campaign, and "bounded extra forgetting" / broadly matched downstream behavior on MMLU/GSM8K/HumanEval versus standard full-layer LoRA. The speedup comes specifically from eliminating adapter forward/backward computation in skipped layers (the frozen base model still processes all tokens through all layers — only adapter overhead is removed). A compute-matched follow-up analysis (Campaign 2) found the speed savings don't clearly translate into a downstream-quality advantage at matched compute — i.e., the win is efficiency, not accuracy. One documented failure case: Pythia/GPT-NeoX had fp16 instability affecting both Aletheia and standard LoRA (an architecture-specific issue, not a method failure).

**Positioning note:** this is a **training-time analog of your idea 2.1** — Aletheia identifies task-relevant layers via gradient-norm probing for *where to add adapters*, whereas your idea uses agreement/acceptance signal to decide *which layers to keep in a drafter*. Same underlying question (which layers matter, cheaply diagnosed), different downstream use (parameter-efficient fine-tuning vs. speculative-decoding draft construction). Their finding that "~half the layers are pass-through" for typical downstream tasks is a useful empirical prior if you want to bound your own search space before applying an acceptance-length criterion.

---

# Paper 22 — GradPruner# Paper 22 — GradPruner: Gradient-Guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs (arXiv:2601.19503)

**1. Challenges it's trying to solve**

Structured pruning methods speed up **inference** but typically require extra time/memory for training, knowledge distillation, or structure search on top of already-expensive fine-tuning — so you pay a training-cost tax to get an inference-cost win. GradPruner asks: can pruning be derived essentially **for free**, as a byproduct of the fine-tuning process you're already running, so both training *and* inference get faster simultaneously — with no separate pruning stage?

**2. Methods**

- **Initial Gradient Information Accumulation (IGIA) Matrix:** during the **early stage of LoRA fine-tuning** (before pruning), accumulate per-parameter gradients over some initial steps to build the IGIA-Matrix, which scores each layer's importance for the specific downstream task — layers with small cumulative gradient contribute little to the task and are pruning candidates.
- **Sparsify + merge, not just delete:** rather than crudely dropping the identified low-importance layers, GradPruner **sparsifies** them based on the IGIA-Matrix (keeping only the elements that matter) and then **merges** these sparsified remnants into the remaining layers — specifically only combining elements that share the same sign, which is a lightweight, training-free way to fold residual signal from pruned layers into kept layers rather than discarding it outright. This lets more layers be pruned than a naive "just delete the unimportant ones" approach would tolerate, since some of their contribution is preserved via merging.
- Result: a smaller, merged model is produced directly from the byproduct of the fine-tuning run already underway, without an added distillation phase, knowledge-distillation loss, or dedicated structure search.

**3. Loss**

The **underlying fine-tuning loss stays exactly whatever it already was** (standard LoRA cross-entropy fine-tuning on the downstream task) — GradPruner adds no new loss term. Its contribution is entirely in how the **existing gradients**, already computed as a normal part of training, are accumulated (IGIA-Matrix) and used post-hoc to prune and merge, not in any new training objective.

**4. Training data**

Downstream fine-tuning datasets — evaluated on **two LLMs and eight downstream datasets** (their comparisons involve models like Llama3.1-8B and Llama3.2-3B). The IGIA-Matrix is computed from the gradients naturally produced during that same fine-tuning run — no separate calibration or distillation corpus is required.

**5. Final results**

**40% parameter reduction with only a 0.99% accuracy decrease** on downstream tasks — outperforming compared structured-pruning baselines (APT, SAT, LLM-Pruner-style gradient-based structural pruning, LaCo layer-merging, and others) across the two-model/eight-dataset evaluation. Notably, some pruned models (e.g., pruned Llama3.1-8B) **outperform directly fine-tuned versions of Llama3.2-3B** — i.e., the pruned-and-merged larger model beats a smaller model fine-tuned from scratch on the same task, suggesting the merging step retains genuinely useful capacity rather than just discarding it.

**Positioning note:** GradPruner is the closest published precedent to your idea 2.1 (Acceptance-Guided Layer Shearing) among everything we've covered — it prunes layers using a gradient-based importance signal computed *during* an existing training process, then **merges** (rather than discards) pruned-layer information into survivors. The two open gaps relative to your idea: (a) GradPruner's criterion is task-loss gradient magnitude, not speculative-decoding acceptance length, and (b) it isn't evaluated for inference-time draft/target agreement at all — it's a general efficient-fine-tuning method, not built for the drafter-construction setting. Your contribution would be porting this gradient-guided-prune-and-merge recipe to the acceptance-length objective specifically, which nothing in your list does yet.


# Paper 23 — Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting (arXiv:2404.18911, NeurIPS 2024, Huawei Noah's Ark)

**1. Challenges it's trying to solve**

Speculative decoding accelerates LLM inference losslessly, but training a separate draft model to reach a satisfactory acceptance rate is costly. Self-speculative approaches (like fixed early-exit — LayerSkip) avoid a second model, but they surface a subtler problem: once you skip most of the depth, **the shallow sub-network's own inference latency stops being negligible** relative to the target — every drafted token still costs a real forward pass through the shallow part, and if the shallow network is weak, you spend a lot of steps drafting tokens that get rejected anyway. So the real challenge becomes: how do you raise acceptance rate *while also* minimizing the number of costly drafting steps the shallow model has to take, especially on hard tokens where it's likely to be wrong regardless?

**2. Methods**

**Double early exiting** — two separate exit mechanisms working together:

- **Exit #1 (architectural / "which sub-network is the drafter"):** a **fixed shallow sub-network** (the first few layers of the target) serves as the self-draft model, with the rest of the target's layers serving as the verifier — same basic self-speculative structure as LayerSkip. But instead of relying on the raw shallow layers' hidden state directly, Kangaroo adds a **lightweight trained adapter module** on top of the sub-network to bridge the representational gap between the shallow sub-network and the full model — the adapter is just **one multi-head attention layer + two normalization layers**, deliberately minimal. Training follows the Medusa/EAGLE-style recipe (train the adapter against target model's outputs), but critically the shallow *base* layers themselves are frozen/untouched — only this small adapter is trained. Because draft and target share the same early layers, they also **share KV-cache and computation** for that shared prefix — the only extra deployment cost is the tiny adapter.
- **Exit #2 (dynamic, at the token level, during drafting):** to avoid wasting compute on tokens the shallow model is unlikely to get right, Kangaroo **halts the small model's further autoregressive drafting mid-step** whenever its confidence for the current token drops below a threshold — i.e., it stops drafting further tokens in this round rather than pushing on through a token it's unsure about. This second exit applies in both single-sequence and tree-decoding verification settings, trimming wasted drafting steps on hard tokens specifically.

**3. Loss**

Adapter-only training loss following the Medusa/EAGLE training recipe: cross-entropy against the target model's token distribution (the shallow sub-network + adapter learn to approximate what the full target would predict). The frozen shallow layers are untouched — this is architecturally closer to EAGLE (small trained module bridging to a frozen backbone) than to LayerSkip (which retrains the base model's layers with dropout + multi-depth loss).

**4. Training data**

Following EAGLE/Medusa-style training data practice — instruction/dialogue data used to train only the small adapter (67M parameters, vs. Medusa-1's 591M) — the base target LLM stays entirely frozen throughout.

**5. Final results**

Under single-sequence verification, up to **1.68×** speedup on Spec-Bench; with tree-based decoding, walltime speedups up to **2.04×**, both while **outperforming Medusa-1 with 88.7% fewer additional parameters** (67M vs 591M) — a strong parameter-efficiency result, since the "drafter" is mostly free (shared shallow layers + shared KV-cache), with only a tiny adapter as genuinely new capacity. The paper explicitly notes that raw compression rate (how few layers/params you use) doesn't reliably predict acceptance rate — the confidence-based second exit is what actually controls the useful trade-off between drafting cost and acceptance.

**Positioning note:** Kangaroo is the missing link between LayerSkip (Paper 9) and the SWIFT/CLaSp/CAS-Spec/LEAP lineage (Papers 15–18) — architecturally it's LayerSkip's shared-prefix idea (draft = fixed shallow layers of the target, full cache sharing), but instead of retraining the base model with layer dropout, it adds a **tiny trained adapter** (EAGLE-style) on top of the frozen shallow layers, and instead of a fixed drafting length, it adds a **confidence-based early stop within the draft phase itself** — the "double" exit. This is directly relevant to your idea 2.1: Kangaroo shows that a frozen shallow prefix + small trained bridge can beat far larger dedicated draft models, which supports top-truncated shearing (rather than scattered pruning) as the more practical starting point, since it gets KV-cache sharing for free exactly as Kangaroo does.



# Token Compression Methods for Visual Tasks — 3 Papers

---

## Paper A — FastV: An Image is Worth 1/2 Tokens After Layer 2 (arXiv:2403.06764, ECCV 2024)

**1. Challenges it's trying to solve**

Vision-language models process hundreds to thousands of visual tokens per image, and because Transformer cost scales roughly quadratically with sequence length, this visual token count dominates inference FLOPs — often far more than the accompanying text. The question FastV asks is whether all of that visual computation is actually being used by the model in later layers, or whether it's wasted.

**2. Methods**

FastV starts from an empirical observation: visual signal redundancy leads image-related, instruction-specific features to aggregate onto certain "anchor" tokens through self-attention in the shallow layers, and these anchor tokens are rarely image tokens themselves — so in deep layers, attention concentrates on the anchors and largely abandons the original image tokens. Based on this, FastV is a **training-free, plug-and-play pruning method**: computation proceeds normally up through a chosen early layer, and beyond that layer image tokens are re-ranked by the average attention score they've received; tokens below a threshold are discarded from all subsequent layers. Because tokens are dropped outright (not just masked in attention), this also skips their FFN compute in deep layers, not just their self-attention cost.

**3. Loss**

None — FastV requires no training or fine-tuning; it is a pure inference-time intervention on a frozen pretrained model.

**4. Training data**

Not applicable (training-free).

**5. Final results**

On LLaVA-1.5-13B, filtering 50% of image tokens after layer 2 causes no loss in average performance across a combination of benchmarks, and overall FastV achieves up to a 45% FLOPs reduction on LLaVA-1.5-13B without sacrificing performance across a wide range of image and video understanding tasks, and can compress a 13B model's FLOPs below a 7B model's while still outperforming it.

---

## Paper B — LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models (arXiv:2403.15388, ICCV 2025)

**1. Challenges it's trying to solve**

Same root problem as FastV — LMM visual token counts are large and costly — but PruMerge asks whether a *fixed* pruning ratio is even the right frame, since different images carry very different amounts of information density (a text-heavy screenshot vs. a plain sky photo shouldn't need the same number of tokens).

**2. Methods**

PruMerge operates at the vision-encoder output, before the LLM, and is training-free by default. It exploits sparsity in the CLIP ViT's own attention: most spatial visual tokens have near-zero attention with the [CLS] token, so the unpruned/kept tokens are first selected based on their similarity to the class token and to other spatial tokens, giving an **adaptive** budget per image (dense images keep more tokens, simple ones keep fewer). Rather than simply throwing away the discarded tokens, PruMerge then clusters the pruned tokens by key similarity and merges them back into the retained set to supplement their information — a prune-then-merge combination, not pruning alone. A "PruMerge+" variant additionally adds spatially uniform sampling from initially-discarded regions, guided by the distribution of outlier tokens, to give more comprehensive coverage and reduce performance loss under aggressive compression.

**3. Loss**

Training-free in its base form. Where fine-tuning is used (to adapt the LLM to the reduced token budget), it follows standard LMM instruction-tuning cross-entropy loss — the paper's core contribution is the selection/merge mechanism, not a new objective.

**4. Training data**

None required for the base method; any fine-tuned variant reuses the standard LLaVA instruction-tuning data.

**5. Final results**

Applied to LLaVA-1.5, the approach compresses visual tokens by 18× on average while achieving comparable performance across diverse visual question-answering and reasoning tasks, and more broadly reduces LMM prefill FLOPs by roughly 4–10× while maintaining comparable performance.

---

## Paper C — VisionZip: Longer is Better but Not Necessary in Vision Language Models (arXiv:2412.04467, CVPR 2025)

**1. Challenges it's trying to solve**

VisionZip pushes back on the implicit assumption that "more visual tokens = better performance." It targets the same redundancy problem as FastV/PruMerge but designs specifically to avoid the two common failure modes of naive pruning: (a) losing small-but-important details when only the most-attended tokens are kept, and (b) a train/inference mismatch when token count is cut sharply.

**2. Methods**

VisionZip is **text-agnostic** and works at the vision-encoder side, with training-free, fine-tuning, and train-from-scratch variants. It has two stages: first select dominant tokens — those receiving significant attention and aggregating most of the image information — then, to avoid missing small but potentially important details, merge the remaining tokens by similarity into "contextual tokens" that supplement the dominant set. The attention signal used for stage one is the same CLS-token attention pattern PruMerge exploits: in early layers attention is broadly distributed across the image, but by middle layers it suddenly converges onto a few tokens, and in deeper layers attention and information concentrate on this small dominant set. For the fine-tuning variant, because the input token count drops sharply, there's a slight misalignment between the reduced visual input space and the LLM's expected space, which a lightweight projector fine-tuning step corrects.

**3. Loss**

Training-free mode: none. Fine-tuning mode: standard next-token cross-entropy on the projector only (analogous to adapter-style tuning), correcting the input-distribution shift caused by the reduced token count rather than teaching new visual understanding.

**4. Training data**

Training-free mode needs none. The fine-tuning variant reuses standard LLaVA-style instruction-tuning data, applied briefly to the projector.

**5. Final results**

Reported on LLaVA-1.5 and LLaVA-NeXT-class models, VisionZip retains a small fraction of the original 576 visual tokens (down to the tens of tokens) while keeping accuracy close to the full-token baseline, and the fine-tuned variant narrows the gap further versus training-free pruning at the same aggressive budget — consistent with follow-up work benchmarking against it (e.g., OccamToken, GreedyPrune) as one of the standard strong training-free baselines in this space.

---

**Positioning note:** These three sit on a clear lineage. FastV prunes *inside the LLM decoder*, using cross-modal attention collapse in deep layers as its signal — closest in spirit to Kangaroo's "confidence-gated early exit," except the gate here is attention mass, not token-prediction confidence. PruMerge and VisionZip both operate *at the vision-encoder boundary*, using CLS-token attention sparsity to select dominant tokens, then differ mainly in whether/how they recover information from the discarded tokens (similarity-based merge in both, plus PruMerge+'s uniform-sampling supplement). None of the three train a separate drafting network the way Kangaroo does — they're all closer to "prune what's redundant" than "predict then verify," which is the key structural difference between token-compression-for-vision and self-speculative-decoding-for-text.



SparseVLM: Visual Token Sparsification for Efficient Vision-Language Model Inference (arXiv:2410.04417, ICML 2025)

**1. Challenges it's trying to solve**

Existing token-reduction methods either need a trained pruning network with extra training data, or — when they do prune during LLM decoding — ignore the guidance available from the text/instruction tokens entirely, which the authors argue contradicts the multimodal nature of the task: the model should attend to different image regions (foreground vs. background) depending on what the question actually asks.

**2. Methods**

SparseVLM is **training-free and text-guided**. Instead of scoring visual tokens purely by internal image-encoder attention (as FastV/PruMerge/VisionZip do), it selects the visual-relevant *text* tokens and uses those to rate the significance of each visual token within the self-attention matrix extracted from the VLM — i.e., pruning decisions are conditioned on the prompt, not just the image. Pruning is progressive across layers rather than a one-shot cut, and the sparsification ratio per layer is set adaptively via a rank-based strategy rather than a fixed global ratio. Pruned tokens aren't simply discarded: a token-recycling mechanism compresses them into more compact representations so some of their information is retained rather than fully lost.

**3. Loss**

None — no additional parameters or fine-tuning; the method operates entirely on a frozen VLM's existing attention matrices.

**4. Training data**

Not applicable (training-free).

**5. Final results**

LLaVA equipped with SparseVLM achieves a 54% reduction in FLOPs and a 37% decrease in CUDA latency while maintaining 97% of its original accuracy; the method also extends to video (VideoLLaVA) by sparsifying across the temporal dimension.

---

**Updated positioning note:** SparseVLM's distinguishing move relative to PruMerge and VisionZip is *where the pruning signal comes from* — those two score tokens using only the vision encoder's internal (CLS-token) attention, text-agnostic; SparseVLM instead lets the instruction/question steer which visual tokens survive, layer-by-layer, with an adaptive per-layer budget instead of one global ratio. All three remain training-free pruning/merging methods rather than trained-drafter approaches, so the same caveat as before applies: none of them are the "trained small predictor + verify" structure that Kangaroo uses for text.


# Proposed Method: Draft Model Construction via Layer Selection and On-Policy Self-Distillation

## Motivation

Training-based speculative decoding (EAGLE family) achieves the strongest speedups but has two structural weaknesses. First, the drafter is a **freshly initialized module** — it discards the target's internal circuitry and must relearn prediction from scratch, capping acceptance length. Second, methods with larger, more capable drafters pay a **memory and deployment cost** (a second model artifact to store, serve, and keep in sync). Self-speculative methods (LayerSkip, Kangaroo, SWIFT/CLaSp/LEAP) avoid both by reusing the target's own layers, but are limited to contiguous shallow prefixes or training-free inference-time skipping, leaving a large acceptance gap versus trained drafters (≈1.3–2.3× vs. 2.7–3.6×).

**Goal:** EAGLE-level (or better) speedup and acceptance length, at self-speculative-level memory overhead — for VLM targets, where visual tokens make drafting disproportionately expensive.

**Core idea:** construct the drafter *from the target itself* — select an importance-ranked, possibly non-contiguous subset of the target's layers, adapt it with lightweight LoRA/MLP modules (base weights untouched and shared), heal it with on-policy self-distillation, and feed it compressed visual input.

## Method

**Stage 1 — Layer selection (criterion: one of two options, ablated).**
(a) *SFT-gradient probe:* brief fine-tuning of the full model on task data; rank layers by accumulated gradient magnitude. (b) Alternatively, use the gradient for the agreement loss. Entropy- and attention-sum-based rankings as secondary baselines. Depth-only selection. 

**Stage 2 — Parameter-efficient adaptation.**
LoRA + small MLPs on kept layers only; frozen base weights are shared with the target. KV-cache handling, (1) KV-invariant LoRA — adapt q/o/MLP but freeze k/v projections so the target's cache is directly attendable; (2) learned cache projection (Cache-to-Cache) mapping target KV into drafter space.

**Stage 3 — On-policy self-distillation.**
Supervision from the frozen target model during training, evaluated on the drafter's own rollout states including verification-rejected positions (Draft-OPD; EAGLE-3's training-time test). Loss stack:
- *Primary:* divergence on logits — JS as default (balances mode-seeking/greedy vs. mass-covering/sampled acceptance).
- *Attention alignment:* drafter attention over visual tokens matched to the target's via **group matching** — each kept layer matched to the averaged attention of the teacher span it absorbs (kept layer + preceding skipped layers), cosine distance.
- *Intermediate hidden-state supervision:* normalized-MSE matching at kept layers via the inherited identity mapping

**Stage 4 — Visual token compression at the drafter input.**
The full target prefills first (required for verification anyway), so its visual features exist before drafting; the drafter consumes a compressed selection rather than raw visual tokens (SmolVLM: small models want fewer visual tokens). Options: (i) resampler over target visual features, optionally pool-anchored elastic queries (PARCEL) for budget-adaptive compression; (ii) prune-then-resample (attention-scored pruning first); (iii) zero visual tokens, relying on target text-position hidden states (HiViS limiting case). An MLP down-projects fused target features before injection into drafter layers (ViSpec adaptor / EAGLE-3 fusion).

**Inference pipeline.** Target prefill → drafter drafts (kept layers + adapters, compressed visual context, shared/regenerated KV per Stage-2 choice) → target verifies with standard lossless rejection sampling → repeat.

**Evaluation plan.** Metrics: acceptance length τ, relative drafter cost c, wall-clock speedup, drafter memory overhead, per-token-type acceptance (visual-grounded vs. function tokens, MSD diagnostic). Baselines: EAGLE-2/3, ViSpec, HiViS (trained); CLaSp/LEAP (training-free floor); MASSV (independent small drafter). Targets: one LLaVA-family and one Qwen-VL-family model.


{
  "architectures": [
    "Eagle3LlamaForCausalLM"
  ],
  "model_type": "llama",
  "target_model_type": "smolvlm",
  "modal_type": "VLM",
  "torch_dtype": "bfloat16",
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 576,
  "image_token_id": 49190,
  "initializer_range": 0.041666666666666664,
  "intermediate_size": 1536,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "num_attention_heads": 9,
  "num_hidden_layers": 1,
  "num_key_value_heads": 3,
  "pad_token_id": 2,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_interleaved": false,
  "rope_parameters": {
    "rope_theta": 100000,
    "rope_type": "default"
  },
  "tie_word_embeddings": true,
  "transformers_version": "4.51.0",
  "use_cache": true,
  "vocab_size": 49280,
  "draft_vocab_size": 32000
}


