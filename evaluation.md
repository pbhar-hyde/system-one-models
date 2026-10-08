# Evaluation and current open models

**Tags:** #system-one #eval #calibration
**Parent:** [index](INDEX.md)

## Evaluation stack

No single board answers the deployment question. Use:

1. a production-shaped private holdout for thresholds and coverage at your error budget;
2. [Decision Index 0.3](#decision-index-03) for broad capability and hidden-domain transfer;
3. [JevBench v1.6.1](#jevbench-v161) for typed probability quality, latency, and cost;
4. targeted invariance and long-context controls.

Scores across the two boards are not numerically comparable. Strength on both is better evidence than optimizing one public protocol.

## Decision Index 0.3

[Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) evaluates whether a typed-decision interface preserves broad model capability; its public [reproduction kit](https://github.com/apolinario/decision-index) rebuilds the public suite. Version 0.3 contains 110,201 requests and counts 37 public benchmarks across knowledge/reasoning, language understanding, retrieval/classification, tools/automation, and arts/human taste.

Its full score is **20% public benchmarks, 50% private same-skill tests, and 30% private new-domain tests**. Each task keeps an appropriate native metric—accuracy, macro-F1, nDCG, best-move accuracy, and others—then chance-corrects it before area weighting. Unsupported or unanswered requests count as wrong.

The hidden sets make it a stronger transfer test than a fully public suite, but the full score is not independently reproducible: private items, labels, per-item outputs, and scale-mapping details are withheld. It has no explicit calibration, cost, or speed axis; latency is an eligibility gate rather than part of the score.

In the 7 Oct 2026 snapshot, the full-score frontier is:

- **Perplexity Decider v1.1 27B (62.75):** Qwen3.8-27B full fine-tune, lifted causal mask in full-attention layers, and a separate 255-way decision head. The evaluated path is not ordinary causal inference; access is currently gated.
- **Fastino GLiDE no-thinking 28B (60.21):** second-place system, but weights were still announced as forthcoming.
- **Jev (60.11):** closed reference.
- **Torchcast Decision 27B (59.91):** Qwen3.8-27B LoRA, one-pass shared-question serving; non-commercial weights.
- **deck-31B (59.03), Kev-27B (58.78), and Quyet-1.0-Large (58.71):** nearly tied systems with respectively frozen letter logits, a trained pointer head, and a LoRA letter readout.
- **Blink v0.3, Mapika's stock Gemma-4-31B readout, Rune, GEV, Xor, and Decision 2.0 Vega** form the next group.

The area leaders differ: Quyet leads knowledge, Perplexity leads language and retrieval, Fastino leads tools, and Torchcast leads arts. That disagreement is useful—there is no universally best checkpoint.

## JevBench v1.6.1

This note uses [JevBench v1.6.1](https://benchmarkheaven.com/jev-models), published 6 Oct 2026. The user's filtered view sets cost ≤1.6× Jev and median latency ≤1.1× Jev. That means:

- cost ≤$0.052 per 1,000 decisions;
- adjusted p50 latency ≤0.68 s;
- 74 of 124 measured systems qualify.

These caps only decide eligibility. **Capability** is still the mean of Intelligence and Calibration. The official **Composite** is an equal-weight harmonic mean of Intelligence, Calibration, Speed, and Cost, with extra penalties for weak axes.

**Intelligence** is chance-corrected competence, computed within each difficulty/type tier:

\[
100\max\left(0,\frac{\text{accuracy}-\text{chance}}{1-\text{chance}}\right)
\]

**Calibration** rewards both reliability and fidelity to the target distribution. For Choice, the core terms are \(100(1-2\,\mathrm{ECE})\), clipped at zero, and \(100(1-\mathrm{mean\ TVD})\), then averaged. Noul and Score use type-appropriate Brier/RPS and ECE diagnostics. A system can therefore be accurate but lose Capability by assigning poor probabilities.

Under the custom caps, the Capability leaders are:

| Rank | System | Backbone and readout | Intelligence | Calibration | Capability | Estimated $/1k | p50 |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Quyet-1.0-Large | Gemma-4-31B, rank-16 LoRA, letter logits | 73.4 | 90.0 | 81.7 | $0.045 | 0.38 s |
| 2 | deck-31B | frozen Gemma-4-31B, FP8, letter logits | 73.0 | 82.2 | 77.6 | $0.048 | 0.40 s |
| — | Jev 1.13.0 | closed API, reference only | 63.6 | 90.6 | 77.1 | $0.032 | 0.24 s |
| 3 | H2O-Lightning-4B v1.1 | Qwen3.5-4B fine-tune, native letter logits | 60.0 | 90.0 | 75.0 | $0.021 | 0.21 s |
| 4 | Surogate Rune v3 | Gemma-4-26B-A4B full fine-tune | 56.1 | 90.9 | 73.5 | $0.048 | 0.38 s |
| 5 | Xor 26B-A4B | Gemma-4-26B-A4B post-train and weight merge | 58.7 | 86.1 | 72.4 | $0.044 | 0.27 s |
| 6 | Torchcast Decision 12B | Gemma-4-12B LoRA, letter logits | 60.5 | 82.9 | 71.7 | $0.028 | 0.22 s |
| 7 | Jev-Omni | Gemma-4-12B fine-tune and 256-way head | 55.5 | 87.0 | 71.3 | $0.029 | 0.46 s |
| 8 | Winnow-12B Q8 | Gemma-4-12B merged LoRA, GGUF | 59.5 | 83.0 | 71.2 | $0.028 | 0.38 s |
| 9 | Cygnet | frozen Gemma-4-12B, letter logits | 54.8 | 87.0 | 70.9 | $0.028 | 0.22 s |

The official Composite tells a different, useful story. H2O-Lightning-4B is #1 at 72.5 because it is cheap and fast while retaining good intelligence and calibration. Quyet is #2 at 71.4 and is the quality-first choice. Jev is the unranked reference at 71.5.

The main conclusions:

1. Gemma 4 is the strongest open backbone family in this release.
2. Frozen 31B and 12B models are real competitors. Decision fine-tuning is not automatically beneficial.
3. Calibration and the serving shim can move rank as much as training.
4. H2O-Lightning-4B is the best compact system on this board.
5. This board favors letter readouts; Decision Index's pointer and decision-head leaders prevent a general conclusion that custom heads are only niches.

## What JevBench does and does not prove

JevBench is useful because the maintainers run open weights themselves, include sealed decisions, and normalize serving conditions. It is still one benchmark.

- The current release has 1,500 decisions per self-hosted system. Only system-level sealed aggregates are public.
- Most self-hosted costs are estimates, not bills. They combine recorded token use with a base-model price assumption.
- Latency is adjusted for the evaluator's hardware. It does not predict your batch size, GPU, context length, or network.
- Public items influence some model and prompt selection. Torchcast says this explicitly; Cygnet and deck use temperatures fitted elsewhere.
- Rank moved substantially from v1.4 through v1.6. Always quote the release and date.
- Capability ignores speed and cost after applying the eligibility caps. Composite includes all four axes.
- The model cards vary sharply in reproducibility. deck and Cygnet are fully inspectable recipes; Quyet and H2O publish weights but not full training runs; Rune and Winnow keep training data private.

Use JevBench to compare calibration and deployment economics under its protocol. Use Decision Index for breadth, then a production-shaped holdout to choose and threshold a system.

## What to measure on your data

Measure per question type and source:

- **Accuracy or macro-F1.** For `score`, add MAE of the expected level.
- **NLL or Brier score.** These score the entire distribution.
- **ECE with bin counts.** A low aggregate ECE can hide an empty middle and a crowded 0.9–1.0 bin.
- **AURC and coverage at an error budget.** This is the fraction of work you can automate while keeping error below ε.
- **Confident errors:** `P(wrong | top probability ≥ threshold)`.
- **Option-order stability.** Re-run each item under several permutations.
- **Unknown-evidence behavior.** Remove the decisive fact and check whether confidence falls or the model abstains.
- **Context robustness.** Move the same evidence from the start to the middle and end of a realistic document.
- **Question isolation.** A sibling question must not change the answer unless your design intentionally conditions on it.

Do not threshold `confidence` as if it were `P(correct)`. In the TypeSafe-compatible response it usually measures concentration relative to a uniform distribution. Threshold the calibrated probability of the chosen answer.

For `noul`, do not assume 0.5 is optimal. In the [37-dataset Jev study](https://arxiv.org/html/2609.37647v1), a tuned threshold raised micro-F1 on UNFAIR-ToS from 0.50 to 0.75.

## Controls worth keeping

- **Frozen backbone:** same prompt and readout, no fine-tuning. deck and Cygnet show why this control is mandatory.
- **Generated JSON:** compare one-token logits with schema-constrained generation. [mini-jev](https://github.com/r-ms/mini-jev) found 0.907 vs 0.909 on CLINC150 while the logit path was much faster.
- **Written confidence:** ask the model to print probabilities. In mini-jev this collapsed to 0.346 accuracy and chose the first option 62% of the time. Never use this as the baseline.
- **Option permutation:** report accuracy, probability drift, and threshold-crossing flips.
- **Question ablation:** withhold the question while keeping the state. Jev falls to chance.
- **Evidence ablation:** delete or contradict the decisive fact. A model that stays confident is recalling a prior, not reading the state.
- **Long context:** leaderboard prompts are not a substitute for your policy documents or agent traces.

## Models to run first

**H2O-Lightning-4B** is the compact JevBench candidate. It is Apache-2.0, uses about 9 GB of weights, runs on stock vLLM with an open shim, and now supports images. On the public 231 items its author reports 205 correct. It is the right small baseline, but its JevBench lead is not a substitute for a Decision Index or production-domain result.

**Quyet-1.0-Large** is the cross-board quality candidate. It leads JevBench Capability and Decision Index knowledge, needs an 80 GB-class deployment, supports only ten options, and exposes just 8k context in its packaged decision prompt.

**deck-31B** is the most informative cross-board baseline: no training, transparent code, strong intelligence. Use it to determine whether a fine-tune adds anything beyond Gemma-4-31B.

**Cygnet** is the equivalent frozen 12B control. It is simple, reproducible, and supports a production shim with grouped reads for large menus. A 48 GB GPU is the documented setup.

**Kev-27B** and **Torchcast Decision 27B** are Decision Index candidates. Kev uses a trained pointer head and publishes unusually useful training details; Torchcast uses a Qwen3.8-27B LoRA and custom one-pass serving. Torchcast's weights are CC BY-NC 4.0.

**Perplexity Decider v1.1** is the Decision Index leader and the main learned-head reference. Reproduction requires its non-causal full-attention path, not stock generation; the repository is currently gated. **Fastino GLiDE no-thinking** is not yet downloadable, so neither is the first reproducible baseline.

**Torchcast Decision 12B** is the strongest measured 12B by JevBench Intelligence, but its weights are also CC BY-NC 4.0. Use it for research or non-commercial deployments.

**Winnow-12B Q8** is the practical 16 GB option. It combines typed decisions, chat, images, GGUF serving, and a 65k tested context. Its training pipeline is private.

**Xor** and **Rune** are the larger MoE choices. Xor publishes detailed merge provenance and has a 131k served context; Rune supports vision and optional reasoning. Both require roughly 50 GB of weights, and neither publishes a complete training recipe.

**Jev-Omni** is the broad modality option: text, image, audio, and video. It requires CUDA and has no public training pipeline.

**Imajev-4B** remains useful when photo-versus-record checks and explicit `unknown_probability` matter. See [imajev](imajev.md). Its current text-only JevBench position is not the reason to choose it.

**Mapika/decider** is both a competitive frozen 31B readout on Decision Index and the broadest reproducible training toolkit. Laya and Verdict remain specialist encoder toolkits.

## Other evaluations worth keeping

[Evaluating and Benchmarking Jev](https://arxiv.org/html/2609.37647v1) covers 37 full datasets and 346k requests. Jev beats the tested Qwen model on 27 of 37 datasets and Gemma-4-E4B on all 37. It also shows where the class struggles: low-resource languages, fine-grained labels, noisy labels, and rubric judgments.

Kev's frozen transfer suites remain useful for regression testing during fine-tuning because they separate trained sources from new sources and publish the harness and reports.

The separate `jevbench` coherence package checks invariances and probability laws without gold labels. It is a diagnostic, not a capability benchmark: a uniform predictor can score highly while deciding nothing.
