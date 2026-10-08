# Fine-tuning

**Tags:** #system-one #finetune #calibration #rlcd
**Parent:** [index](INDEX.md) · mechanism in [architecture](architecture.md)

## First ask whether you need it

JevBench and Decision Index both make a strong case for starting with frozen logits.

- **deck-31B** is frozen Gemma-4-31B-it, nearly matches the JevBench intelligence leader, and remains near the Decision Index frontier.
- **Cygnet** is frozen Gemma-4-12B-it and remains competitive with trained 12B models.
- Mapika's stock Gemma-4-31B readout ranks close to fully fine-tuned 26–31B systems on Decision Index 0.3.

Fine-tuning is justified when it improves your own holdout, not because the interface is called a decision model. Run the base model with an option-letter readout, fit a temperature, and keep that as the control.

Training is most useful for:

- a private taxonomy or policy that the base model has never seen;
- reliable abstention on missing evidence;
- short models, where task data can compensate for weaker base knowledge;
- multimodal checks with a fixed operational shape;
- calibration and coverage on your production distribution.

It is less useful when the errors are missing world knowledge or multi-step reasoning. A larger frozen backbone may fix those more cheaply than a domain LoRA.

## What is disclosed by the leading models

The best checkpoints are not necessarily the best recipes to copy.

**H2O-Lightning-4B** is the leading compact system and the official JevBench Composite leader. Its card describes a held-out lockbox, exact and near-duplicate screening, gap-driven data collection, human labels or agreement among strong independent models, and calibration as a release gate. It does **not** publish the dataset, optimizer, learning rate, or full training script.

**Quyet-1.0-Large** discloses a merged rank-16 LoRA on Gemma-4-31B-it and fitted temperatures. Its data and training recipe are not public.

**Perplexity Decider v1.1 27B**, the Decision Index leader, is a full fine-tune of Qwen3.8-27B with the causal mask lifted in full-attention layers and a separate 255-way decision head. The evaluated inference path is specialized, and the weights are currently gated; the full data pipeline is not public.

**Kev-27B** is the most informative trained-head system near the Decision Index frontier. It is a one-epoch Qwen3.8-27B full-weight fine-tune with a 2.6M-parameter pointer head over 145,840 records; the dataset manifest and training details are public, but the full corpus is not.

**Torchcast Decision 12B/27B** gives a clear partial recipe: LoRA, mixed hard labels and teacher option distributions, procedural decisions, Open-Jev data, and public train splits. The 27B Qwen3.8 model is near the Decision Index top; the 12B Gemma model is strong on JevBench. Exact optimization settings are not published, and the weights are non-commercial.

**Winnow-12B** is a merged rank-32, alpha-64 LoRA over attention and MLP projections. Its synthetic, teacher-supervised, and semantic-task data are private.

**Xor 26B-A4B** publishes its model composition rather than its training run. Two rank-32 LoRA lineages are merged; one also adapts the router and every expert with ranks based on routing frequency. Their average is blended 15% back toward the base. The merge scripts are public, but the adapters and data are not.

**Imajev-4B** is unusually transparent for multimodal training: open base, adapter and readout, staged data counts, optimizer family, learning rates, replay, soft targets, and compute are documented in its [model card](https://huggingface.co/mohit67890/imajev-4b). It is not a current text-leaderboard leader, but it is a useful recipe for photo-plus-record decisions and learned `unknown`.

## The reproducible general recipe

[Mapika/decider](https://github.com/Mapika/decider) is currently the broadest public training implementation. Its builders convert about 95 public datasets into typed decisions. The reproducible mixture is 1.47M examples and 455M tokens; one epoch takes about 5.3 hours on a GH200, followed by roughly 45 minutes of evaluation.

The recipe is intentionally ordinary:

1. Start from a strong base or instruct model.
2. Render the state, instructions, and options exactly as they will appear at inference.
3. Read option-label logits at one answer position.
4. Train with cross-entropy on hard labels or soft cross-entropy on teacher distributions.
5. Permute option order.
6. Fit temperatures on a separate calibration split.
7. Select checkpoints on a private lockbox, not the public benchmark.

Mapika's own results also show the limit: its trained 2B, 4B, and 35B models trail stock Gemma-4-31B with the same readout on broad knowledge. Training improves the task distribution; it does not replace a capable backbone.

Treat the prompt, answer cue, tokenizer binding, and index scheme as part of the model. Verify that every chosen label is one token at the answer boundary, marginalize duplicate token IDs, and keep the rendering byte-identical across training and serving. The full contract is in [architecture](architecture.md#prompt-contract).

## Data that transfers

One record should contain the same state, question, options, and rendering that the server will receive.

```jsonl
{"state": {"subject": "Charged twice", "body": "I see two charges for order #4411."},
 "questions": {
   "team": {"type": "choice", "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments and refunds", "shipping": "Delivery problems"},
            "label": "billing"},
   "angry": {"type": "noul", "instructions": "Is the customer angry?", "label": false},
   "priority": {"type": "score", "instructions": "How urgent?",
                "criteria": ["low", "normal", "high"], "label": 1}}}
```

Use:

- **Hard labels** for verifiable facts and policy decisions.
- **Soft targets** when several answers are defensible. Torchcast, Imajev, and Kev-27B all use teacher distributions.
- **Minimal pairs** where the decisive evidence is added, removed, or contradicted.
- **Unknown examples** where no listed answer follows from the state. Do not use “none of the above” as a substitute if abstention is operationally distinct.
- **Long-context examples** with the evidence buried in realistic noise.
- **Replay** from the general mixture during domain continuation to limit forgetting.

Shuffle answer order during training and evaluation. Otherwise the model can improve by learning slot priors.

Split by source, template, and scenario family—not only by row. Synthetic variants of the same scenario on both sides of a random split are leakage.

## A practical domain continuation

Kev still provides the cleanest small-data continuation workflow:

```bash
uv run python -m kev.train \
  --data train.jsonl \
  --base Qwen/Qwen3.5-4B-Base \
  --init_from jaredpalmer/kev-4b \
  --epochs 2 --lr 2e-5 --batch 1 --accum 8 \
  --dtype bf16 --checkpointing 1 --device cuda \
  --out runs/mine

uv run python -m kev.benchmark \
  --run runs/mine --data heldout.jsonl --out runs/mine-eval
```

Continue from the released checkpoint. In Kev's support-tool experiment, training from the base damaged the general evaluation, while `--init_from` preserved it and improved the new domain. Their examples suggest that roughly 400 records can be noise; 1k–5k well-labelled records produce measurable in-domain gains.

For a new model family, prefer a standard LoRA over a new head until the letter-logit control fails. Decision Index shows that pointer and 255-way heads can lead, but not that they beat a matched LM-head baseline because backbone, data, and inference also changed. Compare readouts directly and test index permutations; see [improvement directions](improvement-directions.md).

## Calibration is a separate stage

Fine-tuning does not make probabilities trustworthy by itself.

Fit temperature on data excluded from both gradient updates and checkpoint selection. Use separate temperatures by question type when the holdout supports it; Noul often needs a very different correction from Choice and Score. If calibration changes by option count, fit a smooth function or bucket only when each bucket has enough samples.

Report:

- NLL or Brier score over the full distribution;
- ECE with bin counts;
- option-order flip rate;
- AURC and coverage at the error budget you will ship;
- accuracy and calibration on unseen sources;
- false abstention and missed-abstention rates.

Temperature changes confidence, not the winning label. If accuracy rises while AURC worsens, the model became harder to automate.

## When reasoning is part of the decision

[OpenJev-RLCD](https://arxiv.org/html/2609.38850) addresses a different case: generate a rationale, then read a distribution over answers. Its two-stage recipe first calibrates the readout with a proper score, then reinforces reasoning with the same score, a reduced policy-gradient weight (`c=0.3`), and a KL anchor (`β=0.04`).

On GSM8K answer verification, two-stage RLCD and GRPO have similar accuracy (~92%), but RLCD covers 81% of items at a 5% error budget versus 19% for GRPO. Correctness-only RL makes the model confident; it does not teach it when to abstain.

Use this only when reasoning materially changes the answer. For routing, moderation, extraction, and ordinary policy checks, a direct one-pass scorer is simpler and faster.

There is a middle ground between direct scoring and a generated rationale: **latent non-autoregressive reasoning**. Recurrent-depth or Coconut-style models spend a few hidden-state iterations without emitting text; LRT-style models predict a latent trajectory in parallel; diffusion models refine masked latent slots. This is well matched to a decision-only output, but it is not free reasoning—iterative variants add sequential latency, supervision is weaker, and hidden computation can make confident errors harder to diagnose. Compare at equal FLOPs against a larger direct scorer, and train the final distribution with a proper scoring rule. See [improvement directions](improvement-directions.md#3-add-latent-reasoning-without-text-decoding) and [efficient reasoning](../knowledge/efficient-reasoning.md).

## Encoder fine-tuning

ModernBERT/GLiClass remains sensible for a stable, short-text workload with tight CPU or latency constraints. Laya's public notebook uses soft cross-entropy plus proper-score policy gradients; Verdict uses a GLiClass-style option head and held-out calibration. These models can beat generalists on the workflow they were trained for, but their zero-shot performance and context length are poor. Treat them as distilled specialists, not as the starting point for a general System One model.
