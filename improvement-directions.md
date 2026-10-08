# Improvement directions

**Tags:** #system-one #research #inference #latent-reasoning
**Parent:** [index](INDEX.md)

Option-letter logits are a strong baseline, not a finished architecture. The useful research question is: what improves accuracy, calibration, invariance, or throughput at the same compute?

## 1. Improve the readout

Comparing the next-token probabilities of `A`, `B`, and `C` reuses the LM head and needs no training, but it inherits tokenizer frequency, spelling, and position biases. Compare it against:

- **Pointer or energy scoring:** encode each option and score it against a decision state. This handles arbitrary labels without routing through alphabet tokens; Kev is the existing reference.
- **Learned slot tokens:** map options to reserved, equally trained codes rather than natural-language letters. Imajev's fixed slots are one version, although a fixed maximum and retraining cost remain.
- **Sequence scoring:** score each option text under the LM, with length normalization. This removes arbitrary labels but repeats work and can reward fluent wording rather than correctness.
- **Set-equivariant readout:** jointly encode the options and return one logit per option with shared parameters. The output should permute exactly when the options do.
- **Hybrid shortlist:** retrieve candidates with a bi-encoder, then apply a listwise scorer to the shortlist. This is more plausible than a flat 255-way decision for very large menus.

Do not replace the letter baseline merely because a head is more elegant. Require gains on unseen-source accuracy, NLL/Brier, option-permutation stability, and latency.

## 2. Stop treating the alphabet as the natural index

`A/B/C` works because instruction-tuned LMs have seen multiple-choice text, not because letters are optimal identifiers. Natural letters have unequal tokenization and pretraining priors; semantic labels add wording priors; fixed numeric slots may have their own frequency bias.

The desired interface is **permutation equivariant**: reordering the options should reorder the probabilities and change nothing else. Test:

1. single-token letters, digits, and reserved tokens;
2. semantic labels versus opaque labels;
3. all option permutations, not one canonical order;
4. training-time random indexing versus inference-time rotation averaging;
5. duplicate-token marginalization for every textual index.

A learned, exchangeable slot representation is the clean target. Until it wins, randomize indices during training, marginalize equivalent tokens, and measure the residual order effect.

## 3. Add latent reasoning without text decoding

Some decisions need more computation than one transformer pass but do not need a visible chain of thought. Existing LM work suggests three routes:

- **Recurrent latent steps:** feed a hidden state back through a shared block, as in Coconut or recurrent-depth models.
- **Parallel latent trajectories:** predict several latent thought states at once, as in LRT, then let the decision head attend to them.
- **Masked or diffusion refinement:** iteratively refine hidden answer slots without autoregressive text.

This is especially natural here because the only required output is a distribution. A controller could allocate zero latent steps to easy routing and a small fixed budget to hard verification. It is still extra sequential compute when refinement is iterative, and latent reasoning is harder to supervise and inspect. Compare at equal FLOPs and latency against (a) a larger direct model and (b) an explicit-rationale model. Optimize with a proper scoring rule so extra compute does not only make wrong answers more confident. See [efficient reasoning](../knowledge/efficient-reasoning.md).

## 4. Make prefill the serving primitive

The common state should be processed once where the architecture permits it:

- cache the stable system/template prefix;
- prefill the state once, then fork the KV cache for independent questions;
- batch variable-length question suffixes with continuous batching;
- use tree/branch attention or packed suffixes when the attention kernels support them;
- group many option reads from the same final state;
- quantize weights and compile the fixed readout path.

Prefix sharing is not free for every backbone. Hybrid linear-attention or recurrent layers carry state that ordinary 2-D tree masks cannot safely share; cache forking or re-encoding may be required. Benchmark end-to-end latency across realistic state lengths, question counts, and batch sizes.

## 5. Compress long-lived state

For multi-step workflows, keep three tiers:

1. canonical structured facts and decisions;
2. a compact narrative summary rewritten periodically by a stronger LLM;
3. a recent raw-event tail plus retrieved source spans.

The decision model reads the compact state; high-risk or low-confidence cases retrieve raw evidence or escalate. Compression must preserve provenance and uncertainty—an assertive but lossy summary can improve latency while silently worsening calibration. Evaluate decisions before and after compression, especially evidence-ablation and contradiction cases.

## 6. Standardize and search the prompt

The baseline prompt is not merely “state and question.” It needs a stable contract:

1. task rule: use only the supplied state; return one indexed option;
2. state, with explicit delimiters;
3. question/instructions;
4. complete option criteria in a fixed rendering;
5. answer cue ending exactly where logits are read.

Keep this rendering identical across training, calibration, and serving. Search block order, delimiters, answer cue, index scheme, and whether criteria precede the question on a development split; lock the template before final evaluation. Prompt search can overfit a public board as easily as weight training, so confirm on Decision Index's hidden transfer score and a private production holdout.

## 7. Evaluate the system, not only the checkpoint

Use [Decision Index](evaluation.md#decision-index-03) for broad and hidden-domain transfer, [JevBench](evaluation.md#jevbench-v161) for calibration and serving economics, and a production-shaped holdout for the actual routing threshold. Track coverage at an error budget, confident errors, abstention, option-order drift, context robustness, and cost/latency. A direction is promising only if it improves the operational frontier, not one public score.
