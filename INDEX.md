# System One models

**Snapshot:** early Oct 2026, about three weeks after TypeSafe's 15 Sep 2026 launch of Jev. The open ecosystem is moving weekly; treat leaderboards as dated.

A System One model is a fast decision function: unstructured state in, typed probabilities out, no generated text. The name is Kahneman's System 1 (snap judgment) against System 2 (slow verbal reasoning). The product claim, from [TypeSafe's launch](https://typesafe.ai/blog/introducing-system-one-models-and-jev), is that software can branch on those probabilities the way it branches on a function return — classify, route, score, gate — at roughly the cost of a prefill.

The interface that everything below speaks:

| Primitive | Returns | Typical use |
|---|---|---|
| `choice` | one of ≤255 options, a distribution, a confidence | routing, intent, tool pick |
| `score` | expected level on an ordered 2–10 scale, plus the distribution | rubrics, urgency, quality |
| `noul` | P(yes) ∈ [0, 1] | gates, entailment, "does any option fit?" |

Several questions ride one request. They share the state and cannot read each other. **Confidence is not P(correct).** For a choice with K>1 options the official adapter sets `c = (p_max − 1/K) / (1 − 1/K)`: how peaked the distribution is. A peaked wrong answer still has high confidence. Act on the probability, after you have measured calibration on your own labels.

Jev itself is closed. Parameter count, base model, and the RLCD training recipe are unpublished. The open frontier now uses decoder models—mainly Gemma 4 or Qwen—with several readouts. Frozen letter-logit systems such as deck-31B remain surprisingly competitive; Decision Index is led by trained 27B systems with letter, pointer, and 255-way heads, while H2O-Lightning-4B leads JevBench's compact deployment trade-off. BERT-style encoders remain useful for CPU serving and stable short-text taxonomies.

## Notes

- [architecture](architecture.md) — exact no-decode readout, prompt contract, prefill sharing, alternative heads, and leading systems
- [fine-tuning](fine-tuning.md) — reproducible recipe, disclosed frontier recipes, calibration, RLCD, and latent reasoning
- [evaluation](evaluation.md) — Decision Index plus JevBench, production metrics, controls, and which open weights to run
- [routing](routing.md) — where a decision call replaces an LLM call; router vs cascade vs harness gate
- [imajev](imajev.md) — how to run mohit67890/imajev locally and call it like Jev, including images
- [improvement directions](improvement-directions.md) — better readouts and indices, latent reasoning, prefill optimization, state compression, and prompt search

Use both major boards: [Decision Index](evaluation.md#decision-index-03) tests broad capability and hidden-domain transfer; [JevBench](evaluation.md#jevbench-v161) tests the typed decision primitive, calibration, latency, and cost. Neither replaces a production-shaped holdout.

## Primary sources

Official, or as close as exists:

- [Introducing System One Models and Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) — Diogo Almeida, 15 Sep 2026. The claims, and the caveats TypeSafe itself prints under them.
- [Jev's Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/) — Archer Hume, 17 Sep 2026. Black-box probes (~10k calls). The reconstruction Kev implements. Read the "what would change my mind" section; sparse-MoE is the weakest claim.
- [A counterpoint on Jev's architecture](https://jevresearch.github.io/Jev-Research/counterpoint/) — agrees on shared state, isolated questions, interacting options; rejects a specific public tokenizer or a named Qwen parent.
- [Kev](https://github.com/jaredpalmer/kev) — Jared Palmer. The documented open decoder: weights, recipe, frozen evals, serving numbers. Apache-2.0.
- [OpenJev-RLCD](https://arxiv.org/html/2609.38850) — Gao & Wang, arXiv:2609.38850. Code: [ZimmyGao/openjev-rlcd](https://github.com/ZimmyGao/openjev-rlcd). An independent "calibrate, then reinforce" recipe for reasoning models. It is not TypeSafe's unpublished method.
- [Evaluating and Benchmarking the System One Model Jev](https://arxiv.org/html/2609.37647v1) — arXiv:2609.37647. Jev 1.13.0, zero-shot, 37 datasets, 346k requests. The cleanest external accuracy/calibration study.
- [Decision Index 0.3](https://huggingface.co/spaces/multimodalart/jev-decision-index) and its [reproduction kit](https://github.com/apolinario/decision-index) — 37 counted public benchmarks plus private same-skill and new-domain tests; the broad-transfer board.
- [JevBench](https://github.com/fstandhartinger/jevbench) — sealed typed decisions with explicit intelligence, calibration, speed, and cost axes; the operational board.

Open servers and encoder checkpoints:

- [razorback16/openjev](https://github.com/razorback16/openjev) — `/v1/systemone` on DiffusionGemma, plus routes to Laya, Verdict, CLM, JevK5.
- [daseinlabs/open-jev](https://github.com/daseinlabs/open-jev) — one-pass option scoring on Gemma 3 4B; frozen-feature head training.
- [Laya typed-decisions](https://huggingface.co/convaiinnovations/laya-typed-decisions) and [fine-tune notes](https://jevlaya.cc/en/laya/finetune) — ModernBERT-large.
- [Verdict](https://github.com/Heman10x-NGU/Verdict-open-jev) / `heman10x/rlcd-modernbert-151m` — ModernBERT-base + GLiClass head.
- [mini-jev](https://github.com/r-ms/mini-jev) — frozen Qwen3-4B, letter logits vs constrained JSON. Preregistered.
- [Your language model is already a decision model](https://shadowgraph.io/ntlm1686/Your-language-model-is-already-a-decision-model) — unfine-tuned Qwen3.5-9B next-token scoring vs Jev.

Secondary blogs (LangChain, MindStudio, "learn Jev" sites) are useful for harness patterns and unreliable for architecture or benchmarks. Both leaderboards move quickly and optimize different objectives; always record the protocol version and date. See [evaluation](evaluation.md).
