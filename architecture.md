# Architecture

**Tags:** #system-one #jev #gemma #qwen
**Parent:** [index](INDEX.md)

## The design that is winning

The current open frontier is simpler than the first wave of Jev reproductions:

1. Render the state, question, and allowed answers in the prompt.
2. Bind each answer to a one-token label such as `A`, `B`, or `C`.
3. Stop at the answer position and read the next-token logits for those labels.
4. Renormalize over the allowed labels and apply a held-out calibration temperature.

Nothing is generated. The model does one prefill and exposes the distribution it already has at the answer position.

This **option-letter readout** is the dominant simple baseline on both [JevBench v1.6.1](evaluation.md#jevbench-v161) and [Decision Index 0.3](evaluation.md#decision-index-03). JevBench's capped leaders include Quyet, deck-31B, H2O-Lightning, Rune, Xor, and Torchcast; Decision Index's leaders include Perplexity Decider v1.1, Fastino GLiDE, Torchcast Decision 27B, deck-31B, Kev-27B, and Quyet. They are decoder models, but they do not all use the same readout: the frontier now includes frozen LM heads, pointer heads, and learned decision heads.

The strongest control is **deck-31B**: frozen `google/gemma-4-31B-it`, no decision training, TorchAO FP8, Cygnet's letter prompt, and one temperature. It is near the JevBench intelligence leader and remains near the Decision Index frontier. A custom decision head is therefore optional, not the defining feature.

## Exact mechanics

Suppose the rendered prompt ends at `Answer:` and the final hidden state is \(h\). The ordinary LM head computes one logit for every vocabulary token:

\[
z = W_{\text{vocab}}h
\]

For option \(i\), collect every token ID whose decoded text is its label after the prompt boundary. If `A` and ` A` are both valid there, combine them:

\[
z_i^*=\operatorname{logsumexp}_{t:\,\operatorname{canon}(\operatorname{decode}(t))=\ell_i} z_t,\qquad
p(i)=\operatorname{softmax}_i(z_i^*/T)
\]

This is one ordinary forward pass over the prompt. There is no `generate()`, sampled token, appended token, or decode-phase KV-cache growth. The answer is the argmax and the full softmax is returned. Summing duplicate token mass matters because choosing one arbitrary token ID discards valid probability and makes results depend on tokenizer accidents. Disable thinking: if the model predicts a thinking marker before the label, these are not the answer logits.

## Prompt contract

It is more than state plus question. A robust template contains:

1. a task rule: use the supplied state and choose exactly one indexed option;
2. the state inside stable delimiters;
3. the question or decision instruction;
4. every option and its criterion, in a fixed rendering;
5. an answer cue that ends exactly where the logits are read.

```text
Choose the option best supported by STATE. Answer with one option index.
<STATE>
{state}
</STATE>
QUESTION: {instruction}
OPTIONS:
A. {criterion_a}
B. {criterion_b}
ANSWER:
```

The exact template is model-specific: Cygnet/deck, Quyet, and learned-head systems were trained or tuned with different renderings. Keep the chosen template identical during training, calibration, and serving. Search delimiters, block order, answer cue, and index scheme on development data, then lock them before evaluation; a public leaderboard is not a prompt-development set.

## The API is not the architecture

`POST /v1/systemone` accepts one state and several independent questions. That does not mean every implementation encodes the state once.

- Jev appears to share the state and isolate question branches.
- Winnow and some custom runtimes explicitly cache or fork a shared prefix.
- Cygnet and Xor score each question as a separate prompt and re-encode the state.
- H2O-Lightning's shim accepts several questions and reports shared usage, but each decision is still described as one forward pass.

State sharing is a serving optimization. The invariant is narrower: each question ends in a bounded probability distribution without autoregressive output.

Questions must remain independent. If question B depends on answer A, make B a second request or encode the joint decision as one question. The workflow graph belongs in code.

### Prefill optimizations

For independent questions over the same state, cache the stable template, prefill the state once, and fork its KV cache for each question suffix. Continuous batching handles unequal suffix lengths; tree/branch attention or packed suffixes can avoid physically copying shared KV; grouped reads can score many menu chunks from one state. Quantization and compilation help further because the readout shape is fixed and there is no decode loop.

The practical gain grows with state length and question count. It can be small for one short question, and it is not universally available: hybrid linear-attention or recurrent layers carry state that ordinary tree masks cannot share. Those models may need explicit state forking or re-encoding. Measure end-to-end latency, not only the final matrix multiply. See [routing](routing.md#what-the-surrounding-code-has-to-do) and [improvement directions](improvement-directions.md#4-make-prefill-the-serving-primitive).

## What the leading systems actually do

**Quyet-1.0-Large** ([model card](https://huggingface.co/chinhnc/Quyet-1.0-Large)) is the Capability leader: 81.7, from 73.4 Intelligence and 90.0 Calibration. It is Gemma-4-31B-it with a merged rank-16 LoRA, a fixed option-letter prompt, and fitted temperatures. It needs about 62.5 GB of bf16 weights, accepts at most ten options, and keeps the state within a 6k-token budget. The training data and optimizer are not published.

**deck-31B** ([repo](https://github.com/krishna-gogineni-765/deck31b)) is the clean control. It uses the same 31B base with unchanged weights, dynamic FP8 quantization, the public Cygnet prompt, and `T=3.4`. It needs an 80 GB FP8-capable GPU, supports up to 26 options, and rejects inputs over 16,384 tokens. Its near-tie with Quyet says that the 31B backbone and readout protocol explain much of the result.

**H2O-Lightning-4B** ([model card](https://huggingface.co/h2oai/h2o-lightning-4b)) is the practical standout. It is a Qwen3.5-4B fine-tune served by stock vLLM plus a small shim. It leads the official four-axis Composite Score because it combines 60.0 Intelligence, 90.0 Calibration, low estimated cost, and low latency. The current v1.2 checkpoint also accepts images. This is the best default when a 31B model is excessive.

**The Decision Index 27B frontier** is more architecturally varied. Perplexity Decider v1.1 full-fine-tunes Qwen3.8-27B, lifts the causal mask in full-attention layers, and adds a 255-way head. Kev-27B uses the same backbone family with a pointer head. Torchcast Decision 27B uses a merged LoRA and shared one-pass question handling. Fastino GLiDE no-thinking also ranks near the top, but its evaluated weights were not yet public. These results make custom heads credible directions, but there is no matched experiment showing that the head—rather than data, backbone, or bidirectional attention—caused the gain.

**Gemma-4-12B systems** occupy the middle:

- [Torchcast Decision 12B](https://huggingface.co/torchcast-ai/torchcast-decision-12b): LoRA, option-letter readout, 50/50 hard and teacher-distribution targets. Strong but CC BY-NC 4.0.
- [Winnow-12B](https://huggingface.co/EldanRing/Winnow-12B): merged rank-32 LoRA, GGUF serving, chat and vision from the same model. Q8 fits a 16 GB GPU.
- [Cygnet](https://github.com/blockbrain-ai/cygnet-recipe): frozen Gemma-4-12B-it plus a letter-readout shim and `T=3.4`. It is the 12B baseline to run before training.
- [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni): 30k-question fine-tune with a 256-way head; supports text, images, audio, and video. The training pipeline is not published.

**Gemma-4-26B-A4B systems** trade total memory for roughly 4B active parameters:

- [Surogate Rune v3](https://huggingface.co/surogate/rune-26b-a4b-GGUF): full fine-tune with text and vision; data and training recipe are private.
- [Xor 26B-A4B](https://huggingface.co/juspay/Xor-26B-A4B): two LoRA lineages, including expert-specific adapters, merged into bf16 and blended 15% back toward the base. The merge recipe is public; the adapters and training data are not.

## Alternative readouts

Letter logits are the cheapest heuristic, not an optimality result. They inherit token-frequency, spelling, and position biases. More specialized heads are useful when they beat that control:

- **Pointer head:** [Kev](https://github.com/jaredpalmer/kev) scores option representations against a decision representation. This naturally handles labels supplied at request time and can share a state prefix. Its transparent training and fine-tuning tools remain valuable.
- **Fixed slot head:** [Imajev](imajev.md) uses 255 answer codes plus a trained `unknown` code. It is useful when explicit abstention and image-to-record comparison matter.
- **Joint schema head:** Cloudflare's Clef scores several typed fields through a trained schema head. This can reduce repeated work when the output schema is fixed, but it is not the design used by the current leaders.
- **Diffusion canvas:** OpenJev reads masked answer slots from DiffusionGemma. It supports images, but the current leaderboard favors ordinary decoders.
- **Encoder classifier:** Laya and Verdict remain attractive for a narrow, stable taxonomy on CPU. They re-encode the state per question, have 512–1024-token limits, and need task-specific fine-tuning. They are deployment specialists, not current generalist leaders.

Natural alphabetic indices are also arbitrary. The target property is permutation equivariance: reordering options should only reorder probabilities. Reserved learned slots, pointer scoring, and set-equivariant heads are cleaner candidates; until they win, randomize option order, marginalize duplicate tokens, and report permutation drift. See [improvement directions](improvement-directions.md).

## What is known about Jev

TypeSafe has published the interface and behavior, not the model:

- parallel typed outputs rather than token generation;
- a post-training method called RLCD, aimed at calibrated decisions;
- synthetic training data;
- up to 255 options;
- quoted latency of 70–500 ms and input-only pricing.

Parameter count, backbone, dense vs MoE, readout, loss, and training recipe remain undisclosed. [Black-box probes](https://archerhume.com/posts/jevs-architecture-unmasked/) support shared state, isolated question branches, and listwise option interaction. The [counterpoint](https://jevresearch.github.io/Jev-Research/counterpoint/) confirms those behaviors but finds no evidence for a specific public parent model. Treat causal decoder and sparse MoE as plausible reconstructions, not facts.

## Choosing an architecture

- **Strongest reproducible cross-board choice:** Quyet-1.0-Large, if an 80 GB-class deployment and ten-option limit are acceptable.
- **Decision Index head research:** Perplexity Decider v1.1 for the leading specialized path; Kev-27B for the more inspectable pointer-head path.
- **Best compact JevBench trade-off:** H2O-Lightning-4B; validate breadth separately.
- **Best no-training baseline:** deck-31B for maximum quality; Cygnet for a 12B footprint.
- **Local 16 GB deployment:** Winnow-12B Q8.
- **Images plus explicit abstention:** Imajev-4B; for broader visual tasks, H2O-Lightning v1.2 or Jev-Omni.
- **Stable labels, short text, CPU:** a ModernBERT/GLiClass encoder.

Start with option-letter logits and temperature scaling because they are the strongest cheap control, not because `A/B/C` is the final architecture. Add training or a different readout only after it wins on a representative holdout and both broad-transfer and operational tests. See [fine-tuning](fine-tuning.md), [evaluation](evaluation.md), and [improvement directions](improvement-directions.md).
