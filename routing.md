# Using them to cut LLM calls

**Tags:** #system-one #routing #cascade #agents
**Parent:** [index](INDEX.md) · numbers in [evaluation](evaluation.md)

A System One call replaces an LLM call only where the output was always a branch: which model, which tool, whether to escalate, whether the draft is good enough, which of N candidates to keep. It does not replace the call that has to write the draft. The gain is large exactly when the decision used to be "generate a JSON object, parse it, hope the confidence string is a probability."

TypeSafe's pricing makes the arithmetic concrete. Input is $0.042 / MTok and the output is unbilled, against frontier rates of roughly $0.20–$10 / MTok input and ~5× that on output. A 1k-token gate in front of every request is a rounding error next to one frontier generation. Latency of 70–500 ms (hosted) or ~20 ms (Kev-4B on an H100) fits in front of a user-visible call; it also fits inside an agent step, which a second frontier call often does not.

## Four places it earns its keep

**1. Pick the model once, at the start of the thread.**

LangChain's `ModelRouterMiddleware` ([harness writeup](https://www.langchain.com/blog/building-a-harness-with-jev), [router writeup](https://www.langchain.com/blog/how-to-build-a-model-router-in-the-harness)) sends the first user message through one choice question whose criteria are the tiers you actually have. The chosen model stays for the thread. Their earlier classifier was an LLM with structured output; swapping in Jev made that classification ~50× faster, on their measurement. The criteria are task-specific ("direct lookups and local edits" vs "architecture and high-stakes"), which is why this lives in the harness and not in a generic gateway. A gateway does not know that your agent is Open SWE.

```python
router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(model="openai:luna",
                            criteria="Direct lookups, extraction, and localized changes."),
        "powerful": ModelChoice(model="openai:sol",
                                criteria="Architecture and high-stakes decisions."),
    },
    instructions="Choose the least costly model that can complete the task.",
)
```

The probabilities stay on the agent state. A low `confidence` (peakedness) or a close top-two is a reason to bump up a tier even when the argmax says fast. That policy is three lines and it is the whole point of getting a distribution.

MindStudio's [router walkthrough](https://www.mindstudio.ai/blog/how-to-build-model-router-with-jev) is the same pattern with the rules written out: one request asks task type (`choice`), difficulty (`score`), and privacy (`noul`); privacy above a threshold forces the local model no matter what the other two say; high difficulty goes to the strong or web-enabled model; very long prompts go to a generalist. The ordering of those overrides is the product. Write them in code, in priority order, and log which rule fired.

**2. Gate a tool call before it runs.**

Claude Code, Codex, and Cursor already do this with a private classifier: the agent proposes a tool call, something cheap decides whether to run it, ask, or refuse. `AutoModeMiddleware` is LangChain's version of that gate on Jev. The useful shape is a fan-out, not a single "is this safe?" bit:

- `noul`: does this call leave the workspace, spend money, delete, or send a message?
- `choice`: allow / ask / deny, with criteria that name the failure modes.
- `score`: blast radius.

Escalate when the *probability* crosses your cost ratio, not when the argmax says "risky." If a missed dangerous call costs nine times an unnecessary ask, you ask whenever P(dangerous) > 0.1. That threshold is only meaningful after you have measured calibration on your own tool traces ([fine-tuning](fine-tuning.md)). Also escalate when confidence is low, separately from the label: a 0.4/0.35/0.25 split is "I don't know," and the argmax will still name a winner.

A choice over tools always returns a winner, including when none of the tools fit. Pair it with a `noul` ("does any of these tools apply?") and let the noul veto. TypeSafe's own skill-suggestion note ranks all 182 Hermes skills in one choice, fetches the top three in full, and judges again against the fuller text. Two stages: a cheap shortlist over the catalogue, then a real decision over a short menu with the evidence attached. Above 255 options you have to do this anyway; Jev's WikiRacing demo is that pattern.

**3. Decide whether the cheap model's answer is good enough.**

This is the cascade, and it is a different algorithm from the router.

- A **router** (RouteLLM) looks at the prompt and sends it to one model. One generation. Right when difficulty is visible from the prompt: chitchat vs a proof, a lookup vs a migration.
- A **cascade** (FrugalGPT) always generates with the cheap model, scores the result, and escalates if the score is below a threshold fit on a validation set under a cost budget. Right when you cannot tell difficulty from the prompt, or when the answer is verifiable (tests, schema, a citation check). You pay a generation you might throw away, and the tail latency is the strong model's.

System One sits in both slots, as the router and as the scorer `g(x, y)`. It should not sit in the slot of the model being scored. AutoMix's variant — the small model grades its own draft, then maybe escalates — has the conflict of interest OpenJev-RLCD measures: a correctness-trained grader is overconfident exactly on the items you needed it to abstain on. Use a separate decision model, trained to score the draft, and read Cov@ε on a labelled set of (prompt, draft, correct?) before you pick the threshold. GSM8K-Verify in that paper is this job.

RouteLLM (Ong et al., arXiv:2406.18665, [lm-sys/RouteLLM](https://github.com/lm-sys/RouteLLM)) is the pre-Jev reference for learned routing. Four routers on preference data between a strong and a weak model: matrix factorization (the one they recommend), a BERT classifier, a causal-LLM classifier, and similarity-weighted Elo. The pitch is ~2× cost reduction, and up to ~85% cost cut at 95% of GPT-4 quality on MT-Bench, with the routers transferring to model pairs they were not trained on. The BERT and causal-LLM routers are System One models avant la lettre. What Jev/Kev add is a menu and a calibrated distribution defined per request, so you are not stuck with the pair the router was trained on. What RouteLLM still does better, if you have preference data for your exact pair, is a router trained on who actually won. A zero-shot "how hard is this?" score is a prior; win-rate data is the update. If you have the data, fine-tune ([fine-tuning](fine-tuning.md)) rather than prompting a generalist.

**4. Supervise the loop without another generation.**

An agent that has taken the same action four times is a state you can describe in a few hundred tokens. A `noul` ("is this loop stuck?") plus a `choice` (continue / change strategy / stop) is cheap enough to run every step. Keep the hard step limit in code beside it. The decision model is reading a trace produced by the agent it supervises; it is a better stop condition and a bad only stop condition.

Other fan-outs that are the same shape: which retrieved chunk is supported by the source (`noul` per chunk, or one choice), whether a memory should be written, which subagent a task goes to, whether a user turn is prompt injection (Kev's devtools eval includes a prompt-injection source), whether to compress history. Batch them. One request with eight questions costs the extra question tokens and, on a prefix-cached decoder, almost no extra state prefill. Kev's server runs questions in token-budgeted batches so the answers do not depend on the split; OpenJev reads in chunks of ~12. Do not split what is actually one question. If the decision is "which of these three labels, given that the urgency is high," that is one choice whose criteria mention urgency, or a second request that sees the first answer. Independent questions that share only the state are the ones that batch.

## Long-lived state

Do not keep pasting an entire agent trace into a small decision model. Maintain:

1. canonical structured facts, commitments, and tool results;
2. a compact narrative summary periodically rewritten by a stronger LLM;
3. a recent raw tail and retrievable source spans.

The System One model reads the compact state. On low confidence, contradiction, or a high-risk branch, retrieve the raw evidence or escalate to the stronger model. Trigger rewriting by token budget or semantic change, not every turn. Store provenance with each compressed fact and test the same decisions before and after compression: a fluent summary that deletes a caveat can reduce cost while making calibration worse.

For each decision batch, render the shared state first so the server can reuse its prefix, then fork independent question suffixes. This matters more than micro-optimizing the final softmax when the state is long. Architecture details and caveats are in [prefill optimizations](architecture.md#prefill-optimizations).

## What the surrounding code has to do

The model returns distributions. The policy is yours.

- Put overrides in a fixed order: safety and privacy first, then cost, then the argmax. Log the rule, the distribution, and the threshold.
- Thresholds come from a labelled slice and an error budget, not from 0.5 and not from the `confidence` field. Refit when the menu or the domain changes.
- Keep a deterministic backstop for anything reversible only at high cost: max steps, allow-listed tools, schema validation. The decision model is an additional gate.
- For high cardinality, score in two stages. Stage one can be a cheaper encoder or a bi-encoder (CLM caches action embeddings; that is the right tool when the action set is large and fixed). Stage two is a listwise decision over the shortlist, where options need to interact.
- Do not route on a generated self-rating ("I am 90% sure"). mini-jev's preregistered result: letter logits match constrained JSON; asking for a written probability collapses onto the first option. If the only model you have is a generative one, read its next-token distribution over a fixed menu. [evaluation](evaluation.md) has the protocol, including option rotation.
- Arithmetic, exact dates, and long-document needles are the known holes of the small decoders. Append computed facts (`KEV_DATE_FACTS`) or retrieve the span into the state. Do not spend a fine-tune trying to make a 4B subtract dates.
- When the task is aleatoric (annotators disagree, the rubric is taste), a direct scorer trained on the label distribution is the right tool, and a reasoning model trained with RLCD will correctly abandon the rationale (ChaosNLI in the OpenJev-RLCD paper). When the task needs computation (check a proof, verify a multi-step answer), the direct scorer is the wrong tool: the same paper's probe goes from 56% direct to 87% after a rationale. Escalate those to a System 2 call, and use the decision model on the result.

## A small design sequence

1. Write down the branches your code already computes by prompting an LLM. Each one is a candidate question.
2. Run the frozen letter-readout baseline (a Qwen you already have) and Kev-4B on a few hundred real states. Use [Decision Index and JevBench](evaluation.md#evaluation-stack) as external checks, but measure Cov@ε on your own data.
3. If coverage at your error budget is enough, ship the generalist and keep the labels for monitoring.
4. If a particular question is far from the budget, fine-tune that question's data with `--init_from`, replay the generalist set, refit temperature on a holdout.
5. If the text is short and the menu will not change, distill that one question into a ModernBERT and serve it on CPU. Leave the shifting questions on the decoder.
6. Re-measure option-order flips and confident errors on the holdout before you move the threshold.

## Prior art, if you want the theory around the product

- RouteLLM, arXiv:2406.18665. Preference-trained routers; BERT and causal-LLM classifiers; pre-inference, one model per query.
- FrugalGPT, Chen et al. 2023. Cascades as a budget-constrained threshold problem. The scoring function is the piece a calibrated decision model replaces.
- AutoMix, Aggarwal et al. 2024. Small model self-verifies, then maybe escalates. Pays two calls when it escalates; the self-grade is the weak point.
- LLM-Blender, Jiang et al. 2023. Query several models, pick a winner. The picker can be a decision model; the cost model only works when you were going to fan out anyway.
- FIRST, Reddy et al. 2024. Listwise reranking from a single token's logits. The immediate ancestor of "read a distribution, do not decode a ranking."
- ModernBERT; GLiClass; DeBERTa-v3-MNLI; SetFit. The encoder stack for when step 5 above is the whole system.
- Guo et al. 2017 on calibration; Gneiting & Raftery 2007 on proper scores. Why accuracy is not the objective.
