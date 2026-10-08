# Using imajev

**Repo:** [mohit67890/imajev](https://github.com/mohit67890/imajev) · **Weights:** [imajev-4b](https://huggingface.co/mohit67890/imajev-4b) · Apache-2.0
**Parent:** [index](INDEX.md)

Local Jev-compatible API. Imajev uses learned fixed answer slots plus a trained "can't tell", not the alphabet-token readout used by deck/Cygnet. It also accepts up to two images. There is no hosted inference API. The [Hugging Face Space](https://huggingface.co/spaces/mohit67890/imajev) is a browser demo of the 4B; production calls go to a server you run. Weights and photos stay on that machine.

Use **imajev-4b**. It is the current adapter (the 2B and 9B are the previous generation). One question is one forward pass, no token generation: about 96 ms raw on an H100, about 350 ms with the four-rotation serving mode, about 1.2 s on a Mac Studio in that same mode.

## Run it

Needs the Qwen3.5-4B base (~9 GB) and the adapter (~500 MB). MLX on Apple silicon, PyTorch on NVIDIA.

```sh
git clone https://github.com/mohit67890/imajev && cd imajev
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[serve,mlx]"     # Mac. CUDA: pip install -e ".[serve,torch]"
python scripts/download_model.py --model 4b
hf download mohit67890/imajev-4b --local-dir adapters/imajev-4b

# Mac
PYTHONPATH=src:scripts python scripts/playground/server.py \
  --model-bundle artifacts/model-qwen4b.json \
  --adapter adapters/imajev-4b/mlx \
  --calibration adapters/imajev-4b/calibration.json \
  --model-name imajev-4b --port 8765

# CUDA: add --backend torch and point --adapter at adapters/imajev-4b (the PEFT dir, not mlx/)
```

Playground: http://127.0.0.1:8765/. API: `POST http://127.0.0.1:8765/v1/systemone`.

`--rotations 4` averages four option orders. This reduces answer-index bias rather than adding knowledge. That is how the README's published numbers were measured, at about 3× the latency (+0.9 points on JevBench hard for the 4B). Pair it with `calibration-rot4.json` rather than `calibration.json`. On an H100, `--fast --merge-lora` is the same computation, faster (about 11 ms text / 91 ms image per decision in their check); it wants a C compiler for the Triton kernels. See [answer indexing and serving directions](improvement-directions.md#2-stop-treating-the-alphabet-as-the-natural-index).

Text-only on a machine without torchvision: `pip install -e ".[serve,torch-text]"`. The full checkpoint still loads. CPU works and is slow.

## Request

Text is JSON, the same shape as Jev. `state` is a string or an object. One to eight questions. Question ids are yours; the model does not see them.

```python
import requests
r = requests.post("http://127.0.0.1:8765/v1/systemone", json={
    "state": "Charged twice for one order, need the duplicate refunded.",
    "questions": {
        "queue": {"type": "choice", "instructions": "Route the ticket.",
                  "criteria": {"billing": None, "shipping": None, "account": None, "other": None}},
        "urgent": {"type": "noul", "instructions": "Does this need a human today?"},
        "urgency": {"type": "score", "instructions": "How urgent is this ticket?",
                    "criteria": ["can wait a week", "this week", "today", "right now"]},
    },
})
```

| Type | `criteria` | You read |
|---|---|---|
| `choice` | 2–254 options, `{name: description or null}` | `choice`, `probabilities` (sum to 1 over your options) |
| `noul` | none required | `noul`: P(yes). Half of the unknown mass is folded in, same as Jev |
| `score` | 2–10 ordered level strings | `score` (expected level index from 0), `probabilities` |

Images: at most two. The first is the reference, the second the target. Multipart, or data URLs inside JSON.

```python
request = {"state": {"listing": {"title": "Men's suede boat shoes", "color": "red"}},
           "questions": { ... }}
with open("listing.jpg", "rb") as photo:
    r = requests.post(URL, files={"image": photo}, data={"request": json.dumps(request)})

# two photos, JSON
request["images"] = ["data:image/jpeg;base64,...", "data:image/jpeg;base64,..."]
r = requests.post(URL, json=request)
```

Multipart file fields it accepts: `image`, `images`, `image[]`. A `data:image/...;base64,...` string buried in `state` is pulled out and replaced with `[image N]`.

Limits, refused rather than truncated: state ≤ 32 KB, ≤ 4,096 tokens, ≤ 8 questions, English. Images are resized to ≤ 400,000 pixels; over 20 MB is a 413. A third image is a 422.

## Response

```json
{
  "model": "imajev-4b",
  "answers": {
    "queue": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {"billing": 0.91, "shipping": 0.03, "account": 0.02, "other": 0.04},
      "confidence": 0.88,
      "unknown_probability": 0.01,
      "abstained": false
    },
    "urgent": {"type": "noul", "noul": 0.22, "unknown_probability": 0.04, "abstained": false}
  },
  "usage": {"total_ms": 350.0, "input_tokens": 180}
}
```

`unknown_probability` is mass on a trained can't-tell code (missing, contradictory, or out-of-scope evidence). `abstained` is true when that code was the most likely outcome. `confidence` is how peaked the distribution is, the same idea as Jev's field; it is not P(correct). The option probabilities are conditional on the model answering.

Act on the probability that survives abstention. Their own snippet:

```python
a = r.json()["answers"]["queue"]
p = a["probabilities"][a["choice"]] * (1 - a["unknown_probability"])
if a["abstained"] or p < 0.85:
    route_to_person(a)
else:
    send_to(a["choice"])
```

On their 279-item bench, the 4B at 90% sure automates 58% of decisions and is right on 97.5% of those. Fit the threshold on a few hundred of your own cases, including ones whose honest answer is "can't tell". `--calibration` applies a temperature (1.305 for the 4B) and does not change the argmax. Photo-against-record traffic was already calibrated raw on an earlier check; if that is all you serve, try without the file or fit your own (`scripts/v1_text/fit_temperature_calibration.py`).

## What it will not do

One pass, no reasoning trace. Multi-step arithmetic and "is this answer good?" trail a thinking model. Two-photo comparison and shelf counts are the weak visual tasks. Context is 32 KB, not Jev's 32k tokens. The 9B is the one for knowledge-heavy text, and it wants ~19 GB resident.
