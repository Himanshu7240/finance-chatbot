# Guide 04 — LoRA and 4-bit Fine-Tuning

Day 5 fine-tunes `Llama-3.2-3B-Instruct` on the 8,292 triplets from Day 4, on a free Kaggle
GPU. This guide is the *why* behind every number in that notebook.

## Why not just fine-tune the whole model

Because it does not fit, and it is not close. Full fine-tuning a 3.21B-parameter model with
Adam needs, roughly:

| | |
|---|---|
| Weights (fp16) | 6.4 GB |
| Gradients (fp16) | 6.4 GB |
| Adam optimizer states — `m` and `v`, fp32 | 25.7 GB |
| fp32 master weights | 12.8 GB |
| **Total, before a single activation** | **~51 GB** |

A Kaggle T4 has 16 GB. Even both T4s together fall well short. And that is before activations,
which scale with batch size and sequence length.

The optimizer states are the offender: **four bytes per parameter, twice over**, for numbers
that exist only to compute an update. That's where parameter-efficient fine-tuning attacks.

## LoRA — the actual idea

The insight behind LoRA (Hu et al., 2021) is that the *update* a model needs to adapt to a
new task is much lower-rank than the model itself. You don't need to change all 3.21B
parameters in arbitrary directions; the useful change lives in a small subspace.

So instead of learning a full update matrix `ΔW` of shape `(d_in, d_out)`, LoRA factors it:

```
ΔW = B · A          A: (r, d_in)      B: (d_out, r)      r << min(d_in, d_out)
W_effective = W_frozen + (alpha / r) · B · A
```

`A` and `B` are the only things trained. `W` is frozen — never updated, never needs a
gradient, never needs an optimizer state. `B` is initialized to zeros, so at step 0 `ΔW = 0`
and the model is exactly the base model: training starts from a known-good point rather than
a perturbed one.

### What r, alpha and dropout actually do

**`r = 32`** — the rank, i.e. how much capacity the adapter has. Too low and the adapter
can't represent the change the task needs; too high and you lose the regularization benefit
(and drift toward the cost of full fine-tuning). For domain adaptation on a few thousand
examples, 8–64 is the usual band. This project uses 32, matching the original.

**`alpha = 32`** — a scaling factor. The update is multiplied by `alpha / r`, here `32/32 = 1`.
Its real purpose is decoupling: if you later change `r`, keeping `alpha/r` fixed keeps the
update magnitude comparable, so you don't have to re-tune the learning rate as well. A common
convention is `alpha = 2r`; `alpha = r` is a more conservative update.

**`dropout = 0.05`** — dropout on the LoRA input. Mild regularization; with 6,368 training
examples and 48M trainable parameters, some is warranted.

### Which layers get adapters

Targeting all seven projection matrices — `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, `down_proj` — rather than attention only. The QLoRA paper found
adapting *all* linear layers matters more for matching full fine-tuning quality than making
`r` large. Adapting only `q` and `v` is a widely-copied habit from the original LoRA paper
that later work superseded.

For Llama-3.2-3B (28 layers, hidden 3072, intermediate 8192, 24 heads with 8 KV heads), the
LoRA parameter count at `r = 32` works out to `r × (d_in + d_out)` summed over those seven
matrices, times 28 layers:

```
per layer:  q 196,608 + k 131,072 + v 131,072 + o 196,608
          + gate 360,448 + up 360,448 + down 360,448   =  1,736,704
× 28 layers                                            = 48,627,712
```

**~48.6M trainable parameters — about 1.5% of the model.** The notebook prints the real figure
via `model.print_trainable_parameters()`; treat that output as the source of truth over this
arithmetic.

## 4-bit quantization — the other half

LoRA removes the optimizer-state cost. The frozen weights still have to live in memory, and
6.4 GB in fp16 plus activations is uncomfortable on a 16 GB card. So the base model is loaded
in **4-bit** (QLoRA, Dettmers et al., 2023): ~1.6 GB instead of 6.4 GB.

This is not "training a 4-bit model". The frozen base is quantized; the LoRA adapters stay in
16-bit and are what actually trains. During the forward pass, 4-bit weights are dequantized
block-by-block to the compute dtype, used, and discarded — you pay compute for memory.

Three settings matter:

**Quant type — `nf4` vs `fp4`.** NF4 ("normal float 4") uses quantization levels spaced so
each bin holds equal probability mass *for normally-distributed weights* — which is what
trained neural network weights approximately are. FP4 spaces levels by float exponent
instead, which wastes resolution where weights are dense. The QLoRA paper measured NF4 as
consistently better at identical memory cost. **This project uses `nf4`, a documented
deviation from the original's `fp4`** — same memory, strictly better precision, no reason to
reproduce the weaker choice.

**Double quantization.** The quantization constants are themselves quantized. Saves roughly
0.4 bits per parameter — about 150 MB here. Free, so it's on.

**Compute dtype — `float16`, not `bfloat16`.** The T4 is Turing (sm_75) and has no bf16
support. This is the same constraint that shaped the Day 4 generation notebook.

### The memory budget on one T4

| | |
|---|---|
| Base model, 4-bit + double quant | ~1.7 GB |
| LoRA adapters (fp16) + gradients | ~0.2 GB |
| Adam states for 48.6M params (fp32) | ~0.4 GB |
| Activations, batch 4 × 1024 tokens, with gradient checkpointing | ~2–4 GB |
| **Total** | **~5–7 GB** |

Comfortably inside 16 GB — which is the entire point. Gradient checkpointing recomputes
activations during the backward pass instead of storing them: roughly 30% slower, and it is
what keeps activation memory bounded as sequence length grows.

## What the model actually learns from

Each example becomes one sequence, following the Day 2 schema:

```
Question: {question}  Context: {context}          <- input
{answer}                                          <- label
```

**Loss is computed on the answer tokens only.** The prompt tokens are masked out with `-100`.
This matters more than it sounds: if you train on the whole sequence, most of the loss comes
from predicting the *question and context* — text the model will always be given at inference
time. You would be spending most of the gradient signal teaching it to generate news
paragraphs, which is not the task. Masking focuses every update on the thing being learned:
extracting the answer from the context.

## Evaluate the base model first

Before training anything, run the **un-fine-tuned** `Llama-3.2-3B-Instruct` on the test set
and record its score.

This is the step most projects skip, and skipping it makes every later number
uninterpretable. Our examples hand the model the context in the prompt — and an
instruction-tuned 3B is already reasonably good at "extract the answer from this paragraph".
So a strong post-training score does not, by itself, demonstrate that fine-tuning did
anything. Only the *delta* does.

It also gives you an honest decision rule for what to do next:

| Result | What it means | Action |
|---|---|---|
| Fine-tuned ≫ base | Training worked | Ship it |
| Fine-tuned ≈ base | Dataset too small, or LR/epochs off | More data, or tune training |
| Fine-tuned < base | Config is wrong — overfitting, bad LR, broken masking | Fix training; more data won't help |

That last row is why the baseline is worth 20 minutes: without it, a bad result looks like a
data problem, and you can waste a day generating examples that were never the bottleneck.

## Talking points

- *"What is LoRA and why use it?"* — Full fine-tuning a 3B model needs ~51 GB, mostly Adam
  optimizer states. LoRA freezes the base and learns a low-rank update `ΔW = BA`, training
  ~48.6M parameters (1.5%) instead of 3.21B, which fits on a free 16 GB GPU.
- *"What does alpha do?"* — It scales the update by `alpha/r`. Its purpose is decoupling rank
  from update magnitude, so changing `r` doesn't force you to re-tune the learning rate.
- *"Why 4-bit, and what's NF4?"* — Quantizing the frozen base cuts it from 6.4 GB to ~1.6 GB.
  NF4's levels are spaced for normally-distributed weights, so it's more accurate than FP4 at
  the same size. Adapters stay in 16-bit; only the frozen base is quantized.
- *"Why mask the prompt in the loss?"* — Otherwise most of the gradient signal goes to
  predicting the question and context, which are always supplied at inference. Masking spends
  every update on the actual task.
- *"How do you know fine-tuning helped?"* — By scoring the base model on the same held-out
  test set first. The absolute number means nothing; the delta is the result.
