# Guide 01 — Dataset Design for LLM Fine-Tuning

## Why this comes before any code

Every later step — fine-tuning quality, how the model generalizes, even how you'll defend this
project in an interview — is downstream of decisions made here. A bigger model or fancier
training loop cannot fix a badly-designed dataset. This is the step most tutorials skip, and
the step interviewers actually probe.

## Fine-tuning vs. prompting vs. RAG — where this project sits

There are three ways to make a general-purpose LLM good at a narrow task:

1. **Prompting** — give the base model instructions and examples at inference time. No training
   needed, but you're limited by context length and the model's existing knowledge.
2. **RAG (Retrieval-Augmented Generation)** — keep the base model frozen, but fetch relevant
   documents at query time and stuff them into the prompt. Good for facts that change often.
3. **Fine-tuning** — actually update the model's weights (or a small adapter, see LoRA in
   Guide 03) on task-specific examples, so the behavior is baked in.

This project does **fine-tuning** for the core Q&A behavior (so the model *is* a finance
assistant, not a general assistant pretending to be one), and layers a lightweight **live
data lookup** on top for stock prices, which change every second and could never be baked into
static weights. That split — fine-tune for durable domain knowledge and reasoning style,
fetch live for anything time-sensitive — is the practical version of the RAG-vs-fine-tuning
question you'll get asked about.

## Why "instruction-tuned" base model, not the raw pretrained one

We fine-tune `Llama-3.2-3B-Instruct`, not the base `Llama-3.2-3B`. The `-Instruct` variant has
already been trained (via SFT + RLHF-style methods) to follow instructions and hold a
conversation. Fine-tuning on top of that means our small QA dataset only has to *specialize*
the model, not teach it how to follow instructions from scratch — the latter would need orders
of magnitude more data than we can produce ourselves.

## The Question / Answer / Context triplet — and why "just Q&A" isn't enough

The original project started with plain `(question, answer)` pairs and found the model gave
inaccurate or unsupported answers. The fix was adding a **context** field:

```json
{
  "question": "How much might Tata Steel need to pay as minerals tax dues to Odisha?",
  "answer": "More than Rs 17,000 crore",
  "context": "Tata Steel might need to pay more than Rs 17,000 crore as minerals tax dues to the state of Odisha."
}
```

Why this matters: without context, the model can only answer from whatever it memorized during
training — and with a few thousand examples, memorization is unreliable and doesn't generalize.
With context, training teaches the model a *skill*: "extract the answer to this question from
the text you're given." That skill transfers to questions the model never saw verbatim during
training, as long as it's given relevant context at inference time — which is exactly what the
`RAGPipeline` will do later (Guide 06) by retrieving live stock data as context.

This is also the actual mechanism referenced in the project report's cited paper (Ye et al.,
2024, "Empirical Insights on Fine-Tuning LLMs for QA"): the paper distinguishes *memorized*
knowledge from *in-context* extraction, and finds that a "diagonal effect" holds — models do
best on data whose memorization profile matches what they're being asked to do. Since we can't
control what Llama already memorized about NIFTY 50 companies, giving explicit context sidesteps
the question entirely: the model is trained to read and extract, not just recall.

## How much data is actually needed?

The same paper found that effective QA fine-tuning can happen with **as few as 60 examples** —
because the base model already has broad world knowledge, and fine-tuning mostly needs to
activate and shape that knowledge, not install new facts from zero. More data isn't automatically
better: low-quality or redundant examples can *hurt* performance.

The original project scaled to 17,000 triplets anyway, because the goal wasn't "can the model
answer one question type" but "can it handle open-ended questions across 50 different
companies and several question styles (revenue, ratings, tax disputes, etc.)" — that breadth is
what needs volume, not the fine-tuning mechanism itself. For this rebuild, we'll size the
dataset to the same reasoning: enough per-company diversity to generalize across companies and
question phrasing, not an arbitrary large number.

## The schema we're using

Defined in [`src/dataset/schema.py`](../../src/dataset/schema.py):

```python
@dataclass
class QAExample:
    company: str        # canonical company name, e.g. "Tata Steel"
    ticker: str          # NSE ticker, e.g. "TATASTEEL"
    question: str
    answer: str
    context: str          # source paragraph the Q/A was derived from
    source_url: str        # provenance — which article this came from
```

`source_url` isn't in the original project's schema, but it's a small addition: keeping
provenance means you can always trace a bad answer back to its source article, which matters
both for debugging and for being able to say, in an interview, exactly how the data was
sourced and verified.

## Talking points for interviews / resume

- "I designed the fine-tuning dataset as question/answer/context triplets instead of bare Q&A
  pairs, because giving the model context at training time teaches it to extract answers from
  supplied text rather than rely purely on memorization — which generalizes better and reduces
  hallucination."
- "I sized the dataset based on per-entity and per-question-style diversity rather than a raw
  target count, since research on QA fine-tuning shows small datasets can be sufficient once the
  base model already has relevant world knowledge — the bottleneck is coverage, not volume."
- "I kept source provenance in the schema so every training example is traceable back to its
  source article."
