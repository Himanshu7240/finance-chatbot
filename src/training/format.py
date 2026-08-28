"""How a QAExample becomes tokens - the single source of truth.

Every place that touches this format has to agree: the training notebook, evaluation,
and the Gradio app on Day 10. If the app prompts the model even slightly differently
from how it was trained, quality drops for reasons that are invisible in the code and
maddening to debug. So the format lives here and everything imports it.

Day 2 specified the input as ``Question: {question} Context: {context}`` with the answer
as the label. This adds newlines and a trailing ``Answer:`` cue - the model needs an
unambiguous signal for where its output begins, which a space alone doesn't give.
"""

from __future__ import annotations

PROMPT_TEMPLATE = "Question: {question}\nContext: {context}\nAnswer:"

# Masked-out positions in the label tensor; the loss function ignores them.
IGNORE_INDEX = -100


def build_prompt(question: str, context: str) -> str:
    """The model's input. Identical at training, evaluation and inference time."""
    return PROMPT_TEMPLATE.format(question=question.strip(), context=context.strip())


def build_example(tokenizer, example: dict, max_length: int = 1024) -> dict:
    """Tokenize one training example, masking the prompt out of the loss.

    Loss is computed on the answer tokens only. Training on the whole sequence would
    spend most of the gradient signal teaching the model to reproduce the question and
    the context - text it is always *given* at inference time. Masking points every
    update at the thing actually being learned: pulling the answer out of the context.
    """
    prompt = build_prompt(example["question"], example["context"])
    answer = example["answer"].strip()

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(" " + answer, add_special_tokens=False)["input_ids"]
    answer_ids = answer_ids + [tokenizer.eos_token_id]

    input_ids = (prompt_ids + answer_ids)[:max_length]
    # -100 over the prompt span; real token ids over the answer span.
    labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def collate(batch: list[dict], pad_token_id: int) -> dict:
    """Pad a batch to equal length, padding labels with IGNORE_INDEX.

    Padding labels with the pad token instead of -100 is a classic silent bug: the model
    gets trained to emit padding, and the loss curve looks fine while it happens.
    """
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    padded = {"input_ids": [], "labels": [], "attention_mask": []}
    for item in batch:
        gap = width - len(item["input_ids"])
        padded["input_ids"].append(item["input_ids"] + [pad_token_id] * gap)
        padded["labels"].append(item["labels"] + [IGNORE_INDEX] * gap)
        padded["attention_mask"].append(item["attention_mask"] + [0] * gap)
    return {key: torch.tensor(value) for key, value in padded.items()}
