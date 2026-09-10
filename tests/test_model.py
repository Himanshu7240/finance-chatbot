"""The prompt the model is served must be the prompt it was trained on.

Guide 06's "one definition, three call sites" argument is worth nothing if nothing checks
it, so this stubs the tokenizer and the model and asserts on the exact string that reaches
them. No weights are downloaded.
"""

from __future__ import annotations

import torch

from src.app.model import FinanceLLM
from src.training.format import build_prompt

QUESTION = "How much does Tata Steel owe Odisha?"
CONTEXT = "Tata Steel might need to pay more than Rs 17,000 crore as minerals tax dues."


class FakeEncoding(dict):
    """What a tokenizer call returns: a mapping that can be moved to a device."""

    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, reply: str):
        self.reply = reply
        self.prompts: list[str] = []

    def __call__(self, text, return_tensors=None, **kwargs):
        self.prompts.append(text)
        return FakeEncoding(input_ids=torch.tensor([[1, 2, 3, 4]]),
                            attention_mask=torch.tensor([[1, 1, 1, 1]]))

    def decode(self, token_ids, skip_special_tokens=True):
        return self.reply


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.kwargs = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        prompt_length = kwargs["input_ids"].shape[1]
        return torch.tensor([[0] * prompt_length + [5, 6, 7]])


def build(reply: str = "More than Rs 17,000 crore"):
    llm = FinanceLLM()
    llm._tokenizer = FakeTokenizer(reply)
    llm._model = FakeModel()                 # set, so load() is a no-op
    return llm


def test_the_served_prompt_is_exactly_the_training_prompt():
    llm = build()
    llm(QUESTION, CONTEXT)
    assert llm._tokenizer.prompts == [build_prompt(QUESTION, CONTEXT)]


def test_the_prompt_ends_with_the_answer_cue():
    llm = build()
    llm(QUESTION, CONTEXT)
    assert llm._tokenizer.prompts[0].endswith("\nAnswer:")


def test_decoding_is_greedy():
    # The model was trained to emit one short span; sampling can only add variance the
    # task does not want.
    llm = build()
    llm(QUESTION, CONTEXT)
    assert llm._model.kwargs["do_sample"] is False


def test_only_the_generated_continuation_is_returned():
    llm = build(reply="More than Rs 17,000 crore")
    assert llm(QUESTION, CONTEXT) == "More than Rs 17,000 crore"


def test_a_model_that_rambles_past_its_lesson_is_trimmed_to_the_first_line():
    llm = build(reply="More than Rs 17,000 crore\nQuestion: what else can I tell you?")
    assert llm(QUESTION, CONTEXT) == "More than Rs 17,000 crore"


def test_surrounding_whitespace_is_stripped():
    assert build(reply="  Rs 340.7 crore  ").__call__(QUESTION, CONTEXT) == "Rs 340.7 crore"
