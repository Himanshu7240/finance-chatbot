"""Loading the fine-tuned model and prompting it exactly the way it was trained.

    python -m src.app.model "How much does Tata Steel owe Odisha?" --context "..."

The merged Day 8 model is a plain causal LM - no PEFT, no gated base repo - so this is a
thin wrapper. The part that matters is that it builds its prompt with
:func:`src.training.format.build_prompt`, the same function the training notebook and the
evaluation used. Three call sites, one definition; prompt drift between training and
serving is a quality bug that leaves no trace in the code.

Decoding is greedy. The model was trained to emit one short extracted span and stop, so
sampling can only invent variation the task does not want, and a low token budget keeps a
model that has learned to be terse from padding its answer.
"""

from __future__ import annotations

import argparse
import logging

from ..training.format import build_prompt

log = logging.getLogger("model")

DEFAULT_MODEL_ID = "Himanshu724006/Llama-3.2-3B-finance-india"
MAX_NEW_TOKENS = 64
# 3.2B parameters: fp16/bf16 is ~6.4 GB of weights, fp32 twice that. On a 12 GB laptop the
# fp32 copy simply does not fit, so CPU defaults to bfloat16 - which torch supports natively
# on CPU, unlike fp16, where several ops fall back or are unimplemented.
DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}


class FinanceLLM:
    """The merged fine-tune, loaded lazily and prompted in the training format."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        token: str | None = None,
        dtype: str = "auto",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.token = token
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Download and load the weights. ~6.4 GB, so this is never done at import time."""
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self.dtype == "auto":
            dtype = torch.float16 if device == "cuda" else torch.bfloat16
        else:
            dtype = getattr(torch, DTYPES[self.dtype])
        log.info("loading %s on %s (%s, ~6.4 GB)", self.model_id, device, dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.token)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
            low_cpu_mem_usage=True,
            token=self.token,
        ).to(device)
        self._model.eval()
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def __call__(self, question: str, context: str) -> str:
        """The answer span for a question, given retrieved context."""
        import torch

        self.load()
        prompt = build_prompt(question, context)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        # Only the continuation; everything before is the prompt we just built.
        generated = output[0][inputs["input_ids"].shape[1]:]
        answer = self._tokenizer.decode(generated, skip_special_tokens=True).strip()
        # Training labels were a single line ending in EOS. If a stray newline shows up,
        # the first line is the answer and the rest is the model rambling past its lesson.
        return answer.split("\n")[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--context", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="auto", choices=["auto", *DTYPES])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    llm = FinanceLLM(model_id=args.model_id, device=args.device, dtype=args.dtype)
    log.info("%s", llm(args.question, args.context))


if __name__ == "__main__":
    main()
