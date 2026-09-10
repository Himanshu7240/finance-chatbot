# Finance Chatbot — Indian Stock Market

A chatbot that answers questions about NIFTY 50 companies and the Indian stock market, built by
fine-tuning Meta's Llama-3.2-3B-Instruct on a custom question/answer/context dataset, with a Gradio
interface that also pulls live stock prices via `yfinance`.

This is an original rebuild of the WIDS (Winter in Data Science) project described in
[`docs/project-report.pdf`](docs/project-report.pdf) — same methodology (data collection, fine-tuning
approach, model choice, deployment), reimplemented from scratch with a proper project structure,
no hardcoded secrets, and a dataset built from the ground up.

## Approach

- **Data**: scrape recent news articles for NIFTY 50 companies from public financial news sources,
  clean the text, and generate question/answer/context triplets from each article paragraph via an LLM.
- **Model**: Llama-3.2-3B-Instruct, fine-tuned with LoRA (r=32, alpha=32, dropout=0.05) on top of a
  4-bit quantized base model (BitsAndBytes, NF4, double quantization) — memory-efficient enough to
  train on a single Kaggle T4.
- **Interface**: a Gradio chat UI. Stock-price questions are detected via embedding similarity and
  answered with live data from `yfinance`; other financial questions go to the fine-tuned model.

## Project layout

```
src/            # scraping, dataset building, training, and app code
tests/          # pytest suite — no network, no model download
notebooks/      # Kaggle notebooks (fine-tuning, data prep)
data/           # raw/ and processed/ datasets (gitignored — see data/README.md)
docs/           # original project report + supporting docs
```

## Model

Published on the Hub, merged into fp16 so it loads with a plain `from_pretrained` and needs
neither `peft` nor access to the gated base repo:

- **[Himanshu724006/Llama-3.2-3B-finance-india](https://huggingface.co/Himanshu724006/Llama-3.2-3B-finance-india)** — merged model, 6.43 GB
- **[Himanshu724006/Llama-3.2-3B-finance-india-lora](https://huggingface.co/Himanshu724006/Llama-3.2-3B-finance-india-lora)** — the LoRA adapter alone, 195 MB

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Himanshu724006/Llama-3.2-3B-finance-india", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Himanshu724006/Llama-3.2-3B-finance-india")
```

Against the un-fine-tuned base on 300 held-out examples: exact match 11.7 → **76.3**, token F1
40.8 → **91.5**, numeric accuracy 63.8 → **91.9**. Read
[the limitations](docs/training-results.md) before trusting those numbers — the test answers
came from the same teacher model as the training data, and ~8% of numeric answers are wrong.

## Running it

```bash
python -m src.app.ui          # chat UI on http://127.0.0.1:7860
```

The app starts in about a second and answers from retrieved evidence; the fine-tuned model sits
behind a checkbox and loads (6.4 GB, once) only if you tick it — so you can see, side by side,
what the fine-tune actually contributes. Every answer shows its route, its sources and the exact
context the model was given.

Conversation state is deliberately thin: the model was trained on single turns and has no chat
template, so history never enters the prompt. The UI carries only the last company and route, per
session, which is enough for *"What is Reliance trading at?" → "and Wipro?"* to work — and it says
when it assumed one.

## Asking it from the command line

The same pipeline, without the UI — it answers from the evidence it retrieved, which is also what
makes it testable:

```bash
python -m src.app.pipeline "What is Tata Steel trading at?"      # live NSE quote via yfinance
python -m src.app.pipeline "Why is Tata Steel restructuring?"    # the Day 4 news corpus
python -m src.app.pipeline "How is Wipro's AI business doing?" --model   # add the fine-tune
```

A question is routed to the live price feed only when it has both price wording *and* a resolvable
NIFTY 50 company; anything else with a company goes to the article corpus; anything with no company
is refused rather than answered from a lexically similar paragraph. Every answer carries its source
— an article URL and date, or the quote's fetch time. The reasoning is in
[Guide 06](docs/guides/06-retrieval-and-the-app-layer.md).

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN etc.
pytest                 # 80 tests, no network and no model download
```

The suite injects a fake price feed, a fake clock, a fake corpus and a fake generator, so every
routing and refusal path is covered without touching Yahoo or downloading weights. Tests that need
the built dataset skip themselves on a fresh clone.

## Docs

- [`docs/PLAN.md`](docs/PLAN.md) — build plan and progress
- [`docs/training-results.md`](docs/training-results.md) — fine-tuning setup, results and limitations
- [`docs/dataset-design.md`](docs/dataset-design.md) — dataset schema, scope, sourcing and compliance
- [`docs/guides/`](docs/guides/) — concept guides for each technique used in this project
