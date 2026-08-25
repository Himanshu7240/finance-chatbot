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
  4-bit quantized base model (BitsAndBytes, fp4, double quantization) — memory-efficient enough to
  train on a single Kaggle GPU.
- **Interface**: a Gradio chat UI. Stock-price questions are detected via embedding similarity and
  answered with live data from `yfinance`; other financial questions go to the fine-tuned model.

## Project layout

```
src/            # scraping, dataset building, training, and app code
notebooks/      # Kaggle notebooks (fine-tuning, data prep)
data/           # raw/ and processed/ datasets (gitignored — see data/README.md)
docs/           # original project report + supporting docs
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in HF_TOKEN etc.
```

## Status

See [`docs/PLAN.md`](docs/PLAN.md) for the current build plan and progress.
