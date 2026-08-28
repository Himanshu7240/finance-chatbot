# Data

- `raw/articles/<TICKER>.jsonl` — one scraped article per line (title, publish date, body
  paragraphs, source URL), before cleaning (gitignored).
- `raw/http/` — cached HTTP responses, so re-runs and parser changes cost the source site
  nothing (gitignored).
- `raw/collection-report.json` — per-company article and paragraph counts from the last run.
- `processed/paragraphs.jsonl` — cleaned, deduplicated, fact-bearing paragraphs; the
  contexts QA pairs are generated from (`src.dataset.clean`).
- `processed/splits.json` — article id → train/val/test. Splitting happens at article
  level to prevent leakage between splits (`src.dataset.split`).
- `processed/generations.jsonl` — raw Qwen2.5-7B output from the Kaggle notebook, one
  `{para_id, response}` per line.
- `processed/{train,val,test}.json` — the final question/answer/context dataset in the
  `QAExample` schema, ready for training (`src.dataset.build`).

`raw/` and `processed/` are gitignored because the datasets are large and regenerable via `src/scraping` and
`src/dataset`. The final processed dataset is uploaded as a Kaggle Dataset for the training
notebook to consume.
