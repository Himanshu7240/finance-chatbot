# Data

- `raw/articles/<TICKER>.jsonl` — one scraped article per line (title, publish date, body
  paragraphs, source URL), before cleaning (gitignored).
- `raw/http/` — cached HTTP responses, so re-runs and parser changes cost the source site
  nothing (gitignored).
- `raw/collection-report.json` — per-company article and paragraph counts from the last run.
- `processed/` — cleaned question/answer/context JSON, ready for training (gitignored).

`raw/` and `processed/` are gitignored because the datasets are large and regenerable via `src/scraping` and
`src/dataset`. The final processed dataset is uploaded as a Kaggle Dataset for the training
notebook to consume.
