# Data

- `raw/` — scraped article text per company, before cleaning (gitignored).
- `processed/` — cleaned question/answer/context JSON, ready for training (gitignored).

Both are gitignored because the datasets are large and regenerable via `src/scraping` and
`src/dataset`. The final processed dataset is uploaded as a Kaggle Dataset for the training
notebook to consume.
