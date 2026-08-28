# 12-Day Build Plan

Rebuilding the WIDS finance chatbot from scratch — same documented methodology
(`project-report.pdf`), original code and dataset, proper repo hygiene.

- [x] **Day 1 — Scaffolding**: repo structure, `requirements.txt`, `.gitignore`, `.env.example`,
      GitHub repo created, original leaked HF token flagged for revocation.
- [x] **Day 2 — Dataset design**: QA/context schema (`src/dataset/schema.py`), NIFTY 50 company
      list (`data/companies.json`), source + compliance decisions (`docs/dataset-design.md`),
      Guide 01 on dataset design.
- [x] **Day 3 — Dataset collection**: `src/scraping/` — sitemap discovery, robots-aware
      rate-limited fetcher with disk cache, company matching, article extraction; corpus in
      `data/raw/articles/`, Guide 02 on collecting a corpus.
- [x] **Day 4 — Dataset cleaning, generation & split**: `src/dataset/` — cleaning, near-
      duplicate and attribution filtering (10,257 paragraphs from 18,098), leakage-safe
      article-level split, QA generation with Qwen2.5-7B-Instruct on Kaggle, groundedness
      validation. **Final dataset: 8,292 triplets** (train 6,368 / val 973 / test 951)
      across all 50 companies. Guide 03.
- [ ] **Day 5 — Fine-tuning notebook**: LoRA + 4-bit config, env-based HF login.
- [ ] **Day 6–7 — Run + monitor training**: Kaggle GPU, checkpointing, evaluation.
- [ ] **Day 8 — Merge & publish model**: push merged model to Hugging Face Hub.
- [ ] **Day 9 — Retrieval + app layer**: `StockDataRetriever` / `RAGPipeline` as `src/` modules.
- [ ] **Day 10 — Gradio UI + integration testing**.
- [ ] **Day 11 — Documentation**: README, dataset provenance, setup docs.
- [ ] **Day 12 — Final review & push**: demo, screenshots, release tag.

## Known deviations from the original

- HF token is read from `.env` / Kaggle secrets, never hardcoded.
- Dataset is rebuilt from scratch (not copied) — see Day 2–4.
- Article discovery uses Moneycontrol's published news sitemaps instead of the paginated
  listing URLs the original used, which `robots.txt` disallows. See `dataset-design.md`.
- The scraper sends a browser user-agent (the CDN 403s any other kind) and identifies itself
  in an `X-Purpose` header instead. Documented in `dataset-design.md` and Guide 02.
- Each day ships a concept guide in `docs/guides/` alongside the code.
- Flask interface mentioned in the report is out of scope unless time permits after Day 12.
