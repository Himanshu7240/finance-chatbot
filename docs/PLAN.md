# 12-Day Build Plan

Rebuilding the WIDS finance chatbot from scratch — same documented methodology
(`project-report.pdf`), original code and dataset, proper repo hygiene.

- [x] **Day 1 — Scaffolding**: repo structure, `requirements.txt`, `.gitignore`, `.env.example`,
      GitHub repo created, original leaked HF token flagged for revocation.
- [ ] **Day 2 — Dataset design**: QA/context schema, topic scope (NIFTY 50 companies), source list.
- [ ] **Day 3 — Dataset collection**: scrape news articles per company.
- [ ] **Day 4 — Dataset cleaning & split**: dedupe, structure into JSON, train/val/test split.
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
- Flask interface mentioned in the report is out of scope unless time permits after Day 12.
