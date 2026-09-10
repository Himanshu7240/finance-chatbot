# Reproducing This Project End to End

Every artifact in this repo — the corpus, the dataset, the adapter, the merged model — can be
rebuilt from an empty `data/` directory. This is the ordered runbook: what to run, where it runs,
roughly how long it takes, and what each step should produce.

Nothing here is a summary of the guides. Each step links to the guide that explains *why* it works
the way it does; this document is only concerned with getting it to run.

## What you need

| | |
|---|---|
| Python | 3.11+ (developed on 3.14) |
| Local hardware | Anything. No step below needs a local GPU |
| Kaggle account | Two notebooks need a free GPU: QA generation and fine-tuning |
| Hugging Face account | With access to the gated `meta-llama/Llama-3.2-3B-Instruct` repo |
| Wall-clock | ~3.5 h of compute, spread over three days of scraping if you are polite about it |

```bash
git clone https://github.com/Himanshu7240/finance-chatbot && cd finance-chatbot
pip install -r requirements.txt
cp .env.example .env       # fill in HF_TOKEN
pytest                     # 96 tests, no network, no downloads
```

See [`setup.md`](setup.md) for secrets, gated-repo access and the failure modes worth knowing
about before you start.

---

## Step 1 — Collect the news corpus

**Where:** local · **Time:** ~2 h for a 12-month corpus, dominated by deliberate rate limiting ·
**Guide:** [02](guides/02-collecting-a-training-corpus.md)

```bash
python -m src.scraping.collect --months 12 --max-articles 40
```

Discovers articles through Moneycontrol's published news sitemaps (the paginated listing URLs the
original project used are `robots.txt`-disallowed), matches each URL to one of the 50 companies in
`data/companies.json`, fetches at ~1 request / 3 s with jitter, and caches every response under
`data/raw/http/`.

**Produces:** `data/raw/articles/<TICKER>.jsonl`, `data/raw/collection-report.json`.
**Reference run:** 2,014 articles / 18,098 paragraphs across all 50 companies, published
2025-09-02 → 2026-08-25.

The HTTP cache is the important part of this step. Re-running after a parser change costs zero
requests, so iterate on extraction freely — just don't delete `data/raw/http/`.

## Step 2 — Clean and deduplicate

**Where:** local · **Time:** ~1 min · **Guide:** [03](guides/03-generating-a-qa-dataset.md)

```bash
python -m src.dataset.clean
```

Drops paragraphs that carry no fact, that cannot be attributed to a company, and that duplicate
another paragraph (exact, then Jaccard near-duplicate at `--jaccard 0.8`).

**Produces:** `data/processed/paragraphs.jsonl`, `data/processed/clean-report.json`.
**Reference run:** 18,098 → **10,257** paragraphs (4,201 not fact-bearing, 3,525 unattributable,
115 duplicates).

## Step 3 — Assign splits, before generating anything

**Where:** local · **Time:** seconds · **Guide:** [03](guides/03-generating-a-qa-dataset.md)

```bash
python -m src.dataset.split --seed 42
```

Splits are drawn **per article, not per paragraph**, and they are drawn *now* — before any QA pair
exists. Two questions generated from the same article can otherwise land in train and test at once,
and every metric downstream becomes a lie you cannot detect.

**Produces:** `data/processed/splits.json`.

## Step 4 — Generate QA pairs (Kaggle GPU)

**Where:** Kaggle · **Time:** ~2 h on a T4 · **Guide:** [03](guides/03-generating-a-qa-dataset.md)

Upload `data/processed/paragraphs.jsonl` as a Kaggle Dataset, then run
[`notebooks/qa_generation_qwen.ipynb`](../notebooks/qa_generation_qwen.ipynb).

The teacher is **Qwen2.5-7B-Instruct** — Apache 2.0, so the derived dataset has no licensing
ambiguity, and strong enough to write a grounded question from a single paragraph. The notebook
writes one `{para_id, response}` per line and nothing else; all validation happens locally in the
next step, where it can be re-run without a GPU.

**Produces:** `data/processed/generations.jsonl` (download it back into the repo).
**Reference run:** 13,608 responses over 10,257 paragraphs; 3,225 paragraphs the model correctly
declined to write a question for.

## Step 5 — Validate and assemble the dataset

**Where:** local · **Time:** ~1 min · **Guide:** [03](guides/03-generating-a-qa-dataset.md)

```bash
python -m src.dataset.build
```

Every generated pair runs a gauntlet: the answer must be grounded in its paragraph, every number in
the answer must appear in the source, the company must be named, the question must stand alone.
Rejections are counted by reason rather than discarded silently — the report is how you find out
your prompt is broken.

**Produces:** `data/processed/{train,val,test}.json`, `data/processed/dataset-report.json`.
**Reference run:** 11,590 pairs generated → **8,292 kept** (train 6,368 / val 973 / test 951),
all 50 companies present in every split. Provenance and the rejection breakdown:
[`dataset-design.md`](dataset-design.md).

## Step 6 — Fine-tune (Kaggle GPU)

**Where:** Kaggle · **Time:** ~1 h 20 m on a T4 · **Guide:**
[04](guides/04-lora-and-4bit-fine-tuning.md)

Upload `data/processed/` as a Kaggle Dataset, set the `HF_TOKEN` Kaggle secret, and run
[`notebooks/finetune_llama_lora.ipynb`](../notebooks/finetune_llama_lora.ipynb).

LoRA r=32 / alpha=32 / dropout 0.05 on all seven projections, over a 4-bit NF4 base with fp16
compute. Loss is masked to the answer tokens only — the model is not spending gradient on
reproducing a question it is always handed. The notebook evaluates the *base* model before training
so the comparison is real rather than remembered.

**Produces:** a ~195 MB adapter. Save it as a Kaggle notebook output.
**Reference run:** 796 steps, final train loss 0.134. Against the base model on 300 held-out
examples: exact match 11.7 → 76.3, token F1 40.8 → 91.5, numeric accuracy 63.8 → 91.9. The full
curve, including why the second epoch was wasted:
[`training-results.md`](training-results.md).

## Step 7 — Merge and publish

**Where:** Kaggle (CPU is fine) · **Time:** ~15 min · **Guide:**
[05](guides/05-merging-and-publishing-a-model.md)

Run [`notebooks/merge_and_publish.ipynb`](../notebooks/merge_and_publish.ipynb), or locally:

```bash
python -m src.training.merge --adapter models/lora-adapter --out models/merged
python -m src.training.merge --adapter models/lora-adapter --out models/merged \
    --push <your-username>/Llama-3.2-3B-finance-india
```

The merge loads the base in **fp16, not 4-bit**. Merging into quantized weights re-quantizes the
adapter's contribution through a 4-bit grid and silently degrades exactly the small, precise update
you spent an hour training. Nothing errors; the model just gets worse.

The published name must begin with `Llama` and the card must say "Built with Llama" — required by
the Llama 3.2 Community License for distributed derivatives, and enforced in `merge.py`.

**Produces:** a 6.4 GB fp16 model. Reference artifacts:
[merged](https://huggingface.co/Himanshu724006/Llama-3.2-3B-finance-india) ·
[adapter](https://huggingface.co/Himanshu724006/Llama-3.2-3B-finance-india-lora).

## Step 8 — Serve it

**Where:** local · **Time:** ~7 s to start · **Guides:**
[06](guides/06-retrieval-and-the-app-layer.md), [07](guides/07-serving-a-small-model.md)

```bash
python -m src.app.ui
```

Starts in retrieval-only mode — the TF-IDF index over Step 2's paragraphs plus live quotes from
yfinance — and loads the model only when you tick the box. Point it at your own model with
`--model <your-username>/Llama-3.2-3B-finance-india`.

---

## The short version

```bash
python -m src.scraping.collect --months 12 --max-articles 40   # ~2 h,  local
python -m src.dataset.clean                                    # ~1 min, local
python -m src.dataset.split --seed 42                          # seconds, local
#  -> notebooks/qa_generation_qwen.ipynb                        ~2 h,   Kaggle T4
python -m src.dataset.build                                    # ~1 min, local
#  -> notebooks/finetune_llama_lora.ipynb                       ~1h20m, Kaggle T4
#  -> notebooks/merge_and_publish.ipynb                         ~15 min, Kaggle CPU
python -m src.app.ui                                           # local
```

## What will not reproduce exactly

- **The corpus.** Moneycontrol's sitemaps are a moving window, so a run today collects a different
  set of articles than the reference run did. Every downstream count will differ.
- **The generated QA pairs.** The teacher samples; the same paragraph can yield a differently
  phrased question. The validators are deterministic, so the *rejection reasons* should look
  similar even when the counts do not.
- **Training.** Seeded, but GPU non-determinism and a different dataset make the final loss and the
  metrics move by a point or two.

What *should* reproduce is the shape of the thing: ~10k clean paragraphs from ~2k articles, ~70–80%
of generated pairs surviving validation, and a fine-tune that takes exact match from roughly 12 to
roughly 75 on held-out data. If your numbers are far from that, the pipeline reports at each step
are where to look first.
