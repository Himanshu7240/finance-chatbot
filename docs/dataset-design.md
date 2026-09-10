# Dataset Design

Concrete decisions for the training dataset. For the *why* behind the triplet format and
sizing reasoning, see [Guide 01](guides/01-dataset-design-for-llm-finetuning.md).

## Schema

Defined in [`src/dataset/schema.py`](../src/dataset/schema.py):

| Field | Type | Purpose |
|-------|------|---------|
| `company` | str | Canonical company name, e.g. `"Tata Steel"` |
| `ticker` | str | NSE ticker, e.g. `"TATASTEEL"` |
| `question` | str | The user-style question |
| `answer` | str | Concise answer, derived from `context` |
| `context` | str | Source paragraph the Q/A was generated from |
| `source_url` | str | Article the example came from (provenance) |

At training time these become a single sequence: `"Question: {question} Context: {context}"`
as the input, with `answer` as the label.

## Scope

**Companies**: all 50 NIFTY 50 constituents, listed in
[`data/companies.json`](../data/companies.json) with ticker and URL slug.

**Question styles** to cover per company — breadth here is what makes the model generalize
across phrasings rather than memorize:

- Financial results (revenue, profit, margins, YoY/QoQ changes)
- Analyst coverage (ratings, target prices, brokerage views)
- Corporate actions (acquisitions, expansions, capex, dividends)
- Regulatory / legal (tax disputes, penalties, compliance)
- Leadership & strategy (appointments, guidance, business direction)
- Sector/market positioning

**Target size**: ~10,000–17,000 triplets, matching the original project's range. Sized for
coverage across 50 companies × 6 question styles rather than an arbitrary count — see Guide 01
on why volume alone isn't the goal.

## Source and compliance

**Source**: Moneycontrol news articles for NIFTY 50 companies (same source as the original
project).

**How we discover articles** — this is where we deliberately differ from the original:

Moneycontrol's `robots.txt` has `Disallow: /*/page-*/` for all user-agents, which covers the
paginated listing URLs the original project used to enumerate articles. Instead we discover
article URLs through the **news sitemaps that Moneycontrol explicitly publishes in its own
`robots.txt`**:

- `https://www.moneycontrol.com/news/news-sitemap.xml` (recent articles)
- `https://www.moneycontrol.com/news/index-sitemap-2026.xml` (archive index)

Rules the scraper follows:

1. Honor every `Disallow` rule under `User-agent: *` — no `/page-*/` pagination, no
   `/news/printpage/`, no `/stocks/company_info/*`.
2. Discover URLs via published sitemaps only.
3. Rate-limit requests (~3s between fetches, plus jitter) and back off exponentially on
   403/429/5xx rather than hammering through them.
4. Cache raw fetched HTML in `data/raw/http/` so re-runs don't re-hit the site.

**User-agent — a documented compromise.** The Day 2 plan was to identify the scraper with a
descriptive user-agent. That turned out to be impossible: Moneycontrol's CDN returns `403` for
any user-agent that isn't a recognised browser string, including a browser string with a
project token appended. The scraper therefore sends a plain browser user-agent and identifies
itself in an `X-Purpose` header, plus an optional `From:` contact address read from
`SCRAPER_CONTACT` in `.env`. Every other constraint above is unchanged. See
[Guide 02](guides/02-collecting-a-training-corpus.md) for the full reasoning.

**Disclosure**: Moneycontrol's `robots.txt` additionally sets `Disallow: /` for named
AI-training crawlers (`GPTBot`, `CCBot`, `Google-Extended`, `ChatGPT-User`). This project is a
non-commercial student/portfolio project that does not operate any of those crawlers, and it
scrapes at low volume under the general `User-agent: *` rules. It is recorded here for
transparency about data provenance. Anyone reusing this pipeline commercially, or at scale,
should seek permission from the publisher first — and no article text is redistributed in this
repository (`data/` is gitignored; only derived Q/A triplets are used, and only for training).

## Collection pipeline (Day 3)

Implemented in `src/scraping/`, run with:

```bash
python -m src.scraping.collect --months 6 --max-articles 25
```

| Module | Responsibility |
|--------|----------------|
| `robots.py` | Loads and applies `robots.txt` per host; fails closed if it can't be read |
| `fetcher.py` | Rate limiting, exponential backoff, on-disk response cache, robots enforcement |
| `sitemaps.py` | Article discovery via the news sitemap and the monthly archive sitemaps |
| `matcher.py` | Maps an article URL to a NIFTY 50 company from its headline slug |
| `article.py` | Extracts title, publish date and clean body paragraphs; drops boilerplate |
| `collect.py` | CLI that runs discovery -> fetch -> parse -> `data/raw/articles/<TICKER>.jsonl` |

**Company matching** works on whole slug tokens (so `itc` doesn't match `switch`), resolves
ambiguity by longest alias (`sbi life` beats `sbi`), and carries per-company exclusion tokens
for same-brand entities that aren't the constituent (`reliance-power`). Articles that pass slug
matching are re-checked against the parsed body text before being stored.

**Output**: one JSONL file per ticker, one article per line:

```json
{"company": "Tata Steel", "ticker": "TATASTEEL", "url": "...", "title": "...",
 "published": "2026-07-31T20:51:04+05:30", "paragraphs": ["...", "..."]}
```

Runs are resumable — already-collected URLs are skipped and every response is cached — and each
run writes `data/raw/collection-report.json` with per-company article and paragraph counts.

## QA generation

Article paragraphs are turned into question/answer/context triplets by an LLM, with a prompt
that constrains output to strict JSON and requires the company name to appear in the question
(so examples stay attributable). The prompt lives in
[`src/dataset/qa_prompt.py`](../src/dataset/qa_prompt.py) — in the repo rather than in the
notebook, because changing it changes every downstream number.

**Generator**: `Qwen2.5-7B-Instruct` (Apache 2.0), run with vLLM on a free Kaggle GPU via
[`notebooks/qa_generation_qwen.ipynb`](../notebooks/qa_generation_qwen.ipynb). A 7B teacher
generating training data for the 3B student — knowledge distillation. Chosen over a hosted
API so that anyone cloning this repo can rebuild the dataset with no API key and no spend,
and so the derived dataset carries no licensing ambiguity. The trade-off is a higher rate of
malformed JSON, which the validator absorbs.

**Validation** ([`src/dataset/qa_validate.py`](../src/dataset/qa_validate.py)): every pair
must parse as JSON, name the company, ask something other than a yes/no question, answer in
≤25 words, and be *grounded* — at least 70% of the answer's content words present in the
source paragraph, and **every number in the answer present in the paragraph**, with no
tolerance. A fabricated figure is the worst example a finance dataset can contain. Rejections
are counted by reason in `data/processed/dataset-report.json`.

## Pipeline

```
src.scraping.collect   ->  data/raw/articles/<TICKER>.jsonl   2,014 articles (Day 3)
src.dataset.clean      ->  data/processed/paragraphs.jsonl    10,257 from 18,098
src.dataset.split      ->  data/processed/splits.json         80/10/10, by article
  [Kaggle notebook]    ->  data/processed/generations.jsonl   11,590 pairs
src.dataset.build      ->  data/processed/{train,val,test}.json  8,292 triplets
```

**Attribution filtering**: multi-company roundup articles ("Stocks to Watch Today: …") label
every paragraph with a single company, so paragraphs about other companies inherit the wrong
name. Groundedness checks cannot detect this — the answer is genuinely supported by the text;
only the label is wrong. `clean.py` drops paragraphs from roundup headlines unless they name
their own company (3,525 removed), and `qa_validate.py` catches the residue involving
non-NIFTY companies at validation time.

Splits are assigned **by article, not by paragraph**: paragraphs from one article overlap
heavily, so a random per-paragraph split would put near-identical text in both train and
test and inflate eval scores. See [Guide 03](guides/03-generating-a-qa-dataset.md).

## The finished dataset

What the pipeline above actually produced, for the record. Everything here comes from
`data/processed/dataset-report.json` and `clean-report.json`, both regenerated on every run.

| | |
|---|---|
| Triplets | **8,292** — train 6,368 / val 973 / test 951 |
| Companies | 50, every one present in all three splits |
| Source articles | 2,014 collected, 1,910 surviving into the final paragraphs |
| Article publication range | 2025-09-02 → 2026-08-25 |
| Source paragraphs | 10,257 clean, from 18,098 extracted |
| Teacher model | Qwen2.5-7B-Instruct (Apache 2.0), free Kaggle T4 |
| Generated pairs | 11,590 from 13,608 responses; 3,225 paragraphs correctly declined |
| Kept after validation | 8,292 (71.5% of generated pairs) |

### Why 3,298 pairs were thrown away

| Reason | Count |
|---|---|
| `unknown_paragraph` — generated before the attribution filter existed | 3,368 |
| `company_not_named` — question not attributable on its own | 1,729 |
| `invented_number` — a figure not present in the source paragraph | 538 |
| `ungrounded` — under 70% of answer content words in the source | 449 |
| `not_self_contained` — "the company", "this quarter" with no referent | 353 |
| `duplicate_question` | 123 |
| `answer_too_long` / `yes_no_question` / other | 106 |

(The counts exceed 3,298 because `unknown_paragraph` includes responses for paragraphs the
attribution filter later removed entirely.)

The **538 invented numbers** are the line that justifies the whole validator. Those are answers a
groundedness check on words alone would have passed, and every one of them would have taught a
finance model that inventing a figure is acceptable.

### Coverage is uneven, and that matters

Per-company triplet counts run from **36 (SBI Life) to 254 (Maruti Suzuki)**, median 169. The
spread is inherited from the news: Maruti and Reliance are written about constantly, SBI Life
rarely. Two consequences worth stating rather than hiding:

- The model has seen far more of some companies than others, so answer quality is not uniform
  across the NIFTY 50 — it is roughly proportional to how much press each company got in the year
  to August 2026.
- Per-company counts do not track paragraph counts. State Bank of India yields 202 triplets from
  276 paragraphs, while SBI Life yields 36 from 25 — the `company_not_named` rule is stricter on
  companies that the press refers to obliquely ("the bank", "the lender").

### Provenance and redistribution

The paragraphs are extracted from Moneycontrol articles and remain their copyright. The repo
therefore ships **no article text**: `data/raw/` and `data/processed/` are gitignored, and what is
published is the *code that rebuilds them* plus the reports describing them. Anyone reproducing the
dataset collects it themselves, under the same `robots.txt` terms and rate limits, from the same
public sitemaps.

The questions and answers are generated by Qwen2.5-7B-Instruct (Apache 2.0), so the derived
annotation carries no licensing ambiguity of its own — but each triplet's `context` field is source
text, which is why the assembled dataset is treated the same way as the corpus.

The trained artifacts are a different matter: model weights are a transformation, published under
the Llama 3.2 Community License with the required "Built with Llama" attribution, and the model
card names this dataset and its limitations. The `source_url` on every triplet means any answer can
be traced back to the article it came from.
