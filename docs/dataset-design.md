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
3. Rate-limit requests with a delay between fetches; identify with a descriptive user-agent.
4. Cache raw fetched HTML in `data/raw/` so re-runs don't re-hit the site.

**Disclosure**: Moneycontrol's `robots.txt` additionally sets `Disallow: /` for named
AI-training crawlers (`GPTBot`, `CCBot`, `Google-Extended`, `ChatGPT-User`). This project is a
non-commercial student/portfolio project that does not operate any of those crawlers, and it
scrapes at low volume under the general `User-agent: *` rules. It is recorded here for
transparency about data provenance. Anyone reusing this pipeline commercially, or at scale,
should seek permission from the publisher first — and no article text is redistributed in this
repository (`data/` is gitignored; only derived Q/A triplets are used, and only for training).

## QA generation

Article paragraphs are turned into question/answer/context triplets by an LLM, with a prompt
that constrains output to strict JSON and requires the company name to appear in the question
(so examples stay attributable). Details and the prompt itself land on Day 3–4.
