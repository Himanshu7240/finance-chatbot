# Guide 02 — Collecting a Training Corpus

Guide 01 decided *what* the dataset should contain: question / answer / context triplets
covering 50 companies and six question styles. This guide is about actually getting the raw
material — a few thousand news articles about specific companies — off the public web, in a
way that is reproducible, defensible, and doesn't get you blocked.

## Why scrape at all

The alternative is a ready-made dataset from Hugging Face or Kaggle. Two reasons this project
collects its own:

1. **Recency and specificity.** There's no off-the-shelf QA dataset about NIFTY 50 companies'
   last two quarters. The value of this model is that it knows *these* companies, and that
   knowledge has a shelf life measured in months.
2. **Provenance.** Because we collected each paragraph, we know exactly which article every
   training example came from — that's the `source_url` field in the schema. When someone asks
   "where did this answer come from?", there's an answer. Most public datasets can't do this.

The cost is that data collection becomes an engineering problem with its own failure modes,
which is what the rest of this guide is about.

## Discovery: sitemaps, not pagination

The naive scraper walks a listing page — `/news/business/page-2/`, `/page-3/`, and so on —
and follows every article link. That's what the original project did, and it's the thing
Moneycontrol's `robots.txt` explicitly disallows:

```
User-Agent: *
Disallow: /*/page-*/
```

The same `robots.txt` **advertises sitemaps** — machine-readable indexes of everything the
publisher wants crawlers to find:

```
Sitemap: https://www.moneycontrol.com/news/news-sitemap.xml
Sitemap: https://www.moneycontrol.com/news/index-sitemap-2026.xml
```

So we use those instead (`src/scraping/sitemaps.py`). The year index points to one sitemap per
month, each listing ~15,000 article URLs with publish dates. This is better on every axis:
it's the access path the publisher sanctioned, it's one request per 15,000 URLs instead of one
per 20, and the URLs come with dates so the corpus can be scoped to a time window.

**The general lesson**: before writing a crawler, read `robots.txt` — not just for what's
forbidden, but for what's offered. Publishers usually maintain a front door.

## Politeness: rate limiting, backoff, caching

`src/scraping/fetcher.py` wraps every request in three behaviours, and each one exists because
of a specific failure:

- **Rate limiting** — one request every ~3 seconds plus jitter. Moneycontrol's CDN starts
  returning `403` when requests arrive faster than that. A 403 here isn't a permission
  decision, it's a "slow down".
- **Exponential backoff** — on 403/429/5xx, wait, then double the wait. Without this, a
  throttled scraper interprets rate limiting as permanent failure and silently collects
  nothing.
- **On-disk caching** — every response is stored under `data/raw/http/`, keyed by a hash of
  the URL. Re-runs read from disk. This matters more than it sounds: during development you
  will run the pipeline dozens of times, and without a cache each run is a fresh burst of
  traffic at someone else's server. It also makes parser changes cheap to test — fix the
  parser, re-run over cached HTML, zero requests.

## The user-agent problem — an honest note

Standard scraping etiquette says: identify yourself in the `User-Agent` header, so the site
operator can see who you are and contact you.

Moneycontrol's CDN doesn't allow that. Any user-agent that isn't a recognised browser string
is rejected with `403` — including a descriptive one, and including a browser string with a
project token appended. The only header that gets through is a plain browser user-agent.

The compromise this project makes: send a normal browser user-agent, but identify the client in
an `X-Purpose` header (and an optional `From:` contact address from `.env`), which the WAF
doesn't police. It's weaker identification than the ideal, and it's written down here and in
`docs/dataset-design.md` rather than hidden. The other constraints — honoring every `Disallow`
rule, low request volume, caching, non-commercial use, no redistribution of article text — are
what actually carry the ethical weight, and none of them are relaxed.

This is worth being able to discuss: "we followed robots.txt" is a stronger claim than "we set
a polite UA string", and knowing the difference is the point.

## Entity matching: which company is this article about?

An article URL carries the headline as a slug:

```
/news/business/earnings/tata-steel-q1-results-profit-jumps-13990630.html
```

So the company is usually right there — but naive substring matching goes wrong fast, in ways
that quietly poison a dataset:

| Failure | Example | Fix in `matcher.py` |
|---------|---------|---------------------|
| Substring collisions | `itc` matches `switch` | Match whole slug **tokens**, never substrings |
| Sibling entities | `sbi-life-insurance` matched to State Bank of India | **Longest alias wins** — `sbi life` beats `sbi` |
| Same-brand, different company | `reliance-power` (Anil Ambani group) matched to Reliance Industries | Per-company **exclusion tokens** |
| Renames | Zomato is now "Eternal" | Alias list carries both names |
| Passing mentions | An article about a rival that names Infosys once | Second-stage check on the **parsed body** |

That last row is the important one architecturally: slug matching is a cheap first filter over
~90,000 URLs, and the expensive, accurate check only runs on articles we already fetched. Cheap
filter first, precise filter second, is the standard shape for this kind of pipeline.

## Extraction: the boilerplate problem

The article body sits in one container, but not every paragraph in it is article text. Mixed in
are newsletter pitches, "Also Read" links, disclaimers, and widget captions. If those survive
into `context` fields, the model learns to generate them — a fine-tuned model that emits
"Views and investment tips expressed by experts are their own" mid-answer is a data cleaning
bug, not a model bug.

`src/scraping/article.py` filters on two signals: **length** (a paragraph under ~90 characters
can't carry a fact worth asking about) and **shape** (a boilerplate prefix list, promo phrases,
link-dump punctuation). Being aggressive is the right default — we have far more paragraphs
than we need, so throwing away a good one costs nothing and keeping a bad one costs quality.

## Reproducibility

The run is resumable and idempotent: article URLs already in the output are skipped, HTTP
responses come from cache, and each run writes `data/raw/collection-report.json` with per
company counts. Because output is JSONL (one article per line), an interrupted run leaves a
valid file, and a re-run appends rather than rebuilds.

None of the raw article text is committed to the repo — `data/` is gitignored. What ships is
the *pipeline*, so anyone can regenerate the corpus, and the derived triplets used for
training.

## Talking points

- **"Why sitemaps instead of crawling?"** — robots.txt disallowed the paginated listings and
  advertised sitemaps; sitemaps are the sanctioned path, and 750× more efficient per request.
- **"How did you handle being rate-limited?"** — the 403s were throttling, not blocking:
  fixed delay with jitter, exponential backoff, and a disk cache so development iterations
  don't generate traffic at all.
- **"How do you know an article is about the right company?"** — token-level slug matching with
  longest-alias resolution and exclusion lists for sibling entities, then a second verification
  against the parsed body text.
- **"What did you do about data cleanliness?"** — filtered boilerplate at extraction time,
  because anything left in `context` is something the model learns to imitate.
- **"Is this legal/ethical?"** — non-commercial, robots-compliant, rate-limited, cached, no
  redistribution of source text; the one compromise (user-agent) is documented rather than
  glossed over.
