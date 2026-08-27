"""Day 3 entry point: collect news articles for NIFTY 50 companies.

    python -m src.scraping.collect --months 6 --max-articles 40

Pipeline: discover URLs from published sitemaps -> match each URL to a company by
its headline slug -> fetch (rate-limited, cached, robots-checked) -> parse into
clean paragraphs -> append to ``data/raw/articles/<TICKER>.jsonl``.

The whole thing is resumable: URLs already present in the JSONL files are skipped,
and every HTTP response is cached on disk, so an interrupted run costs the site
nothing when restarted.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from .article import parse_article
from .fetcher import Blocked, Fetcher
from .matcher import Company, load_companies, match_company, mentions
from .sitemaps import month_articles, news_sitemap, recent_months

log = logging.getLogger("collect")

OUTPUT_DIR = Path("data/raw/articles")
REPORT_PATH = Path("data/raw/collection-report.json")


def existing_urls(ticker: str) -> set[str]:
    """URLs already collected for a company, so re-runs don't duplicate work."""
    path = OUTPUT_DIR / f"{ticker}.jsonl"
    if not path.exists():
        return set()
    urls = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                urls.add(json.loads(line)["url"])
            except (json.JSONDecodeError, KeyError):
                continue
    return urls


def discover(
    fetcher: Fetcher,
    companies: tuple[Company, ...],
    months: list[str],
    max_articles: int,
    use_news_sitemap: bool,
) -> dict[str, list[str]]:
    """Candidate article URLs per ticker, newest months first."""
    by_ticker: dict[str, list[str]] = defaultdict(list)
    already = {company.ticker: existing_urls(company.ticker) for company in companies}
    seen: set[str] = set().union(*already.values()) if already else set()

    sources: list[tuple[str, list[str]]] = []
    if use_news_sitemap:
        sources.append(("news-sitemap", news_sitemap(fetcher)))
    for month in months:
        sources.append((month, month_articles(fetcher, month)))

    for label, urls in sources:
        hits = 0
        for url in urls:
            if url in seen:
                continue
            company = match_company(url, companies)
            if company is None:
                continue
            if len(by_ticker[company.ticker]) + len(already[company.ticker]) >= max_articles:
                continue
            by_ticker[company.ticker].append(url)
            seen.add(url)
            hits += 1
        log.info("%s: %s articles, %s matched a company", label, len(urls), hits)

    return by_ticker


def collect(
    fetcher: Fetcher,
    companies: tuple[Company, ...],
    candidates: dict[str, list[str]],
) -> dict[str, dict]:
    """Fetch, parse and store candidate articles. Returns per-company stats."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    by_ticker = {company.ticker: company for company in companies}
    stats: dict[str, dict] = {}

    for ticker, urls in sorted(candidates.items()):
        company = by_ticker[ticker]
        kept = paragraphs = skipped = 0
        with (OUTPUT_DIR / f"{ticker}.jsonl").open("a", encoding="utf-8") as out:
            for url in urls:
                try:
                    html = fetcher.get(url)
                except Blocked:
                    log.warning("robots.txt disallows %s", url)
                    skipped += 1
                    continue
                if not html:
                    skipped += 1
                    continue

                article = parse_article(html, url)
                if article is None:
                    skipped += 1
                    continue
                # The slug said this was about the company; confirm the text agrees.
                if not mentions(f"{article.title} {article.text}", company):
                    skipped += 1
                    continue

                record = {"company": company.company, "ticker": ticker, **article.to_dict()}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
                paragraphs += len(article.paragraphs)

        stats[ticker] = {
            "company": company.company,
            "articles": kept,
            "paragraphs": paragraphs,
            "skipped": skipped,
        }
        log.info("%-12s %3d articles, %4d paragraphs (%d skipped)",
                 ticker, kept, paragraphs, skipped)

    return stats


def corpus_stats() -> dict:
    """Per-company totals for everything on disk, not just this run.

    The corpus is usually built over several runs (a first pass, then top-ups), so
    the run's own counts alone never describe what the dataset actually holds.
    """
    per_company: dict[str, dict] = {}
    oldest = newest = None
    for path in sorted(OUTPUT_DIR.glob("*.jsonl")):
        articles = paragraphs = 0
        company = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            company = record.get("company", "")
            articles += 1
            paragraphs += len(record.get("paragraphs", []))
            published = record.get("published")
            if published:
                oldest = min(oldest or published, published)
                newest = max(newest or published, published)
        if articles:
            per_company[path.stem] = {
                "company": company,
                "articles": articles,
                "paragraphs": paragraphs,
            }
    return {
        "totals": {
            "articles": sum(entry["articles"] for entry in per_company.values()),
            "paragraphs": sum(entry["paragraphs"] for entry in per_company.values()),
            "companies": len(per_company),
        },
        "published_range": [oldest, newest],
        "per_company": per_company,
    }


def write_report(stats: dict[str, dict], fetcher: Fetcher) -> dict:
    """Persist what this run collected, alongside the state of the whole corpus."""
    run_totals = {
        "articles": sum(entry["articles"] for entry in stats.values()),
        "paragraphs": sum(entry["paragraphs"] for entry in stats.values()),
        "skipped": sum(entry["skipped"] for entry in stats.values()),
        "companies_with_articles": sum(1 for entry in stats.values() if entry["articles"]),
    }
    payload = {
        "generated": date.today().isoformat(),
        "run": {"totals": run_totals, "http": dict(fetcher.stats), "per_company": stats},
        "corpus": corpus_stats(),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect NIFTY 50 news articles.")
    parser.add_argument("--months", type=int, default=6,
                        help="how many months of archive to walk back (default: 6)")
    parser.add_argument("--until", default=date.today().strftime("%Y-%m"),
                        help="newest month to include, YYYY-MM (default: this month)")
    parser.add_argument("--companies", nargs="*", metavar="TICKER",
                        help="restrict to these tickers (default: all 50)")
    parser.add_argument("--max-articles", type=int, default=40,
                        help="cap on stored articles per company (default: 40)")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="seconds between requests (default: 3.0)")
    parser.add_argument("--news-sitemap", action="store_true",
                        help="also pull today's news sitemap for a fresh top-up")
    parser.add_argument("--dry-run", action="store_true",
                        help="discover and report candidate counts without fetching articles")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    load_dotenv()          # picks up SCRAPER_CONTACT, if set
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")

    companies = load_companies()
    if args.companies:
        wanted = {ticker.upper() for ticker in args.companies}
        companies = tuple(c for c in companies if c.ticker.upper() in wanted)
        if not companies:
            raise SystemExit(f"no companies matched {sorted(wanted)}")

    months = recent_months(args.months, args.until)
    fetcher = Fetcher(delay=args.delay)
    log.info("discovering articles for %d companies across %s", len(companies), months)

    candidates = discover(fetcher, companies, months, args.max_articles, args.news_sitemap)
    found = sum(len(urls) for urls in candidates.values())
    log.info("%d new candidate articles across %d companies", found, len(candidates))

    if args.dry_run:
        for ticker, urls in sorted(candidates.items(), key=lambda item: -len(item[1])):
            log.info("%-12s %3d candidates", ticker, len(urls))
        return

    stats = collect(fetcher, companies, candidates)
    payload = write_report(stats, fetcher)
    run, corpus = payload["run"]["totals"], payload["corpus"]["totals"]
    log.info("this run: %d articles, %d paragraphs, %d/%d companies covered",
             run["articles"], run["paragraphs"], run["companies_with_articles"], len(companies))
    log.info("corpus now: %d articles, %d paragraphs across %d companies",
             corpus["articles"], corpus["paragraphs"], corpus["companies"])
    log.info("http: %s", fetcher.stats)


if __name__ == "__main__":
    main()
