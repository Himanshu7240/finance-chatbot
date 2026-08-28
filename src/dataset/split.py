"""Day 4 step 2: assign every article to a train/val/test split.

    python -m src.dataset.split

The important decision here is **splitting by article, never by paragraph**.

Paragraphs from one article overlap heavily - the same figures, the same quotes,
the same framing restated. Split them randomly and near-identical text lands in
both train and test, the model recognises what it already memorised, and the eval
score becomes fiction. Grouping by article keeps every paragraph of a story on one
side of the wall.

Splits are also stratified by company, so all 50 appear in train, val and test -
otherwise a company held out entirely would look like catastrophic failure at eval
time when it is really just absence.

Balancing is done on *paragraph* counts rather than article counts: articles vary
from 1 to 30 usable paragraphs, so equal article counts would give lopsided splits.

Output: ``data/processed/splits.json``, mapping article id -> split name.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

log = logging.getLogger("split")

PARAGRAPHS_PATH = Path("data/processed/paragraphs.jsonl")
OUTPUT_PATH = Path("data/processed/splits.json")

RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
SEED = 20260827


def article_sizes(paragraphs_path: Path = PARAGRAPHS_PATH):
    """Paragraph count per article, grouped by company ticker."""
    sizes: dict[str, int] = defaultdict(int)
    ticker_of: dict[str, str] = {}
    for line in paragraphs_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sizes[row["article_id"]] += 1
        ticker_of[row["article_id"]] = row["ticker"]

    by_ticker: dict[str, list[str]] = defaultdict(list)
    for article_id, ticker in ticker_of.items():
        by_ticker[ticker].append(article_id)
    return sizes, by_ticker


def load_existing(path: Path = OUTPUT_PATH) -> dict[str, str]:
    """Split assignments from a previous run, so they can be kept.

    Splits must be **sticky** across dataset revisions. If the corpus grows and every
    article is re-drawn, articles the previous model trained on can land in the new test
    set - and then run 2 looks better than run 1 for reasons that have nothing to do with
    the model. Comparing two training runs requires the same test set underneath both.
    """
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("splits", {})
    except (json.JSONDecodeError, KeyError):
        return {}


def assign(sizes, by_ticker, ratios=RATIOS, seed: int = SEED,
           existing: dict[str, str] | None = None):
    """Greedily place each company's articles into the split furthest below target.

    Largest articles first: placing the big ones while there is room left keeps the
    final paragraph counts close to the requested ratios. Articles named in ``existing``
    keep the split they already had; only new ones are drawn.
    """
    rng = random.Random(seed)
    existing = existing or {}
    splits: dict[str, str] = {}
    totals = {name: 0 for name in ratios}

    for ticker in sorted(by_ticker):
        articles = sorted(by_ticker[ticker], key=lambda a: (-sizes[a], a))
        rng.shuffle(articles)
        articles.sort(key=lambda a: -sizes[a])

        company_total = sum(sizes[a] for a in articles)
        placed = {name: 0 for name in ratios}

        # Honour prior assignments first, so the counters below account for them when
        # placing the new articles.
        for article_id in articles:
            if article_id in existing:
                name = existing[article_id]
                splits[article_id] = name
                placed[name] += sizes[article_id]
                totals[name] += sizes[article_id]

        for article_id in articles:
            if article_id in splits:
                continue
            # Deficit against this company's own target share, so stratification
            # holds per company rather than only in aggregate.
            name = max(
                ratios,
                key=lambda s: ratios[s] * company_total - placed[s],
            )
            splits[article_id] = name
            placed[name] += sizes[article_id]
            totals[name] += sizes[article_id]

    return splits, totals


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--paragraphs", type=Path, default=PARAGRAPHS_PATH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--reassign", action="store_true",
                        help="redraw every split from scratch, discarding previous "
                             "assignments (invalidates comparisons with earlier runs)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    existing = {} if args.reassign else load_existing()
    sizes, by_ticker = article_sizes(args.paragraphs)
    splits, totals = assign(sizes, by_ticker, seed=args.seed, existing=existing)
    if existing:
        kept = sum(1 for a in splits if a in existing)
        log.info("kept %d existing assignments, drew %d new", kept, len(splits) - kept)

    # A company missing from a split would make its eval numbers meaningless.
    per_company_splits = defaultdict(set)
    for article_id, name in splits.items():
        for ticker, articles in by_ticker.items():
            if article_id in articles:
                per_company_splits[ticker].add(name)
                break
    missing = {t: sorted(set(RATIOS) - s) for t, s in per_company_splits.items()
               if set(RATIOS) - s}

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"seed": args.seed, "totals": totals, "splits": splits}, indent=2),
        encoding="utf-8",
    )

    grand = sum(totals.values())
    for name, count in totals.items():
        log.info("%-5s %6d paragraphs (%.1f%%)", name, count, 100 * count / grand)
    log.info("%d articles assigned -> %s", len(splits), OUTPUT_PATH)
    if missing:
        log.warning("companies absent from a split: %s", missing)


if __name__ == "__main__":
    main()
