"""Day 4 step 1: turn the raw article corpus into QA-ready paragraphs.

    python -m src.dataset.clean

Three jobs, in order:

1. **Normalize** - NFKC, real spaces for non-breaking ones, straight quotes, tidy
   currency. Left alone, these become tokenizer noise the model spends capacity on
   for no gain.
2. **Keep only fact-bearing paragraphs** - a paragraph with no number, date or
   business event in it cannot produce an answerable question. This filter, more
   than any later step, decides how good the dataset turns out.
3. **Deduplicate** - wire copy repeats near-verbatim across outlets and days. Exact
   hashing misses that, so near-duplicates are caught with Jaccard overlap over word
   shingles, narrowed by an inverted index so it stays tractable.

Output: ``data/processed/paragraphs.jsonl``, one candidate context per line.
See docs/guides/03-cleaning-and-splitting-a-dataset.md for the reasoning.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

log = logging.getLogger("clean")

INPUT_DIR = Path("data/raw/articles")
OUTPUT_PATH = Path("data/processed/paragraphs.jsonl")
REPORT_PATH = Path("data/processed/clean-report.json")

MIN_CHARS = 90
MAX_CHARS = 1200
SHINGLE_SIZE = 5
JACCARD_THRESHOLD = 0.8

# Business events that make a paragraph answerable even with no digits in it.
_EVENT_WORDS = re.compile(
    r"\b(acquir\w+|acquisition|merger|merge\w*|appoint\w*|resign\w*|stepped down|"
    r"launch\w*|approv\w*|partner\w+|stake|divest\w*|demerger|expansion|expand\w*|"
    r"dividend|buyback|order win|contract|guidance|outlook|upgrad\w*|downgrad\w*|"
    r"rating|target price|invest\w*|capex|plant|facility|regulat\w*|penalt\w*|"
    r"probe|lawsuit|tribunal|licence|license|tariff|subsidiary)\b",
    re.IGNORECASE,
)
_HAS_DIGIT = re.compile(r"\d")

# Multi-company roundup articles ("Stocks to Watch Today: Reliance, Wipro, Axis Bank...").
# Each paragraph in these covers a *different* company, but every paragraph inherits the
# article's single company label - so "The company has secured domestic orders worth Rs 128
# crore" gets filed under Axis Bank when it belongs to Indian Hume Pipe. Generation then
# produces a confident false statement that groundedness checks cannot catch, because the
# answer really is supported by the paragraph. The only defence is to not use these
# paragraphs unless they name their own company outright.
_ROUNDUP_TITLE = re.compile(
    r"stocks to watch|buzzing stocks|top gainers|top losers|market live|closing bell|"
    r"opening bell|trade spotlight|stocks in news|in focus on|hot stocks|f&o cues|"
    r"market wrap|nifty today|share market (live|today)|stock radar|money morning",
    re.IGNORECASE,
)

# Paragraphs that are pure attribution or navigation leftovers.
_LOW_VALUE = re.compile(
    r"^(the company said|the statement said|it said|he said|she said|they said|"
    r"according to the (statement|filing)|in a (regulatory )?filing)[.,]?$",
    re.IGNORECASE,
)
_WORD = re.compile(r"[a-z0-9]+")

_REPLACEMENTS = {
    " ": " ",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": " - ",
    "…": "...",
    "₹": "Rs ",
}


@dataclass
class Paragraph:
    para_id: str
    article_id: str
    company: str
    ticker: str
    url: str
    title: str
    published: str | None
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize(text: str) -> str:
    """Canonical form for text that will end up in training data."""
    text = unicodedata.normalize("NFKC", text)
    for source, target in _REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"\bRs\.?\s*(?=\d)", "Rs ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_fact_bearing(text: str) -> bool:
    """Whether a paragraph carries something a question could be asked about."""
    if not (MIN_CHARS <= len(text) <= MAX_CHARS):
        return False
    if _LOW_VALUE.match(text):
        return False
    # A paragraph that is nothing but a quote is opinion, a poor answer target.
    if text.startswith('"') and text.endswith('"'):
        return False
    return bool(_HAS_DIGIT.search(text) or _EVENT_WORDS.search(text))


def _shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset[int]:
    words = _WORD.findall(text.lower())
    if len(words) < size:
        return frozenset({hash(" ".join(words))})
    return frozenset(hash(" ".join(words[i:i + size])) for i in range(len(words) - size + 1))


def deduplicate(paragraphs: list[Paragraph], threshold: float = JACCARD_THRESHOLD):
    """Drop exact and near-duplicate paragraphs, keeping the first occurrence.

    Comparing every pair would be ~160M comparisons at this corpus size, so
    candidates are narrowed with an inverted index: two paragraphs are only
    compared if they already share a shingle.
    """
    kept: list[Paragraph] = []
    seen_exact: set[str] = set()
    index: dict[int, list[int]] = defaultdict(list)
    signatures: list[frozenset[int]] = []
    dropped = {"exact": 0, "near": 0}

    for paragraph in paragraphs:
        key = " ".join(_WORD.findall(paragraph.text.lower()))
        if key in seen_exact:
            dropped["exact"] += 1
            continue
        seen_exact.add(key)

        signature = _shingles(paragraph.text)
        candidates = {i for shingle in signature for i in index[shingle]}
        if any(
            len(signature & signatures[i]) / len(signature | signatures[i]) >= threshold
            for i in candidates
        ):
            dropped["near"] += 1
            continue

        position = len(kept)
        kept.append(paragraph)
        signatures.append(signature)
        for shingle in signature:
            index[shingle].append(position)

    return kept, dropped


def load_corpus(input_dir: Path = INPUT_DIR):
    """Every stored article, paired with a stable article id."""
    for path in sorted(input_dir.glob("*.jsonl")):
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                yield json.loads(line), f"{path.stem}-{line_number}"
            except json.JSONDecodeError:
                log.warning("bad JSON at %s:%s", path, line_number)


def attributable(text: str, title: str, names_own, names_other) -> bool:
    """Whether this paragraph can be trusted to be about the company it is filed under.

    Two ways a label goes wrong, both caught here:

    * the paragraph names a *different* NIFTY 50 company and not its own - it is about
      that company, whatever the article was filed under;
    * the paragraph names nobody and came from a roundup, where the label is a guess.

    A paragraph that names its own company is always fine: the text speaks for itself.
    """
    if names_own(text):
        return True
    if names_other(text):
        return False
    return not (_ROUNDUP_TITLE.search(title) or _company_count(title) >= 3)


def _company_count(title: str) -> int:
    """How many NIFTY 50 companies a headline names - 3+ means it's a roundup."""
    try:
        from ..scraping.matcher import load_companies, mentions
    except (ImportError, FileNotFoundError):
        return 0
    return sum(1 for company in load_companies() if mentions(title, company))


def clean(input_dir: Path = INPUT_DIR, threshold: float = JACCARD_THRESHOLD):
    candidates: list[Paragraph] = []
    stats = {"articles": 0, "paragraphs_in": 0, "not_fact_bearing": 0, "unattributable": 0}

    try:
        from ..scraping.matcher import load_companies, mentions
        companies = load_companies()
        by_ticker = {c.ticker: c for c in companies}
    except (ImportError, FileNotFoundError):
        companies, by_ticker = (), {}

    for article, article_id in load_corpus(input_dir):
        stats["articles"] += 1
        title = normalize(article["title"])
        company = by_ticker.get(article["ticker"])
        if company is None:
            names_own = names_other = lambda text: False
        else:
            others = [c for c in companies if c.ticker != company.ticker]
            names_own = lambda text: mentions(text, company)
            names_other = lambda text: any(mentions(text, c) for c in others)

        for index, raw in enumerate(article["paragraphs"]):
            stats["paragraphs_in"] += 1
            text = normalize(raw)
            if not is_fact_bearing(text):
                stats["not_fact_bearing"] += 1
                continue
            if not attributable(text, title, names_own, names_other):
                stats["unattributable"] += 1
                continue
            candidates.append(
                Paragraph(
                    para_id=f"{article_id}-p{index}",
                    article_id=article_id,
                    company=article["company"],
                    ticker=article["ticker"],
                    url=article["url"],
                    title=normalize(article["title"]),
                    published=article.get("published"),
                    text=text,
                )
            )

    kept, dropped = deduplicate(candidates, threshold)
    stats.update(dropped)
    stats["paragraphs_out"] = len(kept)
    return kept, stats


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jaccard", type=float, default=JACCARD_THRESHOLD,
                        help="near-duplicate threshold, 0-1 (default: 0.8)")
    parser.add_argument("--input", type=Path, default=INPUT_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    kept, stats = clean(args.input, args.jaccard)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for paragraph in kept:
            out.write(json.dumps(paragraph.to_dict(), ensure_ascii=False) + "\n")

    per_company: dict[str, int] = defaultdict(int)
    for paragraph in kept:
        per_company[paragraph.ticker] += 1
    REPORT_PATH.write_text(
        json.dumps({"stats": stats, "per_company": dict(sorted(per_company.items()))}, indent=2),
        encoding="utf-8",
    )

    log.info("%(articles)s articles, %(paragraphs_in)s paragraphs in", stats)
    log.info("dropped %(not_fact_bearing)s not fact-bearing, %(unattributable)s "
             "unattributable, %(exact)s exact dupes, %(near)s near dupes", stats)
    log.info("kept %s paragraphs across %s companies -> %s",
             stats["paragraphs_out"], len(per_company), OUTPUT_PATH)


if __name__ == "__main__":
    main()
