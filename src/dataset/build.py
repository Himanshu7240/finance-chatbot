"""Day 4 final step: assemble validated triplets into the training dataset.

    python -m src.dataset.build

Inputs:
  * ``data/processed/paragraphs.jsonl``  - cleaned contexts (src.dataset.clean)
  * ``data/processed/splits.json``       - article -> split map (src.dataset.split)
  * ``data/processed/generations.jsonl`` - raw model output from the Kaggle notebook,
                                           one ``{"para_id", "response"}`` per line

Output: ``data/processed/{train,val,test}.json`` in the Day 2 :class:`QAExample`
schema, plus a report showing exactly why every rejected pair was rejected.

Splits are inherited from the *paragraph's article*, never re-drawn here - that is
what keeps two questions generated from the same article out of train and test at
once. See src/dataset/split.py for why that matters.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path

from .qa_validate import clean_answer, parse_response, validate
from .schema import QAExample

log = logging.getLogger("build")

PROCESSED = Path("data/processed")
PARAGRAPHS_PATH = PROCESSED / "paragraphs.jsonl"
SPLITS_PATH = PROCESSED / "splits.json"
GENERATIONS_PATH = PROCESSED / "generations.jsonl"
REPORT_PATH = PROCESSED / "dataset-report.json"

_WORD = re.compile(r"[a-z0-9]+")


def _normalized(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def load_paragraphs(path: Path = PARAGRAPHS_PATH) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["para_id"]] = row
    return rows


def load_generations(path: Path = GENERATIONS_PATH):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def company_matcher(company: str, ticker: str):
    """A callable that says whether a question names this company.

    Built from the scraper's alias table when available - so "Airtel" counts for
    Bharti Airtel - and falling back to a plain name check if it isn't importable
    (the notebook environment doesn't need the scraping package).
    """
    try:
        from ..scraping.matcher import load_companies, mentions

        for entry in load_companies():
            if entry.ticker == ticker:
                return lambda text: mentions(text, entry)
    except (ImportError, FileNotFoundError):
        pass

    needle = _normalized(company)
    first = needle.split()[0] if needle else ""
    return lambda text: needle in _normalized(text) or (
        len(first) > 3 and first in _normalized(text).split()
    )


def build(paragraphs: dict[str, dict], splits: dict[str, str], generations,
          max_per_company: int | None = None):
    """Validate every generated pair and group the survivors by split."""
    by_split: dict[str, list[QAExample]] = defaultdict(list)
    rejections: Counter[str] = Counter()
    per_company: Counter[str] = Counter()
    seen_questions: set[str] = set()
    stats = Counter()

    matchers: dict[str, object] = {}

    for generation in generations:
        stats["responses"] += 1
        paragraph = paragraphs.get(generation.get("para_id", ""))
        if paragraph is None:
            rejections["unknown_paragraph"] += 1
            continue

        pairs = parse_response(generation.get("response", ""))
        if pairs is None:
            rejections["unparseable_json"] += 1
            continue
        if not pairs:
            # The model was allowed to decline a thin paragraph; that is a success
            # of the prompt, not a failure, so it is counted separately.
            stats["declined"] += 1
            continue

        ticker = paragraph["ticker"]
        if ticker not in matchers:
            matchers[ticker] = company_matcher(paragraph["company"], ticker)
        mentions_company = matchers[ticker]

        for pair in pairs:
            stats["pairs"] += 1
            reason = validate(pair, paragraph["text"], mentions_company)
            if reason:
                rejections[reason] += 1
                continue

            question = pair["question"].strip()
            key = _normalized(question)
            if key in seen_questions:
                rejections["duplicate_question"] += 1
                continue
            if max_per_company is not None and per_company[ticker] >= max_per_company:
                rejections["company_cap"] += 1
                continue

            seen_questions.add(key)
            per_company[ticker] += 1
            split = splits.get(paragraph["article_id"], "train")
            by_split[split].append(
                QAExample(
                    company=paragraph["company"],
                    ticker=ticker,
                    question=question,
                    answer=clean_answer(pair["answer"]),
                    context=paragraph["text"],
                    source_url=paragraph["url"],
                )
            )

    return by_split, rejections, per_company, stats


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--generations", type=Path, default=GENERATIONS_PATH)
    parser.add_argument("--paragraphs", type=Path, default=PARAGRAPHS_PATH)
    parser.add_argument("--splits", type=Path, default=SPLITS_PATH)
    parser.add_argument("--max-per-company", type=int, default=None,
                        help="cap examples per company, to even out coverage")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.generations.exists():
        raise SystemExit(
            f"{args.generations} not found - run notebooks/qa_generation_qwen.ipynb "
            "on Kaggle first, then download its output here."
        )

    paragraphs = load_paragraphs(args.paragraphs)
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    by_split, rejections, per_company, stats = build(
        paragraphs, splits, load_generations(args.generations), args.max_per_company
    )

    totals = {}
    for split, examples in sorted(by_split.items()):
        path = PROCESSED / f"{split}.json"
        path.write_text(
            json.dumps([example.to_dict() for example in examples], indent=2,
                       ensure_ascii=False),
            encoding="utf-8",
        )
        totals[split] = len(examples)
        log.info("%-5s %6d examples -> %s", split, len(examples), path)

    kept = sum(totals.values())
    REPORT_PATH.write_text(
        json.dumps(
            {
                "totals": totals,
                "kept": kept,
                "generated_pairs": stats["pairs"],
                "responses": stats["responses"],
                "declined_paragraphs": stats["declined"],
                "rejections": dict(rejections.most_common()),
                "per_company": dict(sorted(per_company.items())),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rejected = sum(rejections.values())
    log.info("kept %d of %d generated pairs (%.1f%% rejected)",
             kept, stats["pairs"], 100 * rejected / max(stats["pairs"], 1))
    for reason, count in rejections.most_common():
        log.info("  rejected %-20s %5d", reason, count)
    log.info("report -> %s", REPORT_PATH)


if __name__ == "__main__":
    main()
