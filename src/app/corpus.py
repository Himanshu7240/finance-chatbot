"""Retrieval over the Day 4 news corpus - the context for everything that is not a price.

    python -m src.app.corpus "Why does Tata Steel owe Odisha money?"

Sparse TF-IDF, not embeddings, and the reasoning is in
docs/guides/06-retrieval-and-the-app-layer.md: these queries are carried by proper nouns
and numbers, where exact-token overlap *is* the relevance signal and IDF weights the rare
decisive token ("Odisha") on its own. A general-purpose sentence encoder would also place
"Q1 profit" next to "Q2 profit", which for financial questions is a bug.

Three things do most of the work, and all three were sized by measurement over 300 held-out
questions (Guide 06 has the tables):

* the **ticker pre-filter** - when a company resolves, search its ~200 paragraphs instead
  of all 10,257. The strongest distractor for any financial query is the same sentence
  about a different company, and this removes that class entirely: 53.3% -> 69.0%;
* **dropping the company name from the query** once that filter has consumed it, or the
  corpus-wide IDF that makes "tata steel" a decisive term makes it match everything inside
  TATASTEEL's own slice: 61.0% -> 70.0% at top-3;
* the **score floor**, which detects *retrieval failure* - below it, 4.5% of questions have
  their answer in the retrieved text, against 70.9% above. What it does not detect is an
  off-topic question: nonsense queries score 0.2-0.4 against a corpus this size, higher than
  many real ones. That job belongs to the entity check in pipeline.py.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..scraping.matcher import strip_company

log = logging.getLogger("corpus")

PARAGRAPHS_PATH = Path("data/processed/paragraphs.jsonl")

# Cosine below this means retrieval found nothing inside the company's own paragraphs.
# Measured on 300 held-out test questions: of the 22 that fall below 0.10, 4.5% had their
# answer in the retrieved text, against 70.9% of the 278 above it. Raising the floor to
# 0.15 would throw away 46 more questions that are answerable 38% of the time, so 0.10 is
# where the separation is. What this floor does *not* do is detect an off-topic question -
# see the guide, and the entity check in pipeline.py that does.
MIN_SCORE = 0.10
# Training contexts were single paragraphs, so the prompt stays close to that shape. The
# budget is the measured knee: on the same 300 questions the answer is present in 56% of
# contexts at 600 chars, 66% at 1200 and 69.7% at 1500, which is already the whole of
# top-3 (70.0%). Going to top-5 buys 5 more points for 50% more context.
CONTEXT_CHARS = 1500
TOP_K = 3


@dataclass(frozen=True)
class Passage:
    """One retrieved paragraph, with enough provenance for a user to check it."""

    para_id: str
    company: str
    ticker: str
    title: str
    url: str
    published: str
    text: str
    score: float

    def citation(self) -> str:
        return f"{self.title} ({self.published[:10]}) - {self.url}"


class ArticleRetriever:
    """TF-IDF search over the cleaned news paragraphs, optionally scoped to one company."""

    def __init__(
        self,
        paragraphs_path: str | Path = PARAGRAPHS_PATH,
        min_score: float = MIN_SCORE,
    ) -> None:
        self.paragraphs_path = Path(paragraphs_path)
        self.min_score = min_score
        self._rows: list[dict] = []
        self._matrix = None
        self._vectorizer = None
        self._by_ticker: dict[str, list[int]] = {}

    # Building is lazy so importing the app - or constructing the pipeline in a test -
    # does not pay for reading 10k paragraphs and fitting a vectorizer.
    def _ensure_index(self) -> None:
        if self._matrix is not None:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not self.paragraphs_path.exists():
            raise FileNotFoundError(
                f"{self.paragraphs_path} not found - run the Day 3/4 pipeline first "
                "(see data/README.md)"
            )
        with self.paragraphs_path.open(encoding="utf-8") as handle:
            self._rows = [json.loads(line) for line in handle if line.strip()]

        by_ticker = defaultdict(list)
        for index, row in enumerate(self._rows):
            by_ticker[row["ticker"]].append(index)
        self._by_ticker = dict(by_ticker)

        # Unigrams + bigrams: "profit" and "net profit" are different questions. Sublinear
        # tf stops a paragraph that repeats a company name from outranking one that
        # answers the question. min_df=2 drops scrape debris that appears exactly once.
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, min_df=2, stop_words="english",
        )
        self._matrix = self._vectorizer.fit_transform(row["text"] for row in self._rows)
        log.info(
            "indexed %d paragraphs, %d terms, %d companies",
            self._matrix.shape[0], self._matrix.shape[1], len(self._by_ticker),
        )

    @property
    def size(self) -> int:
        self._ensure_index()
        return len(self._rows)

    def search(self, question: str, ticker: str | None = None, top_k: int = TOP_K) -> list[Passage]:
        """Top passages for a question, best first, all scoring at or above the floor."""
        self._ensure_index()
        from sklearn.metrics.pairwise import linear_kernel

        candidates = self._by_ticker.get(ticker) if ticker else None
        if ticker and not candidates:
            log.debug("no paragraphs for %s, searching the whole corpus", ticker)
            candidates = None

        # The name only comes out of the query if the filter is actually going to apply;
        # in a whole-corpus search it is the most useful term there is.
        text = strip_company(question, ticker) if candidates is not None else question
        query = self._vectorizer.transform([text])
        if query.nnz == 0:                    # no query term survived the vocabulary
            return []

        matrix = self._matrix if candidates is None else self._matrix[candidates]
        # Both sides are L2-normalized by TfidfVectorizer, so the dot product is cosine.
        scores = linear_kernel(query, matrix).ravel()

        order = scores.argsort()[::-1][:top_k]
        passages = []
        for position in order:
            score = float(scores[position])
            if score < self.min_score:
                break                          # sorted, so everything after is worse too
            row = self._rows[position if candidates is None else candidates[position]]
            passages.append(
                Passage(
                    para_id=row["para_id"],
                    company=row["company"],
                    ticker=row["ticker"],
                    title=row.get("title", ""),
                    url=row.get("url", ""),
                    published=row.get("published", ""),
                    text=row["text"],
                    score=score,
                )
            )
        return passages


def build_context(passages: list[Passage], budget: int = CONTEXT_CHARS) -> str:
    """Passages as one context string, best first, truncated to a budget.

    The model was fine-tuned on single-paragraph contexts. Concatenating everything
    retrieved would drift from that shape and bury the answer span; the budget keeps the
    prompt close to what training looked like while leaving room for a second passage
    when the first one is short.
    """
    parts: list[str] = []
    used = 0
    for passage in passages:
        text = passage.text.strip()
        if parts and used + len(text) > budget:
            break
        parts.append(text)
        used += len(text) + 1
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question")
    parser.add_argument("--ticker", help="restrict to one company, e.g. TATASTEEL")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    retriever = ArticleRetriever()
    passages = retriever.search(args.question, ticker=args.ticker, top_k=args.top_k)
    if not passages:
        log.info("nothing above the score floor of %.2f", retriever.min_score)
        return
    for passage in passages:
        log.info("[%.3f] %s | %s", passage.score, passage.ticker, passage.text)
        log.info("        %s", passage.citation())


if __name__ == "__main__":
    main()
