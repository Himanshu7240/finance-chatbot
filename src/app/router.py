"""Deciding whether a question wants a live price or the news corpus.

    python -m src.app.router "what is reliance trading at"

The project report specifies embedding similarity against prototype price questions, and
that is the primary path here. It is backed by a lexical rule that is a real fallback, not
a stub: `sentence-transformers` pulls a ~90 MB model on first use, and the app has to start
and route sensibly without it.

The two signals are combined with OR, then vetoed by tense. Reasoning, and the numbers, in
docs/guides/06-retrieval-and-the-app-layer.md:

* **OR, not AND** - they have complementary error profiles. On a 36-question battery the
  keyword rule fired on no news question at all but missed 11 of 18 price questions
  ("wipro price", "what am i paying for a share of trent"); the encoder recovers most of
  those. High-precision rule, high-recall model, union.
* **The company name comes out first.** Embedding "how is TCS doing today" scores 0.54
  against the price prototypes; embedding "how is doing today" scores 0.79. The name pulls
  the vector towards "a company" and away from "a price question", exactly as it pulls
  corpus search towards "any paragraph about this company".
* **The tense veto** - "what *was* the price after Q1 results" is a question about the
  past, and the live feed cannot answer it however price-shaped it looks.

Intent is only half of a route: :class:`~src.app.pipeline.RAGPipeline` also requires a
resolved NIFTY 50 company before it will call the price feed.
"""

from __future__ import annotations

import argparse
import logging
import re
import threading
from dataclasses import dataclass

from ..scraping.matcher import match_text, strip_company

log = logging.getLogger("router")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Measured, not guessed. On a 36-question battery (18 price, 18 news) the company-stripped
# question scores a median 0.66 against the prototypes when it is a price question and 0.20
# when it is not; 0.40 sits in that gap and gives the best combined accuracy, 34/36. The
# 0.55 that felt right by eye scores 32/36 - it drops "how is TCS doing today" at 0.54.
SIMILARITY_THRESHOLD = 0.40

# The shapes a live-price question comes in. Deliberately varied in wording - they are the
# encoder's whole notion of the class.
PRICE_PROTOTYPES = [
    "what is the share price",
    "what is the stock trading at right now",
    "how much does one share cost today",
    "give me the current quote",
    "is the stock up or down today",
    "what is the market price of this company",
    "how is the stock doing today",
    "current valuation of the share",
]

_PRICE_RE = re.compile(
    r"\b(share|stock)\s*price\b|\bprice\s*of\b|\btrading\s+at\b|\bcurrent\s+price\b"
    r"|\bstock\s+quote\b|\bquote\s+for\b|\bmarket\s+price\b|\bshare\s+value\b"
    r"|\b(up|down)\s+today\b|\bltp\b|\bhow\s+much\s+.*\b(share|stock)s?\b",
    re.IGNORECASE,
)

# Past-tense and period markers. A price question in the past tense is a corpus question:
# the feed only knows about now, and answering it from now would be answering a different
# question than the one asked.
_HISTORICAL_RE = re.compile(
    r"\b(was|were|had|did)\b|\byesterday\b|\blast\s+(week|month|year|quarter)\b"
    r"|\bq[1-4]\b|\bfy\s?\d{2,4}\b|\bin\s+(19|20)\d{2}\b|\bafter\s+the\b|\bhistor",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Intent:
    """Why the router decided what it decided - carried through for logging and the UI."""

    is_price: bool
    reason: str
    score: float | None = None


class IntentRouter:
    """Price question or not. Embeddings when available, keywords when not."""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        threshold: float = SIMILARITY_THRESHOLD,
        use_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self.use_embeddings = use_embeddings
        self._model = None
        self._prototypes = None
        self._embedding_failed = False
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Start loading the encoder in the background.

        Loading reaches the Hugging Face Hub even when the model is cached, and when the Hub
        is unreachable its client retries with backoff - measured at *hours*, not seconds.
        Doing that inside the first request looks like a hung app; doing it synchronously at
        startup looks like an app that will not start. So it happens on a daemon thread, and
        every question that arrives before it finishes is routed by the keyword rule.
        """
        if self._model is None and not self._embedding_failed and self.use_embeddings:
            threading.Thread(target=self._load, name="router-warm", daemon=True).start()

    def _load(self) -> None:
        """Load the encoder once. Falls back permanently and quietly if it is unavailable."""
        with self._lock:
            if self._model is not None or self._embedding_failed:
                return
            try:
                from sentence_transformers import SentenceTransformer

                try:
                    # A cached model loads instantly and offline. Only reach the network if
                    # it genuinely is not on disk yet.
                    model = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception:
                    log.info("downloading the intent-router encoder (%s)", self.model_name)
                    model = SentenceTransformer(self.model_name)
                self._prototypes = model.encode(
                    PRICE_PROTOTYPES, normalize_embeddings=True, show_progress_bar=False
                )
                self._model = model
            except Exception as exc:            # not installed, no network, no disk space
                log.info("embedding router unavailable (%s); using the keyword rule", exc)
                self._embedding_failed = True
                self._model = None

    def _ensure_model(self):
        """The encoder if it is ready, or ``None`` - never blocking behind a warm-up."""
        if self._model is not None or self._embedding_failed or not self.use_embeddings:
            return self._model
        if not self._lock.acquire(blocking=False):
            # A warm-up holds the lock. This turn takes the keyword rule rather than
            # waiting on a network call of unknown length.
            return None
        self._lock.release()
        self._load()
        return self._model

    def similarity(self, question: str) -> float | None:
        """Highest cosine against the price prototypes, or ``None`` without an encoder.

        The company name is removed first - see the module docstring; it costs the price
        prototypes about 0.15 of cosine and buys nothing, since the pipeline resolves the
        company separately anyway.
        """
        model = self._ensure_model()
        if model is None:
            return None
        company = match_text(question)
        text = strip_company(question, company.ticker if company else None)
        vector = model.encode([text], normalize_embeddings=True, show_progress_bar=False)
        return float((vector @ self._prototypes.T).max())

    def classify(self, question: str) -> Intent:
        if _HISTORICAL_RE.search(question):
            return Intent(False, "past tense - the live feed cannot answer it")

        if _PRICE_RE.search(question):
            return Intent(True, "price wording")

        score = self.similarity(question)
        if score is None:
            return Intent(False, "no price wording (keyword rule)")
        if score >= self.threshold:
            return Intent(True, f"similar to a price question ({score:.2f})", score)
        return Intent(False, f"unlike the price prototypes ({score:.2f})", score)

    def is_price_question(self, question: str) -> bool:
        return self.classify(question).is_price


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--no-embeddings", action="store_true", help="keyword rule only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The encoder logs every Hub request at INFO; that is not this CLI's output.
    for noisy in ("httpx", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    router = IntentRouter(use_embeddings=not args.no_embeddings)
    for question in args.questions:
        intent = router.classify(question)
        log.info("%-5s %-55s %s", "PRICE" if intent.is_price else "NEWS", question, intent.reason)


if __name__ == "__main__":
    main()
