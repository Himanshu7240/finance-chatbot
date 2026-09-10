"""The RAG pipeline: question in, grounded answer plus its sources out.

    python -m src.app.pipeline "What is Tata Steel trading at?"

The fine-tune taught one skill - pull the answer span out of a passage it is given
(Guide 01). This module's whole job is to make sure the model never runs without such a
passage, and to refuse out loud when it cannot find one. Routing:

    price wording AND a resolved NIFTY 50 company   -> live quote from yfinance
    a resolved company, anything else               -> the Day 4 news corpus
    no company resolved                             -> refuse, and say why

That last line is a deliberate product decision backed by measurement, not caution for its
own sake. See docs/guides/06-retrieval-and-the-app-layer.md: on 300 held-out questions the
company matcher resolved 99.7% of them and never picked the wrong company, while off-topic
questions ("who won the world cup?") resolve none - but they still score 0.2-0.4 cosine
against *something* in a 10,257-paragraph news corpus. The entity check separates in-domain
from out-of-domain; the similarity score does not.

Without a generator the pipeline still works and answers from the retrieved text directly.
That is what makes Day 9 runnable, and testable, without a 6.4 GB download.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass, field
from enum import Enum

from ..scraping.matcher import Company, strip_company
from .corpus import ArticleRetriever, Passage, build_context
from .router import Intent, IntentRouter
from .stock import Quote, QuoteUnavailable, StockDataRetriever

log = logging.getLogger("pipeline")


class Route(str, Enum):
    LIVE_PRICE = "live_price"
    CORPUS = "corpus"
    REFUSED = "refused"


NO_COMPANY = (
    "I can only answer questions about the 50 NIFTY companies in my sources - "
    "name one and I'll look it up."
)
NO_PRICE_COMPANY = "Which company's price? I can look up any NIFTY 50 stock."
NO_PASSAGE = "I don't have anything on that in my sources."

# What makes a question a *follow-up* rather than a new one. Only these inherit the previous
# turn's company; without the gate, every off-topic question would borrow it and get answered
# from that company's paragraphs instead of refused.
_FOLLOW_UP_RE = re.compile(
    r"^\s*(and|what about|how about|ok(ay)?[, ]+and)\b"
    r"|\b(it|its|it's|they|their|them|the same|that one)\b",
    re.IGNORECASE,
)
# Words that carry no question of their own. If nothing but these is left after the company
# name is removed, the turn is a bare entity switch - "and Wipro?" - and inherits the last
# route rather than being sent to the corpus with an empty query.
_FILLER = {"and", "what", "about", "how", "ok", "okay", "the", "a", "an", "of", "for", "now",
           "then", "also", "too", "please", "is", "are", "was", "s", "so", "well", "hey"}
_WORD = re.compile(r"[a-z0-9]+")


def _is_follow_up(question: str) -> bool:
    """Whether a question refers back to the previous turn rather than standing alone."""
    return bool(_FOLLOW_UP_RE.search(question)) or len(_WORD.findall(question.lower())) <= 4


def _is_bare_mention(question: str, ticker: str) -> bool:
    """Whether the question is only a company name plus filler - "and Wipro?"."""
    remainder = strip_company(question, ticker)
    if remainder.strip().lower() == question.strip().lower():
        return False                      # nothing was stripped, so the name was not there
    return not (set(_WORD.findall(remainder.lower())) - _FILLER)


@dataclass
class Answer:
    """An answer and everything needed to check it."""

    text: str
    route: Route
    question: str
    company: Company | None = None
    context: str = ""
    passages: list[Passage] = field(default_factory=list)
    quote: Quote | None = None
    reason: str = ""
    #: True when the company came from the previous turn rather than from this question.
    assumed_company: bool = False

    @property
    def assumption(self) -> str:
        """What the answer silently relied on, phrased for the user - or nothing."""
        if self.assumed_company and self.company is not None:
            return f"Assuming {self.company.company}, from your last question."
        return ""

    @property
    def sources(self) -> list[str]:
        """One line per source, de-duplicated - two passages often share an article."""
        if self.quote is not None:
            return [f"yfinance {self.quote.symbol}, fetched {self.quote.fetched_at_text}"]
        return list(dict.fromkeys(passage.citation() for passage in self.passages))


class RAGPipeline:
    """Routes a question to a retriever, then to the model, then reports its sources."""

    def __init__(
        self,
        generator=None,
        stock: StockDataRetriever | None = None,
        corpus: ArticleRetriever | None = None,
        router: IntentRouter | None = None,
        top_k: int = 3,
    ) -> None:
        self.generator = generator            # callable(question, context) -> str
        self.stock = stock or StockDataRetriever()
        self.corpus = corpus or ArticleRetriever()
        self.router = router or IntentRouter()
        self.top_k = top_k

    def answer(self, question: str, last_company: Company | None = None,
               last_route: "Route | None" = None) -> Answer:
        """Answer one question, with the previous turn's company and route for context.

        The model was fine-tuned on single turns and has no chat template, so conversation
        state never enters the prompt (Guide 07). Two narrow pieces of it live here instead,
        because "and Wipro?" is how people actually ask:

        * a question that names no company inherits the previous one - but only if it reads
          as a follow-up, so an off-topic question is still refused rather than answered
          from whichever company was last mentioned;
        * a question that is *only* a company name inherits the previous route, so a price
          question followed by a bare name stays a price question.

        Neither ever overrides something the question says explicitly, and the answer records
        what it assumed so the interface can say so.
        """
        question = question.strip()
        if not question:
            return Answer(NO_COMPANY, Route.REFUSED, question, reason="empty question")

        intent = self.router.classify(question)
        company = self.stock.resolve(question)
        assumed = False
        if company is None and last_company is not None and _is_follow_up(question):
            company, assumed = last_company, True

        # "What is Reliance trading at?" -> "and Wipro?" is a new company with the old
        # question. Without this the second turn reaches the corpus with an empty query.
        if (company is not None and not intent.is_price and last_route is Route.LIVE_PRICE
                and _is_bare_mention(question, company.ticker)):
            intent = Intent(True, "follow-up to a price question")

        if intent.is_price:
            if company is None:
                # Never fall through to the corpus here: it would answer a "what is it
                # trading at" question with a price printed in a news story weeks ago.
                return Answer(NO_PRICE_COMPANY, Route.REFUSED, question, reason=intent.reason)
            return self._answer_live(question, company, intent.reason, assumed)

        if company is None:
            return Answer(NO_COMPANY, Route.REFUSED, question, reason="no company resolved")
        return self._answer_corpus(question, company, assumed)

    def _answer_live(self, question: str, company: Company, reason: str,
                     assumed: bool = False) -> Answer:
        try:
            quote = self.stock.quote(company)
        except QuoteUnavailable as exc:
            log.warning("quote failed: %s", exc)
            return Answer(
                f"Couldn't reach the price feed for {company.company} just now.",
                Route.REFUSED, question, company=company, reason=str(exc),
            )

        context = quote.as_context()
        # With no model loaded the quote sentence *is* the answer - it was written to be
        # read by a person as much as by the model.
        text = self._generate(question, context) or context
        return Answer(text, Route.LIVE_PRICE, question, company=company,
                      context=context, quote=quote, reason=reason, assumed_company=assumed)

    def _answer_corpus(self, question: str, company: Company,
                       assumed: bool = False) -> Answer:
        passages = self.corpus.search(question, ticker=company.ticker, top_k=self.top_k)
        if not passages:
            return Answer(NO_PASSAGE, Route.REFUSED, question, company=company,
                          reason="nothing above the score floor", assumed_company=assumed)

        context = build_context(passages)
        # Falling back to the passage itself is honest: it is the retrieved evidence,
        # unedited, rather than a guess about what the model would have extracted.
        text = self._generate(question, context) or passages[0].text
        return Answer(text, Route.CORPUS, question, company=company,
                      context=context, passages=passages,
                      reason=f"top score {passages[0].score:.3f}", assumed_company=assumed)

    def _generate(self, question: str, context: str) -> str | None:
        if self.generator is None:
            return None
        try:
            return self.generator(question, context).strip() or None
        except Exception as exc:              # a broken model must not lose the evidence
            log.warning("generation failed (%s); returning the retrieved text", exc)
            return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--model", nargs="?", const="", default=None,
                        help="load the fine-tuned model (optionally a model id); "
                             "without it, answers are the retrieved text itself")
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-embeddings", action="store_true",
                        help="route with the keyword rule only")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The encoder logs every Hub request at INFO; that is not this CLI's output.
    for noisy in ("httpx", "sentence_transformers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    generator = None
    if args.model is not None:
        from .model import DEFAULT_MODEL_ID, FinanceLLM

        generator = FinanceLLM(model_id=args.model or DEFAULT_MODEL_ID, device=args.device)

    pipeline = RAGPipeline(
        generator=generator,
        router=IntentRouter(use_embeddings=not args.no_embeddings),
        top_k=args.top_k,
    )
    last_company = None
    for question in args.questions:
        answer = pipeline.answer(question)
        log.info("\nQ %s", question)
        log.info("  route   %s (%s)", answer.route.value, answer.reason)
        log.info("  answer  %s", answer.text)
        for source in answer.sources:
            log.info("  source  %s", source)


if __name__ == "__main__":
    main()
