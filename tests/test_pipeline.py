"""Routing, refusals and conversation state - the contract the UI depends on.

Every dependency is injected: a fake feed, a fake corpus, a fake generator. No network,
no dataset, no 6.4 GB of weights. A refusal that silently becomes an answer is the worst
regression this project can have and raises no exception anywhere, so it is tested here.
"""

from __future__ import annotations

import pytest

from src.app.corpus import Passage
from src.app.pipeline import (
    NO_COMPANY,
    NO_CORPUS,
    NO_PASSAGE,
    NO_PRICE_COMPANY,
    RAGPipeline,
    Route,
)
from src.app.router import IntentRouter
from src.app.stock import QuoteUnavailable, StockDataRetriever

FEED = {"lastPrice": 100.0, "previousClose": 99.0, "dayLow": 98.0, "dayHigh": 101.0,
        "currency": "INR"}


class FakeCorpus:
    """Returns a canned passage for anything, unless told to return nothing."""

    def __init__(self, passages=None):
        self.passages = passages if passages is not None else [
            Passage(para_id="X-0-p0", company="Tata Steel", ticker="TATASTEEL",
                    title="Tata Steel Q1", url="https://example.invalid/a",
                    published="2026-07-31T10:00:00+05:30",
                    text="Tata Steel reported a Q1 loss of Rs 340.7 crore.", score=0.42)
        ]
        self.calls = []

    def search(self, question, ticker=None, top_k=3):
        self.calls.append((question, ticker))
        return list(self.passages)


def build(generator=None, feed=FEED, corpus=None):
    def fetch(symbol):
        if isinstance(feed, Exception):
            raise feed
        return feed

    return RAGPipeline(
        generator=generator,
        stock=StockDataRetriever(fetch=fetch),
        corpus=corpus if corpus is not None else FakeCorpus(),
        router=IntentRouter(use_embeddings=False),
    )


class TestRouting:
    @pytest.mark.parametrize("question", [
        "what is tata steel trading at",
        "tata steel share price",
        "is tata steel up or down today",
    ])
    def test_price_wording_plus_a_company_goes_to_the_feed(self, question):
        assert build().answer(question).route is Route.LIVE_PRICE

    @pytest.mark.parametrize("question", [
        "why is tata steel restructuring in the uk",
        "what did the tata steel md announce",
    ])
    def test_everything_else_with_a_company_goes_to_the_corpus(self, question):
        assert build().answer(question).route is Route.CORPUS

    def test_a_past_tense_price_question_goes_to_the_corpus_not_the_feed(self):
        # The feed only knows about now; answering "what was it" from now answers a
        # different question than the one asked.
        answer = build().answer("what was tata steel's share price after q1 results")
        assert answer.route is Route.CORPUS

    def test_the_corpus_search_is_scoped_to_the_resolved_company(self):
        corpus = FakeCorpus()
        build(corpus=corpus).answer("why is tata steel restructuring")
        assert corpus.calls[0][1] == "TATASTEEL"


class TestRefusals:
    def test_a_price_question_with_no_company_refuses_instead_of_reading_the_news(self):
        # The expensive failure: a three-week-old price presented as today's.
        answer = build().answer("what is the share price")
        assert answer.route is Route.REFUSED
        assert answer.text == NO_PRICE_COMPANY

    def test_an_off_topic_question_refuses(self):
        answer = build().answer("who won the 2026 football world cup")
        assert answer.route is Route.REFUSED
        assert answer.text == NO_COMPANY

    def test_an_empty_question_refuses(self):
        assert build().answer("   ").route is Route.REFUSED

    def test_nothing_retrieved_refuses_rather_than_handing_over_a_weak_passage(self):
        answer = build(corpus=FakeCorpus([])).answer("why is tata steel restructuring")
        assert answer.route is Route.REFUSED
        assert answer.text == NO_PASSAGE

    def test_a_feed_outage_refuses_and_names_the_reason(self):
        answer = build(feed=QuoteUnavailable("simulated outage")).answer(
            "what is tata steel trading at"
        )
        assert answer.route is Route.REFUSED
        assert "Couldn't reach the price feed" in answer.text
        assert "simulated outage" in answer.reason

    def test_a_refusal_never_calls_the_model(self):
        def generator(question, context):
            raise AssertionError("the model must not be consulted without grounding")

        assert build(generator=generator).answer("who won the world cup").route is Route.REFUSED


class TestEvidenceAndSources:
    def test_without_a_generator_the_answer_is_the_retrieved_evidence(self):
        answer = build().answer("why is tata steel restructuring")
        assert answer.text == "Tata Steel reported a Q1 loss of Rs 340.7 crore."

    def test_the_generator_receives_the_retrieved_context(self):
        seen = {}

        def generator(question, context):
            seen["question"], seen["context"] = question, context
            return "Rs 340.7 crore"

        answer = build(generator=generator).answer("what was the q1 loss for tata steel")
        assert seen["context"] == "Tata Steel reported a Q1 loss of Rs 340.7 crore."
        assert answer.text == "Rs 340.7 crore"

    def test_a_broken_generator_falls_back_to_the_evidence_rather_than_erroring(self):
        def generator(question, context):
            raise RuntimeError("CUDA out of memory")

        answer = build(generator=generator).answer("why is tata steel restructuring")
        assert answer.route is Route.CORPUS
        assert answer.text == "Tata Steel reported a Q1 loss of Rs 340.7 crore."

    def test_corpus_answers_cite_the_article_with_its_date(self):
        answer = build().answer("why is tata steel restructuring")
        assert answer.sources == ["Tata Steel Q1 (2026-07-31) - https://example.invalid/a"]

    def test_duplicate_articles_are_cited_once(self):
        passage = FakeCorpus().passages[0]
        answer = build(corpus=FakeCorpus([passage, passage])).answer("why restructuring tata steel")
        assert len(answer.sources) == 1

    def test_live_answers_cite_the_feed_and_the_fetch_time(self):
        answer = build().answer("what is tata steel trading at")
        assert answer.sources[0].startswith("yfinance TATASTEEL.NS, fetched ")


class TestConversationState:
    def test_an_explicit_company_always_wins_over_the_previous_one(self):
        pipeline = build()
        first = pipeline.answer("what is tata steel trading at")
        second = pipeline.answer("and wipro?", last_company=first.company,
                                 last_route=first.route)
        assert second.company.ticker == "WIPRO"
        assert not second.assumed_company

    def test_a_bare_company_switch_inherits_the_previous_route(self):
        pipeline = build()
        first = pipeline.answer("what is tata steel trading at")
        second = pipeline.answer("and wipro?", last_company=first.company,
                                 last_route=first.route)
        assert second.route is Route.LIVE_PRICE

    def test_a_pronoun_follow_up_inherits_the_company_and_says_so(self):
        pipeline = build()
        first = pipeline.answer("why is tata steel restructuring")
        second = pipeline.answer("what is its share price", last_company=first.company,
                                 last_route=first.route)
        assert second.route is Route.LIVE_PRICE
        assert second.company.ticker == "TATASTEEL"
        assert second.assumed_company
        assert second.assumption == "Assuming Tata Steel, from your last question."

    def test_an_off_topic_question_does_not_borrow_the_sticky_company(self):
        # Without the follow-up gate this would answer from Tata Steel's paragraphs.
        pipeline = build()
        first = pipeline.answer("why is tata steel restructuring")
        second = pipeline.answer("who won the 2026 football world cup",
                                 last_company=first.company, last_route=first.route)
        assert second.route is Route.REFUSED
        assert second.company is None

    def test_no_history_means_no_assumption(self):
        answer = build().answer("what is its share price")
        assert answer.route is Route.REFUSED
        assert answer.assumption == ""


class TestMissingCorpus:
    """A fresh clone has no data/processed/ - gitignored article text we don't redistribute."""

    class AbsentCorpus:
        def search(self, question, ticker=None, top_k=3):
            raise FileNotFoundError("data/processed/paragraphs.jsonl not found")

    def test_a_corpus_question_refuses_instead_of_crashing(self):
        answer = build(corpus=self.AbsentCorpus()).answer("why is tata steel restructuring")
        assert answer.route is Route.REFUSED
        assert "isn't built on this machine" in answer.text

    def test_live_prices_still_work_without_a_corpus(self):
        answer = build(corpus=self.AbsentCorpus()).answer("what is tata steel trading at")
        assert answer.route is Route.LIVE_PRICE
