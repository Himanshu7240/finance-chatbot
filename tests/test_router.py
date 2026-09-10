"""Intent routing, and the guarantee that a slow encoder never blocks a question.

The keyword rule and the tense veto are pure functions of the text, so they are tested
directly. The encoder is not loaded here - what is tested is what happens *while* it loads,
which is the case that produced a 19-second hang in the Day 12 demo.
"""

from __future__ import annotations

import time

import pytest

from src.app.router import IntentRouter


@pytest.fixture
def keyword_router():
    return IntentRouter(use_embeddings=False)


@pytest.mark.parametrize("question", [
    "what is reliance trading at",
    "tata steel share price",
    "is itc up or down today",
    "give me the current quote for maruti",
    "tell me the ltp of tata steel",
])
def test_price_wording_is_recognised(keyword_router, question):
    assert keyword_router.classify(question).is_price


@pytest.mark.parametrize("question", [
    "why is tata steel restructuring in the uk",
    "who is the ceo of axis bank",
    "explain grasim's demerger",
])
def test_news_questions_are_not_price_questions(keyword_router, question):
    assert not keyword_router.classify(question).is_price


@pytest.mark.parametrize("question", [
    "what was tata steel's share price after q1 results",
    "what did the tcs ceo announce",
    "how did reliance stock do last week",
    "what was the share price in 2025",
])
def test_the_tense_veto_sends_historical_questions_to_the_corpus(keyword_router, question):
    # The live feed only knows about now, so a past-tense price question is a corpus question
    # however price-shaped it looks.
    intent = keyword_router.classify(question)
    assert not intent.is_price
    assert "past tense" in intent.reason


def test_without_embeddings_similarity_is_none(keyword_router):
    assert keyword_router.similarity("how is tcs doing today") is None


class TestWarmUp:
    def test_a_warm_up_in_flight_never_blocks_a_question(self):
        router = IntentRouter()
        router._lock.acquire()                  # stand in for a load on the daemon thread
        try:
            started = time.monotonic()
            assert router.similarity("how is tcs doing today") is None
            assert time.monotonic() - started < 0.5, "a question waited on the encoder"
            # ...and the question is still routed, by the rule that needs no model.
            assert router.classify("tata steel share price").is_price
        finally:
            router._lock.release()

    def test_warm_returns_immediately(self):
        router = IntentRouter(use_embeddings=False)   # no thread, no download
        started = time.monotonic()
        router.warm()
        assert time.monotonic() - started < 0.5

    def test_a_failed_load_falls_back_permanently_and_quietly(self):
        router = IntentRouter(model_name="not-a-real-model/does-not-exist")
        router._embedding_failed = True          # what _load records on failure
        assert router.similarity("how is tcs doing today") is None
        assert router.classify("tata steel share price").is_price
