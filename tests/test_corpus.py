"""Corpus retrieval, against the real index when the dataset is present.

These are skipped on a fresh clone - `data/processed/` is gitignored and regenerable - but
kept, because on a machine that has the data they check the two things that actually break:
the ticker filter and the score floor.
"""

from __future__ import annotations

import pytest

from src.app.corpus import PARAGRAPHS_PATH, ArticleRetriever, Passage, build_context

pytestmark = pytest.mark.skipif(
    not PARAGRAPHS_PATH.exists(),
    reason=f"{PARAGRAPHS_PATH} not built - run the Day 3/4 pipeline",
)


@pytest.fixture(scope="module")
def retriever():
    found = ArticleRetriever()
    found.size                     # build the index once for the whole module
    return found


def test_the_index_covers_the_whole_corpus(retriever):
    assert retriever.size > 10_000


def test_results_come_back_best_first(retriever):
    passages = retriever.search("west asia crisis coking coal", ticker="TATASTEEL", top_k=3)
    assert passages
    assert [p.score for p in passages] == sorted((p.score for p in passages), reverse=True)


def test_the_ticker_filter_confines_results_to_that_company(retriever):
    for passage in retriever.search("quarterly results", ticker="INFY", top_k=5):
        assert passage.ticker == "INFY"


def test_nothing_below_the_floor_is_returned(retriever):
    for passage in retriever.search("west asia crisis", ticker="TATASTEEL"):
        assert passage.score >= retriever.min_score


def test_an_unanswerable_question_returns_nothing(retriever):
    # High floor, so this is a statement about the floor doing its job, not about luck.
    strict = ArticleRetriever(min_score=0.9)
    strict._rows, strict._matrix = retriever._rows, retriever._matrix
    strict._vectorizer, strict._by_ticker = retriever._vectorizer, retriever._by_ticker
    assert strict.search("blockchain yoga retreat", ticker="TATASTEEL") == []


def test_an_unknown_ticker_falls_back_to_the_whole_corpus(retriever):
    assert retriever.search("quarterly results", ticker="NOT-A-TICKER")


def test_every_passage_carries_provenance(retriever):
    for passage in retriever.search("quarterly results", ticker="WIPRO"):
        assert passage.url.startswith("http")
        assert passage.published
        assert passage.citation().endswith(passage.url)


class TestBuildContext:
    def make(self, *texts) -> list[Passage]:
        return [
            Passage(para_id=f"T-{i}", company="Tata Steel", ticker="TATASTEEL",
                    title="t", url="u", published="2026-07-31", text=text, score=0.5)
            for i, text in enumerate(texts)
        ]

    def test_joins_passages_best_first(self):
        assert build_context(self.make("first.", "second.")) == "first. second."

    def test_stops_at_the_budget(self):
        long_text = "x" * 800
        context = build_context(self.make(long_text, long_text), budget=1000)
        assert context == long_text

    def test_always_keeps_the_best_passage_even_when_it_exceeds_the_budget(self):
        long_text = "x" * 2000
        assert build_context(self.make(long_text), budget=100) == long_text

    def test_no_passages_is_an_empty_context(self):
        assert build_context([]) == ""
