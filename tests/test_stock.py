"""The quote retriever, with an injected feed and an injected clock.

No network: ``StockDataRetriever`` takes its ``fetch`` and ``clock`` as constructor
arguments precisely so the cache and the failure paths can be tested in microseconds.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from src.app.stock import IST, Quote, QuoteUnavailable, StockDataRetriever
from src.scraping.matcher import match_text

FIXED_TIME = datetime(2026, 9, 10, 15, 30, tzinfo=IST)

FULL_FEED = {
    "lastPrice": 1309.5,
    "previousClose": 1322.0,
    "dayLow": 1301.9,
    "dayHigh": 1324.2,
    "lastVolume": 10209500,
    "currency": "INR",
}


def quote(**overrides) -> Quote:
    base = dict(
        company="Reliance Industries", ticker="RELIANCE", symbol="RELIANCE.NS",
        price=1309.5, fetched_at=FIXED_TIME, previous_close=1322.0,
        day_low=1301.9, day_high=1324.2, volume=10209500,
    )
    return Quote(**{**base, **overrides})


class TestQuoteProse:
    """The context string is both what the model reads and what the user sees."""

    def test_full_quote_reads_as_a_sentence(self):
        text = quote().as_context()
        assert "Reliance Industries (RELIANCE) is trading at Rs 1,309.50 on the NSE" in text
        assert "down Rs 12.50 (-0.95%) from the previous close of Rs 1,322.00" in text
        assert "The day's range is Rs 1,301.90 to Rs 1,324.20." in text
        assert "Volume so far is 10,209,500 shares." in text

    def test_every_quote_carries_its_fetch_time_and_the_delay(self):
        text = quote().as_context()
        assert "15:30 IST on 10 September 2026" in text
        assert "may be delayed by up to 15 minutes" in text

    def test_missing_previous_close_omits_the_change_rather_than_inventing_zero(self):
        text = quote(previous_close=None).as_context()
        assert "previous close" not in text
        assert "up Rs" not in text and "down Rs" not in text and "%" not in text
        assert quote(previous_close=None).change is None
        assert quote(previous_close=None).change_pct is None

    def test_unchanged_price_is_phrased_as_unchanged(self):
        text = quote(price=1322.0).as_context()
        assert "unchanged from the previous close of Rs 1,322.00" in text
        assert "+0.00%" not in text

    def test_rising_price_says_up(self):
        assert "up Rs 8.00 (+0.61%)" in quote(price=1330.0).as_context()

    def test_missing_day_range_is_simply_absent(self):
        assert "day's range" not in quote(day_low=None, day_high=None).as_context()

    def test_non_inr_currency_is_not_labelled_rs(self):
        assert "USD 1,309.50" in quote(currency="USD").as_context()

    def test_zero_previous_close_does_not_divide_by_zero(self):
        assert quote(previous_close=0.0).change_pct is None


class TestRetriever:
    def test_appends_the_exchange_suffix(self):
        seen = []

        def fetch(symbol):
            seen.append(symbol)
            return FULL_FEED

        retriever = StockDataRetriever(fetch=fetch)
        retriever.quote(match_text("m&m"))
        assert seen == ["M&M.NS"]           # awkward tickers survive the round trip

    def test_cache_hits_inside_the_ttl_and_refetches_after_it(self):
        calls = []
        now = [1000.0]

        def fetch(symbol):
            calls.append(symbol)
            return FULL_FEED

        retriever = StockDataRetriever(ttl_seconds=60, fetch=fetch, clock=lambda: now[0])
        company = match_text("reliance")

        retriever.quote(company)
        now[0] += 30
        retriever.quote(company)
        assert len(calls) == 1, "a second question inside the TTL must not hit the feed"

        now[0] += 31
        retriever.quote(company)
        assert len(calls) == 2, "past the TTL the quote must be refetched"

    def test_a_feed_with_no_price_raises_rather_than_returning_nothing(self):
        retriever = StockDataRetriever(fetch=lambda symbol: {"currency": "INR"})
        with pytest.raises(QuoteUnavailable):
            retriever.quote(match_text("wipro"))

    def test_an_outage_propagates_as_quote_unavailable(self):
        def dead(symbol):
            raise QuoteUnavailable("simulated outage")

        with pytest.raises(QuoteUnavailable):
            StockDataRetriever(fetch=dead).quote(match_text("wipro"))

    def test_unparseable_fields_become_none_instead_of_crashing(self):
        feed = {**FULL_FEED, "previousClose": "n/a", "dayLow": None}
        result = StockDataRetriever(fetch=lambda symbol: feed).quote(match_text("reliance"))
        assert result.previous_close is None
        assert result.day_low is None
        assert result.price == 1309.5

    def test_resolve_uses_the_chat_alias_table(self):
        assert StockDataRetriever().resolve("what is infy trading at").ticker == "INFY"

    def test_quote_for_text_returns_none_when_no_company_is_named(self):
        retriever = StockDataRetriever(fetch=lambda symbol: FULL_FEED)
        assert retriever.quote_for_text("what is the share price") is None
