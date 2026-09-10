"""The alias table encodes domain knowledge that breaks silently.

Nothing here needs the network or the dataset - just data/companies.json.
"""

from __future__ import annotations

import pytest

from src.scraping.matcher import (
    load_companies,
    match_company,
    match_text,
    strip_company,
)


def test_every_canonical_name_resolves_to_its_own_ticker():
    for company in load_companies():
        matched = match_text(f"what is the share price of {company.company} today")
        assert matched is not None, company.company
        assert matched.ticker == company.ticker


@pytest.mark.parametrize("text, ticker", [
    ("what is infy trading at", "INFY"),
    ("how did m&m do today", "M&M"),
    ("sbi life", "SBILIFE"),          # not SBI - the whole point of the exclusion list
    ("sbi", "SBIN"),
    ("kotak", "KOTAKBANK"),
    ("l&t", "LT"),
    ("dr reddy", "DRREDDY"),
    ("interglobe", "INDIGO"),
    ("zomato", "ETERNAL"),            # renamed; people still type the old name
    ("tech mahindra", "TECHM"),       # longest alias wins over "mahindra"
])
def test_chat_forms_resolve(text, ticker):
    matched = match_text(text)
    assert matched is not None, text
    assert matched.ticker == ticker


@pytest.mark.parametrize("text", [
    "reliance power",                 # a different group entirely
    "itc hotels",                     # separately listed after the demerger
    "what is the capital of France",
    "tell me a joke",
])
def test_non_constituents_resolve_to_nothing(text):
    assert match_text(text) is None


@pytest.mark.parametrize("url, ticker", [
    ("https://www.moneycontrol.com/news/business/tata-steel-q1-results-13990630.html", "TATASTEEL"),
    ("https://www.moneycontrol.com/news/business/sbi-life-insurance-shares-rise-1234567.html", "SBILIFE"),
    ("https://www.moneycontrol.com/news/business/kotak-mahindra-bank-q1-1234567.html", "KOTAKBANK"),
])
def test_url_matching_still_works(url, ticker):
    """Day 3 depends on this; the chat aliases must not leak into slug matching."""
    matched = match_company(url)
    assert matched is not None and matched.ticker == ticker


def test_url_matching_ignores_chat_only_aliases():
    # "kotak" alone is a chat form. In a slug it is ambiguous, so it must not match.
    assert match_company("https://www.moneycontrol.com/news/kotak-general-insurance-1234.html") is None


class TestStripCompany:
    def test_removes_the_name_and_keeps_the_question(self):
        assert strip_company("Why might Tata Steel owe money to Odisha?", "TATASTEEL") == (
            "why might owe money to odisha"
        )

    def test_handles_possessives(self):
        assert "tata" not in strip_company("What was Tata Steel's Q1 profit?", "TATASTEEL")

    def test_falls_back_when_nothing_would_be_left(self):
        assert strip_company("Tata Steel", "TATASTEEL") == "Tata Steel"

    def test_no_ticker_is_a_no_op(self):
        assert strip_company("What did Infosys say?", None) == "What did Infosys say?"

    def test_unknown_ticker_is_a_no_op(self):
        assert strip_company("What did Infosys say?", "NOPE") == "What did Infosys say?"
