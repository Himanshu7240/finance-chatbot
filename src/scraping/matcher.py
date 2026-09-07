"""Mapping article URLs to NIFTY 50 companies.

Moneycontrol article URLs carry the headline as a slug, so the company is usually
right there in the path:

    /news/business/earnings/tata-steel-q1-results-profit-jumps-...-13990630.html

Matching is done on **whole slug tokens**, not substrings, so "itc" doesn't match
"switch". Where several companies match, the longest alias wins — that is what keeps
"sbi-life-insurance" with SBI Life instead of State Bank of India. A few names need an
exclusion list for the same reason ("reliance-power" is a different group entirely).

Slug matching is only the first filter; :func:`mentions` re-checks the parsed article
text so an article that merely name-drops a company in its URL is dropped later.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

COMPANIES_PATH = Path("data/companies.json")

# Extra ways each company is written in headlines, beyond its canonical name.
# Single-token aliases are only used where the token is unambiguous on its own.
ALIASES: dict[str, list[str]] = {
    "ADANIPORTS": ["adani ports", "adani ports and sez"],
    "BHARTIARTL": ["bharti airtel", "airtel"],
    "DRREDDY": ["dr reddys laboratories", "dr reddys", "dr reddy s"],
    "EICHERMOT": ["eicher motors", "royal enfield"],
    "ETERNAL": ["eternal", "zomato", "blinkit"],
    "GRASIM": ["grasim"],
    "HCLTECH": ["hcl technologies", "hcl tech", "hcltech"],
    "HINDALCO": ["hindalco"],
    "HINDUNILVR": ["hindustan unilever", "hul"],
    "INDIGO": ["indigo", "interglobe aviation"],
    "JIOFIN": ["jio financial"],
    "KOTAKBANK": ["kotak mahindra bank", "kotak mahindra", "kotak bank"],
    "LT": ["larsen and toubro", "larsen toubro"],
    "M&M": ["mahindra and mahindra", "mahindra mahindra"],
    "MARUTI": ["maruti suzuki", "maruti"],
    "MAXHEALTH": ["max healthcare"],
    "NESTLEIND": ["nestle india", "nestle"],
    "ONGC": ["ongc", "oil and natural gas"],
    "POWERGRID": ["power grid"],
    "RELIANCE": ["reliance industries", "reliance jio", "ril", "reliance"],
    "SBIN": ["state bank of india", "sbi"],
    "SUNPHARMA": ["sun pharma", "sun pharmaceutical"],
    "TCS": ["tata consultancy services", "tcs"],
    "TATACONSUM": ["tata consumer"],
    "TMPV": ["tata motors"],
    "TITAN": ["titan company", "titan"],
    "ULTRACEMCO": ["ultratech cement", "ultratech"],
}

# Tokens that, if present in the slug, disqualify a match — sibling companies and
# same-brand entities that are *not* the NIFTY 50 constituent.
EXCLUSIONS: dict[str, list[str]] = {
    # "self" catches "self-reliance" in headline slugs; "defence" is the Anil Ambani group.
    "RELIANCE": ["power", "infrastructure", "capital", "communications", "anil",
                 "self", "defence", "defense"],
    "SBIN": ["life", "cards", "mutual", "general", "funds"],
    "TITAN": ["titanium"],
    "ITC": ["hotels"],   # ITC Hotels is a separately listed entity post-demerger
    "M&M": ["tech", "kotak"],
}

# Short forms people type into a chat box. Kept apart from ALIASES because they are
# only safe *at the keyboard*: a bare "kotak" or "mahindra" in an article slug is
# ambiguous, while a user asking one question about one company is not. Used by
# :func:`match_text` only - URL matching (Day 3) never sees these.
CHAT_ALIASES: dict[str, list[str]] = {
    "DRREDDY": ["dr reddy"],
    "EICHERMOT": ["eicher"],
    "HCLTECH": ["hcl"],
    "INDIGO": ["interglobe"],
    "KOTAKBANK": ["kotak"],
    "LT": ["l t"],          # "L&T" - the ampersand is not a token character
    "M&M": ["mahindra"],
    "SBILIFE": ["sbi life"],
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")
# Trailing article id, e.g. ".../tata-steel-q1-results-13990630.html"
_ARTICLE_ID_RE = re.compile(r"^\d{6,}$")


@dataclass(frozen=True)
class Company:
    company: str
    ticker: str
    slug: str
    aliases: tuple[tuple[str, ...], ...]   # each alias as a token tuple, longest first
    exclusions: tuple[str, ...]


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@lru_cache(maxsize=4)
def load_companies(path: str | Path = COMPANIES_PATH, chat: bool = False) -> tuple[Company, ...]:
    """NIFTY 50 companies with their match aliases resolved.

    ``chat=True`` adds the keyboard-only forms: :data:`CHAT_ALIASES` plus each ticker
    itself, since a user will type "INFY" or "M&M" where a URL never would.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    companies = []
    for entry in raw:
        ticker = entry["ticker"]
        forms = {entry["slug"], entry["company"], *ALIASES.get(ticker, [])}
        if chat:
            forms |= {ticker, *CHAT_ALIASES.get(ticker, [])}
        aliases = {tuple(_tokens(form)) for form in forms}
        aliases = {alias for alias in aliases if alias}
        companies.append(
            Company(
                company=entry["company"],
                ticker=ticker,
                slug=entry["slug"],
                aliases=tuple(sorted(aliases, key=len, reverse=True)),
                exclusions=tuple(EXCLUSIONS.get(ticker, [])),
            )
        )
    return tuple(companies)


def slug_tokens(url: str) -> list[str]:
    """Headline tokens from an article URL, with the trailing article id dropped."""
    path = urlparse(url).path
    tokens = _tokens(path)
    return [token for token in tokens if not _ARTICLE_ID_RE.match(token)]


def _contains(haystack: list[str], needle: tuple[str, ...]) -> bool:
    span = len(needle)
    return any(tuple(haystack[i:i + span]) == needle for i in range(len(haystack) - span + 1))


def _best_match(tokens: list[str], companies) -> Company | None:
    """Longest matching alias wins, so more specific names beat shorter ones."""
    best: tuple[int, Company] | None = None
    for company in companies:
        if any(exclusion in tokens for exclusion in company.exclusions):
            continue
        for alias in company.aliases:            # already longest-first
            if _contains(tokens, alias):
                if best is None or len(alias) > best[0]:
                    best = (len(alias), company)
                break
    return best[1] if best else None


def match_company(url: str, companies=None) -> Company | None:
    """The NIFTY 50 company an article URL is about, if any."""
    return _best_match(slug_tokens(url), companies or load_companies())


def match_text(text: str, companies=None) -> Company | None:
    """The NIFTY 50 company a free-text question is about, if any.

    Same alias table, longest-wins rule and exclusion list as :func:`match_company`,
    applied to a user's words instead of a URL slug. The app layer resolves company
    mentions through here so the domain knowledge - that "SBI Life" is not "SBI",
    that "Reliance Power" is a different group - lives in exactly one place.
    """
    return _best_match(_tokens(text), companies or load_companies(chat=True))


def strip_company(text: str, ticker: str | None) -> str:
    """``text`` with the named company's own aliases removed.

    Two callers, one reason: once a question has been *routed* by the company it names,
    the name itself is spent signal and actively harmful downstream. In corpus search the
    corpus-wide IDF makes "tata steel" a heavily weighted term that matches every
    paragraph inside TATASTEEL's own slice; in intent routing it pulls the sentence
    embedding towards "a company" and away from "a price question". Removing it lifted
    both - see docs/guides/06-retrieval-and-the-app-layer.md.

    Falls back to the original text if the question was nothing but the company name.
    """
    if not ticker:
        return text
    company = next((c for c in load_companies(chat=True) if c.ticker == ticker), None)
    if company is None:
        return text

    tokens = _tokens(text)
    for alias in company.aliases:            # longest first, so "tata steel" before "tata"
        span = len(alias)
        index = 0
        while index <= len(tokens) - span:
            if tuple(tokens[index:index + span]) == alias:
                del tokens[index:index + span]
            else:
                index += 1
    return " ".join(tokens) or text


def mentions(text: str, company: Company) -> bool:
    """Whether the article text actually talks about the company."""
    tokens = _tokens(text)
    return any(_contains(tokens, alias) for alias in company.aliases)
