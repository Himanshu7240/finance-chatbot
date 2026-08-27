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


@lru_cache(maxsize=1)
def load_companies(path: str | Path = COMPANIES_PATH) -> tuple[Company, ...]:
    """NIFTY 50 companies with their match aliases resolved."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    companies = []
    for entry in raw:
        ticker = entry["ticker"]
        forms = {entry["slug"], entry["company"], *ALIASES.get(ticker, [])}
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


def match_company(url: str, companies=None) -> Company | None:
    """The NIFTY 50 company an article URL is about, if any.

    Longest matching alias wins, so more specific names beat shorter ones.
    """
    companies = companies or load_companies()
    tokens = slug_tokens(url)
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


def mentions(text: str, company: Company) -> bool:
    """Whether the article text actually talks about the company."""
    tokens = _tokens(text)
    return any(_contains(tokens, alias) for alias in company.aliases)
