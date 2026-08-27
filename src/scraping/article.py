"""Turning a Moneycontrol article page into clean paragraphs.

The body lives in ``div#contentdata``, but not every ``<p>`` in there is article
text — the page interleaves promo widgets, "Also Read" links, disclaimers and
newsletter pitches. Those become garbage training context if they survive, so this
module is deliberately aggressive about dropping them: a paragraph has to be long
enough to carry a fact, and must not look like boilerplate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

from bs4 import BeautifulSoup

# A paragraph shorter than this can't hold enough of a fact to generate a Q/A from.
MIN_PARAGRAPH_CHARS = 90
MAX_PARAGRAPH_CHARS = 1200

_BOILERPLATE = re.compile(
    r"^(also read|also watch|read more|watch\b|disclaimer|catch all the|stay tuned|"
    r"follow us|download the|discover the latest|views and investment tips|"
    r"moneycontrol news|first published|click here|subscribe|market mastery|"
    r"find the weak links|check out|advertisement)",
    re.IGNORECASE,
)
_PROMO_PHRASES = re.compile(
    r"(expressed by investment experts|do not represent the views of moneycontrol|"
    r"users to check with certified experts|copyright ©|all rights reserved)",
    re.IGNORECASE,
)
_WHITESPACE = re.compile(r"\s+")
_LDJSON_DATE_RE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"')

_BODY_SELECTORS = ("div#contentdata", "div.content_wrapper", "article")
_DATE_SELECTORS = (
    ("meta[property='article:published_time']", "content"),
    ("meta[property='og:article:published_time']", "content"),
    ("meta[name='publish-date']", "content"),
    ("meta[itemprop='datePublished']", "content"),
    ("time[datetime]", "datetime"),
)


@dataclass
class Article:
    url: str
    title: str
    published: str | None
    paragraphs: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _is_body_text(paragraph: str) -> bool:
    if not (MIN_PARAGRAPH_CHARS <= len(paragraph) <= MAX_PARAGRAPH_CHARS):
        return False
    if _BOILERPLATE.match(paragraph) or _PROMO_PHRASES.search(paragraph):
        return False
    # Link-dump paragraphs and tickers-only lines read as text but aren't prose.
    if paragraph.count("|") > 2 or paragraph.endswith(":"):
        return False
    return True


def _published(soup: BeautifulSoup) -> str | None:
    for selector, attribute in _DATE_SELECTORS:
        element = soup.select_one(selector)
        if element and element.get(attribute):
            return element[attribute].strip()
    for script in soup.select("script[type='application/ld+json']"):
        raw = script.string or ""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            # Some of Moneycontrol's JSON-LD blocks are malformed; fall back to a
            # straight read of the field rather than losing the date entirely.
            match = _LDJSON_DATE_RE.search(raw)
            if match:
                return match.group(1)
            continue
        for node in data if isinstance(data, list) else [data]:
            if isinstance(node, dict) and node.get("datePublished"):
                return str(node["datePublished"])
    return None


def _title(soup: BeautifulSoup) -> str:
    heading = soup.select_one("h1")
    if heading:
        # Headlines can carry a section badge ("MC INTERVIEW", "EXCLUSIVE") in a span.
        for badge in heading.select("span.slug_elmnt"):
            badge.decompose()
        return _clean(heading.get_text(" "))
    meta = soup.select_one("meta[property='og:title']")
    if meta and meta.get("content"):
        return _clean(meta["content"])
    return _clean(soup.title.get_text()) if soup.title else ""


def parse_article(html: str, url: str) -> Article | None:
    """Parse article HTML into title, publish date and body paragraphs.

    Returns ``None`` if the page has no usable body — a paywalled page, a video
    or photo gallery, or a layout this parser doesn't understand.
    """
    soup = BeautifulSoup(html, "html.parser")

    body = None
    for selector in _BODY_SELECTORS:
        body = soup.select_one(selector)
        if body:
            break
    if body is None:
        return None

    for tag in body.select("script, style, figure, figcaption, aside, .related_stories"):
        tag.decompose()

    paragraphs = []
    seen = set()
    for element in body.find_all("p"):
        text = _clean(element.get_text(" "))
        if _is_body_text(text) and text not in seen:
            seen.add(text)
            paragraphs.append(text)

    if not paragraphs:
        return None

    return Article(url=url, title=_title(soup), published=_published(soup), paragraphs=paragraphs)
