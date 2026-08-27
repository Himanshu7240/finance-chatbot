"""Article discovery through Moneycontrol's published sitemaps.

Two entry points into the same corpus:

* the **news sitemap** (`/news/news-sitemap.xml`) — the last ~900 articles across all
  sections, useful for incremental top-ups;
* the **archive** — `/news/index-sitemap-<year>.xml` lists one sitemap per month
  (`/news/sitemap/sitemap-post-<year>-<month>.xml`), each holding ~15k article URLs.

The archive is what gives us enough per-company volume to hit the dataset target.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET

from .fetcher import Fetcher

log = logging.getLogger(__name__)

BASE = "https://www.moneycontrol.com"
NEWS_SITEMAP = f"{BASE}/news/news-sitemap.xml"
YEAR_INDEX = f"{BASE}/news/index-sitemap-{{year}}.xml"

_NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
_MONTH_RE = re.compile(r"sitemap-post-(\d{4})-(\d{2})\.xml")


def _locs(xml: str) -> list[str]:
    """Every <loc> in a urlset or sitemapindex document."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        log.error("could not parse sitemap XML: %s", exc)
        return []
    return [el.text.strip() for el in root.findall(".//s:loc", _NS) if el.text]


def news_sitemap(fetcher: Fetcher) -> list[str]:
    """Recently published article URLs (all sections)."""
    # Always refetch: this file changes by the hour, so a cached copy is worthless.
    xml = fetcher.get(NEWS_SITEMAP, use_cache=False)
    return _locs(xml) if xml else []


def month_sitemaps(fetcher: Fetcher, year: int) -> dict[str, str]:
    """Map of ``"YYYY-MM"`` -> monthly sitemap URL for one year."""
    xml = fetcher.get(YEAR_INDEX.format(year=year))
    if not xml:
        return {}
    found = {}
    for url in _locs(xml):
        match = _MONTH_RE.search(url)
        if match:
            found[f"{match.group(1)}-{match.group(2)}"] = url
    return found


def month_articles(fetcher: Fetcher, month: str) -> list[str]:
    """Every article URL published in ``month`` (format ``"YYYY-MM"``)."""
    year = int(month.split("-")[0])
    sitemaps = month_sitemaps(fetcher, year)
    url = sitemaps.get(month)
    if not url:
        log.warning("no sitemap published for %s", month)
        return []
    xml = fetcher.get(url)
    return _locs(xml) if xml else []


def recent_months(count: int, until: str) -> list[str]:
    """The ``count`` months ending at ``until`` (``"YYYY-MM"``), newest first."""
    year, month = (int(part) for part in until.split("-"))
    months = []
    for _ in range(count):
        months.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return months
