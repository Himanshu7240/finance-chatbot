"""robots.txt enforcement.

Every URL the scraper touches is checked against the publisher's robots.txt before
the request goes out — not as a formality, but because the Day 2 sourcing decision
(docs/dataset-design.md) rests on it: we discover articles through the sitemaps
Moneycontrol publishes, and never through the `/page-*/` listing URLs it disallows.
"""

from __future__ import annotations

import logging
from typing import Callable
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

log = logging.getLogger(__name__)


class RobotsPolicy:
    """Caches and applies robots.txt rules, one parser per host.

    ``fetch`` is the raw, rate-limited GET from :class:`~src.scraping.fetcher.Fetcher`
    — robots.txt is served by the same CDN as everything else and gets throttled the
    same way, so it needs the same backoff behaviour.
    """

    def __init__(self, fetch: Callable[[str], str | None], user_agent: str) -> None:
        self.fetch = fetch
        self.user_agent = user_agent
        self._parsers: dict[str, RobotFileParser | None] = {}

    def _parser(self, url: str) -> RobotFileParser | None:
        parts = urlparse(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._parsers:
            self._parsers[host] = self._load(host)
        return self._parsers[host]

    def _load(self, host: str) -> RobotFileParser | None:
        body = self.fetch(f"{host}/robots.txt")
        if body is None:
            # No robots.txt we can read: fail closed rather than assume permission.
            log.error("could not read robots.txt for %s - refusing to crawl it", host)
            return None
        parser = RobotFileParser()
        parser.parse(body.splitlines())
        return parser

    def allowed(self, url: str) -> bool:
        parser = self._parser(url)
        if parser is None:
            return False
        return parser.can_fetch(self.user_agent, url)

    def sitemaps(self, host: str) -> list[str]:
        """The sitemaps the host advertises in its own robots.txt."""
        parser = self._parser(host)
        if parser is None:
            return []
        return list(parser.site_maps() or [])
