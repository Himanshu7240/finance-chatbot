"""Polite HTTP fetching for the article scraper.

Three things every scraper in this project goes through:

1. robots.txt is checked before each request (see :mod:`src.scraping.robots`).
2. Requests are rate-limited and retried with backoff — Moneycontrol's CDN returns
   403 when hit too fast, which is a soft "slow down", not a hard block.
3. Responses are cached on disk, so re-runs of the pipeline never re-hit the site.

See docs/guides/02-collecting-a-training-corpus.md for the reasoning.
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from pathlib import Path

import requests

from .robots import RobotsPolicy

log = logging.getLogger(__name__)

# Moneycontrol's CDN 403s any user-agent that isn't a recognised browser string --
# including a descriptive one, and including a browser string with a project token
# appended. So the UA has to be a plain browser UA, and the client identifies itself
# in headers the WAF doesn't police instead. See docs/dataset-design.md.
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
PROJECT_ID = "finance-chatbot-research/0.1 (non-commercial student project)"

DEFAULT_DELAY = 3.0          # seconds between requests, plus jitter
DEFAULT_TIMEOUT = 45         # seconds
DEFAULT_RETRIES = 4
CACHE_DIR = Path("data/raw/http")


class Blocked(Exception):
    """Raised when a URL is disallowed by robots.txt."""


class Fetcher:
    """A rate-limited, disk-cached, robots-respecting HTTP client."""

    def __init__(
        self,
        cache_dir: Path | str = CACHE_DIR,
        delay: float = DEFAULT_DELAY,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self._last_request = 0.0

        self.session = requests.Session()
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "X-Purpose": PROJECT_ID,
        }
        # Optional, from .env: an address the site operator could reach you at.
        contact = os.environ.get("SCRAPER_CONTACT", "").strip()
        if contact:
            headers["From"] = contact
        self.session.headers.update(headers)
        self.robots = RobotsPolicy(self._request, USER_AGENT)

        self.stats = {"cache_hits": 0, "fetched": 0, "failed": 0, "blocked": 0}

    # -- cache ---------------------------------------------------------------

    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()
        return self.cache_dir / digest[:2] / f"{digest}.html"

    def cached(self, url: str) -> str | None:
        path = self._cache_path(url)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        return None

    def _store(self, url: str, body: str) -> None:
        path = self._cache_path(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8", errors="replace")

    # -- fetching ------------------------------------------------------------

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request
        pause = self.delay + random.uniform(0, self.delay / 2) - elapsed
        if pause > 0:
            time.sleep(pause)
        self._last_request = time.monotonic()

    def get(self, url: str, use_cache: bool = True) -> str | None:
        """Return the body of ``url``, or ``None`` if it could not be fetched.

        Raises :class:`Blocked` if robots.txt disallows the URL.
        """
        if not self.robots.allowed(url):
            self.stats["blocked"] += 1
            raise Blocked(url)

        if use_cache:
            body = self.cached(url)
            if body is not None:
                self.stats["cache_hits"] += 1
                return body

        body = self._request(url)
        if body is None:
            self.stats["failed"] += 1
            return None
        self._store(url, body)
        self.stats["fetched"] += 1
        return body

    def _request(self, url: str) -> str | None:
        """Rate-limited GET with backoff, no robots check and no caching.

        robots.txt itself is fetched through here — it can't be gated on a policy
        it hasn't supplied yet — and the CDN rate-limits it like any other URL.
        """
        backoff = self.delay
        for attempt in range(1, self.retries + 1):
            self._wait()
            try:
                response = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                log.warning("request failed (%s/%s) %s: %s", attempt, self.retries, url, exc)
            else:
                if response.ok:
                    return response.text
                if response.status_code in (403, 429) or response.status_code >= 500:
                    # Rate limiting / transient — back off and try again.
                    log.warning(
                        "HTTP %s (%s/%s) %s - backing off %.0fs",
                        response.status_code, attempt, self.retries, url, backoff,
                    )
                else:
                    log.warning("HTTP %s %s - giving up", response.status_code, url)
                    break
            time.sleep(backoff)
            backoff *= 2

        return None
