"""Live NSE quotes, formatted as context the fine-tuned model can read.

    python -m src.app.stock "Tata Steel" Reliance

The model was trained to extract an answer span from a passage of financial prose
(Guide 01), so this retriever's job is not only to fetch a price but to write it into
a sentence that looks like the training contexts. See
docs/guides/06-retrieval-and-the-app-layer.md.

Two things this module refuses to paper over:

* Yahoo's NSE data is **delayed**, and ``fast_info`` carries no trade timestamp. Every
  quote therefore records the time we *asked*, and says so - a price with an implied
  "right now" is a claim we cannot support.
* Missing fields stay missing. ``previousClose`` is absent for some symbols; the change
  sentence is then omitted rather than computed against a ``None`` read as zero.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

from ..scraping.matcher import Company, match_text

log = logging.getLogger("stock")

IST = ZoneInfo("Asia/Kolkata")
# Yahoo needs an exchange suffix for Indian equities; data/companies.json stores bare
# NSE tickers. Appending is safer than reconstructing - "M&M" and "BAJAJ-AUTO" survive.
YAHOO_SUFFIX = ".NS"
# Not an optimization. Without it every user turn hits an undocumented endpoint with
# unpublished rate limits; a minute of staleness is well inside the feed's own delay.
CACHE_TTL_SECONDS = 60
QUOTE_DELAY_NOTE = "may be delayed by up to 15 minutes"


class QuoteUnavailable(RuntimeError):
    """No usable quote - unknown symbol, network failure, or a feed with no price."""


@dataclass(frozen=True)
class Quote:
    """One point-in-time look at a stock, with the fetch time attached."""

    company: str
    ticker: str
    symbol: str
    price: float
    fetched_at: datetime
    currency: str = "INR"
    previous_close: float | None = None
    day_low: float | None = None
    day_high: float | None = None
    volume: float | None = None

    @property
    def change(self) -> float | None:
        if self.previous_close is None:
            return None
        return self.price - self.previous_close

    @property
    def change_pct(self) -> float | None:
        if not self.previous_close:          # None or 0.0 - neither can be divided by
            return None
        return 100.0 * (self.price - self.previous_close) / self.previous_close

    @property
    def fetched_at_text(self) -> str:
        return self.fetched_at.strftime("%H:%M IST on %d %B %Y")

    def as_context(self) -> str:
        """The quote as declarative prose, in the register of a news paragraph.

        This string goes into the ``Context:`` slot of the training prompt, so it has to
        read like the article paragraphs the model was fine-tuned on - facts in
        sentences, numbers in full - rather than like a serialized record.
        """
        money = f"{self.currency_symbol} {self.price:,.2f}"
        parts = [f"{self.company} ({self.ticker}) is trading at {money} on the NSE"]

        change, pct = self.change, self.change_pct
        if change is not None and pct is not None:
            close = f"the previous close of {self.currency_symbol} {self.previous_close:,.2f}"
            if change == 0:
                parts.append(f", unchanged from {close}")
            else:
                direction = "up" if change > 0 else "down"
                parts.append(
                    f", {direction} {self.currency_symbol} {abs(change):,.2f} "
                    f"({pct:+.2f}%) from {close}"
                )
        sentences = ["".join(parts) + "."]

        if self.day_low is not None and self.day_high is not None:
            sentences.append(
                f"The day's range is {self.currency_symbol} {self.day_low:,.2f} to "
                f"{self.currency_symbol} {self.day_high:,.2f}."
            )
        if self.volume:
            sentences.append(f"Volume so far is {self.volume:,.0f} shares.")
        sentences.append(
            f"The quote was fetched at {self.fetched_at_text} and {QUOTE_DELAY_NOTE}."
        )
        return " ".join(sentences)

    @property
    def currency_symbol(self) -> str:
        return "Rs" if self.currency == "INR" else self.currency


def _fetch_fast_info(symbol: str) -> dict:
    """Yahoo's lightweight quote endpoint, via yfinance.

    Imported inside the function so the module - and everything that imports it for the
    dataclass alone - stays usable without yfinance installed.
    """
    try:
        import yfinance
    except ImportError as exc:                                  # pragma: no cover
        raise QuoteUnavailable("yfinance is not installed") from exc

    try:
        return dict(yfinance.Ticker(symbol).fast_info)
    except Exception as exc:                                    # network, JSON, 404...
        raise QuoteUnavailable(f"could not reach the price feed for {symbol}") from exc


class StockDataRetriever:
    """Resolves a company mention to a NIFTY 50 ticker and fetches its live quote."""

    def __init__(
        self,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        suffix: str = YAHOO_SUFFIX,
        fetch=_fetch_fast_info,
        clock=time.monotonic,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.suffix = suffix
        self._fetch = fetch          # injectable, so the pipeline is testable offline
        self._clock = clock
        self._cache: dict[str, tuple[float, Quote]] = {}

    def resolve(self, text: str) -> Company | None:
        """The NIFTY 50 company a question mentions, via the Day 3 alias table.

        ``match_text`` loads the table in chat mode, which adds the forms people type
        but URLs never contain - "INFY", "m&m", "sbi life".
        """
        return match_text(text)

    def quote(self, company: Company) -> Quote:
        """Live quote for a company. Raises :class:`QuoteUnavailable` rather than guessing."""
        symbol = f"{company.ticker}{self.suffix}"
        cached = self._cache.get(symbol)
        now = self._clock()
        if cached and now - cached[0] < self.ttl_seconds:
            log.debug("cache hit %s", symbol)
            return cached[1]

        info = self._fetch(symbol)
        price = info.get("lastPrice")
        if price is None:
            raise QuoteUnavailable(f"the feed returned no price for {symbol}")

        quote = Quote(
            company=company.company,
            ticker=company.ticker,
            symbol=symbol,
            price=float(price),
            fetched_at=datetime.now(IST),
            currency=info.get("currency") or "INR",
            previous_close=_as_float(info.get("previousClose")),
            day_low=_as_float(info.get("dayLow")),
            day_high=_as_float(info.get("dayHigh")),
            volume=_as_float(info.get("lastVolume")),
        )
        self._cache[symbol] = (now, quote)
        return quote

    def quote_for_text(self, text: str) -> Quote | None:
        """Convenience path: mention -> company -> quote. ``None`` if no company matched."""
        company = self.resolve(text)
        return self.quote(company) if company else None


def _as_float(value) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mentions", nargs="+", help="company names as a user would type them")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    retriever = StockDataRetriever()
    for mention in args.mentions:
        company = retriever.resolve(mention)
        if company is None:
            log.info("%-20s no NIFTY 50 company matched", mention)
            continue
        try:
            log.info("%s", retriever.quote(company).as_context())
        except QuoteUnavailable as exc:
            log.info("%-20s %s", mention, exc)


if __name__ == "__main__":
    main()
