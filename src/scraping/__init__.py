"""Article collection pipeline for the NIFTY 50 news corpus.

See docs/guides/02-collecting-a-training-corpus.md for the design, and run with:

    python -m src.scraping.collect --months 6 --max-articles 25
"""

from .article import Article, parse_article
from .fetcher import Blocked, Fetcher
from .matcher import Company, load_companies, match_company, mentions
from .robots import RobotsPolicy

__all__ = [
    "Article",
    "Blocked",
    "Company",
    "Fetcher",
    "RobotsPolicy",
    "load_companies",
    "match_company",
    "mentions",
    "parse_article",
]
