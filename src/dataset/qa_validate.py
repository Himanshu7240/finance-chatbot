"""Validating what the generator model produced.

A 7B model asked for strict JSON will still, sometimes, wrap it in a code fence,
invent a figure that is not in the paragraph, or ask a question the paragraph cannot
answer. None of that is a reason to avoid a local model - it is a reason to check its
output before training on it.

Every generated pair runs this gauntlet, and each rejection is counted by reason so
the report shows exactly how the generator behaved (that breakdown is worth more at
review time than a single "kept N" number).

The strictest check is groundedness: **every number in the answer must appear in the
paragraph**. In a finance dataset a hallucinated figure is the worst possible training
example - it teaches the student to state numbers with confidence and no support.
"""

from __future__ import annotations

import json
import re

# Answers longer than this stop being answers and start being paragraphs.
MAX_ANSWER_WORDS = 25
# "5" or "-" is not an answer. Short numeric answers like "6.89%" are.
MIN_ANSWER_CHARS = 2
MIN_QUESTION_CHARS = 25
MAX_QUESTION_CHARS = 250
GROUNDEDNESS_THRESHOLD = 0.7

_YES_NO_OPENERS = {
    "is", "are", "was", "were", "does", "do", "did", "has", "have", "had",
    "will", "can", "could", "should", "would", "may", "might",
}
# Phrases that only make sense with the article in front of you.
_DEICTIC = re.compile(
    r"\b(the company|the firm|this article|the article|the paragraph|the passage|"
    r"as mentioned|as stated|mentioned above|the above|the report says|according to "
    r"the (paragraph|passage|text))\b",
    re.IGNORECASE,
)
_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "been", "by", "for", "from", "had", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "that", "the", "to", "was",
    "were", "will", "with", "which", "this", "these", "those", "their", "than",
}
_WORD = re.compile(r"[a-z0-9][a-z0-9.,%-]*")
_NUMBER = re.compile(r"\d[\d,.]*")
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_response(text: str) -> list[dict] | None:
    """Pull the JSON list out of a model response, or ``None`` if there isn't one.

    Tolerant of code fences and of trailing commentary, because those are the two
    things instruct models do most often despite being told not to. Anything more
    malformed than that is rejected rather than repaired - guessing at what the
    model meant is how bad examples sneak into a dataset.
    """
    if not text:
        return None
    cleaned = _FENCE.sub("", text.strip()).strip()

    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, list):
        return None
    return [item for item in parsed if isinstance(item, dict)]


def clean_answer(answer: str) -> str:
    """Trim an extracted span to the answer itself.

    Models often end a span on the comma or colon it was cut at; keep this in one
    place so validation and dataset assembly can never disagree about the text.
    """
    return (answer or "").strip().strip(",;: ")


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _content_tokens(text: str) -> list[str]:
    return [token.strip(".,") for token in _tokens(text)
            if token not in _STOPWORDS and len(token.strip(".,")) > 1]


def _numbers(text: str) -> set[str]:
    return {match.replace(",", "").rstrip(".") for match in _NUMBER.findall(text)}


def groundedness(answer: str, context: str) -> float:
    """Share of the answer's content words that actually appear in the paragraph."""
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return 0.0
    context_tokens = set(_content_tokens(context))
    return sum(token in context_tokens for token in answer_tokens) / len(answer_tokens)


def validate(pair: dict, context: str, mentions_company) -> str | None:
    """Return a rejection reason, or ``None`` if the pair is good.

    ``mentions_company`` is a callable so this module stays independent of the
    scraper's company table - the caller binds it to the right company.
    """
    question = (pair.get("question") or "").strip()
    answer = clean_answer(pair.get("answer"))

    if not question or not answer:
        return "empty"
    if len(answer) < MIN_ANSWER_CHARS:
        return "answer_too_short"
    if not question.endswith("?"):
        return "not_a_question"
    if not (MIN_QUESTION_CHARS <= len(question) <= MAX_QUESTION_CHARS):
        return "question_length"
    if len(answer.split()) > MAX_ANSWER_WORDS:
        return "answer_too_long"
    if _DEICTIC.search(question):
        return "not_self_contained"

    opener = _tokens(question)[0] if _tokens(question) else ""
    if opener in _YES_NO_OPENERS:
        return "yes_no_question"
    if not mentions_company(question):
        return "company_not_named"

    # A number in the answer that is not in the paragraph is invented. No exceptions:
    # this is the check that keeps fabricated figures out of the training data.
    if not _numbers(answer) <= _numbers(context):
        return "invented_number"
    if groundedness(answer, context) < GROUNDEDNESS_THRESHOLD:
        return "ungrounded"

    return None
