"""QA/context dataset schema — see docs/guides/01-dataset-design-for-llm-finetuning.md."""

from dataclasses import dataclass, asdict


@dataclass
class QAExample:
    company: str      # canonical company name, e.g. "Tata Steel"
    ticker: str        # NSE ticker, e.g. "TATASTEEL"
    question: str
    answer: str
    context: str        # source paragraph the Q/A was derived from
    source_url: str      # article this example was generated from

    def to_dict(self) -> dict:
        return asdict(self)
