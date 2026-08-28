"""The prompt that turns an article paragraph into question/answer pairs.

Kept in the repo rather than inside the Kaggle notebook on purpose: the prompt *is*
the dataset design. Change it and every downstream number changes, so it belongs in
version control where the diff is visible.

Generation runs on Qwen2.5-7B-Instruct (Apache 2.0) on a Kaggle GPU - a 7B teacher
producing training data for the 3B student we fine-tune later, which is ordinary
knowledge distillation. See docs/guides/03-generating-a-qa-dataset.md.

The rules below are not stylistic preferences; each one exists to stop a specific
failure mode that would otherwise poison training:

1. **Answer only from the paragraph** - the model must not use outside knowledge,
   or we train the student to state facts its context does not support.
2. **Company name in the question** - keeps every example attributable, and stops
   "What was the revenue?" questions that are meaningless out of context.
3. **Short answers** - the schema wants a concise answer, not a restated paragraph.
4. **No yes/no questions** - a one-word answer carries almost no training signal.
5. **Self-contained questions** - no "the company" or "this article"; at inference
   time there is no article to refer back to.
6. **Empty list allowed** - a paragraph with no clean fact should produce nothing.
   Forcing output from thin text is where hallucinated training data comes from.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You build question-answer pairs for a financial question-answering dataset about \
Indian stock market companies. You are given one paragraph from a news article about \
a company, and you write questions that the paragraph itself answers.

Rules:
1. Use ONLY information stated in the paragraph. Never add outside knowledge.
2. The company name must appear in every question.
3. Each answer must be short - at most 25 words - and taken directly from the \
paragraph's wording.
4. Never write yes/no questions. Ask what, how much, when, who, or why.
5. Questions must stand alone. Do not write "the company", "this article" or "as \
mentioned" - a reader with no article in front of them must understand the question.
6. Write 1 or 2 pairs. Write 2 only when the paragraph genuinely contains two \
separate facts.
7. If the paragraph states no clear fact about the company, return an empty list.

Reply with strict JSON only - a list of objects with "question" and "answer" keys. \
No markdown, no code fences, no commentary."""

EXAMPLE_CONTEXT = (
    "Tata Steel might need to pay more than Rs 17,000 crore as minerals tax dues to "
    "the state of Odisha, after the Supreme Court allowed states to levy taxes on "
    "mineral rights with retrospective effect from April 2005."
)
EXAMPLE_OUTPUT = (
    '[{"question": "How much might Tata Steel need to pay as minerals tax dues to '
    'Odisha?", "answer": "More than Rs 17,000 crore"}, '
    '{"question": "From when did the Supreme Court allow states to levy retrospective '
    'taxes on mineral rights in the Tata Steel case?", "answer": "April 2005"}]'
)


def build_messages(company: str, context: str, title: str | None = None) -> list[dict]:
    """Chat messages for one paragraph, in the format Qwen's template expects.

    The one-shot example does most of the formatting work: it shows the JSON shape,
    the answer length, and a question that names the company - all far more reliably
    than another paragraph of instructions would.
    """
    example_user = build_user_turn("Tata Steel", EXAMPLE_CONTEXT)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": example_user},
        {"role": "assistant", "content": EXAMPLE_OUTPUT},
        {"role": "user", "content": build_user_turn(company, context, title)},
    ]


def build_user_turn(company: str, context: str, title: str | None = None) -> str:
    parts = [f"Company: {company}"]
    if title:
        parts.append(f"Article headline: {title}")
    parts.append(f"Paragraph: {context}")
    return "\n".join(parts)
