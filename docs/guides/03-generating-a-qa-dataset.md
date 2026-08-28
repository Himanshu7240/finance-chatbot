# Guide 03 — Generating and Splitting a QA Dataset

Day 3 produced 2,014 articles of raw news text. That is not a dataset. This guide covers
the four steps that turn it into one — cleaning, generation, validation, splitting — and
the reasoning behind each, because every one of them is a place where a plausible-looking
shortcut quietly ruins the model you train later.

## Step 1 — Cleaning: the filter that decides dataset quality

`src/dataset/clean.py` does three things. Two are housekeeping; one is the real work.

**Normalization** (housekeeping): NFKC, non-breaking spaces to real spaces, curly quotes
to straight, `₹` to `Rs `. Left alone these are tokenizer noise — the model burns capacity
learning that `₹17,000` and `Rs 17,000` mean the same thing, for no benefit.

**Deduplication** (housekeeping, with a catch): financial wire copy repeats near-verbatim
across outlets and days. Exact hashing catches almost none of it, because one changed word
defeats it. So near-duplicates are caught with **Jaccard overlap over 5-word shingles** at
a 0.8 threshold. The catch is cost: comparing all pairs is ~160M comparisons. An inverted
index from shingle to paragraph fixes that — only paragraphs already sharing a shingle get
compared.

**The fact-bearing filter** (the real work): a paragraph with no number, date, or business
event in it *cannot* produce an answerable question. Ask a model to generate a Q&A pair
from "The company remains focused on long-term value creation" and it will produce one —
by inventing something. This filter removed **4,201 of 18,098 paragraphs (23%)**, and it
matters more to final quality than any hyperparameter chosen later.

## Step 1b — The attribution bug, and why validation found it instead of cleaning

This one was only caught by reading actual generated output, which is the argument for
always doing that. An early sample contained:

> **Axis Bank** | "The company has secured domestic orders worth Rs 128 crore from
> Hyderabad-based companies."
> → *"What orders has Axis Bank secured from Hyderabad-based companies?"*

Axis Bank is a bank. It does not win Rs 128 crore equipment orders. The source article was:

> *"Stocks to Watch Today: Reliance Ind, Paytm, Wipro, Indian Hume Pipe, JK Cement,
> Alembic Pharma, Axis Bank, IndusInd, Can Fin, Tanla Platforms in focus on 27 April"*

A **multi-company roundup**, where each paragraph covers a different company. Day 3's scraper
labels an article by its URL slug, and every paragraph inherits that one label — so a
paragraph about Indian Hume Pipe got filed under Axis Bank. The generator then obeyed the
rule "put the company name in the question" and produced a confident false statement.

What makes this dangerous is that **no groundedness check can catch it**. The answer *is*
supported by the paragraph. Every number *does* appear in the source. The text is real; only
the label is wrong. A validator that checks answer-against-context is structurally blind to
a wrong context-to-company mapping.

Two things worth internalising:

- **Checking the article title doesn't help.** 99.3% of paragraphs that don't name their own
  company have it in the title — *because roundup headlines list every company they cover*.
  The obvious fix was worthless, and only measuring showed that.
- The filter had to be built on the roundup signature instead: known headline patterns
  ("Stocks to Watch", "Buzzing stocks", "Market Live", …) or a title naming 3+ NIFTY 50
  companies, combined with "does this paragraph name a *different* company than its label".

That removed a further **3,525 paragraphs (19%)**.

Result: **10,257 clean paragraphs** from 1,910 articles.

The residue is instructive: the cleaner only knows the 50 NIFTY companies, so paragraphs
about *non-NIFTY* entities (Old Bridge Mutual Fund, Urban Company, Oil India) still slip
through — and get caught downstream at validation as `company_not_named`, 1,729 of them.
Two independent filters catching different slices of one problem is not redundancy; it is
what defence in depth looks like when neither filter can see the whole thing.

## Step 2 — Generation: a 7B teacher for a 3B student

We generate the Q/A pairs with **Qwen2.5-7B-Instruct** running on a free Kaggle GPU, rather
than calling a hosted API.

This is **knowledge distillation**: a larger teacher model produces training data for a
smaller student (our Llama-3.2-3B). It's a standard and well-studied technique — the student
learns the teacher's behaviour on this task without needing the teacher's parameter count at
inference time.

Why local rather than an API, given a frontier model would produce cleaner output:

- **Reproducibility.** Anyone who clones this repo can rebuild the dataset. No API key, no
  spend, no "the results depended on a model I can't share."
- **Licensing.** Qwen2.5 is Apache 2.0, so there's no ambiguity about the derived dataset.
- **Cost.** Zero, against roughly $6–$29 for the same job through a hosted API.

The honest cost: a 7B model breaks strict JSON more often and needs harder filtering.
That's a runtime expense, not a quality ceiling — provided the validation in step 3 is real.

### The prompt is part of the dataset design

`src/dataset/qa_prompt.py` lives in the repo, not in the notebook, because changing it
changes every downstream number. Each rule in it exists to stop a specific failure:

| Rule | Failure it prevents |
|---|---|
| Answer only from the paragraph | Model uses outside knowledge; student learns to assert unsupported facts |
| Company name must appear in the question | "What was the revenue?" — unanswerable without the article |
| Answers ≤ 25 words | Answers that are just the paragraph restated |
| No yes/no questions | One-word answers carry almost no training signal |
| Self-contained questions | "The company" has no referent at inference time |
| **May return an empty list** | Forcing output from thin text is exactly where hallucinated training data comes from |

The one-shot example does more work than any of the prose: it demonstrates the JSON shape,
the answer length, and a question naming the company, all at once.

## Step 3 — Validation: assume the generator lies

Every generated pair runs a gauntlet in `src/dataset/qa_validate.py`, and each rejection is
counted **by reason** — that breakdown tells you whether to fix the prompt or the filter,
which a single "kept N" number never does.

The strictest check: **every number in the answer must appear in the paragraph.** In a
finance dataset a hallucinated figure is the worst possible training example — it teaches
the student to state numbers with confidence and no support. There is no tolerance band
on this one.

Alongside it, a **groundedness score**: the share of the answer's content words that appear
in the paragraph, thresholded at 0.7. Measured on real examples:

| Answer | Score |
|---|---|
| "More than Rs 17,000 crore" (exact span) | 1.00 |
| "April 2005" (date span) | 1.00 |
| "Around Rs 25,000 crore" (invented figure) | 0.50 |
| "The company will expand its steel capacity in Europe" (off-topic) | 0.20 |

Real spans and fabrications separate cleanly, which is what makes a fixed threshold safe here.

## Step 4 — Splitting: the mistake that inflates your eval score

**Split by article, never by paragraph.**

Paragraphs from one article overlap heavily — the same figures, the same quotes, the same
framing restated. Split them randomly and near-identical text lands in both train and test.
The model then recognises at eval time what it memorised at train time, the test score comes
out high, and the number is meaningless. This is **data leakage**, and it is the single most
common way a fine-tuning project reports results it cannot reproduce in the real world.

Grouping by article puts every paragraph of a story on one side of the wall. Two further
details in `src/dataset/split.py`:

- **Stratified by company**, so all 50 appear in train, val and test. A company held out
  entirely looks like catastrophic failure at eval when it is really just absence.
- **Balanced on paragraph counts, not article counts.** Articles range from 1 to 30 usable
  paragraphs, so equal article counts would give lopsided splits.

Result: 80.0% / 10.1% / 9.9%, all 50 companies present in each — and verified empirically:
**zero source articles are shared between train and test**.

## What actually came out

| | |
|---|---|
| Raw paragraphs scraped | 18,098 |
| After cleaning + attribution filtering | 10,257 |
| Pairs generated by Qwen2.5-7B | 11,590 |
| **Final dataset** | **8,292** (train 6,368 / val 973 / test 951) |

Rejections, in order: `unknown_paragraph` 3,368 (generated before the attribution filter
existed, then discarded by it), `company_not_named` 1,729, `invented_number` 538,
`ungrounded` 449, `not_self_contained` 353, `duplicate_question` 123, and a long tail.

**Only 3 responses out of 11,590 were unparseable JSON.** The headline worry about using a
local 7B — that it would mangle structured output — turned out to be a non-issue. The real
attrition came from attribution, which no model choice would have fixed.

8,292 is below the 10,000–17,000 target set on Day 2. That target was chosen to match the
original project's range, not from any technical requirement, and Guide 01 argued from the
start that volume alone isn't the goal. 8,292 well-attributed examples are worth more than
13,000 where one in five carries the wrong company name — and the alternative was training a
finance model on statements that were confidently, checkably false.

## Talking points

- *"Why generate data with a model instead of labelling it?"* — Knowledge distillation. A 7B
  teacher produces supervision for a 3B student; hand-labelling 13,000 examples isn't viable,
  and the alternative — training on raw article text — teaches language modelling, not
  question answering.
- *"How do you know the generated data isn't hallucinated?"* — Every number in an answer must
  appear in the source paragraph, and answers are scored for word-level groundedness against
  it. Rejections are tracked by reason, so generator behaviour is measured, not assumed.
- *"How did you split the data?"* — By article, stratified by company. Splitting by paragraph
  would leak near-duplicate text between train and test and inflate the eval score.
- *"What was the highest-leverage decision?"* — The fact-bearing filter. Dropping 23% of
  paragraphs that couldn't support a real question did more for dataset quality than any
  later modelling choice.
- *"What would you do differently at scale?"* — Generate with a stronger teacher and keep the
  same validator. The filtering is the durable part; the generator is swappable.
- *"Tell me about a bug you found."* — The roundup attribution bug. A "Stocks to Watch"
  listicle labelled every paragraph with one company, so a Rs 128 crore order belonging to
  Indian Hume Pipe was attributed to Axis Bank. It was invisible to groundedness checking
  because the answer genuinely was supported by the text — only the label was wrong. Found
  by reading raw output, not by any metric, and it cost 19% of the corpus to fix properly.
