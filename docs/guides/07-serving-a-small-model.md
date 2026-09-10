# Guide 07 — Serving a Small Model, and Testing Something You Can't Run

Day 10 puts a Gradio interface on top of the Day 9 pipeline and writes the tests that keep it
honest. Neither half is decoration: the UI is where a 3B model either earns trust or quietly
misleads someone, and the test suite is the only thing that will notice when a refactor breaks
the refusal that stops it doing so.

## The cold-start problem shapes the whole interface

The merged model is 6.4 GB. Downloading it takes minutes on a first run, loading it takes tens of
seconds after that, and on a CPU box a single answer takes a few seconds more. A chat UI that
blocks on all of that before its first paint is a bad interface, and the usual fixes — a spinner,
a progress bar — only make waiting more transparent, not shorter.

The Day 9 pipeline already gave us something better to do. `RAGPipeline` runs without a generator
and answers from the retrieved evidence itself: the quote sentence, or the retrieved paragraph.
That is not a degraded mode bolted on for the demo — it is *the retrieved evidence*, which is the
grounding the model would have been given anyway. So the app starts in retrieval-only mode, is
useful in about a second, and the model loads when the user asks for it:

```
start (≈1s)  ->  TF-IDF index built, yfinance live, answers are evidence
toggle       ->  6.4 GB model loads once, answers become extracted spans
```

This also makes the model's contribution *visible*, which is more than most demos manage. Ask the
same question twice, once each way, and you see exactly what fine-tuning bought: the paragraph
versus the span pulled out of it. When the difference is small, that is worth knowing too.

## Chat state is a trap for a model trained on single turns

Every training example was one `{question, context, answer}` triplet with the prompt format from
`src/training/format.py`. The model has never seen a conversation. It has no chat template, no
turn markers, and no reason to know what "it" refers to.

The tempting move — concatenating history into the prompt, the way you would for an instruct model
— breaks the exact format the model was fine-tuned on, and Guide 06 already covered what that
costs. So **each turn is independent**: the UI keeps a visible transcript for the person, and
sends the model one question and one context, every time.

That leaves a real usability hole, though. "What is Reliance trading at?" followed by "and Wipro?"
is how people actually talk, and the second question resolves no company under the Day 9 rules, so
it gets refused. The fix that does *not* involve touching the prompt is a **sticky entity**: the
pipeline accepts the previous turn's company and falls back to it when the current question names
none.

```python
pipeline.answer("and how is it doing today?", last_company=previous)
```

It is carried in the UI's session state, not in the prompt, and the answer says which company it
assumed. Two properties worth insisting on:

- **It never overrides an explicit mention.** If the question names a company, that one wins; the
  carry-over only fills a gap.
- **It is visible.** "Assuming Reliance Industries, from your last question" is a sentence the user
  can disagree with. A silent assumption about *which company's share price you just quoted* is the
  kind of thing that makes a finance tool dangerous rather than convenient.

Sticky state also has to be scoped per session — Gradio's `gr.State` is per browser session, so two
users never inherit each other's company. That is one line, and getting it wrong would be a
cross-user data leak of exactly the sort that makes demos embarrassing.

## What the interface has to show

A retrieval-grounded answer is only as good as the user's ability to check it, so three things
appear alongside every response rather than being buried behind a "details" toggle:

| Shown | Why |
|---|---|
| **Route** — live price / news / refused | The user is entitled to know whether a number came from a feed or from a July article |
| **Sources** — article title, date, URL; or the quote's fetch time | The claim can be checked; the *date* is what defuses stale news |
| **The context** the model was given | The whole answer is supposed to be extracted from it; this is what makes hallucination visible instead of invisible |

A refusal must look like a refusal, too — not like an answer that happens to be evasive. The three
refusal messages from Guide 06 stay verbatim, and the route badge says `refused`, because the honest
failure is the product feature that separates this from a chatbot that always has something to say.

And one line of standing text: the data is delayed, the model is a 3B fine-tune that gets roughly
8% of numeric answers wrong, and none of it is investment advice. Not a modal, not a checkbox —
just always on screen, where the numbers are.

## Testing an application whose two inputs are a live market and a 6.4 GB file

This is where the Day 9 constructor signatures start paying off. Nothing worth testing here needs
either dependency:

```python
StockDataRetriever(fetch=..., clock=...)     # injected feed, injected time
RAGPipeline(generator=..., stock=..., corpus=...)   # injected everything
```

The seams were put there for exactly this. A fake `fetch` returns whatever shape the test needs —
a normal quote, a quote with `previousClose` missing, an outage that raises. A fake `clock` makes
the TTL cache testable in microseconds instead of a minute of real waiting. A fake `generator`
lets every routing test run without a single model file on disk.

What the suite covers, and why each one is worth a test rather than a glance:

- **The alias table**, because it encodes domain knowledge that is easy to break and silent when
  broken: "SBI Life" must not resolve to SBI, "Reliance Power" must resolve to nothing at all, and
  the chat forms ("m&m", "INFY") must keep working alongside the URL slugs Day 3 depends on.
- **Routing**, as a table of question → expected route. This is the contract that keeps a price
  question away from a three-week-old article, and it is one careless regex edit from breaking.
- **Every refusal path**, because a refusal that silently becomes an answer is the worst possible
  regression here and produces no error, no exception, and no failing assertion anywhere else.
- **Quote formatting**, including the missing-field and zero-change cases, since that string is
  what the model reads and what the user sees.
- **The cache**, with a fake clock: a hit inside the TTL, a refetch after it.
- **Prompt format**, asserting the served prompt is byte-identical to what `build_prompt` produces.
  The whole "one definition, three call sites" argument from Guide 06 is worth precisely nothing if
  nothing checks it.

Tests that need the built corpus are skipped when `data/processed/` is absent, so the suite runs on
a fresh clone — but they are not deleted, because on a machine that *has* the data they check the
thing that actually breaks: the score floor and the ticker filter.

Notice what is *not* tested: yfinance's real responses, the model's answer quality, and Gradio's
rendering. The first is someone else's service, the second is what the Day 6–7 evaluation is for
and is not a pass/fail property, and the third is a framework. Tests that would need the network or
the weights to pass are tests that will be skipped in practice, and a skipped test protects nothing.

## Talking points

- **Ship the app before the model.** Retrieval-only mode makes the interface useful in a second and
  makes the fine-tune's contribution visible by comparison, instead of hiding it behind a spinner.
- **A single-turn model gets single-turn prompts.** Conversation state lives in the UI as a sticky
  entity, never in the prompt, because the prompt format is a contract with training.
- **An assumption the user can see is a feature; a silent one is a liability.** "Assuming Reliance,
  from your last question" is the whole difference.
- **The route and the source date are part of the answer**, not metadata. A price from a feed and a
  price from a July article are different claims and must not look alike.
- **Dependency injection is what made this testable.** A fake feed, a fake clock and a fake
  generator cover every routing and refusal path with no network and no weights — and the tests
  that would need either are the tests worth not writing.
