# Guide 06 — Retrieval: Giving a Fine-Tuned Extractor Something to Read

Day 9 builds the layer between a user's question and the model. It is worth being precise
about why this layer exists, because "add RAG" is the kind of phrase that hides the actual
engineering decision.

## The model we actually trained

Guide 01 chose a `{question, answer, context}` schema, and Day 5 masked the loss to the answer
tokens. The thing that was optimized, 6,368 examples at a time, was one narrow skill:

> given a question and a passage that contains the answer, produce the answer span.

That is *not* the same as "knows about Indian markets". Ask the merged model a bare question
with no context and you are testing what Llama-3.2-3B memorized during pretraining, plus
whatever leaked in from three months of Moneycontrol articles. Both are unreliable, and the
fine-tuning made it *worse* at admitting that — it was trained to always emit a confident short
span, because every training label was one.

So the app's job is not to decorate the model. It is to make sure the model never runs without
the context its training assumes. Everything below follows from that.

## Two retrievers, because there are two kinds of question

```
"What is Reliance trading at?"          -> a number that changes every second
"Why did Tata Steel owe Odisha money?"  -> a fact frozen in an August 2026 article
```

No single store answers both. A vector index over news articles will happily return a share
price — the one printed in a story from three weeks ago — and the model, doing exactly what it
was trained to do, will extract it and state it as today's price. That is the single worst
failure this application can produce, and it looks completely fluent.

Hence two retrievers with different guarantees:

| | `StockDataRetriever` | `ArticleRetriever` |
|---|---|---|
| Source | yfinance / Yahoo Finance | the 10,257 cleaned paragraphs from Day 4 |
| Answers | price, change, day range, volume | events, results, statements, reasons |
| Freshness | minutes (delayed, see below) | frozen at scrape time |
| Fails by | network error — loudly | returning a weak passage — quietly |

The asymmetry in that last row drives the design. A yfinance outage is obvious and can be
reported. A bad corpus hit is invisible: retrieval returns *something* for every query, the
cosine score means nothing to the user, and the model will extract an answer from whatever it
is handed. Somewhere the pipeline has to be able to say "I don't have anything on that" instead
of handing over the fifth-best paragraph — and working out *which* signal can actually say that
turned out to be the most interesting measurement of the day. It is not the one you'd reach for.

## Routing: the cheap decision that carries the risk

Which retriever gets the question? The project report specifies embedding similarity: embed the
question, compare it against prototype price questions, take the cosine. `IntentRouter` does that,
backed by a keyword rule for when `sentence-transformers` (a ~90 MB model download) is not
installed — the app has to start and route sensibly without it.

Which of the two is actually pulling its weight is a measurable question, so it got measured: 36
questions, 18 price and 18 news, hand-labelled.

| Router | Correct |
|---|---|
| keyword rule alone | 25/36 |
| embeddings alone, raw question, best threshold | 33/36 |
| embeddings alone, company name stripped, best threshold | 33/36 |
| **keyword OR embeddings @ 0.40, company stripped** | **34/36** |

Three things fell out of that, none of which were obvious beforehand.

**The keyword rule has perfect precision and terrible recall.** It fired on *zero* news questions
and missed 11 of 18 price questions — "wipro price", "what does BEL cost on the exchange", "what
am I paying for a share of Trent". That profile is exactly what you want from the cheap half of a
union: it never causes the expensive error, it just doesn't catch much. So the two are combined
with OR, high-precision rule beside high-recall model, rather than one replacing the other.

**Strip the company name before embedding, too.** "How is TCS doing today" scores 0.54 against the
price prototypes; "how is doing today" scores 0.79. The name drags the sentence vector towards
"this is about a company" and away from "this is about a price" — the same double-counting as in
corpus search, in a different coordinate system, and fixed the same way. Median cosine for price
questions rose from 0.50 to 0.66 while news questions stayed at 0.20.

The two questions the shipped router still gets wrong are worth naming, because they fail in the
expensive direction: "what would one share of HDFC Bank set me back" (0.32) and "give me today's
number for ONGC" (0.35) both fall through to the corpus, where a price printed in a July article
can come back as an answer. There is no threshold that catches them without dragging news
questions along — "tell me about Maruti's new launch" scores 0.32 too. This is the residual risk
the design carries, and it is the reason every corpus answer is returned with its passage's
publication date attached rather than as a bare sentence.

**The threshold that felt right was wrong.** 0.55 is the number you would pick by eye for a
sentence-similarity task; it scores 32/36 and drops "how is TCS doing today" at 0.54 — one
hundredth short. The measured gap between the classes puts the decision boundary at **0.40**.
There is a general lesson in that: a cosine threshold has no absolute meaning, only a meaning
relative to the distribution of the two classes you are separating, and you cannot know it without
labelling a few dozen examples. It takes twenty minutes.

Intent is only half of a route, though. A price route also needs a *company*: "what is the share
price?" with nothing resolved cannot be answered by either retriever, and sending it to the corpus
produces the stale-price failure above. So the route is a conjunction:

```
LIVE_PRICE   <- price intent AND a resolved NIFTY 50 ticker
CORPUS       <- everything else that resolves to something retrievable
REFUSE       <- no company resolved; or nothing retrieved above the score floor
```

with one more guard in front of it: a **tense veto**. "What *was* the price after Q1 results" is
price-shaped but historical, and the live feed cannot answer it however it looks. Past-tense
markers route to the corpus before either signal is consulted.

Company resolution reuses `src/scraping/matcher.py` — the same alias table, longest-alias-wins rule
and exclusion list that Day 3 used to decide which company an article was about. It already knows
that "SBI Life" is not "SBI", that "Reliance Power" is a different group, and that ITC Hotels
demerged. Rebuilding that in the app layer would mean two copies of the same domain knowledge,
drifting. What it did need was a *keyboard* layer: a URL slug always spells "mahindra-and-mahindra"
where a person types "M&M" or "INFY", so `match_text` loads the table with the tickers themselves
and a handful of chat forms added. On 300 held-out test questions it resolves a company for 99.7%
of them and never picks the wrong one.

## Why TF-IDF and not embeddings for the corpus

The corpus retriever inverts the usual advice. Dense embeddings are the default answer for RAG;
here, sparse lexical retrieval is the better tool, for reasons specific to this data:

- **The queries are dominated by proper nouns and numbers.** "Odisha", "Q1", "17,000 crore",
  "Shivakumar". Exact-token overlap *is* the relevance signal. IDF gives rare tokens their weight
  automatically — "Odisha" is decisive precisely because it appears in few paragraphs.
- **Near-synonyms are the enemy, not the goal.** A general-purpose sentence encoder puts "Q1
  profit" and "Q2 profit" close together; for financial questions that similarity is a bug.
- **Fifty companies, one shared vocabulary.** Every paragraph is business news in the same
  register, so dense vectors cluster tightly and discriminate poorly. Lexical scoring stays sharp.
- It builds in about a second over 10k paragraphs, needs no model download, and is inspectable —
  you can print the terms that fired.

The larger point: dense retrieval earns its keep when the user's words differ from the corpus's
words. Here they mostly do not.

The **ticker pre-filter** matters more than the scoring function. When a company resolves, search
only that company's paragraphs — roughly 200 instead of 10,257. Recall is unaffected (a question
about Tata Steel is not answered by an Infosys paragraph) and precision jumps, because the
strongest distractor for any financial query is the *same sentence about a different company*.

### Then delete the company name from the query

The filter creates a problem that took a measurement to see. IDF is fitted over the whole corpus,
where "tata steel" appears in ~200 of 10,257 paragraphs and is therefore weighted as a rare,
decisive term. But *inside* TATASTEEL's own paragraphs it is in nearly every one — it discriminates
nothing, while its high weight drowns out the token that does. Ask "why might Tata Steel owe money
to Odisha" and the top hits come back about Tata Steel, generically, with no Odisha in sight.

The fix follows from what the filter already did: the company name has been consumed by the
metadata filter, so letting it score again is double-counting. Strip the matched aliases out of
the query, keep everything else. Measured over 300 held-out test questions, answer-in-context at
top-3 went **61.0% → 70.0%**. With no ticker filter the name stays in — there it is doing real work.

## Tuning against numbers instead of taste

Every constant in `corpus.py` came from the same 300-question harness: retrieve, then check whether
the gold answer's normalized text appears anywhere in the context the model would have been given.
It is a strict proxy — the gold answers are themselves model-written paraphrases, so a passage can
support an answer without containing it — but it is a *consistent* one, which is what a knob needs.

| Change | Answer present in context |
|---|---|
| top-1 only | 49.3% |
| top-3 | 70.0% |
| top-5 | 76.0% |
| without the company-name strip | 61.0% |
| unigrams instead of unigrams+bigrams | 69.7% |
| keeping stopwords / `min_df=1` / no sublinear tf | 69.3–71.0% |

Two things worth reading off that table. The **vectorizer knobs are noise** — every reasonable
TF-IDF configuration lands within a point of the others, so time spent tuning them would have been
wasted. The **structural choices are not**: how many passages you keep, and whether you stop the
company name from scoring twice, move the number by 9–20 points.

Passage count is the one real trade-off. Top-5 retrieves 6 points more answers, but training
contexts were *single paragraphs*, and a context four times that length is a distribution shift
away from what the model was fine-tuned on — bought with tokens, and with more distractors in the
window. The default is top-3 inside a 1,500-character budget, which measures at 69.0% end to end -
essentially the whole of what top-3 has to offer, and against 53.3% for the same search without
the ticker filter. Both are constructor arguments; when the model is in the loop
end-to-end this is the first thing worth re-measuring against answer quality rather than against
retrieval containment.

## The context-format contract

Whatever the retrievers return has to reach the model in the shape training used:

```
Question: {question}
Context: {context}
Answer:
```

This is not a detail. The model was trained on that exact string, down to the newlines and the
trailing `Answer:` cue, and it has 6,368 gradient updates of expectation about it. Prompt it with
`Q:` / `A:`, or with a chat template wrapped around it, and quality drops for reasons that are
invisible in the code. So the app imports `build_prompt` from `src/training/format.py` — the same
function the training notebook and the evaluation used. One definition, three call sites, no
chance of drift.

The live-data retriever therefore has a formatting job, not just a fetching job: turn a quote into
prose in the same register as the training contexts.

```
Reliance Industries (RELIANCE) is trading at Rs 1,309.50 on the NSE, down Rs 12.50
(-0.95%) from the previous close of Rs 1,322.00. The day's range is Rs 1,301.90 to
Rs 1,324.20.
```

Declarative sentences with the numbers in them — the same thing an article paragraph looks like.
The model then does what it was trained to do: extract the requested span. It is not being asked
to do arithmetic or to know what a stock is.

## What a quote must say, and must not

`fast_info` gives a price with no timestamp, and Yahoo's NSE data is delayed — commonly by about
15 minutes. A finance chatbot that presents a delayed price as "the current price" is making a
claim it cannot support, so the retriever records the fetch time explicitly, and the app surfaces
it. The distinction — *this is when we asked, not when it traded* — is small to implement and
exactly the kind of thing that separates a demo from something you would let someone act on.

Related traps handled in `stock.py`:

- **Market closed.** `lastPrice` keeps returning Friday's close all weekend. Without the
  timestamp this reads as live.
- **A cache is mandatory, not an optimization.** Every user turn would otherwise hit Yahoo, from
  an undocumented endpoint with unpublished rate limits. A short TTL cache makes repeated
  questions about the same stock free and keeps the app well-behaved.
- **Missing fields.** `previousClose` is absent for some symbols; the change line is then simply
  omitted rather than computed from a `None` treated as zero.
- **The `.NS` suffix.** Yahoo needs the exchange suffix for Indian equities; `data/companies.json`
  stores bare NSE tickers, so the retriever appends it. `M&M` and `BAJAJ-AUTO` survive the round
  trip; they are the reason the suffix is appended rather than the symbol being reconstructed.

## What can actually say "I don't know"

The obvious guard is a similarity floor: refuse when the best passage scores below some cosine.
Measured, it does one of the two jobs people expect from it and completely fails the other.

**It detects retrieval failure.** Of 300 held-out questions, 22 scored below 0.10 against their
own company's paragraphs; those 22 had the answer in the retrieved context **4.5%** of the time,
against **70.9%** for the 278 above the line. That is a clean separation, and it is why
`MIN_SCORE = 0.10` exists. (Raising it to 0.15 throws away 46 more questions that are answerable
38% of the time — the floor is calibrated, not picked.)

**It does not detect an off-topic question.** Ask the corpus things it has no business answering
and the top cosine comes back at 0.22 for "what is the capital of France", 0.25 for "tell me a
joke about cats", 0.43 for "who won the 2026 football world cup". Those sit *above* the median
score of questions the retriever answers correctly. A 10,257-paragraph news corpus contains
enough English that any query finds a lexical friend — and "world cup" genuinely matches a
paragraph, because Tata Steel sold a football club.

So the topicality gate is not the score. It is the **entity check**: does the question name a
NIFTY 50 company at all? On the same 300 questions the matcher resolved one for **99.7%** of them
and never picked the wrong one, while every off-topic control resolves nothing. That asymmetry is
what the pipeline refuses on:

- no company resolved → *"I can only answer questions about the 50 NIFTY companies in my sources."*
- price intent, no company → *"Which company's price?"* — never a corpus fallback, which is
  exactly how a three-week-old price gets read out as today's
- yfinance error or unknown symbol → *"Couldn't reach the price feed just now."*
- best passage below the floor → *"I don't have anything on that in my sources."*

The generalizable bit: a similarity score is a measure of *relative* rank within a corpus, and it
was never a measure of whether the corpus is the right one. When you need a "should I answer this
at all" signal, look for something categorical — an entity that resolves or does not — rather than
thresholding a continuous score that has no zero.

This is also the one place the fine-tuned model must not be consulted, because it will always
produce a confident short answer. The guard sits in the pipeline, above the model, where it can
still be enforced. And the pipeline returns its sources alongside every answer for the same
reason — a passage's URL and date, or a quote's fetch time, let a user check the claim, which is
what makes an unverifiable 3B model's output usable at all.

## Talking points

- **"Fine-tuning taught a skill, not facts"** — the model is an extractor; the retrieval layer's
  job is to guarantee it never runs without the input its training assumes.
- **The routing failure that costs the most** is a price question answered from a news article:
  fluent, confident, weeks stale. Routing is a conjunction of intent *and* a resolved entity, and
  anything ambiguous refuses instead of guessing.
- **A high-precision rule and a high-recall model, unioned.** The keyword rule caught 7 of 18 price
  questions and zero false alarms; embeddings took the pair to 34/36. Knowing which half of a
  hybrid is contributing what requires labelling a few dozen examples - and the threshold that
  "looks right" (0.55) measured worse than the one the data picked (0.40).
- **Sparse beat dense here**, and knowing *why* (proper nouns, numbers, single-domain corpus,
  near-synonyms being a liability) is the point — not the choice itself.
- **The metadata filter did more than the ranker.** Restricting to one company's ~200 paragraphs
  removes the strongest distractor class: the same sentence about a different company — and then
  the company name has to be *removed from the query*, or the corpus-wide IDF that made it a
  decisive term inside a filtered search makes it a decisive term that matches everything
  (61.0% → 70.0%).
- **A similarity floor cannot tell you the question is off-topic.** It detects retrieval failure
  well (4.5% vs 70.9% answer-in-context either side of it) and out-of-domain not at all — nonsense
  queries outscore real ones. The entity check does that job; measure before trusting a threshold.
- **One prompt-format definition** shared by training, evaluation and serving. Drift between how
  a model is trained and how it is prompted is a silent quality bug.
- **Delayed data labelled as delayed.** A timestamp on every quote, because "current price" is a
  claim.
