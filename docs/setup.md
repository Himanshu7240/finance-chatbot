# Setup, Secrets and Things That Will Bite You

Getting the app running takes two commands. Everything else on this page is about the parts that
fail in ways whose error messages point somewhere other than the cause — collected as they were
actually hit, over twelve days.

## Install

```bash
pip install -r requirements.txt
pytest                      # 80 tests; no network, no model download, ~12 s
python -m src.app.ui        # http://127.0.0.1:7860
```

Python 3.11+. No local GPU is needed for anything in this repo — the two GPU steps run on Kaggle
(see [`reproducing.md`](reproducing.md)), and the merged model runs on CPU, slowly but correctly.

## What needs a token, and what doesn't

| Task | Token | Why |
|---|---|---|
| Running the chat app | **none** | The merged model is a public repo and loads with a plain `from_pretrained` — no `peft`, no access to the gated base |
| Scraping | none (optional `SCRAPER_CONTACT`) | Sent as a contact header so the site operator can reach you |
| Fine-tuning, merging | **`HF_TOKEN`** | `meta-llama/Llama-3.2-3B-Instruct` is gated, and pushing needs write access |

Locally the token comes from `.env` (`cp .env.example .env`, then fill in `HF_TOKEN`); `.env` is
gitignored and must stay that way. On Kaggle it comes from a **notebook secret named `HF_TOKEN`**,
read through `UserSecretsClient` — never pasted into a cell. The original project this rebuilds
leaked its token into a committed notebook, which is the reason this rule exists rather than an
abstract preference.

Gated access has to be requested once, from the model page, with the same account the token
belongs to. Approval is usually minutes. Without it the download fails with a 401/403 that reads
like a bad token rather than a missing entitlement.

## Optional dependencies, and what you lose without them

| Package | Without it |
|---|---|
| `sentence-transformers` | The intent router falls back to its keyword rule — logged once, at startup. Routing accuracy drops from 34/36 to 25/36 on the battery in [Guide 06](guides/06-retrieval-and-the-app-layer.md), all of it recall: price questions get answered from news instead of the feed |
| `yfinance` | Price questions refuse with "couldn't reach the price feed". Everything else works |
| A built `data/processed/` | Corpus answers are unavailable and those tests skip. Live quotes still work |
| `transformers` + `torch` | Retrieval-only mode. The app is designed to start this way anyway |

## Traps

**`ImportError: Found an incompatible version of torchao ... only versions above 0.16.0`, on a
model that has nothing to do with torchao.** PEFT probes every quantization backend while wrapping
each `Linear` layer, and its `is_torchao_available()` *raises* when torchao is installed but old
instead of returning `False`. Kaggle's image ships 0.10.0. The fix is to make the probe not find
it — `sys.modules["torchao"] = None` before importing PEFT — not to upgrade torchao, which drags
Kaggle's torch build with it. Implemented and explained in the first cell of
`notebooks/merge_and_publish.ipynb`; the general lesson is in
[Guide 05](guides/05-merging-and-publishing-a-model.md).

**Merging into the 4-bit base.** Loading the quantized base, calling `merge_and_unload()` and
saving *works* — and quietly costs you most of the fine-tune, because the adapter's small precise
update gets rounded through a 4-bit grid. Merge in fp16 against the original weights. Nothing warns
you; the only symptom is a worse model.

**Kaggle installs that downgrade torch.** `pip install peft` will happily resolve a different torch
and break the whole image. Use `--no-deps` for anything you add on top of Kaggle's stack, and check
`torch.__version__` afterwards.

**Your Hugging Face username is probably not your GitHub username.** Costs an afternoon of
debugging 401s that look like permission problems, because the Hub returns 401 for both private and
nonexistent repos. Check the account page — it 404s honestly.

**Windows: `huggingface_hub` symlink warning.** Harmless — caching just uses more disk. Silence it
with `HF_HUB_DISABLE_SYMLINKS_WARNING=1`, or enable Developer Mode.

**Windows console: `UnicodeEncodeError` on emoji.** The route badges are emoji and `cp1252` cannot
encode them. Only affects printing to a terminal, never the app; set `PYTHONIOENCODING=utf-8` if
you are piping module output around.

**A scraper that looks blocked.** Moneycontrol's CDN 403s non-browser user agents, so the fetcher
sends a browser UA and identifies its purpose in an `X-Purpose` header instead — documented as a
deviation in [`dataset-design.md`](dataset-design.md). If you see 403s anyway, you are probably
requesting faster than one page per ~3 seconds; the rate limit is not optional politeness, it is
what keeps the run working.

## Where things live

```
src/scraping/    sitemaps, robots-aware fetcher, company matching, article extraction
src/dataset/     cleaning, splitting, QA validation, dataset assembly
src/training/    prompt format (the single source of truth), metrics, LoRA merge
src/app/         retrieval, routing, pipeline, model wrapper, Gradio UI
tests/           80 tests, all offline
notebooks/       the three Kaggle notebooks: QA generation, fine-tuning, merge + publish
data/            gitignored; rebuildable via reproducing.md
models/          gitignored; adapter and merged weights land here
```
