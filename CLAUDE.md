# CLAUDE.md

## Project
A multi-step **markets research agent** for crypto and US stocks, built on **LangGraph**.
Ask it a markets question; it plans the research, gathers live evidence, critiques its own
coverage and loops back to fill gaps, writes a cited report, and finishes with a
professional-style analyst view. Portfolio project #3 for the owner (an AI engineer), and
the one that demonstrates *agentic* skills specifically: stateful orchestration,
reflection, tool use, human-in-the-loop, and evaluation.

The owner is comfortable with Python but new-ish to agent frameworks: explain changes in
plain language and prefer small, readable code over clever abstractions.

## Current state
All planned milestones are **built, verified end to end, and committed** (13 commits):

1. Scaffold + **Planner** node
2. **Researcher** node + live-data tools
3. **Critic** node + conditional **loop-back** (reflection) — both branches verified
4. **Synthesizer** node — cited report
5. Computed **technical indicators** (RSI/EMA)
6. **Human-in-the-loop** plan review (`--review`)
7. **RSS news** tool (dated headlines, incl. Yahoo per-ticker feeds)
8. **Fundamentals** live (Alpha Vantage key set)
9. **Analyst** node (professional-investor view)
10. **Evaluation harness** (trajectory metrics + LLM-as-judge) + recorded baseline

## The graph
```
START → planner → [human_review*] → researcher → critic ─┬─(gaps, under cap)→ researcher
                                                          └─(complete/cap)──→ synthesizer
                                                                              → analyst → END
* only when built with review=True (the --review flag)
```
- **Planner** — decomposes the question into ≤5 sub-questions, each tagged
  `price | technicals | fundamentals | earnings | news | macro` plus the tickers it
  concerns. `earnings` is stocks-only (quarterly results); `fundamentals` is valuation
  ratios. They're separate tags so a question only pays the extra Alpha Vantage calls
  when it actually needs quarterly data.
- **Researcher** — routes each sub-question to the right tool, then distills all raw tool
  output into findings in **one batched LLM call**. Appends to `evidence`.
- **Critic** — judges coverage; if important gaps remain it emits targeted follow-up
  sub-questions and the conditional edge loops back.
- **Synthesizer** — balanced cited report + "what's uncertain" + fixed disclaimer.
- **Analyst** — stance (Accumulate/Hold/Trim/Avoid/Watch), conviction, horizon, reasoning,
  **invalidation conditions**, key risks.

## Environment
- Use the venv in `./venv` for every install and run (`venv\Scripts\python.exe` on Windows).
- Python 3.10+ (built on 3.13).
- All config via environment variables in `.env` (see `.env.example`). Never hardcode,
  print, or commit secrets. `.env` is git-ignored.
- Run (CLI): `python main.py "your question"`, add `--review` for the human gate.
- Run (UI): `streamlit run streamlit_app.py` — streams progress node-by-node and turns
  the review gate into approve/revise/cancel buttons. It imports the agent directly;
  there is deliberately **no separate backend** to keep the stack small.
- Evaluate: `python -m eval.evaluate [--no-judge] [--limit N]`.

## Data sources (all free tiers)
| Concern | Source | Notes |
|---|---|---|
| Crypto price + technicals | CoinGecko | keyless; slow (~19s/call), aggressive rate limits |
| Stock price + technicals | Twelve Data | free key; ~8 req/min |
| Stock fundamentals | Alpha Vantage `OVERVIEW` | free key; **~25 req/day** — degrades to "unavailable" |
| Quarterly results | Alpha Vantage `EARNINGS` + `INCOME_STATEMENT` | **2 calls** per stock; only for `earnings` sub-questions |
| News | RSS via feedparser | CoinDesk/Cointelegraph/Decrypt; CNBC/Yahoo/MarketWatch **+ Yahoo per-ticker** |
| Macro / supplementary | DuckDuckGo (`ddgs`) | keyless |

Indicators are computed locally: **Wilder's RSI(14)**, **EMA13/EMA21**, with a
plain-language momentum read — same approach as the owner's Telegram bot.

## Model gotchas (2026, learned the hard way)
- The API key has the **Gemini 3.x** family. `gemini-2.5-flash` and `gemini-2.0-flash`
  return **404 "no longer available to new users"**; `gemini-flash-latest` has 503'd under
  load. **Pinned `GEMINI_MODEL=gemini-3.5-flash-lite`** (works, more free headroom).
- `gemini-3.6-flash` free tier is only **20 requests/day** — it's used for the *judge*
  only, deliberately different from the generation model.
- Gemini 3.x **ignores `temperature`** (warns) and returns `content` as a **list of
  blocks**, not a string → always normalize with `to_text()` in `agent/llm.py`.
- Windows consoles are cp1252: `main.py` forces UTF-8 stdout, and printed output uses
  ASCII arrows, or LLM text with `→`/curly quotes crashes the CLI.
- The checkpointer must be given a `JsonPlusSerializer(allowed_msgpack_modules=[…])`
  listing our Pydantic state models, or LangGraph warns that restoring them will be
  blocked in a future version.

## Decisions and rules
- **No RAG.** Markets are a real-time domain, so a fixed vector store would be stale on
  arrival. The backbone is live-data tools. (Revisit only if document-heavy sub-tasks like
  SEC filings or earnings transcripts are added — that's the one place RAG would fit.)
- **The loop must always terminate.** The critic increments `loops`; the router forces the
  exit at `MAX_CRITIC_LOOPS`. Never remove that guard.
- **One shared LLM** powers the reasoning nodes today. Per-node models are supported
  (`get_llm(temperature=…)`) but not needed; the **judge must stay a different model**.
- **Honesty is a feature, not decoration.** Findings must say "unavailable" rather than
  guess; the report keeps a "what's uncertain" section; the analyst must weigh conflicting
  signals and state **invalidation conditions**; every output carries a disclaimer.
- **The analyst gives no position sizes, no price targets, and no personalised
  instructions.** It is an interpretation of evidence, not advice — an LLM persona has no
  validated track record. (Same posture as the owner's Telegram bot, whose backtest found
  no validated edge.)
- **Don't dress up eval numbers.** The bundled dataset is small and easy; a 1.00 baseline
  is a *green baseline*, not proof of quality. Its value is regression detection.
- Ask before adding dependencies beyond: langgraph, langchain-core,
  langchain-google-genai, httpx, feedparser, ddgs, pydantic(-settings), python-dotenv,
  streamlit (added for the UI).
- Update this file whenever a decision changes; it is the project's single source of truth.

## Measured baseline
Bundled 3-question dataset (NVDA valuation, BTC risks/catalysts, AAPL vs ETH):
ticker recall **1.00**, focus coverage **1.00**, data availability **1.00**, source
diversity **7.7**/run, 1 critic pass each; judge scores — report faithfulness **1.00**,
relevance **1.00**, analyst groundedness **1.00**. See the honesty caveat above.

## Possible next steps (none blocking)
- **Harder eval cases** (obscure tickers, ambiguous questions, deliberately missing data) —
  the single most useful improvement, since the current set is too easy to be informative.
- A **non-Gemini judge** for genuine independence.
- Persisted run history to track metrics over time.
- A simple UI (the owner has a Next.js frontend pattern from the RAG project).
- Sector/peer comparison, or SEC-filing analysis (the one RAG-shaped extension).
