# Markets Research Analyst — a LangGraph agent

A multi-step research agent for **crypto and US stocks**. Ask it a markets question and
it plans the research, gathers live evidence (price, computed technicals, news, macro),
**critiques its own coverage and loops back to fill gaps**, then writes a **cited,
balanced report** — built as a stateful [LangGraph](https://langchain-ai.github.io/langgraph/)
graph, not a linear script.

> **Not financial advice.** This is a best-effort research assistant. It surfaces data
> and context; it does not predict prices or tell you to buy or sell.

## Why no RAG?
Markets are a **real-time** domain — prices, indicators, and news change constantly, so
a fixed vector store would be stale on arrival. The backbone here is **live-data tools**,
not retrieval over a static corpus.

## The agent graph
```
question
   │
[Planner]      decompose into focused sub-questions, tagged price / technicals /
   │           fundamentals / news / macro, with the tickers each concerns
   ▼
[Researcher]   route each sub-question to the right tool → evidence + sources
   │           (one batched LLM call distills all the raw data into findings)
   ▼
[Critic]  ─────── coverage sufficient? ───────────────────────────────┐
   │  ▲                                                                │
   │  └── no: propose targeted follow-ups, loop back (up to N times) ──┘
   ▼  yes / loop-cap reached
[Synthesizer]  balanced, cited report + "what's uncertain" + disclaimer
   │
   ▼
[Analyst]      a portfolio-manager-style read: stance, conviction, horizon,
   │           reasoning, invalidation conditions, key risks
   ▼
 report + analyst view
```
The **Critic → loop-back** is a real conditional edge: the agent *decides* whether to
research more, and a loop counter guarantees it terminates. That reflection is what makes
this an agent rather than a pipeline.

### The Analyst view
The synthesizer answers *"what does the data say?"*; the analyst answers *"how would an
experienced investor read it?"* — a stance (Accumulate / Hold / Trim / Avoid / Watch),
conviction, horizon, reasoning, **what would change the view**, and key risks.

Two deliberate constraints keep it honest: it must weigh conflicting signals rather than
cherry-pick, and it must state **invalidation conditions** — a view you can't disprove is
a vibe, not analysis. It gives no position sizes, price targets, or personalised
instructions, and carries its own disclaimer: a language model persona has **no validated
track record**, so this is an interpretation of the evidence, not advice.

### Human-in-the-loop (`--review`)
```
START → planner → human_review ─┬─ approve → researcher → … → report
             ▲                  ├─ revise  → planner (re-plans with your feedback)
             └──────────────────┘
                                └─ cancel  → stop, nothing researched
```
Run with `--review` and the agent **pauses after planning** and shows you the plan. The
gate sits here on purpose: the plan decides what everything downstream spends time and
API calls on, so correcting a wrong angle is cheapest at this point. Implemented with
LangGraph's `interrupt()` — the run suspends, its state is saved by a checkpointer, and
resuming with your decision continues exactly where it left off.

## Stack
- **LangGraph** — stateful multi-node orchestration with a typed shared state
- **Gemini** (`gemini-3.5-flash-lite`) — the reasoning model (free tier; ~3–4 calls/run)
- **Tools** (all keyless or free):
  - **CoinGecko** — crypto price + **computed RSI(14) / EMA13 / EMA21** momentum
  - **Twelve Data** — US-stock price + computed technicals *(free key)*
  - **Alpha Vantage** — stock fundamentals *(optional free key)*
  - **RSS news** — dated headlines from CoinDesk / Cointelegraph / Decrypt (crypto) and
    CNBC / Yahoo Finance / MarketWatch **plus Yahoo's per-ticker feed** (stocks), so news
    findings are real, recent, and company-specific
  - **DuckDuckGo** — macro & supplementary web search *(keyless)*

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env           # add GEMINI_API_KEY (+ optional TWELVE_DATA / ALPHAVANTAGE keys)
```

## Run
```bash
python main.py "Is now a risky time to hold NVDA and BTC?"
```
It prints the research plan, the evidence gathered (with sources), the critic's verdict
and how many research passes it took, then the final cited report.

### Web UI
```bash
streamlit run streamlit_app.py
```
Opens at http://localhost:8501 — type a question and watch the agent work **node by
node** (planning → gathering → critiquing → reporting) instead of waiting on a silent
terminal. Results render as a plan, an expandable evidence list with sources, the critic's
verdict, the report, and the analyst view with its stance/conviction/horizon. Toggle
**"Review the plan before researching"** in the sidebar to get approve / revise / cancel
buttons before any research runs. The UI calls the agent directly — there's no separate
backend.

### CLI
To approve the plan before any research runs:
```bash
python main.py --review "Is now a risky time to hold NVDA and BTC?"
```
You'll be shown the plan and asked for **[a]** approve, **[r]** revise with feedback
(e.g. *"focus only on BTC"* — it re-plans), or **[c]** cancel.

## Example (abridged)
```
RESEARCH PLAN  →  tickers: NVDA, BTC; 5 tagged sub-questions
EVIDENCE       →  [technicals] NVDA RSI 57 neutral, bullish vs EMAs; BTC RSI 72 overbought…
                  (sources: CoinGecko/Twelve Data computed, DuckDuckGo web search)
CRITIC         →  passes: 1  |  coverage OK: true
REPORT         →  balanced risk summary • cited bullet findings • what's uncertain • disclaimer
```

## Evaluation
```bash
python -m eval.evaluate                 # trajectory + LLM-as-judge
python -m eval.evaluate --no-judge      # trajectory only (fewer API calls)
python -m eval.evaluate --limit 1       # single question while iterating
```
Two layers, because they answer different questions:

| Layer | Measures | Examples |
|---|---|---|
| **Trajectory** (deterministic) | Did the agent *behave* correctly? | ticker recall, focus coverage, data availability, source diversity, critic passes |
| **Quality** (LLM-as-judge) | Was the *output* any good? | report faithfulness (hallucination check), relevance, analyst groundedness |

The judge runs on a **different model** from the one that produced the output, so it
isn't grading itself with the same blind spots. Both are Gemini here (that's what's
free), so independence is partial — a different provider would be stronger.

### Measured baseline
Run on the bundled 3-question dataset (NVDA valuation, BTC risks/catalysts, AAPL vs ETH
— the last exercising mixed stock+crypto routing):

| Metric | Score |
|---|---:|
| Ticker recall | **1.00** |
| Focus coverage | **1.00** |
| Data availability | **1.00** |
| Source diversity | **7.7** sources/run |
| Critic passes | **1.0** |
| Report faithfulness *(judge)* | **1.00** |
| Report relevance *(judge)* | **1.00** |
| Analyst groundedness *(judge)* | **1.00** |

**Read this honestly:** three straightforward questions all scoring 1.00 is a *green
baseline*, not proof of excellence — the dataset contains no adversarial cases (obscure
tickers, ambiguous questions, deliberately unavailable data). Its value is as a
**regression check**: change a prompt or a tool, re-run, and see whether anything moves.
Adding harder cases is the obvious way to make these numbers more informative.

## Design notes & honest limitations
- **The reflection loop is bounded** by `MAX_CRITIC_LOOPS` so it always terminates, even
  when some data is structurally unavailable.
- **Free-tier limits**: the Gemini free tier has a modest daily request cap; heavy
  iteration can hit it. Fundamentals need an Alpha Vantage key or they show "unavailable".
- **Not backtested / not advice**: signals like RSI/EMA are informational context, not a
  validated edge or a recommendation.

## Roadmap
All six nodes (Planner, Researcher, Critic with loop-back, Synthesizer, Analyst) plus
computed technicals, RSS news, human-in-the-loop plan review, and the evaluation harness
are built. Natural next steps: a larger eval dataset for a meaningful baseline, a
non-Gemini judge for true independence, and persistent run history to track metrics over
time.
