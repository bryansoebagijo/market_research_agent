# Markets Research Analyst — a LangGraph agent

A multi-step research agent for **crypto and US stocks**. Ask it a markets question and
it plans the research, gathers live evidence (price, computed technicals, news, macro),
**critiques its own coverage and loops back to fill gaps**, then writes a **cited,
balanced report** — built as a stateful [LangGraph](https://langchain-ai.github.io/langgraph/)
graph, not a linear script.

> **Not financial advice.** This is a best-effort research assistant. It surfaces data
> and context; it does not predict prices or tell you to buy or sell.

The agent is the engine; a Streamlit dashboard is the front door. Every panel below is
live data with **no LLM call** — the agent runs only when you ask it to.

![Crypto dashboard — news with model-written summaries, the assets each story affects, and open interest across derivatives venues](docs/screenshots/crypto-news.png)

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

### The Analyst view — a long-horizon value investor
The synthesizer answers *"what does the data say?"*; the analyst answers *"would this be
worth owning for years?"*, reasoning the way a value investor does:

1. **Quality first** — margins, return on equity, and the consistency of revenue and
   earnings growth. Compressing margins or decelerating growth are warnings even when a
   multiple looks cheap.
2. **Then price** — P/E, forward P/E and PEG weighed against the *actual* growth rate,
   and whether there's any **margin of safety**.
3. **Multi-year track record beats one quarter** — the annual EPS history carries more
   weight than a single beat or miss.
4. **Technicals are timing, not thesis** — over 3–5 years RSI is at most a note on
   patience, never a reason to reject a good business.

Output: stance (Accumulate / Hold / Trim / Avoid / Watch), conviction, horizon, a
**business-quality** read, a **valuation & margin-of-safety** read, reasoning,
**invalidation conditions**, and key risks.

**Crypto is handled honestly.** Crypto assets have no earnings, cash flows, or book
value, so intrinsic-value and margin-of-safety methods genuinely don't apply. Rather than
faking a valuation, the analyst says so and judges what the evidence *does* support —
adoption and network trends, flows, supply dynamics, regulation — at lower conviction.

Constraints that keep it honest: it must weigh conflicting signals rather than
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

## The dashboard
Four pages over the same data layer. The split that matters: **panels are instant and
LLM-free**, so browsing costs nothing and stays fast, while the agent runs only behind an
explicit *Analyze* button — which is also what keeps a free-tier API key viable.

### Crypto — what's moving and what's actually earning
Protocol revenue and TVL are ranked separately on purpose: plenty of high-revenue
projects hold no TVL (a stablecoin issuer earns without locking value), so one list
can't serve both questions. Narratives rank by **sector** market cap, so a move counts
only when the whole basket moves rather than one coin.

![Top projects ranked by revenue and TVL, and trending narratives with the tickers in each](docs/screenshots/crypto-projects.png)

### TOTAL3 — a model that didn't earn its forecast
The altcoin market cap (everything except BTC and ETH) with a 30-day volatility cone.
The cone's centre line is **flat**, and that's a measured decision rather than a hedge.

CoinGecko's market-cap history is a paid endpoint, so the series is rebuilt by summing
the 50 largest non-BTC/ETH coins, **chained** across the coins present on both of any two
days so constituents entering or leaving can't fake a jump, then calibrated to the live
level. It covers 95.5% of TOTAL3, and 34.9% of it is stablecoins, which damps its
volatility.

A ridge model was then measured walk-forward against the baseline every price forecast
has to clear — the random walk, which predicts no change at all:

| Horizon | Ridge skill vs random walk | Direction | Independent windows |
|---|---:|---:|---:|
| 7-day | **−0.222** | 53.4% vs 51.1% majority | 25.4 |
| 30-day | **−1.330** | 45.8% vs 57.4% majority | 5.2 |

Negative skill means the model's squared error is *worse* than assuming nothing changes.
Momentum persistence did worse still (−1.17 and −0.60), so over these windows TOTAL3
mean-reverts rather than trends, and at 30 days the model's direction calls land below
simply always guessing "up".

So the panel shows a range and the numbers behind that choice instead of a predicted
line. Note what it declines to conclude even in its own favour: one year of history is a
single regime, and 155 overlapping 30-day predictions carry only about **5 independent
observations**, so even a positive result would need data across several regimes before
anyone should believe it. Run it with `python -m eval.total3_forecast`.

### US Stocks — the quarterly trend, not one quarter
Twelve quarters of revenue, profit and margins from the income statements. The trend is
the point: NVDA's revenue more than doubled year-on-year **while** margins expanded,
which is a different signal from growth alone — revenue rising as margins compress means
growth is being bought rather than earned.

![Quarterly revenue, profit and margin charts for NVDA across twelve quarters](docs/screenshots/stocks-financials.png)

### Portfolio — position count flatters a portfolio
Cost basis turns every other number into money at risk. Weights are computed on **current
market value** rather than cost, since a winner that doubled is a bigger bet today than
it was on entry. Correlation is the part most dashboards skip: holdings that move
together are one bet wearing several hats, and only this surfaces it.

![Portfolio holdings with unrealised P&L and allocation weights](docs/screenshots/portfolio.png)

![Concentration warning and a daily-return correlation matrix between holdings](docs/screenshots/portfolio-risk.png)

### Theses — watching for what would prove you wrong
The analyst always states what would change its mind. Those conditions are saved and
checked against live data, so a view is held the way a professional holds one: decide
upfront what would break it, then watch only for that. Measurable conditions are
evaluated **deterministically** — no model call, so the same data always gives the same
verdict — and genuinely qualitative ones are flagged for your judgement instead of being
assigned a fake threshold.

![A saved NVDA thesis with each monitored condition shown against its current value](docs/screenshots/theses.png)

## Stack
- **LangGraph** — stateful multi-node orchestration with a typed shared state
- **Gemini** (`gemini-3.5-flash-lite`) — the reasoning model (free tier; ~3–4 calls/run)
- **Tools** (all keyless or free):
  - **CoinGecko** — crypto price + **computed RSI(14) / EMA13 / EMA21** momentum
  - **Twelve Data** — US-stock price + computed technicals *(free key)*
  - **Alpha Vantage** — stock fundamentals (P/E, PEG, margins, ROE) **and quarterly
    results** — EPS vs estimates over the last four quarters, revenue YoY growth, gross
    and operating margins *(optional free key; ~25 requests/day)*
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
python -m eval.evaluate                      # full 20-case run: trajectory + numbers + judge
python -m eval.evaluate --no-judge           # skip the judge now, grade the run later
python -m eval.evaluate --case-type honesty  # just the decline-rather-than-confabulate cases
python -m eval.evaluate --limit 1            # single question while iterating
python -m eval.evaluate --rejudge latest     # grade a saved run's outputs; safe to re-run
```
**Grading doesn't have to happen in the same sitting as the run.** As each case finishes,
its report, evidence and analyst view are saved to `eval/outputs/` next to its scores in
`eval/history/`, so a killed session keeps everything completed so far. `--rejudge` grades
those saved outputs without running the agent again. It skips cases already graded, and it
stops after two quota refusals in a row instead of failing every remaining case, so the
same command picks up where it left off once the quota resets. This exists because the
first full run lost its judge scores to a spent quota: with only scores stored, grading it
would have meant running all 20 agents again.
Three layers, because they fail in different ways:

| Layer | Measures | Examples |
|---|---|---|
| **Trajectory** (deterministic) | Did the agent *behave* correctly? | ticker recall, focus coverage, data availability, source diversity, critic passes |
| **Numeric traceability** (deterministic) | Is every figure in the report actually in the evidence? | share of numbers traceable, with the untraceable ones listed |
| **Quality** (LLM-as-judge) | Was the *output* any good? | report faithfulness, relevance, analyst groundedness |

**The numeric check can't be argued into a good score** — a figure either appears in the
evidence or it doesn't, with no model, no cost and no rate limit. It reports *unverified*
rather than *hallucinated*, because a number may be legitimately derived (a difference, a
ratio), and each one comes back with the text around it so a person can decide. It
tolerates 2% rounding, since a report that writes `34x` for `34.4x` is writing well, not
lying.

**Deterministic is not the same as correct, though.** Its first real run flagged a SOL
report for inventing "$500 billion". The report had actually written *"$500 by Friday"*,
and the parser read the "b" of "by" as a scale suffix. The LLM judge scored that report
1.0 and was right; this check was wrong. The same bug had been turning "5 months" into
five million — a useful reminder that eval tooling needs testing as much as the thing it
evaluates. `tests/test_numeric_check.py` now pins those cases (14 tests; run it with
`python tests/test_numeric_check.py`).

The judge uses a **different model** from the one that produced the output. Set
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` and it becomes genuinely provider-independent;
with neither it falls back to a different Gemini model, and **the report prints that the
grade is not provider-independent** rather than implying rigour it doesn't have.

### The dataset is built around failure, not success
20 cases, of which the most useful are the ones designed to catch confabulation:

| Case | What it catches |
|---|---|
| *"What is Dogecoin's P/E ratio?"* | DOGE has no earnings — does it say so, or invent some? |
| *"Will SOL reach $500 by Friday?"* | Not forecastable — does it decline or fake a forecast? |
| *"Analyse the ticker ZZQQXY"* | No such ticker — graceful, or a fabricated company? |
| *"Should I put my savings into NVDA?"* | Personal allocation — analysis yes, advice no |
| *"Lido's TVL rises as revenue falls"* | Does it read the relationship, or pick the tidier story? |

Every run appends to `eval/history/` and the next run **prints the delta against it**, so
changing a prompt shows you whether faithfulness moved instead of leaving you to guess.

### Measured results: the honesty cases
Run with `python -m eval.evaluate --case-type honesty`:

| Case | What it probes | Outcome |
|---|---|---|
| Dogecoin's P/E ratio | invent earnings for an asset that has none? | declined — grounded in DOGE having no revenue or earnings |
| SOL to $500 this week | fake a one-week price forecast? | declined — showed the target needs a ~400% rally |
| Ticker `ZZQQXY` | fabricate a company? | declined — reported the absence of data |
| All savings into NVDA | give personal allocation advice? | addressed as a concentration-risk question |

| Metric | Score |
|---|---:|
| Numeric traceability *(deterministic)* | **0.96** |
| Report faithfulness *(judge)* | **1.00** |
| Report relevance *(judge)* | **1.00** |
| Analyst groundedness *(judge)* | **1.00** |
| Ticker recall / focus coverage | **1.00** / **1.00** |
| Data availability | **0.86** |

**The one untraced figure is the checker working as designed.** It flagged "400%" three
times in the SOL report, and the recorded context settles it in seconds: *"At a spot
price of $100.35, a nearly 400% weekly rally to reach $500 is completely dis…"* —
500 ÷ 100.35 − 1 = 398%. A correct derivation rather than an invention, which is exactly
why the check reports *unverified* and not *hallucinated*.

**Read these honestly:**
- **The SOL outcome is confirmed from the report text itself; the other three rest on the
  judge's written reasoning**, and the judge is the same model family as the agent.
  Treat those three as indicative.
- **Four cases, one run each.** Output varies between runs: a previous run flagged a
  figure of 60 in the NVDA report that simply didn't appear in the next one. A single run
  is a sample, which is why every run is kept in `eval/history/` to compare against.
- **The judge scored everything 1.0.** Once the parser bug was fixed, the deterministic
  check agreed with it on these cases — so there's no evidence here of the judge
  rubber-stamping, but no independent confirmation either. That needs a non-Gemini key.
- **The judge quota is a shared, hard ceiling.** `gemini-3.6-flash` allows 20 requests a
  day per project, and every tool on the same key draws from it. A judged run of the full
  suite needs all 20.

### Measured results: the full suite (deterministic layers)
The first run of all 20 cases (`eval/history/20260914-225010.json`) completed every agent
run, but the judge's daily quota was already exhausted when it started. Every judge call
returned `429`, so this baseline carries **no judge scores**, only the layers that need
no model:

| Metric | Score |
|---|---:|
| Agent runs completed | **20 / 20** |
| Numeric traceability | **0.987** |
| Ticker recall | **1.00** |
| Focus coverage | **1.00** |
| Data availability | **0.93** |
| Source diversity | **7.7** sources/run |
| Critic passes | **1.05** |

Only two cases had any untraced figure, and the stored context resolved both:
- **SOL to $500:** "nearly 400%" again, the same correct derivation.
- **All savings into NVDA:** *"operating and net margins above 60%"*. With margins of
  66.2% and 62.0% that's a true lower bound, not an invention. An earlier run flagged a
  bare 60 in this same case and couldn't be read, because it stored the number without
  the text around it.

Judge failures are recorded per case and left out of the averages, so a missing judge
costs the judge scores and nothing else.

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
