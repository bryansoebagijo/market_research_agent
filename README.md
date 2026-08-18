# Markets Research Analyst — a LangGraph agent

A multi-step research agent for **crypto and US stocks**. Ask it a markets question
and it plans the research, gathers live evidence from data + news tools, critiques its
own coverage, and writes a **cited, balanced report** — built as a stateful
[LangGraph](https://langchain-ai.github.io/langgraph/) graph, not a linear script.

> **Not financial advice.** This is a best-effort research assistant. It surfaces data
> and context; it does not predict prices or tell you to buy or sell.

## Why no RAG?
Markets are a **real-time** domain — prices, indicators, and news change constantly, so
a fixed vector store would be stale on arrival. The backbone here is **live-data tools**
(market data, fundamentals, news, web), not retrieval over a static corpus.

## The agent graph
```
question
   │
[Planner]      decompose into focused sub-questions (price / technicals / news / …)
   │
[Researcher]   run the right tool per sub-question → evidence + sources        (next)
   │
[Critic]       coverage & contradictions? loop back if thin                    (next)
   │
[Synthesizer]  cited, balanced report with disclaimers                         (next)
```
*Built incrementally — the Planner is live; Researcher/Critic/Synthesizer are next.*

## Stack
- **LangGraph** — stateful multi-node orchestration
- **Gemini free tier** — the reasoning model
- **Tools:** CoinGecko (crypto), Twelve Data (stocks), news RSS, web search (keyless)

## Setup
```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
copy .env.example .env           # add your GEMINI_API_KEY
```

## Run
```bash
python main.py "Should I be cautious on NVDA and BTC this week?"
```
