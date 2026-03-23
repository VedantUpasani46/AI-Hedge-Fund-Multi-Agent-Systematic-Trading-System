# AI Hedge Fund — Multi-Agent Systematic Trading System

**Vedant Upasani** | Quantitative Developer & Financial Engineer  
📧 upasani99@protonmail.ch |vedant.upasani46@outlook.com| 🔗 [LinkedIn](https://linkedin.com/in/VedantUpasani) | 🐙 [GitHub](https://github.com/VedantUpasani46)

---

> A production-grade, end-to-end AI-powered hedge fund system built over 2024–2026.
> Ten self-contained parts: multi-agent LLM decision-making, event-driven backtesting
> with zero look-ahead bias by architecture, live risk monitoring with circuit breakers,
> an Interactive Brokers execution engine with Almgren-Chriss optimal trajectory, a FastAPI
> investor dashboard with real-time WebSocket streaming, cloud deployment on AWS ECS,
> daily NAV accounting, and regulatory compliance infrastructure.
>
> **~35,000 lines of production Python across 10 parts.**

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AI HEDGE FUND SYSTEM                                 │
│                                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │  PM Agent    │   │ Risk Agent   │   │ Research     │   │ Execution   │ │
│  │  (Part 1)    │◄──│  (Part 2)    │   │ Agent        │   │ Agent       │ │
│  │  LLM-driven  │   │  Pre-trade   │   │  (Part 3)    │   │  (Part 4)   │ │
│  │  allocation  │   │  VaR checks  │   │  RAG / SEC   │   │  IB + AC    │ │
│  └──────┬───────┘   └──────────────┘   └──────────────┘   └─────────────┘ │
│         │                                                                   │
│         ▼  Consensus Protocol (Part 2)                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌─────────────┐ │
│  │ Alt Assets   │   │ Backtest     │   │ Live Risk    │   │ Dashboard   │ │
│  │  (Part 5)    │   │  (Part 6)    │   │  (Part 7)    │   │  (Part 8)   │ │
│  │ Cat bonds    │   │ Walk-forward │   │ Circuit brkr │   │ FastAPI     │ │
│  │ ILS / alt    │   │ Attribution  │   │ Factor mon.  │   │ PDF reports │ │
│  └──────────────┘   └──────────────┘   └──────────────┘   └─────────────┘ │
│                                                                             │
│  ┌──────────────┐   ┌──────────────────────────────────────────────────┐   │
│  │  Cloud Prod  │   │  Fund Operations                                 │   │
│  │  (Part 9)    │   │  (Part 10)                                       │   │
│  │  Docker/ECS  │   │  NAV engine · HWM fees · Compliance · LP portal  │   │
│  │  CI/CD/CW    │   │                                                  │   │
│  └──────────────┘   └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Part 1 — Foundation](#part-1--foundation)
2. [Part 2 — Multi-Agent System](#part-2--multi-agent-system)
3. [Part 3 — RAG & Data Intelligence](#part-3--rag--data-intelligence)
4. [Part 4 — Execution Engine](#part-4--execution-engine)
5. [Part 5 — Alternative Assets & Data](#part-5--alternative-assets--data)
6. [Part 6 — Backtesting Engine](#part-6--backtesting-engine)
7. [Part 7 — Real-Time Risk Management](#part-7--real-time-risk-management)
8. [Part 8 — Investor Dashboard](#part-8--investor-dashboard)
9. [Part 9 — Cloud Production](#part-9--cloud-production)
10. [Part 10 — Fund Operations & Compliance](#part-10--fund-operations--compliance)
11. [Installation](#installation)
12. [Quick Start](#quick-start)
13. [Architecture Decisions](#architecture-decisions)
14. [Related Repositories](#related-repositories)

---

## Part 1 — Foundation

**~3,500 lines | `part1/`**

The base layer: data models, market data fetching, LLM client, and the Portfolio Manager Agent.

### What it does

- **`data_models.py`** — `Portfolio`, `Position`, `Order`, `Fill`, `Decision` dataclasses. The canonical data schema shared by all 10 parts.
- **`market_data.py`** — Yahoo Finance fetcher with 90+ engineered features: momentum at 1d/5d/21d/63d/126d/252d, RSI, MACD, Bollinger %B, volume ratio, 52-week high/low distance. Caches to SQLite.
- **`llm_client.py`** — Unified client for Anthropic Claude and OpenAI GPT-4. Tracks token usage and cost per call. Supports tool-use / function calling.
- **`portfolio_manager_agent.py`** — The PM Agent. Makes allocation decisions using real OHLCV data, Kelly sizing, and LLM reasoning. Every decision is persisted to SQLite with full audit trail.

### Running

```bash
pip install numpy pandas yfinance anthropic openai python-dotenv

# Set LLM key
export ANTHROPIC_API_KEY=sk-ant-...

# Single ticker decision
python run_part1.py --ticker AAPL

# Universe scan
python run_part1.py --scan AAPL MSFT NVDA GOOGL JPM

# Demo (no LLM key needed)
python run_part1.py --demo
```

---

## Part 2 — Multi-Agent System

**~8,400 lines | `part2/`**

Three independent agents reach consensus before any trade is approved. No single agent can approve a trade alone.

### Agents

| Agent | Role | Decision |
|-------|------|----------|
| **Portfolio Manager** | Fundamental + technical analysis | BUY / PASS / SHORT |
| **Research Analyst** | Deep security analysis, growth drivers | BUY / PASS / SHORT |
| **Risk Manager** | Pre-trade VaR, stress tests, concentration | APPROVE / REJECT / REDUCE |

### Consensus Protocol

```
Unanimous BUY (3/3)          → Full Kelly weight
2/3 BUY, high confidence      → 75% Kelly
2/3 BUY, normal confidence    → 50% Kelly
1/3 BUY                       → 5% max (exploratory)
Risk Manager REJECT            → PASS, regardless of other votes
```

### MessageBus (`message_bus.py`)

SQLite-backed publish/subscribe. Priority queuing: `CRITICAL` alerts fire before `HIGH` before `NORMAL`. Full message threading. Drop-in upgrade to Redis (Part 9).

### `BaseAgent`

Abstract base with the full tool-use loop: `think()` → `use_tool()` → `observe()` → repeat until `final_answer`. Every LLM call has a cost budget; the agent halts if it exceeds it.

### Running

```bash
# Risk checks only (no LLM needed)
python run_part2.py --risk-only --ticker AAPL

# Full consensus decision
python run_part2.py --ticker AAPL

# Universe scan, top 5 opportunities
python run_part2.py --scan --top 5
```

---

## Part 3 — RAG & Data Intelligence

**~3,900 lines | `part3/`**

A knowledge base built from live public sources, queried by the Research Analyst Agent to ground every decision in real documents.

### Document Sources (all free, no API key)

| Source | What is fetched |
|--------|----------------|
| SEC EDGAR | 10-K, 10-Q, 8-K filings, earnings call transcripts |
| Federal Reserve | FOMC meeting minutes |
| Yahoo Finance | Financial news, press releases |

### Pipeline

```
SEC EDGAR → Raw text → Sentence-aware chunking (400 words, 80-word overlap)
                     → sentence-transformers (all-MiniLM-L6-v2, 384 dims)
                     → ChromaDB vector store
                     → Metadata filters (ticker, doc type, date)
                     → LLM synthesis from retrieved chunks
```

Falls back to SQLite keyword search when ChromaDB is unavailable.

### Running

```bash
pip install sentence-transformers chromadb yfinance

# Ingest filings for a ticker
python run_part3.py --ingest AAPL

# Query the knowledge base
python run_part3.py --query "Apple AI revenue guidance" --ticker AAPL --no-llm

# Full demo
python run_part3.py --demo AAPL
```

---

## Part 4 — Execution Engine

**~3,400 lines | `part4/`**

Optimal execution via Almgren-Chriss, wired to Interactive Brokers. Full Transaction Cost Analysis on every fill.

### Almgren-Chriss Implementation

The closed-form **sinh-based** trajectory — not a numerical approximation:

```
x*(t) = x₀ · sinh(κ(T−t)) / sinh(κT)

where κ = sqrt(λσ²/η)
      λ = risk aversion parameter
      σ = daily volatility (from Yahoo Finance, recomputed at decision time)
      η = temporary impact coefficient (from 21-day ADV)
```

Parameters auto-calibrate from live 21-day ADV and realised volatility fetched at decision time.

### IB Broker Interface

- Connects to Interactive Brokers TWS/Gateway via `ib_insync`
- **Paper trading**: port 7497 (default — safe)
- **Live trading**: port 7496 (requires explicit `require_paper=False`)
- Falls back to Yahoo Finance simulation when IB is not running
- Full order lifecycle management: PENDING → SUBMITTED → PARTIALLY_FILLED → FILLED

### Transaction Cost Analysis

Every fill is benchmarked against:
- **Arrival price** (decision-time mid)
- **TWAP** over the execution window
- **VWAP** weighted by volume
- **Close price**

Implementation Shortfall computed in basis points and persisted to `execution.db`.

### Running

```bash
pip install ib_insync yfinance

# Test Almgren-Chriss optimiser only (no broker needed)
python run_part4.py --ac-test AAPL 10000 BUY

# Full pipeline with simulation
python run_part4.py --execute AAPL BUY 5

# Compare execution algorithms
python run_part4.py --compare NVDA 5000 BUY
```

---

## Part 5 — Alternative Assets & Data

**~3,200 lines | `part5/`**

Two capabilities: catastrophe bond pricing for uncorrelated return diversification, and alternative data signals for equity enrichment.

### Cat Bond / ILS Pricing

Cat bonds are the capital-markets-adjacent end of the actuarial world. This part prices them using the standard industry methodology.

**Loss Model:**
```
Annual loss = Σᵢ₌₁ᴺ Severityᵢ

N ~ Poisson(λ)           — event frequency
Severity ~ GPD(ξ, β)     — directly reuses the EVT/GPD module from quant-portfolio
```

**Six calibrated peril models** (from Swiss Re Sigma reports):
Florida hurricane, Gulf hurricane, Northeast hurricane, California earthquake, European windstorm, Japan earthquake.

**Pricing output:** Attachment probability, exhaustion probability, expected loss in basis points, fair spread using Lane's empirical risk-multiple table (1.5×–7× EL), cat-adjusted duration, OEP curve (return periods: 10/25/50/100/200yr).

**ILS Portfolio:** Gaussian copula for correlated multi-peril loss simulation. Intra-ILS correlations empirically calibrated (Florida hurricane vs. California earthquake ≈ 0.04). Quantified diversification benefit: adding 10% ILS to a 60/40 portfolio improves Sharpe by ~0.08–0.12 with ~8–12% vol reduction. Source: Cummins & Weiss (2009), Swiss Re ILS data 2002–2023.

### Alternative Data Signals

Four live signal generators from free sources, each returning a −1 to +1 signal:

| Signal | Source | Academic Basis | Confidence |
|--------|--------|----------------|-----------|
| **Options Flow** | Yahoo Finance options chain | Pan & Poteshman (2006) | PCR, IV skew |
| **Insider Transactions** | OpenInsider.com (SEC Form 4) | Seyhun (1998) | Purchase clustering |
| **Short Interest** | Yahoo Finance | Desai et al. (2002) | % float, squeeze detection |
| **Analyst Revisions** | Yahoo Finance | Stickel (1995) | Consensus, target price |

### Running

```bash
pip install numpy pandas scipy yfinance requests

# Price a cat bond
python run_part5.py --price-bond PELICAN-2024-A

# Screen all cat bonds
python run_part5.py --screen-ils

# ILS portfolio diversification analysis
python run_part5.py --diversification

# Alt data signals for a ticker
python run_part5.py --alt-data AAPL

# Rank universe by alt data
python run_part5.py --rank-universe AAPL MSFT NVDA GOOGL JPM
```

---

## Part 6 — Backtesting Engine

**~4,000 lines | `part6/`**

Event-driven backtesting with zero look-ahead bias by architecture, walk-forward validation, performance attribution, and stress testing.

### Why Event-Driven

Vector-based backtests (pandas operations on full dataframes) are chronically infected with look-ahead bias. The architecture here prevents it: at time *t*, the strategy receives a `DataView` containing **only** data available up to *t*. It cannot access the loader or any future data. This is enforced at the type level, not by convention.

### Walk-Forward Validation

The only honest method for fitted strategies. Anchored and rolling flavours:

```
Anchored (default):
  Train on [t₀, t₁]  → test on [t₁, t₂]
  Train on [t₀, t₂]  → test on [t₂, t₃]
  ...

Rolling (for ML models — old data is noise):
  Train on [t₀, t₀+W]  → test on [t₀+W, t₀+W+S]
  Train on [t₁, t₁+W]  → test on [t₁+W, t₁+W+S]
  ...

Report ONLY the stitched OOS result.
IS Sharpe is for internal calibration only.
IS/OOS ratio target: > 0.7 (Lopez de Prado 2018)
```

Computes **Deflated Sharpe Ratio** (Bailey & Lopez de Prado 2014) correcting for multiple comparisons across parameter combinations.

### Transaction Cost Model

Fills at next-day open plus:
- IB-style commission: $0.005/share, min $1
- Half bid-ask spread: 3–10 bps by ADV rank
- Almgren-Chriss square-root market impact
- Random slippage noise: ±0–5 bps

### Performance Attribution

Three layers:
1. **Brinson-Hood-Beebower** — Allocation effect + Selection effect + Interaction
2. **Carhart 4-factor OLS** — α with t-stat and p-value; β_mkt, β_smb, β_hml, β_mom
3. **Transaction Cost Attribution** — Gross vs. net return; drag by commission/spread/impact/slippage

### Stress Testing

**Historical scenarios:** 2008 GFC acute (−44% S&P in 3 months), 2020 COVID crash (−34% in 33 days), 2022 rate shock (full year), 2018 Q4 selloff, 2011 Eurozone crisis.

**Monte Carlo:** Student-t (ν≈4 fat tails) + GARCH-like volatility clustering → 10,000 forward paths → 1st/5th percentile returns, 99% VaR, 99% CVaR, P(loss > 20%), P(loss > 40%).

### Running

```bash
pip install numpy pandas scipy yfinance pyarrow

# Run a backtest
python run_part6.py --backtest momentum AAPL MSFT NVDA GOOGL JPM BAC XOM

# Walk-forward validation
python run_part6.py --walk-forward momentum AAPL MSFT NVDA GOOGL JPM

# Stress test
python run_part6.py --stress-test momentum AAPL MSFT NVDA GOOGL JPM

# Performance attribution
python run_part6.py --attribution momentum AAPL MSFT NVDA GOOGL JPM

# Compare strategies
python run_part6.py --compare momentum mean_reversion AAPL MSFT NVDA GOOGL JPM

# Full demo
python run_part6.py --demo
```

---

## Part 7 — Real-Time Risk Management

**~2,400 lines | `part7/`**

Continuous portfolio risk monitoring during market hours. Distinct from Part 2's pre-trade checks — this runs on a background thread, updating every 30 seconds.

### Live Risk Metrics

Every poll cycle computes:
- NAV, daily P&L, intraday P&L
- Parametric VaR (95% and 99%) — correlation-adjusted using empirical avg correlation 0.35
- Liquidity-adjusted VaR (adds exit cost to standard VaR)
- Portfolio beta to SPY (from 3-month rolling regression)
- Intraday drawdown from today's peak NAV
- Trailing drawdown from 52-week high
- Concentration (HHI, top position weight)
- Margin utilisation

### Circuit Breakers

Seven hard and soft limits:

| Breaker | Type | Threshold | Action |
|---------|------|-----------|--------|
| `DAILY_LOSS_LIMIT` | HARD | −2% NAV | Halt all trading |
| `VAR_LIMIT_BREACH` | HARD | VaR 95% > 2% NAV | Halt all trading |
| `INTRADAY_DRAWDOWN` | HARD | −1.5% from peak | Halt all trading |
| `MARGIN_UTILISATION` | HARD | > 80% | Halt all trading |
| `BETA_SPIKE_HIGH` | SOFT | β > 1.5 | Alert PM Agent |
| `BETA_SPIKE_LOW` | SOFT | β < −0.5 | Alert PM Agent |
| `CONCENTRATION_BREACH` | SOFT | Single position > 20% | Alert PM Agent |

When a HARD breaker fires: trading halts, alert callbacks fire synchronously, and a `CIRCUIT_BREAKER_TRIGGERED` message is broadcast to all agents via MessageBus.

### Factor Monitor

Real-time factor exposure tracking using six ETF proxies (all free via Yahoo Finance):

| Factor | Proxy | Interpretation |
|--------|-------|---------------|
| MKT | SPY | Market exposure |
| SMB | IWM − SPY | Size tilt |
| HML | IWD − IWF | Value tilt |
| MOM | MTUM − SPY | Momentum loading |
| QMJ | QUAL − SPY | Quality tilt |
| BAB | USMV − SPY | Low-vol tilt |

Detects regime changes when MKT beta drifts > 0.3 from its 5-day average.

### Running

```bash
pip install numpy pandas scipy yfinance

# One-shot risk snapshot
python run_part7.py --snapshot

# Factor exposure report
python run_part7.py --factors

# Position reduction plan (simulated VaR breach)
python run_part7.py --reduction-plan

# Live monitor (Ctrl+C to stop)
python run_part7.py --monitor

# Full dashboard
python run_part7.py --dashboard
```

---

## Part 8 — Investor Dashboard

**~2,300 lines | `part8/`**

FastAPI REST + WebSocket server, automated PDF report generation, and LLM-powered investor commentary.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check + component status |
| GET | `/portfolio` | Current NAV, positions, weights |
| GET | `/portfolio/history?days=30` | NAV time series |
| GET | `/portfolio/sector-allocation` | Sector weights |
| GET | `/risk` | Live risk snapshot |
| GET | `/risk/factors` | Current factor exposures |
| GET | `/risk/history?hours=8` | Intraday risk time series |
| GET | `/risk/circuit-breakers` | Circuit breaker status |
| POST | `/risk/reset-circuit-breakers` | Manual reset |
| GET | `/performance` | Sharpe, return, vol, drawdown |
| GET | `/performance/monthly` | Monthly return table |
| GET | `/performance/benchmark?benchmark=SPY` | vs benchmark |
| GET | `/trades?limit=50` | Execution history |
| GET | `/trades/{order_id}` | Single trade detail |
| POST | `/reports/generate?report_type=monthly` | Trigger PDF |
| GET | `/reports/list` | Available reports |
| GET | `/reports/{report_id}` | Download PDF |
| WS | `/ws/risk` | Live risk stream (every 30s) |
| WS | `/ws/portfolio` | Portfolio updates (every 60s) |

**Authentication:** `X-API-Key` header. Dev mode bypasses auth. Production keys in `.env` as `API_KEYS=key1,key2,key3`.

**Interactive docs** at `http://localhost:8000/docs` (Swagger UI — useful for investor demos).

### PDF Reports

**Monthly Investor Letter:**
- Cover page (NAV, period, key stats)
- Performance summary table (colour-coded positive/negative)
- NAV line chart + drawdown chart (matplotlib → embedded PNG)
- Holdings table with sector column
- Sector allocation pie chart
- Risk metrics section
- Disclaimer

**Daily Risk Report:**
- Risk status banner (colour-coded GREEN/YELLOW/RED)
- Portfolio snapshot and VaR vs limits
- Circuit breaker status
- Factor exposure table

### Dashboard Agent

LLM layer on top of the data API:

- `generate_monthly_commentary()` — writes the investor letter commentary in professional fund-manager prose
- `answer_investor_query(question)` — answers natural language LP questions with data-backed responses
- `generate_eod_summary()` — 4–6 bullet daily summary
- `explain_circuit_breaker(name, metrics)` — calm, factual breach explanation for LP email

### Running

```bash
pip install fastapi uvicorn pydantic reportlab matplotlib yfinance

# Start API server
python run_part8.py --server

# Generate monthly PDF
python run_part8.py --report monthly

# Generate daily risk report
python run_part8.py --report daily_risk

# Test all endpoints
python run_part8.py --test-api

# Answer investor question (requires LLM key)
python run_part8.py --query "How is the portfolio performing?"

# Full demo
python run_part8.py --demo
```

---

## Part 9 — Cloud Production

**~2,900 lines | `part9/`**

Docker containerisation, AWS ECS deployment, CI/CD pipeline, and production observability.

### Container Architecture

**Multi-stage Dockerfile:**
- `builder` stage: installs gcc, build tools, all Python deps
- `production` stage: copies only installed packages — ~400MB vs ~1.2GB naive single-stage
- Runs as non-root user (`hedgefund:hedgefund`, uid=1000)
- Built-in `HEALTHCHECK` polls `/health` every 30 seconds

**Docker Compose — 5 services:**

| Service | Description | Port |
|---------|-------------|------|
| `api` | FastAPI server | 8000 |
| `monitor` | LiveRiskEngine background process | — |
| `strategy` | AgentCoordinator scan loop | — |
| `redis` | MessageBus backend (upgrade from SQLite) | 6379 |
| `nginx` | Reverse proxy + TLS termination | 80/443 |

Internal services sit on an `internal: true` Docker network — no external access.

### AWS Deployment

**Architecture (optimised for sub-$100M AUM):**
- ECR: container registry
- EC2 t3.medium: single instance, all containers (~$30/month)
- EBS gp3 50GB: persistent SQLite volumes (~$4/month)
- CloudWatch: logs + metrics + alarms (~$5–15/month)
- Total: **~$40–60/month**

**Zero-downtime deployment (ECS rolling update):**
```
Build → Tag with git SHA → Push to ECR
→ Register new task definition revision
→ ECS rolling replacement (new up before old stops)
→ Health check passes → old containers terminated
```

**CloudWatch alarms created automatically:**

| Alarm | Metric | Threshold |
|-------|--------|-----------|
| `HF-DailyLossBreached` | `daily_loss_pct` | > 2% |
| `HF-VaRBreached` | `var_95_pct` | > 2% |
| `HF-DrawdownWarning` | `intraday_drawdown` | > 1% |
| `HF-APIHighLatency` | `api_latency_p99` | > 2000ms |
| `HF-APIErrorRate` | `api_error_rate` | > 5% |
| `HF-LLMCostSpike` | `llm_cost_hourly` | > $5/hr |

### CI/CD Pipeline (GitHub Actions)

6-stage pipeline on push to `main`:

```
Stage 1: pytest (no LLM calls, no broker)
Stage 2: flake8 + black + isort
Stage 3: Docker build + smoke test
Stage 4: Push to ECR (tagged with git SHA)
Stage 5: ECS rolling deploy (manual approval gate via GitHub environment)
Stage 6: Health check live API + Slack notification
```

### Production Monitoring

**Structured logging** — every log line is JSON, queryable in CloudWatch Insights:
```json
{"timestamp":"2024-01-15T14:23:11","level":"INFO","service":"risk_monitor",
 "event":"risk_snapshot","metric":"var_95_pct","value":0.0142}
```

**LLM cost tracking** — every API call logged with agent, model, purpose, token count, cost. Alerts if hourly spend > configurable threshold.

**Health check hierarchy:**
- `/health` — is process alive?
- `/health/deep` — database, Redis, market data, disk space
- `/health/risk` — risk monitor running and snapshot < 120s old

### Running

```bash
# Docker (local)
python run_part9.py --docker-build
python run_part9.py --docker-run
python run_part9.py --docker-compose-up

# Health checks
python run_part9.py --health

# LLM cost summary
python run_part9.py --llm-costs

# AWS cost estimate
python run_part9.py --aws-costs

# Deploy to AWS (requires credentials)
export AWS_ACCOUNT_ID=123456789012
python run_part9.py --deploy

# Full demo
python run_part9.py --demo
```

---

## Part 10 — Fund Operations & Compliance

**~3,200 lines | `part10/`**

Daily fund accounting, fee calculation, LP account management, and regulatory compliance infrastructure.

### NAV Engine

Follows the exact institutional NAV calculation sequence:

```
1. Gross Asset Value = mark all positions at closing prices + cash
2. Management fee accrual = (1/252) × annual_rate × prior_NAV
3. Performance fee accrual = perf_rate × daily_gain_above_HWM (per LP)
4. Other expenses = (1/252) × (admin + legal + prime) × prior_NAV
5. Net Asset Value = GAV − all accrued liabilities
6. NAV per share = NAV / shares_outstanding
7. Update each LP's capital account and high-water mark
```

**High-Water Mark (HWM) mechanics:**
- HWM ratchets upward at every new peak — never downward
- Performance fee accrues only on gains above the LP's individual HWM
- Different HWMs per LP (they may have invested at different times)

**Four fee structures:**

| Structure | Management Fee | Performance Fee | For |
|-----------|---------------|----------------|-----|
| `FOUNDERS` | 1.0% | 15% | Early investors |
| `STANDARD` | 2.0% | 20% | Default |
| `INSTITUTIONAL` | 1.5% | 10% | Large allocators |
| `NO_PERFORMANCE` | 1.0% | 0% | Pension mandates |

**Subscriptions and redemptions:** Capital converts to fund shares at current NAV/share. Redemptions convert shares back to cash at current NAV/share, optionally with an early redemption fee if within lock-up.

### Compliance Engine

**Pre-trade checks** (before any order):
- Single-name concentration limit (default 15% of NAV)
- Approaching-limit warning at 85% of limit
- 13D/13G ownership threshold flag (>5% of outstanding shares)
- Gross exposure limit (default 150% of NAV)
- Prohibited instruments list

**Daily post-trade surveillance:**
- All position concentration
- Sector concentration (default 35% max in any GICS sector)
- Leverage / gross exposure
- Wash sale detection (buy within 30 days of sell)
- Best execution review (flags IS > 20bps)

**Personal account trading pre-clearance:**
- Supervised persons pre-clear via `preclear_personal_trade()`
- Checks restricted list and fund's current holdings
- Logged to `compliance.db` with timestamp

**Regulatory report generators (templates — require legal review):**
- `generate_form_pf_summary()` — Section 1A data for SEC filing (required for RIAs > $150M AUM)
- `generate_13f_holdings()` — Schedule 13F holdings table (required if > $100M equity)
- `generate_13d_check()` — 5% ownership threshold alert with 10-day filing deadline
- `generate_best_execution_report()` — Quarterly avg IS by broker and ticker

> ⚠️ These produce data summaries only. Actual regulatory filings require legal counsel and submission through the appropriate SEC systems (PFRD, EDGAR).

### Fund Operations Agent

LLM layer that automates the daily ops workflow:

```
4:30pm ET daily workflow:
  1. run_daily_nav        → official NAV with fee accruals
  2. run_compliance_check → post-trade surveillance
  3. get_compliance_alerts → review open items
  4. get_nav_summary      → final fund summary
```

### Running

```bash
pip install numpy pandas yfinance

# Daily NAV calculation
python run_part10.py --daily-nav

# Compliance surveillance
python run_part10.py --compliance

# Investor statements (all LPs)
python run_part10.py --investor-statements

# Regulatory data
python run_part10.py --regulatory form_pf
python run_part10.py --regulatory 13f
python run_part10.py --regulatory best_execution

# Run ops agent EOD workflow (requires LLM key)
python run_part10.py --agent eod

# Full demo
python run_part10.py --demo

# Complete system summary
python run_part10.py --system-summary
```

---

## Installation

### Prerequisites

```bash
Python 3.11+
```

### Core dependencies (all parts)

```bash
pip install numpy pandas scipy yfinance anthropic openai python-dotenv
```

### Part-specific dependencies

```bash
# Part 3 (RAG)
pip install sentence-transformers chromadb sec-edgar-downloader

# Part 4 (Execution)
pip install ib_insync

# Part 5 (Alt Assets)
pip install requests

# Part 6 (Backtesting)
pip install pyarrow

# Part 7 (Risk — no extras beyond core)

# Part 8 (Dashboard)
pip install fastapi uvicorn pydantic websockets reportlab matplotlib

# Part 9 (Cloud)
pip install boto3 redis docker

# Part 10 (Fund Ops — no extras beyond core)
```

### Environment setup

```bash
cp .env.example .env
# Edit .env with your keys:
```

```dotenv
# LLM (required for agent mode, optional for data-only mode)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...           # optional fallback

# API security (Part 8)
API_KEYS=prod-key-1,investor-key-2,admin-key-3
ENV=development

# IB Broker (Part 4, optional)
IB_HOST=127.0.0.1
IB_PORT=7497                    # 7497=paper, 7496=live

# AWS (Part 9, optional)
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012
ECR_REPO=hedgefund-api
ECS_CLUSTER=hedgefund-cluster
ECS_SERVICE=hedgefund-api

# Risk limits (Part 7)
MAX_DAILY_LOSS_PCT=0.02
MAX_VAR_PCT=0.02
MAX_INTRADAY_DD=0.015

# Fund details (Part 10)
FUND_NAME=AI Systematic Fund LP
INITIAL_CAPITAL=1000000

# Alerting (optional)
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

---

## Quick Start

### No API key — data and risk only

```bash
git clone https://github.com/VedantUpasani46/AI_HEDGE_FUND.git
cd AI_HEDGE_FUND
pip install numpy pandas scipy yfinance reportlab matplotlib fastapi uvicorn

# Backtest the momentum strategy
python part6/run_part6.py --backtest momentum AAPL MSFT NVDA GOOGL JPM BAC XOM

# Risk snapshot
python part7/run_part7.py --snapshot

# Generate monthly PDF report
python part8/run_part8.py --report monthly

# Daily NAV demo
python part10/run_part10.py --daily-nav

# Complete system summary
python part10/run_part10.py --system-summary
```

### With API key — full agent mode

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Single allocation decision
python part1/run_part1.py --ticker AAPL

# Multi-agent consensus scan
python part2/run_part2.py --scan --top 5

# EOD fund operations
python part10/run_part10.py --agent eod

# Investor question
python part8/run_part8.py --query "What is our Sharpe ratio and biggest current risk?"
```

### With Docker — full production stack

```bash
cp .env.example .env
# Fill in your keys

docker-compose up -d                       # Start all services
docker-compose logs -f api                 # Stream API logs
curl -H "X-API-Key: dev-key-1" http://localhost:8000/portfolio
```

---

## Architecture Decisions

**Why event-driven backtesting?**
Vector-based backtests (pandas apply on full DataFrames) leak future data constantly. The `DataView` object makes it impossible: the strategy can only see data the engine explicitly passes to it at time *t*.

**Why consensus before every trade?**
A single agent making allocation decisions has no check on overconfidence or hallucination. Three agents disagreeing triggers investigation. The risk agent's veto is unconditional — it can block any trade regardless of the other two votes.

**Why SQLite rather than PostgreSQL?**
At sub-$50M AUM, a single-process system is sufficient. SQLite in WAL mode handles concurrent reads with one writer trivially. No database server to maintain. Upgrade path to PostgreSQL is one connection string change.

**Why Almgren-Chriss closed-form rather than numerical?**
The sinh-based closed form is exact, fast (microseconds), and produces the globally optimal solution. Numerical methods introduce approximation error and are slower. The closed form is also more auditable — every intermediate variable has a closed mathematical interpretation.

**Why free data sources?**
Institutional data vendors (Bloomberg, Refinitiv) cost $25K–$100K/year. The system demonstrates that real, meaningful quant work is possible with Yahoo Finance + SEC EDGAR + Federal Reserve public data. This matters at the seed stage. Data vendors come later.

**Why separate circuit breakers from pre-trade risk checks?**
Pre-trade checks (Part 2) answer: "should we execute this trade?" Circuit breakers (Part 7) answer: "is the portfolio in a state where it is safe to trade at all?" They run on different clocks (per-decision vs. every 30 seconds) and require different responses (reject order vs. halt system).

**Why is the NAV calculation per-LP?**
Different LPs invest at different times, at different NAV/share levels. A performance fee calculated at the fund level would incorrectly charge late investors for gains they didn't participate in. Each LP has their own HWM, and performance fees accrue only on gains above their individual HWM.

---

## Part-by-Part Summary

| Part | Module | Lines | Key Technology | No-LLM Mode |
|------|--------|-------|---------------|-------------|
| 1 | Foundation | ~3,500 | Yahoo Finance, SQLite, LLM client | ✓ |
| 2 | Multi-Agent System | ~8,400 | Consensus protocol, MessageBus | ✓ (risk-only) |
| 3 | RAG & Knowledge Base | ~3,900 | ChromaDB, sentence-transformers | ✓ |
| 4 | Execution Engine | ~3,400 | IB TWS, Almgren-Chriss | ✓ (simulation) |
| 5 | Alternative Assets | ~3,200 | GPD severity, ILS Gaussian copula | ✓ |
| 6 | Backtesting | ~4,000 | Walk-forward, Carhart 4-factor OLS | ✓ |
| 7 | Real-Time Risk | ~2,400 | Threading, circuit breakers | ✓ |
| 8 | Investor Dashboard | ~2,300 | FastAPI, reportlab, WebSocket | ✓ |
| 9 | Cloud Production | ~2,900 | Docker, AWS ECS, GitHub Actions | ✓ |
| 10 | Fund Operations | ~3,200 | NAV engine, HWM, compliance | ✓ |
| **Total** | | **~35,000** | | |

---

## Related Repositories

| Repository | Description | Lines |
|-----------|-------------|-------|
| [quant-portfolio](https://github.com/VedantUpasani46/quant-portfolio) | Core quant library: Heston (Little Trap), SABR, DCC-GARCH, EVT/GPD, Almgren-Chriss, Garleanu-Pedersen, Kalman pairs trading | ~20,675 |
| [ML-QUANTITATIVE-PORTFOLIO](https://github.com/VedantUpasani46/ML-QUANTITATIVE-PORTFOLIO) | ML alpha research: XGBoost ensemble, VAE anomaly detection, DQN/PPO portfolio RL, FinBERT sentiment, Avellaneda-Stoikov market making | ~17,500 |
| [alpha-research-library](https://github.com/VedantUpasani46) | 30 alpha signals + LLM discovery factory: BAB, QMJ, TSMOM, PEAD, GEX pinning, eigenportfolio stat arb, VPIN-filtered momentum | ~25,000 |
| **This repository** | **AI Hedge Fund — full 10-part production system** | **~35,000** |

**Combined codebase: ~115,000 lines of production Python across 4 repositories.**

---

## Academic References

Implementations used throughout the system:

- **Almgren & Chriss (2001)** — *Optimal Execution of Portfolio Transactions* → Part 4 (sinh closed-form)
- **Engle (2002)** — *Dynamic Conditional Correlation* → risk infrastructure
- **Albrecher et al. (2007)** — *The Little Heston Trap* → quant-portfolio pricing
- **Avellaneda & Lee (2008)** — *Statistical Arbitrage in the U.S. Equities Market* → quant-portfolio
- **Pan & Poteshman (2006)** — *The Information in Option Volume for Future Stock Prices* → Part 5 options flow
- **Seyhun (1998)** — *Investment Intelligence from Insider Trading* → Part 5 insider signal
- **Desai et al. (2002)** — *Short-Sellers, Fundamental Analysis and Stock Returns* → Part 5 short interest
- **Stickel (1995)** — *The Anatomy of the Performance of Buy and Sell Recommendations* → Part 5 analyst signal
- **Cummins & Weiss (2009)** — *Convergence of Insurance and Financial Markets* → Part 5 ILS calibration
- **Lane (2006)** — *Pricing Risk Transfer Transactions* → Part 5 cat bond risk-multiple table
- **Garleanu & Pedersen (2013)** — *Dynamic Trading with Predictable Returns and TC* → Part 2 position sizing
- **Bailey & Lopez de Prado (2014)** — *The Deflated Sharpe Ratio* → Part 6 walk-forward
- **Lopez de Prado (2018)** — *Advances in Financial Machine Learning* → Part 6 validation protocol
- **Brinson, Hood & Beebower (1986)** — *Determinants of Portfolio Performance* → Part 6 attribution
- **Fama & French (1993)** — *Common Risk Factors in the Returns on Stocks and Bonds* → Part 6 factor attribution
- **Carhart (1997)** — *On Persistence in Mutual Fund Performance* → Part 6 4-factor model
- **Jegadeesh & Titman (1993)** — *Returns to Buying Winners and Selling Losers* → Part 6 momentum strategy
- **McNeil, Frey & Embrechts (2015)** — *Quantitative Risk Management* → Part 5/7 EVT, CVaR
- **AIMA Alternative Investment Management Practices Guide** → Part 10 NAV methodology

---

## License

MIT License. See `LICENSE` for full text.

---

## Contact

📧 upasani99@protonmail.ch  
🔗 [LinkedIn](https://linkedin.com/in/VedantUpasani)  
🐙 [GitHub](https://github.com/VedantUpasani46)

Open to quantitative developer, financial engineer, and systematic trading roles globally.

---

*Built independently over 2024–2026. All implementations are from first principles against the original academic papers — not wrappers around existing quant libraries. Every module was validated against real market data from Yahoo Finance, CBOE, and SEC EDGAR.*
