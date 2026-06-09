# Quant Research Platform with AI Research Assistant

A local-first MVP for equity factor research. It downloads and caches market data, calculates factor signals, backtests long-only factor strategies, evaluates risk and performance, and generates institutional-style equity research reports with or without an LLM API key.

## Why This Matters

Quant research workflows often split data ingestion, signal research, backtesting, risk review, and reporting across disconnected notebooks. This project turns the workflow into one usable application that is modular enough to extend and simple enough to run locally.

## Features

- Daily OHLCV ingestion from Yahoo Finance via `yfinance`
- Local DuckDB cache with refresh support
- Multi-ticker factor engine
- Momentum, volatility, mean reversion, moving average, and relative strength signals
- Long-only top-N backtester with monthly, weekly, or daily rebalancing
- Configurable starting capital, benchmark ticker, transaction costs, and top-N selection
- Equal weight, inverse volatility, mean-variance, and risk parity weighting
- User-visible fallback warnings when optimizer inputs are insufficient
- Transaction cost and turnover tracking
- Benchmark comparison against SPY or any available ticker
- CAGR, volatility, Sharpe, Sortino, drawdown, Calmar, beta, alpha, information ratio, win rate
- Rolling volatility, rolling Sharpe, drawdown, VaR, CVaR
- Streamlit dashboard with Data, Factors, Backtest, Risk Analytics, and AI Research Report tabs
- Modern dark institutional-style dashboard with visible navigation, KPI cards, factor analytics, benchmark-aware backtesting, risk analytics, and AI-assisted equity reports
- Deterministic research reports without paid APIs
- Optional OpenAI-compatible completion when environment variables are configured
- Markdown, text, and CSV exports for reports, backtest summaries, and metrics

## Architecture

```text
app.py                         Streamlit dashboard
src/data_loader.py             yfinance ingestion and DuckDB cache
src/factors.py                 Factor calculations and rankings
src/backtester.py              Signal-driven portfolio simulation
src/portfolio.py               Portfolio construction methods
src/metrics.py                 Performance metrics
src/risk.py                    Risk analytics helpers
src/ai_research.py             Template and optional LLM report generation
src/report_generator.py        Local report export helpers
tests/                         Pytest coverage for core logic
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run

```bash
streamlit run app.py
```

## Example Workflow

1. Enter a ticker universe such as `AAPL, MSFT, NVDA, AMZN, GOOGL, JPM, XOM, SPY`.
2. Select a date range.
3. Load data. The first run downloads data and writes to DuckDB. Later runs use the cache unless refresh is selected.
4. Review factor rankings.
5. Run a top-N factor backtest with a selected weighting method, benchmark, transaction cost, and starting capital.
6. Review equity curve, drawdowns, holdings, trades, metrics, turnover, and benchmark-relative risk analytics.
7. Download the backtest summary or metrics table.
8. Generate an AI research report for a selected ticker.
9. Export the report as markdown or text.

## Methodology

The factor engine calculates trailing return momentum over 12, 6, and 3 months, annualized realized volatility over 60 and 20 trading days, a short-term mean reversion signal, a 50 versus 200 day moving average crossover, and cross-sectional relative strength ranking. Backtests shift factor signals by one trading day before applying returns to reduce lookahead bias.

Portfolio construction supports equal weighting, inverse volatility weighting, constrained mean-variance optimization, and a risk parity approximation. Optimizers are long-only and fall back to equal weights if inputs are insufficient or optimization fails. The dashboard surfaces these fallbacks as warnings so the user can distinguish a successful optimized portfolio from a safe fallback.

## AI Research Layer

The report generator works in two modes:

- No API key: deterministic report based on local quantitative data.
- API key available: optional OpenAI-compatible chat completion using `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`.

Copy `.env.example` to `.env` if using an API. The app works without this step.

## Tests

```bash
python -m pytest -q
```

The tests cover data formatting and return calculation, factor ranking, Sharpe ratio, max drawdown, backtest output shape, a basic no-lookahead-bias case, missing benchmark handling, starting capital, and deterministic AI report generation without an API key.

## Limitations

- Yahoo Finance data can be delayed, revised, or unavailable for some tickers.
- The MVP uses price-only factors and placeholder fundamental commentary.
- Backtests are simplified and do not model borrow costs, taxes, market impact, or corporate action edge cases beyond adjusted prices.
- Monthly rebalancing uses first available trading day of the month and prior-day signals.
- This is research tooling, not investment advice or a trading system.

## Future Improvements

- Add fundamental data providers and valuation factor support.
- Add sector, country, and liquidity constraints.
- Add survivorship-bias-free universe support.
- Add walk-forward analysis and parameter sweeps.
- Add report chart exports and PDF generation.
- Add authentication and multi-user project storage.

## Suggested CV Bullet

Built a full-stack quantitative research platform integrating market data ingestion, factor modelling, portfolio construction, benchmark-relative backtesting, risk analytics, and an AI-powered equity research assistant, with local data caching and an interactive Streamlit dashboard.
