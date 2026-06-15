# Strategic Finance & Acquisition Intelligence Platform

A standalone Streamlit MVP built for the Bending Spoons Finance Graduate Program. The platform supports acquisition screening, financial due diligence, scenario modelling, post-M&A integration planning, and AI-assisted board memo generation for technology business acquisitions.

## Why This Project Matters

High-growth technology acquirers need finance teams that can move from ambiguous target data to clear judgement quickly. This project demonstrates first-principles reasoning, financial planning and control, acquisition diligence, reporting readiness assessment, and board-level communication in one local-first workflow.

## Features

- Manual target-company input with sensible demo defaults
- Acquisition attractiveness score and strategic fit summary
- Enterprise value, EV/revenue, EV/EBITDA, net debt, and net leverage analysis
- Financial due diligence dashboard covering FCF, capex, working capital, liquidity runway, reporting complexity, and quality-of-earnings flags
- Post-acquisition integration plan across finance systems, reporting consolidation, intercompany flows, revenue recognition, chart of accounts, FP&A, tax, and operations
- Base, downside, and upside scenario modelling
- Downside resilience score and break-even EBITDA margin sensitivity
- Deterministic board memo generation without paid APIs
- Optional OpenAI-compatible API enhancement when environment variables are configured
- Markdown and text memo exports
- Dark institutional dashboard UI for strategic finance workflows

## Architecture

```text
app.py                               Streamlit dashboard
src/acquisition_screening.py          Strategic fit, valuation, leverage, and risk screening
src/financial_due_diligence.py        FCF, liquidity, reporting, and quality-of-earnings analysis
src/integration_planning.py           Post-M&A workstreams and 100-day plan
src/scenario_analysis.py              Base/downside/upside case modelling
src/board_memo.py                     Deterministic and optional LLM board memo generation
src/report_generator.py               Local report export helpers
src/utils.py                          Formatting and robustness helpers
tests/                                Pytest coverage for core logic
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

1. Open the assumptions menu from the left rail.
2. Enter or adjust the target company, sector, revenue, growth, EBITDA, cash, debt, price, employee count, jurisdictions, accounting framework, and strategic rationale.
3. Review the Overview tab for attractiveness, valuation, leverage, liquidity, integration risk, and suggested next action.
4. Use Acquisition Screening to inspect valuation multiples, risk flags, and diligence priorities.
5. Use Financial Due Diligence to assess FCF, liquidity runway, quality of earnings, and reporting complexity.
6. Use Integration Planning to review post-close finance workstreams and the 100-day plan.
7. Use Scenario Analysis to tune downside, base, and upside assumptions.
8. Generate and export a Board Memo.

## Methodology

The scoring model blends growth, EBITDA margin, leverage, valuation discipline, strategic rationale, jurisdiction complexity, and accounting alignment. It is intentionally transparent so assumptions can be challenged rather than hidden behind a black box.

The diligence module estimates free cash flow from EBITDA, cash conversion, capex, working-capital impact, and liquidity runway. The scenario module links growth, margin, capex, working capital, financing cost, synergies, integration costs, and debt repayment or drawdown into a board-readable range of outcomes.

## AI Board Memo

The Board Memo tab works in two modes:

- No API key: deterministic local memo generation using the platform's analysis.
- API key available: optional OpenAI-compatible enhancement using `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `OPENAI_MODEL`.

Copy `.env.example` to `.env` if using an API. The project works without this step.

## Limitations

- Manual inputs are used instead of proprietary databases.
- The MVP is designed for analysis and communication, not legal, tax, or accounting advice.
- Accounting-risk logic is simplified and should be replaced with detailed policy checklists for production use.
- The scenario model is one-period and deliberately transparent.
- Synergies and integration costs require management validation.

## Future Improvements

- Add multi-year financial statements and forecast periods.
- Add upload support for management accounts and due diligence packs.
- Add entity-level consolidation and statutory reporting trackers.
- Add synergy pipeline ownership and benefits realisation tracking.
- Add automated PDF board pack export.
- Add a structured management-question tracker.

## Suggested CV Bullet

Built a strategic finance and acquisition intelligence platform modelling acquisition screening, financial due diligence, scenario analysis, post-M&A integration planning and AI-assisted board memo generation for technology business acquisitions.
