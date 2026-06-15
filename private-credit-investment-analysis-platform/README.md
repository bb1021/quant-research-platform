# Private Credit Investment Analysis Platform

A standalone Streamlit platform for evaluating direct lending and private credit investment opportunities. The app analyses leverage, liquidity, interest coverage, free cash flow conversion, covenant headroom, downside resilience and memo-ready investment recommendations.

## Role Relevance

Built for a Blackstone Credit & Insurance, Private Credit Strategies Off-Cycle Intern profile. The project demonstrates credit judgement, downside scenario thinking, debt capacity analysis, liquidity assessment, covenant analysis and investment committee memo writing.

## Screenshots

Screenshots can be added to `docs/screenshots/` after running the app locally.

## Features

- Realistic demo borrower assumptions
- Credit screening with EV / EBITDA, gross leverage, net leverage, interest coverage, FCF conversion and debt capacity
- Capital structure, cash flow, leverage, coverage and liquidity analysis
- Base, downside and severe downside scenario modelling
- Covenant headroom by scenario for leverage, coverage and liquidity covenants
- First breach and key risk driver identification
- Transparent rule-based recommendation categories: Attractive, Watchlist, High risk and Avoid
- Deterministic private credit investment memo generation without paid APIs
- Optional OpenAI-compatible memo enhancement if environment variables are configured
- Dark institutional Streamlit dashboard with high-density KPI cards and charts

## Architecture

```text
app.py                       Streamlit dashboard
src/credit_metrics.py         Credit ratios, debt capacity, strengths and risks
src/scenario_analysis.py      Base, downside and severe downside modelling
src/covenants.py              Covenant headroom and breach analysis
src/recommendation.py         Rule-based credit recommendation
src/investment_memo.py        Deterministic and optional LLM memo generation
src/report_generator.py       Local memo export helpers
src/utils.py                  Formatting and robust divide-by-zero helpers
tests/                        Pytest coverage for core logic
```

## Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Optional AI Configuration

The memo generator works without an API key. To enable optional OpenAI-compatible enhancement, copy `.env.example` to `.env` and set:

```text
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
```

## Limitations

- Manual inputs are used instead of proprietary lender data rooms or third-party credit databases.
- The model is a one-period MVP and does not replace full lender underwriting.
- Covenant definitions are simplified and should be tailored to actual credit agreement language.
- Scenario assumptions require diligence validation.
- No legal, tax, ratings or investment advice is provided.

## Future Improvements

- Add multi-year forecast periods and cash sweep mechanics.
- Add debt tranche modelling and pricing grids.
- Add credit agreement covenant definition templates.
- Add upload support for management accounts.
- Add PDF investment committee pack export.
- Add sponsor support and collateral quality modules.

## Suggested CV Bullet

Built a private credit investment analysis platform evaluating leverage, liquidity, covenant headroom, debt capacity and downside resilience, generating AI-assisted investment memoranda for direct lending opportunities.
