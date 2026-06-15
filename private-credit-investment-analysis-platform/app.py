from __future__ import annotations

from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.covenants import analyse_covenants
from src.credit_metrics import CreditAssumptions, calculate_credit_metrics, credit_risks, credit_strengths, metrics_table
from src.investment_memo import generate_investment_memo
from src.recommendation import recommend_credit
from src.report_generator import markdown_to_text, save_report
from src.scenario_analysis import DEFAULT_SCENARIOS, run_scenarios
from src.utils import money, multiple, pct


load_dotenv()

st.set_page_config(page_title="Private Credit Investment Analysis Platform", page_icon="C", layout="wide")

PAGES = ["Overview", "Credit Screening", "Financial Analysis", "Scenario Analysis", "Covenant Headroom", "Investment Memo"]
RAIL_ICONS = {"Overview": "O", "Credit Screening": "C", "Financial Analysis": "F", "Scenario Analysis": "S", "Covenant Headroom": "H", "Investment Memo": "M"}
CHART_COLOURS = ["#58d5b7", "#8da2ff", "#f6c36a", "#ef7e8e", "#6fb7ff", "#9ad66f"]


st.markdown(
    """
    <style>
    :root {
        --bg: #050b12;
        --panel: rgba(16, 30, 43, 0.78);
        --line: rgba(130, 158, 177, 0.22);
        --line-strong: rgba(130, 158, 177, 0.40);
        --text: #eef6f5;
        --muted: #9fb0ba;
        --accent: #58d5b7;
        --danger: #ef7e8e;
        --warn: #f6c36a;
    }
    html, body, .stApp, div[data-testid="stAppViewContainer"] {
        overflow-x: hidden;
        background:
            radial-gradient(circle at 46% -18%, rgba(48, 126, 170, 0.18), transparent 34rem),
            radial-gradient(circle at 95% 9%, rgba(88, 213, 183, 0.10), transparent 28rem),
            linear-gradient(180deg, #07111a 0%, #061019 52%, #040a10 100%) !important;
    }
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, system-ui, sans-serif;
    }
    header[data-testid="stHeader"], div[data-testid="stToolbar"], footer {
        visibility: hidden;
        height: 0;
    }
    .block-container {
        max-width: 1496px !important;
        padding: 1.45rem 1.55rem 6.2rem 7.35rem !important;
    }
    h3 {
        color: #c6d5dc !important;
        font-size: 1.52rem !important;
        margin-top: 1.45rem !important;
    }
    p, .stCaption, [data-testid="stMarkdownContainer"], label {
        color: var(--muted);
    }
    .app-rail {
        position: fixed;
        inset: 1.45rem auto 1.45rem 1.45rem;
        width: 88px;
        z-index: 950;
        border: 1px solid rgba(120, 151, 171, 0.20);
        border-radius: 14px 0 0 14px;
        background: linear-gradient(180deg, rgba(9, 21, 31, 0.92), rgba(5, 13, 21, 0.92));
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-top: 5.85rem;
    }
    div[data-testid="stPopover"] {
        position: fixed !important;
        top: 3.1rem !important;
        left: 3.15rem !important;
        z-index: 1000 !important;
        width: 34px !important;
    }
    div[data-testid="stPopover"] button {
        width: 34px !important;
        height: 34px !important;
        min-height: 34px !important;
        padding: 0 !important;
        border: 0 !important;
        background: transparent !important;
        color: #c9d5dc !important;
        box-shadow: none !important;
        font-size: 0 !important;
    }
    div[data-testid="stPopover"] button * {
        display: none !important;
    }
    div[data-testid="stPopover"] button::before {
        content: "\\2630";
        font-size: 1.55rem;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Primary Navigation"]) {
        position: fixed !important;
        inset: 7.05rem auto auto 1.88rem !important;
        width: 72px !important;
        z-index: 1001 !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Primary Navigation"]) > label {
        display: none !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Primary Navigation"]) div[role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 0.82rem !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Primary Navigation"]) label[data-baseweb="radio"] {
        width: 72px;
        height: 52px;
        display: grid !important;
        place-items: center !important;
        color: #b7c5cc;
        border-radius: 0 999px 999px 0;
        border: 1px solid transparent;
        position: relative;
        margin: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Primary Navigation"]) label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Primary Navigation"]) label[data-baseweb="radio"] p {
        width: 28px;
        height: 28px;
        display: grid;
        place-items: center;
        border: 1px solid currentColor;
        border-radius: 8px;
        font-size: 0.72rem !important;
        color: inherit !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Primary Navigation"]) label[data-baseweb="radio"]:has(input:checked) {
        color: #f0fffb;
        background: linear-gradient(90deg, rgba(92, 116, 143, 0.26), rgba(65, 85, 106, 0.18));
        box-shadow: inset -1px 0 0 rgba(88, 213, 183, 0.45);
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Primary Navigation"]) label[data-baseweb="radio"]:has(input:checked)::after {
        content: "";
        position: absolute;
        left: 72px;
        width: 2px;
        height: 34px;
        border-radius: 999px;
        background: #58d5b7;
        box-shadow: 0 0 16px rgba(88, 213, 183, 0.62);
    }
    .reference-header {
        min-height: 78px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 2.1rem;
        border: 1px solid rgba(120, 151, 171, 0.22);
        border-radius: 0 14px 14px 0;
        background: linear-gradient(180deg, rgba(9, 20, 31, 0.82), rgba(6, 14, 22, 0.86));
        box-shadow: 0 30px 90px rgba(0, 0, 0, 0.26);
    }
    .reference-title-row {
        display: flex;
        align-items: baseline;
        gap: 1.55rem;
    }
    .reference-title {
        color: #f4f7f8;
        font-size: 1.55rem;
        font-weight: 850;
    }
    .reference-page-label {
        color: #aab7c0;
        font-size: 1.01rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.52rem;
        border: 1px solid rgba(88, 213, 183, 0.20);
        border-radius: 999px;
        padding: 0.58rem 1rem;
        background: rgba(20, 55, 58, 0.28);
        color: #f0f6f5;
        font-weight: 800;
        font-size: 0.86rem;
        white-space: nowrap;
    }
    .status-pill span {
        width: 0.58rem;
        height: 0.58rem;
        border-radius: 999px;
        background: #f0f7f5;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Section Navigation"]) {
        position: fixed !important;
        left: 8.8rem !important;
        bottom: 2.15rem !important;
        z-index: 920 !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Section Navigation"]) > label {
        display: none !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Section Navigation"]) div[role="radiogroup"] {
        display: flex !important;
        gap: 1rem !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Section Navigation"]) label[data-baseweb="radio"] {
        height: 52px !important;
        min-width: 96px;
        padding: 0 1.55rem !important;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 11px !important;
        background: rgba(14, 27, 39, 0.72) !important;
        border: 1px solid rgba(120, 151, 171, 0.16) !important;
        color: #bac6cd !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Section Navigation"]) label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Section Navigation"]) label[data-baseweb="radio"] p {
        color: inherit !important;
        font-size: 1rem !important;
        font-weight: 620 !important;
        margin: 0 !important;
    }
    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Section Navigation"]) label[data-baseweb="radio"]:has(input:checked) {
        color: #f5f8f8 !important;
        background: linear-gradient(180deg, rgba(35, 57, 75, 0.78), rgba(16, 31, 44, 0.82)) !important;
        border-color: rgba(88, 213, 183, 0.22) !important;
        box-shadow: inset 0 -2px 0 #58d5b7, 0 10px 26px rgba(0, 0, 0, 0.22) !important;
    }
    .metric-card, .panel-card {
        border: 1px solid rgba(120, 151, 171, 0.18);
        border-radius: 12px;
        background: radial-gradient(circle at 86% 6%, rgba(76, 126, 164, 0.11), transparent 10rem), linear-gradient(180deg, rgba(18, 35, 49, 0.74), rgba(11, 24, 36, 0.80));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
        padding: 1.22rem 1.35rem;
    }
    .metric-card {
        min-height: 118px;
    }
    .metric-label {
        color: #bcc8cf;
        font-size: 0.88rem;
        margin-bottom: 0.82rem;
    }
    .metric-value {
        color: #f6f8f8;
        font-size: 1.55rem;
        font-weight: 850;
        line-height: 1;
        margin-bottom: 0.56rem;
    }
    .metric-note {
        color: #aab6be;
        font-size: 0.86rem;
    }
    .panel-title {
        color: #f3f6f7;
        font-size: 1.08rem;
        font-weight: 820;
        margin-bottom: 1rem;
    }
    .finance-note {
        color: #c9d9dc;
        line-height: 1.58;
        font-size: 0.96rem;
    }
    .flag-list {
        margin: 0;
        padding-left: 1.05rem;
        color: #d7e5e2;
    }
    .flag-list li {
        margin: 0.48rem 0;
    }
    .stTextArea textarea, .stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div, div[data-baseweb="input"] input {
        background: #0f1922 !important;
        color: #edf4f2 !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] span, div[data-baseweb="select"] input, div[data-baseweb="popover"], div[data-baseweb="menu"] {
        color: #edf4f2 !important;
        background-color: #0f1922 !important;
    }
    .stButton > button, .stDownloadButton > button {
        background: linear-gradient(180deg, #5ee0bf 0%, #36b99a 100%) !important;
        color: #06100d !important;
        border: 1px solid rgba(133, 245, 215, 0.5) !important;
        border-radius: 8px !important;
        font-weight: 850 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _normalise_page(page: str | None) -> str:
    return page if page in PAGES else "Overview"


def _sync_from_rail() -> None:
    page = _normalise_page(st.session_state.get("rail_page"))
    st.session_state.active_page = page
    st.session_state.bottom_page = page


def _sync_from_bottom() -> None:
    page = _normalise_page(st.session_state.get("bottom_page"))
    st.session_state.active_page = page
    st.session_state.rail_page = page


def _active_page() -> str:
    page = _normalise_page(st.session_state.get("active_page"))
    st.session_state.active_page = page
    st.session_state.rail_page = page
    st.session_state.bottom_page = page
    return page


def scenario_inputs() -> dict[str, dict[str, float]]:
    return {
        name: {key: float(st.session_state.get(f"{name}_{key}", value)) for key, value in defaults.items()}
        for name, defaults in DEFAULT_SCENARIOS.items()
    }


def get_assumptions() -> CreditAssumptions:
    defaults = CreditAssumptions()
    return CreditAssumptions(
        company_name=st.session_state.get("company_name", defaults.company_name),
        sector=st.session_state.get("sector", defaults.sector),
        revenue=float(st.session_state.get("revenue", defaults.revenue)),
        ebitda=float(st.session_state.get("ebitda", defaults.ebitda)),
        ebit=float(st.session_state.get("ebit", defaults.ebit)),
        free_cash_flow=float(st.session_state.get("free_cash_flow", defaults.free_cash_flow)),
        cash=float(st.session_state.get("cash", defaults.cash)),
        total_debt=float(st.session_state.get("total_debt", defaults.total_debt)),
        interest_expense=float(st.session_state.get("interest_expense", defaults.interest_expense)),
        capex=float(st.session_state.get("capex", defaults.capex)),
        working_capital_impact=float(st.session_state.get("working_capital_impact", defaults.working_capital_impact)),
        revolver_availability=float(st.session_state.get("revolver_availability", defaults.revolver_availability)),
        enterprise_value=float(st.session_state.get("enterprise_value", defaults.enterprise_value)),
        existing_debt_maturity=float(st.session_state.get("existing_debt_maturity", defaults.existing_debt_maturity)),
        selected_leverage_multiple=float(st.session_state.get("selected_leverage_multiple", defaults.selected_leverage_multiple)),
    )


def compute_context():
    assumptions = get_assumptions()
    metrics = calculate_credit_metrics(assumptions)
    scenarios = run_scenarios(assumptions, scenario_inputs())
    covenants = analyse_covenants(
        scenarios,
        max_net_leverage=float(st.session_state.get("max_net_leverage", 5.25)),
        min_interest_coverage=float(st.session_state.get("min_interest_coverage", 1.75)),
        min_liquidity=float(st.session_state.get("min_liquidity", 35.0)),
    )
    recommendation = recommend_credit(metrics, scenarios, covenants)
    strengths = credit_strengths(assumptions, metrics)
    risks = credit_risks(assumptions, metrics)
    return assumptions, metrics, scenarios, covenants, recommendation, strengths, risks


def style_chart(fig, height: int = 330):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0c141c",
        font={"color": "#dce7e4", "family": "Segoe UI, Inter, system-ui, sans-serif", "size": 12},
        colorway=CHART_COLOURS,
        legend={"orientation": "h", "y": 1.05, "x": 1, "xanchor": "right"},
        margin={"l": 40, "r": 18, "t": 46, "b": 38},
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)")
    return fig


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"""<div class="metric-card"><div class="metric-label">{escape(label)}</div><div class="metric-value">{escape(value)}</div><div class="metric-note">{escape(note)}</div></div>"""


def bullets(items: list[str]) -> str:
    return '<ul class="flag-list">' + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def format_metric(metric: str, value: object) -> str:
    if isinstance(value, str):
        return value
    numeric = float(value)
    if "Margin" in metric or "Conversion" in metric or "Debt / Enterprise" in metric or "FCF / Debt" in metric:
        return pct(numeric)
    if "Coverage" in metric or "Interest" in metric or "EBITDA" in metric and "/" in metric:
        return multiple(numeric)
    if "Runway" in metric:
        return f"{numeric:.0f} months"
    if "Score" in metric:
        return f"{numeric:.1f}/100"
    if any(token in metric for token in ["Revenue", "EBITDA", "EBIT", "Cash", "Debt", "Liquidity", "Value", "Need", "Capacity", "Flow"]):
        return money(numeric)
    return f"{numeric:.1f}"


def formatted_table(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["Value"] = [format_metric(metric, value) for metric, value in out[["Metric", "Value"]].to_numpy()]
    return out


def render_navigation(active_page: str) -> None:
    st.markdown('<nav class="app-rail" aria-label="Primary"></nav>', unsafe_allow_html=True)
    st.radio("Primary Navigation", PAGES, index=PAGES.index(active_page), key="rail_page", format_func=lambda page: RAIL_ICONS[page], label_visibility="collapsed", on_change=_sync_from_rail)
    st.radio("Section Navigation", PAGES, index=PAGES.index(active_page), key="bottom_page", horizontal=True, label_visibility="collapsed", on_change=_sync_from_bottom)


def render_settings() -> None:
    defaults = CreditAssumptions()
    with st.popover("Menu", help="Open credit assumptions"):
        st.markdown("#### Credit Assumptions")
        st.caption("Values are in USD millions unless otherwise stated.")
        st.text_input("Company name", value=st.session_state.get("company_name", defaults.company_name), key="company_name")
        st.text_input("Sector", value=st.session_state.get("sector", defaults.sector), key="sector")
        for left, right in [
            ("revenue", "ebitda"),
            ("ebit", "free_cash_flow"),
            ("cash", "total_debt"),
            ("interest_expense", "capex"),
            ("working_capital_impact", "revolver_availability"),
            ("enterprise_value", "existing_debt_maturity"),
        ]:
            c1, c2 = st.columns(2)
            c1.number_input(left.replace("_", " ").title(), value=float(st.session_state.get(left, getattr(defaults, left))), step=5.0, key=left)
            c2.number_input(right.replace("_", " ").title(), value=float(st.session_state.get(right, getattr(defaults, right))), step=5.0, key=right)
        st.number_input("Selected leverage multiple", value=float(st.session_state.get("selected_leverage_multiple", defaults.selected_leverage_multiple)), step=0.25, key="selected_leverage_multiple")
        st.markdown("#### Covenant Package")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Maximum net leverage covenant", value=float(st.session_state.get("max_net_leverage", 5.25)), step=0.25, key="max_net_leverage")
        c2.number_input("Minimum interest coverage covenant", value=float(st.session_state.get("min_interest_coverage", 1.75)), step=0.25, key="min_interest_coverage")
        c3.number_input("Minimum liquidity covenant", value=float(st.session_state.get("min_liquidity", 35.0)), step=5.0, key="min_liquidity")


def render_header(active_page: str, recommendation: dict[str, object]) -> None:
    st.markdown(
        f"""<header class="reference-header"><div class="reference-title-row"><div class="reference-title">Private Credit Investment Analysis Platform</div><div class="reference-page-label">{escape(active_page)}</div></div><div class="status-pill"><span></span>{escape(str(recommendation["category"]))}</div></header>""",
        unsafe_allow_html=True,
    )


def render_overview(assumptions, metrics, scenarios, covenants, recommendation, strengths, risks) -> None:
    severe = scenarios.loc[scenarios["Case"] == "Severe Downside"]
    severe_score = float(severe["Downside Resilience Score"].iloc[0]) if not severe.empty else 0.0
    kpis = [
        ("Company", assumptions.company_name, assumptions.sector),
        ("Credit recommendation", recommendation["category"], f"Score {recommendation['score']}/100"),
        ("Revenue", money(metrics["Revenue"]), f"EBITDA {money(metrics['EBITDA'])}"),
        ("EBITDA margin", pct(assumptions.ebitda / assumptions.revenue), "Current run-rate"),
        ("Net debt", money(metrics["Net Debt"]), money(metrics["Total Debt"])),
        ("Gross leverage", multiple(metrics["Debt / EBITDA"]), "Debt / EBITDA"),
        ("Net leverage", multiple(metrics["Net Debt / EBITDA"]), "Net debt / EBITDA"),
        ("Interest coverage", multiple(metrics["EBITDA / Interest"]), "EBITDA basis"),
        ("Liquidity runway", f"{metrics['Liquidity Runway Months']:.0f} months", money(metrics["Liquidity"])),
        ("Downside resilience", f"{severe_score:.1f}/100", "Severe downside"),
    ]
    for start in range(0, len(kpis), 5):
        cols = st.columns(5)
        for col, (label, value, note) in zip(cols, kpis[start:start + 5]):
            col.markdown(metric_card(str(label), str(value), str(note)), unsafe_allow_html=True)
    st.markdown("### Credit Dashboard")
    c1, c2, c3 = st.columns([1.0, 0.95, 0.95], gap="medium")
    with c1:
        chart_df = pd.DataFrame({"Metric": ["Cash", "Revolver", "Debt", "Maturity"], "Value": [assumptions.cash, assumptions.revolver_availability, assumptions.total_debt, assumptions.existing_debt_maturity]})
        st.plotly_chart(style_chart(px.bar(chart_df, x="Metric", y="Value", title="Capital Structure and Liquidity"), 300), use_container_width=True)
    with c2:
        st.markdown('<div class="panel-card"><div class="panel-title">Suggested Diligence Action</div>', unsafe_allow_html=True)
        action = "Validate recurring EBITDA, cash conversion, covenant definitions, and maturity wall refinancing options."
        st.markdown(f"<p class='finance-note'>{escape(action)}</p>", unsafe_allow_html=True)
        st.markdown(bullets(recommendation["rationale"]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="panel-card"><div class="panel-title">Key Credit View</div>', unsafe_allow_html=True)
        st.markdown(bullets([strengths[0], risks[0], f"First covenant breach: {covenants['first_breach']}", f"Key risk driver: {covenants['key_driver']}"]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("### Downside Resilience")
    st.plotly_chart(style_chart(px.bar(scenarios, x="Case", y="Downside Resilience Score", title="Scenario Resilience Score"), 310), use_container_width=True)


def render_credit_screening(assumptions, metrics, recommendation) -> None:
    st.markdown("### Credit Screening")
    st.caption("Use the left menu to adjust borrower assumptions, capital structure and covenant package.")
    cols = st.columns(4)
    cols[0].markdown(metric_card("EV / EBITDA", multiple(metrics["EV / EBITDA"]), money(assumptions.enterprise_value)), unsafe_allow_html=True)
    cols[1].markdown(metric_card("Gross leverage", multiple(metrics["Debt / EBITDA"]), money(assumptions.total_debt)), unsafe_allow_html=True)
    cols[2].markdown(metric_card("Net leverage", multiple(metrics["Net Debt / EBITDA"]), money(metrics["Net Debt"])), unsafe_allow_html=True)
    cols[3].markdown(metric_card("Debt capacity", money(metrics["Debt Capacity"]), f"At {multiple(assumptions.selected_leverage_multiple)}"), unsafe_allow_html=True)
    st.dataframe(formatted_table(metrics_table(metrics)), use_container_width=True, hide_index=True)
    st.markdown("#### Recommendation Logic")
    st.markdown(bullets(recommendation["rationale"]), unsafe_allow_html=True)


def render_financial_analysis(assumptions, metrics, strengths, risks) -> None:
    st.markdown("### Financial Analysis")
    c1, c2 = st.columns(2)
    profile = pd.DataFrame({"Metric": ["Revenue", "EBITDA", "EBIT", "FCF", "Capex", "Working Capital"], "Value": [assumptions.revenue, assumptions.ebitda, assumptions.ebit, assumptions.free_cash_flow, assumptions.capex, assumptions.working_capital_impact]})
    c1.plotly_chart(style_chart(px.bar(profile, x="Metric", y="Value", title="Cash Flow Profile"), 330), use_container_width=True)
    leverage = pd.DataFrame({"Metric": ["Gross Leverage", "Net Leverage", "EBITDA Coverage", "EBIT Coverage"], "Value": [metrics["Debt / EBITDA"], metrics["Net Debt / EBITDA"], metrics["EBITDA / Interest"], metrics["EBIT / Interest"]]})
    c2.plotly_chart(style_chart(px.bar(leverage, x="Metric", y="Value", title="Leverage and Coverage"), 330), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel-card"><div class="panel-title">Credit Strengths</div>', unsafe_allow_html=True)
        st.markdown(bullets(strengths), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="panel-card"><div class="panel-title">Credit Risks</div>', unsafe_allow_html=True)
        st.markdown(bullets(risks), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_scenario_analysis(assumptions, scenarios) -> None:
    st.markdown("### Scenario Analysis")
    with st.expander("Scenario assumptions", expanded=False):
        for name, defaults in DEFAULT_SCENARIOS.items():
            st.markdown(f"#### {name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input(f"{name} revenue growth", value=float(st.session_state.get(f"{name}_revenue_growth", defaults["revenue_growth"])), step=0.01, key=f"{name}_revenue_growth")
            c2.number_input(f"{name} EBITDA margin", value=float(st.session_state.get(f"{name}_ebitda_margin", defaults["ebitda_margin"])), step=0.01, key=f"{name}_ebitda_margin")
            c3.number_input(f"{name} interest rate", value=float(st.session_state.get(f"{name}_interest_rate", defaults["interest_rate"])), step=0.005, key=f"{name}_interest_rate")
            c4.number_input(f"{name} capex % revenue", value=float(st.session_state.get(f"{name}_capex_pct_revenue", defaults["capex_pct_revenue"])), step=0.005, key=f"{name}_capex_pct_revenue")
            c5, c6, c7 = st.columns(3)
            c5.number_input(f"{name} working capital impact", value=float(st.session_state.get(f"{name}_working_capital_impact", defaults["working_capital_impact"])), step=1.0, key=f"{name}_working_capital_impact")
            c6.number_input(f"{name} cost savings", value=float(st.session_state.get(f"{name}_cost_savings", defaults["cost_savings"])), step=1.0, key=f"{name}_cost_savings")
            c7.number_input(f"{name} debt repayment / drawdown", value=float(st.session_state.get(f"{name}_debt_repayment", defaults["debt_repayment"])), step=1.0, key=f"{name}_debt_repayment")
    scenarios = run_scenarios(assumptions, scenario_inputs())
    st.dataframe(format_scenarios(scenarios), use_container_width=True, hide_index=True)
    st.plotly_chart(style_chart(px.line(scenarios, x="Case", y=["EBITDA", "Free Cash Flow", "Liquidity"], markers=True, title="Scenario Outputs"), 350), use_container_width=True)


def format_scenarios(scenarios: pd.DataFrame) -> pd.DataFrame:
    out = scenarios.copy()
    for col in ["Revenue", "EBITDA", "Free Cash Flow", "Net Debt", "Liquidity"]:
        out[col] = out[col].map(money)
    for col in ["Net Leverage", "Interest Coverage"]:
        out[col] = out[col].map(multiple)
    out["Downside Resilience Score"] = out["Downside Resilience Score"].map(lambda value: f"{value:.1f}/100")
    return out


def render_covenants(scenarios, covenants) -> None:
    st.markdown("### Covenant Headroom")
    c1, c2, c3 = st.columns(3)
    max_net_leverage = c1.number_input("Maximum net leverage covenant", value=float(st.session_state.get("max_net_leverage_page", st.session_state.get("max_net_leverage", 5.25))), step=0.25, key="max_net_leverage_page")
    min_interest_coverage = c2.number_input("Minimum interest coverage covenant", value=float(st.session_state.get("min_interest_coverage_page", st.session_state.get("min_interest_coverage", 1.75))), step=0.25, key="min_interest_coverage_page")
    min_liquidity = c3.number_input("Minimum liquidity covenant", value=float(st.session_state.get("min_liquidity_page", st.session_state.get("min_liquidity", 35.0))), step=5.0, key="min_liquidity_page")
    page_covenants = analyse_covenants(scenarios, max_net_leverage, min_interest_coverage, min_liquidity)
    st.dataframe(page_covenants["table"], use_container_width=True, hide_index=True)
    st.markdown(f"<p class='finance-note'>First breach: <strong>{escape(str(page_covenants['first_breach']))}</strong>. Key risk driver: <strong>{escape(str(page_covenants['key_driver']))}</strong>.</p>", unsafe_allow_html=True)


def render_memo(assumptions, metrics, strengths, risks, scenarios, covenants, recommendation) -> None:
    st.markdown("### Investment Memo")
    use_llm = st.checkbox("Use optional OpenAI-compatible API if configured", value=False)
    if st.button("Generate investment memo"):
        memo = generate_investment_memo(assumptions, metrics, strengths, risks, scenarios, covenants, recommendation, use_llm=use_llm)
        st.session_state.memo = memo
        st.session_state.memo_path = save_report(memo, assumptions.company_name)
    memo = st.session_state.get("memo") or generate_investment_memo(assumptions, metrics, strengths, risks, scenarios, covenants, recommendation)
    st.markdown(memo)
    c1, c2 = st.columns(2)
    c1.download_button("Download markdown", memo, file_name=f"{assumptions.company_name.lower().replace(' ', '_')}_investment_memo.md", mime="text/markdown", use_container_width=True)
    c2.download_button("Download text", markdown_to_text(memo), file_name=f"{assumptions.company_name.lower().replace(' ', '_')}_investment_memo.txt", mime="text/plain", use_container_width=True)


active_page = _active_page()
render_navigation(active_page)
render_settings()
assumptions, metrics, scenarios, covenants, recommendation, strengths, risks = compute_context()
render_header(active_page, recommendation)

if active_page == "Overview":
    render_overview(assumptions, metrics, scenarios, covenants, recommendation, strengths, risks)
elif active_page == "Credit Screening":
    render_credit_screening(assumptions, metrics, recommendation)
elif active_page == "Financial Analysis":
    render_financial_analysis(assumptions, metrics, strengths, risks)
elif active_page == "Scenario Analysis":
    render_scenario_analysis(assumptions, scenarios)
elif active_page == "Covenant Headroom":
    render_covenants(scenarios, covenants)
elif active_page == "Investment Memo":
    render_memo(assumptions, metrics, strengths, risks, scenarios, covenants, recommendation)
