from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.acquisition_screening import AcquisitionAssumptions, screen_acquisition, screening_table
from src.board_memo import generate_board_memo
from src.financial_due_diligence import analyse_financial_due_diligence, diligence_summary_table
from src.integration_planning import generate_integration_plan, hundred_day_plan
from src.report_generator import markdown_to_text, save_report
from src.scenario_analysis import DEFAULT_SCENARIOS, run_scenarios, scenario_interpretation
from src.utils import money, multiple, pct


load_dotenv()

st.set_page_config(
    page_title="Strategic Finance & Acquisition Intelligence Platform",
    page_icon="S",
    layout="wide",
)

PAGES = [
    "Overview",
    "Acquisition Screening",
    "Financial Due Diligence",
    "Integration Planning",
    "Scenario Analysis",
    "Board Memo",
]
RAIL_ICONS = {
    "Overview": "O",
    "Acquisition Screening": "A",
    "Financial Due Diligence": "D",
    "Integration Planning": "I",
    "Scenario Analysis": "S",
    "Board Memo": "M",
}
CHART_COLOURS = ["#58d5b7", "#8da2ff", "#f6c36a", "#ef7e8e", "#9ad66f", "#6fb7ff"]


st.markdown(
    """
    <style>
    :root {
        --bg: #050b12;
        --panel: rgba(16, 30, 43, 0.78);
        --panel-2: rgba(22, 39, 53, 0.78);
        --line: rgba(130, 158, 177, 0.22);
        --line-strong: rgba(130, 158, 177, 0.40);
        --text: #eef6f5;
        --muted: #9fb0ba;
        --faint: #71838e;
        --accent: #58d5b7;
        --accent-2: #8da2ff;
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

    h1, h2, h3, h4, h5, h6, p, span, label, div {
        letter-spacing: 0;
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
        background:
            linear-gradient(180deg, rgba(9, 21, 31, 0.92), rgba(5, 13, 21, 0.92)),
            radial-gradient(circle at 80% 12%, rgba(88, 213, 183, 0.08), transparent 8rem);
        box-shadow: 20px 0 60px rgba(0, 0, 0, 0.18);
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-top: 5.85rem;
        gap: 0.82rem;
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
        transition: background 170ms ease, color 170ms ease, border-color 170ms ease;
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
        border-color: rgba(141, 162, 255, 0.07);
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
        gap: 1rem;
        padding: 0 2.1rem;
        border: 1px solid rgba(120, 151, 171, 0.22);
        border-radius: 0 14px 14px 0;
        background:
            radial-gradient(circle at 54% 0%, rgba(58, 111, 151, 0.12), transparent 30rem),
            linear-gradient(180deg, rgba(9, 20, 31, 0.82), rgba(6, 14, 22, 0.86));
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
        line-height: 1;
    }

    .reference-page-label {
        color: #aab7c0;
        font-size: 1.01rem;
        font-weight: 500;
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
        box-shadow: 0 0 0 3px rgba(88, 213, 183, 0.08);
    }

    .bottom-nav-radio {
        display: none;
    }

    div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Section Navigation"]) {
        position: fixed !important;
        left: 8.8rem !important;
        right: auto !important;
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
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
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
        box-shadow: inset 0 -2px 0 #58d5b7, 0 10px 26px rgba(0, 0, 0, 0.22), 0 8px 24px rgba(88, 213, 183, 0.08) !important;
    }

    .metric-card, .panel-card {
        border: 1px solid rgba(120, 151, 171, 0.18);
        border-radius: 12px;
        background:
            radial-gradient(circle at 86% 6%, rgba(76, 126, 164, 0.11), transparent 10rem),
            linear-gradient(180deg, rgba(18, 35, 49, 0.74), rgba(11, 24, 36, 0.80));
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
        font-size: 1.58rem;
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
        color: #a8bac3;
        line-height: 1.55;
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

    .stTextArea textarea,
    .stTextInput input,
    .stNumberInput input,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background: #0f1922 !important;
        color: #edf4f2 !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="popover"],
    div[data-baseweb="menu"] {
        color: #edf4f2 !important;
        background-color: #0f1922 !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(180deg, #5ee0bf 0%, #36b99a 100%) !important;
        color: #06100d !important;
        border: 1px solid rgba(133, 245, 215, 0.5) !important;
        border-radius: 8px !important;
        font-weight: 850 !important;
        box-shadow: 0 10px 28px rgba(42, 185, 153, 0.18) !important;
    }

    .stDataFrame, div[data-testid="stTable"] {
        border-radius: 12px !important;
        overflow: hidden;
    }

    @media (max-width: 980px) {
        .block-container {
            padding-left: 6.6rem !important;
        }
        div[data-testid="stRadio"]:has(div[role="radiogroup"][aria-label="Section Navigation"]) {
            left: 7.4rem !important;
            max-width: calc(100vw - 8rem);
            overflow-x: auto;
        }
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


def _style_chart(fig, height: int = 330):
    if fig.layout.title is None or fig.layout.title.text in {None, "undefined"}:
        fig.update_layout(title_text="")
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
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.12)", linecolor="rgba(148,163,184,0.25)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.12)", linecolor="rgba(148,163,184,0.25)")
    return fig


def metric_card(label: str, value: str, note: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(value)}</div>
        <div class="metric-note">{escape(note)}</div>
    </div>
    """


def bullets(items: list[str]) -> str:
    return '<ul class="flag-list">' + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ul>"


def format_metric_value(metric: str, value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if "Margin" in metric or "Growth" in metric or "Burden" in metric or "Break-even" in metric:
        return pct(numeric)
    if "Multiple" in metric or "Leverage" in metric or metric.startswith("EV /"):
        return multiple(numeric)
    if "Score" in metric:
        return f"{numeric:.1f}/100"
    if "Runway" in metric:
        return f"{numeric:.0f} months"
    if any(token in metric for token in ["Revenue", "EBITDA", "Debt", "Cash", "Value", "Capex", "Working", "FCF", "Liquidity"]):
        return money(numeric)
    return f"{numeric:.1f}"


def format_table(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    formatted["Value"] = [format_metric_value(metric, value) for metric, value in formatted[["Metric", "Value"]].to_numpy()]
    return formatted


def get_assumptions() -> AcquisitionAssumptions:
    return AcquisitionAssumptions(
        company_name=st.session_state.get("company_name", AcquisitionAssumptions.company_name),
        sector=st.session_state.get("sector", AcquisitionAssumptions.sector),
        revenue=float(st.session_state.get("revenue", AcquisitionAssumptions.revenue)),
        revenue_growth=float(st.session_state.get("revenue_growth", AcquisitionAssumptions.revenue_growth)),
        ebitda=float(st.session_state.get("ebitda", AcquisitionAssumptions.ebitda)),
        ebitda_margin=float(st.session_state.get("ebitda_margin", AcquisitionAssumptions.ebitda_margin)),
        cash=float(st.session_state.get("cash", AcquisitionAssumptions.cash)),
        debt=float(st.session_state.get("debt", AcquisitionAssumptions.debt)),
        purchase_price=float(st.session_state.get("purchase_price", AcquisitionAssumptions.purchase_price)),
        employees=int(st.session_state.get("employees", AcquisitionAssumptions.employees)),
        jurisdictions=int(st.session_state.get("jurisdictions", AcquisitionAssumptions.jurisdictions)),
        accounting_framework=st.session_state.get("accounting_framework", AcquisitionAssumptions.accounting_framework),
        strategic_rationale=st.session_state.get("strategic_rationale", AcquisitionAssumptions.strategic_rationale),
    )


def compute_context() -> tuple[AcquisitionAssumptions, dict[str, object], dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assumptions = get_assumptions()
    screening = screen_acquisition(assumptions)
    diligence = analyse_financial_due_diligence(
        assumptions,
        capex_pct_revenue=float(st.session_state.get("capex_pct_revenue", 0.055)),
        working_capital_pct_revenue=float(st.session_state.get("working_capital_pct_revenue", 0.025)),
        cash_conversion=float(st.session_state.get("cash_conversion", 0.72)),
    )
    scenarios = run_scenarios(assumptions, scenario_inputs())
    integration_plan = generate_integration_plan(assumptions)
    day_plan = hundred_day_plan(integration_plan)
    return assumptions, screening, diligence, scenarios, integration_plan, day_plan


def scenario_inputs() -> dict[str, dict[str, float]]:
    scenarios = {}
    for name, defaults in DEFAULT_SCENARIOS.items():
        scenarios[name] = {
            key: float(st.session_state.get(f"{name}_{key}", value))
            for key, value in defaults.items()
        }
    return scenarios


def render_navigation(active_page: str) -> None:
    st.markdown('<nav class="app-rail" aria-label="Primary"></nav>', unsafe_allow_html=True)
    st.radio(
        "Primary Navigation",
        PAGES,
        index=PAGES.index(active_page),
        key="rail_page",
        format_func=lambda page: RAIL_ICONS.get(page, page),
        label_visibility="collapsed",
        on_change=_sync_from_rail,
    )
    st.radio(
        "Section Navigation",
        PAGES,
        index=PAGES.index(active_page),
        key="bottom_page",
        horizontal=True,
        label_visibility="collapsed",
        on_change=_sync_from_bottom,
    )


def render_header(active_page: str, screening: dict[str, object]) -> None:
    status = screening["metrics"]["score_band"]
    st.markdown(
        f"""
        <header class="reference-header">
            <div class="reference-title-row">
                <div class="reference-title">Strategic Finance & Acquisition Intelligence Platform</div>
                <div class="reference-page-label">{escape(active_page)}</div>
            </div>
            <div class="status-pill"><span></span>{escape(str(status))}</div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_settings() -> None:
    defaults = AcquisitionAssumptions()
    with st.popover("Menu", help="Open acquisition assumptions"):
        st.markdown("#### Acquisition Assumptions")
        st.caption("Values are in EUR millions unless otherwise stated.")
        st.text_input("Company name", value=st.session_state.get("company_name", defaults.company_name), key="company_name")
        st.text_input("Sector", value=st.session_state.get("sector", defaults.sector), key="sector")
        c1, c2 = st.columns(2)
        c1.number_input("Revenue", min_value=0.0, value=float(st.session_state.get("revenue", defaults.revenue)), step=5.0, key="revenue")
        c2.number_input("Revenue growth", min_value=-0.5, max_value=2.0, value=float(st.session_state.get("revenue_growth", defaults.revenue_growth)), step=0.01, key="revenue_growth")
        c3, c4 = st.columns(2)
        c3.number_input("EBITDA", value=float(st.session_state.get("ebitda", defaults.ebitda)), step=2.0, key="ebitda")
        c4.number_input("EBITDA margin", min_value=-0.5, max_value=1.0, value=float(st.session_state.get("ebitda_margin", defaults.ebitda_margin)), step=0.01, key="ebitda_margin")
        c5, c6 = st.columns(2)
        c5.number_input("Cash", min_value=0.0, value=float(st.session_state.get("cash", defaults.cash)), step=2.0, key="cash")
        c6.number_input("Debt", min_value=0.0, value=float(st.session_state.get("debt", defaults.debt)), step=5.0, key="debt")
        st.number_input("Purchase price", min_value=0.0, value=float(st.session_state.get("purchase_price", defaults.purchase_price)), step=10.0, key="purchase_price")
        c7, c8 = st.columns(2)
        c7.number_input("Employees", min_value=1, value=int(st.session_state.get("employees", defaults.employees)), step=10, key="employees")
        c8.number_input("Jurisdictions", min_value=1, value=int(st.session_state.get("jurisdictions", defaults.jurisdictions)), step=1, key="jurisdictions")
        st.selectbox(
            "Accounting framework",
            ["IFRS", "US GAAP", "Mixed/unknown"],
            index=["IFRS", "US GAAP", "Mixed/unknown"].index(st.session_state.get("accounting_framework", defaults.accounting_framework)),
            key="accounting_framework",
        )
        st.text_area("Strategic rationale", value=st.session_state.get("strategic_rationale", defaults.strategic_rationale), height=110, key="strategic_rationale")
        st.markdown("#### Diligence Operating Assumptions")
        c9, c10, c11 = st.columns(3)
        c9.number_input("Capex as % revenue", min_value=0.0, max_value=0.5, value=float(st.session_state.get("capex_pct_revenue", 0.055)), step=0.005, key="capex_pct_revenue")
        c10.number_input("Working capital impact", min_value=-0.2, max_value=0.5, value=float(st.session_state.get("working_capital_pct_revenue", 0.025)), step=0.005, key="working_capital_pct_revenue")
        c11.number_input("Cash conversion", min_value=0.0, max_value=1.5, value=float(st.session_state.get("cash_conversion", 0.72)), step=0.02, key="cash_conversion")


def render_overview(assumptions, screening, diligence, scenarios, integration_plan) -> None:
    metrics = screening["metrics"]
    diligence_metrics = diligence["metrics"]
    kpis = [
        ("Target company", assumptions.company_name, assumptions.sector),
        ("Attractiveness score", f"{metrics['score']}/100", metrics["score_band"]),
        ("Revenue", money(assumptions.revenue), f"Growth {pct(assumptions.revenue_growth)}"),
        ("EBITDA", money(assumptions.ebitda), f"Margin {pct(metrics['ebitda_margin'])}"),
        ("Net debt", money(metrics["net_debt"]), f"Leverage {multiple(metrics['net_leverage'])}"),
        ("Liquidity runway", f"{diligence_metrics['Liquidity Runway Months']:.0f} months", "Cash and FCF view"),
    ]
    cols = st.columns(6)
    for col, (label, value, note) in zip(cols, kpis):
        col.markdown(metric_card(label, str(value), str(note)), unsafe_allow_html=True)

    o1, o2, o3 = st.columns([1.05, 0.95, 0.9], gap="medium")
    with o1:
        st.markdown('<div class="panel-title">Executive Finance Snapshot</div>', unsafe_allow_html=True)
        chart_df = pd.DataFrame(
            {
                "Metric": ["Revenue", "EBITDA", "Cash", "Debt", "Net Debt"],
                "Value": [assumptions.revenue, assumptions.ebitda, assumptions.cash, assumptions.debt, metrics["net_debt"]],
            }
        )
        st.plotly_chart(_style_chart(px.bar(chart_df, x="Metric", y="Value", title="Financial Position"), 300), use_container_width=True)
    with o2:
        st.markdown('<div class="panel-card"><div class="panel-title">Risk and Readiness</div>', unsafe_allow_html=True)
        st.markdown(
            bullets(
                [
                    f"Integration risk: {metrics['integration_complexity_score']}/100",
                    f"Reporting readiness: {metrics['accounting_alignment_risk']} accounting alignment risk",
                    f"Liquidity runway: {diligence_metrics['Liquidity Runway Months']:.0f} months",
                    f"Suggested next action: {screening['diligence_priorities'][0]}",
                ]
            ),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with o3:
        st.markdown('<div class="panel-card"><div class="panel-title">Suggested Next Action</div>', unsafe_allow_html=True)
        st.markdown(f"<p class='finance-note'>{escape(screening['strategic_fit_summary'])}</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Scenario Outlook")
    st.plotly_chart(_style_chart(px.bar(scenarios, x="Case", y=["Revenue", "EBITDA", "FCF"], barmode="group", title="Scenario Financial Outputs"), 330), use_container_width=True)


def render_acquisition_screening(assumptions, screening) -> None:
    st.markdown("### Acquisition Screening")
    st.caption("Enter target-company assumptions and evaluate strategic fit, valuation, leverage, and diligence priorities.")
    st.info("Use the menu button in the left rail to edit assumptions.")
    cols = st.columns(4)
    metrics = screening["metrics"]
    cols[0].markdown(metric_card("Attractiveness", f"{metrics['score']}/100", metrics["score_band"]), unsafe_allow_html=True)
    cols[1].markdown(metric_card("EV / Revenue", multiple(metrics["ev_revenue"]), money(metrics["enterprise_value"])), unsafe_allow_html=True)
    cols[2].markdown(metric_card("EV / EBITDA", multiple(metrics["ev_ebitda"]), "Entry valuation"), unsafe_allow_html=True)
    cols[3].markdown(metric_card("Net leverage", multiple(metrics["net_leverage"]), money(metrics["net_debt"])), unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="medium")
    with left:
        st.dataframe(format_table(screening_table(screening)), use_container_width=True, hide_index=True)
    with right:
        st.markdown('<div class="panel-card"><div class="panel-title">Risk Flags</div>', unsafe_allow_html=True)
        st.markdown(bullets(screening["risk_flags"]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("#### Strategic Fit Summary")
    st.markdown(f"<p class='finance-note'>{escape(screening['strategic_fit_summary'])}</p>", unsafe_allow_html=True)
    st.markdown("#### Recommended Diligence Priorities")
    st.markdown(bullets(screening["diligence_priorities"]), unsafe_allow_html=True)


def render_due_diligence(diligence) -> None:
    st.markdown("### Financial Due Diligence")
    st.caption("Quality of earnings, cash conversion, liquidity, net leverage, and reporting complexity.")
    metrics = diligence["metrics"]
    cols = st.columns(5)
    cols[0].markdown(metric_card("Free cash flow", money(metrics["Free Cash Flow"]), "EBITDA after capex and working capital"), unsafe_allow_html=True)
    cols[1].markdown(metric_card("Net debt", money(metrics["Net Debt"]), multiple(metrics["Net Leverage"])), unsafe_allow_html=True)
    cols[2].markdown(metric_card("Liquidity runway", f"{metrics['Liquidity Runway Months']:.0f} months", "Cash burn view"), unsafe_allow_html=True)
    cols[3].markdown(metric_card("Reporting complexity", f"{metrics['Reporting Complexity Score']:.1f}/100", "Entity and people footprint"), unsafe_allow_html=True)
    cols[4].markdown(metric_card("Accounting risk", str(metrics["Accounting Alignment Risk"]), "Policy alignment"), unsafe_allow_html=True)
    left, right = st.columns([1.05, 0.95], gap="medium")
    with left:
        st.dataframe(format_table(diligence_summary_table(diligence)), use_container_width=True, hide_index=True)
    with right:
        st.markdown('<div class="panel-card"><div class="panel-title">Quality of Earnings Flags</div>', unsafe_allow_html=True)
        st.markdown(bullets(diligence["quality_flags"]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("#### Interpretation")
    st.markdown(f"<p class='finance-note'>{escape(diligence['interpretation'])}</p>", unsafe_allow_html=True)


def render_integration_planning(integration_plan, day_plan) -> None:
    st.markdown("### Integration Planning")
    st.caption("Post-acquisition finance integration workstreams and 100-day execution cadence.")
    st.dataframe(integration_plan, use_container_width=True, hide_index=True)
    st.markdown("#### 100-Day Action Plan")
    st.dataframe(day_plan, use_container_width=True, hide_index=True)
    risk_counts = integration_plan["Risk Level"].value_counts().rename_axis("Risk Level").reset_index(name="Workstreams")
    st.plotly_chart(_style_chart(px.bar(risk_counts, x="Risk Level", y="Workstreams", title="Integration Risk Concentration"), 300), use_container_width=True)


def render_scenario_analysis(assumptions, scenarios) -> None:
    st.markdown("### Scenario Analysis")
    st.caption("Tune base, downside, and upside cases for growth, margin, capex, working capital, financing, synergies, and integration costs.")
    with st.expander("Scenario assumptions", expanded=False):
        for name, defaults in DEFAULT_SCENARIOS.items():
            st.markdown(f"#### {name}")
            c1, c2, c3, c4 = st.columns(4)
            c1.number_input(f"{name} revenue growth", value=float(st.session_state.get(f"{name}_revenue_growth", defaults["revenue_growth"])), step=0.01, key=f"{name}_revenue_growth")
            c2.number_input(f"{name} EBITDA margin", value=float(st.session_state.get(f"{name}_ebitda_margin", defaults["ebitda_margin"])), step=0.01, key=f"{name}_ebitda_margin")
            c3.number_input(f"{name} capex % revenue", value=float(st.session_state.get(f"{name}_capex_pct_revenue", defaults["capex_pct_revenue"])), step=0.005, key=f"{name}_capex_pct_revenue")
            c4.number_input(f"{name} working capital impact", value=float(st.session_state.get(f"{name}_working_capital_pct_revenue", defaults["working_capital_pct_revenue"])), step=0.005, key=f"{name}_working_capital_pct_revenue")
            c5, c6, c7, c8 = st.columns(4)
            c5.number_input(f"{name} interest rate", value=float(st.session_state.get(f"{name}_interest_rate", defaults["interest_rate"])), step=0.005, key=f"{name}_interest_rate")
            c6.number_input(f"{name} synergy savings", value=float(st.session_state.get(f"{name}_synergy_savings", defaults["synergy_savings"])), step=1.0, key=f"{name}_synergy_savings")
            c7.number_input(f"{name} integration costs", value=float(st.session_state.get(f"{name}_integration_costs", defaults["integration_costs"])), step=1.0, key=f"{name}_integration_costs")
            c8.number_input(f"{name} debt repayment/drawdown", value=float(st.session_state.get(f"{name}_debt_repayment", defaults["debt_repayment"])), step=1.0, key=f"{name}_debt_repayment")
    scenarios = run_scenarios(assumptions, scenario_inputs())
    st.dataframe(format_scenario_table(scenarios), use_container_width=True, hide_index=True)
    st.plotly_chart(_style_chart(px.line(scenarios, x="Case", y=["Revenue", "EBITDA", "FCF", "Liquidity"], markers=True, title="Scenario Range"), 350), use_container_width=True)
    st.markdown(f"<p class='finance-note'>{escape(scenario_interpretation(scenarios))}</p>", unsafe_allow_html=True)


def format_scenario_table(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for col in ["Revenue", "EBITDA", "FCF", "Net Debt", "Liquidity"]:
        output[col] = output[col].map(money)
    for col in ["Net Leverage"]:
        output[col] = output[col].map(multiple)
    for col in ["Integration Cost Burden", "Break-even EBITDA Margin"]:
        output[col] = output[col].map(pct)
    output["Downside Resilience Score"] = output["Downside Resilience Score"].map(lambda x: f"{x:.1f}/100")
    return output


def render_board_memo(assumptions, screening, diligence, scenarios, integration_plan) -> None:
    st.markdown("### Board Memo")
    st.caption("Generate a board-ready acquisition memo using deterministic local analysis, with optional API enhancement.")
    use_llm = st.checkbox("Use optional OpenAI-compatible API if configured", value=False)
    if st.button("Generate board memo"):
        memo = generate_board_memo(assumptions, screening, diligence, scenarios, integration_plan, use_llm=use_llm)
        st.session_state.board_memo = memo
        st.session_state.memo_path = save_report(memo, assumptions.company_name)
    memo = st.session_state.get("board_memo")
    if memo:
        st.markdown(memo)
        if st.session_state.get("memo_path"):
            st.caption(f"Saved locally to {st.session_state.memo_path}")
        c1, c2 = st.columns(2)
        c1.download_button("Download markdown", memo, file_name=f"{assumptions.company_name.lower().replace(' ', '_')}_board_memo.md", mime="text/markdown", use_container_width=True)
        c2.download_button("Download text", markdown_to_text(memo), file_name=f"{assumptions.company_name.lower().replace(' ', '_')}_board_memo.txt", mime="text/plain", use_container_width=True)
    else:
        preview = generate_board_memo(assumptions, screening, diligence, scenarios, integration_plan, use_llm=False)
        st.markdown(preview)


active_page = _active_page()
render_navigation(active_page)
render_settings()
assumptions, screening, diligence, scenarios, integration_plan, day_plan = compute_context()
render_header(active_page, screening)

if active_page == "Overview":
    render_overview(assumptions, screening, diligence, scenarios, integration_plan)
elif active_page == "Acquisition Screening":
    render_acquisition_screening(assumptions, screening)
elif active_page == "Financial Due Diligence":
    render_due_diligence(diligence)
elif active_page == "Integration Planning":
    render_integration_planning(integration_plan, day_plan)
elif active_page == "Scenario Analysis":
    render_scenario_analysis(assumptions, scenarios)
elif active_page == "Board Memo":
    render_board_memo(assumptions, screening, diligence, scenarios, integration_plan)
