from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.ai_research import generate_research_report
from src.backtester import run_backtest
from src.data_loader import get_price_data
from src.factors import FACTOR_DIRECTIONS, calculate_factors
from src.metrics import rolling_sharpe, rolling_volatility
from src.report_generator import backtest_summary_markdown, markdown_to_text, metrics_to_markdown, save_report
from src.risk import risk_summary
from src.utils import latest_rows, normalize_tickers


load_dotenv()

st.set_page_config(
    page_title="Quant Research Platform",
    page_icon="Q",
    layout="wide",
)

DEFAULT_TICKERS = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM, UNH, SPY"
CHART_COLORS = ["#58d5b7", "#8da2ff", "#f6c36a", "#ef7e8e", "#9ad66f", "#c792ea", "#6fb7ff", "#f08f5f"]


st.markdown(
    """
    <style>
    :root {
        --bg: #080d12;
        --bg-2: #0d141b;
        --panel: rgba(18, 27, 36, 0.92);
        --panel-2: rgba(22, 33, 44, 0.86);
        --panel-3: #111b24;
        --line: rgba(148, 163, 184, 0.22);
        --line-strong: rgba(148, 163, 184, 0.38);
        --text: #edf4f2;
        --muted: #9baab4;
        --faint: #6f808b;
        --accent: #58d5b7;
        --accent-2: #8da2ff;
        --warn: #f6c36a;
        --danger: #ef7e8e;
        --success: #8de0a6;
        --shadow: 0 18px 55px rgba(0, 0, 0, 0.35);
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, system-ui, sans-serif;
    }

    html, body, .stApp, div[data-testid="stAppViewContainer"] {
        overflow-x: hidden;
    }

    .stApp {
        color: var(--text);
        background:
            radial-gradient(circle at 18% 3%, rgba(44, 176, 255, 0.13), transparent 28rem),
            radial-gradient(circle at 70% -4%, rgba(127, 86, 217, 0.14), transparent 34rem),
            radial-gradient(circle at 88% 18%, rgba(88, 213, 183, 0.12), transparent 28rem),
            linear-gradient(180deg, #050a10 0%, #071019 45%, #05090f 100%);
    }

    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    footer {
        visibility: hidden;
        height: 0;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2.4rem;
        max-width: 1480px;
    }

    h1, h2, h3, h4, h5, h6, p, span, label, div {
        letter-spacing: 0;
    }

    h1 {
        font-size: 1.8rem !important;
        line-height: 1.1 !important;
        margin: 0 0 0.35rem 0 !important;
        color: var(--text);
    }

    h2, h3 {
        color: var(--text);
    }

    p, .stMarkdown, .stCaption, label, [data-testid="stMarkdownContainer"] {
        color: var(--muted);
    }

    section[data-testid="stSidebar"],
    div[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 12% 0%, rgba(88, 213, 183, 0.10), transparent 14rem),
            linear-gradient(180deg, #080d13 0%, #0b1119 100%);
        border-right: 1px solid var(--line);
        transition: transform 280ms cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 220ms ease;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] {
        position: fixed !important;
        inset: 0 auto 0 0 !important;
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
        height: 100vh !important;
        flex-shrink: 0 !important;
        transform: translateX(-286px) !important;
        z-index: 1002 !important;
        opacity: 1 !important;
        pointer-events: auto !important;
        overflow: visible !important;
        box-shadow: none;
        transition: transform 300ms cubic-bezier(0.2, 0.8, 0.2, 1) 180ms, box-shadow 220ms ease 180ms;
    }

    section[data-testid="stSidebar"][aria-expanded="false"]:hover {
        transform: translateX(0) !important;
        box-shadow: 28px 0 72px rgba(0, 0, 0, 0.46), 0 0 42px rgba(88, 213, 183, 0.13);
        transition-delay: 0ms;
    }

    section[data-testid="stSidebar"][aria-expanded="false"]::after {
        content: "";
        position: fixed;
        top: 74px;
        right: -12px;
        width: 12px;
        height: min(68vh, 560px);
        border-radius: 0 999px 999px 0;
        background:
            linear-gradient(180deg, rgba(88, 213, 183, 0.94), rgba(141, 162, 255, 0.72) 55%, rgba(88, 213, 183, 0.55)),
            rgba(8, 13, 18, 0.92);
        box-shadow: 0 0 22px rgba(88, 213, 183, 0.72), 0 0 56px rgba(141, 162, 255, 0.30);
        opacity: 0.92;
        pointer-events: none;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stSidebarContent"],
    section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stSidebarHeader"],
    section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stSidebarUserContent"] {
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
        visibility: visible !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stSidebarContent"] {
        opacity: 0;
        transition: opacity 150ms ease;
    }

    section[data-testid="stSidebar"][aria-expanded="false"]:hover div[data-testid="stSidebarContent"] {
        opacity: 1;
        transition-delay: 110ms;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stSidebarHeader"],
    section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stSidebarUserContent"] {
        padding-left: 20px !important;
        padding-right: 20px !important;
    }

    section[data-testid="stSidebar"][aria-expanded="false"] div[data-testid="stSidebarCollapseButton"] {
        visibility: visible !important;
    }

    button[data-testid="stExpandSidebarButton"] {
        visibility: visible !important;
        opacity: 1 !important;
        position: fixed !important;
        left: 8px !important;
        top: 14px !important;
        z-index: 1003 !important;
        width: 34px !important;
        height: 34px !important;
        border: 1px solid rgba(88, 213, 183, 0.42) !important;
        border-radius: 10px !important;
        background: rgba(8, 16, 24, 0.82) !important;
        color: #ddfff8 !important;
        box-shadow: 0 0 26px rgba(88, 213, 183, 0.28), 0 12px 38px rgba(0, 0, 0, 0.38) !important;
        backdrop-filter: blur(14px);
    }

    section[data-testid="stSidebar"][aria-expanded="false"]:hover + header button[data-testid="stExpandSidebarButton"] {
        opacity: 0.2 !important;
    }

    div[data-testid="stSidebar"] * {
        color: var(--text);
    }

    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #b7c5cc;
    }

    div[data-testid="stSidebar"] h1 {
        font-size: 1.15rem !important;
        margin-top: 0.2rem !important;
    }

    .side-brand {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 0.25rem 0 0.9rem 0;
    }

    .brand-mark {
        width: 2.15rem;
        height: 2.15rem;
        display: grid;
        place-items: center;
        border-radius: 12px;
        border: 1px solid rgba(88, 213, 183, 0.35);
        background:
            radial-gradient(circle at 35% 25%, rgba(88, 213, 183, 0.36), transparent 0.6rem),
            linear-gradient(135deg, rgba(22, 36, 50, 0.9), rgba(8, 14, 22, 0.9));
        color: #ddfff8 !important;
        font-weight: 900;
        box-shadow: 0 0 28px rgba(88, 213, 183, 0.14);
    }

    .brand-copy {
        display: flex;
        flex-direction: column;
        gap: 0.08rem;
    }

    .brand-title {
        color: #edf4f2 !important;
        font-size: 1rem;
        font-weight: 850;
        line-height: 1.1;
    }

    .brand-subtitle {
        color: #7f919b !important;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .sidebar-section {
        color: #dce7e4;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin: 1.1rem 0 0.35rem 0;
    }

    .ticker-chip {
        display: inline-flex;
        align-items: center;
        border: 1px solid rgba(88, 213, 183, 0.28);
        background: rgba(88, 213, 183, 0.08);
        color: #dff8f2 !important;
        border-radius: 999px;
        padding: 0.16rem 0.5rem;
        margin: 0.12rem 0.15rem 0.12rem 0;
        font-size: 0.72rem;
        font-weight: 650;
    }

    .sidebar-note {
        color: #81939d !important;
        font-size: 0.72rem;
        margin-top: 0.45rem;
        line-height: 1.35;
    }

    .stTextArea textarea,
    .stTextInput input,
    .stNumberInput input,
    .stDateInput input,
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        background: #0f1922 !important;
        color: #edf4f2 !important;
        border: 1px solid var(--line-strong) !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    .stTextArea textarea::placeholder,
    .stTextInput input::placeholder,
    .stNumberInput input::placeholder {
        color: #7d8c96 !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="select"] input,
    div[data-baseweb="popover"],
    div[data-baseweb="menu"] {
        color: #edf4f2 !important;
        background-color: #0f1922 !important;
    }

    [data-baseweb="checkbox"] span,
    [data-testid="stCheckbox"] label p {
        color: #dbe7e4 !important;
    }

    .stButton > button,
    .stDownloadButton > button {
        background: linear-gradient(180deg, #5ee0bf 0%, #36b99a 100%) !important;
        color: #06100d !important;
        border: 1px solid rgba(133, 245, 215, 0.5) !important;
        border-radius: 8px !important;
        font-weight: 800 !important;
        box-shadow: 0 10px 28px rgba(42, 185, 153, 0.18) !important;
        transition: transform 160ms ease, border-color 160ms ease, filter 160ms ease !important;
    }

    .stButton > button *,
    .stDownloadButton > button * {
        color: #06100d !important;
        font-weight: 850 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        color: #06100d !important;
        filter: brightness(1.05);
        transform: translateY(-1px);
    }

    .stButton > button:focus,
    .stDownloadButton > button:focus {
        color: #06100d !important;
        outline: 2px solid rgba(141, 162, 255, 0.65) !important;
        outline-offset: 2px !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.28rem;
        background: rgba(4, 9, 15, 0.72);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.24rem;
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.32);
        width: fit-content;
    }

    .stTabs [data-baseweb="tab"] {
        height: 2.18rem;
        border-radius: 999px;
        padding: 0 0.9rem;
        background: rgba(17, 27, 38, 0.84);
        border: 1px solid rgba(148, 163, 184, 0.14);
        color: #c9d7dc !important;
        font-weight: 750;
    }

    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #c9d7dc !important;
        font-weight: 750 !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(28, 42, 55, 0.98);
        border-color: rgba(88, 213, 183, 0.35);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, rgba(88, 213, 183, 0.20), rgba(88, 213, 183, 0.10)) !important;
        border-color: rgba(88, 213, 183, 0.58) !important;
        box-shadow: inset 0 0 0 1px rgba(88, 213, 183, 0.18);
    }

    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #effffb !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: var(--accent) !important;
        height: 2px !important;
        border-radius: 999px !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(110, 147, 176, 0.25);
        background:
            linear-gradient(135deg, rgba(15, 25, 36, 0.94), rgba(6, 11, 18, 0.92)),
            radial-gradient(circle at 100% 0%, rgba(45, 159, 255, 0.18), transparent 23rem);
        border-radius: 16px;
        padding: 0.82rem 1rem;
        box-shadow: var(--shadow);
        margin-bottom: 0.62rem;
    }

    .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background:
            linear-gradient(110deg, transparent 0%, rgba(88, 213, 183, 0.04) 38%, transparent 64%),
            repeating-linear-gradient(90deg, rgba(255,255,255,0.035) 0 1px, transparent 1px 84px);
        opacity: 0.45;
    }

    .hero-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: start;
        gap: 1rem;
    }

    .hero-title {
        color: var(--text);
        font-size: 1.52rem;
        font-weight: 900;
        line-height: 1.1;
        margin-bottom: 0.25rem;
    }

    .hero-subtitle {
        color: #adc0c7;
        font-size: 0.86rem;
        max-width: 76ch;
    }

    .status-cluster {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        justify-content: flex-end;
        max-width: 27rem;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.34rem 0.58rem;
        border-radius: 999px;
        border: 1px solid rgba(88, 213, 183, 0.35);
        background: rgba(88, 213, 183, 0.10);
        color: #dff8f2;
        font-size: 0.72rem;
        font-weight: 760;
        white-space: nowrap;
    }

    .status-dot {
        width: 0.52rem;
        height: 0.52rem;
        border-radius: 999px;
        background: var(--accent);
        box-shadow: 0 0 18px rgba(88, 213, 183, 0.72);
    }

    .status-dot.idle {
        background: #687986;
        box-shadow: none;
    }

    .activity-indicator {
        position: fixed;
        top: 0.86rem;
        right: 5.1rem;
        z-index: 999;
        display: inline-flex;
        align-items: center;
        gap: 0.52rem;
        min-height: 2rem;
        padding: 0.34rem 0.68rem 0.34rem 0.46rem;
        border: 1px solid rgba(88, 213, 183, 0.32);
        border-radius: 999px;
        background:
            radial-gradient(circle at 20% 25%, rgba(88, 213, 183, 0.16), transparent 1.2rem),
            linear-gradient(180deg, rgba(11, 21, 31, 0.93), rgba(6, 12, 18, 0.88));
        color: #dff8f2;
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.32), 0 0 28px rgba(88, 213, 183, 0.12);
        backdrop-filter: blur(18px);
        animation: activityIn 260ms ease both;
        pointer-events: none;
    }

    .activity-indicator.processing {
        border-color: rgba(88, 213, 183, 0.58);
        box-shadow: 0 16px 42px rgba(0, 0, 0, 0.36), 0 0 34px rgba(88, 213, 183, 0.24);
    }

    .activity-orbit {
        position: relative;
        width: 1.38rem;
        height: 1.38rem;
        display: grid;
        place-items: center;
        border-radius: 999px;
        border: 1px solid rgba(141, 162, 255, 0.34);
        background: rgba(88, 213, 183, 0.08);
        color: #effffb;
        font-size: 0.86rem;
        line-height: 1;
    }

    .activity-indicator.processing .activity-orbit {
        animation: activityPulse 1.25s ease-in-out infinite;
    }

    .activity-indicator.processing .activity-orbit::before {
        content: "🏃";
        animation: activityIconCycle 5s steps(1, end) infinite;
    }

    .activity-indicator.ready .activity-orbit::before {
        content: "✓";
        color: #8de0a6;
        font-weight: 900;
    }

    .activity-copy {
        display: flex;
        flex-direction: column;
        gap: 0.02rem;
    }

    .activity-label {
        color: #effffb;
        font-size: 0.75rem;
        font-weight: 850;
        line-height: 1.1;
        white-space: nowrap;
    }

    .activity-sub {
        color: #8fa1ac;
        font-size: 0.62rem;
        font-weight: 720;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }

    .activity-indicator.processing .activity-sub {
        color: #9debd7;
    }

    @keyframes activityIconCycle {
        0%, 19% { content: "🏃"; }
        20%, 39% { content: "🏊"; }
        40%, 59% { content: "🚴"; }
        60%, 79% { content: "🧠"; }
        80%, 100% { content: "⚙️"; }
    }

    @keyframes activityPulse {
        0%, 100% {
            transform: scale(1);
            box-shadow: 0 0 0 rgba(88, 213, 183, 0);
        }
        50% {
            transform: scale(1.06);
            box-shadow: 0 0 24px rgba(88, 213, 183, 0.42);
        }
    }

    @keyframes activityIn {
        from {
            opacity: 0;
            transform: translateY(-6px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .metric-card, .intel-card {
        border: 1px solid var(--line);
        background:
            radial-gradient(circle at 100% 0%, rgba(88, 213, 183, 0.08), transparent 9rem),
            linear-gradient(180deg, rgba(18, 29, 41, 0.92), rgba(9, 16, 24, 0.94));
        border-radius: 14px;
        padding: 0.68rem 0.78rem;
        min-height: 4.45rem;
        box-shadow: 0 14px 42px rgba(0, 0, 0, 0.24);
    }

    .metric-label {
        color: var(--faint);
        font-size: 0.69rem;
        font-weight: 800;
        letter-spacing: 0.075em;
        text-transform: uppercase;
        margin-bottom: 0.36rem;
    }

    .metric-value {
        color: var(--text);
        font-size: 1.06rem;
        font-weight: 850;
        line-height: 1.15;
    }

    .metric-note {
        color: var(--muted);
        font-size: 0.68rem;
        margin-top: 0.28rem;
    }

    .kpi-strip {
        margin-bottom: 0.52rem;
    }

    .command-grid {
        display: grid;
        grid-template-columns: minmax(26rem, 1.02fr) minmax(28rem, 1.35fr);
        gap: 0.75rem;
        margin: 0.05rem 0 0.7rem 0;
    }

    .copilot-panel, .market-panel {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(88, 213, 183, 0.24);
        border-radius: 16px;
        padding: 0.82rem;
        min-height: 10.8rem;
        background:
            radial-gradient(circle at 8% 12%, rgba(127, 86, 217, 0.23), transparent 8rem),
            radial-gradient(circle at 95% 0%, rgba(88, 213, 183, 0.12), transparent 16rem),
            linear-gradient(180deg, rgba(15, 27, 40, 0.94), rgba(7, 13, 21, 0.94));
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.36);
    }

    .copilot-panel::before {
        content: "";
        position: absolute;
        inset: 0;
        pointer-events: none;
        background: linear-gradient(120deg, transparent, rgba(88, 213, 183, 0.045), transparent);
    }

    .copilot-head {
        display: flex;
        align-items: center;
        gap: 0.9rem;
        margin-bottom: 0.62rem;
    }

    .copilot-orb {
        width: 2.85rem;
        height: 2.85rem;
        border-radius: 999px;
        background:
            radial-gradient(circle at 35% 32%, rgba(255,255,255,0.86), transparent 0.25rem),
            radial-gradient(circle at 50% 48%, rgba(88, 213, 183, 0.75), transparent 0.8rem),
            radial-gradient(circle at 62% 35%, rgba(141, 162, 255, 0.86), transparent 1.25rem),
            radial-gradient(circle at 45% 68%, rgba(199, 146, 234, 0.80), transparent 1.3rem),
            #111b24;
        border: 1px solid rgba(182, 222, 255, 0.52);
        box-shadow: 0 0 40px rgba(141, 162, 255, 0.34), 0 0 70px rgba(88, 213, 183, 0.14);
    }

    .copilot-title {
        color: #edf4f2;
        font-size: 1rem;
        font-weight: 900;
    }

    .copilot-subtitle {
        color: #91a6b1;
        font-size: 0.74rem;
        margin-top: 0.1rem;
    }

    .copilot-metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.48rem;
        margin-bottom: 0.58rem;
    }

    .copilot-item {
        border: 1px solid rgba(148, 163, 184, 0.17);
        border-radius: 11px;
        background: rgba(8, 15, 23, 0.52);
        padding: 0.5rem;
    }

    .copilot-label, .market-label {
        color: #8fa1ac;
        font-size: 0.66rem;
        font-weight: 850;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .copilot-value, .market-value {
        color: #edf4f2;
        font-size: 0.92rem;
        font-weight: 850;
        margin-top: 0.22rem;
    }

    .ai-summary {
        color: #c8d8dd;
        font-size: 0.82rem;
        line-height: 1.48;
        border-top: 1px solid rgba(148, 163, 184, 0.16);
        padding-top: 0.58rem;
    }

    .market-panel {
        border-color: rgba(141, 162, 255, 0.22);
        background:
            radial-gradient(circle at 92% 6%, rgba(45, 159, 255, 0.13), transparent 16rem),
            linear-gradient(180deg, rgba(14, 24, 36, 0.92), rgba(8, 14, 22, 0.94));
    }

    .market-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.65rem;
    }

    .market-callout {
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 12px;
        padding: 0.56rem 0.62rem;
        background: rgba(8, 15, 23, 0.55);
    }

    .market-action {
        margin-top: 0.55rem;
        border: 1px solid rgba(88, 213, 183, 0.26);
        border-radius: 12px;
        padding: 0.58rem 0.68rem;
        background: rgba(88, 213, 183, 0.08);
    }

    .market-action .market-value {
        color: #dff8f2;
    }

    .intel-title {
        color: #dce8e5;
        font-size: 0.73rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        margin-bottom: 0.35rem;
    }

    .intel-value {
        color: var(--text);
        font-size: 0.98rem;
        font-weight: 820;
    }

    .intel-note {
        color: var(--muted);
        font-size: 0.72rem;
        margin-top: 0.25rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid var(--line);
        background: rgba(16, 25, 34, 0.92);
        border-radius: 12px;
        padding: 0.72rem 0.8rem;
    }

    div[data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-weight: 750;
    }

    div[data-testid="stMetricValue"] {
        color: var(--text);
        font-size: 1.25rem;
    }

    div[data-testid="stDataFrame"],
    div[data-testid="stTable"] {
        border: 1px solid var(--line);
        border-radius: 10px;
        overflow: hidden;
    }

    div[data-testid="stAlert"] {
        background: rgba(21, 31, 42, 0.95);
        border: 1px solid var(--line);
        color: var(--text);
    }

    code {
        color: #dff8f2 !important;
        background: rgba(88, 213, 183, 0.08) !important;
    }

    @media (max-width: 1050px) {
        .command-grid {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 720px) {
        .copilot-metrics,
        .market-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 1.3rem;
        }
    }

    /* Reference-dashboard shell */
    html, body, .stApp, div[data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 50% -18%, rgba(34, 108, 151, 0.18), transparent 36rem),
            radial-gradient(circle at 95% 8%, rgba(42, 196, 179, 0.08), transparent 28rem),
            linear-gradient(180deg, #07111a 0%, #061019 48%, #040a10 100%) !important;
    }

    .block-container {
        max-width: 1496px !important;
        padding: 1.45rem 1.55rem 6.2rem 7.35rem !important;
    }

    section[data-testid="stSidebar"] {
        display: none !important;
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

    .rail-icon {
        width: 72px;
        height: 52px;
        display: grid;
        place-items: center;
        color: #b7c5cc;
        font-size: 0;
        font-weight: 820;
        border-radius: 0 999px 999px 0;
        border: 1px solid transparent;
        transition: background 170ms ease, color 170ms ease, border-color 170ms ease;
    }

    .rail-icon::before {
        content: attr(title);
        width: 28px;
        height: 28px;
        display: grid;
        place-items: center;
        border: 1px solid currentColor;
        border-radius: 8px;
        font-size: 0.72rem;
        line-height: 1;
        letter-spacing: 0;
        opacity: 0.95;
    }

    .rail-icon[title="Overview"]::before { content: "O"; }
    .rail-icon[title="Data"]::before { content: "D"; }
    .rail-icon[title="Factors"]::before { content: "F"; }
    .rail-icon[title="Backtest"]::before { content: "B"; }
    .rail-icon[title="Risk Analytics"]::before { content: "R"; }
    .rail-icon[title="AI Research Report"]::before { content: "AI"; }

    .rail-icon.active {
        color: #f0fffb;
        border-color: rgba(141, 162, 255, 0.07);
        background: linear-gradient(90deg, rgba(92, 116, 143, 0.26), rgba(65, 85, 106, 0.18));
        box-shadow: inset -1px 0 0 rgba(88, 213, 183, 0.45);
    }

    .rail-icon.active::after {
        content: "";
        position: absolute;
        left: 87px;
        width: 2px;
        height: 34px;
        border-radius: 999px;
        background: #58d5b7;
        box-shadow: 0 0 16px rgba(88, 213, 183, 0.62);
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
        line-height: 1 !important;
    }

    div[data-testid="stPopover"] button * {
        display: none !important;
    }

    div[data-testid="stPopover"] button::before {
        content: "\\2630";
        font-size: 1.55rem;
        line-height: 1;
    }

    div[data-testid="stPopover"] button:hover {
        color: #effffb !important;
        background: rgba(148, 163, 184, 0.08) !important;
        transform: none !important;
    }

    .reference-shell {
        min-height: 0;
        margin-top: -4rem;
        border: 1px solid rgba(120, 151, 171, 0.22);
        border-radius: 0 14px 14px 0;
        background:
            radial-gradient(circle at 54% 0%, rgba(58, 111, 151, 0.12), transparent 30rem),
            linear-gradient(180deg, rgba(9, 20, 31, 0.82), rgba(6, 14, 22, 0.86));
        box-shadow: 0 30px 90px rgba(0, 0, 0, 0.26);
        overflow: visible;
    }

    div[data-testid="stTabs"] {
        margin-top: 0;
    }

    .reference-header {
        min-height: 78px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0 2.1rem;
        border-bottom: 1px solid rgba(120, 151, 171, 0.16);
        background: rgba(7, 16, 25, 0.34);
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

    .reference-market-pill {
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

    .reference-market-pill span {
        width: 0.58rem;
        height: 0.58rem;
        border-radius: 999px;
        background: #f0f7f5;
        box-shadow: 0 0 0 3px rgba(88, 213, 183, 0.08);
    }

    .reference-body {
        padding: 1.38rem 1.72rem 2.3rem 1.72rem;
    }

    .overview-kpis {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 1rem;
        margin-bottom: 1.18rem;
    }

    .ref-card {
        border: 1px solid rgba(120, 151, 171, 0.18);
        border-radius: 12px;
        background:
            radial-gradient(circle at 86% 6%, rgba(76, 126, 164, 0.11), transparent 10rem),
            linear-gradient(180deg, rgba(18, 35, 49, 0.74), rgba(11, 24, 36, 0.80));
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.025);
    }

    .ref-kpi {
        min-height: 126px;
        padding: 1.35rem 1.45rem;
    }

    .ref-kpi-label {
        color: #bcc8cf;
        font-size: 0.93rem;
        font-weight: 500;
        margin-bottom: 1.06rem;
    }

    .ref-kpi-value {
        color: #f6f8f8;
        font-size: 1.65rem;
        font-weight: 850;
        line-height: 1;
        margin-bottom: 0.75rem;
    }

    .ref-kpi-note {
        color: #aab6be;
        font-size: 0.92rem;
    }

    .overview-grid {
        display: grid;
        grid-template-columns: 1.18fr 0.82fr 0.96fr;
        gap: 1rem;
        margin-bottom: 1.18rem;
    }

    .overview-bottom-grid {
        display: grid;
        grid-template-columns: 0.74fr 1.16fr;
        gap: 1rem;
    }

    .panel-card {
        min-height: 292px;
        padding: 1.7rem 1.7rem 1.45rem;
    }

    .panel-card.short {
        min-height: 128px;
    }

    .panel-title {
        color: #f3f6f7;
        font-size: 1.12rem;
        font-weight: 820;
        margin-bottom: 1.35rem;
    }

    .empty-chart {
        height: 206px;
        position: relative;
        display: grid;
        place-items: center;
        color: #c6d1d7;
        font-size: 1.02rem;
        background:
            linear-gradient(rgba(148, 163, 184, 0.045) 1px, transparent 1px),
            linear-gradient(90deg, rgba(148, 163, 184, 0.035) 1px, transparent 1px);
        background-size: 100% 41px, 84px 100%;
        border-radius: 8px;
    }

    .chart-legend {
        display: inline-flex;
        gap: 1.2rem;
        color: #b8c5cc;
        font-size: 0.82rem;
        margin-bottom: 0.8rem;
    }

    .legend-line {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }

    .legend-line::before {
        content: "";
        width: 18px;
        height: 3px;
        border-radius: 999px;
        background: #58d5b7;
    }

    .legend-line.spy::before {
        background: rgba(120, 151, 171, 0.58);
    }

    .factor-row, .risk-row, .action-row {
        display: grid;
        grid-template-columns: 1fr minmax(100px, 0.52fr) auto;
        align-items: center;
        gap: 1rem;
        margin: 1.25rem 0;
        color: #c4d0d7;
        font-size: 1rem;
    }

    .factor-bar {
        height: 18px;
        border-radius: 999px;
        background: rgba(120, 151, 171, 0.13);
        overflow: hidden;
    }

    .factor-bar span {
        display: block;
        height: 100%;
        width: var(--w, 0%);
        border-radius: inherit;
        background: linear-gradient(90deg, rgba(88, 213, 183, 0.22), rgba(88, 213, 183, 0.72));
    }

    .risk-toggle {
        width: 18px;
        height: 12px;
        border-radius: 999px;
        background: rgba(120, 151, 171, 0.12);
    }

    .action-row {
        display: flex;
        gap: 2.4rem;
        flex-wrap: wrap;
        margin-top: 0.4rem;
    }

    .action-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.72rem;
        color: #cbd5da;
        font-size: 0.94rem;
        white-space: nowrap;
    }

    .action-icon {
        width: 29px;
        height: 29px;
        display: grid;
        place-items: center;
        border: 1px solid rgba(180, 195, 204, 0.42);
        border-radius: 7px;
        color: #dbe5e8;
        font-size: 1rem;
    }

    .settings-note {
        color: #8ea0aa;
        font-size: 0.82rem;
        line-height: 1.38;
        margin: 0.3rem 0 0.6rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        position: fixed !important;
        left: 8.8rem !important;
        right: auto !important;
        bottom: 2.15rem !important;
        z-index: 920 !important;
        gap: 1rem !important;
        padding: 0 !important;
        width: auto !important;
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: 52px !important;
        min-width: 96px;
        padding: 0 1.55rem !important;
        border-radius: 11px !important;
        background: rgba(14, 27, 39, 0.72) !important;
        border: 1px solid rgba(120, 151, 171, 0.16) !important;
        color: #bac6cd !important;
        font-size: 1rem !important;
        font-weight: 620 !important;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.02);
    }

    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #bac6cd !important;
        font-size: 1rem !important;
        font-weight: 620 !important;
    }

    .stTabs [aria-selected="true"] {
        color: #f5f8f8 !important;
        background: linear-gradient(180deg, rgba(35, 57, 75, 0.78), rgba(16, 31, 44, 0.82)) !important;
        border-color: rgba(88, 213, 183, 0.22) !important;
        box-shadow: inset 0 -2px 0 #58d5b7, 0 10px 26px rgba(0, 0, 0, 0.22), 0 8px 24px rgba(88, 213, 183, 0.08) !important;
    }

    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: #f5f8f8 !important;
        font-weight: 760 !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    .activity-indicator {
        top: 3.05rem;
        right: 2.95rem;
        transform: scale(0.92);
        transform-origin: top right;
    }

    .activity-indicator.ready {
        display: none;
    }

    div[data-testid="stElementContainer"]:has(.activity-indicator),
    div[data-testid="stElementContainer"]:has(.app-rail),
    div[data-testid="stElementContainer"]:has(div[data-testid="stPopover"]) {
        height: 0 !important;
        min-height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        overflow: visible !important;
    }

    .panel-card.short {
        min-height: 118px;
        padding: 1.25rem 1.5rem;
    }

    .action-row {
        gap: 1.75rem;
        margin-top: 0.15rem;
    }

    .action-chip {
        font-size: 0.88rem;
    }

    .action-icon {
        width: 27px;
        height: 27px;
        font-size: 0.78rem;
    }

    @media (max-width: 1180px) {
        .overview-kpis,
        .overview-grid,
        .overview-bottom-grid {
            grid-template-columns: 1fr;
        }
        .block-container {
            padding-left: 6.7rem !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            left: 7.4rem !important;
            gap: 0.55rem !important;
            overflow-x: auto;
            max-width: calc(100vw - 8.5rem);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _metrics_frame(metrics: dict[str, float]) -> pd.DataFrame:
    frame = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
    pct_like = {
        "CAGR",
        "Annualized Volatility",
        "Max Drawdown",
        "Win Rate",
        "Average Turnover",
        "Alpha vs Benchmark",
        "Daily VaR 95%",
        "Daily CVaR 95%",
        "Worst Day",
        "Best Day",
        "Current Drawdown",
    }
    formatted = []
    for _, row in frame.iterrows():
        value = row["Value"]
        if pd.isna(value):
            formatted.append("n/a")
        elif row["Metric"] in pct_like:
            formatted.append(f"{value:.2%}")
        else:
            formatted.append(f"{value:.2f}")
    frame["Value"] = formatted
    frame["Metric"] = frame["Metric"].replace(
        {
            "Annualized Volatility": "Annualised Volatility",
            "Alpha vs Benchmark": "Alpha vs Benchmark",
        }
    )
    return frame


def _money(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"${value:,.0f}"


def _pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def _need_data() -> bool:
    return "price_data" not in st.session_state or st.session_state.price_data.empty


def _style_chart(fig, height: int = 360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0c141c",
        font={"color": "#dce7e4", "family": "Segoe UI, Inter, system-ui, sans-serif", "size": 12},
        title={"font": {"size": 15, "color": "#edf4f2"}, "x": 0.01},
        colorway=CHART_COLORS,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"color": "#b7c5cc"},
        },
        margin={"l": 36, "r": 18, "t": 54, "b": 36},
        hoverlabel={"bgcolor": "#111b24", "font_color": "#edf4f2", "bordercolor": "#334655"},
    )
    fig.update_xaxes(
        gridcolor="rgba(148,163,184,0.12)",
        linecolor="rgba(148,163,184,0.25)",
        tickfont={"color": "#aebdc4"},
        title_font={"color": "#b7c5cc"},
        zerolinecolor="rgba(148,163,184,0.16)",
    )
    fig.update_yaxes(
        gridcolor="rgba(148,163,184,0.12)",
        linecolor="rgba(148,163,184,0.25)",
        tickfont={"color": "#aebdc4"},
        title_font={"color": "#b7c5cc"},
        zerolinecolor="rgba(148,163,184,0.16)",
    )
    return fig


def _card(label: str, value: str, note: str = "Awaiting data") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _status_label() -> tuple[str, str]:
    if _need_data():
        return "Idle", "idle"
    rows = len(st.session_state.price_data)
    tickers_loaded = st.session_state.price_data["ticker"].nunique()
    return f"Market data ready, {tickers_loaded} tickers, {rows:,} rows", "ready"


def _activity_html(mode: str, label: str) -> str:
    mode_class = "processing" if mode == "processing" else "ready"
    subcopy = "Live task" if mode == "processing" else "System status"
    return f"""
        <div class="activity-indicator {mode_class}">
            <div class="activity-orbit"></div>
            <div class="activity-copy">
                <div class="activity-label">{escape(label)}</div>
                <div class="activity-sub">{subcopy}</div>
            </div>
        </div>
    """


def _update_activity(slot, mode: str, label: str) -> None:
    st.session_state.activity_mode = mode
    st.session_state.activity_label = label
    slot.markdown(_activity_html(mode, label), unsafe_allow_html=True)


def _render_rail() -> None:
    st.markdown(
        """
        <nav class="app-rail" aria-label="Primary">
            <div class="rail-icon active" title="Overview">O</div>
            <div class="rail-icon" title="Data">D</div>
            <div class="rail-icon" title="Factors">F</div>
            <div class="rail-icon" title="Backtest">B</div>
            <div class="rail-icon" title="Risk Analytics">R</div>
            <div class="rail-icon" title="AI Research Report">AI</div>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def _overview_status() -> tuple[str, str]:
    if _need_data():
        return "Awaiting Data", "Load data to begin"
    if st.session_state.get("backtest_result"):
        return "Backtest Ready", "Risk analytics available"
    return "Data Loaded", "Run a strategy test"


def _overview_kpis(benchmark: str) -> list[tuple[str, str, str]]:
    result = st.session_state.get("backtest_result")
    status_value, status_note = _overview_status()
    if not result:
        return [
            ("Total Return", "n/a", f"vs {benchmark or 'SPY'}   n/a"),
            ("Sharpe Ratio", "n/a", ""),
            ("Max Drawdown", "n/a", ""),
            (f"Alpha (vs {benchmark or 'SPY'})", "n/a", ""),
            ("Status", status_value, status_note),
        ]

    equity = result["equity_curve"]
    initial_capital = float(result.get("config", {}).get("initial_capital", equity["portfolio_value"].iloc[0]))
    total_return = equity["portfolio_value"].iloc[-1] / initial_capital - 1.0
    metrics = result.get("metrics", {})
    return [
        ("Total Return", _pct(total_return), f"vs {benchmark or 'SPY'}   {_pct(metrics.get('Benchmark CAGR')) if 'Benchmark CAGR' in metrics else 'n/a'}"),
        ("Sharpe Ratio", f"{metrics.get('Sharpe Ratio', np.nan):.2f}" if pd.notna(metrics.get("Sharpe Ratio", np.nan)) else "n/a", ""),
        ("Max Drawdown", _pct(metrics.get("Max Drawdown")), ""),
        (f"Alpha (vs {benchmark or 'SPY'})", _pct(metrics.get("Alpha vs Benchmark")), ""),
        ("Status", status_value, status_note),
    ]


def _overview_card(label: str, value: str, note: str = "") -> str:
    return (
        f'<div class="ref-card ref-kpi">'
        f'<div class="ref-kpi-label">{escape(label)}</div>'
        f'<div class="ref-kpi-value">{escape(value)}</div>'
        f'<div class="ref-kpi-note">{escape(note)}</div>'
        f"</div>"
    )


def _factor_snapshot_html(benchmark: str) -> str:
    latest, leader, high_vol, weakest = _latest_factor_snapshot(benchmark)

    def val(column: str) -> tuple[str, float]:
        if latest.empty or column not in latest.columns:
            return "n/a", 0
        series = pd.to_numeric(latest[column], errors="coerce").dropna()
        if series.empty:
            return "n/a", 0
        if column.startswith("vol"):
            raw = float(series.max())
        elif column == "relative_strength_rank":
            raw = float(series.min())
        else:
            raw = float(series.max())
        width = max(8, min(100, abs(raw) * 100 if abs(raw) <= 1.5 else 100))
        return _pct(raw) if column != "relative_strength_rank" else f"{raw:.0f}", width

    rows = [
        ("Momentum", *val("mom_12m")),
        ("Value", leader if leader != "n/a" else "n/a", 38 if leader != "n/a" else 0),
        ("Volatility", *val("vol_60d")),
        ("Quality", *val("ma_crossover")),
    ]
    row_html = "".join(
        f'<div class="factor-row"><div>{escape(label)}</div><div class="factor-bar" style="--w:{width:.0f}%"><span></span></div><strong>{escape(value)}</strong></div>'
        for label, value, width in rows
    )
    return f'<div class="ref-card panel-card"><div class="panel-title">Top Factors (Snapshot)</div>{row_html}</div>'


def _risk_snapshot_html(benchmark: str) -> str:
    result = st.session_state.get("backtest_result")
    if result:
        metrics = result.get("metrics", {})
        rows = [
            ("Volatility", _pct(metrics.get("Annualized Volatility"))),
            (f"Beta (vs {benchmark or 'SPY'})", f"{metrics.get('Beta vs Benchmark', np.nan):.2f}" if pd.notna(metrics.get("Beta vs Benchmark", np.nan)) else "n/a"),
            ("Information Ratio", f"{metrics.get('Information Ratio', np.nan):.2f}" if pd.notna(metrics.get("Information Ratio", np.nan)) else "n/a"),
        ]
        footer = "Risk analytics available"
    else:
        rows = [("Volatility", "n/a"), (f"Beta (vs {benchmark or 'SPY'})", "n/a"), ("Information Ratio", "n/a")]
        footer = "Run backtest to view"
    row_html = "".join(
        f'<div class="risk-row"><div>{escape(label)}</div><strong>{escape(value)}</strong><div class="risk-toggle"></div></div>'
        for label, value in rows
    )
    return f'<div class="ref-card panel-card"><div class="panel-title">Risk Snapshot</div>{row_html}<div class="ref-kpi-note" style="margin-top:2.4rem;font-size:1rem;">{escape(footer)}</div></div>'


def _popular_actions_html() -> str:
    return (
        '<div class="ref-card panel-card short"><div class="panel-title">Popular Actions</div><div class="action-row">'
        '<span class="action-chip"><span class="action-icon">LD</span>Load Data</span>'
        '<span class="action-chip"><span class="action-icon">BT</span>Run Backtest</span>'
        '<span class="action-chip"><span class="action-icon">VF</span>View Factors</span>'
        '<span class="action-chip"><span class="action-icon">AI</span>AI Research Report</span>'
        "</div></div>"
    )


def _performance_placeholder_html() -> str:
    return (
        '<div class="ref-card panel-card">'
        '<div class="panel-title">Performance (vs SPY)</div>'
        '<div class="chart-legend"><span class="legend-line">Strategy</span><span class="legend-line spy">SPY</span></div>'
        '<div class="empty-chart">No data loaded</div>'
        "</div>"
    )


def _render_overview(benchmark: str) -> None:
    kpi_cols = st.columns(5, gap="medium")
    for col, (label, value, note) in zip(kpi_cols, _overview_kpis(benchmark)):
        with col:
            st.markdown(_overview_card(label, value, note), unsafe_allow_html=True)

    o1, o2, o3 = st.columns([1.18, 0.82, 0.96], gap="medium")
    with o1:
        result = st.session_state.get("backtest_result")
        if result:
            st.markdown('<div class="ref-card panel-card"><div class="panel-title">Performance (vs SPY)</div>', unsafe_allow_html=True)
            equity = result["equity_curve"].copy()
            chart_cols = ["date", "portfolio_value"]
            if equity["benchmark_value"].notna().any():
                chart_cols.append("benchmark_value")
            chart_data = equity[chart_cols].melt(id_vars="date", var_name="series", value_name="value")
            st.plotly_chart(
                _style_chart(px.line(chart_data, x="date", y="value", color="series"), height=216),
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(_performance_placeholder_html(), unsafe_allow_html=True)
    with o2:
        st.markdown(_factor_snapshot_html(benchmark), unsafe_allow_html=True)
    with o3:
        st.markdown(_risk_snapshot_html(benchmark), unsafe_allow_html=True)

    b1, b2 = st.columns([0.74, 1.16], gap="medium")
    status_value, status_note = _overview_status()
    with b1:
        st.markdown(
            f"""
            <div class="ref-card panel-card short">
                <div class="panel-title">Get Started</div>
                <div class="ref-kpi-note" style="font-size:1rem;">{escape(status_note if status_value != "Awaiting Data" else "Load data to generate insights and reports.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(_popular_actions_html(), unsafe_allow_html=True)


def _kpi_values() -> list[tuple[str, str, str]]:
    if _need_data():
        return [
            ("Universe", "Not loaded", "Load prices from sidebar"),
            ("Date range", "Pending", "Select dates and load data"),
            ("Portfolio return", "n/a", "Run a backtest"),
            ("Sharpe", "n/a", "Run a backtest"),
            ("Max drawdown", "n/a", "Run a backtest"),
            ("Alpha", "n/a", "Benchmark required"),
        ]

    price_data = st.session_state.price_data
    start = price_data["date"].min().date()
    end = price_data["date"].max().date()
    values = [
        ("Universe", str(price_data["ticker"].nunique()), f"{len(price_data):,} observations"),
        ("Date range", f"{start} to {end}", "Cached local market history"),
    ]

    result = st.session_state.get("backtest_result")
    if result:
        equity = result["equity_curve"]
        initial_capital = float(result.get("config", {}).get("initial_capital", equity["portfolio_value"].iloc[0]))
        total_return = equity["portfolio_value"].iloc[-1] / initial_capital - 1.0
        metrics = result.get("metrics", {})
        values.extend(
            [
                ("Portfolio return", _pct(total_return), _money(equity["portfolio_value"].iloc[-1])),
                ("Sharpe", f"{metrics.get('Sharpe Ratio', np.nan):.2f}" if pd.notna(metrics.get("Sharpe Ratio", np.nan)) else "n/a", "Risk adjusted return"),
                ("Max drawdown", _pct(metrics.get("Max Drawdown")), "Peak to trough"),
                ("Alpha", _pct(metrics.get("Alpha vs Benchmark")), "Annualised vs benchmark"),
            ]
        )
    else:
        values.extend(
            [
                ("Portfolio return", "n/a", "Run a backtest"),
                ("Sharpe", "n/a", "Run a backtest"),
                ("Max drawdown", "n/a", "Run a backtest"),
                ("Alpha", "n/a", "Benchmark required"),
            ]
        )
    return values


def _latest_factor_snapshot(benchmark: str) -> tuple[pd.DataFrame, str, str, str]:
    factor_data = st.session_state.get("factor_data", pd.DataFrame())
    if factor_data is None or factor_data.empty:
        return pd.DataFrame(), "n/a", "n/a", "n/a"
    latest = latest_rows(factor_data, group_col="ticker")
    if benchmark:
        latest = latest.loc[latest["ticker"] != benchmark]
    mom = latest.dropna(subset=["mom_12m"]) if "mom_12m" in latest.columns else pd.DataFrame()
    vol = latest.dropna(subset=["vol_60d"]) if "vol_60d" in latest.columns else pd.DataFrame()
    leader = mom.sort_values("mom_12m", ascending=False)["ticker"].iloc[0] if not mom.empty else "n/a"
    weakest = mom.sort_values("mom_12m", ascending=True)["ticker"].iloc[0] if not mom.empty else "n/a"
    high_vol = vol.sort_values("vol_60d", ascending=False)["ticker"].iloc[0] if not vol.empty else "n/a"
    return latest, leader, high_vol, weakest


def _market_regime(benchmark: str) -> tuple[str, str]:
    if _need_data():
        return "Awaiting Data", "Load market data to classify regime."

    price_data = st.session_state.price_data.copy()
    benchmark_df = price_data.loc[price_data["ticker"] == benchmark].sort_values("date")
    if benchmark_df.empty:
        benchmark_df = price_data.sort_values("date").groupby("ticker").tail(63)
    returns = pd.to_numeric(benchmark_df["returns"], errors="coerce").dropna()
    if returns.empty:
        return "Unclassified", "Insufficient returns for regime estimate."

    recent_return = (1.0 + returns.tail(63)).prod() - 1.0 if len(returns) >= 20 else (1.0 + returns).prod() - 1.0
    recent_vol = returns.tail(60).std() * np.sqrt(252) if len(returns) >= 20 else np.nan
    if pd.notna(recent_return) and recent_return > 0.03 and (pd.isna(recent_vol) or recent_vol < 0.28):
        return "Growth / Risk-On", "Constructive tape with manageable realised volatility."
    if pd.notna(recent_return) and recent_return < -0.03:
        return "Defensive / Risk-Off", "Benchmark pressure favours tighter risk budgets."
    if pd.notna(recent_vol) and recent_vol >= 0.32:
        return "Volatile / Transitional", "Elevated volatility suggests selective position sizing."
    return "Mixed / Factor-Led", "Cross-sectional signals likely matter more than broad beta."


def _render_command_center(benchmark: str) -> None:
    latest, leader, high_vol, weakest = _latest_factor_snapshot(benchmark)
    result = st.session_state.get("backtest_result")
    if _need_data():
        action = "Load market data"
    elif not result:
        action = "Run a factor backtest"
    elif "report_text" not in st.session_state:
        action = "Generate an AI report"
    else:
        action = "Review risk analytics"
    regime, regime_note = _market_regime(benchmark)

    def note_for(ticker: str, column: str) -> str:
        if latest.empty or ticker == "n/a" or column not in latest.columns:
            return "Awaiting factor history"
        value = latest.loc[latest["ticker"] == ticker, column]
        if value.empty or pd.isna(value.iloc[0]):
            return "Insufficient observations"
        return _pct(float(value.iloc[0]))

    if _need_data():
        summary = "Load a universe to activate the research copilot. It will surface momentum leadership, volatility concentration, market regime, and the next most useful workflow action."
    else:
        summary = (
            f"{leader} leads the momentum screen, {high_vol} carries the highest realised volatility, "
            f"and {weakest} is the weakest momentum constituent. The current regime reads as {regime.lower()}. "
            f"Next best action: {action.lower()}."
        )

    st.markdown(
        f"""
        <div class="command-grid">
            <section class="copilot-panel">
                <div class="copilot-head">
                    <div class="copilot-orb"></div>
                    <div>
                        <div class="copilot-title">AI Research Copilot</div>
                        <div class="copilot-subtitle">Deterministic market intelligence from local factor and risk data</div>
                    </div>
                </div>
                <div class="copilot-metrics">
                    <div class="copilot-item">
                        <div class="copilot-label">Momentum Leader</div>
                        <div class="copilot-value">{leader}</div>
                        <div class="intel-note">{note_for(leader, "mom_12m")}</div>
                    </div>
                    <div class="copilot-item">
                        <div class="copilot-label">Highest Volatility</div>
                        <div class="copilot-value">{high_vol}</div>
                        <div class="intel-note">{note_for(high_vol, "vol_60d")}</div>
                    </div>
                    <div class="copilot-item">
                        <div class="copilot-label">Weakest Momentum</div>
                        <div class="copilot-value">{weakest}</div>
                        <div class="intel-note">{note_for(weakest, "mom_12m")}</div>
                    </div>
                </div>
                <div class="ai-summary"><strong>AI Summary:</strong> {summary}</div>
            </section>
            <section class="market-panel">
                <div class="market-grid">
                    <div class="market-callout">
                        <div class="market-label">Market Regime</div>
                        <div class="market-value">{regime}</div>
                        <div class="intel-note">{regime_note}</div>
                    </div>
                    <div class="market-callout">
                        <div class="market-label">Benchmark</div>
                        <div class="market-value">{benchmark or "n/a"}</div>
                        <div class="intel-note">Reference for beta, alpha, IR</div>
                    </div>
                    <div class="market-callout">
                        <div class="market-label">Opportunity Focus</div>
                        <div class="market-value">{leader if leader != "n/a" else "Pending"}</div>
                        <div class="intel-note">Best current cross-sectional momentum candidate</div>
                    </div>
                    <div class="market-callout">
                        <div class="market-label">Risk Watch</div>
                        <div class="market-value">{high_vol if high_vol != "n/a" else "Pending"}</div>
                        <div class="intel-note">Highest realised volatility constituent</div>
                    </div>
                </div>
                <div class="market-action">
                    <div class="market-label">Suggested Next Action</div>
                    <div class="market-value">{action}</div>
                    <div class="intel-note">Designed to answer what to inspect next before scrolling.</div>
                </div>
            </section>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _tab_header(title: str, subtitle: str) -> None:
    st.markdown(f"### {title}")
    st.caption(subtitle)


if "activity_mode" not in st.session_state:
    st.session_state.activity_mode = "ready"
if "activity_label" not in st.session_state:
    st.session_state.activity_label = "Ready"
activity_slot = st.empty()
_update_activity(activity_slot, st.session_state.activity_mode, st.session_state.activity_label)

_render_rail()

if "control_tickers" not in st.session_state:
    st.session_state.control_tickers = DEFAULT_TICKERS
if "control_benchmark" not in st.session_state:
    st.session_state.control_benchmark = "SPY"
if "control_start" not in st.session_state:
    st.session_state.control_start = pd.Timestamp("2018-01-01")
if "control_end" not in st.session_state:
    st.session_state.control_end = pd.Timestamp.today()

with st.popover("Menu", help="Open research settings"):
    st.markdown("#### Research Settings")
    st.markdown('<div class="settings-note">Universe, benchmark, dates, and local cache controls.</div>', unsafe_allow_html=True)
    ticker_text = st.text_area("Ticker universe", height=118, key="control_tickers")
    parsed_tickers = normalize_tickers(st.session_state.control_tickers)
    if parsed_tickers:
        chips = "".join(f'<span class="ticker-chip">{ticker}</span>' for ticker in parsed_tickers[:16])
        if len(parsed_tickers) > 16:
            chips += f'<span class="ticker-chip">+{len(parsed_tickers) - 16}</span>'
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown(f'<div class="sidebar-note">{len(parsed_tickers)} securities parsed for local factor research.</div>', unsafe_allow_html=True)

    benchmark_input = st.text_input("Benchmark", key="control_benchmark")
    date_cols = st.columns(2)
    start_date = date_cols[0].date_input("Start", key="control_start")
    end_date = date_cols[1].date_input("End", key="control_end")

    refresh = st.checkbox("Refresh market data", value=False)
    run_data = st.button("Load market data", use_container_width=True)

parsed_tickers = normalize_tickers(st.session_state.control_tickers)
benchmark = str(st.session_state.control_benchmark).strip().upper()
start_date = st.session_state.control_start
end_date = st.session_state.control_end
tickers = parsed_tickers
if benchmark and benchmark not in tickers:
    tickers_with_benchmark = [*tickers, benchmark]
else:
    tickers_with_benchmark = tickers

if run_data:
    _update_activity(activity_slot, "processing", "Refreshing data" if refresh else "Loading data")
    with st.spinner("Loading market data..."):
        try:
            st.session_state.price_data = get_price_data(
                tickers_with_benchmark,
                start=str(start_date),
                end=str(end_date),
                refresh=refresh,
            )
            _update_activity(activity_slot, "processing", "Calculating factors")
            st.session_state.factor_data = calculate_factors(st.session_state.price_data)
            st.session_state.last_data_status = "ready"
            _update_activity(activity_slot, "ready", "Data loaded")
        except Exception as exc:
            st.session_state.last_data_status = "error"
            _update_activity(activity_slot, "ready", "Action failed")
            st.error(str(exc))

status_text, status_mode = _status_label()
dot_class = "" if status_mode == "ready" else " idle"
regime_label, _ = _market_regime(benchmark)
st.markdown(
    f"""
    <section class="reference-shell">
        <header class="reference-header">
            <div class="reference-title-row">
                <div class="reference-title">Quant Research Platform</div>
                <div class="reference-page-label">Overview</div>
            </div>
            <div class="reference-market-pill"><span></span>Market: {escape(regime_label)}</div>
        </header>
    </section>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["Overview", "Data", "Factors", "Backtest", "Risk Analytics", "AI Research Report"])

with tabs[0]:
    _render_overview(benchmark)

with tabs[1]:
    _tab_header("Data", "Cached market data, adjusted close history, and clean OHLCV preview.")
    if _need_data():
        st.info("Load market data from the sidebar to start.")
    else:
        price_data = st.session_state.price_data
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickers", price_data["ticker"].nunique())
        c2.metric("Rows", f"{len(price_data):,}")
        c3.metric("Start", str(price_data["date"].min().date()))
        c4.metric("End", str(price_data["date"].max().date()))

        close_prices = price_data.pivot(index="date", columns="ticker", values="adj_close").reset_index()
        melted = close_prices.melt(id_vars="date", var_name="ticker", value_name="adj_close")
        st.plotly_chart(
            _style_chart(px.line(melted, x="date", y="adj_close", color="ticker", title="Adjusted Close"), height=390),
            use_container_width=True,
        )
        st.caption("Recent normalised OHLCV records")
        st.dataframe(price_data.tail(500), use_container_width=True, hide_index=True)

with tabs[2]:
    _tab_header("Factors", "Cross-sectional signal rankings, latest factor values, and single-name factor history.")
    if _need_data():
        st.info("Load market data first.")
    else:
        factor_data = st.session_state.get("factor_data")
        if factor_data is None or factor_data.empty:
            _update_activity(activity_slot, "processing", "Calculating factors")
            factor_data = calculate_factors(st.session_state.price_data)
            st.session_state.factor_data = factor_data
            _update_activity(activity_slot, "ready", "Factors ready")

        f1, f2 = st.columns([1, 1])
        factor_choice = f1.selectbox("Ranking factor", options=list(FACTOR_DIRECTIONS), index=0)
        history_ticker = f2.selectbox(
            "Factor history ticker",
            options=sorted(factor_data["ticker"].dropna().unique()),
            index=0,
        )
        latest = latest_rows(factor_data.dropna(subset=[factor_choice]), group_col="ticker")
        direction = FACTOR_DIRECTIONS.get(factor_choice, "higher")
        ranked = latest.sort_values(factor_choice, ascending=(direction == "lower"))
        factor_cols = [
            "date",
            "ticker",
            "mom_12m",
            "mom_6m",
            "mom_3m",
            "vol_60d",
            "vol_20d",
            "mean_reversion_5d",
            "ma_crossover",
            "relative_strength_rank",
        ]
        available_cols = [col for col in factor_cols if col in factor_data.columns]
        latest_matrix = latest_rows(factor_data, group_col="ticker")[available_cols].sort_values("ticker")

        c1, c2 = st.columns([1.05, 0.95])
        with c1:
            st.caption("Latest factor matrix")
            st.dataframe(latest_matrix, use_container_width=True, hide_index=True)
        with c2:
            heat_cols = [col for col in ["mom_12m", "mom_6m", "mom_3m", "vol_60d", "vol_20d", "ma_crossover"] if col in latest_matrix.columns]
            heat_source = latest_matrix.set_index("ticker")[heat_cols].apply(pd.to_numeric, errors="coerce")
            if heat_source.dropna(how="all").empty:
                st.info("Factor heatmap will appear once enough history is available.")
            else:
                z = (heat_source - heat_source.mean()) / heat_source.std(ddof=0).replace(0, np.nan)
                z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
                fig = px.imshow(
                    z.T,
                    color_continuous_scale=["#ef7e8e", "#17222d", "#58d5b7"],
                    aspect="auto",
                    title="Latest Factor Z-Score Heatmap",
                )
                st.plotly_chart(_style_chart(fig, height=360), use_container_width=True)

        st.caption("Selected factor ranking")
        st.dataframe(
            ranked[["date", "ticker", factor_choice, f"{factor_choice}_rank", f"{factor_choice}_score"]].head(50),
            use_container_width=True,
            hide_index=True,
        )
        if not ranked.empty:
            c1, c2 = st.columns([1, 1])
            c1.plotly_chart(
                _style_chart(px.bar(ranked.head(25), x="ticker", y=factor_choice, title=f"Latest {factor_choice} Ranking"), height=340),
                use_container_width=True,
            )
            history = factor_data.loc[factor_data["ticker"] == history_ticker, ["date", factor_choice]].dropna()
            if history.empty:
                c2.info("Not enough history for this factor yet.")
            else:
                c2.plotly_chart(
                    _style_chart(px.line(history, x="date", y=factor_choice, title=f"{history_ticker} {factor_choice} History"), height=340),
                    use_container_width=True,
                )

with tabs[3]:
    _tab_header("Backtest", "Configure factor strategy construction, rebalance cadence, benchmark, and capital base.")
    if _need_data():
        st.info("Load market data first.")
    else:
        factor_data = st.session_state.get("factor_data")
        available_tickers = sorted(st.session_state.price_data["ticker"].dropna().unique())
        investable_tickers = [t for t in available_tickers if t != benchmark]
        left, mid, right = st.columns(3)
        selected_factor = left.selectbox("Signal", options=list(FACTOR_DIRECTIONS), index=0)
        top_n_limit = max(len(investable_tickers), 1)
        top_n = mid.number_input("Top N", min_value=1, max_value=top_n_limit, value=min(5, top_n_limit))
        weighting = right.selectbox("Weighting", options=["equal", "volatility_weighted", "mean_variance", "risk_parity"])
        b1, b2, b3 = st.columns(3)
        rebalance = b1.selectbox("Rebalance", options=["M", "W", "D"], format_func={"M": "Monthly", "W": "Weekly", "D": "Daily"}.get)
        transaction_bps = b2.number_input("Transaction cost (bps)", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
        benchmark_for_backtest = b3.text_input("Backtest benchmark", value=benchmark or "SPY").strip().upper()
        c1, c2 = st.columns([1, 2])
        starting_capital = c1.number_input("Starting capital", min_value=1_000.0, value=100_000.0, step=10_000.0)
        run_bt = c2.button("Run backtest", use_container_width=True)

        if run_bt:
            _update_activity(activity_slot, "processing", "Running backtest")
            try:
                config = {
                    "selected_factor": selected_factor,
                    "top_n": int(top_n),
                    "rebalance_frequency": rebalance,
                    "transaction_cost": float(transaction_bps) / 10000.0,
                    "weighting_method": weighting,
                    "benchmark_ticker": benchmark_for_backtest,
                    "initial_capital": float(starting_capital),
                    "start_date": str(start_date),
                    "end_date": str(end_date),
                }
                st.session_state.backtest_result = run_backtest(st.session_state.price_data, factor_data, config)
                _update_activity(activity_slot, "processing", "Generating risk analytics")
                _update_activity(activity_slot, "ready", "Backtest complete")
            except Exception as exc:
                _update_activity(activity_slot, "ready", "Backtest failed")
                st.error(str(exc))

        result = st.session_state.get("backtest_result")
        if result:
            equity = result["equity_curve"]
            warnings = result.get("warnings", [])
            for warning in warnings:
                st.warning(warning)
            m1, m2, m3 = st.columns(3)
            initial_capital = float(result.get("config", {}).get("initial_capital", equity["portfolio_value"].iloc[0]))
            m1.metric("Final value", _money(equity["portfolio_value"].iloc[-1]))
            m2.metric("Total return", f"{equity['portfolio_value'].iloc[-1] / initial_capital - 1:.2%}")
            m3.metric("Average turnover", _metrics_frame({"Average Turnover": result["metrics"].get("Average Turnover")})["Value"].iloc[0])

            chart_cols = ["date", "portfolio_value"]
            if equity["benchmark_value"].notna().any():
                chart_cols.append("benchmark_value")
            chart_data = equity[chart_cols].melt(id_vars="date", var_name="series", value_name="value")
            st.plotly_chart(_style_chart(px.line(chart_data, x="date", y="value", color="series", title="Equity Curve"), height=380), use_container_width=True)
            st.plotly_chart(_style_chart(px.area(equity, x="date", y="drawdown", title="Portfolio Drawdown"), height=300), use_container_width=True)
            metrics_frame = _metrics_frame(result["metrics"])
            st.dataframe(metrics_frame, use_container_width=True, hide_index=True)
            if not result["weights"].empty:
                latest_weights = latest_rows(result["weights"], group_col="ticker")
                st.caption("Latest holdings")
                st.dataframe(latest_weights.sort_values("weight", ascending=False), use_container_width=True, hide_index=True)
                st.caption("Holdings over time")
                st.dataframe(result["weights"].tail(500), use_container_width=True, hide_index=True)
            trades = result.get("trades", pd.DataFrame())
            if not trades.empty:
                st.caption("Recent rebalances")
                st.dataframe(trades.tail(200), use_container_width=True, hide_index=True)

            summary_md = backtest_summary_markdown(result)
            d1, d2, d3 = st.columns(3)
            d1.download_button("Download backtest summary", data=summary_md, file_name="backtest_summary.md", mime="text/markdown", use_container_width=True)
            d2.download_button("Download metrics CSV", data=metrics_frame.to_csv(index=False), file_name="backtest_metrics.csv", mime="text/csv", use_container_width=True)
            d3.download_button("Download metrics text", data=markdown_to_text(metrics_to_markdown(metrics_frame)), file_name="backtest_metrics.txt", mime="text/plain", use_container_width=True)

with tabs[4]:
    _tab_header("Risk Analytics", "Performance, tail risk, rolling risk, drawdown, and benchmark-relative comparison.")
    result = st.session_state.get("backtest_result")
    if not result:
        st.info("Run a backtest to view risk analytics.")
    else:
        equity = result["equity_curve"].copy()
        equity["rolling_vol_63d"] = rolling_volatility(equity["portfolio_returns"], window=63).to_numpy()
        equity["rolling_sharpe_63d"] = rolling_sharpe(equity["portfolio_returns"], window=63).to_numpy()
        perf_metrics = _metrics_frame(result["metrics"])
        tail_metrics = _metrics_frame(risk_summary(equity["portfolio_returns"], equity["portfolio_value"]))
        st.caption("Performance and benchmark-relative metrics")
        st.dataframe(perf_metrics, use_container_width=True, hide_index=True)
        r1, r2 = st.columns(2)
        r1.plotly_chart(_style_chart(px.line(equity, x="date", y="rolling_vol_63d", title="Rolling 63-Day Volatility"), height=320), use_container_width=True)
        r2.plotly_chart(_style_chart(px.line(equity, x="date", y="rolling_sharpe_63d", title="Rolling 63-Day Sharpe"), height=320), use_container_width=True)
        d1, d2 = st.columns(2)
        d1.plotly_chart(_style_chart(px.area(equity, x="date", y="drawdown", title="Drawdown Series"), height=320), use_container_width=True)
        if equity["benchmark_value"].notna().any():
            relative = equity[["date", "portfolio_value", "benchmark_value"]].copy()
            relative["relative_performance"] = relative["portfolio_value"] / relative["benchmark_value"] - 1.0
            d2.plotly_chart(
                _style_chart(px.line(relative, x="date", y="relative_performance", title="Portfolio vs Benchmark Relative Return"), height=320),
                use_container_width=True,
            )
        else:
            d2.warning("Benchmark data is unavailable for this backtest.")
        st.caption("Tail risk summary")
        st.dataframe(tail_metrics, use_container_width=True, hide_index=True)
        trades = result.get("trades", pd.DataFrame())
        if not trades.empty:
            st.dataframe(trades.tail(200), use_container_width=True, hide_index=True)

with tabs[5]:
    _tab_header("AI Research Report", "Deterministic institutional equity note with optional OpenAI-compatible enhancement.")
    if _need_data():
        st.info("Load market data first.")
    else:
        if "factor_data" not in st.session_state or st.session_state.factor_data.empty:
            _update_activity(activity_slot, "processing", "Calculating factors")
            st.session_state.factor_data = calculate_factors(st.session_state.price_data)
            _update_activity(activity_slot, "ready", "Factors ready")
        available = sorted(st.session_state.price_data["ticker"].dropna().unique())
        if not available:
            st.warning("No tickers are available for report generation.")
        else:
            selected_ticker = st.selectbox("Ticker", available)
            use_llm = st.checkbox("Use optional OpenAI-compatible API if configured", value=False)
            if st.button("Generate report"):
                _update_activity(activity_slot, "processing", "Generating report")
                try:
                    report_text = generate_research_report(
                        selected_ticker,
                        st.session_state.price_data,
                        st.session_state.factor_data,
                        st.session_state.get("backtest_result"),
                        use_llm=use_llm,
                    )
                    st.session_state.report_text = report_text
                    st.session_state.report_ticker = selected_ticker
                    st.session_state.report_path = save_report(report_text, selected_ticker, extension="md")
                    _update_activity(activity_slot, "ready", "Report generated")
                except Exception as exc:
                    _update_activity(activity_slot, "ready", "Report failed")
                    st.error(str(exc))

            report_text = st.session_state.get("report_text")
            if report_text:
                st.markdown(report_text)
                file_path = st.session_state.get("report_path")
                if file_path:
                    st.caption(f"Saved locally to {file_path}")
                st.download_button(
                    "Download markdown",
                    data=report_text,
                    file_name=file_path.name if file_path else f"{selected_ticker}_research_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
                st.download_button(
                    "Download text",
                    data=markdown_to_text(report_text),
                    file_name=file_path.with_suffix(".txt").name if file_path else f"{selected_ticker}_research_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
