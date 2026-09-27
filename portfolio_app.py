"""
╔══════════════════════════════════════════════════════════════╗
║     PORTFOLIO ANALYZER — Streamlit App (Bloomberg Style)     ║
║                                                              ║
║  Ejecución: streamlit run portfolio_app.py                   ║
║  Requisitos: pip install streamlit plotly yfinance pandas numpy║
╚══════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
import warnings
import time
warnings.filterwarnings('ignore')



def safe(text):
    """Remove non-latin-1 characters for PDF compatibility."""
    if not isinstance(text, str):
        return str(text)
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2019": "\x27", "\u2018": "\x27",
        "\u201c": "\x22", "\u201d": "\x22", "\u2026": "...", "\u00b2": "2",
        "\u03c3": "sigma", "\u03b2": "Beta", "\u03c1": "r",
        "\u221a": "sqrt", "\u03a3": "Sigma", "\u2248": "~",
        "\u25c4": "<", "\u25ba": ">",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    try:
        text.encode("latin-1")
    except UnicodeEncodeError:
        text = text.encode("latin-1", errors="replace").decode("latin-1")
    return text

def generate_pdf_report(metrics, portfolio, portfolio_value, scenario, ticker_info, kpi_thresholds, evaluate_kpi_fn):
    """Generate a PDF report of the portfolio analysis."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None
    
    m = metrics
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Page 1: Summary
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 20)
    pdf.cell(0, 12, 'PORTFOLIO ANALYZER - Report', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.cell(0, 6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ln=True)
    pdf.cell(0, 6, safe(f'Scenario: {scenario} | Value: EUR {portfolio_value:,.0f}'), ln=True)
    pdf.ln(8)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'PORTFOLIO OVERVIEW', ln=True)
    pdf.set_draw_color(255, 140, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('Helvetica', '', 10)
    key_metrics = [
        ('Expected Return E(Rp)', f"{m['erp_portfolio']*100:.2f}%"),
        ('Volatility (Ann.)', f"{m['vol']*100:.2f}%"),
        ('Sharpe Ratio (Forward)', f"{m['sharpe_fwd']:.2f}"),
        ('Sharpe Ratio (Historical)', f"{m['sharpe_hist']:.2f}"),
        ('Sortino Ratio', f"{m['sortino']:.2f}"),
        ('Max Drawdown', f"{m['max_dd']*100:.1f}%"),
        ('CAGR (Historical)', f"{m['cagr']*100:.1f}%"),
        ('Diversification Benefit', f"{m['div_benefit']*100:.1f}%"),
        ('VaR 5% (Monthly)', f"{m['var_5']*100:.1f}%"),
        ('Win Rate', f"{m['pos_periods']/m['total_periods']*100:.0f}%"),
        ('Gain/Loss Ratio', f"{m['gain_loss']:.2f}"),
        ('Risk-Free Rate', f"{m['rf']*100:.2f}%"),
        ('Market Return (Rm)', f"{m['rm']*100:.2f}%"),
        ('Equity Risk Premium', f"{m['erp']*100:.2f}%"),
        ('Observations', f"{m['n_months']} months"),
    ]
    for label, value in key_metrics:
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(90, 7, label, border=0)
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 7, value, ln=True)
    
    # Page 2: Holdings
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'HOLDINGS & EXPECTED RETURNS', ln=True)
    pdf.set_draw_color(255, 140, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    col_w = [22, 38, 16, 14, 14, 16, 18, 20, 22]
    headers = ['Ticker', 'Name', 'Weight', 'Beta', 'R2', 'E(Ri)', 'Vol%', 'Bench', 'Source']
    pdf.set_font('Helvetica', 'B', 7)
    pdf.set_text_color(255, 140, 0)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=1)
    pdf.ln()
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 7)
    bm_map = {'US': '^GSPC', 'MC': '^IBEX', 'DE': '^GDAXI', 'SW': '^SSMI'}
    for t in sorted(m['available']):
        info = ticker_info.get(t, {})
        beta = m['betas'].get(t)
        r2 = m['r_squareds'].get(t)
        er = m['expected_returns'].get(t, 0)
        src = m['data_sources'].get(t, '')
        mkt = info.get('market', 'US')
        bm = bm_map.get(mkt, 'N/A')
        vol_ann = (m['cov_matrix'].loc[t, t] ** 0.5) * 100 if t in m['cov_matrix'].columns else 0
        row = [safe(t), safe(info.get('name', t)[:20]), f"{portfolio.get(t,0)*100:.1f}%",
               f"{beta:.2f}" if beta else '-', f"{r2:.2f}" if r2 else '-',
               f"{er*100:.1f}%", f"{vol_ann:.1f}%", bm if beta else 'N/A', src[:12]]
        for i, val in enumerate(row):
            pdf.cell(col_w[i], 6, val, border=1)
        pdf.ln()
    
    # Page 3: Risk
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'RISK CONTRIBUTION', ln=True)
    pdf.set_draw_color(255, 140, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    rc_w = [25, 50, 25, 30, 25]
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(255, 140, 0)
    for i, h in enumerate(['Ticker', 'Name', 'Weight', 'Risk Contr.', 'Ratio']):
        pdf.cell(rc_w[i], 7, h, border=1)
    pdf.ln()
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 8)
    for t, rc_val in sorted(m['risk_contrib'].items(), key=lambda x: x[1], reverse=True):
        info = ticker_info.get(t, {})
        w = portfolio.get(t, 0)
        ratio = rc_val / w if w > 0 else 0
        for i, val in enumerate([safe(t), safe(info.get('name', t)[:25]), f"{w*100:.1f}%", f"{rc_val*100:.1f}%", f"{ratio:.2f}x"]):
            pdf.cell(rc_w[i], 6, val, border=1)
        pdf.ln()
    
    # Exposure
    pdf.ln(6)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'EXPOSURE', ln=True)
    pdf.set_draw_color(255, 140, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, 'Sector:', ln=True)
    pdf.set_font('Helvetica', '', 9)
    for s, w in sorted(m['sector_weights'].items(), key=lambda x: x[1], reverse=True):
        pdf.cell(80, 6, safe(s))
        pdf.cell(0, 6, f"{w*100:.1f}%", ln=True)
    
    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.cell(0, 7, 'Geography:', ln=True)
    pdf.set_font('Helvetica', '', 9)
    mn = {'US': 'United States', 'MC': 'Spain', 'DE': 'Germany', 'SW': 'Switzerland',
          'COMMODITY': 'Commodities', 'CRYPTO': 'Crypto', 'CASH': 'Cash'}
    geo = {}
    for t, w in portfolio.items():
        info = ticker_info.get(t, {})
        c = mn.get(info.get('market', 'US'), info.get('market', 'US'))
        geo[c] = geo.get(c, 0) + w
    for c, w in sorted(geo.items(), key=lambda x: x[1], reverse=True):
        pdf.cell(80, 6, safe(c))
        pdf.cell(0, 6, f"{w*100:.1f}%", ln=True)
    
    # KPIs
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, 'KPI EVALUATION', ln=True)
    pdf.set_draw_color(255, 140, 0)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    
    kpi_vals = {
        'sharpe_fwd': m['sharpe_fwd'], 'sharpe_hist': m['sharpe_hist'],
        'volatility': m['vol'] * 100, 'max_dd': m['max_dd'] * 100,
        'expected_ret': m['erp_portfolio'] * 100, 'div_benefit': m['div_benefit'] * 100,
        'sector_conc': m['max_sector'] * 100, 'max_position': m['max_position'] * 100,
        'max_risk_contr': m['max_rc'] * 100,
        'pos_periods': m['pos_periods']/m['total_periods']*100 if m['total_periods'] > 0 else 0,
        'gain_loss': m['gain_loss'],
    }
    
    kw = [55, 25, 20, 25, 25]
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(255, 140, 0)
    for i, h in enumerate(['KPI', 'Value', 'Status', 'Green', 'Yellow']):
        pdf.cell(kw[i], 7, h, border=1)
    pdf.ln()
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 8)
    for key, thresh in kpi_thresholds.items():
        val = kpi_vals.get(key, 0)
        status = evaluate_kpi_fn(val, thresh)
        st_text = {'green': 'PASS', 'yellow': 'WATCH', 'red': 'FAIL'}[status]
        if key in ('sharpe_fwd', 'sharpe_hist', 'gain_loss'):
            vs = f"{val:.2f}"
        else:
            vs = f"{val:.1f}%"
        sign = '>' if thresh['target'] == 'above' else '<'
        for i, v in enumerate([safe(thresh['name']), vs, st_text, f"{sign} {thresh['green']}", f"{sign} {thresh['yellow']}"]):
            pdf.cell(kw[i], 6, v, border=1)
        pdf.ln()
    
    pdf.ln(8)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(0, 4,
        "DISCLAIMER: These returns are estimates based on models, not predictions. "
        "CAPM has well-known limitations. Past performance is not indicative of future results.")
    
    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Portfolio Analyzer",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ════════════════════════════════════════════════════════════════
# BLOOMBERG TERMINAL CSS
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&display=swap');

    /* ─── BLOOMBERG FONT: JetBrains Mono everywhere ─── */
    *, *::before, *::after,
    html, body, [class*="css"],
    .stApp, .stApp *,
    div, span, p, label, input, button, select, textarea, option,
    td, th, tr, li, ul, ol, a, code, pre,
    h1, h2, h3, h4, h5, h6,
    .stMarkdown, .stMarkdown *,
    .stDataFrame, .stDataFrame *,
    .stTable, .stTable *,
    .stSelectbox, .stSelectbox *,
    .stTextInput, .stTextInput *,
    .stNumberInput, .stNumberInput *,
    .stSlider, .stSlider *,
    .stTabs, .stTabs *,
    .stExpander, .stExpander *,
    .stRadio, .stRadio *,
    .stCheckbox, .stCheckbox *,
    .stButton, .stButton *,
    .stDownloadButton, .stDownloadButton *,
    .stMetric, .stMetric *,
    .stAlert, .stAlert *,
    .stSuccess, .stSuccess *,
    .stWarning, .stWarning *,
    .stError, .stError *,
    .stInfo, .stInfo *,
    [data-testid], [data-testid] *,
    [data-baseweb], [data-baseweb] *,
    .row-widget, .row-widget * {
        font-family: 'JetBrains Mono', 'Consolas', 'Courier New', 'Lucida Console', monospace !important;
    }

    /* ─── Global ─── */
    .stApp {
        background-color: #0a0a0a !important;
        color: #d4d4d4 !important;
    }
    
    /* ─── Hide Streamlit defaults ─── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stToolbar"] {display: none;}
    
    /* ─── Tabs Bloomberg style ─── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0px;
        background-color: #0a0a0a;
        border-bottom: 2px solid #FF8C00;
        padding: 0;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
        font-size: 0.8rem;
        color: #808080;
        border-bottom: 2px solid transparent;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background-color: #0a0a0a;
    }
    .stTabs [aria-selected="true"] {
        color: #FF8C00 !important;
        border-bottom: 2px solid #FF8C00 !important;
        background-color: #1a1a1a !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFa500 !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background-color: #0a0a0a;
    }
    
    /* ─── Bloomberg Header ─── */
    .bb-header {
        background: #1a1a1a;
        border-bottom: 2px solid #FF8C00;
        padding: 10px 20px;
        margin: -1rem -1rem 1rem -1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .bb-title {
        color: #FF8C00;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.1em;
    }
    .bb-subtitle {
        color: #808080;
        font-size: 0.7rem;
        font-weight: 400;
        letter-spacing: 0.05em;
    }
    .bb-version {
        color: #555;
        font-size: 0.7rem;
    }
    
    /* ─── Metric cards Bloomberg ─── */
    .bb-metric {
        background: #141414;
        border: 1px solid #2a2a2a;
        border-left: 3px solid #FF8C00;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .bb-metric-label {
        font-size: 0.65rem;
        font-weight: 600;
        color: #FF8C00;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 2px;
    }
    .bb-metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .bb-metric-value-sm {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
    }
    .bb-metric-delta-pos { color: #00C853; font-size: 0.75rem; font-weight: 600; }
    .bb-metric-delta-neg { color: #FF1744; font-size: 0.75rem; font-weight: 600; }
    .bb-metric-delta-neutral { color: #808080; font-size: 0.75rem; font-weight: 600; }
    
    /* ─── Section headers ─── */
    .bb-section {
        font-size: 0.75rem;
        font-weight: 700;
        color: #FF8C00;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin: 1.5rem 0 0.75rem 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #2a2a2a;
    }
    
    /* ─── KPI Semaphore rows ─── */
    .bb-kpi-row {
        display: flex;
        align-items: center;
        padding: 6px 12px;
        border-bottom: 1px solid #1a1a1a;
        font-size: 0.78rem;
    }
    .bb-kpi-row:hover { background: #1a1a1a; }
    .bb-kpi-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 10px;
        flex-shrink: 0;
    }
    .bb-kpi-dot-green { background: #00C853; box-shadow: 0 0 4px #00C853; }
    .bb-kpi-dot-yellow { background: #FFD600; box-shadow: 0 0 4px #FFD600; }
    .bb-kpi-dot-red { background: #FF1744; box-shadow: 0 0 4px #FF1744; }
    .bb-kpi-name { flex: 1; color: #d4d4d4; }
    .bb-kpi-value { font-weight: 700; color: #FFFFFF; min-width: 80px; text-align: right; }
    
    /* ─── Card container ─── */
    .bb-card {
        background: #141414;
        border: 1px solid #2a2a2a;
        padding: 16px;
        margin-bottom: 10px;
    }
    .bb-card-title {
        font-size: 0.7rem;
        font-weight: 700;
        color: #FF8C00;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 10px;
    }
    
    /* ─── Data source badge ─── */
    .bb-badge {
        display: inline-block;
        font-size: 0.6rem;
        font-weight: 700;
        padding: 2px 6px;
        letter-spacing: 0.05em;
    }
    .bb-badge-calc { background: #1a3a1a; color: #00C853; border: 1px solid #00C853; }
    .bb-badge-est { background: #3a2a00; color: #FFD600; border: 1px solid #FFD600; }
    .bb-badge-proxy { background: #002244; color: #4488FF; border: 1px solid #4488FF; }
    .bb-badge-manual { background: #2a1a2a; color: #CC66FF; border: 1px solid #CC66FF; }
    .bb-badge-fallback { background: #3a1a1a; color: #FF1744; border: 1px solid #FF1744; }
    
    /* ─── Alert boxes ─── */
    .bb-alert-critical {
        background: #1a0a0a;
        border-left: 3px solid #FF1744;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.78rem;
        color: #FF1744;
    }
    .bb-alert-warning {
        background: #1a1500;
        border-left: 3px solid #FFD600;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.78rem;
        color: #FFD600;
    }
    .bb-alert-info {
        background: #0a0a1a;
        border-left: 3px solid #4488FF;
        padding: 8px 12px;
        margin-bottom: 6px;
        font-size: 0.78rem;
        color: #4488FF;
    }
    
    /* ─── Progress bar ─── */
    .bb-progress-bg {
        background: #2a2a2a;
        height: 4px;
        width: 100%;
        margin-top: 6px;
    }
    .bb-progress-fill {
        height: 4px;
        transition: width 0.3s;
    }
    
    /* ─── Expander styling ─── */
    .streamlit-expanderHeader {
        background: #141414 !important;
        border: 1px solid #2a2a2a !important;
        color: #FF8C00 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }
    .streamlit-expanderContent {
        background: #0f0f0f !important;
        border: 1px solid #2a2a2a !important;
        border-top: none !important;
    }
    
    /* ─── Scrollbar ─── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #FF8C00; }
    
    /* ─── Number input buttons ─── */
    .stNumberInput button { color: #FF8C00 !important; }
    
    /* ─── Radio buttons ─── */
    .stRadio label { color: #d4d4d4 !important; }
    
    /* ─── DataFrames ─── */
    .stDataFrame { border: 1px solid #2a2a2a; }
    
    /* ─── Banner ─── */
    .bb-banner {
        position: relative;
        width: 100%;
        height: 80px;
        background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 50%, #1a1200 100%);
        border-bottom: 2px solid #FF8C00;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
        margin-bottom: 8px;
        overflow: hidden;
    }
    .bb-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: repeating-linear-gradient(
            90deg,
            transparent, transparent 60px,
            rgba(255,140,0,0.03) 60px, rgba(255,140,0,0.03) 61px
        );
    }
    .bb-banner-title {
        font-size: 1.2rem;
        font-weight: 800;
        letter-spacing: 0.15em;
        color: #FFFFFF;
        z-index: 1;
    }
    .bb-banner-title span { color: #FF8C00; }
    .bb-banner-sub {
        font-size: 0.65rem;
        color: #808080;
        letter-spacing: 0.08em;
        margin-top: 2px;
    }
    .bb-banner-version {
        font-size: 0.65rem;
        color: #555;
        z-index: 1;
        text-align: right;
    }
    .bb-banner-accent {
        position: absolute;
        right: 0; top: 0; bottom: 0;
        width: 200px;
        background: linear-gradient(90deg, transparent, rgba(255,140,0,0.05));
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# DEFAULT PORTFOLIO
# ════════════════════════════════════════════════════════════════
DEFAULT_PORTFOLIO = {
    'ORCL':    0.042,
    'ABBN.SW': 0.043,
    'ITX.MC':  0.159,
    'MAP.MC':  0.065,
    'RHM.DE':  0.040,
    'JEF':     0.039,
    'SAN.MC':  0.145,
    'DTE.DE':  0.021,
    'FER':     0.018,
    'EEM':     0.111,
    'BTU':     0.049,
    'GLD':     0.127,
    'EUFN':    0.078,
    'ITA':     0.063,
}

TICKER_INFO = {
    'ORCL':    {'market': 'US', 'name': 'Oracle Corp.',         'asset_class': 'Equity',    'sector': 'Technology'},
    'ABBN.SW': {'market': 'SW', 'name': 'ABB Ltd',              'asset_class': 'Equity',    'sector': 'Industrials'},
    'ITX.MC':  {'market': 'MC', 'name': 'Inditex',              'asset_class': 'Equity',    'sector': 'Consumer Discretionary'},
    'MAP.MC':  {'market': 'MC', 'name': 'Mapfre',               'asset_class': 'Equity',    'sector': 'Financial Services'},
    'RHM.DE':  {'market': 'DE', 'name': 'Rheinmetall AG',       'asset_class': 'Equity',    'sector': 'Industrials'},
    'JEF':     {'market': 'US', 'name': 'Jefferies Financial',  'asset_class': 'Equity',    'sector': 'Financial Services'},
    'SAN.MC':  {'market': 'MC', 'name': 'Banco Santander',      'asset_class': 'Equity',    'sector': 'Financial Services'},
    'DTE.DE':  {'market': 'DE', 'name': 'Deutsche Telekom',     'asset_class': 'Equity',    'sector': 'Communication Services'},
    'FER':     {'market': 'US', 'name': 'Ferrovial',            'asset_class': 'Equity',    'sector': 'Industrials'},
    'EEM':     {'market': 'US', 'name': 'iShares MSCI EM ETF',  'asset_class': 'Equity',    'sector': 'Diversified EM'},
    'BTU':     {'market': 'US', 'name': 'Peabody Energy',       'asset_class': 'Equity',    'sector': 'Energy'},
    'GLD':     {'market': 'COMMODITY', 'name': 'SPDR Gold ETF', 'asset_class': 'Commodity', 'sector': 'Gold'},
    'EUFN':    {'market': 'US', 'name': 'iShares EU Fin ETF',   'asset_class': 'Equity',    'sector': 'Financial Services'},
    'ITA':     {'market': 'US', 'name': 'iShares US A&D ETF',   'asset_class': 'Equity',    'sector': 'Industrials'},
    'BTC-USD': {'market': 'CRYPTO', 'name': 'Bitcoin',          'asset_class': 'Crypto',    'sector': 'Cryptocurrency'},
    'CASH':    {'market': 'CASH', 'name': 'Cash',               'asset_class': 'Cash',      'sector': 'Cash'},
}

BENCHMARK_MAP = {
    'US': '^GSPC', 'MC': '^IBEX', 'DE': '^GDAXI', 'SW': '^SSMI',
    'L': '^FTSE', 'AS': '^AEX', 'PA': '^FCHI',
}

GORDON_SCENARIOS = {
    'Conservador': 0.030, 'Moderado': 0.050, 'Base': 0.060,
    'Crecimiento': 0.073, 'Optimista': 0.090,
}

KPI_THRESHOLDS = {
    'sharpe_fwd':    {'name': 'Sharpe (Forward)',      'target': 'above', 'green': 0.50, 'yellow': 0.25},
    'sharpe_hist':   {'name': 'Sharpe (Historical)',    'target': 'above', 'green': 0.80, 'yellow': 0.50},
    'volatility':    {'name': 'Portfolio Volatility',   'target': 'below', 'green': 15.0, 'yellow': 20.0},
    'max_dd':        {'name': 'Max Drawdown',           'target': 'above', 'green': -15.0,'yellow': -25.0},
    'expected_ret':  {'name': 'Expected Return E(Rp)',  'target': 'above', 'green': 8.0,  'yellow': 6.0},
    'div_benefit':   {'name': 'Diversification Benefit','target': 'above', 'green': 30.0, 'yellow': 20.0},
    'sector_conc':   {'name': 'Max Sector Concentration','target':'below', 'green': 25.0, 'yellow': 35.0},
    'max_position':  {'name': 'Max Single Position',    'target': 'below', 'green': 10.0, 'yellow': 15.0},
    'max_risk_contr':{'name': 'Max Risk Contribution',  'target': 'below', 'green': 15.0, 'yellow': 20.0},
    'pos_periods':   {'name': '% Positive Periods',     'target': 'above', 'green': 60.0, 'yellow': 50.0},
    'gain_loss':     {'name': 'Gain/Loss Ratio',        'target': 'above', 'green': 1.50, 'yellow': 1.00},
}

MARKET_NAMES = {
    'US': 'United States', 'MC': 'Spain', 'DE': 'Germany', 'SW': 'Switzerland',
    'L': 'United Kingdom', 'AS': 'Netherlands', 'PA': 'France',
    'COMMODITY': 'Commodities', 'CRYPTO': 'Crypto', 'CASH': 'Cash',
}


# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════

def get_risk_free_rate():
    try:
        tnx = yf.download("^TNX", period="5d", progress=False)
        if tnx is not None and len(tnx) > 0:
            close = tnx['Close']
            if hasattr(close, 'columns'):
                close = close.iloc[:, 0]
            val = float(close.dropna().iloc[-1]) / 100.0
            if 0.001 < val < 0.15:
                return val
    except:
        pass
    return 0.0425


def get_sp500_dividend_yield():
    try:
        spy = yf.Ticker("SPY")
        divs = spy.dividends
        if divs is not None and len(divs) > 0:
            one_year_ago = datetime.now() - timedelta(days=400)
            recent = divs[divs.index >= one_year_ago.strftime('%Y-%m-%d')]
            if len(recent) >= 2:
                annual_div = float(recent.tail(4).sum())
                hist = spy.history(period="5d")
                if hist is not None and len(hist) > 0:
                    price = float(hist['Close'].iloc[-1])
                    if price > 0:
                        dy = annual_div / price
                        if 0.005 < dy < 0.05:
                            return dy, f"SPY calculado (divs/precio: ${annual_div:.2f}/${price:.0f})"
    except:
        pass
    
    try:
        spy = yf.Ticker("SPY")
        info = spy.info or {}
        for field in ['dividendYield', 'trailingAnnualDividendYield']:
            val = info.get(field)
            if val and 0.005 < float(val) < 0.05:
                return float(val), f"SPY {field}"
    except:
        pass
    
    return 0.013, "FALLBACK (1.3%)"


def download_prices(tickers, years=5):
    end = datetime.now()
    start = end - timedelta(days=years*365)
    
    real_tickers = [t for t in tickers if t != 'CASH']
    if not real_tickers:
        return pd.DataFrame()
    
    for attempt in range(3):
        try:
            data = yf.download(real_tickers, start=start, end=end, progress=False)
            if data is not None and len(data) > 0:
                if 'Close' in data.columns or (hasattr(data.columns, 'levels') and 'Close' in data.columns.get_level_values(0)):
                    if len(real_tickers) == 1:
                        close = data['Close']
                        if isinstance(close, pd.DataFrame):
                            close = close.iloc[:, 0]
                        prices = pd.DataFrame({real_tickers[0]: close})
                    else:
                        prices = data['Close']
                    return prices.dropna(how='all')
            time.sleep(2)
        except:
            time.sleep(2)
    return pd.DataFrame()


def calculate_portfolio_metrics(prices, weights, rf, scenario_g=0.06):
    if prices.empty:
        return None
    
    monthly = prices.resample('ME').last()
    returns = monthly.pct_change().dropna()
    
    available = [t for t in weights if t in returns.columns]
    if not available:
        return None
    
    w_available = {t: weights[t] for t in available}
    cash_weight = weights.get('CASH', 0)
    non_cash_total = sum(w_available.values())
    
    if non_cash_total > 0:
        w_norm = {t: v/non_cash_total for t, v in w_available.items()}
    else:
        w_norm = w_available
    
    ret_df = returns[available]
    w_arr = np.array([w_norm.get(t, 0) for t in available])
    
    port_returns = ret_df.values @ w_arr
    
    n_months = len(port_returns)
    cagr_factor = (1 + port_returns).prod()
    years = n_months / 12
    cagr = cagr_factor ** (1/years) - 1 if years > 0 else 0
    
    vol_monthly = np.std(port_returns, ddof=1)
    vol_annual = vol_monthly * np.sqrt(12)
    
    sharpe_hist = (np.mean(port_returns) - rf/12) / vol_monthly if vol_monthly > 0 else 0
    sharpe_hist_ann = sharpe_hist * np.sqrt(12)
    
    downside = port_returns[port_returns < 0]
    down_dev = np.std(downside, ddof=1) if len(downside) > 1 else vol_monthly
    sortino = (np.mean(port_returns) - rf/12) / down_dev * np.sqrt(12) if down_dev > 0 else 0
    
    cumulative = np.cumprod(1 + port_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns)
    
    cov_matrix = ret_df.cov() * 12
    vol_portfolio = np.sqrt(w_arr @ cov_matrix.values @ w_arr)
    vol_undiversified = sum(w_arr[i] * np.sqrt(cov_matrix.values[i,i]) for i in range(len(available)))
    div_benefit = 1 - vol_portfolio / vol_undiversified if vol_undiversified > 0 else 0
    
    # Risk contribution
    marginal = cov_matrix.values @ w_arr
    risk_contrib = w_arr * marginal / vol_portfolio if vol_portfolio > 0 else w_arr
    
    # Correlation matrix
    corr_matrix = ret_df.corr()
    
    # CAPM
    dy, dy_source = get_sp500_dividend_yield()
    buyback = 0.015
    total_yield = dy + buyback
    rm = total_yield + scenario_g
    erp = rm - rf
    
    benchmarks_needed = set()
    for t in available:
        info = TICKER_INFO.get(t, {})
        mkt = info.get('market', 'US')
        if mkt in BENCHMARK_MAP:
            benchmarks_needed.add(BENCHMARK_MAP[mkt])
    
    bench_prices = {}
    if benchmarks_needed:
        try:
            end = datetime.now()
            start = end - timedelta(days=5*365)
            for bm in benchmarks_needed:
                bdata = yf.download(bm, start=start, end=end, progress=False)
                if bdata is not None and len(bdata) > 0:
                    close = bdata['Close']
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    bench_prices[bm] = close
        except:
            pass
    
    betas = {}
    r_squareds = {}
    expected_returns = {}
    data_sources = {}
    
    for t in available:
        info = TICKER_INFO.get(t, {})
        mkt = info.get('market', 'US')
        asset_class = info.get('asset_class', 'Equity')
        
        if asset_class in ('Commodity', 'Crypto', 'Cash'):
            betas[t] = None
            r_squareds[t] = None
            
            if asset_class == 'Cash':
                expected_returns[t] = rf
                data_sources[t] = 'CASH (Rf)'
            elif asset_class == 'Commodity':
                hist_ret = float(ret_df[t].mean() * 12) if t in ret_df.columns else 0.05
                expected_returns[t] = hist_ret * 0.5 + 0.03 * 0.5
                data_sources[t] = 'ESTIMATED'
            elif asset_class == 'Crypto':
                hist_ret = float(ret_df[t].mean() * 12) if t in ret_df.columns else 0.10
                expected_returns[t] = hist_ret * 0.3
                data_sources[t] = 'ESTIMATED'
        else:
            bm_ticker = BENCHMARK_MAP.get(mkt, '^GSPC')
            if bm_ticker in bench_prices:
                bm_monthly = bench_prices[bm_ticker].resample('ME').last().pct_change().dropna()
                common_idx = ret_df[t].dropna().index.intersection(bm_monthly.index)
                if len(common_idx) > 12:
                    x = bm_monthly.loc[common_idx].values
                    y = ret_df[t].loc[common_idx].values
                    cov_xy = np.cov(x, y, ddof=1)
                    beta = cov_xy[0,1] / cov_xy[0,0] if cov_xy[0,0] > 0 else 1.0
                    ss_res = np.sum((y - (np.mean(y) + beta * (x - np.mean(x))))**2)
                    ss_tot = np.sum((y - np.mean(y))**2)
                    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
                    betas[t] = beta
                    r_squareds[t] = r2
                else:
                    betas[t] = 1.0
                    r_squareds[t] = 0
            else:
                betas[t] = 1.0
                r_squareds[t] = 0
            
            expected_returns[t] = rf + betas[t] * erp
            data_sources[t] = 'CALCULATED'
    
    # Portfolio expected return
    erp_portfolio = sum(w_available.get(t, 0) * expected_returns.get(t, rf) for t in available)
    erp_portfolio += cash_weight * rf
    
    # Normalize for display
    total_w = sum(weights.values())
    if total_w > 0:
        erp_norm = erp_portfolio / total_w
    else:
        erp_norm = erp_portfolio
    
    sharpe_fwd = (erp_norm - rf) / vol_portfolio if vol_portfolio > 0 else 0
    
    pos_months = sum(1 for r in port_returns if r > 0)
    gains = [r for r in port_returns if r > 0]
    losses = [abs(r) for r in port_returns if r < 0]
    gain_loss = np.mean(gains) / np.mean(losses) if losses else 999
    
    var_5 = np.percentile(port_returns, 5)
    
    sector_weights = {}
    for t in weights:
        info = TICKER_INFO.get(t, {})
        sector = info.get('sector', 'Other')
        sector_weights[sector] = sector_weights.get(sector, 0) + weights[t]
    max_sector = max(sector_weights.values()) if sector_weights else 0
    max_sector_name = max(sector_weights, key=sector_weights.get) if sector_weights else ''
    max_position = max(weights.values()) if weights else 0
    max_position_ticker = max(weights, key=weights.get) if weights else ''
    max_rc = max(risk_contrib) if len(risk_contrib) > 0 else 0
    max_rc_ticker = available[np.argmax(risk_contrib)] if len(risk_contrib) > 0 else ''
    
    return {
        'returns': ret_df,
        'port_returns': port_returns,
        'cumulative': cumulative,
        'dates': ret_df.index,
        'cagr': cagr,
        'vol': vol_portfolio,
        'vol_undiv': vol_undiversified,
        'div_benefit': div_benefit,
        'sharpe_hist': sharpe_hist_ann,
        'sortino': sortino,
        'max_dd': max_dd,
        'skewness': float(pd.Series(port_returns).skew()),
        'kurtosis': float(pd.Series(port_returns).kurtosis()),
        'var_5': var_5,
        'pos_periods': pos_months,
        'total_periods': n_months,
        'gain_loss': gain_loss,
        'corr_matrix': corr_matrix,
        'cov_matrix': cov_matrix,
        'risk_contrib': dict(zip(available, risk_contrib)),
        'betas': betas,
        'r_squareds': r_squareds,
        'expected_returns': expected_returns,
        'data_sources': data_sources,
        'erp_portfolio': erp_norm,
        'sharpe_fwd': sharpe_fwd,
        'rf': rf,
        'rm': rm,
        'erp': erp,
        'dy': dy,
        'dy_source': dy_source,
        'sector_weights': sector_weights,
        'max_sector': max_sector,
        'max_sector_name': max_sector_name,
        'max_position': max_position,
        'max_position_ticker': max_position_ticker,
        'max_rc': max_rc,
        'max_rc_ticker': max_rc_ticker,
        'available': available,
        'w_norm': w_norm,
        'n_months': n_months,
    }


def evaluate_kpi(value, threshold):
    if threshold['target'] == 'above':
        if value >= threshold['green']:
            return 'green'
        elif value >= threshold['yellow']:
            return 'yellow'
        else:
            return 'red'
    else:
        if value <= threshold['green']:
            return 'green'
        elif value <= threshold['yellow']:
            return 'yellow'
        else:
            return 'red'


def bb_metric_html(label, value, delta="", delta_class="neutral"):
    return f"""
    <div class="bb-metric">
        <div class="bb-metric-label">{label}</div>
        <div class="bb-metric-value">{value}</div>
        <div class="bb-metric-delta-{delta_class}">{delta}</div>
    </div>
    """


def bb_metric_sm_html(label, value, delta="", delta_class="neutral"):
    return f"""
    <div class="bb-metric">
        <div class="bb-metric-label">{label}</div>
        <div class="bb-metric-value-sm">{value}</div>
        <div class="bb-metric-delta-{delta_class}">{delta}</div>
    </div>
    """


def bb_section(title):
    return f'<div class="bb-section">{title}</div>'


def plotly_bloomberg(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#141414',
        font=dict(family="JetBrains Mono, Consolas, monospace", size=11, color="#d4d4d4"),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a'),
        yaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a'),
    )
    return fig


# ════════════════════════════════════════════════════════════════
# BANNER
# ════════════════════════════════════════════════════════════════
import base64, os

banner_html = """
<div class="bb-banner">
    <div>
        <div class="bb-banner-title">◆ PORTFOLIO <span>ANALYZER</span></div>
        <div class="bb-banner-sub">Quantitative Portfolio Management</div>
    </div>
    <div class="bb-banner-version">v2.0</div>
    <div class="bb-banner-accent"></div>
</div>
"""

# Try to load banner image
banner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'banner.png')
if os.path.exists(banner_path):
    with open(banner_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    banner_html = f"""
    <div style="position:relative; width:100%; height:100px; margin-bottom:8px; overflow:hidden;
                border-bottom: 2px solid #FF8C00;">
        <img src="data:image/png;base64,{b64}" style="width:100%; height:100%; object-fit:cover; opacity:0.6;">
        <div style="position:absolute; top:50%; left:24px; transform:translateY(-50%);">
            <div class="bb-banner-title">◆ PORTFOLIO <span>ANALYZER</span></div>
            <div class="bb-banner-sub">Quantitative Portfolio Management</div>
        </div>
        <div style="position:absolute; top:50%; right:24px; transform:translateY(-50%);">
            <div class="bb-banner-version">v2.0</div>
        </div>
    </div>
    """

st.markdown(banner_html, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SESSION STATE
# ════════════════════════════════════════════════════════════════
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'portfolio_value' not in st.session_state:
    st.session_state.portfolio_value = 100000
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'metrics' not in st.session_state:
    st.session_state.metrics = None
if 'prices' not in st.session_state:
    st.session_state.prices = None


# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════
tabs = st.tabs(["PORTFOLIO", "DASHBOARD", "RETURNS", "RISK", "EXPOSURE", "SIMULATOR", "METHODOLOGY"])


# ═══════════════════════════════════════════════════════════
# TAB 0: PORTFOLIO
# ═══════════════════════════════════════════════════════════
with tabs[0]:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown(bb_section("PORTFOLIO CONFIGURATION"), unsafe_allow_html=True)
        
        pv = st.number_input("Portfolio Value (€)", value=st.session_state.portfolio_value,
                             step=1000, min_value=1000, key="pv_input")
        st.session_state.portfolio_value = pv
        
        input_mode = st.radio("Input mode", ["Percentage (%)", "Amount (€)"], horizontal=True, key="input_mode")
        
        st.markdown(bb_section("HOLDINGS"), unsafe_allow_html=True)
        
        # Dynamic column headers based on mode
        if input_mode == "Percentage (%)":
            col_header_1 = "WEIGHT %"
            col_header_2 = "AMOUNT €"
        else:
            col_header_1 = "AMOUNT €"
            col_header_2 = "WEIGHT %"
        
        st.markdown(f"""
        <div style="display:flex; padding:4px 0; border-bottom:1px solid #2a2a2a; margin-bottom:8px;">
            <div style="flex:2.5; color:#FF8C00; font-size:0.7rem; font-weight:700; letter-spacing:0.1em;">TICKER</div>
            <div style="flex:1.5; color:#FF8C00; font-size:0.7rem; font-weight:700; letter-spacing:0.1em;">{col_header_1}</div>
            <div style="flex:1.5; color:#808080; font-size:0.7rem; font-weight:700; letter-spacing:0.1em;">{col_header_2}</div>
            <div style="flex:0.5; color:#FF8C00; font-size:0.7rem; font-weight:700; text-align:center;">✕</div>
        </div>
        """, unsafe_allow_html=True)
        
        tickers_to_remove = []
        portfolio_copy = dict(st.session_state.portfolio)
        
        for i, (ticker, weight) in enumerate(portfolio_copy.items()):
            c1, c2, c3, c4 = st.columns([2.5, 1.5, 1.5, 0.5])
            
            with c1:
                new_ticker = st.text_input("", value=ticker, key=f"ticker_{i}", label_visibility="collapsed")
            
            if input_mode == "Percentage (%)":
                with c2:
                    new_weight = st.number_input("", value=round(weight*100, 2), step=0.1,
                                                 key=f"weight_{i}", label_visibility="collapsed")
                with c3:
                    amount = pv * new_weight / 100
                    st.markdown(f"<div style='padding:8px; color:#808080;'>€{amount:,.0f}</div>", unsafe_allow_html=True)
                
                if new_ticker != ticker:
                    tickers_to_remove.append(ticker)
                st.session_state.portfolio[new_ticker if new_ticker != ticker else ticker] = new_weight / 100
            else:
                with c2:
                    current_amount = pv * weight
                    new_amount = st.number_input("", value=round(current_amount, 0), step=100.0,
                                                 key=f"amount_{i}", label_visibility="collapsed")
                with c3:
                    calc_weight = (new_amount / pv * 100) if pv > 0 else 0
                    st.markdown(f"<div style='padding:8px; color:#808080;'>{calc_weight:.1f}%</div>", unsafe_allow_html=True)
                
                if new_ticker != ticker:
                    tickers_to_remove.append(ticker)
                st.session_state.portfolio[new_ticker if new_ticker != ticker else ticker] = new_amount / pv if pv > 0 else 0
            
            with c4:
                if st.button("✕", key=f"del_{i}", type="secondary"):
                    tickers_to_remove.append(ticker)
        
        for t in tickers_to_remove:
            if t in st.session_state.portfolio:
                del st.session_state.portfolio[t]
        
        # Buttons
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            if st.button("+ ADD ASSET", type="primary"):
                new_key = f"NEW_{len(st.session_state.portfolio)}"
                st.session_state.portfolio[new_key] = 0.05
                st.rerun()
        with bc2:
            if st.button("+ ADD CASH", type="primary"):
                if 'CASH' not in st.session_state.portfolio:
                    st.session_state.portfolio['CASH'] = 0.05
                    st.rerun()
        with bc3:
            if st.button("↻ RESET"):
                st.session_state.portfolio = {}
                st.rerun()
    
    with col_right:
        st.markdown(bb_section("PORTFOLIO SUMMARY"), unsafe_allow_html=True)
        
        total = sum(st.session_state.portfolio.values())
        total_pct = total * 100
        
        pct_color = "#00C853" if abs(total_pct - 100) < 0.5 else ("#FFD600" if total_pct < 100 else "#FF1744")
        fill_w = min(total_pct, 100)
        
        st.markdown(f"""
        <div class="bb-card">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span class="bb-card-title">Total Allocated</span>
                <span style="color:{pct_color}; font-weight:700; font-size:0.85rem;">{total_pct:.1f}%</span>
            </div>
            <div class="bb-progress-bg">
                <div class="bb-progress-fill" style="width:{fill_w}%; background:{pct_color};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if abs(total_pct - 100) < 0.5:
            st.success("Portfolio weights sum to 100% ✓")
        elif total_pct < 100:
            st.warning(f"Unallocated: {100-total_pct:.1f}% (consider adding Cash)")
        else:
            st.error(f"Over-allocated by {total_pct-100:.1f}%")
        
        n_assets = len(st.session_state.portfolio)
        st.markdown(f"""
        <div class="bb-card">
            <div class="bb-card-title">Holdings</div>
            <div class="bb-metric-value-sm">{n_assets} assets</div>
            <div style="color:#808080; font-size:0.75rem;">Value: €{pv:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.portfolio:
            max_pos = max(st.session_state.portfolio, key=st.session_state.portfolio.get)
            max_val = st.session_state.portfolio[max_pos] * 100
            st.markdown(f"""
            <div class="bb-card">
                <div class="bb-card-title">Largest Position</div>
                <div class="bb-metric-value-sm">{max_val:.1f}%</div>
                <div style="color:#808080; font-size:0.75rem;">{max_pos}</div>
            </div>
            """, unsafe_allow_html=True)
            
            sector_w = {}
            for t, w in st.session_state.portfolio.items():
                info = TICKER_INFO.get(t, {})
                s = info.get('sector', 'Other')
                sector_w[s] = sector_w.get(s, 0) + w
            if sector_w:
                top_s = max(sector_w, key=sector_w.get)
                st.markdown(f"""
                <div class="bb-card">
                    <div class="bb-card-title">Top Sector</div>
                    <div class="bb-metric-value-sm">{sector_w[top_s]*100:.1f}%</div>
                    <div style="color:#808080; font-size:0.75rem;">{top_s}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Analysis controls
    st.markdown("<br>", unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns([1, 1, 1])
    with ac1:
        scenario = st.selectbox("Growth Scenario (g)", list(GORDON_SCENARIOS.keys()), index=2)
    with ac2:
        lookback = st.slider("Historical Data (years)", 1, 10, 5)
    with ac3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("▶ ANALYZE", type="primary", use_container_width=True):
            with st.spinner("Downloading market data..."):
                prices = download_prices(list(st.session_state.portfolio.keys()), lookback)
                rf = get_risk_free_rate()
                g = GORDON_SCENARIOS[scenario]
                metrics = calculate_portfolio_metrics(prices, st.session_state.portfolio, rf, g)
                
                if metrics:
                    st.session_state.metrics = metrics
                    st.session_state.prices = prices
                    st.session_state.analyzed = True
                    st.session_state.scenario = scenario
                    st.success("Analysis complete ✓")
                else:
                    st.error("Error: Could not calculate metrics. Check tickers.")


# ═══════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first --> go to PORTFOLIO tab and click ANALYZE</div>',
                    unsafe_allow_html=True)
    else:
        m = st.session_state.metrics
        
        st.markdown(bb_section("PORTFOLIO OVERVIEW"), unsafe_allow_html=True)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(bb_metric_html("E(Rp)", f"{m['erp_portfolio']*100:.1f}%", f"CAPM {st.session_state.get('scenario','Base')}", "pos"), unsafe_allow_html=True)
        with c2:
            st.markdown(bb_metric_html("Volatility", f"{m['vol']*100:.1f}%", "Annualized", "neutral"), unsafe_allow_html=True)
        with c3:
            sc = "pos" if m['sharpe_fwd'] > 0.5 else ("neutral" if m['sharpe_fwd'] > 0.25 else "neg")
            st.markdown(bb_metric_html("Sharpe (Fwd)", f"{m['sharpe_fwd']:.2f}", f"Rf={m['rf']*100:.1f}%", sc), unsafe_allow_html=True)
        with c4:
            st.markdown(bb_metric_html("Max Drawdown", f"{m['max_dd']*100:.1f}%", "Historical", "neg"), unsafe_allow_html=True)
        with c5:
            st.markdown(bb_metric_html("Diversification", f"{m['div_benefit']*100:.1f}%", "Vol reduction", "pos"), unsafe_allow_html=True)
        
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(bb_metric_sm_html("CAGR (Historical)", f"{m['cagr']*100:.1f}%"), unsafe_allow_html=True)
        with c2:
            st.markdown(bb_metric_sm_html("Sharpe (Historical)", f"{m['sharpe_hist']:.2f}"), unsafe_allow_html=True)
        with c3:
            st.markdown(bb_metric_sm_html("Sortino", f"{m['sortino']:.2f}"), unsafe_allow_html=True)
        with c4:
            st.markdown(bb_metric_sm_html("VaR 5% (Monthly)", f"{m['var_5']*100:.1f}%"), unsafe_allow_html=True)
        with c5:
            wr = m['pos_periods']/m['total_periods']*100 if m['total_periods'] > 0 else 0
            st.markdown(bb_metric_sm_html("Win Rate", f"{wr:.0f}%"), unsafe_allow_html=True)
        
        # KPI Semaphore
        st.markdown(bb_section("KPI SEMAPHORE"), unsafe_allow_html=True)
        
        kpi_values = {
            'sharpe_fwd': m['sharpe_fwd'],
            'sharpe_hist': m['sharpe_hist'],
            'volatility': m['vol'] * 100,
            'max_dd': m['max_dd'] * 100,
            'expected_ret': m['erp_portfolio'] * 100,
            'div_benefit': m['div_benefit'] * 100,
            'sector_conc': m['max_sector'] * 100,
            'max_position': m['max_position'] * 100,
            'max_risk_contr': m['max_rc'] * 100,
            'pos_periods': m['pos_periods']/m['total_periods']*100 if m['total_periods'] > 0 else 0,
            'gain_loss': m['gain_loss'],
        }
        
        greens = yellows = reds = 0
        kpi_html = ""
        for key, thresh in KPI_THRESHOLDS.items():
            val = kpi_values.get(key, 0)
            status = evaluate_kpi(val, thresh)
            if status == 'green':
                greens += 1
            elif status == 'yellow':
                yellows += 1
            else:
                reds += 1
            
            dot_class = f"bb-kpi-dot-{status}"
            
            if key in ('sharpe_fwd', 'sharpe_hist', 'gain_loss'):
                val_str = f"{val:.2f}"
            elif key in ('volatility', 'max_dd', 'expected_ret', 'div_benefit', 'sector_conc', 'max_position', 'max_risk_contr', 'pos_periods'):
                val_str = f"{val:.1f}%"
            else:
                val_str = f"{val:.2f}"
            
            kpi_html += f"""
            <div class="bb-kpi-row">
                <div class="bb-kpi-dot {dot_class}"></div>
                <div class="bb-kpi-name">{thresh['name']}</div>
                <div class="bb-kpi-value">{val_str}</div>
            </div>
            """
        
        st.markdown(kpi_html, unsafe_allow_html=True)
        st.markdown(f"""<div style="padding:8px 12px; color:#808080; font-size:0.75rem;">
            Score: {greens} <span style="color:#00C853;">●</span> {yellows} <span style="color:#FFD600;">●</span> {reds} <span style="color:#FF1744;">●</span> ({greens}/{greens+yellows+reds} passing)
        </div>""", unsafe_allow_html=True)
        
        # PDF Download
        pdf_buf = generate_pdf_report(
            m, st.session_state.portfolio, st.session_state.portfolio_value,
            st.session_state.get('scenario', 'Base'), TICKER_INFO,
            KPI_THRESHOLDS, evaluate_kpi
        )
        if pdf_buf:
            st.download_button(
                label="DOWNLOAD REPORT (PDF)",
                data=pdf_buf,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True,
            )
        
        # Portfolio Growth Chart
        st.markdown(bb_section("PORTFOLIO GROWTH"), unsafe_allow_html=True)
        
        cum_series = pd.Series(m['cumulative'], index=m['dates'][:len(m['cumulative'])])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_series.index, y=cum_series.values,
                                  mode='lines', line=dict(color='#FF8C00', width=2),
                                  fill='tozeroy', fillcolor='rgba(255,140,0,0.1)'))
        fig.update_layout(yaxis_title="Growth of €1", showlegend=False, height=350)
        st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 2: RETURNS
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first --> go to PORTFOLIO tab and click ANALYZE</div>',
                    unsafe_allow_html=True)
    else:
        m = st.session_state.metrics
        
        st.markdown(bb_section("MARKET RETURN — GORDON GROWTH MODEL"), unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(bb_metric_html("Risk-Free Rate", f"{m['rf']*100:.2f}%", "US Treasury 10Y"), unsafe_allow_html=True)
        with c2:
            st.markdown(bb_metric_html("Dividend Yield", f"{m['dy']*100:.2f}%", m['dy_source']), unsafe_allow_html=True)
        with c3:
            st.markdown(bb_metric_html("Total Yield", f"{(m['dy']+0.015)*100:.2f}%", "Div + Buyback"), unsafe_allow_html=True)
        with c4:
            st.markdown(bb_metric_html("Rm (Base)", f"{m['rm']*100:.2f}%", f"g = {GORDON_SCENARIOS.get(st.session_state.get('scenario','Base'),0.06)*100:.1f}%"), unsafe_allow_html=True)
        
        # Scenarios table
        st.markdown(bb_section("RETURN SCENARIOS"), unsafe_allow_html=True)
        
        sc_data = []
        for name, g in GORDON_SCENARIOS.items():
            rm_sc = m['dy'] + 0.015 + g
            erp_sc = rm_sc - m['rf']
            marker = " ◄" if name == st.session_state.get('scenario', 'Base') else ""
            sc_data.append({
                'Scenario': name + marker,
                'g': f"{g*100:.1f}%",
                'Rm': f"{rm_sc*100:.2f}%",
                'ERP': f"{erp_sc*100:.2f}%",
            })
        st.dataframe(pd.DataFrame(sc_data), use_container_width=True, hide_index=True)
        
        # Betas & Expected Returns
        st.markdown(bb_section("CAPM — BETAS & EXPECTED RETURNS"), unsafe_allow_html=True)
        
        beta_data = []
        for t in sorted(m['available']):
            info = TICKER_INFO.get(t, {})
            mkt = info.get('market', 'US')
            bm = BENCHMARK_MAP.get(mkt, 'N/A')
            beta = m['betas'].get(t)
            r2 = m['r_squareds'].get(t)
            er = m['expected_returns'].get(t, 0)
            src = m['data_sources'].get(t, '')
            
            beta_data.append({
                'Ticker': t,
                'Name': info.get('name', t),
                'Weight': f"{st.session_state.portfolio.get(t,0)*100:.1f}%",
                'β': f"{beta:.2f}" if beta is not None else '–',
                'R²': f"{r2:.2f}" if r2 is not None else '–',
                'E(Ri)': f"{er*100:.2f}%",
                'Benchmark': bm if beta is not None else 'N/A',
                'Source': src,
            })
        
        st.dataframe(pd.DataFrame(beta_data), use_container_width=True, hide_index=True)
        
        # Portfolio Expected Return
        st.markdown(bb_section("PORTFOLIO EXPECTED RETURN"), unsafe_allow_html=True)
        st.markdown(f"""
        <div class="bb-metric" style="border-left-color:#FF8C00;">
            <div class="bb-metric-label">E(Rp)</div>
            <div class="bb-metric-value">{m['erp_portfolio']*100:.2f}%</div>
            <div class="bb-metric-delta-pos">Scenario: {st.session_state.get('scenario','Base')}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Data Source Transparency
        st.markdown(bb_section("DATA SOURCE TRANSPARENCY"), unsafe_allow_html=True)
        for t in sorted(m['data_sources'].keys()):
            src = m['data_sources'][t]
            info = TICKER_INFO.get(t, {})
            badge_class = 'bb-badge-calc'
            if src == 'ESTIMATED':
                badge_class = 'bb-badge-est'
            elif src == 'PROXY':
                badge_class = 'bb-badge-proxy'
            elif 'FALLBACK' in src:
                badge_class = 'bb-badge-fallback'
            
            st.markdown(f"""
            <div style="padding:4px 0; display:flex; align-items:center; gap:8px;">
                <span style="color:#d4d4d4; min-width:70px; font-size:0.78rem;">{t}</span>
                <span class="bb-badge {badge_class}">{src}</span>
                <span style="color:#808080; font-size:0.75rem;">{info.get('name', '')}</span>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 3: RISK
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first --> go to PORTFOLIO tab and click ANALYZE</div>',
                    unsafe_allow_html=True)
    else:
        m = st.session_state.metrics
        
        # Correlation Heatmap
        st.markdown(bb_section("CORRELATION MATRIX"), unsafe_allow_html=True)
        
        corr = m['corr_matrix']
        names = [TICKER_INFO.get(t, {}).get('name', t)[:12] for t in corr.columns]
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values, x=names, y=names,
            colorscale=[[0, '#FF1744'], [0.5, '#0a0a0a'], [1, '#00C853']],
            zmin=-1, zmax=1,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont=dict(size=9, color="#d4d4d4"),
        ))
        fig.update_layout(height=500)
        st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)
        
        n = m['n_months']
        thresh = round(2 / np.sqrt(n), 2)
        st.caption(f"Observations: {n} months | Significance threshold (95%): |r| > {thresh}")
        
        # Risk Contribution
        st.markdown(bb_section("RISK CONTRIBUTION BY ASSET"), unsafe_allow_html=True)
        
        rc = m['risk_contrib']
        weights_dict = {t: st.session_state.portfolio.get(t, 0) for t in rc}
        
        rc_sorted = sorted(rc.items(), key=lambda x: x[1], reverse=True)
        tickers_rc = [t for t, _ in rc_sorted]
        rc_vals = [v*100 for _, v in rc_sorted]
        w_vals = [weights_dict.get(t, 0)*100 for t in tickers_rc]
        names_rc = [TICKER_INFO.get(t, {}).get('name', t) for t in tickers_rc]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Risk Contribution', x=names_rc, y=rc_vals,
                              marker_color='#FF8C00', text=[f"{v:.1f}%" for v in rc_vals],
                              textposition='outside', textfont=dict(size=9)))
        fig.add_trace(go.Bar(name='Weight', x=names_rc, y=w_vals,
                              marker_color='#555', text=[f"{v:.1f}%" for v in w_vals],
                              textposition='outside', textfont=dict(size=9)))
        fig.update_layout(barmode='group', height=400, yaxis_title="%", showlegend=True)
        st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)
        
        # Individual Asset Volatility Table
        st.markdown(bb_section("INDIVIDUAL ASSET VOLATILITY"), unsafe_allow_html=True)
        
        vol_data = []
        cov = m['cov_matrix']
        for t in sorted(m['available']):
            info = TICKER_INFO.get(t, {})
            if t in cov.columns:
                vol_ann = np.sqrt(cov.loc[t, t]) * 100
                vol_mon = vol_ann / np.sqrt(12)
            else:
                vol_ann = 0
                vol_mon = 0
            w = st.session_state.portfolio.get(t, 0) * 100
            bar_len = int(vol_ann / 2)
            bar = '█' * min(bar_len, 30)
            vol_data.append({
                'Ticker': t,
                'Name': info.get('name', t),
                'Weight': f"{w:.1f}%",
                'Vol (Ann.)': f"{vol_ann:.1f}%",
                'Vol (Mon.)': f"{vol_mon:.1f}%",
                'Risk Level': bar,
            })
        
        vol_data.sort(key=lambda x: float(x['Vol (Ann.)'].replace('%','')), reverse=True)
        st.dataframe(pd.DataFrame(vol_data), use_container_width=True, hide_index=True)
        
        # Volatility
        st.markdown(bb_section("PORTFOLIO VOLATILITY"), unsafe_allow_html=True)
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.markdown(bb_metric_html("σ(p) Diversified", f"{m['vol']*100:.2f}%"), unsafe_allow_html=True)
        with vc2:
            st.markdown(bb_metric_html("σ(p) Undiversified", f"{m['vol_undiv']*100:.2f}%"), unsafe_allow_html=True)
        with vc3:
            st.markdown(bb_metric_html("Diversification Benefit", f"{m['div_benefit']*100:.1f}%"), unsafe_allow_html=True)
        
        # Drawdowns
        st.markdown(bb_section("DRAWDOWNS"), unsafe_allow_html=True)
        
        cum = pd.Series(m['cumulative'], index=m['dates'][:len(m['cumulative'])])
        running_max = cum.cummax()
        dd_series = (cum - running_max) / running_max * 100
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values,
                                  fill='tozeroy', fillcolor='rgba(255,23,68,0.3)',
                                  line=dict(color='#FF1744', width=1)))
        fig.update_layout(yaxis_title="Drawdown %", height=300, showlegend=False)
        st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 4: EXPOSURE
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first --> go to PORTFOLIO tab and click ANALYZE</div>',
                    unsafe_allow_html=True)
    else:
        m = st.session_state.metrics
        
        st.markdown(bb_section("ASSET CLASS ALLOCATION"), unsafe_allow_html=True)
        
        ac1, ac2 = st.columns(2)
        
        # Asset Class
        with ac1:
            ac_weights = {}
            for t, w in st.session_state.portfolio.items():
                info = TICKER_INFO.get(t, {})
                ac = info.get('asset_class', 'Other')
                ac_weights[ac] = ac_weights.get(ac, 0) + w
            
            fig = go.Figure(data=[go.Pie(
                labels=list(ac_weights.keys()),
                values=[v*100 for v in ac_weights.values()],
                hole=0.5,
                textinfo='label+percent',
                marker=dict(colors=['#FF8C00', '#FFD600', '#00C853', '#4488FF', '#CC66FF']),
            )])
            fig.update_layout(title="Asset Class", height=350, showlegend=False)
            st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)
        
        # Sector
        with ac2:
            sector_w = m['sector_weights']
            fig = go.Figure(data=[go.Pie(
                labels=list(sector_w.keys()),
                values=[v*100 for v in sector_w.values()],
                hole=0.5,
                textinfo='label+percent',
            )])
            fig.update_layout(title="Sector", height=350, showlegend=False)
            st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)
        
        # Geography
        st.markdown(bb_section("GEOGRAPHIC EXPOSURE"), unsafe_allow_html=True)
        
        geo_weights = {}
        for t, w in st.session_state.portfolio.items():
            info = TICKER_INFO.get(t, {})
            mkt = info.get('market', 'US')
            country = MARKET_NAMES.get(mkt, mkt)
            geo_weights[country] = geo_weights.get(country, 0) + w
        
        fig = go.Figure(data=[go.Pie(
            labels=list(geo_weights.keys()),
            values=[v*100 for v in geo_weights.values()],
            hole=0.5,
            textinfo='label+percent',
            marker=dict(colors=['#FF8C00', '#4488FF', '#00C853', '#FFD600', '#CC66FF', '#FF1744']),
        )])
        fig.update_layout(title="Geography", height=350, showlegend=False)
        st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)
        
        # Holdings table
        st.markdown(bb_section("HOLDINGS DETAIL"), unsafe_allow_html=True)
        
        holdings = []
        for t, w in st.session_state.portfolio.items():
            info = TICKER_INFO.get(t, {})
            holdings.append({
                'Ticker': t,
                'Name': info.get('name', t),
                'Weight': f"{w*100:.1f}%",
                'Class': info.get('asset_class', 'Other'),
                'Sector': info.get('sector', 'Other'),
                'Market': MARKET_NAMES.get(info.get('market', ''), info.get('market', '')),
                'Amount': f"€{st.session_state.portfolio_value * w:,.0f}",
            })
        st.dataframe(pd.DataFrame(holdings), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 5: SIMULATOR
# ═══════════════════════════════════════════════════════════
with tabs[5]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first --> go to PORTFOLIO tab and click ANALYZE</div>',
                    unsafe_allow_html=True)
    else:
        m = st.session_state.metrics
        
        st.markdown(bb_section("WEIGHT SIMULATOR"), unsafe_allow_html=True)
        st.markdown('<div style="color:#808080; font-size:0.78rem; margin-bottom:16px;">Adjust weights below to simulate alternative allocations. Metrics update in real-time.</div>', unsafe_allow_html=True)
        
        sim_weights = {}
        cols = st.columns(4)
        for i, (t, w) in enumerate(st.session_state.portfolio.items()):
            if t == 'CASH':
                continue
            with cols[i % 4]:
                new_w = st.slider(t, 0.0, 30.0, round(w*100, 2), 0.1, key=f"sim_{t}")
                sim_weights[t] = new_w / 100
        
        # Calculate simulated metrics
        if st.session_state.prices is not None and len(sim_weights) > 0:
            rf = m['rf']
            g = GORDON_SCENARIOS.get(st.session_state.get('scenario', 'Base'), 0.06)
            sim_metrics = calculate_portfolio_metrics(st.session_state.prices, sim_weights, rf, g)
            
            if sim_metrics:
                st.markdown(bb_section("COMPARISON: CURRENT VS SIMULATED"), unsafe_allow_html=True)
                
                comp_data = [
                    {'Metric': 'E(Rp) Forward', 'Current': f"{m['erp_portfolio']*100:.2f}%", 'Simulated': f"{sim_metrics['erp_portfolio']*100:.2f}%"},
                    {'Metric': 'CAGR Historical', 'Current': f"{m['cagr']*100:.1f}%", 'Simulated': f"{sim_metrics['cagr']*100:.1f}%"},
                    {'Metric': 'Volatility', 'Current': f"{m['vol']*100:.2f}%", 'Simulated': f"{sim_metrics['vol']*100:.2f}%"},
                    {'Metric': 'Sharpe (Forward)', 'Current': f"{m['sharpe_fwd']:.2f}", 'Simulated': f"{sim_metrics['sharpe_fwd']:.2f}"},
                    {'Metric': 'Sharpe (Historical)', 'Current': f"{m['sharpe_hist']:.2f}", 'Simulated': f"{sim_metrics['sharpe_hist']:.2f}"},
                    {'Metric': 'Max Drawdown', 'Current': f"{m['max_dd']*100:.1f}%", 'Simulated': f"{sim_metrics['max_dd']*100:.1f}%"},
                    {'Metric': 'Diversification', 'Current': f"{m['div_benefit']*100:.1f}%", 'Simulated': f"{sim_metrics['div_benefit']*100:.1f}%"},
                ]
                st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 6: METHODOLOGY
# ═══════════════════════════════════════════════════════════
with tabs[6]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first to see methodology details for your specific portfolio.</div>',
                    unsafe_allow_html=True)
    else:
        m = st.session_state.metrics
        st.markdown(bb_section("METHODOLOGY & SOURCES"), unsafe_allow_html=True)
        
        # Build dynamic methodology based on actual portfolio
        method_sections = []
        
        # 1. Gordon Growth - always present
        method_sections.append(("Market Return (Rm) - Gordon Growth Model",
            f"Formula: Rm = Shareholder Yield + g<br><br>"
            f"• Dividend Yield: {m['dy']*100:.2f}% ({m['dy_source']})<br>"
            f"• Buyback Yield: 1.50% (S&P 500 average estimate)<br>"
            f"• g: {GORDON_SCENARIOS.get(st.session_state.get('scenario','Base'),0.06)*100:.1f}% "
            f"(Scenario: {st.session_state.get('scenario','Base')})<br>"
            f"• Rm = {m['rm']*100:.2f}%<br>"
            f"• g is the primary source of uncertainty in the model"))
        
        # 2. Rf - always present
        method_sections.append(("Risk-Free Rate (Rf)",
            f"Source: US Treasury 10Y (^TNX via Yahoo Finance)<br><br>"
            f"• Current value: {m['rf']*100:.2f}%<br>"
            f"• Updated on each analysis run<br>"
            f"• For EUR portfolios, German Bund 10Y would be more appropriate<br>"
            f"• High Rf compresses ERP and reduces forward Sharpe"))
        
        # 3. Betas - dynamic based on which benchmarks are used
        benchmarks_used = set()
        low_r2_assets = []
        for t in m['available']:
            info = TICKER_INFO.get(t, {})
            mkt = info.get('market', 'US')
            if mkt in BENCHMARK_MAP and BENCHMARK_MAP[mkt]:
                benchmarks_used.add(BENCHMARK_MAP[mkt])
            r2 = m['r_squareds'].get(t)
            if r2 is not None and r2 < 0.10:
                low_r2_assets.append(f"{t} (R2={r2:.2f})")
        
        beta_text = (f"Method: Cov(Ri, Rm) / Var(Rm)<br><br>"
            f"• Period: {m['n_months']} months, monthly frequency<br>"
            f"• Benchmarks used: {', '.join(sorted(benchmarks_used))}<br>"
            f"• Local beta + global ERP is a simplification")
        if low_r2_assets:
            beta_text += f"<br>• Low R2 (unreliable beta): {', '.join(low_r2_assets)}"
        method_sections.append(("Betas", beta_text))
        
        # 4. Non-CAPM Assets - only if portfolio has them
        non_capm = []
        for t in m['available']:
            info = TICKER_INFO.get(t, {})
            ac = info.get('asset_class', 'Equity')
            if ac in ('Commodity', 'Crypto', 'Cash'):
                non_capm.append((t, info.get('name', t), ac))
        
        has_cash = 'CASH' in st.session_state.portfolio
        if has_cash:
            non_capm.append(('CASH', 'Cash Position', 'Cash'))
        
        if non_capm:
            nc_text = "The following assets do not have interpretable market betas:<br><br>"
            for t, name, ac in non_capm:
                src = m['data_sources'].get(t, '')
                er = m['expected_returns'].get(t, 0)
                if ac == 'Cash':
                    nc_text += f"• {t} ({name}): Return = Rf ({m['rf']*100:.2f}%)<br>"
                elif ac == 'Commodity':
                    nc_text += f"• {t} ({name}): Estimated return = {er*100:.2f}% (blend of CAGR + long-term assumption)<br>"
                elif ac == 'Crypto':
                    nc_text += f"• {t} ({name}): Estimated return = {er*100:.2f}% (CAGR with 70% discount)<br>"
            nc_text += "<br>All non-CAPM assets are included in risk calculations (correlations, volatility)."
            method_sections.append(("Non-CAPM Assets", nc_text))
        
        # 5. Correlations - always present
        n = m['n_months']
        thresh = round(2 / (n ** 0.5), 2)
        method_sections.append(("Correlations",
            f"• Period: {n} months ({n/12:.1f} years) of monthly observations<br>"
            f"• Significance threshold (95%): |r| > {thresh}<br>"
            f"• Correlations are historical, not forward-looking<br>"
            f"• In crisis, correlations tend to converge toward +1"))
        
        # 6. Volatility - always present
        method_sections.append(("Portfolio Volatility",
            f"Method: sigma_p = sqrt(w' Sigma w) with annualized covariances<br><br>"
            f"• Portfolio volatility: {m['vol']*100:.2f}%<br>"
            f"• Undiversified: {m['vol_undiv']*100:.2f}%<br>"
            f"• Diversification benefit: {m['div_benefit']*100:.1f}%<br>"
            f"• Uses historical volatility, not implied/forward"))
        
        # 7. KPIs - always present
        method_sections.append(("KPI Thresholds",
            "Thresholds are configurable reference values, not universal rules.<br><br>"
            "• Forward Sharpe mixes forward returns with backward volatility<br>"
            "• This is the best approximation without options data (implied vol)"))
        
        # 8. Proxies - only if proxies are used
        proxy_map = {
            'ABBN.SW': 'ABBN:CH (Swiss exchange listing)',
            'SAN.MC': 'BNC:GB (Same bank, Madrid listing)',
            'EEM': 'XMEM:DE (iShares EM equivalent)',
            'GLD': 'SGLN:GB (Gold ETC equivalent)',
            'EUFN': 'X7PP:GB (EU Financials equivalent)',
            'ITA': 'DFEN:DE (Defense sector equivalent)',
        }
        proxies_in_use = []
        for t in m['available']:
            if t in proxy_map:
                proxies_in_use.append(f"• {proxy_map[t]} --> {t}")
        
        if proxies_in_use:
            proxy_text = "The following assets use proxy tickers for data availability:<br><br>"
            proxy_text += "<br>".join(proxies_in_use)
            proxy_text += "<br><br>Proxies may not exactly replicate the original asset."
            method_sections.append(("Proxies Used", proxy_text))
        
        # 9. Disclaimer - always present
        method_sections.append(("Disclaimer",
            "• These returns are estimates based on models, not predictions<br>"
            "• CAPM has well-known limitations<br>"
            "• Inputs (g, Rf, beta) are uncertain<br>"
            "• Using scenarios is more honest than a single number<br>"
            "• Past performance is not indicative of future results"))
        
        for title, meth_content in method_sections:
            st.markdown(f'''<div class="bb-card" style="margin-bottom:6px;">
                <div class="bb-card-title">{title}</div>
                <div style="font-size:0.78rem; color:#d4d4d4; line-height:1.6;">{meth_content}</div>
            </div>''', unsafe_allow_html=True)
        
        # Dynamic data sources
        benchmarks_list = ", ".join(sorted(benchmarks_used)) if benchmarks_used else "N/A"
        st.markdown(bb_section("DATA SOURCES"), unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:0.78rem; color:#808080; line-height:1.8;">
            <span style="color:#FF8C00; font-weight:600;">PRICES</span> Yahoo Finance (yfinance) - daily adjusted close<br>
            <span style="color:#FF8C00; font-weight:600;">RISK-FREE</span> US Treasury 10Y (^TNX) - Yahoo Finance ({m['rf']*100:.2f}%)<br>
            <span style="color:#FF8C00; font-weight:600;">DIVIDENDS</span> SPY trailing 12-month dividends ({m['dy']*100:.2f}%)<br>
            <span style="color:#FF8C00; font-weight:600;">BENCHMARKS</span> {benchmarks_list} - Yahoo Finance<br>
            <span style="color:#FF8C00; font-weight:600;">OBSERVATIONS</span> {m['n_months']} months of monthly data
        </div>
        """, unsafe_allow_html=True)
