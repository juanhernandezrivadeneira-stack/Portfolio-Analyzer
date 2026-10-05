"""
╔══════════════════════════════════════════════════════════════╗
║     PORTFOLIO ANALYZER — Streamlit App (Bloomberg Style)     ║
║     v5.0 — KPI Targets + Health Score                        ║
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


# ════════════════════════════════════════════════════════════════
# KPI SYSTEM
# ════════════════════════════════════════════════════════════════

# 7 KPIs with editable targets
KPI_DEFINITIONS = {
    'expected_ret': {'name': 'Expected Return E(Rp)', 'target': 'above', 'unit': '%',  'yellow_factor': 0.75, 'step': 0.5, 'fmt': '%.1f', 'min': 0.0,   'max': 30.0},
    'sharpe_fwd':   {'name': 'Sharpe (Forward)',      'target': 'above', 'unit': '',   'yellow_factor': 0.50, 'step': 0.05,'fmt': '%.2f', 'min': 0.0,   'max': 3.0},
    'volatility':   {'name': 'Portfolio Volatility',  'target': 'below', 'unit': '%',  'yellow_factor': 1.33, 'step': 0.5, 'fmt': '%.1f', 'min': 1.0,   'max': 50.0},
    'max_dd':       {'name': 'Max Drawdown',          'target': 'above', 'unit': '%',  'yellow_factor': 1.67, 'step': 1.0, 'fmt': '%.1f', 'min': -60.0, 'max': 0.0},
    'sector_conc':  {'name': 'Max Sector Conc.',      'target': 'below', 'unit': '%',  'yellow_factor': 1.40, 'step': 1.0, 'fmt': '%.0f', 'min': 5.0,   'max': 100.0},
    'max_position': {'name': 'Max Single Position',   'target': 'below', 'unit': '%',  'yellow_factor': 1.50, 'step': 1.0, 'fmt': '%.0f', 'min': 2.0,   'max': 50.0},
    'avg_corr':     {'name': 'Wtd Avg Correlation',   'target': 'below', 'unit': '',   'yellow_factor': 1.50, 'step': 0.05,'fmt': '%.2f', 'min': 0.0,   'max': 1.0},
}

KPI_TARGETS_BY_SCENARIO = {
    'Conservador': {'expected_ret': 4.0,  'sharpe_fwd': 0.25, 'volatility': 22.0, 'max_dd': -30.0, 'sector_conc': 35.0, 'max_position': 18.0, 'avg_corr': 0.45},
    'Moderado':    {'expected_ret': 6.0,  'sharpe_fwd': 0.35, 'volatility': 18.0, 'max_dd': -20.0, 'sector_conc': 30.0, 'max_position': 15.0, 'avg_corr': 0.35},
    'Base':        {'expected_ret': 8.0,  'sharpe_fwd': 0.50, 'volatility': 15.0, 'max_dd': -15.0, 'sector_conc': 25.0, 'max_position': 10.0, 'avg_corr': 0.30},
    'Crecimiento': {'expected_ret': 10.0, 'sharpe_fwd': 0.65, 'volatility': 13.0, 'max_dd': -12.0, 'sector_conc': 22.0, 'max_position': 10.0, 'avg_corr': 0.28},
    'Optimista':   {'expected_ret': 12.0, 'sharpe_fwd': 0.80, 'volatility': 10.0, 'max_dd': -10.0, 'sector_conc': 20.0, 'max_position': 8.0,  'avg_corr': 0.25},
}

# 8 informational KPIs (no user threshold, used in Health Score)
INFO_KPIS = {
    'div_benefit':    {'name': 'Diversification Benefit', 'unit': '%'},
    'sharpe_hist':    {'name': 'Sharpe (Historical)',     'unit': ''},
    'cagr':           {'name': 'CAGR (Historical)',       'unit': '%'},
    'sortino':        {'name': 'Sortino Ratio',           'unit': ''},
    'var_5':          {'name': 'VaR 5% (Monthly)',        'unit': '%'},
    'pos_periods':    {'name': '% Positive Periods',      'unit': '%'},
    'gain_loss':      {'name': 'Gain/Loss Ratio',         'unit': ''},
    'max_risk_contr': {'name': 'Max Risk Contribution',   'unit': '%'},
}


# ════════════════════════════════════════════════════════════════
# SAFE TEXT
# ════════════════════════════════════════════════════════════════
def safe(text):
    if not isinstance(text, str):
        return str(text)
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u00b2": "2",
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


# ════════════════════════════════════════════════════════════════
# KPI EVALUATION & HEALTH SCORE
# ════════════════════════════════════════════════════════════════
def evaluate_kpi(value, target_value, kpi_key):
    defn = KPI_DEFINITIONS[kpi_key]
    direction = defn['target']
    yellow_val = target_value * defn['yellow_factor']
    if direction == 'above':
        if value >= target_value: return 'green'
        elif value >= yellow_val: return 'yellow'
        else: return 'red'
    else:
        if value <= target_value: return 'green'
        elif value <= yellow_val: return 'yellow'
        else: return 'red'


def normalize_kpi(value, min_val, max_val, inverted=False):
    if inverted:
        if value <= min_val: return 100
        if value >= max_val: return 0
        return ((max_val - value) / (max_val - min_val)) * 100
    else:
        if value <= min_val: return 0
        if value >= max_val: return 100
        return ((value - min_val) / (max_val - min_val)) * 100


def score_from_threshold(value, target_value, kpi_key):
    status = evaluate_kpi(value, target_value, kpi_key)
    return {'green': 100, 'yellow': 60, 'red': 30}[status]


def calculate_health_score(metrics, active_targets):
    m = metrics
    # 1. Return Potential (20%)
    er_s = score_from_threshold(m['erp_portfolio']*100, active_targets['expected_ret'], 'expected_ret')
    sf_s = score_from_threshold(m['sharpe_fwd'], active_targets['sharpe_fwd'], 'sharpe_fwd')
    cagr_s = normalize_kpi(m['cagr']*100, -5, 20)
    ret = er_s * 0.4 + sf_s * 0.4 + cagr_s * 0.2

    # 2. Risk Management (25%)
    vol_s = score_from_threshold(m['vol']*100, active_targets['volatility'], 'volatility')
    dd_s = score_from_threshold(m['max_dd']*100, active_targets['max_dd'], 'max_dd')
    sh_s = normalize_kpi(m['sharpe_hist'], 0, 1.5)
    so_s = normalize_kpi(m['sortino'], 0, 2.0)
    var_s = normalize_kpi(m['var_5']*100, -15, -2)
    risk = vol_s * 0.30 + dd_s * 0.30 + sh_s * 0.15 + so_s * 0.15 + var_s * 0.10

    # 3. Diversification (20%)
    db_s = normalize_kpi(m['div_benefit']*100, 10, 45)
    co_s = score_from_threshold(m.get('avg_corr', 0), active_targets['avg_corr'], 'avg_corr')
    div = db_s * 0.50 + co_s * 0.50

    # 4. Concentration Risk (20%)
    sc_s = score_from_threshold(m['max_sector']*100, active_targets['sector_conc'], 'sector_conc')
    mp_s = score_from_threshold(m['max_position']*100, active_targets['max_position'], 'max_position')
    rc_s = normalize_kpi(m['max_rc']*100, 10, 30, inverted=True)
    conc = sc_s * 0.40 + mp_s * 0.40 + rc_s * 0.20

    # 5. Track Record (15%)
    wr = (m['pos_periods']/m['total_periods']*100) if m['total_periods'] > 0 else 50
    wr_s = normalize_kpi(wr, 40, 70)
    gl_s = normalize_kpi(m['gain_loss'], 0.5, 2.5)
    track = wr_s * 0.50 + gl_s * 0.50

    total = ret * 0.20 + risk * 0.25 + div * 0.20 + conc * 0.20 + track * 0.15

    return {
        'total': total,
        'dimensions': {
            'return': ret, 'risk': risk, 'diversification': div,
            'concentration': conc, 'track_record': track,
        }
    }


def get_active_targets():
    return {k: st.session_state.get(f'kpi_t_{k}', KPI_TARGETS_BY_SCENARIO['Base'][k]) for k in KPI_DEFINITIONS}


# ════════════════════════════════════════════════════════════════
# PDF REPORT
# ════════════════════════════════════════════════════════════════
def generate_pdf_report(metrics, portfolio, portfolio_value, scenario, ticker_info,
                        active_thresholds, evaluate_kpi_fn, health_score=None):
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    m = metrics
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    BG = (10, 10, 10)
    BG2 = (20, 20, 20)
    TEXT = (212, 212, 212)
    ACCENT = (255, 140, 0)
    GREEN = (0, 200, 83)
    RED = (255, 23, 68)
    YELLOW = (255, 214, 0)
    GRAY = (128, 128, 128)

    def dark_page(p):
        p.set_fill_color(*BG)
        p.rect(0, 0, 210, 297, 'F')

    # ── Page 1: Summary ──────────────────────────────────────
    pdf.add_page()
    dark_page(pdf)
    pdf.set_fill_color(*BG2)
    pdf.rect(0, 0, 210, 22, 'F')
    pdf.set_draw_color(*ACCENT)
    pdf.line(0, 22, 210, 22)

    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(*ACCENT)
    pdf.set_y(6)
    pdf.cell(0, 10, 'PORTFOLIO ANALYZER - Report', ln=True)

    pdf.set_font('Helvetica', '', 9)
    pdf.set_text_color(*GRAY)
    pdf.cell(0, 5, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', ln=True)
    pdf.cell(0, 5, safe(f'Scenario: {scenario} | Value: EUR {portfolio_value:,.0f}'), ln=True)
    pdf.ln(6)

    pdf.set_text_color(*ACCENT)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'PORTFOLIO OVERVIEW', ln=True)
    pdf.set_draw_color(*ACCENT)
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
        ('Wtd Avg Correlation', f"{m.get('avg_corr',0):.2f}"),
        ('Risk-Free Rate', f"{m['rf']*100:.2f}%"),
        ('Market Return (Rm)', f"{m['rm']*100:.2f}%"),
        ('Equity Risk Premium', f"{m['erp']*100:.2f}%"),
        ('Observations', f"{m['n_months']} months"),
    ]
    for label, value in key_metrics:
        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(*GRAY)
        pdf.cell(90, 7, label, border=0)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(*TEXT)
        pdf.cell(0, 7, value, ln=True)

    # ── Page 2: Health Score ─────────────────────────────────
    if health_score is not None:
        pdf.add_page()
        dark_page(pdf)
        pdf.set_fill_color(*BG2)
        pdf.rect(0, 0, 210, 18, 'F')
        pdf.set_draw_color(*ACCENT)
        pdf.line(0, 18, 210, 18)
        pdf.set_font('Helvetica', 'B', 12)
        pdf.set_text_color(*ACCENT)
        pdf.set_y(5)
        pdf.cell(0, 8, 'PORTFOLIO HEALTH SCORE', ln=True)
        pdf.ln(10)

        total = health_score['total']
        dims = health_score['dimensions']

        if total >= 90:
            sc_color, sc_label = GREEN, 'EXCELLENT'
        elif total >= 75:
            sc_color, sc_label = GREEN, 'STRONG'
        elif total >= 60:
            sc_color, sc_label = YELLOW, 'GOOD'
        elif total >= 40:
            sc_color, sc_label = ACCENT, 'FAIR'
        else:
            sc_color, sc_label = RED, 'WEAK'

        pdf.set_font('Helvetica', 'B', 54)
        pdf.set_text_color(*sc_color)
        pdf.cell(0, 28, f"{total:.0f}", ln=True, align='C')
        pdf.set_font('Helvetica', 'B', 14)
        pdf.cell(0, 8, sc_label, ln=True, align='C')
        pdf.ln(12)

        dim_labels = [
            ('return', 'Return Potential'),
            ('risk', 'Risk Management'),
            ('diversification', 'Diversification'),
            ('concentration', 'Concentration Risk'),
            ('track_record', 'Track Record'),
        ]
        dim_weights = [20, 25, 20, 20, 15]

        for i, (key, name) in enumerate(dim_labels):
            score = dims.get(key, 0)
            if score >= 75:
                bar_c = GREEN
            elif score >= 60:
                bar_c = YELLOW
            elif score >= 40:
                bar_c = ACCENT
            else:
                bar_c = RED

            y0 = pdf.get_y()
            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(*TEXT)
            pdf.cell(65, 10, safe(f"{name} ({dim_weights[i]}%)"))

            bar_x = 80
            bar_y = y0 + 3
            bar_max = 80
            pdf.set_fill_color(42, 42, 42)
            pdf.rect(bar_x, bar_y, bar_max, 5, 'F')
            bar_w = max(1, score / 100 * bar_max)
            pdf.set_fill_color(*bar_c)
            pdf.rect(bar_x, bar_y, bar_w, 5, 'F')

            pdf.set_xy(165, y0)
            pdf.set_font('Helvetica', 'B', 9)
            pdf.set_text_color(*bar_c)
            pdf.cell(30, 10, f"{score:.0f}/100", align='R')
            pdf.ln(12)

        pdf.ln(6)
        pdf.set_font('Helvetica', 'I', 7)
        pdf.set_text_color(*GRAY)
        pdf.multi_cell(0, 3.5,
            "Health Score: weighted average of 5 dimensions. Threshold-based KPIs score 100 (green), "
            "60 (yellow), 30 (red). Informational KPIs are normalized to 0-100 using predefined ranges.")

    # ── Page 3: Holdings ─────────────────────────────────────
    pdf.add_page()
    dark_page(pdf)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 10, 'HOLDINGS & EXPECTED RETURNS', ln=True)
    pdf.set_draw_color(*ACCENT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    col_w = [22, 38, 16, 14, 14, 16, 18, 20, 22]
    headers = ['Ticker', 'Name', 'Weight', 'Beta', 'R2', 'E(Ri)', 'Vol%', 'Bench', 'Source']
    pdf.set_font('Helvetica', 'B', 7)
    pdf.set_text_color(*ACCENT)
    for i, h in enumerate(headers):
        pdf.cell(col_w[i], 7, h, border=0)
    pdf.ln()
    pdf.set_draw_color(42, 42, 42)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())

    pdf.set_text_color(*TEXT)
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
            pdf.cell(col_w[i], 6, val, border=0)
        pdf.ln()
        pdf.set_draw_color(42, 42, 42)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())

    # Cash row
    if 'CASH' in portfolio:
        cw = portfolio['CASH']
        row = ['CASH', 'Cash Position', f"{cw*100:.1f}%", '-', '-', f"{m['rf']*100:.1f}%", '0.0%', 'N/A', 'CASH']
        for i, val in enumerate(row):
            pdf.cell(col_w[i], 6, val, border=0)
        pdf.ln()

    # ── Page 4: Risk Contribution ────────────────────────────
    pdf.add_page()
    dark_page(pdf)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 10, 'RISK CONTRIBUTION', ln=True)
    pdf.set_draw_color(*ACCENT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    rc_w = [25, 50, 25, 30, 25]
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(*ACCENT)
    for i, h in enumerate(['Ticker', 'Name', 'Weight', 'Risk Contr.', 'Ratio']):
        pdf.cell(rc_w[i], 7, h, border=0)
    pdf.ln()
    pdf.set_draw_color(42, 42, 42)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())

    pdf.set_text_color(*TEXT)
    pdf.set_font('Helvetica', '', 8)
    for t, rc_val in sorted(m['risk_contrib'].items(), key=lambda x: x[1], reverse=True):
        info = ticker_info.get(t, {})
        w = portfolio.get(t, 0)
        ratio = rc_val / w if w > 0 else 0
        for i, val in enumerate([safe(t), safe(info.get('name', t)[:25]), f"{w*100:.1f}%", f"{rc_val*100:.1f}%", f"{ratio:.2f}x"]):
            pdf.cell(rc_w[i], 6, val, border=0)
        pdf.ln()
        pdf.set_draw_color(42, 42, 42)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())

    # ── Page 5: Correlation Matrix (Landscape) ───────────────
    if hasattr(m, '__contains__') and 'corr_matrix' in m and m['corr_matrix'] is not None:
        corr = m['corr_matrix']
        n_assets = len(corr.columns)
        if n_assets > 1:
            pdf.add_page('L')
            pdf.set_fill_color(*BG)
            pdf.rect(0, 0, 297, 210, 'F')
            pdf.set_fill_color(*BG2)
            pdf.rect(0, 0, 297, 14, 'F')
            pdf.set_draw_color(*ACCENT)
            pdf.line(0, 14, 297, 14)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.set_text_color(*ACCENT)
            pdf.set_y(3)
            pdf.cell(0, 8, 'CORRELATION MATRIX', ln=True)
            pdf.ln(4)

            labels = [t[:6] for t in corr.columns]
            label_w = 18
            avail_w = 270 - label_w
            cell_w = min(avail_w / n_assets, 18)
            fs = max(4.5, min(6, 90 / n_assets))

            pdf.set_font('Helvetica', 'B', fs)
            pdf.set_text_color(*ACCENT)
            pdf.cell(label_w, 5, '', border=0)
            for lbl in labels:
                pdf.cell(cell_w, 5, lbl, border=0, align='C')
            pdf.ln()

            for i, row_label in enumerate(labels):
                pdf.set_font('Helvetica', 'B', fs)
                pdf.set_text_color(*ACCENT)
                pdf.cell(label_w, 5, row_label, border=0)
                for j in range(n_assets):
                    val = corr.iloc[i, j]
                    if i == j:
                        pdf.set_fill_color(42, 42, 42)
                        pdf.set_text_color(*GRAY)
                    elif val > 0.6:
                        pdf.set_fill_color(80, 10, 10)
                        pdf.set_text_color(*RED)
                    elif val > 0.3:
                        pdf.set_fill_color(60, 40, 0)
                        pdf.set_text_color(*ACCENT)
                    elif val < -0.1:
                        pdf.set_fill_color(0, 40, 20)
                        pdf.set_text_color(*GREEN)
                    else:
                        pdf.set_fill_color(20, 20, 20)
                        pdf.set_text_color(*TEXT)
                    pdf.set_font('Helvetica', '', fs)
                    pdf.cell(cell_w, 5, f"{val:.2f}", border=0, align='C', fill=True)
                pdf.ln()

            pdf.ln(4)
            pdf.set_font('Helvetica', 'I', 6)
            pdf.set_text_color(*GRAY)
            n_obs = m['n_months']
            pdf.cell(0, 4, safe(f"Observations: {n_obs} months | Significance (95%): |r| > {2/np.sqrt(n_obs):.2f}"), ln=True)

    # ── Page 6: Exposure ─────────────────────────────────────
    pdf.add_page()
    dark_page(pdf)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 10, 'EXPOSURE ANALYSIS', ln=True)
    pdf.set_draw_color(*ACCENT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 7, 'Sector Allocation:', ln=True)
    pdf.set_font('Helvetica', '', 9)
    mn = {'US': 'United States', 'MC': 'Spain', 'DE': 'Germany', 'SW': 'Switzerland',
          'COMMODITY': 'Commodities', 'CRYPTO': 'Crypto', 'CASH': 'Cash',
          'L': 'United Kingdom', 'AS': 'Netherlands', 'PA': 'France'}
    for s, w in sorted(m['sector_weights'].items(), key=lambda x: x[1], reverse=True):
        y0 = pdf.get_y()
        pdf.set_text_color(*GRAY)
        pdf.cell(70, 6, safe(s))
        pdf.set_text_color(*TEXT)
        pdf.cell(20, 6, f"{w*100:.1f}%")
        bar_w = int(w * 100 * 0.8)
        pdf.set_fill_color(*ACCENT)
        pdf.rect(100, y0 + 1.5, max(1, bar_w), 3, 'F')
        pdf.ln()

    pdf.ln(6)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 7, 'Geographic Allocation:', ln=True)
    pdf.set_font('Helvetica', '', 9)
    geo = {}
    for t, w in portfolio.items():
        info = ticker_info.get(t, {})
        c = mn.get(info.get('market', 'US'), info.get('market', 'US'))
        geo[c] = geo.get(c, 0) + w
    for c, w in sorted(geo.items(), key=lambda x: x[1], reverse=True):
        y0 = pdf.get_y()
        pdf.set_text_color(*GRAY)
        pdf.cell(70, 6, safe(c))
        pdf.set_text_color(*TEXT)
        pdf.cell(20, 6, f"{w*100:.1f}%")
        bar_w = int(w * 100 * 0.8)
        pdf.set_fill_color(68, 136, 255)
        pdf.rect(100, y0 + 1.5, max(1, bar_w), 3, 'F')
        pdf.ln()

    # ── Page 7: KPI Evaluation ───────────────────────────────
    pdf.add_page()
    dark_page(pdf)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 10, 'KPI EVALUATION', ln=True)
    pdf.set_draw_color(*ACCENT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)

    kpi_vals = {
        'expected_ret': m['erp_portfolio'] * 100, 'sharpe_fwd': m['sharpe_fwd'],
        'volatility': m['vol'] * 100, 'max_dd': m['max_dd'] * 100,
        'sector_conc': m['max_sector'] * 100, 'max_position': m['max_position'] * 100,
        'avg_corr': m.get('avg_corr', 0),
    }

    kw = [55, 25, 20, 25, 25]
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(*ACCENT)
    for i, h in enumerate(['KPI', 'Value', 'Status', 'Target', 'Direction']):
        pdf.cell(kw[i], 7, h, border=0)
    pdf.ln()
    pdf.set_draw_color(42, 42, 42)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())

    pdf.set_font('Helvetica', '', 8)
    for key, defn in KPI_DEFINITIONS.items():
        val = kpi_vals.get(key, 0)
        target_val = active_thresholds.get(key, 0)
        status = evaluate_kpi_fn(val, target_val, key)
        st_text = {'green': 'PASS', 'yellow': 'WATCH', 'red': 'FAIL'}[status]
        if status == 'green':
            pdf.set_text_color(*GREEN)
        elif status == 'yellow':
            pdf.set_text_color(*YELLOW)
        else:
            pdf.set_text_color(*RED)

        if defn['unit'] == '%':
            vs = f"{val:.1f}%"
            ts = f"{target_val:.1f}%"
        else:
            vs = f"{val:.2f}"
            ts = f"{target_val:.2f}"
        direction = '>' if defn['target'] == 'above' else '<'

        pdf.set_text_color(*TEXT)
        pdf.cell(kw[0], 6, safe(defn['name']), border=0)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(kw[1], 6, vs, border=0)
        if status == 'green':
            pdf.set_text_color(*GREEN)
        elif status == 'yellow':
            pdf.set_text_color(*YELLOW)
        else:
            pdf.set_text_color(*RED)
        pdf.cell(kw[2], 6, st_text, border=0)
        pdf.set_text_color(*GRAY)
        pdf.set_font('Helvetica', '', 8)
        pdf.cell(kw[3], 6, f"{direction} {ts}", border=0)
        pdf.cell(kw[4], 6, safe(f"Scenario: {scenario}"), border=0)
        pdf.ln()
        pdf.set_draw_color(42, 42, 42)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())

    # Additional Metrics
    pdf.ln(6)
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(*ACCENT)
    pdf.cell(0, 7, 'ADDITIONAL METRICS', ln=True)
    pdf.set_draw_color(*ACCENT)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(2)

    info_vals = {
        'div_benefit': m['div_benefit'] * 100, 'sharpe_hist': m['sharpe_hist'],
        'cagr': m['cagr'] * 100, 'sortino': m['sortino'], 'var_5': m['var_5'] * 100,
        'pos_periods': m['pos_periods']/m['total_periods']*100 if m['total_periods'] > 0 else 0,
        'gain_loss': m['gain_loss'], 'max_risk_contr': m['max_rc'] * 100,
    }
    pdf.set_font('Helvetica', '', 8)
    for key, defn in INFO_KPIS.items():
        val = info_vals.get(key, 0)
        pdf.set_text_color(*GRAY)
        pdf.cell(90, 6, safe(defn['name']), border=0)
        pdf.set_text_color(*TEXT)
        pdf.set_font('Helvetica', 'B', 8)
        if defn['unit'] == '%':
            pdf.cell(0, 6, f"{val:.1f}%", ln=True)
        else:
            pdf.cell(0, 6, f"{val:.2f}", ln=True)
        pdf.set_font('Helvetica', '', 8)

    pdf.ln(8)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.set_text_color(*GRAY)
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
st.set_page_config(page_title="Portfolio Analyzer", page_icon="◆", layout="wide", initial_sidebar_state="collapsed")

# ════════════════════════════════════════════════════════════════
# BLOOMBERG TERMINAL CSS
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800&display=swap');

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

    .stApp { background-color: #0a0a0a !important; color: #d4d4d4 !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .stDeployButton {display: none;} div[data-testid="stToolbar"] {display: none;}

    .stTabs [data-baseweb="tab-list"] { gap: 0px; background-color: #0a0a0a; border-bottom: 2px solid #FF8C00; padding: 0; }
    .stTabs [data-baseweb="tab"] { padding: 8px 16px; font-weight: 600; font-size: 0.8rem; color: #808080; border-bottom: 2px solid transparent; letter-spacing: 0.05em; text-transform: uppercase; background-color: #0a0a0a; }
    .stTabs [aria-selected="true"] { color: #FF8C00 !important; border-bottom: 2px solid #FF8C00 !important; background-color: #1a1a1a !important; }
    .stTabs [data-baseweb="tab"]:hover { color: #FFa500 !important; }
    .stTabs [data-baseweb="tab-panel"] { background-color: #0a0a0a; }

    .bb-metric { background: #141414; border: 1px solid #2a2a2a; border-left: 3px solid #FF8C00; padding: 12px 16px; margin-bottom: 8px; }
    .bb-metric-label { font-size: 0.65rem; font-weight: 600; color: #FF8C00; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 2px; }
    .bb-metric-value { font-size: 1.5rem; font-weight: 700; color: #FFFFFF; line-height: 1.2; }
    .bb-metric-value-sm { font-size: 1.1rem; font-weight: 600; color: #FFFFFF; }
    .bb-metric-delta-pos { color: #00C853; font-size: 0.75rem; font-weight: 600; }
    .bb-metric-delta-neg { color: #FF1744; font-size: 0.75rem; font-weight: 600; }
    .bb-metric-delta-neutral { color: #808080; font-size: 0.75rem; font-weight: 600; }

    .bb-section { font-size: 0.75rem; font-weight: 700; color: #FF8C00; text-transform: uppercase; letter-spacing: 0.12em; margin: 1.5rem 0 0.75rem 0; padding-bottom: 4px; border-bottom: 1px solid #2a2a2a; }
    .bb-kpi-row { display: flex; align-items: center; padding: 6px 12px; border-bottom: 1px solid #1a1a1a; font-size: 0.78rem; }
    .bb-kpi-row:hover { background: #1a1a1a; }
    .bb-kpi-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; margin-right: 10px; flex-shrink: 0; }
    .bb-kpi-dot-green { background: #00C853; box-shadow: 0 0 4px #00C853; }
    .bb-kpi-dot-yellow { background: #FFD600; box-shadow: 0 0 4px #FFD600; }
    .bb-kpi-dot-red { background: #FF1744; box-shadow: 0 0 4px #FF1744; }
    .bb-kpi-dot-gray { background: #555; }
    .bb-kpi-name { flex: 1; color: #d4d4d4; }
    .bb-kpi-value { font-weight: 700; color: #FFFFFF; min-width: 80px; text-align: right; }

    .bb-card { background: #141414; border: 1px solid #2a2a2a; padding: 16px; margin-bottom: 10px; }
    .bb-card-title { font-size: 0.7rem; font-weight: 700; color: #FF8C00; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px; }
    .bb-badge { display: inline-block; font-size: 0.6rem; font-weight: 700; padding: 2px 6px; letter-spacing: 0.05em; }
    .bb-badge-calc { background: #1a3a1a; color: #00C853; border: 1px solid #00C853; }
    .bb-badge-est { background: #3a2a00; color: #FFD600; border: 1px solid #FFD600; }
    .bb-badge-proxy { background: #002244; color: #4488FF; border: 1px solid #4488FF; }
    .bb-badge-fallback { background: #3a1a1a; color: #FF1744; border: 1px solid #FF1744; }

    .bb-alert-critical { background: #1a0a0a; border-left: 3px solid #FF1744; padding: 8px 12px; margin-bottom: 6px; font-size: 0.78rem; color: #FF1744; }
    .bb-alert-warning { background: #1a1500; border-left: 3px solid #FFD600; padding: 8px 12px; margin-bottom: 6px; font-size: 0.78rem; color: #FFD600; }
    .bb-alert-info { background: #0a0a1a; border-left: 3px solid #4488FF; padding: 8px 12px; margin-bottom: 6px; font-size: 0.78rem; color: #4488FF; }

    .bb-progress-bg { background: #2a2a2a; height: 4px; width: 100%; margin-top: 6px; }
    .bb-progress-fill { height: 4px; transition: width 0.3s; }

    .streamlit-expanderHeader { background: #141414 !important; border: 1px solid #2a2a2a !important; color: #FF8C00 !important; }
    .streamlit-expanderContent { background: #0f0f0f !important; border: 1px solid #2a2a2a !important; border-top: none !important; }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #FF8C00; }

    .stNumberInput button { color: #FF8C00 !important; }
    .stRadio label { color: #d4d4d4 !important; }
    .stDataFrame { border: 1px solid #2a2a2a; }

    .bb-banner { position: relative; width: 100%; height: 80px; background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 50%, #1a1200 100%); border-bottom: 2px solid #FF8C00; display: flex; align-items: center; justify-content: space-between; padding: 0 24px; margin-bottom: 8px; overflow: hidden; }
    .bb-banner::before { content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: repeating-linear-gradient(90deg, transparent, transparent 60px, rgba(255,140,0,0.03) 60px, rgba(255,140,0,0.03) 61px); }
    .bb-banner-title { font-size: 1.2rem; font-weight: 800; letter-spacing: 0.15em; color: #FFFFFF; z-index: 1; }
    .bb-banner-title span { color: #FF8C00; }
    .bb-banner-sub { font-size: 0.65rem; color: #808080; letter-spacing: 0.08em; margin-top: 2px; }
    .bb-banner-version { font-size: 0.65rem; color: #555; z-index: 1; text-align: right; }
    .bb-banner-accent { position: absolute; right: 0; top: 0; bottom: 0; width: 200px; background: linear-gradient(90deg, transparent, rgba(255,140,0,0.05)); }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIG
# ════════════════════════════════════════════════════════════════
TICKER_INFO = {}

SPECIAL_TICKERS = {
    'GLD':     {'asset_class': 'Commodity', 'sector': 'Gold',           'market': 'COMMODITY'},
    'IAU':     {'asset_class': 'Commodity', 'sector': 'Gold',           'market': 'COMMODITY'},
    'SLV':     {'asset_class': 'Commodity', 'sector': 'Silver',         'market': 'COMMODITY'},
    'BTC-USD': {'asset_class': 'Crypto',    'sector': 'Cryptocurrency', 'market': 'CRYPTO'},
    'ETH-USD': {'asset_class': 'Crypto',    'sector': 'Cryptocurrency', 'market': 'CRYPTO'},
    'CASH':    {'asset_class': 'Cash',      'sector': 'Cash',           'market': 'CASH'},
}

ETF_SECTOR_FALLBACK = {
    'EEM': 'Diversified EM', 'VWO': 'Diversified EM', 'IEMG': 'Diversified EM',
    'EUFN': 'Financial Services', 'XLF': 'Financial Services', 'KBE': 'Financial Services',
    'ITA': 'Industrials', 'PPA': 'Industrials', 'DFEN': 'Industrials',
    'SPY': 'Diversified US', 'VOO': 'Diversified US', 'IVV': 'Diversified US',
    'QQQ': 'Technology', 'XLK': 'Technology', 'VGT': 'Technology',
    'XLE': 'Energy', 'VDE': 'Energy', 'XOP': 'Energy',
    'XLV': 'Healthcare', 'VHT': 'Healthcare', 'IBB': 'Healthcare',
    'XLI': 'Industrials', 'VIS': 'Industrials',
    'XLY': 'Consumer Discretionary', 'XLP': 'Consumer Staples',
    'XLU': 'Utilities', 'VNQ': 'Real Estate', 'XLRE': 'Real Estate',
    'XLC': 'Communication Services', 'XLB': 'Basic Materials',
    'AGG': 'Bonds', 'BND': 'Bonds', 'TLT': 'Bonds', 'LQD': 'Bonds', 'HYG': 'Bonds',
    'GLD': 'Gold', 'IAU': 'Gold', 'SLV': 'Silver',
    'VXUS': 'Diversified Intl', 'EFA': 'Diversified Intl', 'VEA': 'Diversified Intl',
    'VT': 'Diversified Global', 'ACWI': 'Diversified Global',
}

YFINANCE_SECTOR_MAP = {
    'Technology': 'Technology', 'Financial Services': 'Financial Services',
    'Healthcare': 'Healthcare', 'Consumer Cyclical': 'Consumer Discretionary',
    'Consumer Defensive': 'Consumer Staples', 'Industrials': 'Industrials',
    'Energy': 'Energy', 'Basic Materials': 'Basic Materials',
    'Communication Services': 'Communication Services', 'Utilities': 'Utilities',
    'Real Estate': 'Real Estate', 'Financial': 'Financial Services',
    'Finance': 'Financial Services', 'Consumer Goods': 'Consumer Discretionary',
    'Services': 'Other', 'Industrial Goods': 'Industrials', 'Conglomerates': 'Industrials',
}

BENCHMARK_MAP = {
    'US': '^GSPC', 'MC': '^IBEX', 'DE': '^GDAXI', 'SW': '^SSMI',
    'L': '^FTSE', 'AS': '^AEX', 'PA': '^FCHI',
}

GORDON_SCENARIOS = {
    'Conservador': 0.030, 'Moderado': 0.050, 'Base': 0.060,
    'Crecimiento': 0.073, 'Optimista': 0.090,
}

MARKET_NAMES = {
    'US': 'United States', 'MC': 'Spain', 'DE': 'Germany', 'SW': 'Switzerland',
    'L': 'United Kingdom', 'AS': 'Netherlands', 'PA': 'France',
    'COMMODITY': 'Commodities', 'CRYPTO': 'Crypto', 'CASH': 'Cash',
}

EUR_SUFFIXES = {'.MC', '.DE', '.SW', '.L', '.AS', '.PA', '.MI'}


# ════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ════════════════════════════════════════════════════════════════
def get_ticker_info():
    return st.session_state.get('ticker_info', TICKER_INFO)

def detect_market(ticker):
    if '.' in ticker:
        suffix = ticker.split('.')[-1].upper()
        suffix_map = {'MC': 'MC', 'DE': 'DE', 'SW': 'SW', 'L': 'L', 'AS': 'AS', 'PA': 'PA', 'MI': 'MI'}
        return suffix_map.get(suffix, 'US')
    return 'US'

def _classify_sector_from_text(text):
    if not text:
        return None
    t = text.lower()
    if any(k in t for k in ('bank', 'financ', 'insurance', 'capital market')): return 'Financial Services'
    if any(k in t for k in ('tech', 'software', 'semicond', 'internet', 'comput')): return 'Technology'
    if any(k in t for k in ('health', 'pharma', 'biotech', 'medical')): return 'Healthcare'
    if any(k in t for k in ('energy', 'oil', 'gas', 'coal', 'petroleum')): return 'Energy'
    if any(k in t for k in ('defense', 'aerospace', 'industrial', 'manufactur', 'machin')): return 'Industrials'
    if any(k in t for k in ('telecom', 'communicat', 'media', 'entertain')): return 'Communication Services'
    if any(k in t for k in ('consumer', 'retail', 'apparel', 'auto', 'luxury', 'textile')): return 'Consumer Discretionary'
    if any(k in t for k in ('food', 'beverage', 'household', 'tobacco', 'staple')): return 'Consumer Staples'
    if any(k in t for k in ('real estate', 'reit', 'property')): return 'Real Estate'
    if any(k in t for k in ('util', 'electric', 'water', 'power')): return 'Utilities'
    if any(k in t for k in ('material', 'mining', 'metal', 'chemical', 'steel')): return 'Basic Materials'
    if any(k in t for k in ('gold', 'precious')): return 'Gold'
    if any(k in t for k in ('emerging', ' em ')): return 'Diversified EM'
    return None

def detect_currency_mode(portfolio):
    eur_weight, total_weight = 0.0, 0.0
    for t, w in portfolio.items():
        if t == 'CASH': continue
        total_weight += w
        for suffix in EUR_SUFFIXES:
            if t.upper().endswith(suffix):
                eur_weight += w
                break
    if total_weight > 0 and (eur_weight / total_weight) > 0.5:
        return 'EUR'
    return 'USD'

def fetch_ticker_info(tickers):
    info_dict = {}
    for t in tickers:
        if t == 'CASH':
            info_dict[t] = {'market': 'CASH', 'name': 'Cash', 'asset_class': 'Cash', 'sector': 'Cash'}
            continue
        if t in SPECIAL_TICKERS:
            special = SPECIAL_TICKERS[t]
            try:
                yf_info = yf.Ticker(t).info or {}
                name = yf_info.get('shortName', yf_info.get('longName', t))
            except:
                name = t
            info_dict[t] = {'market': special['market'], 'name': name, 'asset_class': special['asset_class'], 'sector': special['sector']}
            continue
        try:
            yf_info = yf.Ticker(t).info or {}
            name = yf_info.get('shortName', yf_info.get('longName', t))
            yf_sector = yf_info.get('sector', '')
            sector = YFINANCE_SECTOR_MAP.get(yf_sector, '')
            qtype = yf_info.get('quoteType', 'EQUITY')
            if qtype == 'ETF':
                asset_class = 'Equity'
                category = yf_info.get('category', '')
                if t.upper() in ETF_SECTOR_FALLBACK:
                    sector = ETF_SECTOR_FALLBACK[t.upper()]
                elif not sector:
                    sector = _classify_sector_from_text(category) or _classify_sector_from_text(name) or 'Diversified'
                if category and any(k in category.lower() for k in ('bond', 'fixed', 'treasury', 'debt')):
                    asset_class = 'Fixed Income'
                    if sector in ('Diversified', 'Other', ''): sector = 'Bonds'
                elif category and any(k in category.lower() for k in ('gold', 'precious', 'commodit')):
                    asset_class = 'Commodity'
                    if sector in ('Diversified', 'Other', ''): sector = 'Commodities'
            else:
                asset_class = 'Equity'
                if not sector:
                    industry = yf_info.get('industry', '')
                    sector = _classify_sector_from_text(industry) or _classify_sector_from_text(yf_sector) or _classify_sector_from_text(name) or 'Other'
            market = detect_market(t)
            info_dict[t] = {'market': market, 'name': name, 'asset_class': asset_class, 'sector': sector}
        except:
            info_dict[t] = {'market': detect_market(t), 'name': t, 'asset_class': 'Equity', 'sector': 'Other'}
    return info_dict

def validate_tickers(tickers):
    valid, invalid = [], []
    for t in tickers:
        if t == 'CASH':
            valid.append(t)
            continue
        try:
            hist = yf.Ticker(t).history(period="5d")
            if hist is not None and len(hist) > 0:
                valid.append(t)
            else:
                invalid.append(t)
        except:
            invalid.append(t)
    return valid, invalid

def get_risk_free_rate(mode='USD'):
    if mode == 'EUR':
        return 0.028, "German Bund 10Y (estimate)", "AUTO EUR"
    try:
        tnx = yf.download("^TNX", period="5d", progress=False)
        if tnx is not None and len(tnx) > 0:
            close = tnx['Close']
            if hasattr(close, 'columns'):
                close = close.iloc[:, 0]
            val = float(close.dropna().iloc[-1]) / 100.0
            if 0.001 < val < 0.15:
                return val, "US Treasury 10Y (^TNX)", "AUTO USD"
    except:
        pass
    return 0.0425, "Fallback (4.25%)", "FALLBACK"

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
                            return dy, f"SPY calc ({annual_div:.2f}/{price:.0f})"
    except:
        pass
    try:
        info = yf.Ticker("SPY").info or {}
        for field in ['dividendYield', 'trailingAnnualDividendYield']:
            val = info.get(field)
            if val and 0.005 < float(val) < 0.05:
                return float(val), f"SPY {field}"
    except:
        pass
    return 0.013, "FALLBACK (1.3%)"

def download_prices(tickers, years=5):
    end = datetime.now()
    start = end - timedelta(days=years * 365)
    real_tickers = [t for t in tickers if t != 'CASH']
    if not real_tickers:
        return pd.DataFrame()
    for attempt in range(3):
        try:
            data = yf.download(real_tickers, start=start, end=end, progress=False)
            if data is not None and len(data) > 0:
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
    returns_raw = monthly.pct_change().iloc[1:]

    available = [t for t in weights if t in returns_raw.columns]
    if not available:
        return None

    w_available = {t: weights[t] for t in available}
    cash_weight = weights.get('CASH', 0)
    non_cash_total = sum(w_available.values())
    w_norm = {t: v / non_cash_total for t, v in w_available.items()} if non_cash_total > 0 else w_available

    ret_df = returns_raw[available].copy()
    for col in ret_df.columns:
        col_mean = ret_df[col].mean()
        ret_df[col] = ret_df[col].fillna(col_mean)
    ret_df = ret_df.dropna(how='all')

    w_arr = np.array([w_norm.get(t, 0) for t in available])
    port_returns = ret_df.values @ w_arr

    n_months = len(port_returns)
    years = n_months / 12
    cagr = (1 + port_returns).prod() ** (1 / years) - 1 if years > 0 else 0

    vol_monthly = np.std(port_returns, ddof=1)
    vol_annual = vol_monthly * np.sqrt(12)

    sharpe_hist = (np.mean(port_returns) - rf / 12) / vol_monthly * np.sqrt(12) if vol_monthly > 0 else 0

    downside = port_returns[port_returns < 0]
    down_dev = np.std(downside, ddof=1) if len(downside) > 1 else vol_monthly
    sortino = (np.mean(port_returns) - rf / 12) / down_dev * np.sqrt(12) if down_dev > 0 else 0

    cumulative = np.cumprod(1 + port_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns)

    cov_matrix = ret_df.cov() * 12
    vol_portfolio = np.sqrt(w_arr @ cov_matrix.values @ w_arr)
    vol_undiversified = sum(w_arr[i] * np.sqrt(cov_matrix.values[i, i]) for i in range(len(available)))
    div_benefit = 1 - vol_portfolio / vol_undiversified if vol_undiversified > 0 else 0

    marginal = cov_matrix.values @ w_arr
    risk_contrib = w_arr * marginal / vol_portfolio if vol_portfolio > 0 else w_arr
    corr_matrix = ret_df.corr()

    # CAPM
    dy, dy_source = get_sp500_dividend_yield()
    buyback = 0.015
    rm = dy + buyback + scenario_g
    erp = rm - rf

    benchmarks_needed = set()
    for t in available:
        info = get_ticker_info().get(t, {})
        mkt = info.get('market', 'US')
        if mkt in BENCHMARK_MAP:
            benchmarks_needed.add(BENCHMARK_MAP[mkt])

    bench_prices = {}
    if benchmarks_needed:
        try:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=5 * 365)
            for bm in benchmarks_needed:
                bdata = yf.download(bm, start=start_dt, end=end_dt, progress=False)
                if bdata is not None and len(bdata) > 0:
                    close = bdata['Close']
                    if isinstance(close, pd.DataFrame):
                        close = close.iloc[:, 0]
                    bench_prices[bm] = close
        except:
            pass

    betas, r_squareds, expected_returns, data_sources = {}, {}, {}, {}

    for t in available:
        info = get_ticker_info().get(t, {})
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
                    beta = cov_xy[0, 1] / cov_xy[0, 0] if cov_xy[0, 0] > 0 else 1.0
                    ss_res = np.sum((y - (np.mean(y) + beta * (x - np.mean(x)))) ** 2)
                    ss_tot = np.sum((y - np.mean(y)) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
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

    erp_portfolio = sum(w_available.get(t, 0) * expected_returns.get(t, rf) for t in available)
    erp_portfolio += cash_weight * rf
    total_w = sum(weights.values())
    erp_norm = erp_portfolio / total_w if total_w > 0 else erp_portfolio

    sharpe_fwd = (erp_norm - rf) / vol_portfolio if vol_portfolio > 0 else 0

    pos_months = sum(1 for r in port_returns if r > 0)
    gains = [r for r in port_returns if r > 0]
    losses = [abs(r) for r in port_returns if r < 0]
    gain_loss = np.mean(gains) / np.mean(losses) if losses else 999
    var_5 = np.percentile(port_returns, 5)

    sector_weights = {}
    for t in weights:
        info = get_ticker_info().get(t, {})
        sector = info.get('sector', 'Other')
        sector_weights[sector] = sector_weights.get(sector, 0) + weights[t]
    max_sector = max(sector_weights.values()) if sector_weights else 0
    max_sector_name = max(sector_weights, key=sector_weights.get) if sector_weights else ''
    max_position = max(weights.values()) if weights else 0
    max_position_ticker = max(weights, key=weights.get) if weights else ''
    max_rc = max(risk_contrib) if len(risk_contrib) > 0 else 0
    max_rc_ticker = available[np.argmax(risk_contrib)] if len(risk_contrib) > 0 else ''

    avg_corr = 0.0
    if len(available) > 1:
        total_pair_weight = 0
        for ii in range(len(available)):
            for jj in range(ii + 1, len(available)):
                pair_w = w_arr[ii] * w_arr[jj]
                avg_corr += pair_w * corr_matrix.iloc[ii, jj]
                total_pair_weight += pair_w
        if total_pair_weight > 0:
            avg_corr = avg_corr / total_pair_weight

    data_warnings = {}
    for t in available:
        t_data = returns_raw[t].dropna() if t in returns_raw.columns else pd.Series()
        actual_months = len(t_data)
        total_months = len(ret_df)
        if actual_months < total_months and actual_months < 24:
            data_warnings[t] = actual_months

    return {
        'returns': ret_df, 'port_returns': port_returns, 'cumulative': cumulative,
        'dates': ret_df.index, 'cagr': cagr, 'vol': vol_portfolio, 'vol_undiv': vol_undiversified,
        'div_benefit': div_benefit, 'sharpe_hist': sharpe_hist, 'sortino': sortino, 'max_dd': max_dd,
        'skewness': float(pd.Series(port_returns).skew()),
        'kurtosis': float(pd.Series(port_returns).kurtosis()),
        'var_5': var_5, 'pos_periods': pos_months, 'total_periods': n_months, 'gain_loss': gain_loss,
        'corr_matrix': corr_matrix, 'cov_matrix': cov_matrix,
        'risk_contrib': dict(zip(available, risk_contrib)),
        'betas': betas, 'r_squareds': r_squareds, 'expected_returns': expected_returns,
        'data_sources': data_sources, 'erp_portfolio': erp_norm, 'sharpe_fwd': sharpe_fwd,
        'rf': rf, 'rm': rm, 'erp': erp, 'dy': dy, 'dy_source': dy_source,
        'sector_weights': sector_weights, 'max_sector': max_sector, 'max_sector_name': max_sector_name,
        'max_position': max_position, 'max_position_ticker': max_position_ticker,
        'max_rc': max_rc, 'max_rc_ticker': max_rc_ticker,
        'available': available, 'w_norm': w_norm, 'n_months': n_months, 'avg_corr': avg_corr,
        'data_warnings': data_warnings,
    }


# ════════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════════
def bb_metric_html(label, value, delta="", delta_class="neutral"):
    return f'<div class="bb-metric"><div class="bb-metric-label">{label}</div><div class="bb-metric-value">{value}</div><div class="bb-metric-delta-{delta_class}">{delta}</div></div>'

def bb_metric_sm_html(label, value, delta="", delta_class="neutral"):
    return f'<div class="bb-metric"><div class="bb-metric-label">{label}</div><div class="bb-metric-value-sm">{value}</div><div class="bb-metric-delta-{delta_class}">{delta}</div></div>'

def bb_section(title):
    return f'<div class="bb-section">{title}</div>'

def plotly_bloomberg(fig):
    fig.update_layout(
        template="plotly_dark", paper_bgcolor='#0a0a0a', plot_bgcolor='#141414',
        font=dict(family="JetBrains Mono, Consolas, monospace", size=11, color="#d4d4d4"),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a'),
        yaxis=dict(gridcolor='#2a2a2a', zerolinecolor='#2a2a2a'),
    )
    return fig

def _on_scenario_change():
    """Callback: load KPI targets from selected scenario."""
    sc = st.session_state.get('scenario_selector', 'Base')
    targets = KPI_TARGETS_BY_SCENARIO.get(sc, KPI_TARGETS_BY_SCENARIO['Base'])
    for kpi_key, val in targets.items():
        st.session_state[f'kpi_t_{kpi_key}'] = val


# ════════════════════════════════════════════════════════════════
# BANNER
# ════════════════════════════════════════════════════════════════
import base64, os

banner_html = """
<div class="bb-banner">
    <div><div class="bb-banner-title">◆ PORTFOLIO <span>ANALYZER</span></div>
    <div class="bb-banner-sub">Quantitative Portfolio Management</div></div>
    <div class="bb-banner-version">v5.0</div>
    <div class="bb-banner-accent"></div>
</div>"""

banner_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'banner.png')
if os.path.exists(banner_path):
    with open(banner_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    banner_html = f"""
    <div style="position:relative; width:100%; height:100px; margin-bottom:8px; overflow:hidden; border-bottom: 2px solid #FF8C00;">
        <img src="data:image/png;base64,{b64}" style="width:100%; height:100%; object-fit:cover; opacity:0.6;">
        <div style="position:absolute; top:50%; left:24px; transform:translateY(-50%);">
            <div class="bb-banner-title">◆ PORTFOLIO <span>ANALYZER</span></div>
            <div class="bb-banner-sub">Quantitative Portfolio Management</div></div>
        <div style="position:absolute; top:50%; right:24px; transform:translateY(-50%);">
            <div class="bb-banner-version">v5.0</div></div>
    </div>"""

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
if 'currency_mode' not in st.session_state:
    st.session_state.currency_mode = 'USD'
if 'rf_source' not in st.session_state:
    st.session_state.rf_source = ''

# Initialize KPI targets from Base scenario
for _kpi_key in KPI_DEFINITIONS:
    _ss_key = f'kpi_t_{_kpi_key}'
    if _ss_key not in st.session_state:
        st.session_state[_ss_key] = KPI_TARGETS_BY_SCENARIO['Base'][_kpi_key]


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

        pv = st.number_input("Portfolio Value (EUR)", value=st.session_state.portfolio_value,
                             step=1000, min_value=1000, key="pv_input")
        st.session_state.portfolio_value = pv

        st.markdown(bb_section("ENTER HOLDINGS"), unsafe_allow_html=True)
        st.markdown("""<div style="color:#808080; font-size:0.75rem; margin-bottom:8px;">
            Enter one asset per line: <span style="color:#FF8C00;">TICKER,WEIGHT</span><br>
            Or semicolon-separated: <span style="color:#FF8C00;">AAPL,30;ITX.MC,20;GLD,10</span><br>
            For cash, use: <span style="color:#FF8C00;">CASH,5</span>
        </div>""", unsafe_allow_html=True)

        default_text = ""
        if st.session_state.portfolio:
            lines = [f"{t},{w*100:.1f}" for t, w in st.session_state.portfolio.items()]
            default_text = "\n".join(lines)

        portfolio_text = st.text_area(
            "Holdings (TICKER,WEIGHT)", value=default_text, height=200,
            key="portfolio_text", label_visibility="collapsed",
            placeholder="AAPL,30\nITX.MC,20\nSAN.MC,15\nGLD,10\nEEM,25"
        )

        parsed = {}
        parse_errors = []
        if portfolio_text.strip():
            text = portfolio_text.strip().replace(';', '\n')
            for line in text.split('\n'):
                line = line.strip()
                if not line: continue
                parts = line.split(',')
                if len(parts) != 2:
                    parse_errors.append(f"Invalid format: '{line}' (use TICKER,WEIGHT)")
                    continue
                ticker = parts[0].strip().upper()
                try:
                    weight = float(parts[1].strip().replace('%', ''))
                    if weight <= 0 or weight > 100:
                        parse_errors.append(f"Invalid weight for {ticker}: {weight}%")
                        continue
                    parsed[ticker] = weight / 100.0
                except ValueError:
                    parse_errors.append(f"Invalid weight for {ticker}: '{parts[1].strip()}'")

        if parse_errors:
            for err in parse_errors:
                st.markdown(f'<div class="bb-alert-warning">{err}</div>', unsafe_allow_html=True)

        st.session_state.portfolio = parsed

        if parsed:
            st.markdown(bb_section("PARSED HOLDINGS"), unsafe_allow_html=True)
            ph_data = [{'Ticker': t, 'Weight': f"{w*100:.1f}%", 'Amount': f"EUR {pv*w:,.0f}"} for t, w in parsed.items()]
            st.dataframe(pd.DataFrame(ph_data), use_container_width=True, hide_index=True)

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
            <div class="bb-progress-bg"><div class="bb-progress-fill" style="width:{fill_w}%; background:{pct_color};"></div></div>
        </div>""", unsafe_allow_html=True)

        if abs(total_pct - 100) < 0.5:
            st.success("Portfolio weights sum to 100% ✓")
        elif total_pct < 100:
            st.warning(f"Unallocated: {100-total_pct:.1f}%")
        else:
            st.error(f"Over-allocated by {total_pct-100:.1f}%")

        n_assets = len(parsed)
        st.markdown(f"""<div class="bb-card"><div class="bb-card-title">Holdings</div>
            <div class="bb-metric-value-sm">{n_assets} assets</div>
            <div style="color:#808080; font-size:0.75rem;">Value: EUR {pv:,.0f}</div></div>""", unsafe_allow_html=True)

        if parsed:
            max_pos = max(parsed, key=parsed.get)
            st.markdown(f"""<div class="bb-card"><div class="bb-card-title">Largest Position</div>
                <div class="bb-metric-value-sm">{parsed[max_pos]*100:.1f}%</div>
                <div style="color:#808080; font-size:0.75rem;">{max_pos}</div></div>""", unsafe_allow_html=True)

            cmode = detect_currency_mode(parsed)
            badge_color = "#4488FF" if cmode == 'EUR' else "#FF8C00"
            st.markdown(f"""<div class="bb-card"><div class="bb-card-title">Detected Mode</div>
                <div class="bb-metric-value-sm" style="color:{badge_color};">{cmode}</div>
                <div style="color:#808080; font-size:0.75rem;">{"Rf: German Bund 10Y est." if cmode == "EUR" else "Rf: US Treasury 10Y"}</div></div>""", unsafe_allow_html=True)

    # ── Analysis Settings ──────────────────────────────────
    st.markdown(bb_section("ANALYSIS SETTINGS"), unsafe_allow_html=True)
    set_c1, set_c2 = st.columns([1, 1])
    with set_c1:
        scenario = st.selectbox("Growth Scenario (g)", list(GORDON_SCENARIOS.keys()), index=2,
                                key='scenario_selector', on_change=_on_scenario_change)
    with set_c2:
        lookback = st.slider("Historical Data (years)", 1, 10, 5)

    # ── KPI Targets (always visible, not expander) ─────────
    st.markdown(bb_section("KPI TARGETS"), unsafe_allow_html=True)
    st.markdown(f'<div style="color:#808080; font-size:0.72rem; margin-bottom:12px;">Targets auto-loaded from <span style="color:#FF8C00;">{scenario}</span> scenario. Edit any value to customize.</div>', unsafe_allow_html=True)

    kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
    kpi_cols = [kpi_c1, kpi_c2, kpi_c3, kpi_c4]
    kpi_keys_list = list(KPI_DEFINITIONS.keys())
    for idx, kpi_key in enumerate(kpi_keys_list):
        defn = KPI_DEFINITIONS[kpi_key]
        with kpi_cols[idx % 4]:
            unit_label = f" ({defn['unit']})" if defn['unit'] else ""
            st.number_input(
                f"{defn['name']}{unit_label}",
                value=st.session_state.get(f'kpi_t_{kpi_key}', KPI_TARGETS_BY_SCENARIO['Base'][kpi_key]),
                step=defn['step'],
                format=defn['fmt'],
                min_value=defn['min'],
                max_value=defn['max'],
                key=f'kpi_t_{kpi_key}',
            )

    # ── Analyze Button ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("▶ ANALYZE", type="primary", use_container_width=True):
        portfolio_tickers = list(st.session_state.portfolio.keys())
        real_tickers = [t for t in portfolio_tickers if t != 'CASH']

        if not real_tickers:
            st.error("Add at least one asset to the portfolio before analyzing.")
        else:
            with st.spinner("Validating tickers..."):
                valid, invalid = validate_tickers(real_tickers)

            if invalid:
                st.error(f"Tickers not found: **{', '.join(invalid)}**")
                st.markdown('<div class="bb-alert-warning">Fix or remove these tickers before analyzing.</div>', unsafe_allow_html=True)
            else:
                # FIX: Set ticker_info BEFORE calculate_portfolio_metrics
                with st.spinner("Fetching asset information..."):
                    fetched_info = fetch_ticker_info(portfolio_tickers)
                    st.session_state.ticker_info = fetched_info  # ← MUST be before metrics calc

                cmode = detect_currency_mode(st.session_state.portfolio)
                st.session_state.currency_mode = cmode

                with st.spinner("Downloading market data..."):
                    prices = download_prices(portfolio_tickers, lookback)
                    rf, rf_desc, rf_badge = get_risk_free_rate(cmode)
                    st.session_state.rf_source = rf_badge
                    g = GORDON_SCENARIOS[scenario]
                    metrics = calculate_portfolio_metrics(prices, st.session_state.portfolio, rf, g)

                    if metrics:
                        st.session_state.metrics = metrics
                        st.session_state.prices = prices
                        st.session_state.analyzed = True
                        st.session_state.scenario = scenario

                        if metrics.get('data_warnings'):
                            for t, months in metrics['data_warnings'].items():
                                st.warning(f"{t} only has {months} months of data.")

                        st.success("Analysis complete ✓")
                    else:
                        st.error("Error: Could not calculate metrics. Check tickers.")


# ═══════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ═══════════════════════════════════════════════════════════
with tabs[1]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first -- go to PORTFOLIO tab and click ANALYZE</div>', unsafe_allow_html=True)
    else:
        m = st.session_state.metrics
        active_targets = get_active_targets()

        # ── Health Score ─────────────────────────────────────
        health = calculate_health_score(m, active_targets)
        total_score = health['total']
        dims = health['dimensions']

        if total_score >= 90: hs_color, hs_label = "#00C853", "EXCELLENT"
        elif total_score >= 75: hs_color, hs_label = "#00C853", "STRONG"
        elif total_score >= 60: hs_color, hs_label = "#FFD600", "GOOD"
        elif total_score >= 40: hs_color, hs_label = "#FF8C00", "FAIR"
        else: hs_color, hs_label = "#FF1744", "WEAK"

        st.markdown(f"""
        <div class="bb-card" style="background: linear-gradient(135deg, #1a1a1a 0%, #0a0a0a 100%); border: 2px solid {hs_color}; padding: 24px; text-align: center; margin-bottom: 16px;">
            <div style="font-size: 0.8rem; color: #808080; letter-spacing: 0.15em; margin-bottom: 8px;">PORTFOLIO HEALTH</div>
            <div style="font-size: 3.5rem; font-weight: 800; color: {hs_color}; line-height: 1;">{total_score:.0f}</div>
            <div style="font-size: 0.9rem; color: {hs_color}; letter-spacing: 0.1em; margin-top: 4px;">{hs_label}</div>
        </div>
        """, unsafe_allow_html=True)

        dim_display = [
            ('return', 'Return Potential', '20%'),
            ('risk', 'Risk Management', '25%'),
            ('diversification', 'Diversification', '20%'),
            ('concentration', 'Concentration Risk', '20%'),
            ('track_record', 'Track Record', '15%'),
        ]
        for key, name, weight in dim_display:
            score = dims.get(key, 0)
            if score >= 75: bar_c = "#00C853"
            elif score >= 60: bar_c = "#FFD600"
            elif score >= 40: bar_c = "#FF8C00"
            else: bar_c = "#FF1744"
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="flex: 0 0 200px; color: #d4d4d4; font-size: 0.78rem;">{name} <span style="color:#555;">({weight})</span></div>
                <div style="flex: 1; background: #2a2a2a; height: 18px; border-radius: 3px; overflow: hidden; margin: 0 12px;">
                    <div style="width: {score}%; height: 100%; background: {bar_c};"></div>
                </div>
                <div style="flex: 0 0 60px; text-align: right; color: {bar_c}; font-weight: 700; font-size: 0.82rem;">{score:.0f}/100</div>
            </div>""", unsafe_allow_html=True)

        # ── Currency badge ───────────────────────────────────
        rf_badge = st.session_state.get('rf_source', '')
        badge_color = "#4488FF" if 'EUR' in rf_badge else "#FF8C00"
        st.markdown(f'<div style="text-align:right; margin-bottom:-10px;"><span class="bb-badge" style="background:#1a1a1a; color:{badge_color}; border:1px solid {badge_color};">{rf_badge} | Rf={m["rf"]*100:.2f}%</span></div>', unsafe_allow_html=True)

        # ── Portfolio Overview ───────────────────────────────
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
            wr = m['pos_periods'] / m['total_periods'] * 100 if m['total_periods'] > 0 else 0
            st.markdown(bb_metric_sm_html("Win Rate", f"{wr:.0f}%"), unsafe_allow_html=True)

        # ── KPI Targets (with semaphore) ─────────────────────
        st.markdown(bb_section("KPI TARGETS"), unsafe_allow_html=True)

        kpi_values = {
            'expected_ret': m['erp_portfolio'] * 100, 'sharpe_fwd': m['sharpe_fwd'],
            'volatility': m['vol'] * 100, 'max_dd': m['max_dd'] * 100,
            'sector_conc': m['max_sector'] * 100, 'max_position': m['max_position'] * 100,
            'avg_corr': m.get('avg_corr', 0),
        }

        greens = yellows = reds = 0
        kpi_html = ""
        for key, defn in KPI_DEFINITIONS.items():
            val = kpi_values.get(key, 0)
            target_val = active_targets[key]
            status = evaluate_kpi(val, target_val, key)
            if status == 'green': greens += 1
            elif status == 'yellow': yellows += 1
            else: reds += 1
            if defn['unit'] == '%':
                val_str = f"{val:.1f}%"
                tgt_str = f"{target_val:.1f}%"
            else:
                val_str = f"{val:.2f}"
                tgt_str = f"{target_val:.2f}"
            direction = ">" if defn['target'] == 'above' else "<"
            kpi_html += f'<div class="bb-kpi-row"><div class="bb-kpi-dot bb-kpi-dot-{status}"></div><div class="bb-kpi-name">{defn["name"]}</div><div style="color:#555; font-size:0.72rem; min-width:80px; text-align:right; margin-right:12px;">target {direction} {tgt_str}</div><div class="bb-kpi-value">{val_str}</div></div>'

        st.markdown(kpi_html, unsafe_allow_html=True)
        total_kpis = greens + yellows + reds
        st.markdown(f'<div style="padding:8px 12px; color:#808080; font-size:0.75rem;">Score: {greens} <span style="color:#00C853;">●</span> {yellows} <span style="color:#FFD600;">●</span> {reds} <span style="color:#FF1744;">●</span> ({greens}/{total_kpis} passing)</div>', unsafe_allow_html=True)

        # ── Additional Metrics (informational, gray dot) ─────
        st.markdown(bb_section("ADDITIONAL METRICS"), unsafe_allow_html=True)

        info_vals = {
            'div_benefit': m['div_benefit'] * 100, 'sharpe_hist': m['sharpe_hist'],
            'cagr': m['cagr'] * 100, 'sortino': m['sortino'], 'var_5': m['var_5'] * 100,
            'pos_periods': m['pos_periods'] / m['total_periods'] * 100 if m['total_periods'] > 0 else 0,
            'gain_loss': m['gain_loss'], 'max_risk_contr': m['max_rc'] * 100,
        }

        info_html = ""
        for key, defn in INFO_KPIS.items():
            val = info_vals.get(key, 0)
            if defn['unit'] == '%':
                val_str = f"{val:.1f}%"
            else:
                val_str = f"{val:.2f}"
            info_html += f'<div class="bb-kpi-row"><div class="bb-kpi-dot bb-kpi-dot-gray"></div><div class="bb-kpi-name">{defn["name"]}</div><div class="bb-kpi-value">{val_str}</div></div>'

        st.markdown(info_html, unsafe_allow_html=True)

        # ── PDF Download ─────────────────────────────────────
        pdf_buf = generate_pdf_report(m, st.session_state.portfolio, st.session_state.portfolio_value,
            st.session_state.get('scenario', 'Base'), get_ticker_info(), active_targets, evaluate_kpi, health)
        if pdf_buf:
            st.download_button(label="DOWNLOAD REPORT (PDF)", data=pdf_buf,
                file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf", type="primary", use_container_width=True)

        # ── Portfolio Growth ─────────────────────────────────
        st.markdown(bb_section("PORTFOLIO GROWTH"), unsafe_allow_html=True)
        cum_series = pd.Series(m['cumulative'], index=m['dates'][:len(m['cumulative'])])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cum_series.index, y=cum_series.values, mode='lines',
            line=dict(color='#FF8C00', width=2), fill='tozeroy', fillcolor='rgba(255,140,0,0.1)'))
        fig.update_layout(yaxis_title="Growth of EUR 1", showlegend=False, height=350)
        st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 2: RETURNS
# ═══════════════════════════════════════════════════════════
with tabs[2]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first -- go to PORTFOLIO tab and click ANALYZE</div>', unsafe_allow_html=True)
    else:
        m = st.session_state.metrics
        rf_badge = st.session_state.get('rf_source', '')

        st.markdown(bb_section("MARKET RETURN - GORDON GROWTH MODEL"), unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(bb_metric_html("Risk-Free Rate", f"{m['rf']*100:.2f}%", rf_badge), unsafe_allow_html=True)
        with c2:
            st.markdown(bb_metric_html("Dividend Yield", f"{m['dy']*100:.2f}%", m['dy_source']), unsafe_allow_html=True)
        with c3:
            st.markdown(bb_metric_html("Total Yield", f"{(m['dy']+0.015)*100:.2f}%", "Div + Buyback"), unsafe_allow_html=True)
        with c4:
            st.markdown(bb_metric_html("Rm (Base)", f"{m['rm']*100:.2f}%", f"g = {GORDON_SCENARIOS.get(st.session_state.get('scenario','Base'),0.06)*100:.1f}%"), unsafe_allow_html=True)

        st.markdown(bb_section("RETURN SCENARIOS"), unsafe_allow_html=True)
        sc_data = []
        for name, g in GORDON_SCENARIOS.items():
            rm_sc = m['dy'] + 0.015 + g
            erp_sc = rm_sc - m['rf']
            marker = " ◄" if name == st.session_state.get('scenario', 'Base') else ""
            sc_data.append({'Scenario': name + marker, 'g': f"{g*100:.1f}%", 'Rm': f"{rm_sc*100:.2f}%", 'ERP': f"{erp_sc*100:.2f}%"})
        st.dataframe(pd.DataFrame(sc_data), use_container_width=True, hide_index=True)

        st.markdown(bb_section("CAPM - BETAS & EXPECTED RETURNS"), unsafe_allow_html=True)
        beta_data = []
        for t in sorted(m['available']):
            info = get_ticker_info().get(t, {})
            mkt = info.get('market', 'US')
            bm = BENCHMARK_MAP.get(mkt, 'N/A')
            beta = m['betas'].get(t)
            r2 = m['r_squareds'].get(t)
            er = m['expected_returns'].get(t, 0)
            src = m['data_sources'].get(t, '')
            beta_data.append({
                'Ticker': t, 'Name': str(info.get('name', t)),
                'Weight': f"{st.session_state.portfolio.get(t,0)*100:.1f}%",
                'Beta': f"{beta:.2f}" if beta is not None else '-',
                'R2': f"{r2:.2f}" if r2 is not None else '-',
                'E(Ri)': f"{er*100:.2f}%",
                'Benchmark': bm if beta is not None else 'N/A',
                'Source': str(src)
            })
        st.dataframe(pd.DataFrame(beta_data), use_container_width=True, hide_index=True)

        st.markdown(bb_section("PORTFOLIO EXPECTED RETURN"), unsafe_allow_html=True)
        st.markdown(f'<div class="bb-metric" style="border-left-color:#FF8C00;"><div class="bb-metric-label">E(Rp)</div><div class="bb-metric-value">{m["erp_portfolio"]*100:.2f}%</div><div class="bb-metric-delta-pos">Scenario: {st.session_state.get("scenario","Base")}</div></div>', unsafe_allow_html=True)

        st.markdown(bb_section("DATA SOURCE TRANSPARENCY"), unsafe_allow_html=True)
        for t in sorted(m['data_sources'].keys()):
            src = m['data_sources'][t]
            info = get_ticker_info().get(t, {})
            bc = 'bb-badge-calc' if src == 'CALCULATED' else ('bb-badge-est' if src == 'ESTIMATED' else 'bb-badge-fallback')
            st.markdown(f'<div style="padding:4px 0; display:flex; align-items:center; gap:8px;"><span style="color:#d4d4d4; min-width:70px; font-size:0.78rem;">{t}</span><span class="bb-badge {bc}">{src}</span><span style="color:#808080; font-size:0.75rem;">{info.get("name", "")}</span></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# TAB 3: RISK
# ═══════════════════════════════════════════════════════════
with tabs[3]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first -- go to PORTFOLIO tab and click ANALYZE</div>', unsafe_allow_html=True)
    else:
        m = st.session_state.metrics

        st.markdown(bb_section("CORRELATION MATRIX"), unsafe_allow_html=True)
        corr = m['corr_matrix']
        names = [get_ticker_info().get(t, {}).get('name', t)[:12] for t in corr.columns]
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=names, y=names,
            colorscale=[[0, '#FF1744'], [0.5, '#0a0a0a'], [1, '#00C853']], zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=9, color="#d4d4d4")))
        fig.update_layout(height=500)
        st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)
        n = m['n_months']
        st.caption(f"Observations: {n} months | Significance threshold (95%): |r| > {2/np.sqrt(n):.2f}")

        st.markdown(bb_section("RISK CONTRIBUTION BY ASSET"), unsafe_allow_html=True)
        rc = m['risk_contrib']
        rc_sorted = sorted(rc.items(), key=lambda x: x[1], reverse=True)
        tickers_rc = [t for t, _ in rc_sorted]
        rc_vals = [v * 100 for _, v in rc_sorted]
        w_vals = [st.session_state.portfolio.get(t, 0) * 100 for t in tickers_rc]
        names_rc = [get_ticker_info().get(t, {}).get('name', t) for t in tickers_rc]
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Risk Contribution', x=names_rc, y=rc_vals, marker_color='#FF8C00',
            text=[f"{v:.1f}%" for v in rc_vals], textposition='outside', textfont=dict(size=9)))
        fig.add_trace(go.Bar(name='Weight', x=names_rc, y=w_vals, marker_color='#555',
            text=[f"{v:.1f}%" for v in w_vals], textposition='outside', textfont=dict(size=9)))
        fig.update_layout(barmode='group', height=400, yaxis_title="%", showlegend=True)
        st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)

        st.markdown(bb_section("INDIVIDUAL ASSET VOLATILITY"), unsafe_allow_html=True)
        vol_data = []
        cov = m['cov_matrix']
        for t in sorted(m['available']):
            info = get_ticker_info().get(t, {})
            vol_ann = np.sqrt(cov.loc[t, t]) * 100 if t in cov.columns else 0
            vol_mon = vol_ann / np.sqrt(12)
            w = st.session_state.portfolio.get(t, 0) * 100
            bar = chr(9608) * min(int(vol_ann / 2), 30)
            vol_data.append({'Ticker': t, 'Name': str(info.get('name', t)), 'Weight': f"{w:.1f}%",
                'Vol Ann.': f"{vol_ann:.1f}%", 'Vol Mon.': f"{vol_mon:.1f}%", 'Risk': bar})
        vol_data.sort(key=lambda x: float(x['Vol Ann.'].replace('%', '')), reverse=True)
        st.dataframe(pd.DataFrame(vol_data), use_container_width=True, hide_index=True)

        st.markdown(bb_section("PORTFOLIO VOLATILITY"), unsafe_allow_html=True)
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.markdown(bb_metric_html("Sigma(p) Diversified", f"{m['vol']*100:.2f}%"), unsafe_allow_html=True)
        with vc2:
            st.markdown(bb_metric_html("Sigma(p) Undiversified", f"{m['vol_undiv']*100:.2f}%"), unsafe_allow_html=True)
        with vc3:
            st.markdown(bb_metric_html("Diversification Benefit", f"{m['div_benefit']*100:.1f}%"), unsafe_allow_html=True)

        st.markdown(bb_section("DRAWDOWNS"), unsafe_allow_html=True)
        cum = pd.Series(m['cumulative'], index=m['dates'][:len(m['cumulative'])])
        running_max = cum.cummax()
        dd_series = (cum - running_max) / running_max * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values, fill='tozeroy',
            fillcolor='rgba(255,23,68,0.3)', line=dict(color='#FF1744', width=1)))
        fig.update_layout(yaxis_title="Drawdown %", height=300, showlegend=False)
        st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)


# ═══════════════════════════════════════════════════════════
# TAB 4: EXPOSURE
# ═══════════════════════════════════════════════════════════
with tabs[4]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first -- go to PORTFOLIO tab and click ANALYZE</div>', unsafe_allow_html=True)
    else:
        m = st.session_state.metrics

        st.markdown(bb_section("ASSET CLASS ALLOCATION"), unsafe_allow_html=True)
        ac1, ac2 = st.columns(2)
        with ac1:
            ac_weights = {}
            for t, w in st.session_state.portfolio.items():
                info = get_ticker_info().get(t, {})
                ac = info.get('asset_class', 'Other')
                ac_weights[ac] = ac_weights.get(ac, 0) + w
            fig = go.Figure(data=[go.Pie(labels=list(ac_weights.keys()), values=[v * 100 for v in ac_weights.values()],
                hole=0.5, textinfo='label+percent', marker=dict(colors=['#FF8C00', '#FFD600', '#00C853', '#4488FF', '#CC66FF']))])
            fig.update_layout(title="Asset Class", height=350, showlegend=False)
            st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)
        with ac2:
            sector_w = m['sector_weights']
            fig = go.Figure(data=[go.Pie(labels=list(sector_w.keys()), values=[v * 100 for v in sector_w.values()],
                hole=0.5, textinfo='label+percent')])
            fig.update_layout(title="Sector", height=350, showlegend=False)
            st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)

        st.markdown(bb_section("GEOGRAPHIC EXPOSURE"), unsafe_allow_html=True)
        geo_weights = {}
        for t, w in st.session_state.portfolio.items():
            info = get_ticker_info().get(t, {})
            country = MARKET_NAMES.get(info.get('market', 'US'), info.get('market', 'US'))
            geo_weights[country] = geo_weights.get(country, 0) + w
        fig = go.Figure(data=[go.Pie(labels=list(geo_weights.keys()), values=[v * 100 for v in geo_weights.values()],
            hole=0.5, textinfo='label+percent', marker=dict(colors=['#FF8C00', '#4488FF', '#00C853', '#FFD600', '#CC66FF', '#FF1744']))])
        fig.update_layout(title="Geography", height=350, showlegend=False)
        st.plotly_chart(plotly_bloomberg(fig), use_container_width=True)

        st.markdown(bb_section("HOLDINGS DETAIL"), unsafe_allow_html=True)
        holdings = []
        for t, w in st.session_state.portfolio.items():
            info = get_ticker_info().get(t, {})
            holdings.append({
                'Ticker': t, 'Name': str(info.get('name', t)),
                'Weight': f"{w*100:.1f}%", 'Class': str(info.get('asset_class', 'Other')),
                'Sector': str(info.get('sector', 'Other')),
                'Market': str(MARKET_NAMES.get(info.get('market', ''), info.get('market', ''))),
                'Amount': f"EUR {st.session_state.portfolio_value * w:,.0f}"
            })
        st.dataframe(pd.DataFrame(holdings), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 5: SIMULATOR
# ═══════════════════════════════════════════════════════════
with tabs[5]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first -- go to PORTFOLIO tab and click ANALYZE</div>', unsafe_allow_html=True)
    else:
        m = st.session_state.metrics

        st.markdown(bb_section("WEIGHT SIMULATOR"), unsafe_allow_html=True)
        st.markdown('<div style="color:#808080; font-size:0.78rem; margin-bottom:16px;">Adjust weights below to simulate alternative allocations.</div>', unsafe_allow_html=True)

        sim_weights = {}
        cols = st.columns(4)
        for i, (t, w) in enumerate(st.session_state.portfolio.items()):
            with cols[i % 4]:
                init_val = round(w * 100, 1)
                new_w = st.slider(t, 0.0, 30.0, init_val, 0.1, key=f"sim_{t}")
                sim_weights[t] = new_w / 100

        sim_total = sum(sim_weights.values()) * 100
        sim_color = "#00C853" if abs(sim_total - 100) < 1 else ("#FFD600" if sim_total < 100 else "#FF1744")
        sim_fill = min(sim_total, 100)
        st.markdown(f"""<div class="bb-card" style="margin:12px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span class="bb-card-title">Simulated Total Weight</span>
                <span style="color:{sim_color}; font-weight:700; font-size:0.9rem;">{sim_total:.1f}%</span>
            </div>
            <div class="bb-progress-bg"><div class="bb-progress-fill" style="width:{sim_fill}%; background:{sim_color};"></div></div>
        </div>""", unsafe_allow_html=True)

        weights_changed = any(abs(sim_weights[t] - st.session_state.portfolio.get(t, 0)) > 0.001 for t in sim_weights)

        st.markdown(bb_section("COMPARISON: CURRENT VS SIMULATED"), unsafe_allow_html=True)

        if weights_changed and st.session_state.prices is not None:
            rf = m['rf']
            g = GORDON_SCENARIOS.get(st.session_state.get('scenario', 'Base'), 0.06)
            sim_metrics = calculate_portfolio_metrics(st.session_state.prices, sim_weights, rf, g)
            sm = sim_metrics if sim_metrics else m
        else:
            sm = m

        comp_data = [
            {'Metric': 'E(Rp) Forward', 'Current': f"{m['erp_portfolio']*100:.2f}%", 'Simulated': f"{sm['erp_portfolio']*100:.2f}%"},
            {'Metric': 'CAGR Historical', 'Current': f"{m['cagr']*100:.1f}%", 'Simulated': f"{sm['cagr']*100:.1f}%"},
            {'Metric': 'Volatility', 'Current': f"{m['vol']*100:.2f}%", 'Simulated': f"{sm['vol']*100:.2f}%"},
            {'Metric': 'Sharpe (Forward)', 'Current': f"{m['sharpe_fwd']:.2f}", 'Simulated': f"{sm['sharpe_fwd']:.2f}"},
            {'Metric': 'Sharpe (Historical)', 'Current': f"{m['sharpe_hist']:.2f}", 'Simulated': f"{sm['sharpe_hist']:.2f}"},
            {'Metric': 'Max Drawdown', 'Current': f"{m['max_dd']*100:.1f}%", 'Simulated': f"{sm['max_dd']*100:.1f}%"},
            {'Metric': 'Diversification', 'Current': f"{m['div_benefit']*100:.1f}%", 'Simulated': f"{sm['div_benefit']*100:.1f}%"},
        ]
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# TAB 6: METHODOLOGY
# ═══════════════════════════════════════════════════════════
with tabs[6]:
    if not st.session_state.analyzed or st.session_state.metrics is None:
        st.markdown('<div class="bb-alert-info">Run analysis first to see methodology details.</div>', unsafe_allow_html=True)
    else:
        m = st.session_state.metrics
        st.markdown(bb_section("METHODOLOGY & SOURCES"), unsafe_allow_html=True)

        method_sections = []

        rf_badge = st.session_state.get('rf_source', '')
        method_sections.append(("Market Return (Rm) - Gordon Growth Model",
            f"Formula: Rm = Shareholder Yield + g<br><br>"
            f"Dividend Yield: {m['dy']*100:.2f}% ({m['dy_source']})<br>"
            f"Buyback Yield: 1.50% (S&P 500 average estimate)<br>"
            f"g: {GORDON_SCENARIOS.get(st.session_state.get('scenario','Base'),0.06)*100:.1f}% (Scenario: {st.session_state.get('scenario','Base')})<br>"
            f"Rm = {m['rm']*100:.2f}%"))

        method_sections.append(("Risk-Free Rate (Rf)",
            f"Mode: {rf_badge}<br>"
            f"Current value: {m['rf']*100:.2f}%<br>"
            f"{'Auto EUR: majority European assets. German Bund 10Y estimate.' if 'EUR' in rf_badge else 'Auto USD: US Treasury 10Y from Yahoo Finance.'}"))

        benchmarks_used = set()
        low_r2_assets = []
        for t in m['available']:
            info = get_ticker_info().get(t, {})
            mkt = info.get('market', 'US')
            if mkt in BENCHMARK_MAP:
                benchmarks_used.add(BENCHMARK_MAP[mkt])
            r2 = m['r_squareds'].get(t)
            if r2 is not None and r2 < 0.10:
                low_r2_assets.append(f"{t} (R2={r2:.2f})")

        beta_text = (f"Method: Cov(Ri, Rm) / Var(Rm)<br>"
            f"Period: {m['n_months']} months, monthly frequency<br>"
            f"Benchmarks: {', '.join(sorted(benchmarks_used))}")
        if low_r2_assets:
            beta_text += f"<br>Low R2: {', '.join(low_r2_assets)}"
        method_sections.append(("Betas", beta_text))

        non_capm = []
        for t in m['available']:
            info = get_ticker_info().get(t, {})
            if info.get('asset_class', 'Equity') in ('Commodity', 'Crypto', 'Cash'):
                non_capm.append((t, info.get('name', t), info.get('asset_class', '')))
        if 'CASH' in st.session_state.portfolio and not any(x[0] == 'CASH' for x in non_capm):
            non_capm.append(('CASH', 'Cash Position', 'Cash'))

        if non_capm:
            nc_text = "Non-CAPM assets:<br><br>"
            for t, name, ac in non_capm:
                er = m['expected_returns'].get(t, 0)
                nc_text += f"{t} ({name}): E(R) = {er*100:.2f}% [{ac}]<br>"
            method_sections.append(("Non-CAPM Assets", nc_text))

        n = m['n_months']
        method_sections.append(("Correlations",
            f"Period: {n} months ({n/12:.1f} years)<br>"
            f"Significance (95%): |r| > {2/np.sqrt(n):.2f}<br>"
            f"Wtd Avg Correlation: {m.get('avg_corr',0):.2f}"))

        method_sections.append(("Portfolio Volatility",
            f"sigma_p = sqrt(w' Sigma w)<br>"
            f"Diversified: {m['vol']*100:.2f}% | Undiversified: {m['vol_undiv']*100:.2f}%<br>"
            f"Diversification benefit: {m['div_benefit']*100:.1f}%"))

        method_sections.append(("Health Score",
            "Weighted composite of 5 dimensions:<br>"
            "- Return Potential (20%): E(Rp), Sharpe Fwd, CAGR<br>"
            "- Risk Management (25%): Vol, Max DD, Sharpe Hist, Sortino, VaR<br>"
            "- Diversification (20%): Div Benefit, Wtd Avg Corr<br>"
            "- Concentration (20%): Sector Conc, Max Position, Max RC<br>"
            "- Track Record (15%): Win Rate, Gain/Loss Ratio<br><br>"
            "Threshold KPIs: green=100, yellow=60, red=30.<br>"
            "Informational KPIs: normalized 0-100 using predefined ranges."))

        method_sections.append(("Disclaimer",
            "These are model-based estimates, not predictions.<br>"
            "CAPM has well-known limitations.<br>"
            "Past performance is not indicative of future results."))

        for title, content in method_sections:
            st.markdown(f'<div class="bb-card" style="margin-bottom:6px;"><div class="bb-card-title">{title}</div><div style="font-size:0.78rem; color:#d4d4d4; line-height:1.6;">{content}</div></div>', unsafe_allow_html=True)

        benchmarks_list = ", ".join(sorted(benchmarks_used)) if benchmarks_used else "N/A"
        st.markdown(bb_section("DATA SOURCES"), unsafe_allow_html=True)
        st.markdown(f"""<div style="font-size:0.78rem; color:#808080; line-height:1.8;">
            <span style="color:#FF8C00; font-weight:600;">PRICES</span> Yahoo Finance (yfinance)<br>
            <span style="color:#FF8C00; font-weight:600;">RISK-FREE</span> {rf_badge} ({m['rf']*100:.2f}%)<br>
            <span style="color:#FF8C00; font-weight:600;">DIVIDENDS</span> SPY trailing 12-month ({m['dy']*100:.2f}%)<br>
            <span style="color:#FF8C00; font-weight:600;">BENCHMARKS</span> {benchmarks_list}<br>
            <span style="color:#FF8C00; font-weight:600;">OBSERVATIONS</span> {m['n_months']} months
        </div>""", unsafe_allow_html=True)
