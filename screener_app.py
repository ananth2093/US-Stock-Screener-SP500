# screener_app.py v16
# ─────────────────────────────────────────────────────────────────────────────
# v16 CHANGES from v15:
#
#  GAP 2 — Quality Score: 3 signals → 7 signals
#    • _fetch_yahoo_deep_one() now fetches 14 new fields needed for
#      Piotroski and Sloan: ocf_ttm, fcf_ttm, net_income_ttm,
#      gross_profit_ttm, gross_margin_now/prev, roa_ttm, roa_prev,
#      total_assets_now/prev, lt_debt_ratio_now/prev,
#      current_ratio_now/prev, shares_now/prev.
#    • compute_piotroski_fscore(d) — 9-point binary financial health
#      score (4 profitability + 3 leverage/liquidity + 2 efficiency).
#      Returns (int 0-9, component dict).
#    • compute_sloan_ratio() — (NI − OCF) / Avg Total Assets.
#      Negative = earnings backed by cash (good). Positive = accrual
#      risk (bad). Academic alpha: 5-8%/yr underperformance for high
#      accrual stocks (Sloan 1996).
#    • compute_quality_score_elite() replaces compute_quality_score().
#      7 sub-signals with explicit weights:
#        Profitability (ROIC/ROE)   25%
#        Interest Coverage          15%
#        Operating Margin           15%  (non-Financials only)
#        Gross Margin + trend       20%
#        Piotroski F-Score          15%
#        Sloan Ratio (accruals)     10%
#    • quality_flag() updated: adds "HighAccruals" flag when
#      Sloan Ratio > 0.05.
#
#  GAP 3 — Valuation Composite: P/E only → FCF Yield + EV/EBITDA + P/E
#    • _fetch_yahoo_info_one() now fetches ev_ebitda, ev_sales, div_yield
#      from Yahoo info dict.
#    • _fetch_yahoo_deep_one() already computes fcf_ttm (added for Gap 2).
#    • merge_yahoo_phases() passes fcf_ttm through to merged dict.
#    • merge_all_sources() passes ev_ebitda, ev_sales, div_yield, fcf_ttm
#      through to merged dict.
#    • build_screener_table() computes fcf_yield inline from fcf_ttm/mc,
#      adds EV/EBITDA, FCF Yield%, EV/Sales, Div Yield%, Piotroski F,
#      Sloan Ratio to rows dict and num_cols.
#    • compute_valuation_subscore(elig) — blends three valuation signals:
#        FCF Yield%   40%  (higher = better)
#        EV/EBITDA    35%  (lower = cheaper)
#        Fwd/Trailing P/E  25%  (lower = cheaper)
#      Weights self-normalise when signals are missing. Falls back to
#      P/E-only if neither FCF nor EV/EBITDA data available.
#    • compute_rank_by_sector() calls compute_valuation_subscore()
#      instead of elite_factor_score(pe_input, ascending=True).
#
#  GAP 4 — Quality Score already covered by compute_quality_score_elite()
#      which adds Gross Margin (moat proxy) as a 20%-weighted sub-signal
#      with a 10% bonus when gross margin is improving YoY.
#
#  DISPLAY
#    • COLS list updated: EV/EBITDA, FCF Yield%, EV/Sales, Div Yield%,
#      Piotroski F, Sloan Ratio added after Quality Flag.
#    • sort_map updated: Piotroski F high, FCF Yield high, EV/EBITDA low.
#    • Reference Guide → Quality tab updated with all 7 sub-signals.
#    • Reference Guide → Valuation tab updated with composite explanation.
#
#  KEPT INTACT from v15
#    • winsorise(), mad_zscore(), elite_factor_score() normalisation
#    • All data-fetch concurrency, FMP integration, conviction score, UI
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
import random
import re
import warnings
import concurrent.futures
from datetime import datetime, date

warnings.filterwarnings("ignore")

try:
    from bs4 import BeautifulSoup
except ImportError:
    st.error("pip install beautifulsoup4")
    st.stop()

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_GROWTH_PCT_FOR_PEG   = 5.0
FETCH_TIMEOUT_PER_TICKER = 45

SECTOR_FACTOR_WEIGHTS = {
    "Information Technology": {
        "valuation": 0.20, "quality": 0.25, "peg": 0.25,
        "earn_traj": 0.15, "momentum": 0.15,
    },
    "Consumer Discretionary": {
        "valuation": 0.20, "quality": 0.20, "peg": 0.22,
        "earn_traj": 0.18, "momentum": 0.20,
    },
    "Communication Services": {
        "valuation": 0.22, "quality": 0.23, "peg": 0.22,
        "earn_traj": 0.18, "momentum": 0.15,
    },
    "Health Care": {
        "valuation": 0.25, "quality": 0.30, "peg": 0.18,
        "earn_traj": 0.15, "momentum": 0.12,
    },
    "Industrials": {
        "valuation": 0.25, "quality": 0.28, "peg": 0.18,
        "earn_traj": 0.17, "momentum": 0.12,
    },
    "Consumer Staples": {
        "valuation": 0.28, "quality": 0.32, "peg": 0.10,
        "earn_traj": 0.15, "momentum": 0.15,
    },
    "Financials": {
        "valuation": 0.30, "quality": 0.25, "peg": 0.18,
        "earn_traj": 0.17, "momentum": 0.10,
    },
    "Energy": {
        "valuation": 0.30, "quality": 0.18, "peg": 0.12,
        "earn_traj": 0.15, "momentum": 0.25,
    },
    "Materials": {
        "valuation": 0.28, "quality": 0.20, "peg": 0.12,
        "earn_traj": 0.15, "momentum": 0.25,
    },
    "Real Estate": {
        "valuation": 0.30, "quality": 0.18, "peg": 0.10,
        "earn_traj": 0.22, "momentum": 0.20,
    },
    "Utilities": {
        "valuation": 0.38, "quality": 0.27, "peg": 0.05,
        "earn_traj": 0.15, "momentum": 0.15,
    },
}

DEFAULT_FACTOR_WEIGHTS = {
    "valuation": 0.25, "quality": 0.25, "peg": 0.20,
    "earn_traj": 0.15, "momentum": 0.15,
}

ROE_PRIMARY_SECTORS = {"Financials"}

QUALITY_THRESHOLDS = {
    "roic_min":         8.0,
    "int_coverage_min": 3.0,
    "op_margin_min":    5.0,
}

OPERATING_CASH_PCT_OF_REV = 0.02


# ── Credentials ────────────────────────────────────────────────────────────────
def get_fmp_key():
    try:
        k = st.secrets["fmp"]["api_key"]
        return k if k and k.strip() and k != "YOUR_KEY_HERE" else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def sf(val):
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def normalise_pct(val):
    if val is None:
        return None
    v = float(val)
    return v * 100.0 if abs(v) < 5.0 else v


def normalise_pct_fmp(val):
    if val is None:
        return None
    return float(val) * 100.0


def fmt_mc(val):
    if pd.isna(val) or val == 0:
        return "N/A"
    if val >= 1e12:
        return "${:.2f}T".format(val / 1e12)
    if val >= 1e9:
        return "${:.1f}B".format(val / 1e9)
    return "${:.0f}M".format(val / 1e6)


# ── v15 Elite normalisation pipeline (unchanged) ──────────────────────────────

def winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return series.copy()
    q_lo = valid.quantile(lower)
    q_hi = valid.quantile(upper)
    return series.clip(lower=q_lo, upper=q_hi)


def mad_zscore(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)
    med = valid.median()
    mad = (valid - med).abs().median()
    if mad == 0:
        return pd.Series(0.0, index=series.index)
    return (series - med) / (1.4826 * mad)


def elite_factor_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    if series.dropna().empty:
        return pd.Series(0.0, index=series.index)
    ws = winsorise(series.copy())
    if ascending:
        ws = -ws
    z = mad_zscore(ws)
    z = z.clip(lower=-3.0, upper=3.0)
    z_min, z_max = z.min(), z.max()
    if z_max > z_min:
        scaled = (z - z_min) / (z_max - z_min) * 100.0
    else:
        scaled = pd.Series(50.0, index=series.index)
    return scaled.fillna(0.0)


def percentile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    """DEPRECATED v15 — kept for backward compat only."""
    result = pd.Series(index=series.index, dtype=float)
    valid  = series.notna()
    if valid.sum() == 0:
        return result.fillna(0.0)
    ranked = series[valid].rank(method="average", ascending=ascending)
    n      = valid.sum()
    result[valid]  = (ranked - 1) / (n - 1) * 100.0 if n > 1 else 50.0
    result[~valid] = 0.0
    return result


def missing_factor_penalty(row, factor_cols):
    missing = sum(1 for c in factor_cols if pd.isna(row.get(c)))
    if missing >= 3: return 0.70
    if missing == 2: return 0.85
    if missing == 1: return 0.95
    return 1.00


def revenue_growth_pct_cagr(rev4):
    try:
        if rev4 is None or len(rev4) != 4:
            return None
        q1, _, _, q4 = rev4
        if q1 is None or q4 is None:
            return None
        q1, q4 = float(q1), float(q4)
        if q1 <= 0 or q4 <= 0:
            return None
        return ((q4 / q1) ** (1 / 3) - 1) * 100.0
    except Exception:
        return None


# ── v16 NEW: Piotroski F-Score ────────────────────────────────────────────────

def compute_piotroski_fscore(d: dict) -> tuple:
    """
    9-point binary financial health score.
    d = merged data dict containing deep financial fields.

    Returns
    -------
    (score: int 0-9, components: dict of binary pass/fail per test)

    Scoring groups:
      Profitability  (P1–P4): ROA>0, OCF>0, ROA improving, OCF>NI
      Leverage       (L1–L3): lower LT debt ratio, higher current ratio,
                               no share dilution (≤2% growth allowed)
      Efficiency     (O1–O2): gross margin improving, asset turnover proxy
    """
    score = 0
    comp  = {}

    # ── Profitability ─────────────────────────────────────────────────────
    roa = d.get("roa_ttm")
    p1  = 1 if (roa is not None and roa > 0) else 0
    score += p1; comp["P1_ROA_pos"] = p1

    ocf = d.get("ocf_ttm")
    p2  = 1 if (ocf is not None and ocf > 0) else 0
    score += p2; comp["P2_OCF_pos"] = p2

    roa_prev = d.get("roa_prev")
    p3 = 1 if (roa is not None and roa_prev is not None and roa > roa_prev) else 0
    score += p3; comp["P3_ROA_improving"] = p3

    ni = d.get("net_income_ttm")
    p4 = 1 if (ocf is not None and ni is not None and ocf > ni) else 0
    score += p4; comp["P4_accruals_ok"] = p4

    # ── Leverage / Liquidity ──────────────────────────────────────────────
    ldr_now  = d.get("lt_debt_ratio_now")
    ldr_prev = d.get("lt_debt_ratio_prev")
    l1 = 1 if (ldr_now is not None and ldr_prev is not None
                and ldr_now < ldr_prev) else 0
    score += l1; comp["L1_leverage_down"] = l1

    cr_now  = d.get("current_ratio_now")
    cr_prev = d.get("current_ratio_prev")
    l2 = 1 if (cr_now is not None and cr_prev is not None
                and cr_now > cr_prev) else 0
    score += l2; comp["L2_liquidity_up"] = l2

    sh_now  = d.get("shares_now")
    sh_prev = d.get("shares_prev")
    l3 = 1 if (sh_now is not None and sh_prev is not None
                and sh_now <= sh_prev * 1.02) else 0
    score += l3; comp["L3_no_dilution"] = l3

    # ── Efficiency ────────────────────────────────────────────────────────
    gm_now  = d.get("gross_margin_now")
    gm_prev = d.get("gross_margin_prev")
    o1 = 1 if (gm_now is not None and gm_prev is not None
                and gm_now > gm_prev) else 0
    score += o1; comp["O1_gross_margin_up"] = o1

    rev_growth = d.get("rev_growth_pct")
    o2 = 1 if (rev_growth is not None and rev_growth > 0) else 0
    score += o2; comp["O2_asset_turn_up"] = o2

    return score, comp


# ── v16 NEW: Sloan Accruals Ratio ─────────────────────────────────────────────

def compute_sloan_ratio(net_income, ocf, total_assets_now, total_assets_prev):
    """
    Sloan Ratio = (Net Income TTM − OCF TTM) / Average Total Assets

    Interpretation
    --------------
    > +0.05 : Earnings significantly exceed cash — HIGH accrual/quality risk
     0.00 to +0.05: Mild accruals — acceptable
    -0.10 to 0.00: OCF slightly exceeds earnings — GOOD quality
    < -0.10 : OCF substantially exceeds earnings — EXCELLENT quality

    Academic: Sloan (1996) — high-accrual stocks underperform by 5-8%/yr.
    """
    try:
        if any(v is None for v in
               [net_income, ocf, total_assets_now, total_assets_prev]):
            return None
        avg_ta = (float(total_assets_now) + float(total_assets_prev)) / 2.0
        if avg_ta <= 0:
            return None
        return (float(net_income) - float(ocf)) / avg_ta
    except Exception:
        return None


# ── v16 NEW: Elite quality score (7 sub-signals) ─────────────────────────────

def compute_quality_score_elite(
    roic, roe, int_coverage, op_margin,
    gross_margin_now=None, gross_margin_prev=None,
    fcf_ni_ratio=None,
    piotroski_f=None,
    sloan_ratio=None,
    sector=None,
):
    """
    Replaces compute_quality_score() from v14/v15.

    Sub-signal weights
    ------------------
    1. Profitability (ROIC or ROE)     25%
    2. Interest Coverage               15%
    3. Operating Margin (non-Fin)      15%
    4. Gross Margin + YoY trend        20%
    5. Piotroski F-Score (0–9)         15%
    6. Sloan Ratio (accruals, inverted) 10%

    All weights normalised to sum to 1.0 so missing signals degrade
    gracefully without collapsing the total score to zero.
    """
    scores  = []
    weights = []

    # 1. Profitability
    profitability = (
        roe if sector in ROE_PRIMARY_SECTORS
        else (roic if roic is not None else roe)
    )
    if profitability is not None and not pd.isna(profitability):
        pf = float(profitability)
        s  = min(100.0, np.log1p(pf) / np.log1p(30.0) * 100.0) if pf > 0 else 0.0
    else:
        s = 0.0
    scores.append(s); weights.append(0.25)

    # 2. Interest Coverage
    if int_coverage is not None and not pd.isna(int_coverage):
        scores.append(min(100.0, max(0.0, float(int_coverage) / 10.0 * 100.0)))
    else:
        scores.append(0.0)
    weights.append(0.15)

    # 3. Operating Margin (non-Financials only)
    if sector not in ROE_PRIMARY_SECTORS:
        if op_margin is not None and not pd.isna(op_margin):
            scores.append(min(100.0, max(0.0, float(op_margin) / 40.0 * 100.0)))
        else:
            scores.append(0.0)
        weights.append(0.15)

    # 4. Gross Margin (moat proxy) — Buffett's pricing power test
    if gross_margin_now is not None and not pd.isna(gross_margin_now):
        gm_score = min(100.0, max(0.0, float(gross_margin_now) / 60.0 * 100.0))
        # +10% bonus if gross margin is improving YoY (capped at 100)
        if gross_margin_prev is not None and float(gross_margin_now) > float(gross_margin_prev):
            gm_score = min(100.0, gm_score * 1.10)
        scores.append(gm_score)
    else:
        scores.append(0.0)
    weights.append(0.20)

    # 5. Piotroski F-Score (0–9 mapped to 0–100)
    if piotroski_f is not None and not pd.isna(piotroski_f):
        scores.append(float(piotroski_f) / 9.0 * 100.0)
    else:
        scores.append(0.0)
    weights.append(0.15)

    # 6. Sloan Ratio — lower (more negative) = better earnings quality
    # Map [-0.15, +0.05] → [100, 0] linearly (inverted)
    if sloan_ratio is not None and not pd.isna(sloan_ratio):
        sr         = float(sloan_ratio)
        sr_clamped = max(-0.15, min(0.05, sr))
        sloan_score = (0.05 - sr_clamped) / 0.20 * 100.0
        scores.append(sloan_score)
    else:
        scores.append(0.0)
    weights.append(0.10)

    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(s * w for s, w in zip(scores, weights)) / total_w


# ── v16 NEW: Composite valuation sub-score ────────────────────────────────────

def compute_valuation_subscore(elig: pd.DataFrame) -> pd.Series:
    """
    Blend three valuation signals using elite_factor_score() normalisation.

    Signal hierarchy (weights self-normalise when signals are missing):
      FCF Yield%   40%  — higher yield = cheaper (ascending=False)
      EV/EBITDA    35%  — lower multiple = cheaper (ascending=True)
      Fwd/Trail PE 25%  — lower multiple = cheaper (ascending=True)

    Minimum data threshold: a signal is included only when ≥5 valid
    values exist in the sector (below that, the signal is noise).
    """
    scores  = pd.DataFrame(index=elig.index)
    weights = []

    if "FCF Yield%" in elig.columns and elig["FCF Yield%"].notna().sum() >= 5:
        scores["fcf"] = elite_factor_score(elig["FCF Yield%"], ascending=False)
        weights.append(("fcf", 0.40))

    if "EV/EBITDA" in elig.columns and elig["EV/EBITDA"].notna().sum() >= 5:
        scores["evebitda"] = elite_factor_score(elig["EV/EBITDA"], ascending=True)
        weights.append(("evebitda", 0.35))

    pe_input = elig["Fwd P/E"].fillna(elig["P/E"])
    if pe_input.notna().sum() >= 5:
        scores["pe"] = elite_factor_score(pe_input, ascending=True)
        weights.append(("pe", 0.25))

    if not weights:
        return pd.Series(50.0, index=elig.index)

    total_w   = sum(w for _, w in weights)
    composite = sum(
        scores[k] * (w / total_w)
        for k, w in weights
        if k in scores.columns
    )
    return composite.fillna(0.0)


# ── S&P 500 universe ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_sp500_constituents():
    url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r    = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    tbl  = soup.find("table", {"id": "constituents"})
    if tbl is None:
        raise RuntimeError("Wikipedia S&P 500 table not found")
    data = []
    for row in tbl.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) >= 4:
            raw     = cols[0].get_text(strip=True).replace(".", "-")
            cleaned = re.sub(r"[^A-Za-z0-9-]", "", raw).upper()
            sector  = cols[2].get_text(strip=True)
            if cleaned and sector and re.match(r"[A-Z][A-Z0-9-]{0,5}$", cleaned):
                data.append({"Ticker": cleaned, "Sector": sector})
    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════════════════════
# PRICES + MOMENTUM  (unchanged from v15)
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_price_momentum_one(t):
    result = {
        "price": None, "hi52": None, "lo52": None,
        "ret_1mo": None, "ret_3mo": None, "ret_6mo": None,
        "trailing_vol": None, "momentum_score": None,
    }
    try:
        obj  = yf.Ticker(t)
        hist = obj.history(period="12mo", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return t, result
        closes = hist["Close"].dropna()
        if closes.empty:
            return t, result

        result["price"] = float(closes.iloc[-1])
        result["hi52"]  = float(closes.max())
        result["lo52"]  = float(closes.min())

        monthly = closes.resample("ME").last().dropna()
        if len(monthly) < 2:
            return t, result
        px_now = float(monthly.iloc[-1])
        if px_now <= 0:
            return t, result

        def ret_mo(n):
            idx = -(n + 1)
            if abs(idx) > len(monthly):
                return None
            px = float(monthly.iloc[idx])
            return (px_now / px - 1) * 100.0 if px > 0 else None

        r1, r3, r6 = ret_mo(1), ret_mo(3), ret_mo(6)
        result["ret_1mo"] = r1
        result["ret_3mo"] = r3
        result["ret_6mo"] = r6

        if len(closes) >= 20:
            daily_rets = closes.pct_change().dropna().tail(90)
            if len(daily_rets) >= 15:
                result["trailing_vol"] = float(
                    daily_rets.std() * np.sqrt(252) * 100.0
                )

        t_vol = result["trailing_vol"]
        if r6 is not None and r1 is not None:
            skip_raw = r6 - r1
            result["momentum_score"] = (
                skip_raw / t_vol if (t_vol and t_vol > 0) else skip_raw
            )
    except Exception:
        pass
    return t, result


@st.cache_data(ttl=3600)
def fetch_price_momentum_all(tickers):
    tl     = list(tickers)
    out    = {t: {} for t in tl}
    CHUNK  = 25; WKRS = 10; SLEEP = 1.0
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog   = st.progress(0); status = st.empty()

    for ci, chunk in enumerate(chunks):
        status.text("Prices + Momentum: chunk {}/{} ({} done)...".format(
            ci + 1, len(chunks), ci * CHUNK))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futs = {ex.submit(_fetch_price_momentum_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(
                    futs, timeout=FETCH_TIMEOUT_PER_TICKER * len(chunk)):
                try:
                    t, d = fut.result(); out[t] = d
                except Exception:
                    out[futs[fut]] = {}
        prog.progress((ci + 1) / len(chunks))
        if ci < len(chunks) - 1:
            time.sleep(SLEEP + random.uniform(0, 0.3))

    prog.empty(); status.empty()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Yahoo info  (v16: adds ev_ebitda, ev_sales, div_yield)
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_yahoo_info_one(t):
    result = {
        "pe": None, "pe_src": None, "fwd_pe": None,
        "peg": None, "peg_src": None,
        "roe": None, "op_margin": None, "debt_eq": None,
        "eps_growth": None, "earn_traj": None,
        "mc": None, "hi52": None, "lo52": None,
        "roic": None, "int_coverage": None,
        "rev4": [None, None, None, None],
        # v16 new ─────────────────────────────
        "ev_ebitda": None,
        "ev_sales":  None,
        "div_yield": None,
    }
    try:
        obj = yf.Ticker(t)
        try:
            fi = obj.fast_info
            if fi is not None:
                mc_fi = sf(getattr(fi, "market_cap", None))
                if mc_fi:
                    result["mc"] = mc_fi
        except Exception:
            pass

        info = {}
        for _ in range(2):
            try:
                info = obj.info or {}
                if info.get("trailingPE") or info.get("pegRatio") or info.get("forwardPE"):
                    break
            except Exception:
                pass
            time.sleep(0.5 + random.uniform(0, 0.5))

        px = sf(info.get("currentPrice") or info.get("regularMarketPrice"))

        t_pe  = sf(info.get("trailingPE"))
        t_eps = sf(info.get("trailingEps"))
        if t_pe and 0 < t_pe <= 10_000:
            result["pe"] = t_pe; result["pe_src"] = "Yahoo"
        elif t_eps and t_eps > 0 and px and px > 0:
            result["pe"] = px / t_eps; result["pe_src"] = "Yahoo(calc)"

        f_pe  = sf(info.get("forwardPE"))
        f_eps = sf(info.get("forwardEps"))
        if f_pe and 0 < f_pe <= 10_000:
            result["fwd_pe"] = f_pe
        elif f_eps and f_eps > 0 and px and px > 0:
            result["fwd_pe"] = px / f_eps

        peg_y = sf(info.get("pegRatio"))
        if peg_y and 0 < peg_y <= 500:
            result["peg"] = peg_y; result["peg_src"] = "Yahoo"

        roe_y = sf(info.get("returnOnEquity"))
        if roe_y is not None:
            result["roe"] = roe_y * 100.0

        om_y = sf(info.get("operatingMargins"))
        if om_y is not None:
            result["op_margin"] = om_y * 100.0

        de_y = sf(info.get("debtToEquity"))
        if de_y is not None:
            result["debt_eq"] = de_y / 100.0

        eg_y = sf(info.get("earningsGrowth"))
        if eg_y is not None:
            result["eps_growth"] = eg_y * 100.0

        fwd_eps_val   = sf(info.get("forwardEps"))
        trail_eps_val = sf(info.get("trailingEps"))
        if (fwd_eps_val is not None and trail_eps_val is not None
                and abs(trail_eps_val) > 0.01):
            earn_traj_raw = (fwd_eps_val - trail_eps_val) / abs(trail_eps_val)
            clipped       = max(-1.0, min(1.0, earn_traj_raw))
            if trail_eps_val < 0 and fwd_eps_val < 0:
                clipped = min(clipped, 0.30)
            result["earn_traj"] = clipped

        if result["mc"] is None:
            mc_y = sf(info.get("marketCap"))
            if mc_y:
                result["mc"] = mc_y

        # ── v16 NEW ───────────────────────────────────────────────────────
        ev_raw     = sf(info.get("enterpriseValue"))
        ebitda_raw = sf(info.get("ebitda"))
        if ev_raw and ebitda_raw and ebitda_raw > 0:
            ev_eb = ev_raw / ebitda_raw
            if 0 < ev_eb < 200:
                result["ev_ebitda"] = ev_eb

        rev_ttm_y = sf(info.get("totalRevenue"))
        if ev_raw and rev_ttm_y and rev_ttm_y > 0:
            ev_s = ev_raw / rev_ttm_y
            if 0 < ev_s < 100:
                result["ev_sales"] = ev_s

        dy = sf(info.get("dividendYield"))
        if dy is not None:
            result["div_yield"] = dy * 100.0

    except Exception:
        pass
    return t, result


@st.cache_data(ttl=86400)
def fetch_yahoo_info_all(tickers, _cache_date=None):
    tl     = list(tickers)
    out    = {}
    CHUNK  = 30; WKRS = 8; SLEEP = 1.5
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog   = st.progress(0); status = st.empty()

    for ci, chunk in enumerate(chunks):
        status.text("Phase 1/2 — Yahoo info: chunk {}/{} ({} done)...".format(
            ci + 1, len(chunks), ci * CHUNK))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futs = {ex.submit(_fetch_yahoo_info_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(
                    futs, timeout=FETCH_TIMEOUT_PER_TICKER * len(chunk)):
                try:
                    t, d = fut.result(); out[t] = d
                except Exception:
                    out[futs[fut]] = {}
        prog.progress((ci + 1) / len(chunks))
        if ci < len(chunks) - 1:
            time.sleep(SLEEP + random.uniform(0, 0.5))

    prog.empty(); status.empty()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Deep financials  (v16: +14 new fields for Piotroski/Sloan/FCF)
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_yahoo_deep_one(t):
    result = {
        # v14/v15 fields
        "roic":         None,
        "int_coverage": None,
        "rev4":         [None, None, None, None],
        # v16 new ─────────────────────────────────────────────────────────
        "ocf_ttm":           None,
        "fcf_ttm":           None,
        "net_income_ttm":    None,
        "gross_profit_ttm":  None,
        "gross_margin_now":  None,
        "gross_margin_prev": None,
        "roa_ttm":           None,
        "roa_prev":          None,
        "total_assets_now":  None,
        "total_assets_prev": None,
        "lt_debt_ratio_now":  None,
        "lt_debt_ratio_prev": None,
        "current_ratio_now":  None,
        "current_ratio_prev": None,
        "shares_now":  None,
        "shares_prev": None,
    }
    try:
        obj  = yf.Ticker(t)
        qfin = obj.quarterly_financials
        qbs  = obj.quarterly_balance_sheet
        qcf  = obj.quarterly_cashflow
        info = {}
        try:
            info = obj.info or {}
        except Exception:
            pass

        # ── Revenue (unchanged from v15) ──────────────────────────────────
        if qfin is not None and not qfin.empty:
            rev_row = next(
                (nm for nm in ["Total Revenue", "Revenue"] if nm in qfin.index),
                None,
            )
            if rev_row:
                rev_series = qfin.loc[rev_row].sort_index().dropna().tail(4)
                if len(rev_series) == 4:
                    result["rev4"] = [float(x) for x in rev_series.values]
                elif len(rev_series) > 0:
                    vals = [float(x) for x in rev_series.values]
                    result["rev4"] = ([None] * (4 - len(vals))) + vals

        # ── Interest coverage (unchanged) ─────────────────────────────────
        if qfin is not None and not qfin.empty:
            ebit_row = next(
                (nm for nm in ["EBIT", "Operating Income", "Ebit"]
                 if nm in qfin.index), None)
            int_row = next(
                (nm for nm in [
                    "Interest Expense",
                    "Interest Expense Non Operating",
                    "Net Interest Income",
                ] if nm in qfin.index), None)
            if ebit_row and int_row:
                ebit_ttm = qfin.loc[ebit_row].dropna().head(4).sum()
                int_ttm  = abs(qfin.loc[int_row].dropna().head(4).sum())
                if int_ttm > 0 and ebit_ttm > 0:
                    result["int_coverage"] = min(float(ebit_ttm / int_ttm), 100.0)

        # ── ROIC (unchanged) ──────────────────────────────────────────────
        if qfin is not None and not qfin.empty and qbs is not None and not qbs.empty:
            op_inc_row = next(
                (nm for nm in ["Operating Income", "EBIT", "Ebit"]
                 if nm in qfin.index), None)
            tax_row = next(
                (nm for nm in ["Tax Provision", "Income Tax Expense", "Tax Expense"]
                 if nm in qfin.index), None)
            pretax_row = next(
                (nm for nm in ["Pretax Income", "Income Before Tax", "EBT"]
                 if nm in qfin.index), None)
            if op_inc_row:
                op_inc_ttm   = float(qfin.loc[op_inc_row].dropna().head(4).sum())
                eff_tax_rate = 0.21
                if tax_row and pretax_row:
                    tax_ttm    = float(qfin.loc[tax_row].dropna().head(4).sum())
                    pretax_ttm = float(qfin.loc[pretax_row].dropna().head(4).sum())
                    if pretax_ttm > 0 and tax_ttm >= 0:
                        cr = tax_ttm / pretax_ttm
                        if 0 < cr < 0.6:
                            eff_tax_rate = cr
                nopat = op_inc_ttm * (1 - eff_tax_rate)

                equity_val = next(
                    (float(qbs.loc[nm].dropna().iloc[0])
                     for nm in ["Total Stockholders Equity", "Stockholders Equity",
                                "Common Stock Equity",
                                "Total Equity Gross Minority Interest"]
                     if nm in qbs.index and len(qbs.loc[nm].dropna()) > 0), None)
                debt_val = next(
                    (float(qbs.loc[nm].dropna().iloc[0])
                     for nm in ["Total Debt", "Net Debt", "Long Term Debt",
                                "Long Term Debt And Capital Lease Obligation"]
                     if nm in qbs.index and len(qbs.loc[nm].dropna()) > 0), None)
                cash_val = next(
                    (float(qbs.loc[nm].dropna().iloc[0])
                     for nm in ["Cash And Cash Equivalents",
                                "Cash Cash Equivalents And Short Term Investments",
                                "Cash Financial", "Cash And Short Term Investments"]
                     if nm in qbs.index and len(qbs.loc[nm].dropna()) > 0), None)

                cash_use = 0
                if cash_val is not None:
                    rev4_vals = result["rev4"]
                    if all(v is not None for v in rev4_vals):
                        rev_ttm = sum(rev4_vals)
                        if rev_ttm > 0:
                            cash_use = max(
                                0.0, cash_val - OPERATING_CASH_PCT_OF_REV * rev_ttm
                            )
                        else:
                            cash_use = cash_val
                    else:
                        cash_use = cash_val

                if equity_val is not None and debt_val is not None:
                    ic_val = equity_val + debt_val - cash_use
                    if ic_val > 0 and nopat != 0:
                        roic_c = (nopat / ic_val) * 100.0
                        if -100 < roic_c < 200:
                            result["roic"] = roic_c

        # ── v16 NEW: FCF = OCF − CapEx ────────────────────────────────────
        if qcf is not None and not qcf.empty:
            ocf_row = next(
                (n for n in ["Operating Cash Flow", "Cash From Operations",
                             "Total Cash From Operating Activities"]
                 if n in qcf.index), None)
            capex_row = next(
                (n for n in ["Capital Expenditure", "Purchase Of PPE",
                             "Capital Expenditures",
                             "Purchases Of Property Plant And Equipment"]
                 if n in qcf.index), None)
            if ocf_row:
                ocf_ttm = float(qcf.loc[ocf_row].dropna().head(4).sum())
                result["ocf_ttm"] = ocf_ttm
                if capex_row:
                    capex_ttm = abs(
                        float(qcf.loc[capex_row].dropna().head(4).sum())
                    )
                    result["fcf_ttm"] = ocf_ttm - capex_ttm

        # ── v16 NEW: Gross Profit + Net Income ───────────────────────────
        if qfin is not None and not qfin.empty:
            gp_row = next(
                (n for n in ["Gross Profit", "Gross Income"]
                 if n in qfin.index), None)
            rev_row2 = next(
                (n for n in ["Total Revenue", "Revenue"]
                 if n in qfin.index), None)
            ni_row = next(
                (n for n in ["Net Income", "Net Income Common Stockholders"]
                 if n in qfin.index), None)

            if gp_row and rev_row2:
                gp_ttm  = float(qfin.loc[gp_row].dropna().head(4).sum())
                rev_ttm2 = float(qfin.loc[rev_row2].dropna().head(4).sum())
                if rev_ttm2 > 0:
                    result["gross_profit_ttm"] = gp_ttm
                    result["gross_margin_now"]  = gp_ttm / rev_ttm2 * 100.0

                # Prior-year gross margin (quarters 5–8)
                rev_all = qfin.loc[rev_row2].dropna()
                gp_all  = qfin.loc[gp_row].dropna()
                if len(rev_all) >= 8 and len(gp_all) >= 8:
                    rev_prev = float(rev_all.iloc[4:8].sum())
                    gp_prev  = float(gp_all.iloc[4:8].sum())
                    if rev_prev > 0:
                        result["gross_margin_prev"] = gp_prev / rev_prev * 100.0

            if ni_row:
                result["net_income_ttm"] = float(
                    qfin.loc[ni_row].dropna().head(4).sum()
                )

        # ── v16 NEW: Balance sheet — Piotroski signals ────────────────────
        if qbs is not None and not qbs.empty:
            ta_row = next(
                (n for n in ["Total Assets", "Assets"] if n in qbs.index), None)
            ltd_row = next(
                (n for n in ["Long Term Debt",
                             "Long Term Debt And Capital Lease Obligation"]
                 if n in qbs.index), None)
            ca_row = next(
                (n for n in ["Current Assets", "Total Current Assets"]
                 if n in qbs.index), None)
            cl_row = next(
                (n for n in ["Current Liabilities", "Total Current Liabilities"]
                 if n in qbs.index), None)

            if ta_row:
                ta_vals = qbs.loc[ta_row].dropna()
                if len(ta_vals) >= 1:
                    ta_now = float(ta_vals.iloc[0])
                    result["total_assets_now"] = ta_now
                    if len(ta_vals) >= 5:
                        ta_prev = float(ta_vals.iloc[4])
                        result["total_assets_prev"] = ta_prev
                        avg_ta = (ta_now + ta_prev) / 2.0
                        ni_val = result.get("net_income_ttm")
                        if avg_ta > 0 and ni_val is not None:
                            result["roa_ttm"] = ni_val / avg_ta * 100.0

            if ltd_row:
                ltd_vals = qbs.loc[ltd_row].dropna()
                ta_n     = result.get("total_assets_now")
                if len(ltd_vals) >= 1 and ta_n and ta_n > 0:
                    result["lt_debt_ratio_now"] = float(ltd_vals.iloc[0]) / ta_n
                if len(ltd_vals) >= 5:
                    ta_p = result.get("total_assets_prev")
                    if ta_p and ta_p > 0:
                        result["lt_debt_ratio_prev"] = float(ltd_vals.iloc[4]) / ta_p

            if ca_row and cl_row:
                ca_vals = qbs.loc[ca_row].dropna()
                cl_vals = qbs.loc[cl_row].dropna()
                if len(ca_vals) >= 1 and len(cl_vals) >= 1:
                    cl_now = float(cl_vals.iloc[0])
                    if cl_now > 0:
                        result["current_ratio_now"] = float(ca_vals.iloc[0]) / cl_now
                if len(ca_vals) >= 5 and len(cl_vals) >= 5:
                    cl_prev = float(cl_vals.iloc[4])
                    if cl_prev > 0:
                        result["current_ratio_prev"] = float(ca_vals.iloc[4]) / cl_prev

        # ── v16 NEW: Shares outstanding for dilution check ────────────────
        result["shares_now"]  = sf(info.get("sharesOutstanding"))
        result["shares_prev"] = sf(info.get("floatShares"))  # proxy

    except Exception:
        pass
    return t, result


@st.cache_data(ttl=86400)
def fetch_yahoo_deep_financials(tickers_filtered, _cache_date=None):
    tl  = list(tickers_filtered)
    out = {}
    if not tl:
        return out

    CHUNK  = 20; WKRS = 6; SLEEP = 2.0
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog   = st.progress(0); status = st.empty()

    for ci, chunk in enumerate(chunks):
        status.text(
            "Phase 2/2 — Deep financials: chunk {}/{} ({}/{})...".format(
                ci + 1, len(chunks), min((ci + 1) * CHUNK, len(tl)), len(tl)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futs = {ex.submit(_fetch_yahoo_deep_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(
                    futs, timeout=FETCH_TIMEOUT_PER_TICKER * len(chunk)):
                try:
                    t, d = fut.result(); out[t] = d
                except Exception:
                    out[futs[fut]] = {}
        prog.progress((ci + 1) / len(chunks))
        if ci < len(chunks) - 1:
            time.sleep(SLEEP + random.uniform(0, 0.5))

    prog.empty(); status.empty()
    return out


def _pre_filter_tickers(info_map, universe_df, mc_min_b, pe_max):
    keep = []
    for t in universe_df["Ticker"]:
        d     = info_map.get(t, {})
        mc_ok = (d.get("mc") is None) or (d.get("mc") >= mc_min_b * 1e9)
        pe_ok = (d.get("pe") is None) or (d.get("pe") <= pe_max)
        if mc_ok and pe_ok:
            keep.append(t)
    return keep


def merge_yahoo_phases(info_map, deep_map, tickers):
    """
    v16: passes all 14 new deep fields through to merged dict.
    """
    NEW_DEEP_FIELDS = [
        "ocf_ttm", "fcf_ttm", "net_income_ttm", "gross_profit_ttm",
        "gross_margin_now", "gross_margin_prev",
        "roa_ttm", "roa_prev",
        "total_assets_now", "total_assets_prev",
        "lt_debt_ratio_now", "lt_debt_ratio_prev",
        "current_ratio_now", "current_ratio_prev",
        "shares_now", "shares_prev",
    ]
    merged = {}
    for t in tickers:
        base = dict(info_map.get(t, {}))
        deep = deep_map.get(t, {})
        base["roic"] = (
            deep.get("roic") if deep.get("roic") is not None
            else base.get("roic")
        )
        base["int_coverage"] = (
            deep.get("int_coverage") if deep.get("int_coverage") is not None
            else base.get("int_coverage")
        )
        base["rev4"] = deep.get("rev4", [None, None, None, None])
        for f in NEW_DEEP_FIELDS:
            base[f] = deep.get(f)
        merged[t] = base
    return merged


# ── FMP /quote bulk ────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_quotes_if_available(tickers, api_key):
    out = {}
    if not api_key:
        return out
    tl     = list(tickers)
    chunks = [tl[i:i+100] for i in range(0, len(tl), 100)]
    for chunk in chunks:
        url = ("https://financialmodelingprep.com/api/v3/quote/{}?apikey={}".format(
            ",".join(chunk), api_key))
        try:
            r = requests.get(url, timeout=20); r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                continue
            for item in data:
                t  = str(item.get("symbol", "")).upper().strip()
                if not t:
                    continue
                pe = sf(item.get("pe")); mc = sf(item.get("marketCap"))
                if pe is not None and (pe <= 0 or pe > 10_000):
                    pe = None
                out[t] = {"pe": pe, "mc": mc,
                          "pe_src": "FMP-quote" if pe is not None else None}
        except Exception:
            pass
        time.sleep(0.3)
    return out


# ── FMP /ratios-ttm ────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_ratios_if_available(tickers, api_key):
    out = {}
    if not api_key:
        return out
    try:
        r    = requests.get(
            "https://financialmodelingprep.com/api/v3/ratios-ttm/AAPL?apikey={}".format(
                api_key), timeout=10)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return out
        st.session_state["fmp_ratios_fields"] = list(data[0].keys())
    except Exception:
        return out

    def fetch_one(t):
        url = ("https://financialmodelingprep.com/api/v3/ratios-ttm/{}?apikey={}".format(
            t, api_key))
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 429:
                time.sleep(3.0)
                r = requests.get(url, timeout=12)
            r.raise_for_status()
            d = r.json()
            if not isinstance(d, list) or len(d) == 0:
                return t, {}
            item = d[0]
            peg_raw = sf(item.get("priceEarningsGrowthRatioTTM"))
            peg     = peg_raw if (peg_raw and 0 < peg_raw <= 500) else None
            roic_raw = sf(item.get("returnOnInvestedCapitalTTM"))
            roic = normalise_pct_fmp(roic_raw) if roic_raw is not None else None
            roe_raw = sf(item.get("returnOnEquityTTM"))
            roe  = normalise_pct_fmp(roe_raw) if roe_raw is not None else None
            om_raw = sf(item.get("operatingProfitMarginTTM"))
            om   = normalise_pct_fmp(om_raw) if om_raw is not None else None
            ic_raw = sf(item.get("interestCoverageTTM"))
            ic   = min(float(ic_raw), 100.0) if (ic_raw and ic_raw > 0) else None
            de   = sf(item.get("debtEquityRatioTTM"))
            fmp_pe_raw = sf(item.get("priceToEarningsRatioTTM"))
            fmp_pe = (fmp_pe_raw if (fmp_pe_raw and 0 < fmp_pe_raw <= 10_000)
                      else None)
            return t, {"peg": peg, "roic": roic, "roe": roe, "op_margin": om,
                       "int_coverage": ic, "debt_eq": de,
                       "fmp_trailing_pe": fmp_pe,
                       "peg_src": "FMP-ratios" if peg else None}
        except Exception:
            return t, {}

    tl = list(tickers)
    for ci, chunk in enumerate([tl[i:i+50] for i in range(0, len(tl), 50)]):
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(fetch_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    t, d = fut.result()
                    if d:
                        out[t] = d
                except Exception:
                    pass
        if ci < (len(tl) // 50):
            time.sleep(1.0)
    return out


# ── Merge all sources (v16: pass-through new fields) ──────────────────────────
def merge_all_sources(yahoo_data, fmp_quotes, fmp_ratios, tickers):
    NEW_FIELDS = [
        "ev_ebitda", "ev_sales", "div_yield",
        "ocf_ttm", "fcf_ttm", "net_income_ttm", "gross_profit_ttm",
        "gross_margin_now", "gross_margin_prev",
        "roa_ttm", "roa_prev",
        "total_assets_now", "total_assets_prev",
        "lt_debt_ratio_now", "lt_debt_ratio_prev",
        "current_ratio_now", "current_ratio_prev",
        "shares_now", "shares_prev",
    ]
    merged = {}
    for t in tickers:
        yb = yahoo_data.get(t, {})
        fq = fmp_quotes.get(t, {})
        fr = fmp_ratios.get(t, {})

        def first(*vals):
            for v in vals:
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return v
            return None

        pe_val = first(fq.get("pe"), fr.get("fmp_trailing_pe"), yb.get("pe"))
        pe_src = (
            "FMP-quote"  if fq.get("pe")              is not None else
            "FMP-ratios" if fr.get("fmp_trailing_pe") is not None else
            yb.get("pe_src", "Yahoo")
        )

        row = {
            "pe": pe_val, "pe_src": pe_src,
            "fwd_pe":    yb.get("fwd_pe"),
            "peg":       first(fr.get("peg"),         yb.get("peg")),
            "peg_src":   ("FMP-ratios" if fr.get("peg") is not None
                          else yb.get("peg_src", "Yahoo")
                          if yb.get("peg") is not None else "—"),
            "roic":      first(fr.get("roic"),         yb.get("roic")),
            "roe":       first(fr.get("roe"),           yb.get("roe")),
            "int_coverage": first(fr.get("int_coverage"), yb.get("int_coverage")),
            "op_margin": first(fr.get("op_margin"),    yb.get("op_margin")),
            "debt_eq":   first(fr.get("debt_eq"),       yb.get("debt_eq")),
            "eps_growth":    yb.get("eps_growth"),
            "growth_src":    "Yahoo" if yb.get("eps_growth") is not None else None,
            "earn_traj":     yb.get("earn_traj"),
            "mc":        first(fq.get("mc"), yb.get("mc")),
            "rev4":          yb.get("rev4", [None, None, None, None]),
        }
        for f in NEW_FIELDS:
            row[f] = yb.get(f)
        merged[t] = row
    return merged


# ── Quality flag (v16: +HighAccruals) ────────────────────────────────────────
def quality_flag(roic, roe, ic, om, sloan_ratio=None, sector=None):
    flags = []
    if sector in ROE_PRIMARY_SECTORS:
        profitability = roe; prof_label = "ROE"
    else:
        profitability = roic if (roic is not None and not pd.isna(roic)) else roe
        prof_label    = "ROIC" if (roic is not None and not pd.isna(roic)) else "ROE"

    if (profitability is not None and not pd.isna(profitability)
            and profitability < QUALITY_THRESHOLDS["roic_min"]):
        flags.append("{}<8%".format(prof_label))
    if ic is not None and not pd.isna(ic) and ic < QUALITY_THRESHOLDS["int_coverage_min"]:
        flags.append("IntCov<3x")
    if sector not in ROE_PRIMARY_SECTORS:
        if om is not None and not pd.isna(om) and om < QUALITY_THRESHOLDS["op_margin_min"]:
            flags.append("Margin<5%")
    # v16 new
    if sloan_ratio is not None and not pd.isna(sloan_ratio) and float(sloan_ratio) > 0.05:
        flags.append("HighAccruals")

    return ", ".join(flags) if flags else "Pass"


# ── Conviction Score (unchanged from v15) ─────────────────────────────────────
def compute_conviction_scores(scr):
    KEY_FACTORS = ["P/E", "Fwd P/E", "PEG", "Quality Score", "Momentum Score", "Earn Traj"]
    n_factors   = len(KEY_FACTORS)
    scr         = scr.copy()

    def completeness(row):
        return sum(1 for c in KEY_FACTORS
                   if c in row.index and pd.notna(row[c])) / n_factors

    scr["_completeness"]  = scr.apply(completeness, axis=1)
    overall_median_pe     = scr["P/E"].median()
    sector_pe_map         = scr.groupby("Sector")["P/E"].median()

    def sector_discount(sector):
        if pd.isna(overall_median_pe) or overall_median_pe == 0:
            return 1.0
        s_pe = sector_pe_map.get(sector)
        if pd.isna(s_pe) or s_pe == 0:
            return 1.0
        return float(np.clip(overall_median_pe / s_pe, 0.7, 1.3))

    scr["_sec_discount"] = scr["Sector"].map(sector_discount)
    raw = scr["Score"] * scr["_completeness"] * scr["_sec_discount"]
    c_min, c_max = raw.min(), raw.max()
    scr["Conviction Score"] = (
        (raw - c_min) / (c_max - c_min) * 100.0
        if c_max > c_min else pd.Series(50.0, index=scr.index)
    )
    return scr.drop(columns=["_completeness", "_sec_discount"])


# ══════════════════════════════════════════════════════════════════════════════
# RANKING  (v16: compute_valuation_subscore replaces plain elite_factor_score
#           for valuation; quality score uses compute_quality_score_elite)
# ══════════════════════════════════════════════════════════════════════════════
def compute_rank_by_sector(scr):
    scr = scr.copy()
    scr["Score"] = pd.NA
    scr["Rank"]  = pd.NA

    for sector in scr["Sector"].dropna().unique():
        g    = scr[scr["Sector"] == sector].copy()
        elig = g[g["Eligible"]].copy()
        if elig.empty:
            continue

        W = SECTOR_FACTOR_WEIGHTS.get(sector, DEFAULT_FACTOR_WEIGHTS)

        # v16: composite valuation (FCF Yield + EV/EBITDA + PE)
        elig["_s_val"]   = compute_valuation_subscore(elig)

        # v15: MAD z-score for PEG / Momentum / EarnTraj (unchanged)
        elig["_s_peg"]   = elite_factor_score(elig["PEG"],            ascending=True)
        elig["_s_mom"]   = elite_factor_score(elig["Momentum Score"], ascending=False)
        elig["_s_etraj"] = elite_factor_score(elig["Earn Traj"],      ascending=False)

        # Quality — min-max rescale (log transform already applied in quality fn)
        qs    = elig["Quality Score"]
        q_min = qs.min(); q_max = qs.max()
        if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min:
            elig["_s_quality"] = (qs - q_min) / (q_max - q_min) * 100.0
        else:
            elig["_s_quality"] = qs.fillna(0.0)
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)

        raw = (
            W["valuation"] * elig["_s_val"]     +
            W["quality"]   * elig["_s_quality"] +
            W["peg"]       * elig["_s_peg"]     +
            W["earn_traj"] * elig["_s_etraj"]   +
            W["momentum"]  * elig["_s_mom"]
        )

        factor_cols = ["P/E", "PEG", "Quality Score", "Earn Traj", "Momentum Score"]
        penalties   = elig.apply(
            lambda r: missing_factor_penalty(r, factor_cols), axis=1
        )
        raw = raw * penalties

        elig["Score"] = raw
        elig = elig.sort_values("Score", ascending=False)
        elig["Rank"] = range(1, len(elig) + 1)
        scr.loc[elig.index, "Score"] = elig["Score"]
        scr.loc[elig.index, "Rank"]  = elig["Rank"]

    return scr


# ── Build screener table (v16: Piotroski, Sloan, FCF, EV/EBITDA) ──────────────
def build_screener_table(universe_df, pm_map, merged_map):
    rows = []
    for _, r in universe_df.iterrows():
        t   = r["Ticker"]
        sec = r["Sector"]

        pm        = pm_map.get(t, {})
        price     = to_num(pm.get("price"))
        hi        = to_num(pm.get("hi52"))
        lo        = to_num(pm.get("lo52"))
        ret_1mo   = to_num(pm.get("ret_1mo"))
        ret_3mo   = to_num(pm.get("ret_3mo"))
        ret_6mo   = to_num(pm.get("ret_6mo"))
        t_vol     = to_num(pm.get("trailing_vol"))
        mom_score = to_num(pm.get("momentum_score"))

        fi        = merged_map.get(t, {})
        mc        = to_num(fi.get("mc"))
        pe        = to_num(fi.get("pe"))
        fwd       = to_num(fi.get("fwd_pe"))
        roic      = to_num(fi.get("roic"))
        roe       = to_num(fi.get("roe"))
        ic        = to_num(fi.get("int_coverage"))
        om        = to_num(fi.get("op_margin"))
        de        = to_num(fi.get("debt_eq"))
        earn_traj = to_num(fi.get("earn_traj"))

        pos52 = None
        if pd.notna(price) and pd.notna(hi) and pd.notna(lo) and hi != lo:
            pos52 = float((price - lo) / (hi - lo) * 100.0)

        rev4 = fi.get("rev4", [None, None, None, None])
        rq1, rq2, rq3, rq4 = [to_num(x) for x in rev4]
        growth = revenue_growth_pct_cagr([rq1, rq2, rq3, rq4])

        peg_direct = to_num(fi.get("peg"))
        peg = None; peg_method = "—"
        if pd.notna(peg_direct):
            peg        = float(peg_direct)
            peg_method = fi.get("peg_src") or "Yahoo"
        else:
            pe_for_peg = fwd if pd.notna(fwd) else pe
            eps_g      = fi.get("eps_growth")
            g_src      = fi.get("growth_src") or ""
            if eps_g is not None:
                eg = float(eps_g)
                if eg >= MIN_GROWTH_PCT_FOR_PEG and pd.notna(pe_for_peg):
                    peg        = float(pe_for_peg) / eg
                    peg_method = "{} EPS growth".format(g_src)
        if peg is not None and (peg <= 0 or peg > 500):
            peg = None

        # ── v16: Piotroski + Sloan ────────────────────────────────────────
        # Inject rev_growth_pct so O2 test works
        fi_with_growth = dict(fi)
        fi_with_growth["rev_growth_pct"] = float(growth) if growth is not None else None

        piotroski_f, _ = compute_piotroski_fscore(fi_with_growth)

        sloan_ratio = compute_sloan_ratio(
            fi.get("net_income_ttm"),
            fi.get("ocf_ttm"),
            fi.get("total_assets_now"),
            fi.get("total_assets_prev"),
        )

        # ── v16: FCF Yield ────────────────────────────────────────────────
        fcf_ttm = fi.get("fcf_ttm")
        fcf_yield = None
        if fcf_ttm is not None and pd.notna(mc) and float(mc) > 0:
            fcf_yield = float(fcf_ttm) / float(mc) * 100.0

        # ── v16: Quality Score (7 sub-signals) ───────────────────────────
        gross_margin_now  = fi.get("gross_margin_now")
        gross_margin_prev = fi.get("gross_margin_prev")
        fcf_ni_ratio = None
        ni_val = fi.get("net_income_ttm")
        if fcf_ttm is not None and ni_val is not None and float(ni_val) != 0:
            fcf_ni_ratio = float(fcf_ttm) / float(ni_val)

        q_score = compute_quality_score_elite(
            float(roic)   if pd.notna(roic) else None,
            float(roe)    if pd.notna(roe)  else None,
            float(ic)     if pd.notna(ic)   else None,
            float(om)     if pd.notna(om)   else None,
            gross_margin_now  = (float(gross_margin_now)
                                 if gross_margin_now is not None else None),
            gross_margin_prev = (float(gross_margin_prev)
                                 if gross_margin_prev is not None else None),
            fcf_ni_ratio = fcf_ni_ratio,
            piotroski_f  = piotroski_f,
            sloan_ratio  = sloan_ratio,
            sector       = sec,
        )

        rows.append({
            "Ticker":             t,
            "Sector":             sec,
            "Price":              price,
            "Mkt Cap":            mc,
            "P/E":                pe,
            "Fwd P/E":            fwd,
            "PEG":                to_num(peg),
            "PEG Method":         peg_method,
            "Earn Traj":          earn_traj,
            "52W Pos%":           to_num(pos52),
            "ROIC%":              roic,
            "ROE%":               roe,
            "Int Coverage":       ic,
            "Op Margin%":         om,
            "Debt/Eq":            de,
            "Quality Score":      to_num(q_score),
            "Momentum Score":     mom_score,
            "Ret 1Mo%":           ret_1mo,
            "Ret 3Mo%":           ret_3mo,
            "Ret 6Mo%":           ret_6mo,
            "Trailing Vol%":      t_vol,
            "Eligible":           True,
            "Rev Q1 Oldest ($B)": rq1,
            "Rev Q2 ($B)":        rq2,
            "Rev Q3 ($B)":        rq3,
            "Rev Q4 Latest ($B)": rq4,
            "Rev Growth% (CAGR)": to_num(growth),
            # v16 new columns
            "EV/EBITDA":   to_num(fi.get("ev_ebitda")),
            "FCF Yield%":  to_num(fcf_yield),
            "EV/Sales":    to_num(fi.get("ev_sales")),
            "Div Yield%":  to_num(fi.get("div_yield")),
            "Piotroski F": to_num(piotroski_f),
            "Sloan Ratio": to_num(sloan_ratio),
        })

    scr = pd.DataFrame(rows)
    if scr.empty:
        return scr

    total_sp500_mc = scr["Mkt Cap"].sum()
    scr["MC% of S&P500"] = (
        (scr["Mkt Cap"] / total_sp500_mc * 100.0) if total_sp500_mc > 0 else None
    )

    num_cols = [
        "Price", "Mkt Cap", "P/E", "Fwd P/E", "PEG", "52W Pos%",
        "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
        "Quality Score", "Earn Traj", "Momentum Score",
        "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
        "MC% of S&P500",
        "Rev Q1 Oldest ($B)", "Rev Q2 ($B)", "Rev Q3 ($B)", "Rev Q4 Latest ($B)",
        "Rev Growth% (CAGR)",
        "EV/EBITDA", "FCF Yield%", "EV/Sales", "Div Yield%",
        "Piotroski F", "Sloan Ratio",
    ]
    for c in num_cols:
        if c in scr.columns:
            scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns:
        scr["Rank"] = pd.NA

    sector_med_pe = scr.groupby("Sector")["P/E"].transform("median")
    scr["P/E vs Sector Med"] = (scr["P/E"] / sector_med_pe).round(2)

    scr = compute_conviction_scores(scr)
    return scr


# ── Reference Guide ────────────────────────────────────────────────────────────
def render_reference_guide():
    st.markdown("## Column Reference Guide")
    st.caption(
        "Every metric explained with formula, real-world numeric example, "
        "sector benchmarks, and how it is used in scoring."
    )

    tab_val, tab_qual, tab_peg, tab_etraj, tab_mom, tab_rank, tab_disp = st.tabs([
        "Valuation", "Quality", "PEG", "Earn Trajectory",
        "Momentum", "Ranking & Score", "Display-Only",
    ])

    with tab_val:
        st.markdown("""
**Valuation Composite (v16) — FCF Yield + EV/EBITDA + P/E blended**

The valuation sub-score is now a weighted blend of three signals:

| Signal | Weight | Lower/Higher = Better | Why |
|---|---|---|---|
| FCF Yield% | 40% | Higher = better | Pure cash — no accounting distortion |
| EV/EBITDA | 35% | Lower = cheaper | Capital-structure neutral — removes debt distortion |
| Fwd P/E (or P/E) | 25% | Lower = cheaper | Consensus expectation anchor |

Weights self-normalise when signals are missing. If only P/E is available,
it carries 100% of the valuation score.

---
**FCF Yield%** = (OCF − CapEx) / Market Cap × 100

**EV/EBITDA** = Enterprise Value / EBITDA TTM. Negative EBITDA → None.

**EV/Sales** = Enterprise Value / Revenue TTM (display only — for negative-EBITDA growth stocks).

**Div Yield%** = Annual Dividend / Price × 100 (display only).

---
**P/E vs Sector Med** = Stock P/E / Median P/E of sector. Display only.

**52W Pos%** = (Price − 52W Low) / (52W High − 52W Low) × 100.
        """)

    with tab_qual:
        st.markdown("""
**Quality Score (0–100) — v16: 7 sub-signals**

| Sub-Signal | Weight | Formula | Why |
|---|---|---|---|
| Profitability (ROIC or ROE) | 25% | log1p(x)/log1p(30)×100 | Core capital efficiency |
| Interest Coverage | 15% | EBIT TTM / Interest TTM | Financial safety margin |
| Operating Margin (non-Fin) | 15% | Op Income / Revenue × 100 | Operational efficiency |
| Gross Margin + YoY trend | 20% | Gross Profit / Revenue × 100 | Moat / pricing power proxy |
| Piotroski F-Score | 15% | 9-point binary score / 9 × 100 | Holistic financial health |
| Sloan Ratio (accruals) | 10% | (NI − OCF) / Avg Assets | Earnings quality / manipulation risk |

---
**Piotroski F-Score (0–9)**

9 binary tests across profitability, leverage/liquidity, and efficiency:
- P1 ROA > 0 · P2 OCF > 0 · P3 ROA improving · P4 OCF > Net Income
- L1 Lower LT debt ratio · L2 Higher current ratio · L3 No dilution
- O1 Gross margin improving · O2 Asset turnover improving

Score 8–9 = strong · 5–7 = average · 0–2 = distressed

---
**Sloan Ratio** = (Net Income TTM − OCF TTM) / Average Total Assets

| Range | Quality Signal |
|---|---|
| > +0.05 | **HighAccruals** flag — earnings manipulation risk |
|  0 to +0.05 | Mild accruals — acceptable |
| -0.10 to 0 | Good — OCF backs earnings |
| < -0.10 | Excellent — OCF substantially exceeds reported income |

Academic source: Sloan (1996) — high accrual stocks underperform by 5–8%/yr.

---
**Gross Margin** = Gross Profit TTM / Revenue TTM × 100

> 40% = strong pricing power / moat (Buffett proxy)
Improving YoY adds a +10% bonus to the sub-score (capped at 100).

---
**Quality Flag** — Flags: `ROIC<8%` · `ROE<8%` · `IntCov<3x` · `Margin<5%` · `HighAccruals` · `Pass`
        """)

    with tab_peg:
        st.markdown("""
**PEG — Price/Earnings-to-Growth Ratio**

Formula: `P/E Ratio / Annual EPS Growth Rate (%)`

| Stock | P/E | EPS Growth | PEG | Verdict |
|---|---|---|---|---|
| Nvidia | 35 | 40%/yr | 0.88 | Potentially undervalued |
| Alphabet | 22 | 20%/yr | 1.10 | Growth well-priced in |
| Tesla | 60 | 25%/yr | 2.40 | Expensive vs growth rate |
| P&G | 26 | 4%/yr | N/A | Below 5% growth floor |

Growth guard: PEG only computed when EPS growth ≥ 5%.
        """)

    with tab_etraj:
        st.markdown("""
**Earn Traj** = (Forward EPS − Trailing EPS) / |Trailing EPS|, clipped [−1, +1].

Both-negative cap at +0.30.

| Scenario | Earn Traj |
|---|---|
| Trail +$2.00 → Fwd +$2.50 | +0.25 |
| Trail −$2.00 → Fwd +$1.00 | +1.0 |
| Trail −$2.00 → Fwd −$0.50 | +0.30 capped |
| Trail +$2.00 → Fwd +$1.50 | −0.25 |
        """)

    with tab_mom:
        st.markdown("""
**Momentum Score** = (6Mo return − 1Mo return) / Trailing 90-day Ann. Vol

| Score | Signal |
|---|---|
| > +1.0 | Exceptional momentum |
| +0.3 to +1.0 | Healthy uptrend |
| −0.3 to +0.3 | Neutral |
| < −0.3 | Downtrend |
        """)

    with tab_rank:
        st.markdown("""
**Score (0–100) — v15 MAD Z-Score + v16 Valuation Composite**

Normalisation pipeline (v15, all 4 factor sub-scores):
1. Winsorise at 1st/99th pct · 2. MAD z-score · 3. Clip [−3,+3] · 4. Rescale [0,100]

Valuation sub-score (v16): weighted blend of FCF Yield, EV/EBITDA, P/E.
Quality sub-score: min-max rescale on 7-signal composite.

| Sector | Val | Quality | PEG | Earn | Mom |
|---|---|---|---|---|---|
| Information Technology | 20% | 25% | 25% | 15% | 15% |
| Consumer Staples | 28% | 32% | 10% | 15% | 15% |
| Financials | 30% | 25% | 18% | 17% | 10% |
| Energy | 30% | 18% | 12% | 15% | 25% |
| Utilities | 38% | 27% | 5% | 15% | 15% |

**Missing Factor Penalty**

| Missing | Multiplier |
|---|---|
| 0 | ×1.00 |
| 1 | ×0.95 |
| 2 | ×0.85 |
| 3+ | ×0.70 |
        """)

    with tab_disp:
        st.markdown("""
**v16 New Columns**

| Column | Source | Notes |
|---|---|---|
| EV/EBITDA | Yahoo info | Capital-structure-neutral valuation |
| FCF Yield% | Phase 2 cashflow | (OCF−CapEx)/MktCap×100 |
| EV/Sales | Yahoo info | For negative-EBITDA growth stocks |
| Div Yield% | Yahoo info | Annual dividend / price |
| Piotroski F | Phase 2 balance sheet + P&L | 0=distressed, 9=excellent |
| Sloan Ratio | Phase 2 P&L + cashflow | Negative = better earnings quality |

**Data Coverage (typical)**

| Metric | Coverage |
|---|---|
| Price, 52W | ~99% |
| Trailing P/E | ~90% |
| Forward P/E | ~78% |
| PEG | ~70% |
| ROE, Op Margin | ~88% |
| Int Coverage | ~65% |
| ROIC | ~60% |
| EV/EBITDA | ~75% |
| FCF Yield | ~65% |
| Piotroski F | ~55% |
| Sloan Ratio | ~55% |
| Earn Traj | ~82% |
| Momentum | ~97% |
        """)

    st.markdown("---")
    st.markdown(
        "**v16:** Yahoo + FMP · MAD Z-Score normalisation (v15) · "
        "Valuation composite: FCF Yield 40% + EV/EBITDA 35% + PE 25% · "
        "Quality: 7 signals incl. Piotroski F-Score + Sloan Accruals + Gross Margin · "
        "_Nothing here is financial advice._"
    )


# ══════════════════════════════════════════════════════════════════════════════
# APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="S&P 500 Screener v16", layout="wide", page_icon="📊"
)
st.markdown(
    "<style>"
    "div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}"
    "</style>",
    unsafe_allow_html=True,
)
st.markdown("## S&P 500 Fundamental Screener v16")

page_screener, page_reference = st.tabs(["Screener", "Column Reference Guide"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════
with page_screener:
    col_r, col_t = st.columns([1, 6])
    with col_r:
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()
    with col_t:
        st.caption(
            "Last loaded: {} · 1hr price cache · 24hr fundamental cache · "
            "v16: MAD scoring · Piotroski · Sloan · FCF · EV/EBITDA".format(
                datetime.now().strftime("%I:%M %p")
            )
        )

    fmp_key = get_fmp_key()
    if fmp_key:
        st.success("FMP API key found — bonus layer active.")
    else:
        st.info("No FMP key. Yahoo Finance only. Add [fmp] api_key to Streamlit Secrets.")

    with st.spinner("Loading S&P 500 universe..."):
        sp500 = fetch_sp500_constituents()
    if sp500.empty:
        st.error("Failed to load S&P 500 universe.")
        st.stop()

    universe_df = sp500.copy().reset_index(drop=True)
    tickers     = tuple(universe_df["Ticker"].tolist())
    today_date  = date.today()

    st.markdown("### Filters")
    all_sectors = sorted(universe_df["Sector"].dropna().unique().tolist())
    f1, f2, f3, f4, f5 = st.columns(5)

    sector_sel = f1.selectbox("Sector", ["All Sectors"] + all_sectors)
    sort_by    = f2.selectbox("Sort by", [
        "Sector then Rank", "Score high to low", "Conviction high to low",
        "MC% of S&P500 high to low", "Price low to high", "Price high to low",
        "Mkt Cap high to low", "PE low to high", "Fwd PE low to high",
        "PEG low to high", "Quality Score high", "ROIC high to low",
        "ROE high to low", "Earn Traj high to low", "Rev Growth high to low",
        "Momentum Score high", "52W Pos low to high",
        "P/E vs Sector Med low to high",
        "Piotroski F high", "FCF Yield high", "EV/EBITDA low",
    ])
    mc_min_b   = f3.number_input("Min Mkt Cap ($B)", value=0, step=10, min_value=0)
    pe_max     = f4.number_input("Max P/E", value=9999, step=50, min_value=0)
    qual_min_f = f5.number_input("Min Quality Score", value=0.0, step=5.0,
                                  min_value=0.0, max_value=100.0)

    with st.spinner("Phase 1/2 — Yahoo info ({} tickers)...".format(len(tickers))):
        yahoo_info = fetch_yahoo_info_all(tickers, _cache_date=today_date)

    filtered_tickers = _pre_filter_tickers(yahoo_info, universe_df, mc_min_b, pe_max)

    with st.spinner("Phase 2/2 — Deep financials ({} tickers)...".format(
            len(filtered_tickers))):
        yahoo_deep = fetch_yahoo_deep_financials(
            tuple(filtered_tickers), _cache_date=today_date)

    yahoo_fundamentals = merge_yahoo_phases(yahoo_info, yahoo_deep, tickers)

    with st.spinner("Prices + Momentum ({} tickers)...".format(len(tickers))):
        pm_data = fetch_price_momentum_all(tickers)

    fmp_quotes = {}; fmp_ratios = {}
    if fmp_key:
        with st.spinner("FMP /quote..."):
            fmp_quotes = fetch_fmp_quotes_if_available(tickers, fmp_key)
        with st.spinner("FMP /ratios-ttm..."):
            fmp_ratios = fetch_fmp_ratios_if_available(tickers, fmp_key)

    with st.spinner("Merging sources..."):
        merged_map = merge_all_sources(yahoo_fundamentals, fmp_quotes, fmp_ratios, tickers)

    # Coverage stats
    total_t  = len(tickers)
    def cov(key, src="merged"):
        if src == "merged":
            return sum(1 for t in tickers
                       if merged_map.get(t, {}).get(key) is not None)
        return sum(1 for t in tickers
                   if pm_data.get(t, {}).get(key) is not None)

    st.info(
        "Coverage — "
        "P/E: {}/{} ({:.0f}%) · Fwd P/E: {}/{} ({:.0f}%) · "
        "PEG: {}/{} ({:.0f}%) · ROIC: {}/{} ({:.0f}%) · "
        "EV/EBITDA: {}/{} ({:.0f}%) · FCF: {}/{} ({:.0f}%) · "
        "Piotroski: display after build · "
        "Momentum: {}/{} ({:.0f}%) · "
        "Yahoo{}".format(
            cov("pe"),      total_t, cov("pe")      / total_t * 100,
            cov("fwd_pe"),  total_t, cov("fwd_pe")  / total_t * 100,
            cov("peg"),     total_t, cov("peg")      / total_t * 100,
            cov("roic"),    total_t, cov("roic")     / total_t * 100,
            cov("ev_ebitda"),total_t,cov("ev_ebitda")/ total_t * 100,
            cov("fcf_ttm"), total_t, cov("fcf_ttm")  / total_t * 100,
            cov("momentum_score", "pm"), total_t,
            cov("momentum_score", "pm") / total_t * 100,
            " + FMP" if fmp_key else "",
        )
    )

    scr = build_screener_table(universe_df, pm_data, merged_map)

    # Filters
    filt = scr.copy()
    if sector_sel != "All Sectors":
        filt = filt[filt["Sector"] == sector_sel]
    filt = filt[(filt["Mkt Cap"].isna())       | (filt["Mkt Cap"]       >= mc_min_b * 1e9)]
    filt = filt[(filt["P/E"].isna())           | (filt["P/E"]           <= pe_max)]
    filt = filt[(filt["Quality Score"].isna()) | (filt["Quality Score"] >= qual_min_f)]

    sort_map = {
        "Sector then Rank":              (["Sector", "Rank"],      [True,  True]),
        "Score high to low":             (["Score"],               [False]),
        "Conviction high to low":        (["Conviction Score"],    [False]),
        "MC% of S&P500 high to low":     (["MC% of S&P500"],      [False]),
        "Price low to high":             (["Price"],               [True]),
        "Price high to low":             (["Price"],               [False]),
        "Mkt Cap high to low":           (["Mkt Cap"],             [False]),
        "PE low to high":                (["P/E"],                 [True]),
        "Fwd PE low to high":            (["Fwd P/E"],             [True]),
        "PEG low to high":               (["PEG"],                 [True]),
        "Quality Score high":            (["Quality Score"],       [False]),
        "ROIC high to low":              (["ROIC%"],               [False]),
        "ROE high to low":               (["ROE%"],                [False]),
        "Earn Traj high to low":         (["Earn Traj"],           [False]),
        "Rev Growth high to low":        (["Rev Growth% (CAGR)"],  [False]),
        "Momentum Score high":           (["Momentum Score"],      [False]),
        "52W Pos low to high":           (["52W Pos%"],            [True]),
        "P/E vs Sector Med low to high": (["P/E vs Sector Med"],   [True]),
        "Piotroski F high":              (["Piotroski F"],         [False]),
        "FCF Yield high":                (["FCF Yield%"],          [False]),
        "EV/EBITDA low":                 (["EV/EBITDA"],           [True]),
    }
    sc, sa = sort_map.get(sort_by, (["Sector", "Rank"], [True, True]))
    filt   = filt.sort_values(sc, ascending=sa, na_position="last")

    st.caption("Showing **{}** of **{}** · Sector: {} · Sort: {}".format(
        len(filt), len(scr), sector_sel, sort_by))

    # Display
    disp = filt.copy()
    disp["Price ($)"]          = disp["Price"].round(2)
    disp["Mkt Cap ($B)"]       = (disp["Mkt Cap"] / 1e9).round(2)
    disp["MC% of S&P500"]      = disp["MC% of S&P500"].round(4)
    disp["Rev Q1 Oldest ($B)"] = (disp["Rev Q1 Oldest ($B)"] / 1e9).round(2)
    disp["Rev Q2 ($B)"]        = (disp["Rev Q2 ($B)"]         / 1e9).round(2)
    disp["Rev Q3 ($B)"]        = (disp["Rev Q3 ($B)"]         / 1e9).round(2)
    disp["Rev Q4 Latest ($B)"] = (disp["Rev Q4 Latest ($B)"]  / 1e9).round(2)

    disp["Quality Flag"] = disp.apply(
        lambda r: quality_flag(
            r.get("ROIC%"), r.get("ROE%"), r.get("Int Coverage"),
            r.get("Op Margin%"),
            sloan_ratio=r.get("Sloan Ratio"),
            sector=r.get("Sector"),
        ), axis=1,
    )

    for c in [
        "P/E", "Fwd P/E", "PEG", "Earn Traj", "52W Pos%",
        "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
        "Quality Score", "Momentum Score", "Ret 1Mo%", "Ret 3Mo%",
        "Ret 6Mo%", "Trailing Vol%", "Score", "Conviction Score",
        "Rev Growth% (CAGR)", "P/E vs Sector Med",
        "EV/EBITDA", "FCF Yield%", "EV/Sales", "Div Yield%", "Sloan Ratio",
    ]:
        if c in disp.columns:
            disp[c] = disp[c].round(2)

    disp["Rank"] = disp["Rank"].apply(
        lambda v: int(v) if pd.notna(v) else pd.NA)

    COLS = [
        "Ticker", "Sector", "Price ($)", "Mkt Cap ($B)", "MC% of S&P500",
        "P/E", "P/E vs Sector Med", "Fwd P/E",
        "EV/EBITDA", "FCF Yield%", "EV/Sales", "Div Yield%",
        "PEG", "PEG Method", "Earn Traj",
        "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
        "Quality Score", "Quality Flag",
        "Piotroski F", "Sloan Ratio",
        "Momentum Score", "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
        "52W Pos%", "Score", "Conviction Score", "Rank",
        "Rev Q1 Oldest ($B)", "Rev Q2 ($B)", "Rev Q3 ($B)", "Rev Q4 Latest ($B)",
        "Rev Growth% (CAGR)",
    ]
    disp_final = disp[[c for c in COLS if c in disp.columns]].copy()
    st.dataframe(disp_final, use_container_width=True, height=680)

    st.download_button(
        label="Download CSV",
        data=disp_final.to_csv(index=False).encode("utf-8"),
        file_name="sp500_screener_v16_{}.csv".format(
            datetime.now().strftime("%Y%m%d_%H%M")),
        mime="text/csv",
    )

with page_reference:
    render_reference_guide()
