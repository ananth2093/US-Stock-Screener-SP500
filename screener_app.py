# screener_app.py v17
# ─────────────────────────────────────────────────────────────────────────────
# v17 CHANGES from v16:
#
#  GAP 5 — Earnings Surprise Trend (new)
#    • extract_earnings_surprise_trend(obj) — reads obj.earnings_history,
#      computes surprise% per quarter, returns:
#        eps_surprise_avg   : mean surprise % over last 4Q
#        eps_beat_rate      : fraction of quarters where actuals beat (0–1)
#        eps_surprise_trend : +1.0 if recent 2Q avg > earlier 2Q avg, else -1.0
#    • Called inside _fetch_yahoo_info_one(); results stored in merged dict
#      and passed through to screener rows.
#    • Display columns: EPS Surp Avg%, EPS Beat Rate, EPS Surp Trend
#
#  GAP 6 — Analyst Revision Momentum (new)
#    • extract_revision_momentum(obj) — reads obj.recommendations_summary,
#      compares latest period vs prior period buy/sell counts, returns a
#      score in [-1.0, +1.0]. +1.0 = strong upgrade trend.
#    • Called inside _fetch_yahoo_info_one(); result stored as
#      revision_momentum in merged dict.
#    • Display column: Revision Mom
#
#  GAP 7 — 4-Signal Momentum Composite (upgrade)
#    • compute_elite_momentum(closes, price, hi52, spy_3mo) replaces the
#      single skip-month score:
#        S1  Skip-month vol-adj momentum     weight 40%
#        S2  52W-high proximity              weight 25%
#        S3  Price vs 200d MA (50d fallback) weight 20%
#        S4  Relative strength vs SPY 3Mo    weight 15%
#      Weights self-normalise when SPY data is unavailable.
#    • fetch_spy_3mo_return() — cached 1hr, fetches SPY once before the
#      ticker loop and passes to each worker via a closure.
#    • _fetch_price_momentum_one() now accepts spy_3mo kwarg, calls
#      compute_elite_momentum(), returns composite_momentum alongside all
#      individual sub-scores (skip_month_raw, hi52_proximity, vs_ma200,
#      rel_strength_spy).
#    • build_screener_table() uses composite_momentum as Momentum Score.
#    • Display columns: Momentum Score (composite), Skip Mo, 52W Prox,
#      vs MA200, Rel Str SPY
#
#  GAP 8 — Elite Conviction Score (architecture fix)
#    • compute_conviction_scores_elite() replaces compute_conviction_scores().
#      Three independent dimensions:
#        D1  Data completeness     (0.5–1.0 multiplier)
#        D2  Signal agreement      (0.7–1.0 multiplier) — cross-signal
#            consistency measured by std-dev of valuation/momentum/earn_traj
#        D3  Anomaly multiplier    (0.70–1.00) — Piotroski ≤2 → ×0.70,
#            Sloan >0.05 → ×0.85
#      Old logic (Score × completeness × sector_PE_discount) removed.
#
#  GAP 9 — Cross-Sectional Score (new)
#    • compute_cross_sectional_scores() — scores all ~500 stocks vs the
#      full S&P 500 universe (not just within sector).
#      Weights: Val 25%, Quality 25%, PEG 20%, EarnTraj 15%, Momentum 15%.
#      Output column: CS Score (0–100).
#
#  GAP 10 — Score History Tracking (new)
#    • Session-state dict score_history stores up to 10 snapshots per ticker
#      (score, conviction, rank, piotroski_f, momentum_score, timestamp).
#    • get_score_delta() returns Score change vs last snapshot.
#    • Display column: Score Δ (None on first load; numeric on refresh).
#
#  KEPT INTACT from v16
#    • winsorise / mad_zscore / elite_factor_score (v15)
#    • Piotroski F-Score, Sloan Ratio, Gross Margin, FCF Yield,
#      EV/EBITDA, compute_valuation_subscore (v16)
#    • All data-fetch concurrency, FMP integration, ROIC computation
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

ROE_PRIMARY_SECTORS      = {"Financials"}
QUALITY_THRESHOLDS       = {"roic_min": 8.0, "int_coverage_min": 3.0, "op_margin_min": 5.0}
OPERATING_CASH_PCT_OF_REV = 0.02


# ── Credentials ────────────────────────────────────────────────────────────────
def get_fmp_key():
    try:
        k = st.secrets["fmp"]["api_key"]
        return k if k and k.strip() and k != "YOUR_KEY_HERE" else None
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — v15 normalisation pipeline (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def sf(val):
    try:
        return float(val) if val is not None else None
    except Exception:
        return None

def normalise_pct(val):
    if val is None: return None
    v = float(val)
    return v * 100.0 if abs(v) < 5.0 else v

def normalise_pct_fmp(val):
    if val is None: return None
    return float(val) * 100.0

def fmt_mc(val):
    if pd.isna(val) or val == 0: return "N/A"
    if val >= 1e12: return "${:.2f}T".format(val / 1e12)
    if val >= 1e9:  return "${:.1f}B".format(val / 1e9)
    return "${:.0f}M".format(val / 1e6)

def winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    valid = series.dropna()
    if valid.empty: return series.copy()
    return series.clip(lower=valid.quantile(lower), upper=valid.quantile(upper))

def mad_zscore(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty: return pd.Series(0.0, index=series.index)
    med = valid.median()
    mad = (valid - med).abs().median()
    if mad == 0: return pd.Series(0.0, index=series.index)
    return (series - med) / (1.4826 * mad)

def elite_factor_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    if series.dropna().empty: return pd.Series(0.0, index=series.index)
    ws = winsorise(series.copy())
    if ascending: ws = -ws
    z = mad_zscore(ws).clip(-3.0, 3.0)
    z_min, z_max = z.min(), z.max()
    scaled = ((z - z_min) / (z_max - z_min) * 100.0
              if z_max > z_min else pd.Series(50.0, index=series.index))
    return scaled.fillna(0.0)

def percentile_score(series: pd.Series, ascending: bool = True) -> pd.Series:
    """DEPRECATED v15."""
    result = pd.Series(index=series.index, dtype=float)
    valid  = series.notna()
    if valid.sum() == 0: return result.fillna(0.0)
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
        if rev4 is None or len(rev4) != 4: return None
        q1, _, _, q4 = rev4
        if q1 is None or q4 is None: return None
        q1, q4 = float(q1), float(q4)
        if q1 <= 0 or q4 <= 0: return None
        return ((q4 / q1) ** (1 / 3) - 1) * 100.0
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# GAP 5 — Earnings Surprise Trend
# ══════════════════════════════════════════════════════════════════════════════
def extract_earnings_surprise_trend(obj):
    """
    Returns (avg_surprise_pct, beat_rate, trend) using obj.earnings_history.

    avg_surprise_pct : mean (Actual − Estimate) / |Estimate| × 100 over last 4Q
    beat_rate        : fraction of quarters with positive surprise (0.0–1.0)
    trend            : +1.0 if recent-2Q avg > earlier-2Q avg, else -1.0, else None

    Returns (None, None, None) on any failure or missing data.
    """
    try:
        hist = obj.earnings_history
        if hist is None or hist.empty:
            return None, None, None
        hist = hist.dropna(subset=["epsActual", "epsEstimate"])
        if len(hist) == 0:
            return None, None, None
        hist = hist.copy()
        hist["surprise_pct"] = (
            (hist["epsActual"] - hist["epsEstimate"])
            / hist["epsEstimate"].abs() * 100.0
        )
        hist = hist.tail(4)
        avg_surprise = float(hist["surprise_pct"].mean())
        beat_rate    = float((hist["surprise_pct"] > 0).sum() / len(hist))
        trend = None
        if len(hist) >= 3:
            recent  = float(hist["surprise_pct"].tail(2).mean())
            earlier = float(hist["surprise_pct"].head(2).mean())
            trend   = 1.0 if recent > earlier else -1.0
        return avg_surprise, beat_rate, trend
    except Exception:
        return None, None, None


# ══════════════════════════════════════════════════════════════════════════════
# GAP 6 — Analyst Revision Momentum
# ══════════════════════════════════════════════════════════════════════════════
def extract_revision_momentum(obj):
    """
    Uses obj.recommendations_summary to measure analyst upgrade/downgrade trend.

    Compares latest period vs prior period buy/sell counts.
    Returns float in [-1.0, +1.0]:
        +1.0 = strong upgrade trend
        -1.0 = strong downgrade trend
         0.0 = neutral / no change
        None  = data unavailable
    """
    try:
        rec = obj.recommendations_summary
        if rec is None or rec.empty or len(rec) < 2:
            return None
        latest = rec.iloc[0]
        prior  = rec.iloc[1]
        sb_chg = (
            float(latest.get("strongBuy", 0) or 0) +
            float(latest.get("buy",       0) or 0) -
            float(prior.get("strongBuy",  0) or 0) -
            float(prior.get("buy",        0) or 0)
        )
        sell_chg = (
            float(latest.get("sell",       0) or 0) +
            float(latest.get("strongSell", 0) or 0) -
            float(prior.get("sell",        0) or 0) -
            float(prior.get("strongSell",  0) or 0)
        )
        total = abs(sb_chg) + abs(sell_chg)
        if total == 0:
            return 0.0
        return float(np.clip((sb_chg - sell_chg) / total, -1.0, 1.0))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# GAP 7 — 4-Signal Elite Momentum
# ══════════════════════════════════════════════════════════════════════════════
def compute_elite_momentum(closes: pd.Series, price: float,
                           hi52: float, spy_3mo: float = None) -> tuple:
    """
    Four-signal momentum composite.  Returns (composite_score, components_dict).

    Signals (weights self-normalise for missing data):
      S1  Skip-month vol-adjusted momentum     40%
      S2  52W-high proximity                   25%
      S3  Price vs 200d MA (50d fallback)       20%
      S4  Relative strength vs SPY 3Mo          15%

    composite_score : float in [-1, +1], or None if no signals available
    components_dict : individual raw signal values for display
    """
    monthly = closes.resample("ME").last().dropna()
    components = {
        "skip_month_raw":  None,
        "hi52_proximity":  None,
        "vs_ma200":        None,
        "rel_strength_spy": None,
    }

    if len(monthly) < 2 or price is None or price <= 0:
        return None, components

    px_now = float(monthly.iloc[-1])
    if px_now <= 0:
        return None, components

    def ret_mo(n):
        idx = -(n + 1)
        if abs(idx) > len(monthly): return None
        px = float(monthly.iloc[idx])
        return (px_now / px - 1) * 100.0 if px > 0 else None

    r1, r3, r6 = ret_mo(1), ret_mo(3), ret_mo(6)

    # S1: Skip-month
    s1 = None
    daily_rets = closes.pct_change().dropna().tail(90)
    t_vol = (float(daily_rets.std() * np.sqrt(252) * 100.0)
             if len(daily_rets) >= 15 else None)
    if r6 is not None and r1 is not None:
        skip_raw = r6 - r1
        raw_score = skip_raw / t_vol if (t_vol and t_vol > 0) else skip_raw
        s1 = float(np.clip(raw_score / 2.0, -1.0, 1.0))
    components["skip_month_raw"] = s1

    # S2: 52W-high proximity
    s2 = None
    if hi52 and hi52 > 0:
        pct_from_high = (price - hi52) / hi52  # always <= 0
        s2 = float(np.clip(1.0 + pct_from_high / 0.30, 0.0, 1.0))
    components["hi52_proximity"] = s2

    # S3: Price vs 200d MA (50d fallback)
    s3 = None
    if len(closes) >= 200:
        ma = float(closes.tail(200).mean())
        if ma > 0:
            s3 = float(np.clip((price - ma) / ma / 0.30, -1.0, 1.0))
    elif len(closes) >= 50:
        ma = float(closes.tail(50).mean())
        if ma > 0:
            s3 = float(np.clip((price - ma) / ma / 0.20, -1.0, 1.0))
    components["vs_ma200"] = s3

    # S4: Relative strength vs SPY
    s4 = None
    if spy_3mo is not None and r3 is not None:
        rs = r3 - spy_3mo
        s4 = float(np.clip(rs / 20.0, -1.0, 1.0))
    components["rel_strength_spy"] = s4

    # Weighted blend
    signal_weights = [
        ("skip_month_raw",   0.40),
        ("hi52_proximity",   0.25),
        ("vs_ma200",         0.20),
        ("rel_strength_spy", 0.15),
    ]
    total_w   = 0.0
    composite = 0.0
    for key, w in signal_weights:
        val = components.get(key)
        if val is not None:
            composite += val * w
            total_w   += w

    if total_w == 0:
        return None, components

    return float(composite / total_w), components


# ── SPY 3-month return (cached, fetched once per session/hour) ────────────────
@st.cache_data(ttl=3600)
def fetch_spy_3mo_return():
    """
    Fetch SPY 12-month daily history, return 3-month price return as float %.
    Returns None on any failure — signal S4 is simply excluded when None.
    """
    try:
        hist = yf.Ticker("SPY").history(period="12mo", interval="1d",
                                         auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        closes  = hist["Close"].dropna()
        monthly = closes.resample("ME").last().dropna()
        if len(monthly) < 4:
            return None
        px_now  = float(monthly.iloc[-1])
        px_3m   = float(monthly.iloc[-4])
        if px_3m <= 0:
            return None
        return (px_now / px_3m - 1) * 100.0
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# GAP 8 — Elite Conviction Score
# ══════════════════════════════════════════════════════════════════════════════
def compute_conviction_scores_elite(scr: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces compute_conviction_scores().

    Conviction measures HOW CONFIDENT we are in each Score — not the Score itself.

    Three independent dimensions (all multiplicative on Score):

    D1  Data completeness     → multiplier 0.5 + 0.5 × completeness_ratio
        If all 6 key factors present: ×1.0. If none: ×0.5.

    D2  Signal agreement      → multiplier 0.7 + 0.3 × agreement
        agreement = 1 − std(sub_scores) / 1.5
        sub_scores: valuation signal (PE vs sector median), momentum, earn_traj
        Low std = signals point the same direction = high confidence.

    D3  Anomaly multiplier    → 0.70–1.00
        Piotroski F ≤ 2     → ×0.70 (financial distress)
        Sloan Ratio > 0.05  → ×0.85 (earnings manipulation risk)
        Both flags          → both multipliers apply (×0.595)

    Final: normalise raw_conv to [0, 100] across universe.
    """
    scr = scr.copy()
    KEY_FACTORS = ["P/E", "Fwd P/E", "PEG", "Quality Score",
                   "Momentum Score", "Earn Traj"]
    n_factors = len(KEY_FACTORS)

    # D1: Data completeness
    def completeness(row):
        return sum(1 for c in KEY_FACTORS
                   if c in row.index and pd.notna(row[c])) / n_factors
    scr["_completeness"] = scr.apply(completeness, axis=1)

    # Pre-compute sector median P/E for D2 valuation signal
    sector_med_pe_map    = scr.groupby("Sector")["P/E"].median().to_dict()
    scr["_sector_med_pe"] = scr["Sector"].map(sector_med_pe_map)

    # D2: Signal agreement
    def signal_agreement(row):
        sub = []
        pe  = row.get("P/E")
        med = row.get("_sector_med_pe", 25)
        if pd.notna(pe) and med and med > 0:
            sub.append(1.0 if pe < med * 0.9 else
                       (-1.0 if pe > med * 1.1 else 0.0))
        mom = row.get("Momentum Score")
        if pd.notna(mom):
            sub.append(float(np.clip(float(mom), -1.0, 1.0)))
        et = row.get("Earn Traj")
        if pd.notna(et):
            sub.append(float(et))
        if len(sub) < 2:
            return 0.5
        return float(np.clip(1.0 - float(np.std(sub)) / 1.5, 0.0, 1.0))
    scr["_signal_agreement"] = scr.apply(signal_agreement, axis=1)

    # D3: Anomaly multiplier
    def anomaly_multiplier(row):
        mult = 1.0
        pf   = row.get("Piotroski F")
        if pd.notna(pf) and float(pf) <= 2:
            mult *= 0.70
        sl   = row.get("Sloan Ratio")
        if pd.notna(sl) and float(sl) > 0.05:
            mult *= 0.85
        return mult
    scr["_anomaly_mult"] = scr.apply(anomaly_multiplier, axis=1)

    # Raw conviction
    raw_conv = (
        scr["Score"]
        * (0.5 + 0.5 * scr["_completeness"])
        * (0.7 + 0.3 * scr["_signal_agreement"])
        * scr["_anomaly_mult"]
    )
    c_min, c_max = raw_conv.min(), raw_conv.max()
    scr["Conviction Score"] = (
        (raw_conv - c_min) / (c_max - c_min) * 100.0
        if c_max > c_min else pd.Series(50.0, index=scr.index)
    )
    return scr.drop(columns=[
        "_completeness", "_signal_agreement",
        "_anomaly_mult", "_sector_med_pe",
    ])


# ══════════════════════════════════════════════════════════════════════════════
# GAP 9 — Cross-Sectional Score
# ══════════════════════════════════════════════════════════════════════════════
def compute_cross_sectional_scores(scr: pd.DataFrame) -> pd.DataFrame:
    """
    Score each stock against the FULL S&P 500 universe (not just its sector).

    Uses elite_factor_score() with same weights as sector scoring but applied
    cross-sectionally, so a top-20 IT stock can be compared directly against
    a top-20 Utilities stock.

    Output column: CS Score (0–100).
    """
    scr    = scr.copy()
    cs_val  = elite_factor_score(
        scr["Fwd P/E"].fillna(scr["P/E"]), ascending=True)
    cs_peg  = elite_factor_score(scr["PEG"],            ascending=True)
    cs_qual = elite_factor_score(scr["Quality Score"],  ascending=False)
    cs_mom  = elite_factor_score(scr["Momentum Score"], ascending=False)
    cs_et   = elite_factor_score(scr["Earn Traj"],      ascending=False)

    cs_raw = (0.25 * cs_val +
              0.25 * cs_qual +
              0.20 * cs_peg  +
              0.15 * cs_et   +
              0.15 * cs_mom)

    cs_min, cs_max = cs_raw.min(), cs_raw.max()
    scr["CS Score"] = (
        (cs_raw - cs_min) / (cs_max - cs_min) * 100.0
        if cs_max > cs_min else pd.Series(50.0, index=scr.index)
    )
    return scr


# ══════════════════════════════════════════════════════════════════════════════
# GAP 10 — Score History helpers
# ══════════════════════════════════════════════════════════════════════════════
def record_score_history(scr: pd.DataFrame) -> None:
    """
    Persist up to 10 snapshots per ticker in st.session_state["score_history"].
    Called once after build_screener_table() completes.
    """
    if "score_history" not in st.session_state:
        st.session_state["score_history"] = {}

    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    for _, row in scr.iterrows():
        t = row["Ticker"]
        if t not in st.session_state["score_history"]:
            st.session_state["score_history"][t] = []
        st.session_state["score_history"][t].append({
            "ts":             ts,
            "score":          row.get("Score"),
            "conviction":     row.get("Conviction Score"),
            "rank":           row.get("Rank"),
            "piotroski_f":    row.get("Piotroski F"),
            "momentum_score": row.get("Momentum Score"),
        })
        # Keep only last 10 snapshots
        st.session_state["score_history"][t] = (
            st.session_state["score_history"][t][-10:]
        )


def get_score_delta(ticker: str, current_score) -> float:
    """
    Returns Score change vs previous snapshot, or None on first load.
    """
    history = st.session_state.get("score_history", {}).get(ticker, [])
    if len(history) < 2:
        return None
    prev = history[-2].get("score")
    if prev is None or current_score is None or pd.isna(current_score):
        return None
    try:
        return round(float(current_score) - float(prev), 1)
    except Exception:
        return None


# ── v16 Piotroski F-Score (unchanged) ────────────────────────────────────────
def compute_piotroski_fscore(d: dict) -> tuple:
    score = 0; comp = {}
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
    gm_now  = d.get("gross_margin_now")
    gm_prev = d.get("gross_margin_prev")
    o1 = 1 if (gm_now is not None and gm_prev is not None
                and gm_now > gm_prev) else 0
    score += o1; comp["O1_gross_margin_up"] = o1
    rev_growth = d.get("rev_growth_pct")
    o2 = 1 if (rev_growth is not None and rev_growth > 0) else 0
    score += o2; comp["O2_asset_turn_up"] = o2
    return score, comp


# ── v16 Sloan Ratio (unchanged) ───────────────────────────────────────────────
def compute_sloan_ratio(net_income, ocf, total_assets_now, total_assets_prev):
    try:
        if any(v is None for v in
               [net_income, ocf, total_assets_now, total_assets_prev]):
            return None
        avg_ta = (float(total_assets_now) + float(total_assets_prev)) / 2.0
        if avg_ta <= 0: return None
        return (float(net_income) - float(ocf)) / avg_ta
    except Exception:
        return None


# ── v16 Elite Quality Score (unchanged) ──────────────────────────────────────
def compute_quality_score_elite(
    roic, roe, int_coverage, op_margin,
    gross_margin_now=None, gross_margin_prev=None,
    fcf_ni_ratio=None, piotroski_f=None, sloan_ratio=None, sector=None,
):
    scores = []; weights = []
    profitability = (roe if sector in ROE_PRIMARY_SECTORS
                     else (roic if roic is not None else roe))
    if profitability is not None and not pd.isna(profitability):
        pf = float(profitability)
        s  = min(100.0, np.log1p(pf) / np.log1p(30.0) * 100.0) if pf > 0 else 0.0
    else:
        s = 0.0
    scores.append(s); weights.append(0.25)

    scores.append(min(100.0, max(0.0, float(int_coverage)/10.0*100.0))
                  if int_coverage is not None and not pd.isna(int_coverage)
                  else 0.0)
    weights.append(0.15)

    if sector not in ROE_PRIMARY_SECTORS:
        scores.append(min(100.0, max(0.0, float(op_margin)/40.0*100.0))
                      if op_margin is not None and not pd.isna(op_margin)
                      else 0.0)
        weights.append(0.15)

    if gross_margin_now is not None and not pd.isna(gross_margin_now):
        gm_score = min(100.0, max(0.0, float(gross_margin_now)/60.0*100.0))
        if (gross_margin_prev is not None
                and float(gross_margin_now) > float(gross_margin_prev)):
            gm_score = min(100.0, gm_score * 1.10)
        scores.append(gm_score)
    else:
        scores.append(0.0)
    weights.append(0.20)

    scores.append(float(piotroski_f)/9.0*100.0
                  if piotroski_f is not None and not pd.isna(piotroski_f)
                  else 0.0)
    weights.append(0.15)

    if sloan_ratio is not None and not pd.isna(sloan_ratio):
        sr = float(sloan_ratio)
        sr_clamped  = max(-0.15, min(0.05, sr))
        sloan_score = (0.05 - sr_clamped) / 0.20 * 100.0
        scores.append(sloan_score)
    else:
        scores.append(0.0)
    weights.append(0.10)

    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_w if total_w else 0.0


# ── v16 Valuation sub-score (unchanged) ──────────────────────────────────────
def compute_valuation_subscore(elig: pd.DataFrame) -> pd.Series:
    scores = pd.DataFrame(index=elig.index); weights = []
    if "FCF Yield%" in elig.columns and elig["FCF Yield%"].notna().sum() >= 5:
        scores["fcf"]     = elite_factor_score(elig["FCF Yield%"], ascending=False)
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
    composite = sum(scores[k] * (w / total_w) for k, w in weights
                    if k in scores.columns)
    return composite.fillna(0.0)


# ── S&P 500 universe ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_sp500_constituents():
    url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r    = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    tbl  = soup.find("table", {"id": "constituents"})
    if tbl is None: raise RuntimeError("Wikipedia S&P 500 table not found")
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
# PRICES + MOMENTUM  (v17: 4-signal composite via compute_elite_momentum)
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_price_momentum_one(t, spy_3mo=None):
    result = {
        "price":           None, "hi52": None, "lo52": None,
        "ret_1mo":         None, "ret_3mo": None, "ret_6mo": None,
        "trailing_vol":    None,
        # v17 composite replaces the single skip-month score
        "momentum_score":  None,   # composite (used in scoring)
        "skip_month_raw":  None,   # individual sub-score display
        "hi52_proximity":  None,
        "vs_ma200":        None,
        "rel_strength_spy": None,
    }
    try:
        obj  = yf.Ticker(t)
        hist = obj.history(period="12mo", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return t, result
        closes = hist["Close"].dropna()
        if closes.empty:
            return t, result

        price_val = float(closes.iloc[-1])
        hi52_val  = float(closes.max())
        lo52_val  = float(closes.min())
        result["price"] = price_val
        result["hi52"]  = hi52_val
        result["lo52"]  = lo52_val

        # Monthly returns for display columns
        monthly = closes.resample("ME").last().dropna()
        if len(monthly) >= 2:
            px_now = float(monthly.iloc[-1])
            if px_now > 0:
                def ret_mo(n):
                    idx = -(n + 1)
                    if abs(idx) > len(monthly): return None
                    px = float(monthly.iloc[idx])
                    return (px_now / px - 1) * 100.0 if px > 0 else None
                result["ret_1mo"] = ret_mo(1)
                result["ret_3mo"] = ret_mo(3)
                result["ret_6mo"] = ret_mo(6)

        # Trailing vol
        if len(closes) >= 20:
            dr = closes.pct_change().dropna().tail(90)
            if len(dr) >= 15:
                result["trailing_vol"] = float(dr.std() * np.sqrt(252) * 100.0)

        # v17: 4-signal momentum
        composite, comps = compute_elite_momentum(
            closes, price_val, hi52_val, spy_3mo
        )
        result["momentum_score"]   = composite
        result["skip_month_raw"]   = comps.get("skip_month_raw")
        result["hi52_proximity"]   = comps.get("hi52_proximity")
        result["vs_ma200"]         = comps.get("vs_ma200")
        result["rel_strength_spy"] = comps.get("rel_strength_spy")

    except Exception:
        pass
    return t, result


@st.cache_data(ttl=3600)
def fetch_price_momentum_all(tickers, spy_3mo=None):
    tl     = list(tickers)
    out    = {t: {} for t in tl}
    CHUNK  = 25; WKRS = 10; SLEEP = 1.0
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog   = st.progress(0); status = st.empty()

    for ci, chunk in enumerate(chunks):
        status.text("Prices + Momentum: chunk {}/{} ({} done)...".format(
            ci + 1, len(chunks), ci * CHUNK))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futs = {ex.submit(_fetch_price_momentum_one, t, spy_3mo): t
                    for t in chunk}
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
# PHASE 1 — Yahoo info  (v17: +earnings surprise + revision momentum)
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
        # v16
        "ev_ebitda": None, "ev_sales": None, "div_yield": None,
        # v17 new ──────────────────────────────────────────────────────────
        "eps_surprise_avg":   None,
        "eps_beat_rate":      None,
        "eps_surprise_trend": None,
        "revision_momentum":  None,
    }
    try:
        obj = yf.Ticker(t)
        try:
            fi = obj.fast_info
            if fi is not None:
                mc_fi = sf(getattr(fi, "market_cap", None))
                if mc_fi: result["mc"] = mc_fi
        except Exception:
            pass

        info = {}
        for _ in range(2):
            try:
                info = obj.info or {}
                if (info.get("trailingPE") or info.get("pegRatio")
                        or info.get("forwardPE")):
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
        if roe_y is not None: result["roe"] = roe_y * 100.0

        om_y = sf(info.get("operatingMargins"))
        if om_y is not None: result["op_margin"] = om_y * 100.0

        de_y = sf(info.get("debtToEquity"))
        if de_y is not None: result["debt_eq"] = de_y / 100.0

        eg_y = sf(info.get("earningsGrowth"))
        if eg_y is not None: result["eps_growth"] = eg_y * 100.0

        fwd_eps_val   = sf(info.get("forwardEps"))
        trail_eps_val = sf(info.get("trailingEps"))
        if (fwd_eps_val is not None and trail_eps_val is not None
                and abs(trail_eps_val) > 0.01):
            earn_traj_raw = ((fwd_eps_val - trail_eps_val)
                             / abs(trail_eps_val))
            clipped = max(-1.0, min(1.0, earn_traj_raw))
            if trail_eps_val < 0 and fwd_eps_val < 0:
                clipped = min(clipped, 0.30)
            result["earn_traj"] = clipped

        if result["mc"] is None:
            mc_y = sf(info.get("marketCap"))
            if mc_y: result["mc"] = mc_y

        # v16 valuation extras
        ev_raw     = sf(info.get("enterpriseValue"))
        ebitda_raw = sf(info.get("ebitda"))
        if ev_raw and ebitda_raw and ebitda_raw > 0:
            ev_eb = ev_raw / ebitda_raw
            if 0 < ev_eb < 200: result["ev_ebitda"] = ev_eb
        rev_ttm_y = sf(info.get("totalRevenue"))
        if ev_raw and rev_ttm_y and rev_ttm_y > 0:
            ev_s = ev_raw / rev_ttm_y
            if 0 < ev_s < 100: result["ev_sales"] = ev_s
        dy = sf(info.get("dividendYield"))
        if dy is not None: result["div_yield"] = dy * 100.0

        # v17 Gap 5: earnings surprise
        avg_surp, beat_rt, surp_trend = extract_earnings_surprise_trend(obj)
        result["eps_surprise_avg"]   = avg_surp
        result["eps_beat_rate"]      = beat_rt
        result["eps_surprise_trend"] = surp_trend

        # v17 Gap 6: revision momentum
        result["revision_momentum"] = extract_revision_momentum(obj)

    except Exception:
        pass
    return t, result


@st.cache_data(ttl=86400)
def fetch_yahoo_info_all(tickers, _cache_date=None):
    tl     = list(tickers); out = {}
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
# PHASE 2 — Deep financials  (unchanged from v16)
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_yahoo_deep_one(t):
    result = {
        "roic": None, "int_coverage": None,
        "rev4": [None, None, None, None],
        "ocf_ttm": None, "fcf_ttm": None, "net_income_ttm": None,
        "gross_profit_ttm": None, "gross_margin_now": None,
        "gross_margin_prev": None, "roa_ttm": None, "roa_prev": None,
        "total_assets_now": None, "total_assets_prev": None,
        "lt_debt_ratio_now": None, "lt_debt_ratio_prev": None,
        "current_ratio_now": None, "current_ratio_prev": None,
        "shares_now": None, "shares_prev": None,
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

        if qfin is not None and not qfin.empty:
            rev_row = next((nm for nm in ["Total Revenue", "Revenue"]
                            if nm in qfin.index), None)
            if rev_row:
                rs = qfin.loc[rev_row].sort_index().dropna().tail(4)
                if len(rs) == 4:
                    result["rev4"] = [float(x) for x in rs.values]
                elif len(rs) > 0:
                    vals = [float(x) for x in rs.values]
                    result["rev4"] = ([None] * (4 - len(vals))) + vals

            ebit_row = next((nm for nm in ["EBIT","Operating Income","Ebit"]
                             if nm in qfin.index), None)
            int_row  = next((nm for nm in ["Interest Expense",
                             "Interest Expense Non Operating",
                             "Net Interest Income"]
                             if nm in qfin.index), None)
            if ebit_row and int_row:
                ebit_ttm = qfin.loc[ebit_row].dropna().head(4).sum()
                int_ttm  = abs(qfin.loc[int_row].dropna().head(4).sum())
                if int_ttm > 0 and ebit_ttm > 0:
                    result["int_coverage"] = min(
                        float(ebit_ttm / int_ttm), 100.0)

        if (qfin is not None and not qfin.empty
                and qbs is not None and not qbs.empty):
            op_inc_row = next((nm for nm in
                               ["Operating Income","EBIT","Ebit"]
                               if nm in qfin.index), None)
            tax_row    = next((nm for nm in
                               ["Tax Provision","Income Tax Expense",
                                "Tax Expense"]
                               if nm in qfin.index), None)
            pretax_row = next((nm for nm in
                               ["Pretax Income","Income Before Tax","EBT"]
                               if nm in qfin.index), None)
            if op_inc_row:
                op_inc_ttm   = float(qfin.loc[op_inc_row].dropna().head(4).sum())
                eff_tax_rate = 0.21
                if tax_row and pretax_row:
                    tax_ttm    = float(qfin.loc[tax_row].dropna().head(4).sum())
                    pretax_ttm = float(qfin.loc[pretax_row].dropna().head(4).sum())
                    if pretax_ttm > 0 and tax_ttm >= 0:
                        cr = tax_ttm / pretax_ttm
                        if 0 < cr < 0.6: eff_tax_rate = cr
                nopat = op_inc_ttm * (1 - eff_tax_rate)

                def _bs_val(names):
                    return next((float(qbs.loc[nm].dropna().iloc[0])
                                 for nm in names
                                 if nm in qbs.index
                                 and len(qbs.loc[nm].dropna()) > 0), None)

                equity_val = _bs_val(["Total Stockholders Equity",
                                      "Stockholders Equity",
                                      "Common Stock Equity",
                                      "Total Equity Gross Minority Interest"])
                debt_val   = _bs_val(["Total Debt","Net Debt","Long Term Debt",
                                      "Long Term Debt And Capital Lease Obligation"])
                cash_val   = _bs_val(["Cash And Cash Equivalents",
                                      "Cash Cash Equivalents And Short Term Investments",
                                      "Cash Financial",
                                      "Cash And Short Term Investments"])
                cash_use = 0
                if cash_val is not None:
                    rv4 = result["rev4"]
                    if all(v is not None for v in rv4):
                        rtm = sum(rv4)
                        if rtm > 0:
                            cash_use = max(
                                0.0, cash_val - OPERATING_CASH_PCT_OF_REV * rtm)
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

        if qcf is not None and not qcf.empty:
            ocf_row   = next((n for n in ["Operating Cash Flow",
                              "Cash From Operations",
                              "Total Cash From Operating Activities"]
                              if n in qcf.index), None)
            capex_row = next((n for n in ["Capital Expenditure",
                              "Purchase Of PPE","Capital Expenditures",
                              "Purchases Of Property Plant And Equipment"]
                              if n in qcf.index), None)
            if ocf_row:
                ocf_ttm = float(qcf.loc[ocf_row].dropna().head(4).sum())
                result["ocf_ttm"] = ocf_ttm
                if capex_row:
                    capex_ttm = abs(float(qcf.loc[capex_row].dropna().head(4).sum()))
                    result["fcf_ttm"] = ocf_ttm - capex_ttm

        if qfin is not None and not qfin.empty:
            gp_row   = next((n for n in ["Gross Profit","Gross Income"]
                             if n in qfin.index), None)
            rev_row2 = next((n for n in ["Total Revenue","Revenue"]
                             if n in qfin.index), None)
            ni_row   = next((n for n in ["Net Income",
                             "Net Income Common Stockholders"]
                             if n in qfin.index), None)
            if gp_row and rev_row2:
                gp_ttm   = float(qfin.loc[gp_row].dropna().head(4).sum())
                rev_ttm2 = float(qfin.loc[rev_row2].dropna().head(4).sum())
                if rev_ttm2 > 0:
                    result["gross_profit_ttm"] = gp_ttm
                    result["gross_margin_now"]  = gp_ttm / rev_ttm2 * 100.0
                rev_all = qfin.loc[rev_row2].dropna()
                gp_all  = qfin.loc[gp_row].dropna()
                if len(rev_all) >= 8 and len(gp_all) >= 8:
                    rp  = float(rev_all.iloc[4:8].sum())
                    gpp = float(gp_all.iloc[4:8].sum())
                    if rp > 0:
                        result["gross_margin_prev"] = gpp / rp * 100.0
            if ni_row:
                result["net_income_ttm"] = float(
                    qfin.loc[ni_row].dropna().head(4).sum())

        if qbs is not None and not qbs.empty:
            ta_row  = next((n for n in ["Total Assets","Assets"]
                            if n in qbs.index), None)
            ltd_row = next((n for n in ["Long Term Debt",
                            "Long Term Debt And Capital Lease Obligation"]
                            if n in qbs.index), None)
            ca_row  = next((n for n in ["Current Assets","Total Current Assets"]
                            if n in qbs.index), None)
            cl_row  = next((n for n in ["Current Liabilities",
                            "Total Current Liabilities"]
                            if n in qbs.index), None)
            if ta_row:
                ta_vals = qbs.loc[ta_row].dropna()
                if len(ta_vals) >= 1:
                    ta_now = float(ta_vals.iloc[0])
                    result["total_assets_now"] = ta_now
                    if len(ta_vals) >= 5:
                        ta_prev = float(ta_vals.iloc[4])
                        result["total_assets_prev"] = ta_prev
                        avg_ta  = (ta_now + ta_prev) / 2.0
                        ni_val  = result.get("net_income_ttm")
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
                        result["current_ratio_prev"] = (
                            float(ca_vals.iloc[4]) / cl_prev)

        result["shares_now"]  = sf(info.get("sharesOutstanding"))
        result["shares_prev"] = sf(info.get("floatShares"))

    except Exception:
        pass
    return t, result


@st.cache_data(ttl=86400)
def fetch_yahoo_deep_financials(tickers_filtered, _cache_date=None):
    tl  = list(tickers_filtered); out = {}
    if not tl: return out
    CHUNK = 20; WKRS = 6; SLEEP = 2.0
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog   = st.progress(0); status = st.empty()
    for ci, chunk in enumerate(chunks):
        status.text("Phase 2/2 — Deep: chunk {}/{} ({}/{})...".format(
            ci + 1, len(chunks), min((ci+1)*CHUNK, len(tl)), len(tl)))
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
    NEW_DEEP = [
        "ocf_ttm","fcf_ttm","net_income_ttm","gross_profit_ttm",
        "gross_margin_now","gross_margin_prev","roa_ttm","roa_prev",
        "total_assets_now","total_assets_prev",
        "lt_debt_ratio_now","lt_debt_ratio_prev",
        "current_ratio_now","current_ratio_prev",
        "shares_now","shares_prev",
    ]
    merged = {}
    for t in tickers:
        base = dict(info_map.get(t, {}))
        deep = deep_map.get(t, {})
        base["roic"]         = (deep.get("roic") if deep.get("roic") is not None
                                 else base.get("roic"))
        base["int_coverage"] = (deep.get("int_coverage")
                                 if deep.get("int_coverage") is not None
                                 else base.get("int_coverage"))
        base["rev4"] = deep.get("rev4", [None, None, None, None])
        for f in NEW_DEEP:
            base[f] = deep.get(f)
        merged[t] = base
    return merged


# ── FMP fetches (unchanged from v16) ──────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_quotes_if_available(tickers, api_key):
    out = {}
    if not api_key: return out
    tl = list(tickers)
    for chunk in [tl[i:i+100] for i in range(0, len(tl), 100)]:
        url = ("https://financialmodelingprep.com/api/v3/quote/{}?apikey={}".format(
            ",".join(chunk), api_key))
        try:
            r = requests.get(url, timeout=20); r.raise_for_status()
            data = r.json()
            if not isinstance(data, list): continue
            for item in data:
                t = str(item.get("symbol","")).upper().strip()
                if not t: continue
                pe = sf(item.get("pe")); mc = sf(item.get("marketCap"))
                if pe is not None and (pe <= 0 or pe > 10_000): pe = None
                out[t] = {"pe": pe, "mc": mc,
                          "pe_src": "FMP-quote" if pe is not None else None}
        except Exception:
            pass
        time.sleep(0.3)
    return out


@st.cache_data(ttl=86400)
def fetch_fmp_ratios_if_available(tickers, api_key):
    out = {}
    if not api_key: return out
    try:
        r = requests.get(
            "https://financialmodelingprep.com/api/v3/ratios-ttm/AAPL?apikey={}".format(
                api_key), timeout=10)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0: return out
        st.session_state["fmp_ratios_fields"] = list(data[0].keys())
    except Exception:
        return out

    def fetch_one(t):
        url = ("https://financialmodelingprep.com/api/v3/ratios-ttm/{}?apikey={}".format(
            t, api_key))
        try:
            r = requests.get(url, timeout=12)
            if r.status_code == 429: time.sleep(3.0); r = requests.get(url, timeout=12)
            r.raise_for_status()
            d = r.json()
            if not isinstance(d, list) or len(d) == 0: return t, {}
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
            fmp_pe = (fmp_pe_raw if (fmp_pe_raw and 0 < fmp_pe_raw <= 10_000) else None)
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
                    if d: out[t] = d
                except Exception:
                    pass
        if ci < (len(tl) // 50): time.sleep(1.0)
    return out


def merge_all_sources(yahoo_data, fmp_quotes, fmp_ratios, tickers):
    NEW_FIELDS = [
        "ev_ebitda","ev_sales","div_yield",
        "ocf_ttm","fcf_ttm","net_income_ttm","gross_profit_ttm",
        "gross_margin_now","gross_margin_prev","roa_ttm","roa_prev",
        "total_assets_now","total_assets_prev",
        "lt_debt_ratio_now","lt_debt_ratio_prev",
        "current_ratio_now","current_ratio_prev",
        "shares_now","shares_prev",
        # v17 new
        "eps_surprise_avg","eps_beat_rate","eps_surprise_trend",
        "revision_momentum",
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
        row = {
            "pe":          pe_val,
            "pe_src":      ("FMP-quote"  if fq.get("pe")              is not None else
                            "FMP-ratios" if fr.get("fmp_trailing_pe") is not None else
                            yb.get("pe_src","Yahoo")),
            "fwd_pe":      yb.get("fwd_pe"),
            "peg":         first(fr.get("peg"),         yb.get("peg")),
            "peg_src":     ("FMP-ratios" if fr.get("peg") is not None else
                            yb.get("peg_src","Yahoo") if yb.get("peg") is not None
                            else "—"),
            "roic":        first(fr.get("roic"),         yb.get("roic")),
            "roe":         first(fr.get("roe"),           yb.get("roe")),
            "int_coverage":first(fr.get("int_coverage"), yb.get("int_coverage")),
            "op_margin":   first(fr.get("op_margin"),    yb.get("op_margin")),
            "debt_eq":     first(fr.get("debt_eq"),       yb.get("debt_eq")),
            "eps_growth":  yb.get("eps_growth"),
            "growth_src":  "Yahoo" if yb.get("eps_growth") is not None else None,
            "earn_traj":   yb.get("earn_traj"),
            "mc":          first(fq.get("mc"), yb.get("mc")),
            "rev4":        yb.get("rev4", [None, None, None, None]),
        }
        for f in NEW_FIELDS:
            row[f] = yb.get(f)
        merged[t] = row
    return merged


# ── Quality flag (v16 + HighAccruals) ────────────────────────────────────────
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
    if sloan_ratio is not None and not pd.isna(sloan_ratio) and float(sloan_ratio) > 0.05:
        flags.append("HighAccruals")
    return ", ".join(flags) if flags else "Pass"


# ══════════════════════════════════════════════════════════════════════════════
# RANKING  (v16 valuation composite + v15 MAD scoring — unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def compute_rank_by_sector(scr):
    scr = scr.copy()
    scr["Score"] = pd.NA; scr["Rank"] = pd.NA

    for sector in scr["Sector"].dropna().unique():
        g    = scr[scr["Sector"] == sector].copy()
        elig = g[g["Eligible"]].copy()
        if elig.empty: continue
        W = SECTOR_FACTOR_WEIGHTS.get(sector, DEFAULT_FACTOR_WEIGHTS)

        elig["_s_val"]   = compute_valuation_subscore(elig)
        elig["_s_peg"]   = elite_factor_score(elig["PEG"],            ascending=True)
        elig["_s_mom"]   = elite_factor_score(elig["Momentum Score"], ascending=False)
        elig["_s_etraj"] = elite_factor_score(elig["Earn Traj"],      ascending=False)

        qs    = elig["Quality Score"]
        q_min = qs.min(); q_max = qs.max()
        if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min:
            elig["_s_quality"] = (qs - q_min) / (q_max - q_min) * 100.0
        else:
            elig["_s_quality"] = qs.fillna(0.0)
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)

        raw = (W["valuation"] * elig["_s_val"]     +
               W["quality"]   * elig["_s_quality"] +
               W["peg"]       * elig["_s_peg"]     +
               W["earn_traj"] * elig["_s_etraj"]   +
               W["momentum"]  * elig["_s_mom"])

        factor_cols = ["P/E","PEG","Quality Score","Earn Traj","Momentum Score"]
        penalties   = elig.apply(
            lambda r: missing_factor_penalty(r, factor_cols), axis=1)
        raw = raw * penalties

        elig["Score"] = raw
        elig = elig.sort_values("Score", ascending=False)
        elig["Rank"] = range(1, len(elig) + 1)
        scr.loc[elig.index, "Score"] = elig["Score"]
        scr.loc[elig.index, "Rank"]  = elig["Rank"]

    return scr


# ── Build screener table ───────────────────────────────────────────────────────
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
        mom_score = to_num(pm.get("momentum_score"))   # composite

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
        if peg is not None and (peg <= 0 or peg > 500): peg = None

        # Piotroski + Sloan
        fi_g = dict(fi)
        fi_g["rev_growth_pct"] = float(growth) if growth is not None else None
        piotroski_f, _ = compute_piotroski_fscore(fi_g)
        sloan_ratio    = compute_sloan_ratio(
            fi.get("net_income_ttm"), fi.get("ocf_ttm"),
            fi.get("total_assets_now"), fi.get("total_assets_prev"))

        # FCF Yield
        fcf_ttm   = fi.get("fcf_ttm")
        fcf_yield = None
        if fcf_ttm is not None and pd.notna(mc) and float(mc) > 0:
            fcf_yield = float(fcf_ttm) / float(mc) * 100.0

        # FCF/NI ratio
        fcf_ni_ratio = None
        ni_val = fi.get("net_income_ttm")
        if fcf_ttm is not None and ni_val is not None and float(ni_val) != 0:
            fcf_ni_ratio = float(fcf_ttm) / float(ni_val)

        # Quality score (7 sub-signals)
        q_score = compute_quality_score_elite(
            float(roic)   if pd.notna(roic) else None,
            float(roe)    if pd.notna(roe)  else None,
            float(ic)     if pd.notna(ic)   else None,
            float(om)     if pd.notna(om)   else None,
            gross_margin_now  = (float(fi["gross_margin_now"])
                                 if fi.get("gross_margin_now") is not None else None),
            gross_margin_prev = (float(fi["gross_margin_prev"])
                                 if fi.get("gross_margin_prev") is not None else None),
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
            # v16 columns
            "EV/EBITDA":    to_num(fi.get("ev_ebitda")),
            "FCF Yield%":   to_num(fcf_yield),
            "EV/Sales":     to_num(fi.get("ev_sales")),
            "Div Yield%":   to_num(fi.get("div_yield")),
            "Piotroski F":  to_num(piotroski_f),
            "Sloan Ratio":  to_num(sloan_ratio),
            # v17 columns
            "EPS Surp Avg%":   to_num(fi.get("eps_surprise_avg")),
            "EPS Beat Rate":   to_num(fi.get("eps_beat_rate")),
            "EPS Surp Trend":  to_num(fi.get("eps_surprise_trend")),
            "Revision Mom":    to_num(fi.get("revision_momentum")),
            "Skip Mo":         to_num(pm.get("skip_month_raw")),
            "52W Prox":        to_num(pm.get("hi52_proximity")),
            "vs MA200":        to_num(pm.get("vs_ma200")),
            "Rel Str SPY":     to_num(pm.get("rel_strength_spy")),
        })

    scr = pd.DataFrame(rows)
    if scr.empty: return scr

    total_sp500_mc = scr["Mkt Cap"].sum()
    scr["MC% of S&P500"] = (
        (scr["Mkt Cap"] / total_sp500_mc * 100.0)
        if total_sp500_mc > 0 else None
    )

    num_cols = [
        "Price","Mkt Cap","P/E","Fwd P/E","PEG","52W Pos%",
        "ROIC%","ROE%","Int Coverage","Op Margin%","Debt/Eq",
        "Quality Score","Earn Traj","Momentum Score",
        "Ret 1Mo%","Ret 3Mo%","Ret 6Mo%","Trailing Vol%","MC% of S&P500",
        "Rev Q1 Oldest ($B)","Rev Q2 ($B)","Rev Q3 ($B)","Rev Q4 Latest ($B)",
        "Rev Growth% (CAGR)",
        "EV/EBITDA","FCF Yield%","EV/Sales","Div Yield%",
        "Piotroski F","Sloan Ratio",
        "EPS Surp Avg%","EPS Beat Rate","EPS Surp Trend","Revision Mom",
        "Skip Mo","52W Prox","vs MA200","Rel Str SPY",
    ]
    for c in num_cols:
        if c in scr.columns:
            scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns: scr["Rank"] = pd.NA

    sector_med_pe = scr.groupby("Sector")["P/E"].transform("median")
    scr["P/E vs Sector Med"] = (scr["P/E"] / sector_med_pe).round(2)

    # v17 Gap 8: elite conviction
    scr = compute_conviction_scores_elite(scr)

    # v17 Gap 9: cross-sectional score
    scr = compute_cross_sectional_scores(scr)

    return scr


# ── Reference Guide ────────────────────────────────────────────────────────────
def render_reference_guide():
    st.markdown("## Column Reference Guide")
    st.caption("All metrics with formula, benchmarks, and scoring role.")

    tabs = st.tabs([
        "Valuation", "Quality", "PEG", "Earn Trajectory",
        "Momentum (v17)", "Earnings Surprise (v17)",
        "Ranking & Score", "Display-Only",
    ])
    tab_val, tab_qual, tab_peg, tab_etraj, tab_mom, tab_surp, tab_rank, tab_disp = tabs

    with tab_val:
        st.markdown("""
**Valuation Composite (v16)** — FCF Yield 40% + EV/EBITDA 35% + Fwd P/E 25%

Weights self-normalise when signals are missing.

| Signal | Better | Why |
|---|---|---|
| FCF Yield% = (OCF−CapEx)/MktCap×100 | Higher | Pure cash — no accounting noise |
| EV/EBITDA = EV/EBITDA TTM | Lower | Capital-structure neutral |
| Fwd P/E (or trailing P/E) | Lower | Consensus anchor |

EV/Sales and Div Yield% are display-only.
        """)

    with tab_qual:
        st.markdown("""
**Quality Score (0–100) — v16: 7 sub-signals**

| Sub-signal | Weight | Formula | Why |
|---|---|---|---|
| Profitability ROIC/ROE | 25% | log1p / log1p(30) | Capital efficiency |
| Interest Coverage | 15% | EBIT / Interest TTM | Safety margin |
| Operating Margin (non-Fin) | 15% | Op Inc / Rev | Operational efficiency |
| Gross Margin + trend | 20% | Gross Profit / Rev | Moat / pricing power |
| Piotroski F (0–9) | 15% | 9 binary tests | Holistic health |
| Sloan Ratio | 10% | (NI − OCF) / Avg Assets | Earnings quality |

Quality Flag: `ROIC<8%` · `IntCov<3x` · `Margin<5%` · `HighAccruals` · `Pass`
        """)

    with tab_peg:
        st.markdown("""
**PEG** = P/E / EPS Growth Rate (%). Guard: growth ≥ 5%.

| PEG | Signal |
|---|---|
| < 1.0 | Potentially undervalued for growth rate |
| 1.0–2.0 | Fair value |
| > 2.0 | Expensive vs growth |
| N/A | Growth below floor |
        """)

    with tab_etraj:
        st.markdown("""
**Earn Traj** = (Fwd EPS − Trail EPS) / |Trail EPS|, clipped [−1, +1].

Both-negative cap at +0.30 (recovery-in-progress signal).
        """)

    with tab_mom:
        st.markdown("""
**Momentum Score (v17) — 4-signal composite**

| Signal | Weight | Formula | Source |
|---|---|---|---|
| Skip Mo (skip-month) | 40% | (6Mo ret − 1Mo ret) / Ann Vol, clipped ÷ 2 | Jegadeesh & Titman 1993 |
| 52W Prox (52W-high proximity) | 25% | 1 − (price−hi52)/(hi52×0.30) | George & Hwang 2004 |
| vs MA200 (price vs 200d MA) | 20% | (price−MA200)/MA200 ÷ 0.30 | Trend confirmation |
| Rel Str SPY (3Mo vs SPY) | 15% | (Stock 3Mo − SPY 3Mo) ÷ 20 | Relative outperformance |

Composite in [-1, +1]. Weights self-normalise when SPY unavailable.
Momentum Score in the table is the composite; sub-scores shown as separate columns.
        """)

    with tab_surp:
        st.markdown("""
**EPS Surprise Signals (v17 — display only)**

| Column | Formula | Interpretation |
|---|---|---|
| EPS Surp Avg% | Mean (Actual − Est) / \|Est\| × 100 over last 4Q | +5% = beat by 5% on average |
| EPS Beat Rate | Count(beats) / 4 | 1.0 = beat all 4 quarters |
| EPS Surp Trend | +1 if recent 2Q avg > earlier 2Q avg, else −1 | Improving/deteriorating |

Academic: Post-Earnings Announcement Drift (Bernard & Thomas 1990) — systematic beaters
outperform for 60 days post-announcement due to analyst anchoring.

**Revision Mom** = analyst upgrade/downgrade momentum.
Uses `recommendations_summary` — compares latest vs prior period buy+sell counts.
Range: −1.0 (strong downgrade trend) to +1.0 (strong upgrade trend).
        """)

    with tab_rank:
        st.markdown("""
**Score (0–100)** — sector-adaptive weighted composite.
**Normalisation pipeline (v15):** Winsorise → MAD z-score → Clip[−3,+3] → Rescale [0,100].

---
**Conviction Score (v17 — rebuilt)**

Three independent multipliers on Score:
1. Data completeness: 0.5–1.0 (fraction of 6 key factors present)
2. Signal agreement: 0.7–1.0 (std-dev of valuation/momentum/earn_traj signals)
3. Anomaly: 0.70–1.00 (Piotroski ≤2 → ×0.70; Sloan >0.05 → ×0.85)

---
**CS Score (v17 — Cross-Sectional)**
Full S&P 500 universe ranking — compares IT stocks against Utilities directly.
Weights: Val 25%, Quality 25%, PEG 20%, EarnTraj 15%, Momentum 15%.

---
**Score Δ** — change in Score vs previous session snapshot.
None on first load; numeric on subsequent refreshes.

| Sector | Val | Quality | PEG | Earn | Mom |
|---|---|---|---|---|---|
| Information Technology | 20% | 25% | 25% | 15% | 15% |
| Consumer Staples | 28% | 32% | 10% | 15% | 15% |
| Financials | 30% | 25% | 18% | 17% | 10% |
| Energy | 30% | 18% | 12% | 15% | 25% |
| Utilities | 38% | 27% | 5% | 15% | 15% |
        """)

    with tab_disp:
        st.markdown("""
**v17 New Columns**

| Column | Typical Coverage |
|---|---|
| EPS Surp Avg% / Beat Rate / Trend | ~65% |
| Revision Mom | ~75% |
| Skip Mo / 52W Prox / vs MA200 / Rel Str SPY | ~97% |
| CS Score | 100% (computed) |
| Score Δ | None first load |

**v16 Columns**

EV/EBITDA (~75%) · FCF Yield% (~65%) · Piotroski F (~55%) · Sloan Ratio (~55%)
        """)

    st.markdown("---")
    st.markdown(
        "**v17:** v15 MAD normalisation · v16 valuation composite + 7-signal quality · "
        "v17 earnings surprise + revision momentum + 4-signal momentum + "
        "elite conviction + CS Score + score history. "
        "_Nothing here is financial advice._"
    )


# ══════════════════════════════════════════════════════════════════════════════
# APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="S&P 500 Screener v17", layout="wide", page_icon="📊"
)
st.markdown(
    "<style>"
    "div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}"
    "</style>",
    unsafe_allow_html=True,
)
st.markdown("## S&P 500 Fundamental Screener v17")

page_screener, page_reference = st.tabs(["Screener", "Column Reference Guide"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1
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
            "v17: 4-signal momentum · EPS surprise · Revision mom · "
            "Elite conviction · CS Score · Score history".format(
                datetime.now().strftime("%I:%M %p")
            )
        )

    fmp_key = get_fmp_key()
    if fmp_key:
        st.success("FMP API key found — bonus layer active.")
    else:
        st.info("No FMP key. Yahoo Finance only. "
                "Add [fmp] api_key to Streamlit Secrets.")

    with st.spinner("Loading S&P 500 universe..."):
        sp500 = fetch_sp500_constituents()
    if sp500.empty:
        st.error("Failed to load S&P 500 universe.")
        st.stop()

    universe_df = sp500.copy().reset_index(drop=True)
    tickers     = tuple(universe_df["Ticker"].tolist())
    today_date  = date.today()

    # ── Filters ───────────────────────────────────────────────────────────
    st.markdown("### Filters")
    all_sectors = sorted(universe_df["Sector"].dropna().unique().tolist())
    f1, f2, f3, f4, f5 = st.columns(5)

    sector_sel = f1.selectbox("Sector", ["All Sectors"] + all_sectors)
    sort_by    = f2.selectbox("Sort by", [
        "Sector then Rank", "Score high to low", "Conviction high to low",
        "CS Score high to low",
        "MC% of S&P500 high to low", "Price low to high", "Price high to low",
        "Mkt Cap high to low", "PE low to high", "Fwd PE low to high",
        "PEG low to high", "Quality Score high", "ROIC high to low",
        "ROE high to low", "Earn Traj high to low", "Rev Growth high to low",
        "Momentum Score high", "52W Pos low to high",
        "P/E vs Sector Med low to high",
        "Piotroski F high", "FCF Yield high", "EV/EBITDA low",
        "EPS Beat Rate high", "Revision Mom high",
        "Score Delta high",
    ])
    mc_min_b   = f3.number_input("Min Mkt Cap ($B)", value=0, step=10, min_value=0)
    pe_max     = f4.number_input("Max P/E", value=9999, step=50, min_value=0)
    qual_min_f = f5.number_input("Min Quality Score", value=0.0, step=5.0,
                                  min_value=0.0, max_value=100.0)

    # ── Data fetching ──────────────────────────────────────────────────────
    with st.spinner("Phase 1/2 — Yahoo info ({} tickers)...".format(len(tickers))):
        yahoo_info = fetch_yahoo_info_all(tickers, _cache_date=today_date)

    filtered_tickers = _pre_filter_tickers(yahoo_info, universe_df, mc_min_b, pe_max)

    with st.spinner("Phase 2/2 — Deep financials ({} tickers)...".format(
            len(filtered_tickers))):
        yahoo_deep = fetch_yahoo_deep_financials(
            tuple(filtered_tickers), _cache_date=today_date)

    yahoo_fundamentals = merge_yahoo_phases(yahoo_info, yahoo_deep, tickers)

    # v17: fetch SPY once, pass to momentum workers
    with st.spinner("Fetching SPY benchmark return..."):
        spy_3mo = fetch_spy_3mo_return()
    if spy_3mo is not None:
        st.caption("SPY 3Mo return: {:.1f}% (used for Rel Str SPY signal)".format(
            spy_3mo))

    with st.spinner("Prices + 4-signal Momentum ({} tickers)...".format(len(tickers))):
        pm_data = fetch_price_momentum_all(tickers, spy_3mo=spy_3mo)

    fmp_quotes = {}; fmp_ratios = {}
    if fmp_key:
        with st.spinner("FMP /quote..."):
            fmp_quotes = fetch_fmp_quotes_if_available(tickers, fmp_key)
        with st.spinner("FMP /ratios-ttm..."):
            fmp_ratios = fetch_fmp_ratios_if_available(tickers, fmp_key)

    with st.spinner("Merging sources..."):
        merged_map = merge_all_sources(
            yahoo_fundamentals, fmp_quotes, fmp_ratios, tickers)

    # Coverage
    total_t = len(tickers)
    def cov_m(key):
        return sum(1 for t in tickers
                   if merged_map.get(t, {}).get(key) is not None)
    def cov_p(key):
        return sum(1 for t in tickers
                   if pm_data.get(t, {}).get(key) is not None)

    st.info(
        "Coverage — "
        "P/E: {}/{} ({:.0f}%) · Fwd P/E: {}/{} ({:.0f}%) · "
        "EV/EBITDA: {}/{} ({:.0f}%) · FCF: {}/{} ({:.0f}%) · "
        "EPS Surprise: {}/{} ({:.0f}%) · Revision Mom: {}/{} ({:.0f}%) · "
        "Momentum: {}/{} ({:.0f}%) · "
        "Yahoo{}".format(
            cov_m("pe"),              total_t, cov_m("pe")              / total_t * 100,
            cov_m("fwd_pe"),          total_t, cov_m("fwd_pe")          / total_t * 100,
            cov_m("ev_ebitda"),       total_t, cov_m("ev_ebitda")       / total_t * 100,
            cov_m("fcf_ttm"),         total_t, cov_m("fcf_ttm")         / total_t * 100,
            cov_m("eps_surprise_avg"),total_t, cov_m("eps_surprise_avg")/ total_t * 100,
            cov_m("revision_momentum"),total_t,cov_m("revision_momentum")/total_t * 100,
            cov_p("momentum_score"),  total_t, cov_p("momentum_score")  / total_t * 100,
            " + FMP" if fmp_key else "",
        )
    )

    # Build
    scr = build_screener_table(universe_df, pm_data, merged_map)

    # Gap 10: record history, add Score Δ
    record_score_history(scr)
    scr["Score Δ"] = scr.apply(
        lambda r: get_score_delta(r["Ticker"], r.get("Score")), axis=1)

    # Filters
    filt = scr.copy()
    if sector_sel != "All Sectors":
        filt = filt[filt["Sector"] == sector_sel]
    filt = filt[(filt["Mkt Cap"].isna())       | (filt["Mkt Cap"]       >= mc_min_b * 1e9)]
    filt = filt[(filt["P/E"].isna())           | (filt["P/E"]           <= pe_max)]
    filt = filt[(filt["Quality Score"].isna()) | (filt["Quality Score"] >= qual_min_f)]

    sort_map = {
        "Sector then Rank":              (["Sector","Rank"],          [True,True]),
        "Score high to low":             (["Score"],                  [False]),
        "Conviction high to low":        (["Conviction Score"],       [False]),
        "CS Score high to low":          (["CS Score"],               [False]),
        "MC% of S&P500 high to low":     (["MC% of S&P500"],         [False]),
        "Price low to high":             (["Price"],                  [True]),
        "Price high to low":             (["Price"],                  [False]),
        "Mkt Cap high to low":           (["Mkt Cap"],                [False]),
        "PE low to high":                (["P/E"],                    [True]),
        "Fwd PE low to high":            (["Fwd P/E"],                [True]),
        "PEG low to high":               (["PEG"],                    [True]),
        "Quality Score high":            (["Quality Score"],          [False]),
        "ROIC high to low":              (["ROIC%"],                  [False]),
        "ROE high to low":               (["ROE%"],                   [False]),
        "Earn Traj high to low":         (["Earn Traj"],              [False]),
        "Rev Growth high to low":        (["Rev Growth% (CAGR)"],     [False]),
        "Momentum Score high":           (["Momentum Score"],         [False]),
        "52W Pos low to high":           (["52W Pos%"],               [True]),
        "P/E vs Sector Med low to high": (["P/E vs Sector Med"],      [True]),
        "Piotroski F high":              (["Piotroski F"],            [False]),
        "FCF Yield high":                (["FCF Yield%"],             [False]),
        "EV/EBITDA low":                 (["EV/EBITDA"],              [True]),
        "EPS Beat Rate high":            (["EPS Beat Rate"],          [False]),
        "Revision Mom high":             (["Revision Mom"],           [False]),
        "Score Delta high":              (["Score Δ"],                [False]),
    }
    sc, sa = sort_map.get(sort_by, (["Sector","Rank"], [True,True]))
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
            sloan_ratio=r.get("Sloan Ratio"), sector=r.get("Sector"),
        ), axis=1,
    )

    for c in [
        "P/E","Fwd P/E","PEG","Earn Traj","52W Pos%",
        "ROIC%","ROE%","Int Coverage","Op Margin%","Debt/Eq",
        "Quality Score","Momentum Score","Ret 1Mo%","Ret 3Mo%",
        "Ret 6Mo%","Trailing Vol%","Score","Conviction Score","CS Score",
        "Rev Growth% (CAGR)","P/E vs Sector Med",
        "EV/EBITDA","FCF Yield%","EV/Sales","Div Yield%","Sloan Ratio",
        "EPS Surp Avg%","EPS Beat Rate","EPS Surp Trend","Revision Mom",
        "Skip Mo","52W Prox","vs MA200","Rel Str SPY","Score Δ",
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
        "EPS Surp Avg%", "EPS Beat Rate", "EPS Surp Trend", "Revision Mom",
        "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
        "Quality Score", "Quality Flag", "Piotroski F", "Sloan Ratio",
        "Momentum Score", "Skip Mo", "52W Prox", "vs MA200", "Rel Str SPY",
        "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
        "52W Pos%", "Score", "Score Δ", "Conviction Score", "CS Score", "Rank",
        "Rev Q1 Oldest ($B)", "Rev Q2 ($B)", "Rev Q3 ($B)", "Rev Q4 Latest ($B)",
        "Rev Growth% (CAGR)",
    ]
    disp_final = disp[[c for c in COLS if c in disp.columns]].copy()
    st.dataframe(disp_final, use_container_width=True, height=680)

    st.download_button(
        label="Download CSV",
        data=disp_final.to_csv(index=False).encode("utf-8"),
        file_name="sp500_screener_v17_{}.csv".format(
            datetime.now().strftime("%Y%m%d_%H%M")),
        mime="text/csv",
    )

with page_reference:
    render_reference_guide()
