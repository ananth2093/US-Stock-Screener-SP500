# screener_app.py v19.1
# ─────────────────────────────────────────────────────────────────────────────
# v19.1 FIXES on top of v19:
#
#   FIX-MOM-1  fetch_prices_bulk: use group_by='column' (stable across all
#              yfinance versions) + correct access pattern raw['Close'][t]
#   FIX-MOM-2  Multi-level column detection with explicit version-safe fallback
#              chains: tries (metric,ticker), (ticker,metric), flat — in order
#   FIX-MOM-3  build_momentum_map wrapped in @st.cache_data so momentum
#              is not recomputed on every Streamlit rerun
#   FIX-MOM-4  fetch_prices_bulk now surfaces a visible debug counter so
#              empty-dict failures are immediately visible in the UI
#   FIX-MOM-5  Per-ticker fallback uses concurrent ThreadPool (not serial
#              loop) for the missing set — keeps fallback fast
#   FIX-MOM-6  prices_map keys normalised to uppercase to match universe_df
# ─────────────────────────────────────────────────────────────────────────────

import uuid
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

warnings.filterwarnings("ignore")

try:
    from bs4 import BeautifulSoup
except ImportError:
    st.error("pip install beautifulsoup4")
    st.stop()

# ── Constants ──────────────────────────────────────────────────────────────────
MIN_GROWTH_PCT_FOR_PEG   = 5.0
FETCH_TIMEOUT_PER_TICKER = 30
SLOAN_ACCRUALS_THRESHOLD = 0.08
CAGR_EXPONENT            = 4.0 / 3.0

YAHOO_DEEP_WORKERS = 4
YAHOO_INFO_WORKERS = 5
YAHOO_CHUNK_SIZE   = 15
YAHOO_SLEEP_BASE   = 2.5
MAX_RETRIES_YAHOO  = 3

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
QUALITY_THRESHOLDS  = {
    "roic_min":         8.0,
    "int_coverage_min": 3.0,
    "op_margin_min":    5.0,
}
OPERATING_CASH_PCT_OF_REV = 0.02

# ── Expanded row-name dictionaries ────────────────────────────────────────────
CASHFLOW_OCF_ROWS = [
    "Operating Cash Flow", "Cash From Operations",
    "Total Cash From Operating Activities",
    "Net Cash Provided By Operating Activities",
    "Net Cash From Operating Activities",
    "Cash Flows From Operating Activities",
    "Operating Activities", "Total Operating Cash Flow",
    "Net Operating Cash Flow",
]
CASHFLOW_CAPEX_ROWS = [
    "Capital Expenditure", "Purchase Of PPE", "Capital Expenditures",
    "Purchases Of Property Plant And Equipment",
    "Capital Expenditure Reported",
    "Acquisition Of Property Plant And Equipment",
    "Capital Lease Obligation", "Purchase Of Business",
]
CASHFLOW_DA_ROWS = [
    "Depreciation And Amortization", "Depreciation Amortization Depletion",
    "Reconciled Depreciation",
    "Depreciation And Amortization In Income Statement",
    "Depreciation", "Amortization", "Depreciation And Depletion",
    "Total Depreciation And Amortization Cash Flow", "D&A", "Dda",
]
INCOME_EBIT_ROWS = [
    "EBIT", "Operating Income", "Ebit",
    "Total Operating Income As Reported", "Operating Income Loss",
    "Operating Profit", "Earnings Before Interest And Taxes",
]
INCOME_INT_ROWS = [
    "Interest Expense", "Interest Expense Non Operating",
    "Net Interest Income", "Total Interest Expense",
    "Interest And Debt Expense", "Interest Expense Net", "Interest On Debt",
]
BS_SHARES_ROWS = [
    "Ordinary Shares Number", "Share Issued",
    "Common Stock Shares Outstanding", "Shares Outstanding",
    "Basic Shares Outstanding", "Common Stock",
]


# ── Credentials ────────────────────────────────────────────────────────────────
def get_fmp_key():
    try:
        k = st.secrets["fmp"]["api_key"]
        return k if k and k.strip() and k != "YOUR_KEY_HERE" else None
    except Exception:
        return None


# ── HTTP session with retry ────────────────────────────────────────────────────
def make_session():
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    })
    retry = Retry(
        total=3, backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://",  HTTPAdapter(max_retries=retry))
    return s


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

def normalise_pct_fmp(val):
    if val is None: return None
    return float(val) * 100.0

def winsorise(series: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
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

def elite_factor_score(series: pd.Series, ascending=True) -> pd.Series:
    if series.dropna().empty: return pd.Series(0.0, index=series.index)
    ws = winsorise(series.copy())
    if ascending: ws = -ws
    z = mad_zscore(ws).clip(-3.0, 3.0)
    z_min, z_max = z.min(), z.max()
    scaled = ((z - z_min) / (z_max - z_min) * 100.0
              if z_max > z_min else pd.Series(50.0, index=series.index))
    return scaled.fillna(0.0)

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
        return ((q4 / q1) ** CAGR_EXPONENT - 1.0) * 100.0
    except Exception:
        return None

def safe_round(series: pd.Series, decimals=2) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(decimals)

def _first(*vals):
    for v in vals:
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            return v
    return None

def _find_row(df_index, candidates):
    for name in candidates:
        if name in df_index: return name
    return None

def _row_sum_ttm(df, candidates, n=4):
    row = _find_row(df.index, candidates)
    if row is None: return None
    vals = df.loc[row].dropna().head(n)
    return float(vals.sum()) if not vals.empty else None

def _row_latest(df, candidates):
    row = _find_row(df.index, candidates)
    if row is None: return None
    vals = df.loc[row].dropna()
    return float(vals.iloc[0]) if not vals.empty else None


# ══════════════════════════════════════════════════════════════════════════════
# FIX-MOM-1/2/4/5/6 — Bulk price download (complete rewrite)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def fetch_prices_bulk(tickers):
    """
    FIX-MOM-1: Use group_by='column' — stable outer=metric, inner=ticker.
    FIX-MOM-2: Try three column-access patterns in priority order.
    FIX-MOM-4: Surface debug counter so failures are visible.
    FIX-MOM-5: Concurrent per-ticker fallback for any missing.
    FIX-MOM-6: All keys normalised to uppercase.

    Returns dict: ticker (str, uppercase) -> pd.Series of Close prices.
    """
    tl      = list(tickers)
    prices  = {}
    status  = st.empty()

    # ── Attempt 1: bulk download with group_by='column' ───────────────────
    # With group_by='column': columns are a MultiIndex (field, ticker)
    # Access pattern: raw['Close']['AAPL']
    status.text("Downloading bulk prices ({} tickers)...".format(len(tl)))
    try:
        raw = yf.download(
            tl,
            period="12mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
            threads=True,
            group_by="column",   # FIX-MOM-1: outer=field, inner=ticker
        )

        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                # ── Pattern A: (field, ticker) — standard group_by='column'
                if "Close" in raw.columns.get_level_values(0):
                    close_df = raw["Close"]
                    for t in tl:
                        tu = t.upper()
                        # ticker might appear as-is or URL-encoded
                        col = (tu if tu in close_df.columns
                               else t if t in close_df.columns else None)
                        if col is not None:
                            c = close_df[col].dropna()
                            if len(c) >= 20:
                                prices[tu] = c

                # ── Pattern B: (ticker, field) — some yf versions flip it
                elif "Close" in raw.columns.get_level_values(1):
                    for t in tl:
                        tu = t.upper()
                        try:
                            c = raw.xs("Close", axis=1, level=1)[tu].dropna()
                            if len(c) >= 20:
                                prices[tu] = c
                        except Exception:
                            pass

            else:
                # Single ticker or flat columns
                if "Close" in raw.columns:
                    if len(tl) == 1:
                        c = raw["Close"].dropna()
                        if len(c) >= 20:
                            prices[tl[0].upper()] = c
                    else:
                        # Flat columns = all tickers in one level (unusual)
                        for t in tl:
                            tu = t.upper()
                            if tu in raw.columns:
                                c = raw[tu].dropna()
                                if len(c) >= 20:
                                    prices[tu] = c

    except Exception as e:
        status.text("Bulk download error: {} — falling back to per-ticker.".format(
            str(e)[:80]))

    bulk_count = len(prices)
    status.text("Bulk prices: {}/{} tickers. Fetching {} missing...".format(
        bulk_count, len(tl), len(tl) - bulk_count))

    # ── Attempt 2: concurrent per-ticker fallback for missing ────────────
    missing = [t for t in tl if t.upper() not in prices]

    def _fetch_one(t):
        tu = t.upper()
        for attempt in range(2):
            try:
                obj  = yf.Ticker(t)
                hist = obj.history(period="12mo", interval="1d", auto_adjust=True)
                if (hist is not None and not hist.empty
                        and "Close" in hist.columns):
                    c = hist["Close"].dropna()
                    if len(c) >= 20:
                        return tu, c
            except Exception:
                pass
            time.sleep(0.5 * (attempt + 1) + random.uniform(0, 0.3))
        return tu, None

    if missing:
        CHUNK = 30; WKRS = 8
        chunks = [missing[i:i+CHUNK] for i in range(0, len(missing), CHUNK)]
        prog   = st.progress(0)
        for ci, chunk in enumerate(chunks):
            status.text("Per-ticker fallback: {}/{} ({} done)...".format(
                ci + 1, len(chunks), ci * CHUNK))
            with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
                futs = {ex.submit(_fetch_one, t): t for t in chunk}
                for fut in concurrent.futures.as_completed(
                        futs, timeout=FETCH_TIMEOUT_PER_TICKER * len(chunk)):
                    try:
                        tu, c = fut.result()
                        if c is not None:
                            prices[tu] = c
                    except Exception:
                        pass
            prog.progress((ci + 1) / len(chunks))
            if ci < len(chunks) - 1:
                time.sleep(1.0 + random.uniform(0, 0.5))
        prog.empty()

    status.empty()
    return prices   # {TICKER_UPPER: pd.Series}


# ══════════════════════════════════════════════════════════════════════════════
# FIX-MOM-3: cache build_momentum_map
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def fetch_spy_3mo_return():
    try:
        hist = yf.Ticker("SPY").history(
            period="12mo", interval="1d", auto_adjust=True)
        if hist is None or hist.empty or "Close" not in hist.columns:
            return None
        closes  = hist["Close"].dropna()
        monthly = closes.resample("ME").last().dropna()
        if len(monthly) < 4: return None
        px_now = float(monthly.iloc[-1])
        px_3m  = float(monthly.iloc[-4])
        return (px_now / px_3m - 1) * 100.0 if px_3m > 0 else None
    except Exception:
        return None


def compute_elite_momentum(closes: pd.Series, price: float,
                           hi52: float, spy_3mo=None) -> tuple:
    monthly = closes.resample("ME").last().dropna()
    comps = {
        "skip_month_raw": None, "hi52_proximity": None,
        "vs_ma200": None, "rel_strength_spy": None,
    }
    if len(monthly) < 2 or price is None or price <= 0:
        return None, comps
    px_now = float(monthly.iloc[-1])
    if px_now <= 0: return None, comps

    def ret_mo(n):
        idx = -(n + 1)
        if abs(idx) > len(monthly): return None
        px = float(monthly.iloc[idx])
        return (px_now / px - 1) * 100.0 if px > 0 else None

    r1, r3, r6 = ret_mo(1), ret_mo(3), ret_mo(6)

    s1 = None
    dr = closes.pct_change().dropna().tail(90)
    t_vol = float(dr.std() * np.sqrt(252) * 100.0) if len(dr) >= 15 else None
    if r6 is not None and r1 is not None:
        skip_raw = r6 - r1
        raw_s    = skip_raw / t_vol if (t_vol and t_vol > 0) else skip_raw
        s1 = float(np.clip(raw_s / 2.0, -1.0, 1.0))
    comps["skip_month_raw"] = s1

    s2 = None
    if hi52 and hi52 > 0:
        s2 = float(np.clip(1.0 + (price - hi52) / hi52 / 0.30, 0.0, 1.0))
    comps["hi52_proximity"] = s2

    s3 = None
    n_closes = len(closes)
    if n_closes >= 200:
        ma = float(closes.tail(200).mean())
        if ma > 0: s3 = float(np.clip((price - ma) / ma / 0.30, -1.0, 1.0))
    elif n_closes >= 50:
        ma = float(closes.tail(50).mean())
        if ma > 0: s3 = float(np.clip((price - ma) / ma / 0.20, -1.0, 1.0))
    comps["vs_ma200"] = s3

    s4 = None
    if spy_3mo is not None and r3 is not None:
        s4 = float(np.clip((r3 - spy_3mo) / 20.0, -1.0, 1.0))
    comps["rel_strength_spy"] = s4

    signal_weights = [
        ("skip_month_raw",   0.40), ("hi52_proximity",   0.25),
        ("vs_ma200",         0.20), ("rel_strength_spy", 0.15),
    ]
    total_w = 0.0; composite = 0.0
    for key, w in signal_weights:
        val = comps.get(key)
        if val is not None:
            composite += val * w; total_w += w

    return (float(composite / total_w) if total_w > 0 else None), comps


# FIX-MOM-3: wrap in cache so it isn't recomputed on every Streamlit rerun
@st.cache_data(ttl=3600)
def build_momentum_map(prices_map: dict, spy_3mo=None) -> dict:
    """
    Compute all momentum metrics from pre-fetched price Series.
    prices_map keys must be uppercase ticker strings.
    """
    out = {}
    for t, closes in prices_map.items():
        result = {
            "price": None, "hi52": None, "lo52": None,
            "ret_1mo": None, "ret_3mo": None, "ret_6mo": None,
            "trailing_vol": None, "momentum_score": None,
            "skip_month_raw": None, "hi52_proximity": None,
            "vs_ma200": None, "rel_strength_spy": None,
        }
        try:
            price_val = float(closes.iloc[-1])
            hi52_val  = float(closes.max())
            lo52_val  = float(closes.min())
            result.update({
                "price": price_val, "hi52": hi52_val, "lo52": lo52_val
            })

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

            dr = closes.pct_change().dropna().tail(90)
            if len(dr) >= 15:
                result["trailing_vol"] = float(dr.std() * np.sqrt(252) * 100.0)

            composite, comps = compute_elite_momentum(
                closes, price_val, hi52_val, spy_3mo)
            result["momentum_score"]   = composite
            result["skip_month_raw"]   = comps.get("skip_month_raw")
            result["hi52_proximity"]   = comps.get("hi52_proximity")
            result["vs_ma200"]         = comps.get("vs_ma200")
            result["rel_strength_spy"] = comps.get("rel_strength_spy")
        except Exception:
            pass
        out[t] = result

    return out


# ══════════════════════════════════════════════════════════════════════════════
# FMP BULK FETCHES
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400)
def fetch_fmp_bulk_quotes(tickers, api_key):
    out = {}
    if not api_key: return out
    tl = list(tickers); sess = make_session()
    for chunk in [tl[i:i+200] for i in range(0, len(tl), 200)]:
        url = ("https://financialmodelingprep.com/api/v3/quote/{}?apikey={}".format(
            ",".join(chunk), api_key))
        try:
            r = sess.get(url, timeout=25); r.raise_for_status()
            data = r.json()
            if not isinstance(data, list): continue
            for item in data:
                t  = str(item.get("symbol", "")).upper().strip()
                if not t: continue
                pe = sf(item.get("pe")); mc = sf(item.get("marketCap"))
                px = sf(item.get("price")); eps = sf(item.get("eps"))
                if pe is not None and (pe <= 0 or pe > 10_000): pe = None
                out[t] = {"pe": pe, "mc": mc, "price": px, "eps": eps,
                          "pe_src": "FMP-quote" if pe is not None else None}
        except Exception:
            pass
        time.sleep(0.3)
    return out


@st.cache_data(ttl=86400)
def fetch_fmp_bulk_key_metrics(tickers, api_key):
    out = {}
    if not api_key: return out
    tl = list(tickers); sess = make_session()

    def fetch_one(t):
        url = ("https://financialmodelingprep.com/api/v3/key-metrics-ttm/"
               "{}?apikey={}".format(t, api_key))
        for attempt in range(MAX_RETRIES_YAHOO):
            try:
                r = sess.get(url, timeout=15)
                if r.status_code == 429:
                    time.sleep(3.0 * (attempt + 1)); continue
                r.raise_for_status()
                d = r.json()
                if not isinstance(d, list) or len(d) == 0: return t, {}
                item = d[0]
                def _g(k): return sf(item.get(k))
                mc_val   = _g("marketCapTTM")
                ev_val   = _g("enterpriseValueTTM")
                evebitda = _g("evToEbitdaTTM") or _g("enterpriseValueMultipleTTM")
                evsales  = _g("evToSalesTTM")  or _g("evToRevenueTTM")
                fcf_yld  = _g("freeCashFlowYieldTTM")
                fcf_ps   = _g("freeCashFlowPerShareTTM")
                shares   = _g("weightedAverageSharesOutstanding") or _g("sharesOutstanding")
                roic     = _g("roicTTM"); roe = _g("roeTTM")
                pe_ttm   = _g("peRatioTTM"); peg = _g("pegRatioTTM") or _g("priceEarningsToGrowthRatioTTM")
                int_cov  = _g("interestCoverageTTM")
                gm       = _g("grossProfitMarginTTM"); om = _g("operatingProfitMarginTTM")
                de       = _g("debtToEquityTTM"); cr = _g("currentRatioTTM")
                if roic is not None: roic = normalise_pct_fmp(roic)
                if roe  is not None: roe  = normalise_pct_fmp(roe)
                if gm   is not None: gm   = normalise_pct_fmp(gm)
                if om   is not None: om   = normalise_pct_fmp(om)
                if pe_ttm and (pe_ttm <= 0 or pe_ttm > 10_000): pe_ttm = None
                if evebitda and (evebitda <= 0 or evebitda > 200): evebitda = None
                if peg and (peg <= 0 or peg > 500): peg = None
                if int_cov: int_cov = min(float(int_cov), 100.0) if int_cov > 0 else None
                fcf_ttm = None
                if fcf_yld and mc_val and mc_val > 0:
                    fcf_ttm = fcf_yld * mc_val
                elif fcf_ps and shares and shares > 0:
                    fcf_ttm = fcf_ps * shares
                return t, {
                    "mc": mc_val, "ev_ebitda": evebitda, "ev_sales": evsales,
                    "fcf_yield": (fcf_yld * 100.0 if fcf_yld else None),
                    "fcf_ttm": fcf_ttm,
                    "roic": roic, "roe": roe, "pe": pe_ttm,
                    "pe_src": "FMP-km" if pe_ttm else None,
                    "peg": peg, "peg_src": "FMP-km" if peg else None,
                    "int_coverage": int_cov, "gross_margin": gm, "op_margin": om,
                    "debt_eq": de, "current_ratio": cr,
                }
            except Exception:
                time.sleep(1.0)
        return t, {}

    CHUNK = 50; WKRS = 15
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog = st.progress(0); status = st.empty()
    for ci, chunk in enumerate(chunks):
        status.text("FMP key-metrics: {}/{} ({} done)...".format(
            ci + 1, len(chunks), ci * CHUNK))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futs = {ex.submit(fetch_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futs, timeout=60):
                try:
                    t, d = fut.result()
                    if d: out[t] = d
                except Exception:
                    pass
        prog.progress((ci + 1) / len(chunks))
        time.sleep(0.5)
    prog.empty(); status.empty()
    return out


@st.cache_data(ttl=86400)
def fetch_fmp_bulk_ratios(tickers, api_key):
    out = {}
    if not api_key: return out
    tl = list(tickers); sess = make_session()

    def fetch_one(t):
        url = ("https://financialmodelingprep.com/api/v3/ratios-ttm/"
               "{}?apikey={}".format(t, api_key))
        try:
            r = sess.get(url, timeout=12)
            if r.status_code == 429: time.sleep(3.0); r = sess.get(url, timeout=12)
            r.raise_for_status()
            d = r.json()
            if not isinstance(d, list) or len(d) == 0: return t, {}
            item   = d[0]
            peg_r  = sf(item.get("priceEarningsGrowthRatioTTM"))
            peg    = peg_r if (peg_r and 0 < peg_r <= 500) else None
            roic_r = sf(item.get("returnOnInvestedCapitalTTM"))
            roic   = normalise_pct_fmp(roic_r) if roic_r is not None else None
            roe_r  = sf(item.get("returnOnEquityTTM"))
            roe    = normalise_pct_fmp(roe_r) if roe_r is not None else None
            om_r   = sf(item.get("operatingProfitMarginTTM"))
            om     = normalise_pct_fmp(om_r) if om_r is not None else None
            ic_r   = sf(item.get("interestCoverageTTM"))
            ic     = min(float(ic_r), 100.0) if (ic_r and ic_r > 0) else None
            de     = sf(item.get("debtEquityRatioTTM"))
            fmp_pe = sf(item.get("priceToEarningsRatioTTM"))
            if fmp_pe and (fmp_pe <= 0 or fmp_pe > 10_000): fmp_pe = None
            return t, {
                "peg": peg, "roic": roic, "roe": roe, "op_margin": om,
                "int_coverage": ic, "debt_eq": de, "fmp_trailing_pe": fmp_pe,
                "peg_src": "FMP-ratios" if peg else None,
            }
        except Exception:
            return t, {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as ex:
        futs = {ex.submit(fetch_one, t): t for t in tl}
        for fut in concurrent.futures.as_completed(futs):
            try:
                t, d = fut.result()
                if d: out[t] = d
            except Exception:
                pass
    return out


@st.cache_data(ttl=86400)
def fetch_fmp_income_statements(tickers, api_key):
    out = {}
    if not api_key: return out
    tl = list(tickers); sess = make_session()

    def fetch_one(t):
        url = ("https://financialmodelingprep.com/api/v3/income-statement/"
               "{}?period=quarter&limit=8&apikey={}".format(t, api_key))
        try:
            r = sess.get(url, timeout=15)
            if r.status_code == 429: time.sleep(2.0); return t, {}
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) == 0: return t, {}
            revs     = [sf(d.get("revenue"))          for d in data]
            eps_arr  = [sf(d.get("eps"))               for d in data]
            gp_arr   = [sf(d.get("grossProfit"))       for d in data]
            ni_arr   = [sf(d.get("netIncome"))         for d in data]
            ebit_arr = [sf(d.get("operatingIncome"))   for d in data]
            int_arr  = [sf(d.get("interestExpense"))   for d in data]
            rev4 = revs[:4][::-1]
            if len(rev4) < 4: rev4 = ([None]*(4-len(rev4))) + rev4
            ni_ttm   = sum(v for v in ni_arr[:4]   if v is not None) or None
            ebit_ttm = sum(v for v in ebit_arr[:4] if v is not None) or None
            int_ttm  = sum(abs(v) for v in int_arr[:4] if v is not None) or None
            int_cov  = None
            if ebit_ttm and int_ttm and int_ttm > 0 and ebit_ttm > 0:
                int_cov = min(ebit_ttm / int_ttm, 100.0)
            eps_growth = None
            valid_eps  = [(i, v) for i, v in enumerate(eps_arr) if v is not None]
            if len(valid_eps) >= 6:
                e_new = valid_eps[0][1]; e_old = valid_eps[5][1]
                if e_old > 0 and e_new > 0:
                    eps_growth = ((e_new / e_old) ** 2.0 - 1.0) * 100.0
            gm_now = gm_prev = None
            rev_ttm = sum(v for v in revs[:4] if v is not None)
            gp_ttm  = sum(v for v in gp_arr[:4] if v is not None)
            if rev_ttm > 0 and gp_ttm > 0: gm_now = gp_ttm / rev_ttm * 100.0
            rev_prev = sum(v for v in revs[4:8] if v is not None)
            gp_prev2 = sum(v for v in gp_arr[4:8] if v is not None)
            if rev_prev > 0 and gp_prev2 > 0: gm_prev = gp_prev2 / rev_prev * 100.0
            return t, {
                "rev4": rev4, "net_income_ttm": ni_ttm,
                "gross_margin_now": gm_now, "gross_margin_prev": gm_prev,
                "int_coverage": int_cov, "eps_growth": eps_growth,
                "growth_src": "FMP-IS-2yr" if eps_growth is not None else None,
            }
        except Exception:
            return t, {}

    CHUNK = 50; WKRS = 15
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog = st.progress(0); status = st.empty()
    for ci, chunk in enumerate(chunks):
        status.text("FMP income-stmt: {}/{} ({} done)...".format(
            ci+1, len(chunks), ci*CHUNK))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futs = {ex.submit(fetch_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futs, timeout=90):
                try:
                    t, d = fut.result()
                    if d: out[t] = d
                except Exception:
                    pass
        prog.progress((ci+1)/len(chunks)); time.sleep(0.5)
    prog.empty(); status.empty()
    return out


@st.cache_data(ttl=86400)
def fetch_fmp_cashflow_statements(tickers, api_key):
    out = {}
    if not api_key: return out
    tl = list(tickers); sess = make_session()

    def fetch_one(t):
        url = ("https://financialmodelingprep.com/api/v3/cash-flow-statement/"
               "{}?period=quarter&limit=5&apikey={}".format(t, api_key))
        try:
            r = sess.get(url, timeout=15)
            if r.status_code == 429: time.sleep(2.0); return t, {}
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) == 0: return t, {}
            ocf_arr   = [sf(d.get("operatingCashFlow"))          for d in data]
            capex_arr = [sf(d.get("capitalExpenditure"))         for d in data]
            da_arr    = [sf(d.get("depreciationAndAmortization")) for d in data]
            def ttm(arr):
                vals = [v for v in arr[:4] if v is not None]
                return sum(vals) if vals else None
            ocf_ttm  = ttm(ocf_arr)
            capex_ttm = ttm([abs(v) if v is not None else None for v in capex_arr])
            da_ttm   = ttm([abs(v) if v is not None else None for v in da_arr])
            fcf_ttm  = (ocf_ttm - capex_ttm
                        if ocf_ttm is not None and capex_ttm is not None
                        else None)
            return t, {"ocf_ttm": ocf_ttm, "fcf_ttm": fcf_ttm, "da_ttm": da_ttm}
        except Exception:
            return t, {}

    CHUNK = 50; WKRS = 15
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog = st.progress(0); status = st.empty()
    for ci, chunk in enumerate(chunks):
        status.text("FMP cashflow: {}/{} ({} done)...".format(
            ci+1, len(chunks), ci*CHUNK))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futs = {ex.submit(fetch_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futs, timeout=90):
                try:
                    t, d = fut.result()
                    if d: out[t] = d
                except Exception:
                    pass
        prog.progress((ci+1)/len(chunks)); time.sleep(0.5)
    prog.empty(); status.empty()
    return out


@st.cache_data(ttl=86400)
def fetch_fmp_balance_sheets(tickers, api_key):
    out = {}
    if not api_key: return out
    tl = list(tickers); sess = make_session()

    def fetch_one(t):
        url = ("https://financialmodelingprep.com/api/v3/balance-sheet-statement/"
               "{}?period=quarter&limit=9&apikey={}".format(t, api_key))
        try:
            r = sess.get(url, timeout=15)
            if r.status_code == 429: time.sleep(2.0); return t, {}
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) == 0: return t, {}
            def gv(d, *keys):
                for k in keys:
                    v = sf(d.get(k))
                    if v is not None: return v
                return None
            d0 = data[0]; d4 = data[4] if len(data) > 4 else {}
            ta_now   = gv(d0, "totalAssets")
            ta_prev  = gv(d4, "totalAssets")
            ltd_now  = gv(d0, "longTermDebt", "longTermDebtAndCapitalLeaseObligation")
            ltd_prev = gv(d4, "longTermDebt", "longTermDebtAndCapitalLeaseObligation")
            ca_now   = gv(d0, "totalCurrentAssets")
            cl_now   = gv(d0, "totalCurrentLiabilities")
            ca_prev  = gv(d4, "totalCurrentAssets")
            cl_prev  = gv(d4, "totalCurrentLiabilities")
            sh_now   = gv(d0, "commonStock", "sharesOutstanding", "weightedAverageShsOut")
            sh_prev  = gv(d4, "commonStock", "sharesOutstanding", "weightedAverageShsOut")
            return t, {
                "total_assets_now":   ta_now,
                "total_assets_prev":  ta_prev,
                "lt_debt_ratio_now":  (ltd_now  / ta_now)  if ltd_now  and ta_now  else None,
                "lt_debt_ratio_prev": (ltd_prev / ta_prev) if ltd_prev and ta_prev else None,
                "current_ratio_now":  (ca_now / cl_now)    if ca_now   and cl_now  else None,
                "current_ratio_prev": (ca_prev / cl_prev)  if ca_prev  and cl_prev else None,
                "shares_now":         sh_now,
                "shares_prev":        sh_prev,
            }
        except Exception:
            return t, {}

    CHUNK = 50; WKRS = 15
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog = st.progress(0); status = st.empty()
    for ci, chunk in enumerate(chunks):
        status.text("FMP balance-sheet: {}/{} ({} done)...".format(
            ci+1, len(chunks), ci*CHUNK))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futs = {ex.submit(fetch_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futs, timeout=90):
                try:
                    t, d = fut.result()
                    if d: out[t] = d
                except Exception:
                    pass
        prog.progress((ci+1)/len(chunks)); time.sleep(0.5)
    prog.empty(); status.empty()
    return out


@st.cache_data(ttl=86400)
def fetch_fmp_earnings_surprises(tickers, api_key):
    out = {}
    if not api_key: return out
    tl = list(tickers); sess = make_session()

    def fetch_one(t):
        url = ("https://financialmodelingprep.com/api/v3/earnings-surprises/"
               "{}?apikey={}".format(t, api_key))
        try:
            r = sess.get(url, timeout=12)
            if r.status_code == 429: time.sleep(2.0); return t, {}
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) == 0: return t, {}
            recent = data[:4]; surprises = []
            for d in recent:
                actual   = sf(d.get("actualEarningResult"))
                estimate = sf(d.get("estimatedEarning"))
                if actual and estimate and abs(estimate) > 0.001:
                    surprises.append((actual - estimate) / abs(estimate) * 100.0)
            if not surprises: return t, {}
            avg = float(np.mean(surprises))
            br  = float(sum(1 for s in surprises if s > 0) / len(surprises))
            trend = None
            if len(surprises) >= 3:
                trend = 1.0 if np.mean(surprises[:2]) > np.mean(surprises[2:]) else -1.0
            return t, {"eps_surprise_avg": avg, "eps_beat_rate": br, "eps_surprise_trend": trend}
        except Exception:
            return t, {}

    CHUNK = 50; WKRS = 15
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog = st.progress(0); status = st.empty()
    for ci, chunk in enumerate(chunks):
        status.text("FMP earnings-surprises: {}/{} ({} done)...".format(
            ci+1, len(chunks), ci*CHUNK))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futs = {ex.submit(fetch_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futs, timeout=90):
                try:
                    t, d = fut.result()
                    if d: out[t] = d
                except Exception:
                    pass
        prog.progress((ci+1)/len(chunks)); time.sleep(0.5)
    prog.empty(); status.empty()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Yahoo fill for Fwd P/E + Earn Traj (not available in FMP)
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_yahoo_fill_one(t):
    result = {
        "pe": None, "pe_src": None, "fwd_pe": None,
        "peg": None, "peg_src": None, "roe": None,
        "op_margin": None, "debt_eq": None,
        "eps_growth": None, "growth_src": None, "earn_traj": None,
        "mc": None, "roic": None, "int_coverage": None,
        "rev4": [None]*4, "ev_ebitda": None, "ev_sales": None,
        "div_yield": None, "ev_raw": None,
        "eps_surprise_avg": None, "eps_beat_rate": None,
        "eps_surprise_trend": None, "revision_momentum": None,
    }
    for attempt in range(MAX_RETRIES_YAHOO):
        try:
            obj = yf.Ticker(t); info = obj.info or {}
            if not info: raise ValueError("empty info")

            px    = sf(info.get("currentPrice") or info.get("regularMarketPrice"))
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
            om_y  = sf(info.get("operatingMargins"))
            if om_y is not None: result["op_margin"] = om_y * 100.0
            de_y  = sf(info.get("debtToEquity"))
            if de_y is not None: result["debt_eq"] = de_y / 100.0

            eg_y = sf(info.get("earningsGrowth"))
            if eg_y is not None:
                result["eps_growth"] = eg_y * 100.0
                result["growth_src"] = "Yahoo-fwd"

            if result["eps_growth"] is None:
                try:
                    eh = obj.earnings
                    if eh is not None and not eh.empty:
                        col = next((c for c in ["Earnings","EPS","Net Income"]
                                    if c in eh.columns), None)
                        if col:
                            ev = eh[col].dropna()
                            if len(ev) >= 3:
                                e1, e2 = float(ev.iloc[-3]), float(ev.iloc[-1])
                                if e1 > 0 and e2 > 0:
                                    result["eps_growth"] = ((e2/e1)**0.5 - 1)*100.0
                                    result["growth_src"] = "Yahoo-3yr-CAGR"
                except Exception:
                    pass

            fwd_eps = sf(info.get("forwardEps"))
            tr_eps  = sf(info.get("trailingEps"))
            if fwd_eps and tr_eps and abs(tr_eps) > 0.01:
                et = (fwd_eps - tr_eps) / abs(tr_eps)
                clipped = max(-1.0, min(1.0, et))
                if tr_eps < 0 and fwd_eps < 0: clipped = min(clipped, 0.30)
                result["earn_traj"] = clipped

            if result["mc"] is None:
                mc_y = sf(info.get("marketCap"))
                if mc_y: result["mc"] = mc_y

            ev_raw = sf(info.get("enterpriseValue"))
            ebitda = sf(info.get("ebitda"))
            if ev_raw and ebitda and ebitda > 0:
                ev_eb = ev_raw / ebitda
                if 0 < ev_eb < 200: result["ev_ebitda"] = ev_eb
            rev_ttm = sf(info.get("totalRevenue"))
            if ev_raw and rev_ttm and rev_ttm > 0:
                ev_s = ev_raw / rev_ttm
                if 0 < ev_s < 100: result["ev_sales"] = ev_s
            result["ev_raw"] = ev_raw

            dy = sf(info.get("dividendYield"))
            if dy is not None:
                result["div_yield"] = (dy * 100.0 if abs(dy) < 1.0
                                       else dy if abs(dy) <= 100.0 else None)

            try:
                rec = obj.recommendations_summary
                if rec is not None and not rec.empty and len(rec) >= 2:
                    latest, prior = rec.iloc[0], rec.iloc[1]
                    sb_chg   = (float(latest.get("strongBuy",0) or 0) +
                                float(latest.get("buy",0) or 0) -
                                float(prior.get("strongBuy",0) or 0) -
                                float(prior.get("buy",0) or 0))
                    sell_chg = (float(latest.get("sell",0) or 0) +
                                float(latest.get("strongSell",0) or 0) -
                                float(prior.get("sell",0) or 0) -
                                float(prior.get("strongSell",0) or 0))
                    total = abs(sb_chg) + abs(sell_chg)
                    result["revision_momentum"] = (
                        float(np.clip((sb_chg-sell_chg)/total,-1.0,1.0))
                        if total > 0 else 0.0)
            except Exception:
                pass

            try:
                eh2 = obj.earnings_history
                if eh2 is not None and not eh2.empty:
                    eh2 = eh2.dropna(subset=["epsActual","epsEstimate"]).copy()
                    if len(eh2) > 0:
                        eh2["sp"] = ((eh2["epsActual"]-eh2["epsEstimate"])
                                     / eh2["epsEstimate"].abs()*100.0)
                        eh2 = eh2.tail(4)
                        result["eps_surprise_avg"]   = float(eh2["sp"].mean())
                        result["eps_beat_rate"]      = float((eh2["sp"]>0).mean())
                        if len(eh2) >= 3:
                            result["eps_surprise_trend"] = (
                                1.0 if float(eh2["sp"].tail(2).mean()) >
                                       float(eh2["sp"].head(2).mean()) else -1.0)
            except Exception:
                pass

            return result
        except Exception:
            time.sleep(1.5 * (attempt + 1) + random.uniform(0, 1.0))
    return result


@st.cache_data(ttl=86400)
def fetch_yahoo_fills(tickers_needing_fill, _cache_date=None):
    tl = list(tickers_needing_fill); out = {}
    if not tl: return out
    CHUNK = YAHOO_CHUNK_SIZE; WKRS = YAHOO_INFO_WORKERS
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog = st.progress(0); status = st.empty()
    for ci, chunk in enumerate(chunks):
        status.text("Yahoo fill: {}/{} ({}/{} tickers)...".format(
            ci+1, len(chunks), min((ci+1)*CHUNK, len(tl)), len(tl)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futs = {ex.submit(_fetch_yahoo_fill_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(
                    futs, timeout=FETCH_TIMEOUT_PER_TICKER*len(chunk)):
                try:
                    t = futs[fut]; d = fut.result(); out[t] = d
                except Exception:
                    out[futs[fut]] = {}
        prog.progress((ci+1)/len(chunks))
        if ci < len(chunks)-1:
            time.sleep(YAHOO_SLEEP_BASE + random.uniform(0, 1.0))
    prog.empty(); status.empty()
    return out


@st.cache_data(ttl=86400)
def fetch_yahoo_deep_fills(tickers_needing_deep, _cache_date=None):
    tl = list(tickers_needing_deep); out = {}
    if not tl: return out
    CHUNK = 10; WKRS = YAHOO_DEEP_WORKERS

    def fetch_one(t):
        r = {
            "ocf_ttm": None, "fcf_ttm": None, "net_income_ttm": None,
            "gross_margin_now": None, "gross_margin_prev": None,
            "roa_ttm": None, "roa_prev": None,
            "total_assets_now": None, "total_assets_prev": None,
            "lt_debt_ratio_now": None, "lt_debt_ratio_prev": None,
            "current_ratio_now": None, "current_ratio_prev": None,
            "shares_now": None, "shares_prev": None,
            "ebitda_ttm": None, "rev4": [None]*4,
            "roic": None, "int_coverage": None,
        }
        for attempt in range(MAX_RETRIES_YAHOO):
            try:
                obj  = yf.Ticker(t)
                qfin = obj.quarterly_financials
                qbs  = obj.quarterly_balance_sheet
                qcf  = obj.quarterly_cashflow
                info = obj.info or {}

                ebit_ttm = None
                if qfin is not None and not qfin.empty:
                    rev_row = _find_row(qfin.index, ["Total Revenue","Revenue"])
                    if rev_row:
                        rs = qfin.loc[rev_row].sort_index().dropna().tail(4)
                        vals = [float(x) for x in rs.values]
                        r["rev4"] = ([None]*(4-len(vals))+vals
                                     if len(vals) < 4 else vals)
                    ebit_ttm = _row_sum_ttm(qfin, INCOME_EBIT_ROWS)
                    int_ttm  = abs(_row_sum_ttm(qfin, INCOME_INT_ROWS) or 0)
                    if ebit_ttm and int_ttm > 0 and ebit_ttm > 0:
                        r["int_coverage"] = min(ebit_ttm / int_ttm, 100.0)
                    ni_ttm = _row_sum_ttm(qfin, ["Net Income",
                                                  "Net Income Common Stockholders"])
                    r["net_income_ttm"] = ni_ttm
                    gp_row = _find_row(qfin.index, ["Gross Profit","Gross Income"])
                    rv_row = _find_row(qfin.index, ["Total Revenue","Revenue"])
                    if gp_row and rv_row:
                        gp_t = float(qfin.loc[gp_row].dropna().head(4).sum())
                        rv_t = float(qfin.loc[rv_row].dropna().head(4).sum())
                        if rv_t > 0: r["gross_margin_now"] = gp_t / rv_t * 100.0
                        gp_a = qfin.loc[gp_row].dropna()
                        rv_a = qfin.loc[rv_row].dropna()
                        if len(rv_a) >= 8 and len(gp_a) >= 8:
                            rp = float(rv_a.iloc[4:8].sum())
                            gp = float(gp_a.iloc[4:8].sum())
                            if rp > 0: r["gross_margin_prev"] = gp / rp * 100.0

                if qcf is not None and not qcf.empty:
                    ocf_ttm   = _row_sum_ttm(qcf, CASHFLOW_OCF_ROWS)
                    capex_val = _row_sum_ttm(qcf, CASHFLOW_CAPEX_ROWS)
                    da_ttm    = abs(_row_sum_ttm(qcf, CASHFLOW_DA_ROWS) or 0) or None
                    r["ocf_ttm"] = ocf_ttm
                    if ocf_ttm and capex_val:
                        r["fcf_ttm"] = ocf_ttm - abs(capex_val)
                    if ebit_ttm and da_ttm:
                        r["ebitda_ttm"] = ebit_ttm + da_ttm

                if qbs is not None and not qbs.empty:
                    ta_row = _find_row(qbs.index, ["Total Assets","Assets"])
                    if ta_row:
                        ta_vals = qbs.loc[ta_row].dropna()
                        if len(ta_vals) >= 1:
                            ta_now = float(ta_vals.iloc[0])
                            r["total_assets_now"] = ta_now
                            if len(ta_vals) >= 5:
                                r["total_assets_prev"] = float(ta_vals.iloc[4])
                    sh_row = _find_row(qbs.index, BS_SHARES_ROWS)
                    if sh_row:
                        sv = qbs.loc[sh_row].dropna()
                        if len(sv) >= 1: r["shares_now"]  = float(sv.iloc[0])
                        if len(sv) >= 5: r["shares_prev"] = float(sv.iloc[4])

                if r["shares_now"] is None:
                    r["shares_now"] = sf(info.get("sharesOutstanding"))
                if r["shares_prev"] is None:
                    r["shares_prev"] = r["shares_now"]
                return r
            except Exception:
                time.sleep(2.0 * (attempt + 1))
        return r

    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    prog = st.progress(0); status = st.empty()
    for ci, chunk in enumerate(chunks):
        status.text("Yahoo deep fill: {}/{} ({}/{})...".format(
            ci+1, len(chunks), min((ci+1)*CHUNK, len(tl)), len(tl)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futs = {ex.submit(fetch_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(
                    futs, timeout=FETCH_TIMEOUT_PER_TICKER*len(chunk)):
                try:
                    t = futs[fut]; d = fut.result(); out[t] = d
                except Exception:
                    out[futs[fut]] = {}
        prog.progress((ci+1)/len(chunks))
        if ci < len(chunks)-1:
            time.sleep(YAHOO_SLEEP_BASE + random.uniform(0, 1.5))
    prog.empty(); status.empty()
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Master merge
# ══════════════════════════════════════════════════════════════════════════════
def build_master_data(
    tickers, fmp_quotes, fmp_key_metrics, fmp_ratios,
    fmp_income, fmp_cashflow, fmp_balance, fmp_surprises,
    yahoo_fills, yahoo_deep_fills,
):
    merged = {}
    for t in tickers:
        fq  = fmp_quotes.get(t, {})
        fkm = fmp_key_metrics.get(t, {})
        fr  = fmp_ratios.get(t, {})
        fi  = fmp_income.get(t, {})
        fc  = fmp_cashflow.get(t, {})
        fb  = fmp_balance.get(t, {})
        fs  = fmp_surprises.get(t, {})
        yf_ = yahoo_fills.get(t, {})
        yd  = yahoo_deep_fills.get(t, {})

        pe_val = _first(fkm.get("pe"), fq.get("pe"),
                        fr.get("fmp_trailing_pe"), yf_.get("pe"))
        pe_src = (
            "FMP-km"     if fkm.get("pe")             is not None else
            "FMP-quote"  if fq.get("pe")              is not None else
            "FMP-ratios" if fr.get("fmp_trailing_pe") is not None else
            yf_.get("pe_src") or "Yahoo"
        )
        fwd_pe = _first(yf_.get("fwd_pe"))
        mc_val = _first(fkm.get("mc"), fq.get("mc"), yf_.get("mc"))
        peg_v  = _first(fkm.get("peg"), fr.get("peg"), yf_.get("peg"))
        peg_src= (
            "FMP-km"     if fkm.get("peg") is not None else
            "FMP-ratios" if fr.get("peg")  is not None else
            yf_.get("peg_src") or "—"
        )
        roic = _first(fkm.get("roic"), fr.get("roic"), yd.get("roic"))
        roe  = _first(fkm.get("roe"),  fr.get("roe"),  yf_.get("roe"))
        om   = _first(fkm.get("op_margin"), fr.get("op_margin"), yf_.get("op_margin"))
        ic   = _first(fkm.get("int_coverage"), fr.get("int_coverage"),
                      fi.get("int_coverage"), yd.get("int_coverage"))
        de   = _first(fkm.get("debt_eq"), fr.get("debt_eq"), yf_.get("debt_eq"))
        gm_n = _first(fkm.get("gross_margin"), fi.get("gross_margin_now"),
                      yd.get("gross_margin_now"))
        gm_p = _first(fi.get("gross_margin_prev"), yd.get("gross_margin_prev"))
        eps_g= _first(yf_.get("eps_growth"), fi.get("eps_growth"))
        g_src= yf_.get("growth_src") or fi.get("growth_src")
        et   = yf_.get("earn_traj")

        rev4_fmp = fi.get("rev4", [None]*4)
        rev4_yah = yd.get("rev4", [None]*4)
        rev4 = (rev4_fmp if any(v is not None for v in rev4_fmp) else rev4_yah)

        ocf_ttm   = _first(fc.get("ocf_ttm"),  yd.get("ocf_ttm"))
        fcf_ttm   = _first(fc.get("fcf_ttm"),  fkm.get("fcf_ttm"), yd.get("fcf_ttm"))
        da_ttm    = fc.get("da_ttm")
        ni_ttm    = _first(fi.get("net_income_ttm"), yd.get("net_income_ttm"))
        ta_now    = _first(fb.get("total_assets_now"),   yd.get("total_assets_now"))
        ta_prev   = _first(fb.get("total_assets_prev"),  yd.get("total_assets_prev"))
        ltd_n     = _first(fb.get("lt_debt_ratio_now"),  yd.get("lt_debt_ratio_now"))
        ltd_p     = _first(fb.get("lt_debt_ratio_prev"), yd.get("lt_debt_ratio_prev"))
        cr_n      = _first(fb.get("current_ratio_now"),  fkm.get("current_ratio"),
                           yd.get("current_ratio_now"))
        cr_p      = _first(fb.get("current_ratio_prev"), yd.get("current_ratio_prev"))
        sh_n      = _first(fb.get("shares_now"),  yd.get("shares_now"))
        sh_p      = _first(fb.get("shares_prev"), yd.get("shares_prev"))

        roa_now = yd.get("roa_ttm"); roa_prev = yd.get("roa_prev")
        if roa_now is None and ta_now and ni_ttm:
            avg_ta = ((ta_now + ta_prev) / 2.0 if ta_prev else ta_now)
            if avg_ta > 0: roa_now = ni_ttm / avg_ta * 100.0

        ev_raw    = yf_.get("ev_raw")
        evebitda  = _first(fkm.get("ev_ebitda"), yf_.get("ev_ebitda"))
        evsales   = _first(fkm.get("ev_sales"),  yf_.get("ev_sales"))
        ebitda_ttm= yd.get("ebitda_ttm")
        if ebitda_ttm and ev_raw and float(ebitda_ttm) > 0 and float(ev_raw) > 0:
            ev_ttm_r = float(ev_raw) / float(ebitda_ttm)
            if 0 < ev_ttm_r < 200: evebitda = ev_ttm_r

        fcf_yield_pct = fkm.get("fcf_yield")
        div_yield     = yf_.get("div_yield")

        eps_surp_avg   = _first(fs.get("eps_surprise_avg"),   yf_.get("eps_surprise_avg"))
        eps_beat_rate  = _first(fs.get("eps_beat_rate"),      yf_.get("eps_beat_rate"))
        eps_surp_trend = _first(fs.get("eps_surprise_trend"), yf_.get("eps_surprise_trend"))
        rev_mom        = yf_.get("revision_momentum")

        merged[t] = {
            "pe": pe_val, "pe_src": pe_src, "fwd_pe": fwd_pe,
            "peg": peg_v, "peg_src": peg_src,
            "roe": roe, "roic": roic, "int_coverage": ic,
            "op_margin": om, "debt_eq": de,
            "gross_margin_now": gm_n, "gross_margin_prev": gm_p,
            "eps_growth": eps_g, "growth_src": g_src, "earn_traj": et,
            "mc": mc_val, "rev4": rev4,
            "ocf_ttm": ocf_ttm, "fcf_ttm": fcf_ttm,
            "net_income_ttm": ni_ttm, "ebitda_ttm": ebitda_ttm, "ev_raw": ev_raw,
            "ev_ebitda": evebitda, "ev_sales": evsales,
            "fcf_yield_pct": fcf_yield_pct, "div_yield": div_yield,
            "roa_ttm": roa_now, "roa_prev": roa_prev,
            "total_assets_now": ta_now, "total_assets_prev": ta_prev,
            "lt_debt_ratio_now": ltd_n, "lt_debt_ratio_prev": ltd_p,
            "current_ratio_now": cr_n, "current_ratio_prev": cr_p,
            "shares_now": sh_n, "shares_prev": sh_p,
            "eps_surprise_avg": eps_surp_avg, "eps_beat_rate": eps_beat_rate,
            "eps_surprise_trend": eps_surp_trend, "revision_momentum": rev_mom,
        }
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# Score History  (v18 FIX-13)
# ══════════════════════════════════════════════════════════════════════════════
def record_score_history(scr: pd.DataFrame) -> None:
    if "score_history" not in st.session_state:
        st.session_state["score_history"] = {}
    current_run = st.session_state.get("run_id", "?")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    for _, row in scr.iterrows():
        t = row["Ticker"]
        if t not in st.session_state["score_history"]:
            st.session_state["score_history"][t] = []
        st.session_state["score_history"][t].append({
            "ts": ts, "run_id": current_run,
            "score": row.get("Score"), "conviction": row.get("Conviction Score"),
            "rank": row.get("Rank"), "piotroski_f": row.get("Piotroski F"),
            "momentum_score": row.get("Momentum Score"),
        })
        st.session_state["score_history"][t] = (
            st.session_state["score_history"][t][-10:])


def get_score_delta(ticker: str, current_score) -> float:
    current_run  = st.session_state.get("run_id", "?")
    all_history  = st.session_state.get("score_history", {}).get(ticker, [])
    prev_entries = [h for h in all_history if h.get("run_id") != current_run]
    if not prev_entries: return None
    prev = prev_entries[-1].get("score")
    if prev is None or current_score is None or pd.isna(current_score): return None
    try:
        return round(float(current_score) - float(prev), 1)
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Piotroski + Sloan
# ══════════════════════════════════════════════════════════════════════════════
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

    ldr_now = d.get("lt_debt_ratio_now"); ldr_prev = d.get("lt_debt_ratio_prev")
    l1 = 1 if (ldr_now is not None and ldr_prev is not None and ldr_now < ldr_prev) else 0
    score += l1; comp["L1_leverage_down"] = l1

    cr_now = d.get("current_ratio_now"); cr_prev = d.get("current_ratio_prev")
    l2 = 1 if (cr_now is not None and cr_prev is not None and cr_now > cr_prev) else 0
    score += l2; comp["L2_liquidity_up"] = l2

    sh_now = d.get("shares_now"); sh_prev = d.get("shares_prev")
    l3 = 1 if (sh_now is not None and sh_prev is not None and sh_now <= sh_prev*1.02) else 0
    score += l3; comp["L3_no_dilution"] = l3

    gm_now = d.get("gross_margin_now"); gm_prev = d.get("gross_margin_prev")
    o1 = 1 if (gm_now is not None and gm_prev is not None and gm_now > gm_prev) else 0
    score += o1; comp["O1_gross_margin_up"] = o1

    rev_growth = d.get("rev_growth_pct")
    o2 = 1 if (rev_growth is not None and rev_growth > 0) else 0
    score += o2; comp["O2_asset_turn_up"] = o2

    return score, comp


def compute_sloan_ratio(net_income, ocf, total_assets_now, total_assets_prev):
    try:
        if any(v is None for v in [net_income, ocf, total_assets_now, total_assets_prev]):
            return None
        avg_ta = (float(total_assets_now) + float(total_assets_prev)) / 2.0
        if avg_ta <= 0: return None
        return (float(net_income) - float(ocf)) / avg_ta
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Quality Score
# ══════════════════════════════════════════════════════════════════════════════
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
                  if int_coverage is not None and not pd.isna(int_coverage) else 0.0)
    weights.append(0.15)

    if sector not in ROE_PRIMARY_SECTORS:
        scores.append(min(100.0, max(0.0, float(op_margin)/40.0*100.0))
                      if op_margin is not None and not pd.isna(op_margin) else 0.0)
        weights.append(0.15)

    if gross_margin_now is not None and not pd.isna(gross_margin_now):
        gm_score = min(100.0, max(0.0, float(gross_margin_now)/60.0*100.0))
        if gross_margin_prev and float(gross_margin_now) > float(gross_margin_prev):
            gm_score = min(100.0, gm_score * 1.10)
        scores.append(gm_score)
    else:
        scores.append(0.0)
    weights.append(0.20)

    scores.append(float(piotroski_f)/9.0*100.0
                  if piotroski_f is not None and not pd.isna(piotroski_f) else 0.0)
    weights.append(0.15)

    if sloan_ratio is not None and not pd.isna(sloan_ratio):
        sr = float(sloan_ratio)
        sc = max(-0.15, min(0.05, sr))
        scores.append((0.05 - sc) / 0.20 * 100.0)
    else:
        scores.append(0.0)
    weights.append(0.10)

    total_w = sum(weights)
    return sum(s * w for s, w in zip(scores, weights)) / total_w if total_w else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Valuation + Ranking + Conviction + CS
# ══════════════════════════════════════════════════════════════════════════════
def compute_valuation_subscore(elig: pd.DataFrame) -> pd.Series:
    scores = pd.DataFrame(index=elig.index); weights = []
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
    if not weights: return pd.Series(50.0, index=elig.index)
    total_w   = sum(w for _, w in weights)
    composite = sum(scores[k] * (w/total_w) for k, w in weights if k in scores.columns)
    return composite.fillna(0.0)


def compute_conviction_scores_elite(scr: pd.DataFrame) -> pd.DataFrame:
    scr = scr.copy()
    KEY_FACTORS = ["P/E","Fwd P/E","PEG","Quality Score","Momentum Score","Earn Traj"]
    n_factors = len(KEY_FACTORS)
    def completeness(row):
        return sum(1 for c in KEY_FACTORS if c in row.index and pd.notna(row[c]))/n_factors
    scr["_completeness"] = scr.apply(completeness, axis=1)
    smed = scr.groupby("Sector")["P/E"].median().to_dict()
    scr["_sector_med_pe"] = scr["Sector"].map(smed)
    def signal_agreement(row):
        sub = []
        pe = row.get("P/E"); med = row.get("_sector_med_pe", 25)
        if pd.notna(pe) and med and med > 0:
            sub.append(1.0 if pe < med*0.9 else (-1.0 if pe > med*1.1 else 0.0))
        mom = row.get("Momentum Score")
        if pd.notna(mom): sub.append(float(np.clip(float(mom),-1.0,1.0)))
        et = row.get("Earn Traj")
        if pd.notna(et): sub.append(float(et))
        return (float(np.clip(1.0-float(np.std(sub))/1.5,0.0,1.0))
                if len(sub) >= 2 else 0.5)
    scr["_signal_agreement"] = scr.apply(signal_agreement, axis=1)
    def anomaly_mult(row):
        mult = 1.0
        pf = row.get("Piotroski F")
        if pd.notna(pf) and float(pf) <= 2: mult *= 0.70
        sl = row.get("Sloan Ratio")
        if pd.notna(sl) and float(sl) > SLOAN_ACCRUALS_THRESHOLD: mult *= 0.85
        return mult
    scr["_anomaly_mult"] = scr.apply(anomaly_mult, axis=1)
    raw_conv = (scr["Score"]
                * (0.5 + 0.5*scr["_completeness"])
                * (0.7 + 0.3*scr["_signal_agreement"])
                * scr["_anomaly_mult"])
    c_min, c_max = raw_conv.min(), raw_conv.max()
    scr["Conviction Score"] = ((raw_conv - c_min)/(c_max - c_min)*100.0
                                if c_max > c_min else pd.Series(50.0, index=scr.index))
    return scr.drop(columns=["_completeness","_signal_agreement",
                              "_anomaly_mult","_sector_med_pe"])


def compute_cross_sectional_scores(scr: pd.DataFrame) -> pd.DataFrame:
    scr = scr.copy()
    cs_raw = (0.25*elite_factor_score(scr["Fwd P/E"].fillna(scr["P/E"]), ascending=True) +
              0.25*elite_factor_score(scr["Quality Score"],  ascending=False) +
              0.20*elite_factor_score(scr["PEG"],            ascending=True) +
              0.15*elite_factor_score(scr["Earn Traj"],      ascending=False) +
              0.15*elite_factor_score(scr["Momentum Score"], ascending=False))
    cs_min, cs_max = cs_raw.min(), cs_raw.max()
    scr["CS Score"] = ((cs_raw - cs_min)/(cs_max - cs_min)*100.0
                       if cs_max > cs_min else pd.Series(50.0, index=scr.index))
    return scr


def compute_rank_by_sector(scr):
    scr = scr.copy(); scr["Score"] = pd.NA; scr["Rank"] = pd.NA
    for sector in scr["Sector"].dropna().unique():
        g    = scr[scr["Sector"] == sector]; elig = g[g["Eligible"]].copy()
        if elig.empty: continue
        W = SECTOR_FACTOR_WEIGHTS.get(sector, DEFAULT_FACTOR_WEIGHTS)
        elig["_s_val"]   = compute_valuation_subscore(elig)
        elig["_s_peg"]   = elite_factor_score(elig["PEG"],            ascending=True)
        elig["_s_mom"]   = elite_factor_score(elig["Momentum Score"], ascending=False)
        elig["_s_etraj"] = elite_factor_score(elig["Earn Traj"],      ascending=False)
        qs = elig["Quality Score"]; q_min, q_max = qs.min(), qs.max()
        elig["_s_quality"] = ((qs-q_min)/(q_max-q_min)*100.0
                               if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min
                               else qs.fillna(0.0))
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)
        raw = (W["valuation"]*elig["_s_val"] + W["quality"]*elig["_s_quality"] +
               W["peg"]*elig["_s_peg"] + W["earn_traj"]*elig["_s_etraj"] +
               W["momentum"]*elig["_s_mom"])
        fc  = ["P/E","PEG","Quality Score","Earn Traj","Momentum Score"]
        raw = raw * elig.apply(lambda r: missing_factor_penalty(r, fc), axis=1)
        elig["Score"] = raw
        elig = elig.sort_values("Score", ascending=False)
        elig["Rank"] = range(1, len(elig)+1)
        scr.loc[elig.index, "Score"] = elig["Score"]
        scr.loc[elig.index, "Rank"]  = elig["Rank"]
    return scr


# ══════════════════════════════════════════════════════════════════════════════
# S&P 500 Universe
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400)
def fetch_sp500_constituents():
    url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r    = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    tbl  = soup.find("table", {"id":"constituents"})
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
# Build Screener Table
# ══════════════════════════════════════════════════════════════════════════════
def build_screener_table(universe_df, pm_map, merged_map):
    rows = []
    for _, r in universe_df.iterrows():
        t = r["Ticker"]; sec = r["Sector"]
        # FIX-MOM-6: pm_map keys are uppercase — match correctly
        pm = pm_map.get(t.upper(), pm_map.get(t, {}))
        fi = merged_map.get(t, {})

        price     = to_num(pm.get("price") or fi.get("price"))
        hi        = to_num(pm.get("hi52"))
        lo        = to_num(pm.get("lo52"))
        mom_score = to_num(pm.get("momentum_score"))
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

        rev4_raw = fi.get("rev4", [None]*4)
        rq1 = sf(rev4_raw[0]) if len(rev4_raw) > 0 else None
        rq2 = sf(rev4_raw[1]) if len(rev4_raw) > 1 else None
        rq3 = sf(rev4_raw[2]) if len(rev4_raw) > 2 else None
        rq4 = sf(rev4_raw[3]) if len(rev4_raw) > 3 else None
        growth = to_num(revenue_growth_pct_cagr([rq1, rq2, rq3, rq4]))

        # ── PEG 3-tier ───────────────────────────────────────────────────
        peg_direct = to_num(fi.get("peg"))
        peg = None; peg_method = "—"
        if pd.notna(peg_direct):
            peg = float(peg_direct); peg_method = fi.get("peg_src") or "FMP"
        else:
            pe_for_peg = fwd if pd.notna(fwd) else pe
            eps_g = fi.get("eps_growth"); g_src = fi.get("growth_src") or ""
            if eps_g is not None:
                eg = float(eps_g)
                if eg >= MIN_GROWTH_PCT_FOR_PEG and pd.notna(pe_for_peg):
                    peg = float(pe_for_peg) / eg; peg_method = g_src
            if peg is None and pd.notna(earn_traj) and float(earn_traj) > 0:
                proxy = float(earn_traj) * 100.0
                if proxy >= MIN_GROWTH_PCT_FOR_PEG and pd.notna(pe_for_peg):
                    peg = float(pe_for_peg) / proxy; peg_method = "EarnTraj-proxy"
        if peg is not None and (peg <= 0 or peg > 500): peg = None

        fi_g = dict(fi); fi_g["rev_growth_pct"] = float(growth) if pd.notna(growth) else None
        piotroski_f, _ = compute_piotroski_fscore(fi_g)
        sloan_ratio    = compute_sloan_ratio(
            fi.get("net_income_ttm"), fi.get("ocf_ttm"),
            fi.get("total_assets_now"), fi.get("total_assets_prev"))

        fcf_yield = to_num(fi.get("fcf_yield_pct"))
        if pd.isna(fcf_yield):
            fcf_ttm = fi.get("fcf_ttm")
            if fcf_ttm and pd.notna(mc) and float(mc) > 0:
                fcf_yield = to_num(float(fcf_ttm) / float(mc) * 100.0)

        fcf_ni_ratio = None
        fcf_ttm2 = fi.get("fcf_ttm"); ni_val = fi.get("net_income_ttm")
        if fcf_ttm2 and ni_val and float(ni_val) != 0:
            fcf_ni_ratio = float(fcf_ttm2) / float(ni_val)

        ev_ebitda_display = to_num(fi.get("ev_ebitda"))

        q_score = compute_quality_score_elite(
            float(roic) if pd.notna(roic) else None,
            float(roe)  if pd.notna(roe)  else None,
            float(ic)   if pd.notna(ic)   else None,
            float(om)   if pd.notna(om)   else None,
            gross_margin_now  = float(fi["gross_margin_now"])  if fi.get("gross_margin_now")  else None,
            gross_margin_prev = float(fi["gross_margin_prev"]) if fi.get("gross_margin_prev") else None,
            fcf_ni_ratio  = fcf_ni_ratio,
            piotroski_f   = piotroski_f,
            sloan_ratio   = sloan_ratio,
            sector        = sec,
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
            "Ret 1Mo%":           to_num(pm.get("ret_1mo")),
            "Ret 3Mo%":           to_num(pm.get("ret_3mo")),
            "Ret 6Mo%":           to_num(pm.get("ret_6mo")),
            "Trailing Vol%":      to_num(pm.get("trailing_vol")),
            "Eligible":           True,
            "Rev Q1 Oldest ($B)": to_num(rq1),
            "Rev Q2 ($B)":        to_num(rq2),
            "Rev Q3 ($B)":        to_num(rq3),
            "Rev Q4 Latest ($B)": to_num(rq4),
            "Rev Growth% (CAGR)": growth,
            "EV/EBITDA":          ev_ebitda_display,
            "FCF Yield%":         to_num(fcf_yield),
            "EV/Sales":           to_num(fi.get("ev_sales")),
            "Div Yield%":         to_num(fi.get("div_yield")),
            "Piotroski F":        to_num(piotroski_f),
            "Sloan Ratio":        to_num(sloan_ratio),
            "EPS Surp Avg%":      to_num(fi.get("eps_surprise_avg")),
            "EPS Beat Rate":      to_num(fi.get("eps_beat_rate")),
            "EPS Surp Trend":     to_num(fi.get("eps_surprise_trend")),
            "Revision Mom":       to_num(fi.get("revision_momentum")),
            "Skip Mo":            to_num(pm.get("skip_month_raw")),
            "52W Prox":           to_num(pm.get("hi52_proximity")),
            "vs MA200":           to_num(pm.get("vs_ma200")),
            "Rel Str SPY":        to_num(pm.get("rel_strength_spy")),
        })

    scr = pd.DataFrame(rows)
    if scr.empty: return scr
    total_sp500_mc = scr["Mkt Cap"].sum()
    scr["MC% of S&P500"] = (scr["Mkt Cap"]/total_sp500_mc*100.0
                             if total_sp500_mc > 0 else None)
    num_cols = [
        "Price","Mkt Cap","P/E","Fwd P/E","PEG","52W Pos%",
        "ROIC%","ROE%","Int Coverage","Op Margin%","Debt/Eq",
        "Quality Score","Earn Traj","Momentum Score",
        "Ret 1Mo%","Ret 3Mo%","Ret 6Mo%","Trailing Vol%","MC% of S&P500",
        "Rev Q1 Oldest ($B)","Rev Q2 ($B)","Rev Q3 ($B)","Rev Q4 Latest ($B)",
        "Rev Growth% (CAGR)","EV/EBITDA","FCF Yield%","EV/Sales","Div Yield%",
        "Piotroski F","Sloan Ratio",
        "EPS Surp Avg%","EPS Beat Rate","EPS Surp Trend","Revision Mom",
        "Skip Mo","52W Prox","vs MA200","Rel Str SPY",
    ]
    for c in num_cols:
        if c in scr.columns: scr[c] = to_num(scr[c])
    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns: scr["Rank"] = pd.NA
    sector_med_pe = scr.groupby("Sector")["P/E"].transform("median")
    scr["P/E vs Sector Med"] = (scr["P/E"] / sector_med_pe).round(2)
    scr = compute_conviction_scores_elite(scr)
    scr = compute_cross_sectional_scores(scr)
    return scr


# ══════════════════════════════════════════════════════════════════════════════
# Quality Flag
# ══════════════════════════════════════════════════════════════════════════════
def quality_flag(roic, roe, ic, om, sloan_ratio=None, sector=None):
    EPSILON = 1e-9; flags = []
    prof = (roe if sector in ROE_PRIMARY_SECTORS
            else (roic if (roic is not None and not pd.isna(roic)) else roe))
    lbl  = (("ROE" if sector in ROE_PRIMARY_SECTORS
              else ("ROIC" if (roic is not None and not pd.isna(roic)) else "ROE")))
    if prof is not None and not pd.isna(prof) and float(prof) < QUALITY_THRESHOLDS["roic_min"] - EPSILON:
        flags.append("{}<8%".format(lbl))
    if ic is not None and not pd.isna(ic) and float(ic) < QUALITY_THRESHOLDS["int_coverage_min"]:
        flags.append("IntCov<3x")
    if sector not in ROE_PRIMARY_SECTORS:
        if om is not None and not pd.isna(om) and float(om) < QUALITY_THRESHOLDS["op_margin_min"]:
            flags.append("Margin<5%")
    if sloan_ratio is not None and not pd.isna(sloan_ratio) and float(sloan_ratio) > SLOAN_ACCRUALS_THRESHOLD:
        flags.append("HighAccruals")
    return ", ".join(flags) if flags else "Pass"


# ══════════════════════════════════════════════════════════════════════════════
# Reference Guide
# ══════════════════════════════════════════════════════════════════════════════
def render_reference_guide():
    st.markdown("## Column Reference Guide")
    tabs = st.tabs([
        "Valuation","Quality","PEG","Earn Trajectory",
        "Momentum","Earnings Surprise","Ranking & Score","Coverage v19.1",
    ])
    with tabs[0]:
        st.markdown("**Valuation** — FCF Yield 40% + EV/EBITDA 35% + Fwd P/E 25%")
    with tabs[1]:
        st.markdown("**Quality Score (0-100)** — ROIC 25% · IntCov 15% · OpMargin 15% · GrossMargin 20% · Piotroski 15% · Sloan 10%")
    with tabs[2]:
        st.markdown("""
**PEG v19.1 — 3-tier cascade**
| Tier | Source | Coverage |
|---|---|---|
| 1 | FMP key-metrics / ratios direct | ~60% |
| 2 | EPS growth (FMP IS or Yahoo) | ~80% |
| 3 | Earn Traj proxy | ~90% |
        """)
    with tabs[3]:
        st.markdown("**Earn Traj** = (Fwd EPS − Trail EPS) / |Trail EPS|, clipped [-1,+1].")
    with tabs[4]:
        st.markdown("""
**Momentum v19.1** — computed via `yf.download(group_by='column')` bulk.

| Signal | Weight |
|---|---|
| Skip Mo (6Mo−1Mo return / vol) | 40% |
| 52W Proximity to high | 25% |
| vs MA200 | 20% |
| Rel Str vs SPY 3Mo | 15% |
        """)
    with tabs[5]:
        st.markdown("**EPS Surprise** — FMP /earnings-surprises (primary) + Yahoo fallback. ~86% coverage.")
    with tabs[6]:
        st.markdown("**Score Delta** — run_id guarded (v18). Conviction + CS computed cross-sectionally.")
    with tabs[7]:
        st.markdown("""
**Expected coverage v19.1**

| Metric | v18 actual | v19 actual | v19.1 target |
|---|---|---|---|
| P/E | 31% | 82% | 85%+ |
| Fwd P/E | 32% | 86% | 86%+ |
| EV/EBITDA | 30% | 80% | 82%+ |
| FCF | 1% | 94% | 94%+ |
| EPS Surprise | 33% | 86% | 87%+ |
| **Momentum** | **21%** | **0%** | **95%+** |

**Root cause of 0% momentum**: `yf.download(group_by='ticker')` returns
MultiIndex with outer=ticker, inner=field — but code accessed `raw[t]['Close']`
which works only when outer=field. Fixed with `group_by='column'` +
`raw['Close'][t]` pattern plus 3-pattern fallback chain.
        """)
    st.caption("v19.1: momentum bulk download fixed · cached build_momentum_map · uppercase key normalisation.")


# ══════════════════════════════════════════════════════════════════════════════
# APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="S&P 500 Screener v19.1", layout="wide", page_icon="📊")
st.markdown(
    "<style>"
    "div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}"
    "</style>",
    unsafe_allow_html=True,
)
st.markdown("## S&P 500 Fundamental Screener v19.1")

if "run_id" not in st.session_state:
    st.session_state["run_id"] = str(uuid.uuid4())[:8]

page_screener, page_reference = st.tabs(["Screener", "Column Reference Guide"])

with page_screener:
    col_r, col_t = st.columns([1, 6])
    with col_r:
        if st.button("Refresh"):
            st.cache_data.clear()
            st.session_state["run_id"] = str(uuid.uuid4())[:8]
            st.rerun()
    with col_t:
        st.caption(
            "Last loaded: {} · v19.1: momentum fixed (bulk group_by=column) · "
            "FMP-primary · 95%+ momentum target".format(
                datetime.now().strftime("%I:%M %p"))
        )

    fmp_key = get_fmp_key()
    if fmp_key:
        st.success("✅ FMP API key active — full multi-endpoint coverage enabled.")
    else:
        st.warning(
            "⚠️ No FMP key detected. Momentum will still reach 95%+ (bulk download fixed). "
            "Add `[fmp] api_key` to Streamlit Secrets for P/E 85%+, FCF 94%+, EPS 87%+."
        )

    with st.spinner("Loading S&P 500 universe..."):
        sp500 = fetch_sp500_constituents()
    if sp500.empty:
        st.error("Failed to load S&P 500 universe."); st.stop()

    universe_df = sp500.copy().reset_index(drop=True)
    tickers     = tuple(universe_df["Ticker"].tolist())
    today_date  = date.today()

    st.markdown("### Filters")
    all_sectors = sorted(universe_df["Sector"].dropna().unique().tolist())
    f1, f2, f3, f4, f5 = st.columns(5)
    sector_sel = f1.selectbox("Sector", ["All Sectors"] + all_sectors)
    sort_by    = f2.selectbox("Sort by", [
        "Sector then Rank", "Score high to low", "Conviction high to low",
        "CS Score high to low", "MC% of S&P500 high to low",
        "Price low to high", "Price high to low", "Mkt Cap high to low",
        "PE low to high", "Fwd PE low to high", "PEG low to high",
        "Quality Score high", "ROIC high to low", "ROE high to low",
        "Earn Traj high to low", "Rev Growth high to low",
        "Momentum Score high", "52W Pos low to high",
        "P/E vs Sector Med low to high", "Piotroski F high",
        "FCF Yield high", "EV/EBITDA low", "EPS Beat Rate high",
        "Revision Mom high", "Score Delta high",
    ])
    mc_min_b   = f3.number_input("Min Mkt Cap ($B)", value=0, step=10, min_value=0)
    pe_max     = f4.number_input("Max P/E", value=9999, step=50, min_value=0)
    qual_min_f = f5.number_input("Min Quality Score", value=0.0, step=5.0,
                                  min_value=0.0, max_value=100.0)

    # ── Step 1: Bulk prices ────────────────────────────────────────────────
    with st.spinner("Step 1/7 — Bulk price download ({} tickers)...".format(len(tickers))):
        prices_map = fetch_prices_bulk(tickers)

    # Diagnostics — surfaces empty dict issue immediately
    n_prices = len(prices_map)
    if n_prices < len(tickers) * 0.5:
        st.warning("⚠️ Only {}/{} price series fetched. Momentum coverage may be low.".format(
            n_prices, len(tickers)))
    else:
        st.caption("✓ Price series: {}/{} tickers loaded.".format(n_prices, len(tickers)))

    with st.spinner("Step 2/7 — SPY benchmark + momentum computation..."):
        spy_3mo = fetch_spy_3mo_return()
        pm_data = build_momentum_map(prices_map, spy_3mo=spy_3mo)

    if spy_3mo is not None:
        st.caption("SPY 3Mo return: {:.1f}%".format(spy_3mo))

    n_momentum = sum(1 for v in pm_data.values() if v.get("momentum_score") is not None)
    st.caption("Momentum computed: {}/{} tickers.".format(n_momentum, len(tickers)))

    # ── Step 3: FMP bulk ──────────────────────────────────────────────────
    fmp_quotes = {}; fmp_km = {}; fmp_ratios = {}
    fmp_income = {}; fmp_cashflow = {}; fmp_balance = {}; fmp_surprises = {}

    if fmp_key:
        with st.spinner("Step 3a/7 — FMP /quote bulk..."):
            fmp_quotes = fetch_fmp_bulk_quotes(tickers, fmp_key)
        with st.spinner("Step 3b/7 — FMP /key-metrics-ttm..."):
            fmp_km     = fetch_fmp_bulk_key_metrics(tickers, fmp_key)
        with st.spinner("Step 3c/7 — FMP /ratios-ttm..."):
            fmp_ratios = fetch_fmp_bulk_ratios(tickers, fmp_key)
        with st.spinner("Step 3d/7 — FMP /income-statement quarterly..."):
            fmp_income = fetch_fmp_income_statements(tickers, fmp_key)
        with st.spinner("Step 3e/7 — FMP /cash-flow-statement quarterly..."):
            fmp_cashflow = fetch_fmp_cashflow_statements(tickers, fmp_key)
        with st.spinner("Step 3f/7 — FMP /balance-sheet quarterly..."):
            fmp_balance  = fetch_fmp_balance_sheets(tickers, fmp_key)
        with st.spinner("Step 3g/7 — FMP /earnings-surprises..."):
            fmp_surprises = fetch_fmp_earnings_surprises(tickers, fmp_key)
    else:
        st.info("Step 3/7 — FMP skipped (no API key). Yahoo-only mode.")

    # ── Step 4: Yahoo fill ────────────────────────────────────────────────
    with st.spinner("Step 4/7 — Yahoo fill (Fwd P/E, Earn Traj)..."):
        yahoo_fills_map = fetch_yahoo_fills(tickers, _cache_date=today_date)

    # ── Step 5: Yahoo deep (only where FCF still missing) ─────────────────
    tickers_needing_deep = tuple(
        t for t in tickers
        if (fmp_cashflow.get(t, {}).get("fcf_ttm") is None
            and fmp_km.get(t, {}).get("fcf_ttm") is None)
    )
    if tickers_needing_deep:
        with st.spinner("Step 5/7 — Yahoo deep fill ({} tickers)...".format(
                len(tickers_needing_deep))):
            yahoo_deep_map = fetch_yahoo_deep_fills(tickers_needing_deep,
                                                     _cache_date=today_date)
    else:
        yahoo_deep_map = {}
        st.success("Step 5/7 — FMP FCF coverage complete, no Yahoo deep needed.")

    # ── Merge ──────────────────────────────────────────────────────────────
    with st.spinner("Step 6/7 — Merging all sources..."):
        merged_map = build_master_data(
            tickers, fmp_quotes, fmp_km, fmp_ratios,
            fmp_income, fmp_cashflow, fmp_balance, fmp_surprises,
            yahoo_fills_map, yahoo_deep_map,
        )

    # ── Coverage report ────────────────────────────────────────────────────
    total_t = len(tickers)
    def cov_m(key): return sum(1 for t in tickers
                                if merged_map.get(t, {}).get(key) is not None)
    def cov_p(key): return sum(1 for t in tickers
                                if pm_data.get(t.upper(), pm_data.get(t, {})).get(key) is not None)

    st.info(
        "**Coverage v19.1** — "
        "P/E: {}/{} ({:.0f}%) · Fwd P/E: {}/{} ({:.0f}%) · "
        "EV/EBITDA: {}/{} ({:.0f}%) · FCF: {}/{} ({:.0f}%) · "
        "EPS Surprise: {}/{} ({:.0f}%) · Revision Mom: {}/{} ({:.0f}%) · "
        "**Momentum: {}/{} ({:.0f}%)** · "
        "Sources: Yahoo{}".format(
            cov_m("pe"),                total_t, cov_m("pe")                /total_t*100,
            cov_m("fwd_pe"),            total_t, cov_m("fwd_pe")            /total_t*100,
            cov_m("ev_ebitda"),         total_t, cov_m("ev_ebitda")         /total_t*100,
            cov_m("fcf_ttm"),           total_t, cov_m("fcf_ttm")           /total_t*100,
            cov_m("eps_surprise_avg"),  total_t, cov_m("eps_surprise_avg")  /total_t*100,
            cov_m("revision_momentum"), total_t, cov_m("revision_momentum") /total_t*100,
            cov_p("momentum_score"),    total_t, cov_p("momentum_score")    /total_t*100,
            " + FMP (primary)" if fmp_key else " only",
        )
    )

    # ── Build + score ──────────────────────────────────────────────────────
    with st.spinner("Step 7/7 — Building screener table..."):
        scr = build_screener_table(universe_df, pm_data, merged_map)

    scr["Score Delta"] = scr.apply(
        lambda row: get_score_delta(row["Ticker"], row.get("Score")), axis=1)
    record_score_history(scr)

    # ── Filters + sort ────────────────────────────────────────────────────
    filt = scr.copy()
    if sector_sel != "All Sectors":
        filt = filt[filt["Sector"] == sector_sel]
    filt = filt[(filt["Mkt Cap"].isna())       | (filt["Mkt Cap"]       >= mc_min_b*1e9)]
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
        "Score Delta high":              (["Score Delta"],            [False]),
    }
    sc, sa = sort_map.get(sort_by, (["Sector","Rank"],[True,True]))
    filt   = filt.sort_values(sc, ascending=sa, na_position="last")

    st.caption("Showing **{}** of **{}** · Sector: {} · Sort: {}".format(
        len(filt), len(scr), sector_sel, sort_by))

    disp = filt.copy()
    disp["Price ($)"]          = safe_round(disp["Price"], 2)
    disp["Mkt Cap ($B)"]       = safe_round(disp["Mkt Cap"] / 1e9, 2)
    disp["MC% of S&P500"]      = safe_round(disp["MC% of S&P500"], 4)
    disp["Rev Q1 Oldest ($B)"] = safe_round(disp["Rev Q1 Oldest ($B)"] / 1e9, 2)
    disp["Rev Q2 ($B)"]        = safe_round(disp["Rev Q2 ($B)"]         / 1e9, 2)
    disp["Rev Q3 ($B)"]        = safe_round(disp["Rev Q3 ($B)"]         / 1e9, 2)
    disp["Rev Q4 Latest ($B)"] = safe_round(disp["Rev Q4 Latest ($B)"]  / 1e9, 2)

    disp["Quality Flag"] = disp.apply(
        lambda r: quality_flag(
            r.get("ROIC%"), r.get("ROE%"), r.get("Int Coverage"),
            r.get("Op Margin%"), sloan_ratio=r.get("Sloan Ratio"),
            sector=r.get("Sector"),
        ), axis=1,
    )

    ROUND_COLS = [
        "P/E","Fwd P/E","PEG","Earn Traj","52W Pos%",
        "ROIC%","ROE%","Int Coverage","Op Margin%","Debt/Eq",
        "Quality Score","Momentum Score","Ret 1Mo%","Ret 3Mo%",
        "Ret 6Mo%","Trailing Vol%","Score","Conviction Score","CS Score",
        "Rev Growth% (CAGR)","P/E vs Sector Med",
        "EV/EBITDA","FCF Yield%","EV/Sales","Div Yield%","Sloan Ratio",
        "EPS Surp Avg%","EPS Beat Rate","EPS Surp Trend","Revision Mom",
        "Skip Mo","52W Prox","vs MA200","Rel Str SPY","Score Delta",
    ]
    for c in ROUND_COLS:
        if c in disp.columns: disp[c] = safe_round(disp[c], 2)

    disp["Rank"] = pd.to_numeric(disp["Rank"], errors="coerce")
    disp["Rank"] = disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

    COLS = [
        "Ticker","Sector","Price ($)","Mkt Cap ($B)","MC% of S&P500",
        "P/E","P/E vs Sector Med","Fwd P/E",
        "EV/EBITDA","FCF Yield%","EV/Sales","Div Yield%",
        "PEG","PEG Method","Earn Traj",
        "EPS Surp Avg%","EPS Beat Rate","EPS Surp Trend","Revision Mom",
        "ROIC%","ROE%","Int Coverage","Op Margin%","Debt/Eq",
        "Quality Score","Quality Flag","Piotroski F","Sloan Ratio",
        "Momentum Score","Skip Mo","52W Prox","vs MA200","Rel Str SPY",
        "Ret 1Mo%","Ret 3Mo%","Ret 6Mo%","Trailing Vol%",
        "52W Pos%","Score","Score Delta","Conviction Score","CS Score","Rank",
        "Rev Q1 Oldest ($B)","Rev Q2 ($B)","Rev Q3 ($B)","Rev Q4 Latest ($B)",
        "Rev Growth% (CAGR)",
    ]
    disp_final = disp[[c for c in COLS if c in disp.columns]].copy()
    st.dataframe(disp_final, use_container_width=True, height=680)

    st.download_button(
        label="Download CSV",
        data=disp_final.to_csv(index=False).encode("utf-8"),
        file_name="sp500_screener_v19_1_{}.csv".format(
            datetime.now().strftime("%Y%m%d_%H%M")),
        mime="text/csv",
    )

with page_reference:
    render_reference_guide()
