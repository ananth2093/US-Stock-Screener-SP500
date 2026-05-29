# screener_app.py
# ─────────────────────────────────────────────────────────────────────────────
# S&P 500 Fundamental Screener v2 — Streamlit Community Cloud
# Improvements over v1:
#   • Quality dimension (ROE, Debt/Equity, Operating Margin)
#   • Percentile-based scoring (preserves magnitude, not just rank)
#   • Missing-data penalty (stocks missing 3+ factors docked)
#   • Consistent PEG (FMP analyst EPS growth estimate, labeled)
#   • Momentum column (1mo, 3mo, 6mo trailing returns)
#   • Data source transparency column
#   • Screener (hard filters) separated from Ranker
#   • Forward returns column (1mo, 3mo, 6mo price change)
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
from datetime import datetime

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_GROWTH_PCT_FOR_PEG = 5.0

FACTOR_WEIGHTS = {
    "valuation":  0.30,
    "peg":        0.20,
    "quality":    0.25,   # NEW: ROE + margin + debt
    "earn_traj":  0.15,
    "momentum":   0.10,   # NEW: replaces 52W pos + adds 3mo/6mo return
}

QUALITY_THRESHOLDS = {
    "roe_min":        10.0,   # ROE >= 10% to pass quality screen
    "debt_eq_max":    2.0,    # Debt/Equity <= 2.0
    "op_margin_min":  5.0,    # Operating Margin >= 5%
}


# ── Credentials ───────────────────────────────────────────────────────────────
def get_fmp_key():
    try:
        return st.secrets["fmp"]["api_key"]
    except Exception:
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────
def to_num(x):
    return pd.to_numeric(x, errors="coerce")


def safe_float(val):
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def revenue_growth_pct_cagr(rev4):
    try:
        if rev4 is None or len(rev4) != 4:
            return None
        q1, _, _, q4 = rev4
        if q1 is None or q4 is None:
            return None
        q1 = float(q1)
        q4 = float(q4)
        if q1 <= 0 or q4 <= 0:
            return None
        return ((q4 / q1) ** (1 / 3) - 1) * 100.0
    except Exception:
        return None


def fmt_mc(val):
    if pd.isna(val) or val == 0:
        return "N/A"
    if val >= 1e12:
        return "${:.2f}T".format(val / 1e12)
    if val >= 1e9:
        return "${:.1f}B".format(val / 1e9)
    return "${:.0f}M".format(val / 1e6)


def percentile_score(series: pd.Series, ascending=True) -> pd.Series:
    """
    Convert a series to 0-100 percentile scores within the non-null values.
    ascending=True  → lower raw value = higher score (valuation: lower PE is better)
    ascending=False → higher raw value = higher score (quality: higher ROE is better)
    NaN → 0 (explicit missing-data penalty)
    """
    result = pd.Series(index=series.index, dtype=float)
    valid_mask = series.notna()
    if valid_mask.sum() == 0:
        return result.fillna(0.0)
    ranked = series[valid_mask].rank(method="average", ascending=ascending)
    n = valid_mask.sum()
    result[valid_mask] = (ranked - 1) / (n - 1) * 100.0 if n > 1 else 50.0
    result[~valid_mask] = 0.0   # explicit penalty for missing data
    return result


def missing_factor_penalty(row, factor_cols):
    """
    Returns a penalty multiplier: 1.0 if 0–1 factors missing,
    0.85 if 2 missing, 0.70 if 3+ missing.
    This prevents a stock with one strong signal and 4 missing signals
    from ranking highly.
    """
    missing = sum(1 for c in factor_cols if pd.isna(row.get(c)))
    if missing >= 3:
        return 0.70
    if missing == 2:
        return 0.85
    return 1.0


# ── S&P 500 universe ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_sp500_constituents():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise RuntimeError("Wikipedia table not found.")
    rows = table.find_all("tr")[1:]
    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 4:
            raw = cols[0].get_text(strip=True).replace(".", "-")
            cleaned = re.sub(r"[^A-Za-z0-9\-]", "", raw).upper()
            sector = cols[2].get_text(strip=True)
            if cleaned and sector and re.match(r"^[A-Z][A-Z0-9\-]{0,5}$", cleaned):
                data.append({"Ticker": cleaned, "Sector": sector})
    return pd.DataFrame(data)

# BeautifulSoup import guard
try:
    from bs4 import BeautifulSoup
except ImportError:
    st.error("Install beautifulsoup4: pip install beautifulsoup4")
    st.stop()


# ── Prices ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices_batch(tickers):
    tickers_list = list(tickers)
    res = {t: None for t in tickers_list}
    try:
        raw = yf.download(
            tickers_list, period="2d", interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
        for t in tickers_list:
            try:
                if len(tickers_list) == 1:
                    res[t] = float(raw["Close"].iloc[-1])
                else:
                    res[t] = float(raw[t]["Close"].iloc[-1])
            except Exception:
                res[t] = None
    except Exception:
        pass
    return res


# ── Momentum: trailing returns via yfinance history ───────────────────────────
@st.cache_data(ttl=3600)
def fetch_momentum_batch(tickers):
    """
    Returns dict of {ticker: {"ret_1mo": float, "ret_3mo": float, "ret_6mo": float}}
    All values in percent. Missing = None.
    """
    tickers_list = list(tickers)
    out = {t: {} for t in tickers_list}
    try:
        raw = yf.download(
            tickers_list, period="7mo", interval="1mo",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
        for t in tickers_list:
            try:
                if len(tickers_list) == 1:
                    closes = raw["Close"].dropna()
                else:
                    closes = raw[t]["Close"].dropna()
                if len(closes) < 2:
                    continue
                px_now = float(closes.iloc[-1])

                def ret(n_months_back):
                    idx = -(n_months_back + 1)
                    if abs(idx) > len(closes):
                        return None
                    px_then = float(closes.iloc[idx])
                    if px_then <= 0:
                        return None
                    return (px_now / px_then - 1) * 100.0

                out[t] = {
                    "ret_1mo": ret(1),
                    "ret_3mo": ret(3),
                    "ret_6mo": ret(6),
                }
            except Exception:
                out[t] = {}
    except Exception:
        pass
    return out


# ── FMP bulk quote (PE, PEG, MC, 52W, EPS) ───────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_bulk_quotes(tickers, api_key):
    out = {t: {} for t in tickers}
    if not api_key:
        return out
    CHUNK_SIZE = 100
    tickers_list = list(tickers)
    chunks = [tickers_list[i:i + CHUNK_SIZE] for i in range(0, len(tickers_list), CHUNK_SIZE)]
    for chunk in chunks:
        url = "https://financialmodelingprep.com/api/v3/quote/{}?apikey={}".format(
            ",".join(chunk), api_key
        )
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                continue
            for item in data:
                t = str(item.get("symbol", "")).upper().strip()
                if not t:
                    continue
                pe  = safe_float(item.get("pe"))
                peg = safe_float(item.get("priceEarningsToGrowthRatio"))
                mc  = safe_float(item.get("marketCap"))
                hi  = safe_float(item.get("yearHigh"))
                lo  = safe_float(item.get("yearLow"))
                px  = safe_float(item.get("price"))
                eps = safe_float(item.get("eps"))
                if pe  is not None and (pe  <= 0 or pe  > 10_000): pe  = None
                if peg is not None and (peg <= 0 or peg > 500):    peg = None
                out[t] = {
                    "pe": pe, "peg": peg, "mc": mc,
                    "hi52": hi, "lo52": lo, "price": px, "eps": eps,
                    "pe_src": "FMP" if pe is not None else None,
                }
        except Exception:
            pass
        time.sleep(0.3)
    return out


# ── FMP key metrics TTM (Fwd PE, ROE, Debt/Eq, Op Margin, EPS growth) ─────────
@st.cache_data(ttl=86400)
def fetch_fmp_key_metrics_bulk(tickers, api_key):
    out = {t: {} for t in tickers}
    if not api_key:
        return out

    def one(t):
        # key-metrics-ttm gives ROE, debt/eq, margins, EPS growth
        url_km = "https://financialmodelingprep.com/api/v3/key-metrics-ttm/{}?apikey={}".format(t, api_key)
        url_rat = "https://financialmodelingprep.com/api/v3/ratios-ttm/{}?apikey={}".format(t, api_key)
        result = {}
        try:
            r = requests.get(url_km, timeout=15)
            r.raise_for_status()
            d = r.json()
            if isinstance(d, list) and len(d) > 0:
                item = d[0]
                result["fwd_pe"]         = safe_float(item.get("peRatioTTM"))
                result["peg_ttm"]        = safe_float(item.get("pegRatioTTM"))
                result["revenue_growth"] = safe_float(item.get("revenueGrowthTTM"))
                result["roe"]            = safe_float(item.get("roeTTM"))
                result["eps_growth"]     = safe_float(item.get("epsgrowthTTM"))
        except Exception:
            pass
        try:
            r2 = requests.get(url_rat, timeout=15)
            r2.raise_for_status()
            d2 = r2.json()
            if isinstance(d2, list) and len(d2) > 0:
                item2 = d2[0]
                result["debt_eq"]   = safe_float(item2.get("debtEquityRatioTTM"))
                result["op_margin"] = safe_float(item2.get("operatingProfitMarginTTM"))
                if result.get("roe") is None:
                    result["roe"] = safe_float(item2.get("returnOnEquityTTM"))
        except Exception:
            pass
        return t, result

    tickers_list = list(tickers)
    CHUNK_SIZE   = 15
    CHUNK_SLEEP  = 1.5
    MAX_WORKERS  = 5
    chunks = [tickers_list[i:i + CHUNK_SIZE] for i in range(0, len(tickers_list), CHUNK_SIZE)]

    for chunk_idx, chunk in enumerate(chunks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(one, t): t for t in chunk}
            for future in concurrent.futures.as_completed(futures):
                try:
                    t, d = future.result()
                    out[t] = d
                except Exception:
                    pass
        if chunk_idx < len(chunks) - 1:
            time.sleep(CHUNK_SLEEP)

    return out


# ── Yahoo Finance fallback ────────────────────────────────────────────────────
def _fetch_yahoo_one(t, max_retries=2):
    result = {}
    try:
        ticker_obj    = yf.Ticker(t)
        fi            = ticker_obj.fast_info
        price_from_fi = None

        if fi is not None:
            mc_fi         = safe_float(getattr(fi, "market_cap", None))
            hi_fi         = safe_float(getattr(fi, "year_high",  None))
            lo_fi         = safe_float(getattr(fi, "year_low",   None))
            price_from_fi = safe_float(getattr(fi, "last_price", None))
            if mc_fi:  result["mc"]   = mc_fi
            if hi_fi:  result["hi52"] = hi_fi
            if lo_fi:  result["lo52"] = lo_fi

        info = {}
        for attempt in range(max_retries):
            try:
                info = ticker_obj.info or {}
                if any(info.get(k) is not None for k in [
                    "trailingPE", "forwardPE", "trailingEps", "forwardEps",
                    "currentPrice", "regularMarketPrice"
                ]):
                    break
            except Exception:
                pass
            if attempt < max_retries - 1:
                time.sleep(1.0 + random.uniform(0.5, 1.5))

        current_price = safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))
        if current_price is None:
            current_price = price_from_fi

        trailing_pe  = safe_float(info.get("trailingPE"))
        trailing_eps = safe_float(info.get("trailingEps"))
        if trailing_pe is not None and 0 < trailing_pe <= 10_000:
            result["pe"]     = trailing_pe
            result["pe_src"] = "Yahoo"
        elif trailing_eps and trailing_eps > 0 and current_price and current_price > 0:
            result["pe"]     = current_price / trailing_eps
            result["pe_src"] = "Yahoo(calc)"

        forward_pe  = safe_float(info.get("forwardPE"))
        forward_eps = safe_float(info.get("forwardEps"))
        if forward_pe is not None and 0 < forward_pe <= 10_000:
            result["fwd_pe"]     = forward_pe
            result["fwd_pe_src"] = "Yahoo"
        elif forward_eps and forward_eps > 0 and current_price and current_price > 0:
            result["fwd_pe"]     = current_price / forward_eps
            result["fwd_pe_src"] = "Yahoo(calc)"

        eg = safe_float(info.get("earningsGrowth"))
        if eg is not None:
            result["eps_growth"]   = eg * 100.0
            result["growth_src"]   = "Yahoo"

        roe = safe_float(info.get("returnOnEquity"))
        if roe is not None:
            result["roe"] = roe * 100.0

        de = safe_float(info.get("debtToEquity"))
        if de is not None:
            result["debt_eq"] = de / 100.0   # Yahoo gives in percent

        om = safe_float(info.get("operatingMargins"))
        if om is not None:
            result["op_margin"] = om * 100.0

    except Exception:
        pass

    return t, result


@st.cache_data(ttl=86400)
def fetch_yahoo_fallback_parallel(tickers):
    tickers_list = list(tickers)
    out          = {}
    CHUNK_SIZE   = 25
    CHUNK_SLEEP  = 2.0
    MAX_WORKERS  = 6
    chunks = [tickers_list[i:i + CHUNK_SIZE] for i in range(0, len(tickers_list), CHUNK_SIZE)]
    for chunk_idx, chunk in enumerate(chunks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(_fetch_yahoo_one, t): t for t in chunk}
            for future in concurrent.futures.as_completed(futures):
                try:
                    t, d = future.result()
                    out[t] = d
                except Exception:
                    t = futures[future]
                    out[t] = {}
        if chunk_idx < len(chunks) - 1:
            time.sleep(CHUNK_SLEEP + random.uniform(0.0, 0.5))
    return out


# ── Merge all sources (with provenance tracking) ──────────────────────────────
def merge_fundamental_data(fmp_quotes, fmp_metrics, yahoo_fallback, tickers):
    merged = {}
    for t in tickers:
        fq = fmp_quotes.get(t, {})
        fm = fmp_metrics.get(t, {})
        yb = yahoo_fallback.get(t, {})

        def first(*vals):
            for v in vals:
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return v
            return None

        # ROE: FMP returns as decimal (0.15 = 15%), Yahoo already *100'd above
        roe_fmp = safe_float(fm.get("roe"))
        if roe_fmp is not None and abs(roe_fmp) < 5.0:
            roe_fmp = roe_fmp * 100.0   # convert from decimal if FMP returned decimal

        op_fmp = safe_float(fm.get("op_margin"))
        if op_fmp is not None and abs(op_fmp) < 1.0:
            op_fmp = op_fmp * 100.0

        # EPS growth: consistent source labeling
        eps_growth_fmp   = safe_float(fm.get("eps_growth"))
        eps_growth_yahoo = safe_float(yb.get("eps_growth"))
        if eps_growth_fmp is not None and abs(eps_growth_fmp) < 1.0:
            eps_growth_fmp = eps_growth_fmp * 100.0

        # PE source label
        pe_val = first(fq.get("pe"), yb.get("pe"))
        pe_src = fq.get("pe_src") if fq.get("pe") is not None else yb.get("pe_src", "Yahoo")

        merged[t] = {
            "pe":             pe_val,
            "pe_src":         pe_src,
            "fwd_pe":         first(fm.get("fwd_pe"), yb.get("fwd_pe")),
            "peg":            first(fq.get("peg"), fm.get("peg_ttm")),
            "peg_src":        "FMP" if (fq.get("peg") or fm.get("peg_ttm")) else None,
            "mc":             first(fq.get("mc"), yb.get("mc")),
            "hi52":           first(fq.get("hi52"), yb.get("hi52")),
            "lo52":           first(fq.get("lo52"), yb.get("lo52")),
            "revenue_growth": first(fm.get("revenue_growth")),
            "eps_growth":     first(eps_growth_fmp, eps_growth_yahoo),
            "growth_src":     "FMP" if eps_growth_fmp is not None else ("Yahoo" if eps_growth_yahoo is not None else None),
            # Quality factors
            "roe":            first(roe_fmp, yb.get("roe")),
            "debt_eq":        first(fm.get("debt_eq"), yb.get("debt_eq")),
            "op_margin":      first(op_fmp, yb.get("op_margin")),
        }
    return merged


# ── Revenue: yfinance quarterly ───────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_last4_revenue_parallel(tickers):
    tickers_list = list(tickers)
    out = {}

    def one(t):
        try:
            qf = yf.Ticker(t).quarterly_financials
            if qf is not None and "Total Revenue" in qf.index:
                s    = qf.loc["Total Revenue"].sort_index().tail(4)
                vals = [float(v) for v in s.values]
                if len(vals) == 4:
                    return t, vals
        except Exception:
            pass
        return t, [None, None, None, None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for t, vals in ex.map(one, tickers_list):
            out[t] = vals
    return out


# ── Quality composite (0–100) ─────────────────────────────────────────────────
def compute_quality_score(roe, debt_eq, op_margin):
    """
    Returns a 0-100 quality score.
    Each of the 3 sub-factors is scored 0-100 individually then averaged.
    Returns (score, components_dict) for transparency.
    """
    scores = []
    components = {}

    # ROE: scored 0-100, capped at 50% ROE = 100
    if roe is not None:
        roe_score = min(max(roe / 50.0 * 100.0, 0), 100)
        scores.append(roe_score)
        components["roe_score"] = roe_score
    else:
        scores.append(0.0)
        components["roe_score"] = None

    # Debt/Equity: inverted — 0 = best, 2+ = 0 score
    if debt_eq is not None:
        de_score = max(0, min(100, (1 - debt_eq / 2.0) * 100.0))
        scores.append(de_score)
        components["de_score"] = de_score
    else:
        scores.append(0.0)
        components["de_score"] = None

    # Operating Margin: 0-100, capped at 40% margin = 100
    if op_margin is not None:
        om_score = min(max(op_margin / 40.0 * 100.0, 0), 100)
        scores.append(om_score)
        components["om_score"] = om_score
    else:
        scores.append(0.0)
        components["om_score"] = None

    return sum(scores) / len(scores), components


# ── Ranking model v2 (percentile-based, with quality + momentum) ───────────────
def compute_rank_by_sector(scr):
    scr = scr.copy()
    scr["Score"]     = pd.NA
    scr["Rank"]      = pd.NA
    scr["DataFlags"] = ""

    W = FACTOR_WEIGHTS

    for sector in scr["Sector"].dropna().unique().tolist():
        g    = scr[scr["Sector"] == sector].copy()
        elig = g[g["Eligible"]].copy()
        if elig.empty:
            continue

        # ── Factor 1: Valuation (lower Fwd PE or PE = better) ────────────────
        val_input = elig["Fwd P/E"].fillna(elig["P/E"])
        elig["_s_val"] = percentile_score(val_input, ascending=True)

        # ── Factor 2: PEG (lower = better) ───────────────────────────────────
        elig["_s_peg"] = percentile_score(elig["PEG"], ascending=True)

        # ── Factor 3: Quality composite ───────────────────────────────────────
        elig["_s_quality"] = elig["Quality Score"]
        # Normalize quality score to same 0-100 scale within sector
        q_min = elig["_s_quality"].min()
        q_max = elig["_s_quality"].max()
        if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min:
            elig["_s_quality"] = (elig["_s_quality"] - q_min) / (q_max - q_min) * 100.0
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)

        # ── Factor 4: Earnings Trajectory (higher = better) ──────────────────
        elig["_s_etraj"] = percentile_score(elig["Earn Traj"], ascending=False)

        # ── Factor 5: Momentum composite ─────────────────────────────────────
        # Average of 1mo, 3mo, 6mo returns — higher is better
        elig["_momentum_avg"] = elig[["Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%"]].mean(axis=1)
        elig["_s_mom"] = percentile_score(elig["_momentum_avg"], ascending=False)

        # ── Composite score ───────────────────────────────────────────────────
        raw_score = (
            W["valuation"]  * elig["_s_val"]     +
            W["peg"]        * elig["_s_peg"]      +
            W["quality"]    * elig["_s_quality"]  +
            W["earn_traj"]  * elig["_s_etraj"]    +
            W["momentum"]   * elig["_s_mom"]
        )

        # ── Missing data penalty ──────────────────────────────────────────────
        factor_cols = ["P/E", "PEG", "Quality Score", "Earn Traj", "_momentum_avg"]
        penalties = elig.apply(
            lambda row: missing_factor_penalty(row, factor_cols), axis=1
        )
        raw_score = raw_score * penalties

        elig["Score"] = raw_score
        elig = elig.sort_values("Score", ascending=False)
        elig["Rank"] = range(1, len(elig) + 1)

        scr.loc[elig.index, "Score"] = elig["Score"]
        scr.loc[elig.index, "Rank"]  = elig["Rank"]

    return scr


# ── Build screener table ──────────────────────────────────────────────────────
def build_screener_table(universe_df, prices_map, merged_map, revenue_map, momentum_map):
    rows = []
    for _, r in universe_df.iterrows():
        t   = r["Ticker"]
        sec = r["Sector"]

        price = to_num(prices_map.get(t, None))
        fi    = merged_map.get(t, {})
        mc    = to_num(fi.get("mc"))
        pe    = to_num(fi.get("pe"))
        fwd   = to_num(fi.get("fwd_pe"))
        hi    = to_num(fi.get("hi52"))
        lo    = to_num(fi.get("lo52"))
        roe   = to_num(fi.get("roe"))
        de    = to_num(fi.get("debt_eq"))
        om    = to_num(fi.get("op_margin"))

        pe_src    = fi.get("pe_src", "—")
        peg_src   = fi.get("peg_src", "—")
        grow_src  = fi.get("growth_src", "—")

        # 52-week position
        pos52 = None
        if pd.notna(price) and pd.notna(hi) and pd.notna(lo) and hi != lo:
            pos52 = float((price - lo) / (hi - lo) * 100.0)

        # Revenue
        rev4                = revenue_map.get(t, [None, None, None, None])
        rq1, rq2, rq3, rq4 = [to_num(x) for x in rev4]
        growth              = revenue_growth_pct_cagr([rq1, rq2, rq3, rq4])

        # PEG — consistent source: prefer FMP analyst PEG, then compute from EPS growth
        peg_direct = to_num(fi.get("peg"))
        peg        = None
        peg_method = "—"
        if pd.notna(peg_direct):
            peg        = float(peg_direct)
            peg_method = "FMP direct"
        else:
            pe_for_peg     = fwd if pd.notna(fwd) else pe
            eps_growth     = fi.get("eps_growth")
            growth_for_peg = None
            if eps_growth is not None:
                eg = float(eps_growth)
                if eg >= MIN_GROWTH_PCT_FOR_PEG:
                    growth_for_peg = eg
                    peg_method     = "{} EPS growth".format(grow_src or "est")
            elif growth is not None and growth >= MIN_GROWTH_PCT_FOR_PEG:
                growth_for_peg = growth
                peg_method     = "Rev CAGR fallback"
            if pd.notna(pe_for_peg) and growth_for_peg is not None:
                peg = float(pe_for_peg) / float(growth_for_peg)

        if peg is not None and (peg <= 0 or peg > 500):
            peg = None

        # Earnings trajectory
        earn_traj = None
        if pd.notna(pe) and pd.notna(fwd) and fwd > 0:
            earn_traj = float(pe) / float(fwd)

        # Quality score
        q_score, _ = compute_quality_score(
            float(roe) if pd.notna(roe) else None,
            float(de)  if pd.notna(de)  else None,
            float(om)  if pd.notna(om)  else None,
        )

        # Momentum
        mom = momentum_map.get(t, {})
        ret_1mo = to_num(mom.get("ret_1mo"))
        ret_3mo = to_num(mom.get("ret_3mo"))
        ret_6mo = to_num(mom.get("ret_6mo"))

        # Data source summary
        sources_used = []
        if pe_src and pe_src != "—":    sources_used.append("PE:{}".format(pe_src))
        if peg_src and peg_src != "—":  sources_used.append("PEG:{}".format(peg_src))
        if grow_src and grow_src != "—": sources_used.append("G:{}".format(grow_src))
        data_src_label = " | ".join(sources_used) if sources_used else "Yahoo only"

        rows.append({
            "Ticker":             t,
            "Sector":             sec,
            "Price":              price,
            "Mkt Cap":            mc,
            "P/E":                pe,
            "Fwd P/E":            fwd,
            "PEG":                to_num(peg),
            "PEG Method":         peg_method,
            "Earn Traj":          to_num(earn_traj),
            "52W Pos%":           to_num(pos52),
            "ROE%":               roe,
            "Debt/Eq":            de,
            "Op Margin%":         om,
            "Quality Score":      to_num(q_score),
            "Ret 1Mo%":           ret_1mo,
            "Ret 3Mo%":           ret_3mo,
            "Ret 6Mo%":           ret_6mo,
            "Data Sources":       data_src_label,
            "Eligible":           True,
            "Rev Q1":             rq1,
            "Rev Q2":             rq2,
            "Rev Q3":             rq3,
            "Rev Q4":             rq4,
            "Rev Growth% (CAGR)": to_num(growth),
        })

    scr = pd.DataFrame(rows)
    if scr.empty:
        return scr

    num_cols = [
        "Price", "Mkt Cap", "P/E", "Fwd P/E", "PEG", "Earn Traj",
        "52W Pos%", "ROE%", "Debt/Eq", "Op Margin%", "Quality Score",
        "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%",
        "Rev Q1", "Rev Q2", "Rev Q3", "Rev Q4", "Rev Growth% (CAGR)",
    ]
    for c in num_cols:
        if c in scr.columns:
            scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)

    if "Rank" not in scr.columns:
        scr["Rank"] = pd.NA

    return scr


# ── Quality pass/fail badge ───────────────────────────────────────────────────
def quality_flag(roe, debt_eq, op_margin):
    flags = []
    if roe is not None and not pd.isna(roe):
        if roe < QUALITY_THRESHOLDS["roe_min"]:
            flags.append("ROE<10%")
    if debt_eq is not None and not pd.isna(debt_eq):
        if debt_eq > QUALITY_THRESHOLDS["debt_eq_max"]:
            flags.append("D/E>2")
    if op_margin is not None and not pd.isna(op_margin):
        if op_margin < QUALITY_THRESHOLDS["op_margin_min"]:
            flags.append("Margin<5%")
    return ", ".join(flags) if flags else "✓ Pass"


# ── KPI panel ─────────────────────────────────────────────────────────────────
def render_sector_kpi_panel(scr, sector_sel):
    def _kpi(label, value, sub, color="#ffffff"):
        return (
            "<div style='background:#1e1e2e;border-radius:10px;padding:14px 16px;"
            "text-align:center;margin:2px;'>"
            "<div style='color:#aaa;font-size:11px;margin-bottom:4px;'>{}</div>"
            "<div style='color:{};font-size:20px;font-weight:700;'>{}</div>"
            "<div style='color:#666;font-size:10px;margin-top:3px;'>{}</div>"
            "</div>"
        ).format(label, color, value, sub)

    is_all      = (sector_sel == "All Sectors")
    label       = "All Sectors (S&P 500)" if is_all else sector_sel
    total_mc    = scr["Mkt Cap"].sum()
    sdata       = scr.copy() if is_all else scr[scr["Sector"] == sector_sel].copy()
    sector_mc   = sdata["Mkt Cap"].sum()
    sector_pct  = 100.0 if is_all else (sector_mc / total_mc * 100.0 if total_mc > 0 else 0.0)

    med_pe      = sdata["P/E"].median()
    med_fwdpe   = sdata["Fwd P/E"].median()
    med_peg     = sdata["PEG"].median()
    med_qual    = sdata["Quality Score"].median()
    med_mom     = sdata[["Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%"]].mean(axis=1).median()

    st.markdown(
        "<div style='background:#12122a;border:1px solid #2a2a4a;border-radius:12px;"
        "padding:16px 20px;margin-bottom:16px;'>"
        "<span style='color:#aaa;font-size:13px;'>Sector Analysis  </span>"
        "<span style='color:#fff;font-size:14px;font-weight:700;'>{}</span>"
        "</div>".format(label),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(_kpi("Sector Mkt Cap",    fmt_mc(sector_mc),    "sector total"),                                        unsafe_allow_html=True)
    c2.markdown(_kpi("S&P 500 Mkt Cap",   fmt_mc(total_mc),     "all 500 stocks"),                                      unsafe_allow_html=True)
    c3.markdown(_kpi("Sector Share",      "{:.1f}%".format(sector_pct), "{} stocks".format(len(sdata))),                unsafe_allow_html=True)
    c4.markdown(_kpi("Median P/E → FwdPE",
                     "{:.1f} → {:.1f}".format(med_pe, med_fwdpe) if pd.notna(med_pe) and pd.notna(med_fwdpe) else "N/A",
                     "trailing → forward", "#facc15"),                                                                  unsafe_allow_html=True)
    c5.markdown(_kpi("Median Quality",    "{:.0f}/100".format(med_qual) if pd.notna(med_qual) else "N/A",
                     "ROE + D/E + Margins", "#4ade80"),                                                                 unsafe_allow_html=True)
    c6.markdown(_kpi("Median 3Mo Return", "{:.1f}%".format(med_mom)   if pd.notna(med_mom)   else "N/A",
                     "avg of 1/3/6mo", "#a78bfa"),                                                                      unsafe_allow_html=True)

    if not is_all:
        top3 = sdata[sdata["Rank"].notna()].sort_values("Rank").head(3)
        badges = "  ".join(
            "<span style='background:#1a2a4a;color:#93c5fd;padding:3px 10px;"
            "border-radius:6px;font-weight:700;font-size:13px;'>{}</span>".format(row["Ticker"])
            for _, row in top3.iterrows()
        )
        st.markdown(
            "<div style='background:#1e1e2e;border-radius:10px;padding:14px 16px;margin-bottom:12px;'>"
            "<div style='color:#aaa;font-size:11px;margin-bottom:8px;'>Top Ranked Stocks in Sector</div>"
            "<div>{}</div>"
            "<div style='color:#555;font-size:10px;margin-top:8px;'>"
            "Score = Valuation 30% + PEG 20% + Quality 25% + Earn Traj 15% + Momentum 10%  ·  "
            "Missing data penalized (−15% per 2 missing factors, −30% for 3+)"
            "</div>"
            "</div>".format(badges or "<span style='color:#555;'>No ranked stocks</span>"),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── APP ───────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="S&P 500 Screener", layout="wide", page_icon="📊")

st.markdown(
    "<style>"
    "div[data-testid='stDataFrame'] table { font-size: 13px; }"
    ".stDataFrame thead th { background:#1a1a2e; color:#93c5fd; font-weight:700; }"
    "</style>",
    unsafe_allow_html=True,
)

st.markdown("## S&P 500 Fundamental Screener v2")
st.caption("5-factor ranking · Quality-aware · Momentum-included · Percentile scoring · Source-transparent")

col_r, col_t = st.columns([1, 6])
with col_r:
    if st.button("🔄 Refresh"):
        st.cache_data.clear()
        st.rerun()
with col_t:
    st.caption(
        "Last loaded: {} · Prices & Momentum: 1hr cache · Fundamentals: 24hr cache".format(
            datetime.now().strftime("%I:%M %p")
        )
    )

fmp_key = get_fmp_key()
if not fmp_key:
    st.warning(
        "No FMP API key found. Add [fmp] api_key to .streamlit/secrets.toml. "
        "Quality metrics (ROE, Debt/Eq, Op Margin) and EPS growth require FMP. "
        "Falling back to Yahoo Finance only — quality scores will be limited."
    )

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading S&P 500 universe from Wikipedia..."):
    sp500 = fetch_sp500_constituents()

if sp500.empty:
    st.error("Failed to load S&P 500 universe.")
    st.stop()

universe_df = sp500.copy().reset_index(drop=True)
tickers     = tuple(universe_df["Ticker"].tolist())

with st.spinner("Fetching prices ({} tickers)...".format(len(tickers))):
    prices = fetch_prices_batch(tickers)

with st.spinner("Fetching momentum data (1mo / 3mo / 6mo returns)..."):
    momentum = fetch_momentum_batch(tickers)

fmp_quotes  = {}
fmp_metrics = {}
if fmp_key:
    with st.spinner("FMP: bulk PE, PEG, Market Cap, 52W..."):
        fmp_quotes = fetch_fmp_bulk_quotes(tickers, fmp_key)

    missing_quality = tuple(
        t for t in tickers
        if fmp_metrics.get(t, {}).get("roe") is None
        or fmp_quotes.get(t, {}).get("fwd_pe") is None
    )
    with st.spinner("FMP: quality metrics (ROE, D/E, Margins, EPS growth) for {} tickers...".format(len(missing_quality))):
        fmp_metrics = fetch_fmp_key_metrics_bulk(missing_quality, fmp_key)

missing_after_fmp = tuple(
    t for t in tickers
    if fmp_quotes.get(t, {}).get("pe") is None or (
        fmp_quotes.get(t, {}).get("fwd_pe") is None
        and fmp_metrics.get(t, {}).get("fwd_pe") is None
    )
)

yahoo_fallback = {}
if missing_after_fmp:
    with st.spinner("Yahoo fallback: {} tickers missing from FMP...".format(len(missing_after_fmp))):
        yahoo_fallback = fetch_yahoo_fallback_parallel(missing_after_fmp)

with st.spinner("Merging data sources..."):
    merged_map = merge_fundamental_data(fmp_quotes, fmp_metrics, yahoo_fallback, tickers)

with st.spinner("Fetching quarterly revenue..."):
    rev_map = fetch_last4_revenue_parallel(tickers)

# ── Coverage banner ───────────────────────────────────────────────────────────
total_t   = len(tickers)
has_pe    = sum(1 for t in tickers if merged_map.get(t, {}).get("pe")         is not None)
has_fwdpe = sum(1 for t in tickers if merged_map.get(t, {}).get("fwd_pe")     is not None)
has_roe   = sum(1 for t in tickers if merged_map.get(t, {}).get("roe")        is not None)
has_peg   = sum(1 for t in tickers if merged_map.get(t, {}).get("peg")        is not None)

st.info(
    "Data coverage — "
    "P/E: {}/{} ({:.0f}%) · "
    "Fwd P/E: {}/{} ({:.0f}%) · "
    "PEG: {}/{} ({:.0f}%) · "
    "ROE: {}/{} ({:.0f}%) · "
    "Sources: FMP primary + Yahoo fallback".format(
        has_pe,    total_t, has_pe    / total_t * 100,
        has_fwdpe, total_t, has_fwdpe / total_t * 100,
        has_peg,   total_t, has_peg   / total_t * 100,
        has_roe,   total_t, has_roe   / total_t * 100,
    )
)

# ── Build table ───────────────────────────────────────────────────────────────
scr = build_screener_table(universe_df, prices, merged_map, rev_map, momentum)

# ── Filters ───────────────────────────────────────────────────────────────────
st.markdown("### Filters")

with st.expander("Valuation & Size Filters", expanded=True):
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    all_sectors = sorted(scr["Sector"].dropna().unique().tolist())
    sector_sel  = fc1.selectbox("Sector", ["All Sectors"] + all_sectors)
    sort_by     = fc2.selectbox("Sort by", [
        "Sector then Rank",
        "Score high to low",
        "Price low to high",
        "Price high to low",
        "Mkt Cap high to low",
        "PE low to high",
        "Fwd PE low to high",
        "PEG low to high",
        "Quality Score high",
        "Earn Traj high to low",
        "Rev Growth high to low",
        "Momentum (3Mo) high",
        "52W Pos low to high",
    ])
    pe_max   = fc3.number_input("Max PE",             value=9999, step=50)
    peg_max  = fc4.number_input("Max PEG",            value=999.0, step=1.0)
    mc_min_b = fc5.number_input("Min Market Cap ($B)", value=0, step=5)

with st.expander("Quality Filters (Hard Screens)", expanded=False):
    qc1, qc2, qc3, qc4 = st.columns(4)
    roe_min_f    = qc1.number_input("Min ROE (%)",         value=0.0,  step=5.0,
                                     help="0 = show all. Filter to profitable, high-return businesses.")
    de_max_f     = qc2.number_input("Max Debt/Equity",     value=99.0, step=0.5,
                                     help="99 = show all. Lower = less financial risk.")
    om_min_f     = qc3.number_input("Min Operating Margin%", value=0.0, step=5.0,
                                     help="0 = show all. Filters out unprofitable or thin-margin stocks.")
    qual_min_f   = qc4.number_input("Min Quality Score",   value=0.0,  step=5.0,
                                     help="0 = show all. Composite of ROE + D/E + Op Margin (0–100).")

with st.expander("Momentum Filters", expanded=False):
    mc1, mc2 = st.columns(2)
    mom_min_3mo = mc1.number_input("Min 3Mo Return (%)",  value=-999.0, step=5.0, help="-999 = show all")
    hide_no_pe  = mc2.checkbox("Hide stocks with no P/E or Fwd P/E data", value=False)

render_sector_kpi_panel(scr, sector_sel)

# ── Apply filters ─────────────────────────────────────────────────────────────
filt = scr.copy()
if sector_sel != "All Sectors":
    filt = filt[filt["Sector"] == sector_sel]
filt = filt[(filt["Mkt Cap"].isna())       | (filt["Mkt Cap"]       >= mc_min_b * 1e9)]
filt = filt[(filt["P/E"].isna())           | (filt["P/E"]           <= pe_max)]
filt = filt[(filt["PEG"].isna())           | (filt["PEG"]           <= peg_max)]
filt = filt[(filt["ROE%"].isna())          | (filt["ROE%"]          >= roe_min_f)]
filt = filt[(filt["Debt/Eq"].isna())       | (filt["Debt/Eq"]       <= de_max_f)]
filt = filt[(filt["Op Margin%"].isna())    | (filt["Op Margin%"]    >= om_min_f)]
filt = filt[(filt["Quality Score"].isna()) | (filt["Quality Score"] >= qual_min_f)]
filt = filt[(filt["Ret 3Mo%"].isna())      | (filt["Ret 3Mo%"]      >= mom_min_3mo)]
if hide_no_pe:
    filt = filt[filt["P/E"].notna() | filt["Fwd P/E"].notna()]

sort_map = {
    "Sector then Rank":       (["Sector", "Rank"],           [True,  True]),
    "Score high to low":      (["Score"],                    [False]),
    "Price low to high":      (["Price"],                    [True]),
    "Price high to low":      (["Price"],                    [False]),
    "Mkt Cap high to low":    (["Mkt Cap"],                  [False]),
    "PE low to high":         (["P/E"],                      [True]),
    "Fwd PE low to high":     (["Fwd P/E"],                  [True]),
    "PEG low to high":        (["PEG"],                      [True]),
    "Quality Score high":     (["Quality Score"],            [False]),
    "Earn Traj high to low":  (["Earn Traj"],                [False]),
    "Rev Growth high to low": (["Rev Growth% (CAGR)"],       [False]),
    "Momentum (3Mo) high":    (["Ret 3Mo%"],                 [False]),
    "52W Pos low to high":    (["52W Pos%"],                 [True]),
}
sort_cols, sort_asc = sort_map.get(sort_by, (["Sector", "Rank"], [True, True]))
filt = filt.sort_values(sort_cols, ascending=sort_asc, na_position="last")

st.caption(
    "Showing {} of {} stocks  ·  Sector: {}  ·  Sort: {}".format(
        len(filt), len(scr), sector_sel, sort_by
    )
)

# ── Build display columns ─────────────────────────────────────────────────────
disp = filt.copy()
disp["Price ($)"]      = disp["Price"].round(2)
disp["Mkt Cap ($B)"]   = (disp["Mkt Cap"]   / 1e9).round(2)
disp["Rev Q1 ($B)"]    = (disp["Rev Q1"]    / 1e9).round(2)
disp["Rev Q2 ($B)"]    = (disp["Rev Q2"]    / 1e9).round(2)
disp["Rev Q3 ($B)"]    = (disp["Rev Q3"]    / 1e9).round(2)
disp["Rev Q4 ($B)"]    = (disp["Rev Q4"]    / 1e9).round(2)
disp["Quality Flag"]   = disp.apply(
    lambda row: quality_flag(row.get("ROE%"), row.get("Debt/Eq"), row.get("Op Margin%")),
    axis=1
)

for c in ["P/E", "Fwd P/E", "PEG", "Earn Traj", "52W Pos%", "Rev Growth% (CAGR)",
          "ROE%", "Debt/Eq", "Op Margin%", "Quality Score",
          "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Score"]:
    if c in disp.columns:
        disp[c] = disp[c].round(2)

disp["Rank"] = disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

DISPLAY_COLS = [
    "Ticker", "Sector",
    "Price ($)", "Mkt Cap ($B)",
    "P/E", "Fwd P/E", "PEG", "PEG Method",
    "Earn Traj",
    "ROE%", "Debt/Eq", "Op Margin%", "Quality Score", "Quality Flag",
    "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%",
    "52W Pos%",
    "Score", "Rank",
    "Rev Q1 ($B)", "Rev Q2 ($B)", "Rev Q3 ($B)", "Rev Q4 ($B)",
    "Rev Growth% (CAGR)",
    "Data Sources",
]
disp_final = disp[[c for c in DISPLAY_COLS if c in disp.columns]].copy()

st.dataframe(disp_final, use_container_width=True, height=680)

# ── Export ────────────────────────────────────────────────────────────────────
csv_data = disp_final.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download filtered results as CSV",
    data=csv_data,
    file_name="sp500_screener_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M")),
    mime="text/csv",
)

# ── Legend ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Column Reference Guide")

tab1, tab2, tab3, tab4 = st.tabs(["Valuation", "Quality", "Momentum & Position", "Ranking"])

with tab1:
    st.markdown(
        "**P/E (Price-to-Earnings)** — Stock price ÷ trailing 12-month EPS. "
        "Lower = cheaper relative to current earnings. e.g. P/E 20 = $20 paid per $1 of annual earnings.\n\n"
        "**Fwd P/E (Forward P/E)** — Price ÷ next 12-month estimated EPS. "
        "Fwd P/E lower than P/E = earnings growing. Higher = earnings shrinking.\n\n"
        "**PEG (Price/Earnings to Growth)** — P/E ÷ EPS growth rate. "
        "PEG < 1 = potentially undervalued for its growth. 1–2 = fairly valued. > 2 = expensive. "
        "Only computed when growth ≥ 5%. The **PEG Method** column tells you the data source used: "
        "'FMP direct' uses analyst consensus EPS estimates; 'Rev CAGR fallback' uses your computed revenue CAGR — "
        "these are not directly comparable so check the method column.\n\n"
        "**Earn Traj (Earnings Trajectory)** — Trailing P/E ÷ Forward P/E. "
        "> 1 = earnings expected to grow. < 1 = earnings expected to shrink."
    )

with tab2:
    st.markdown(
        "**ROE% (Return on Equity)** — Net income ÷ shareholders equity × 100. "
        "Measures how efficiently management generates profit from equity. "
        "Benchmark: > 15% is strong, < 10% is weak.\n\n"
        "**Debt/Eq (Debt-to-Equity)** — Total debt ÷ total equity. "
        "Lower = less financial risk. > 2.0 flags elevated leverage.\n\n"
        "**Op Margin% (Operating Margin)** — Operating income ÷ revenue × 100. "
        "Higher = more profitable core business. < 5% flags thin margins.\n\n"
        "**Quality Score (0–100)** — Composite of ROE (capped at 50% = 100 pts), "
        "Debt/Equity (0 = 100 pts, 2.0+ = 0 pts), and Operating Margin (capped at 40% = 100 pts). "
        "Averaged equally. Missing factors score 0 — penalizing data gaps.\n\n"
        "**Quality Flag** — Pass/Fail against thresholds: ROE ≥ 10%, D/E ≤ 2.0, Op Margin ≥ 5%. "
        "Shows which thresholds a stock fails."
    )

with tab3:
    st.markdown(
        "**Ret 1Mo% / 3Mo% / 6Mo%** — Trailing price return over 1, 3, and 6 months. "
        "Positive momentum (especially 3–6mo) is one of the most documented predictors of "
        "near-term outperformance. Negative momentum = recent underperformer.\n\n"
        "**52W Pos%** — Where price sits between its 52-week low and high. "
        "0% = at 52-week low. 100% = at 52-week high. "
        "Lower = more potential upside room but also possible value trap."
    )

with tab4:
    st.markdown(
        "**Score (0–100)** — Composite percentile score within each sector. "
        "Each factor is converted to a 0–100 percentile within the sector before weighting. "
        "This preserves magnitude: a PE=5 stock scores MUCH higher than PE=14, not just 1 rank better.\n\n"
        "**Rank** — Position within sector based on Score (1 = best in sector).\n\n"
        "**Factor weights:** Valuation 30% · Quality 25% · PEG 20% · Earnings Trajectory 15% · Momentum 10%\n\n"
        "**Missing data penalty:** Stocks missing 2 factors: score × 0.85. "
        "Missing 3+ factors: score × 0.70. "
        "A strong signal on one factor alone cannot offset missing everything else.\n\n"
        "**PEG method consistency note:** If a stock's PEG was computed from 'Rev CAGR fallback', "
        "its PEG is not directly comparable to 'FMP direct' PEG values. "
        "This can distort PEG-factor rankings — treat Rev CAGR PEG stocks with caution in rankings.\n\n"
        "**What this ranking does NOT capture:** Earnings quality, cash flow vs net income divergence, "
        "insider ownership, competitive moat, management track record, macro sector tailwinds. "
        "Use this as a first-pass filter, not a buy signal."
    )

st.markdown(
    "**Data Sources** — Primary: FMP bulk API (PE, PEG, Market Cap, ROE, D/E, Op Margin, EPS growth) · "
    "Fallback: Yahoo Finance (fills gaps where FMP is missing) · "
    "Revenue & Momentum: Yahoo Finance · "
    "S&P 500 Universe: Wikipedia GICS sector list"
)
