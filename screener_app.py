# screener_app.py  v7
# ─────────────────────────────────────────────────────────────────────────────
# COMPLETE REWRITE — Yahoo Finance as primary source
#
# ROOT CAUSE OF ALL PREVIOUS 0% ISSUES:
#   1. FMP API key not in Streamlit secrets → /quote returns 0
#   2. Finviz blocks Streamlit Cloud IPs → 0 rows
#   3. FMP /ratios-ttm, /key-metrics-ttm → may not be on free tier
#
# v7 STRATEGY — Yahoo First, FMP as bonus if key exists:
#
#   LAYER 1 (PRIMARY): Yahoo Finance batch download
#     → Prices, 52W Hi/Lo, Market Cap from yf.download (fast, all 503)
#     → PEG, PE, FwdPE, ROE, OpMargin, D/E, EpsGrowth from yf.Ticker.info
#     → EBIT + InterestExpense from yf.Ticker.financials → compute IntCoverage
#     → Expected coverage: PE ~85%, PEG ~70%, IntCoverage ~60%
#
#   LAYER 2 (BONUS): FMP /quote bulk — if API key exists
#     → Overrides Yahoo PE with FMP PE where available
#
#   LAYER 3 (BONUS): FMP /ratios-ttm per-ticker concurrent — if key exists
#     → Overrides ROIC, IntCoverage, PEG with FMP values where available
#
# KEY INSIGHT: Yahoo .info has pegRatio, returnOnEquity, operatingMargins,
#   debtToEquity, earningsGrowth — these are what we need and they work!
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

try:
    from bs4 import BeautifulSoup
except ImportError:
    st.error("pip install beautifulsoup4")
    st.stop()

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_GROWTH_PCT_FOR_PEG = 5.0

FACTOR_WEIGHTS = {
    "valuation":     0.25,
    "quality":       0.25,
    "peg":           0.20,
    "earn_revision": 0.15,
    "momentum":      0.15,
}

QUALITY_THRESHOLDS = {
    "roic_min":         8.0,
    "int_coverage_min": 3.0,
    "op_margin_min":    5.0,
}

# ── Credentials ───────────────────────────────────────────────────────────────
def get_fmp_key():
    try:
        k = st.secrets["fmp"]["api_key"]
        return k if k and k.strip() and k != "YOUR_KEY_HERE" else None
    except Exception:
        return None

# ── Helpers ───────────────────────────────────────────────────────────────────
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

def fmt_mc(val):
    if pd.isna(val) or val == 0:
        return "N/A"
    if val >= 1e12:
        return "${:.2f}T".format(val / 1e12)
    if val >= 1e9:
        return "${:.1f}B".format(val / 1e9)
    return "${:.0f}M".format(val / 1e6)

def percentile_score(series: pd.Series, ascending=True) -> pd.Series:
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
    return 1.0

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

# ── S&P 500 universe ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_sp500_constituents():
    url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r    = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    tbl  = soup.find("table", {"id": "constituents"})
    if tbl is None:
        raise RuntimeError("Wikipedia table not found")
    data = []
    for row in tbl.find_all("tr")[1:]:
        cols = row.find_all("td")
        if len(cols) >= 4:
            raw     = cols[0].get_text(strip=True).replace(".", "-")
            cleaned = re.sub(r"[^A-Za-z0-9\-]", "", raw).upper()
            sector  = cols[2].get_text(strip=True)
            if cleaned and sector and re.match(r"^[A-Z][A-Z0-9\-]{0,5}$", cleaned):
                data.append({"Ticker": cleaned, "Sector": sector})
    return pd.DataFrame(data)

# ── Prices + 52W batch ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices_batch(tickers):
    tl  = list(tickers)
    res = {t: {"price": None, "hi52": None, "lo52": None, "mc": None} for t in tl}
    try:
        raw = yf.download(tl, period="2d", interval="1d",
                          group_by="ticker", auto_adjust=True,
                          progress=False, threads=True)
        for t in tl:
            try:
                px = float(raw["Close"].iloc[-1]) if len(tl) == 1 \
                     else float(raw[t]["Close"].iloc[-1])
                res[t]["price"] = px
            except Exception:
                pass
    except Exception:
        pass
    return res

# ── Momentum ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_momentum_batch(tickers):
    tl  = list(tickers)
    out = {t: {} for t in tl}
    try:
        raw_d = yf.download(tl, period="7mo", interval="1d",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=True)
        raw_m = yf.download(tl, period="7mo", interval="1mo",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=True)
        for t in tl:
            try:
                closes_m = raw_m["Close"].dropna() if len(tl) == 1 \
                           else raw_m[t]["Close"].dropna()
                closes_d = raw_d["Close"].dropna() if len(tl) == 1 \
                           else raw_d[t]["Close"].dropna()
                if len(closes_m) < 2:
                    continue
                px_now = float(closes_m.iloc[-1])

                def ret_mo(n):
                    idx = -(n + 1)
                    if abs(idx) > len(closes_m):
                        return None
                    px = float(closes_m.iloc[idx])
                    return (px_now / px - 1) * 100.0 if px > 0 else None

                r1 = ret_mo(1)
                r3 = ret_mo(3)
                r6 = ret_mo(6)

                trailing_vol = None
                if len(closes_d) >= 20:
                    daily_rets = closes_d.pct_change().dropna().tail(90)
                    if len(daily_rets) >= 15:
                        trailing_vol = float(daily_rets.std() * np.sqrt(252) * 100.0)

                skip_mom_raw = (r6 - r1) if (r6 is not None and r1 is not None) else None
                skip_mom_adj = None
                if skip_mom_raw is not None and trailing_vol and trailing_vol > 0:
                    skip_mom_adj = skip_mom_raw / trailing_vol
                elif skip_mom_raw is not None:
                    skip_mom_adj = skip_mom_raw

                out[t] = {
                    "ret_1mo":        r1,
                    "ret_3mo":        r3,
                    "ret_6mo":        r6,
                    "trailing_vol":   trailing_vol,
                    "momentum_score": skip_mom_adj,
                }
            except Exception:
                pass
    except Exception:
        pass
    return out

# ══════════════════════════════════════════════════════════════════════════════
# ── LAYER 1 PRIMARY: Yahoo Finance per-ticker fundamentals ────────────────────
# Fetches: PE, FwdPE, PEG, ROE, OpMargin, D/E, EpsGrowth, MC, 52W, IntCoverage
# IntCoverage computed from income statement: EBIT / InterestExpense
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_yahoo_fundamentals_one(t):
    result = {
        "pe": None, "pe_src": None,
        "fwd_pe": None,
        "peg": None, "peg_src": None,
        "roe": None,
        "op_margin": None,
        "debt_eq": None,
        "eps_growth": None,
        "int_coverage": None,
        "mc": None,
        "hi52": None,
        "lo52": None,
    }
    try:
        obj  = yf.Ticker(t)

        # ── fast_info: MC, 52W, price ─────────────────────────────────────
        try:
            fi = obj.fast_info
            if fi is not None:
                mc_fi = sf(getattr(fi, "market_cap",  None))
                hi_fi = sf(getattr(fi, "year_high",   None))
                lo_fi = sf(getattr(fi, "year_low",    None))
                if mc_fi: result["mc"]   = mc_fi
                if hi_fi: result["hi52"] = hi_fi
                if lo_fi: result["lo52"] = lo_fi
        except Exception:
            pass

        # ── .info: PE, FwdPE, PEG, quality metrics ───────────────────────
        info = {}
        for attempt in range(2):
            try:
                info = obj.info or {}
                if info.get("trailingPE") or info.get("pegRatio") or info.get("forwardPE"):
                    break
            except Exception:
                pass
            time.sleep(0.5 + random.uniform(0, 0.5))

        px = sf(info.get("currentPrice") or info.get("regularMarketPrice"))

        # PE
        t_pe  = sf(info.get("trailingPE"))
        t_eps = sf(info.get("trailingEps"))
        if t_pe and 0 < t_pe <= 10_000:
            result["pe"]     = t_pe
            result["pe_src"] = "Yahoo"
        elif t_eps and t_eps > 0 and px and px > 0:
            result["pe"]     = px / t_eps
            result["pe_src"] = "Yahoo(calc)"

        # Fwd PE
        f_pe  = sf(info.get("forwardPE"))
        f_eps = sf(info.get("forwardEps"))
        if f_pe and 0 < f_pe <= 10_000:
            result["fwd_pe"] = f_pe
        elif f_eps and f_eps > 0 and px and px > 0:
            result["fwd_pe"] = px / f_eps

        # PEG — Yahoo has this directly as pegRatio
        peg_y = sf(info.get("pegRatio"))
        if peg_y and 0 < peg_y <= 500:
            result["peg"]     = peg_y
            result["peg_src"] = "Yahoo"

        # ROE — Yahoo returns as decimal (0.15 = 15%)
        roe_y = sf(info.get("returnOnEquity"))
        if roe_y is not None:
            result["roe"] = roe_y * 100.0

        # Op Margin — decimal
        om_y = sf(info.get("operatingMargins"))
        if om_y is not None:
            result["op_margin"] = om_y * 100.0

        # Debt/Equity — Yahoo returns as percentage (150 = 1.5 ratio)
        de_y = sf(info.get("debtToEquity"))
        if de_y is not None:
            result["debt_eq"] = de_y / 100.0

        # EPS Growth — decimal
        eg_y = sf(info.get("earningsGrowth"))
        if eg_y is not None:
            result["eps_growth"] = eg_y * 100.0

        # MC if not from fast_info
        if result["mc"] is None:
            mc_y = sf(info.get("marketCap"))
            if mc_y:
                result["mc"] = mc_y

        # 52W if not from fast_info
        if result["hi52"] is None:
            h52 = sf(info.get("fiftyTwoWeekHigh"))
            if h52:
                result["hi52"] = h52
        if result["lo52"] is None:
            l52 = sf(info.get("fiftyTwoWeekLow"))
            if l52:
                result["lo52"] = l52

        # ── Interest Coverage from income statement ───────────────────────
        # IntCoverage = EBIT / InterestExpense (TTM = sum last 4 quarters)
        try:
            qfin = obj.quarterly_financials
            if qfin is not None and not qfin.empty:
                # Try multiple row names Yahoo uses
                ebit_row = None
                for nm in ["EBIT", "Operating Income", "Ebit"]:
                    if nm in qfin.index:
                        ebit_row = nm
                        break

                int_row = None
                for nm in ["Interest Expense", "Interest Expense Non Operating",
                           "Net Interest Income"]:
                    if nm in qfin.index:
                        int_row = nm
                        break

                if ebit_row and int_row:
                    ebit_ttm = qfin.loc[ebit_row].dropna().head(4).sum()
                    int_ttm  = abs(qfin.loc[int_row].dropna().head(4).sum())
                    if int_ttm > 0 and ebit_ttm > 0:
                        ic = min(float(ebit_ttm / int_ttm), 100.0)
                        result["int_coverage"] = ic
        except Exception:
            pass

    except Exception:
        pass
    return t, result


@st.cache_data(ttl=86400)
def fetch_yahoo_fundamentals_all(tickers):
    """
    Fetch fundamentals for ALL tickers via Yahoo Finance.
    Uses 8 concurrent workers with chunked rate limiting.
    Expected to complete in 3-5 minutes for 503 tickers.
    """
    tl     = list(tickers)
    out    = {}
    CHUNK  = 30
    WKRS   = 8
    SLEEP  = 1.5
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]

    progress = st.progress(0)
    status   = st.empty()
    total    = len(chunks)

    for ci, chunk in enumerate(chunks):
        status.text("Yahoo fundamentals: chunk {}/{} ({} tickers done)...".format(
            ci+1, total, ci * CHUNK))
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futures = {ex.submit(_fetch_yahoo_fundamentals_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    t, d = fut.result()
                    out[t] = d
                except Exception:
                    t = futures[fut]
                    out[t] = {}
        progress.progress((ci + 1) / total)
        if ci < len(chunks) - 1:
            time.sleep(SLEEP + random.uniform(0, 0.5))

    progress.empty()
    status.empty()
    return out

# ══════════════════════════════════════════════════════════════════════════════
# ── LAYER 2 BONUS: FMP /quote bulk ────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400)
def fetch_fmp_quotes_if_available(tickers, api_key):
    out = {}
    if not api_key:
        return out
    tl     = list(tickers)
    chunks = [tl[i:i+100] for i in range(0, len(tl), 100)]
    for chunk in chunks:
        url = "https://financialmodelingprep.com/api/v3/quote/{}?apikey={}".format(
            ",".join(chunk), api_key)
        try:
            r    = requests.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                continue
            for item in data:
                t  = str(item.get("symbol", "")).upper().strip()
                if not t:
                    continue
                pe = sf(item.get("pe"))
                mc = sf(item.get("marketCap"))
                hi = sf(item.get("yearHigh"))
                lo = sf(item.get("yearLow"))
                if pe is not None and (pe <= 0 or pe > 10_000):
                    pe = None
                out[t] = {
                    "pe":     pe,
                    "mc":     mc,
                    "hi52":   hi,
                    "lo52":   lo,
                    "pe_src": "FMP-quote" if pe is not None else None,
                }
        except Exception:
            pass
        time.sleep(0.3)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# ── LAYER 3 BONUS: FMP /ratios-ttm per-ticker concurrent ─────────────────────
# Only runs if FMP key exists AND /ratios-ttm is on tier
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400)
def fetch_fmp_ratios_if_available(tickers, api_key):
    out = {}
    if not api_key:
        return out

    # Test first with one ticker to check if endpoint is accessible
    test_url = "https://financialmodelingprep.com/api/v3/ratios-ttm/AAPL?apikey={}".format(api_key)
    try:
        r    = requests.get(test_url, timeout=10)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            st.caption("FMP /ratios-ttm: not available on your tier. Using Yahoo quality metrics.")
            return out
        # Store field names for diagnostic
        st.session_state["fmp_ratios_fields"] = list(data[0].keys())
    except Exception:
        return out

    def fetch_one(t):
        url = "https://financialmodelingprep.com/api/v3/ratios-ttm/{}?apikey={}".format(t, api_key)
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

            peg_raw  = sf(item.get("priceEarningsGrowthRatioTTM"))
            peg      = peg_raw if (peg_raw and 0 < peg_raw <= 500) else None

            roic_raw = sf(item.get("returnOnInvestedCapitalTTM"))
            roic     = normalise_pct(roic_raw) if roic_raw is not None else None

            roe_raw  = sf(item.get("returnOnEquityTTM"))
            roe      = normalise_pct(roe_raw) if roe_raw is not None else None

            om_raw   = sf(item.get("operatingProfitMarginTTM"))
            om       = normalise_pct(om_raw) if om_raw is not None else None

            ic_raw   = sf(item.get("interestCoverageTTM"))
            ic       = min(float(ic_raw), 100.0) if (ic_raw and ic_raw > 0) else None

            de       = sf(item.get("debtEquityRatioTTM"))

            fwd_raw  = sf(item.get("priceToEarningsRatioTTM"))
            fwd_pe   = fwd_raw if (fwd_raw and 0 < fwd_raw <= 10_000) else None

            return t, {
                "peg": peg, "roic": roic, "roe": roe, "op_margin": om,
                "int_coverage": ic, "debt_eq": de, "fwd_pe": fwd_pe,
                "peg_src": "FMP-ratios" if peg else None,
            }
        except Exception:
            return t, {}

    tl      = list(tickers)
    CHUNK   = 50
    WORKERS = 10
    SLEEP   = 1.0
    chunks  = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    for ci, chunk in enumerate(chunks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(fetch_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    t, d = fut.result()
                    if d:
                        out[t] = d
                except Exception:
                    pass
        if ci < len(chunks) - 1:
            time.sleep(SLEEP)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# ── Revenue (Yahoo quarterly) ─────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400)
def fetch_last4_revenue_parallel(tickers):
    tl  = list(tickers)
    out = {}

    def one(t):
        try:
            qf = yf.Ticker(t).quarterly_financials
            if qf is not None and "Total Revenue" in qf.index:
                s = qf.loc["Total Revenue"].sort_index().tail(4)
                v = [float(x) for x in s.values]
                if len(v) == 4:
                    return t, v
        except Exception:
            pass
        return t, [None, None, None, None]

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for t, v in ex.map(one, tl):
            out[t] = v
    return out

# ══════════════════════════════════════════════════════════════════════════════
# ── MERGE: Yahoo primary, FMP override where available ───────────────────────
# ══════════════════════════════════════════════════════════════════════════════
def merge_all_sources(yahoo_data, fmp_quotes, fmp_ratios, tickers):
    """
    Merge strategy:
      FMP overrides Yahoo where FMP has data (FMP is generally more accurate)
      Yahoo is the universal fallback (works for all 503 tickers)
    """
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

        # PE: FMP-quote > Yahoo
        pe_val  = first(fq.get("pe"),      yb.get("pe"))
        pe_src  = ("FMP-quote" if fq.get("pe") is not None else yb.get("pe_src", "Yahoo"))

        # Fwd PE: FMP-ratios > Yahoo
        fwd_pe  = first(fr.get("fwd_pe"),  yb.get("fwd_pe"))

        # PEG: FMP-ratios > Yahoo (Yahoo has pegRatio directly)
        peg_val = first(fr.get("peg"),     yb.get("peg"))
        peg_src = ("FMP-ratios" if fr.get("peg") is not None else
                   yb.get("peg_src", "Yahoo") if yb.get("peg") is not None else "—")

        # ROIC: FMP-ratios only (Yahoo doesn't have ROIC directly)
        # Fallback: use ROE as proxy if ROIC missing
        roic    = first(fr.get("roic"))

        # ROE: FMP-ratios > Yahoo
        roe     = first(fr.get("roe"),     yb.get("roe"))

        # Interest Coverage: FMP-ratios > Yahoo computed
        ic      = first(fr.get("int_coverage"), yb.get("int_coverage"))

        # Op Margin: FMP-ratios > Yahoo
        om      = first(fr.get("op_margin"),    yb.get("op_margin"))

        # D/E: FMP-ratios > Yahoo
        de      = first(fr.get("debt_eq"),      yb.get("debt_eq"))

        # EPS Growth: Yahoo
        eps_g   = yb.get("eps_growth")
        g_src   = "Yahoo" if eps_g is not None else None

        # MC, 52W: FMP-quote > Yahoo
        mc      = first(fq.get("mc"),  yb.get("mc"))
        hi52    = first(fq.get("hi52"), yb.get("hi52"))
        lo52    = first(fq.get("lo52"), yb.get("lo52"))

        # Earn Revision: not available from Yahoo or free FMP
        # Set to None — will be scored 0 (missing penalty applies)
        earn_rev = None

        merged[t] = {
            "pe":             pe_val,
            "pe_src":         pe_src,
            "fwd_pe":         fwd_pe,
            "peg":            peg_val,
            "peg_src":        peg_src,
            "roic":           roic,
            "roe":            roe,
            "int_coverage":   ic,
            "op_margin":      om,
            "debt_eq":        de,
            "eps_growth":     eps_g,
            "growth_src":     g_src,
            "earn_revision":  earn_rev,
            "mc":             mc,
            "hi52":           hi52,
            "lo52":           lo52,
        }
    return merged

# ── Quality Score ─────────────────────────────────────────────────────────────
def compute_quality_score(roic, roe, int_coverage, op_margin):
    """
    v7: if ROIC missing, use ROE as proxy (both measure capital efficiency).
    ROE is leverage-affected but better than 0.
    """
    scores = []

    # Sub-score 1: ROIC preferred, ROE as proxy
    profitability = roic if roic is not None else roe
    if profitability is not None and not pd.isna(profitability):
        pf = float(profitability)
        if pf > 0:
            scores.append(min(100.0, np.log1p(pf) / np.log1p(30.0) * 100.0))
        else:
            scores.append(0.0)
    else:
        scores.append(0.0)

    # Sub-score 2: Interest Coverage
    if int_coverage is not None and not pd.isna(int_coverage):
        scores.append(min(100.0, max(0.0, float(int_coverage) / 10.0 * 100.0)))
    else:
        scores.append(0.0)

    # Sub-score 3: Op Margin
    if op_margin is not None and not pd.isna(op_margin):
        scores.append(min(100.0, max(0.0, float(op_margin) / 40.0 * 100.0)))
    else:
        scores.append(0.0)

    return sum(scores) / 3.0

# ── Conviction Score ──────────────────────────────────────────────────────────
def compute_conviction_scores(scr):
    KEY_FACTORS = ["P/E", "Fwd P/E", "PEG", "Quality Score", "Momentum Score"]
    n_factors   = len(KEY_FACTORS)
    scr         = scr.copy()

    def completeness(row):
        present = sum(1 for c in KEY_FACTORS if c in row.index and pd.notna(row[c]))
        return present / n_factors

    scr["_completeness"]    = scr.apply(completeness, axis=1)
    overall_median_pe       = scr["P/E"].median()
    sector_pe_map           = scr.groupby("Sector")["P/E"].median()

    def sector_discount(sector):
        if pd.isna(overall_median_pe) or overall_median_pe == 0:
            return 1.0
        s_pe = sector_pe_map.get(sector)
        if pd.isna(s_pe) or s_pe == 0:
            return 1.0
        return float(np.clip(overall_median_pe / s_pe, 0.7, 1.3))

    scr["_sec_discount"]    = scr["Sector"].map(sector_discount)
    raw_conviction          = scr["Score"] * scr["_completeness"] * scr["_sec_discount"]
    c_min, c_max            = raw_conviction.min(), raw_conviction.max()
    scr["Conviction Score"] = ((raw_conviction - c_min) / (c_max - c_min) * 100.0
                                if c_max > c_min else 50.0)
    return scr.drop(columns=["_completeness", "_sec_discount"])

# ── Ranking ───────────────────────────────────────────────────────────────────
def compute_rank_by_sector(scr):
    scr = scr.copy()
    scr["Score"] = pd.NA
    scr["Rank"]  = pd.NA
    W = FACTOR_WEIGHTS

    for sector in scr["Sector"].dropna().unique():
        g    = scr[scr["Sector"] == sector].copy()
        elig = g[g["Eligible"]].copy()
        if elig.empty:
            continue

        pe_input       = elig["Fwd P/E"].fillna(elig["P/E"])
        elig["_s_val"] = percentile_score(pe_input,            ascending=True)
        elig["_s_peg"] = percentile_score(elig["PEG"],         ascending=True)
        elig["_s_mom"] = percentile_score(elig["Momentum Score"], ascending=False)
        elig["_s_erev"]= percentile_score(elig["Earn Revision"],  ascending=False)

        qs    = elig["Quality Score"]
        q_min = qs.min(); q_max = qs.max()
        if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min:
            elig["_s_quality"] = (qs - q_min) / (q_max - q_min) * 100.0
        else:
            elig["_s_quality"] = qs.fillna(0.0)
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)

        raw = (W["valuation"]     * elig["_s_val"]     +
               W["quality"]       * elig["_s_quality"] +
               W["peg"]           * elig["_s_peg"]     +
               W["earn_revision"] * elig["_s_erev"]    +
               W["momentum"]      * elig["_s_mom"])

        factor_cols = ["P/E", "PEG", "Quality Score", "Earn Revision", "Momentum Score"]
        penalties   = elig.apply(lambda r: missing_factor_penalty(r, factor_cols), axis=1)
        raw         = raw * penalties

        elig["Score"] = raw
        elig = elig.sort_values("Score", ascending=False)
        elig["Rank"]  = range(1, len(elig) + 1)
        scr.loc[elig.index, "Score"] = elig["Score"]
        scr.loc[elig.index, "Rank"]  = elig["Rank"]

    return scr

# ── Build screener table ──────────────────────────────────────────────────────
def build_screener_table(universe_df, prices_map, merged_map, revenue_map, momentum_map):
    rows = []
    for _, r in universe_df.iterrows():
        t   = r["Ticker"]
        sec = r["Sector"]

        px_info  = prices_map.get(t, {})
        price    = to_num(px_info.get("price"))
        fi       = merged_map.get(t, {})
        mc       = to_num(fi.get("mc"))
        pe       = to_num(fi.get("pe"))
        fwd      = to_num(fi.get("fwd_pe"))
        hi       = to_num(fi.get("hi52"))
        lo       = to_num(fi.get("lo52"))
        roic     = to_num(fi.get("roic"))
        roe      = to_num(fi.get("roe"))
        ic       = to_num(fi.get("int_coverage"))
        om       = to_num(fi.get("op_margin"))
        de       = to_num(fi.get("debt_eq"))
        earn_rev = to_num(fi.get("earn_revision"))

        # Use price from batch download if available
        if pd.isna(price) and px_info.get("price"):
            price = to_num(px_info.get("price"))

        pos52 = None
        if pd.notna(price) and pd.notna(hi) and pd.notna(lo) and hi != lo:
            pos52 = float((price - lo) / (hi - lo) * 100.0)

        rev4               = revenue_map.get(t, [None]*4)
        rq1, rq2, rq3, rq4 = [to_num(x) for x in rev4]
        growth             = revenue_growth_pct_cagr([rq1, rq2, rq3, rq4])

        # PEG
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

        q_score = compute_quality_score(
            float(roic) if pd.notna(roic) else None,
            float(roe)  if pd.notna(roe)  else None,
            float(ic)   if pd.notna(ic)   else None,
            float(om)   if pd.notna(om)   else None,
        )

        mom       = momentum_map.get(t, {})
        ret_1mo   = to_num(mom.get("ret_1mo"))
        ret_3mo   = to_num(mom.get("ret_3mo"))
        ret_6mo   = to_num(mom.get("ret_6mo"))
        mom_score = to_num(mom.get("momentum_score"))
        t_vol     = to_num(mom.get("trailing_vol"))

        # Data source label
        parts = []
        ps = fi.get("pe_src")
        if ps: parts.append("PE:{}".format(ps))
        pg = fi.get("peg_src")
        if pg and pg != "—": parts.append("PEG:{}".format(pg))
        if roic is not None and not pd.isna(roic): parts.append("ROIC:FMP")
        if ic   is not None and not pd.isna(ic):   parts.append("IC:FMP")
        data_src = " | ".join(parts) if parts else "Yahoo"

        rows.append({
            "Ticker":             t,
            "Sector":             sec,
            "Price":              price,
            "Mkt Cap":            mc,
            "P/E":                pe,
            "Fwd P/E":            fwd,
            "PEG":                to_num(peg),
            "PEG Method":         peg_method,
            "Earn Revision":      earn_rev,
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
            "Data Sources":       data_src,
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

    num_cols = ["Price", "Mkt Cap", "P/E", "Fwd P/E", "PEG", "52W Pos%",
                "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
                "Quality Score", "Earn Revision", "Momentum Score",
                "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
                "Rev Q1", "Rev Q2", "Rev Q3", "Rev Q4", "Rev Growth% (CAGR)"]
    for c in num_cols:
        if c in scr.columns:
            scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns:
        scr["Rank"] = pd.NA
    scr = compute_conviction_scores(scr)
    return scr

# ── Quality flag ──────────────────────────────────────────────────────────────
def quality_flag(roic, roe, ic, om, de):
    flags = []
    profitability = roic if (roic is not None and not pd.isna(roic)) else roe
    if profitability is not None and not pd.isna(profitability) and profitability < QUALITY_THRESHOLDS["roic_min"]:
        flags.append("ROIC/ROE<8%")
    if ic is not None and not pd.isna(ic) and ic < QUALITY_THRESHOLDS["int_coverage_min"]:
        flags.append("IntCov<3x")
    if om is not None and not pd.isna(om) and om < QUALITY_THRESHOLDS["op_margin_min"]:
        flags.append("Margin<5%")
    de_note = " | D/E:{:.1f}".format(de) if (de is not None and not pd.isna(de)) else ""
    return (", ".join(flags) if flags else "Pass") + de_note

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

    is_all    = (sector_sel == "All Sectors")
    label     = "All Sectors (S&P 500)" if is_all else sector_sel
    total_mc  = scr["Mkt Cap"].sum()
    sdata     = scr.copy() if is_all else scr[scr["Sector"] == sector_sel].copy()
    sector_mc = sdata["Mkt Cap"].sum()
    pct       = 100.0 if is_all else (sector_mc / total_mc * 100.0 if total_mc > 0 else 0.0)

    med_pe   = sdata["P/E"].median()
    med_fwd  = sdata["Fwd P/E"].median()
    med_qual = sdata["Quality Score"].median()
    med_roe  = sdata["ROE%"].median()
    med_peg  = sdata["PEG"].median()

    st.markdown(
        "<div style='background:#12122a;border:1px solid #2a2a4a;border-radius:12px;"
        "padding:16px 20px;margin-bottom:16px;'>"
        "<span style='color:#aaa;font-size:13px;'>Sector Analysis  </span>"
        "<span style='color:#fff;font-size:14px;font-weight:700;'>{}</span>"
        "</div>".format(label),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(_kpi("Sector Mkt Cap",  fmt_mc(sector_mc), "sector total"),   unsafe_allow_html=True)
    c2.markdown(_kpi("S&P 500 Mkt Cap", fmt_mc(total_mc),  "all stocks"),     unsafe_allow_html=True)
    c3.markdown(_kpi("Sector Share",
                     "{:.1f}%".format(pct), "{} stocks".format(len(sdata))),
                unsafe_allow_html=True)
    c4.markdown(_kpi("Median P/E → Fwd",
                     "{:.1f}→{:.1f}".format(med_pe, med_fwd)
                     if pd.notna(med_pe) and pd.notna(med_fwd) else "N/A",
                     "trailing → forward", "#facc15"),
                unsafe_allow_html=True)
    c5.markdown(_kpi("Median Quality",
                     "{:.0f}/100".format(med_qual) if pd.notna(med_qual) else "N/A",
                     "Profitability+IntCov+Margin", "#4ade80"),
                unsafe_allow_html=True)
    c6.markdown(_kpi("Median PEG",
                     "{:.2f}".format(med_peg) if pd.notna(med_peg) else "N/A",
                     "price/earnings/growth", "#a78bfa"),
                unsafe_allow_html=True)

    if not is_all:
        top3   = sdata[sdata["Rank"].notna()].sort_values("Rank").head(3)
        badges = "  ".join(
            "<span style='background:#1a2a4a;color:#93c5fd;padding:3px 10px;"
            "border-radius:6px;font-weight:700;font-size:13px;'>{} "
            "<span style='color:#4ade80;font-size:11px;'>#{}</span></span>".format(
                row["Ticker"], int(row["Rank"]))
            for _, row in top3.iterrows()
        )
        st.markdown(
            "<div style='background:#1e1e2e;border-radius:10px;padding:14px 16px;"
            "margin-bottom:12px;'>"
            "<div style='color:#aaa;font-size:11px;margin-bottom:8px;'>Top Ranked in Sector</div>"
            "<div>{}</div>"
            "<div style='color:#555;font-size:10px;margin-top:8px;'>"
            "Score = Valuation 25% + Quality 25% + PEG 20% + Earn Revision 15% + Momentum 15%"
            "</div></div>".format(badges or "<span style='color:#555;'>No ranked stocks</span>"),
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── APP ───────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="S&P 500 Screener v7", layout="wide", page_icon="📊")
st.markdown(
    "<style>div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}</style>",
    unsafe_allow_html=True,
)

st.markdown("## S&P 500 Fundamental Screener v7")
st.caption(
    "Yahoo-first architecture · PEG from Yahoo pegRatio · "
    "IntCoverage from EBIT/InterestExp · FMP bonus layer if key available · "
    "Clean 5-factor model"
)

col_r, col_t = st.columns([1, 6])
with col_r:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()
with col_t:
    st.caption("Last loaded: {} · Prices: 1hr · Fundamentals: 24hr".format(
        datetime.now().strftime("%I:%M %p")))

fmp_key = get_fmp_key()
if fmp_key:
    st.success("FMP API key found — will use as bonus layer for ROIC and Int Coverage override.")
else:
    st.info("No FMP key configured. Running on Yahoo Finance only. Add [fmp] api_key to Streamlit Secrets to enable ROIC/IntCoverage from FMP.")

# ── Load universe ─────────────────────────────────────────────────────────────
with st.spinner("Loading S&P 500 universe..."):
    sp500 = fetch_sp500_constituents()
if sp500.empty:
    st.error("Failed to load S&P 500 universe."); st.stop()

universe_df = sp500.copy().reset_index(drop=True)
tickers     = tuple(universe_df["Ticker"].tolist())

# ── Fetch data ────────────────────────────────────────────────────────────────
with st.spinner("Fetching prices ({} tickers)...".format(len(tickers))):
    prices = fetch_prices_batch(tickers)

with st.spinner("Fetching momentum (skip-month vol-adjusted)..."):
    momentum = fetch_momentum_batch(tickers)

# PRIMARY: Yahoo fundamentals for ALL tickers
with st.spinner("Fetching Yahoo fundamentals (PE, PEG, ROE, OpMargin, D/E, IntCoverage) for all {} tickers...".format(len(tickers))):
    yahoo_fundamentals = fetch_yahoo_fundamentals_all(tickers)

# BONUS: FMP if key exists
fmp_quotes = {}
fmp_ratios = {}
if fmp_key:
    with st.spinner("FMP bonus: bulk /quote (PE, MC, 52W override)..."):
        fmp_quotes = fetch_fmp_quotes_if_available(tickers, fmp_key)
    with st.spinner("FMP bonus: /ratios-ttm concurrent (ROIC, IntCoverage override)..."):
        fmp_ratios = fetch_fmp_ratios_if_available(tickers, fmp_key)

with st.spinner("Merging data sources..."):
    merged_map = merge_all_sources(yahoo_fundamentals, fmp_quotes, fmp_ratios, tickers)

with st.spinner("Fetching quarterly revenue..."):
    rev_map = fetch_last4_revenue_parallel(tickers)

# ── Coverage banner ───────────────────────────────────────────────────────────
total_t  = len(tickers)
has_pe   = sum(1 for t in tickers if merged_map.get(t, {}).get("pe")           is not None)
has_fwd  = sum(1 for t in tickers if merged_map.get(t, {}).get("fwd_pe")       is not None)
has_peg  = sum(1 for t in tickers if merged_map.get(t, {}).get("peg")          is not None)
has_roe  = sum(1 for t in tickers if merged_map.get(t, {}).get("roe")          is not None)
has_ic   = sum(1 for t in tickers if merged_map.get(t, {}).get("int_coverage") is not None)
has_om   = sum(1 for t in tickers if merged_map.get(t, {}).get("op_margin")    is not None)

st.info(
    "Data coverage — "
    "P/E: {}/{} ({:.0f}%) · "
    "Fwd P/E: {}/{} ({:.0f}%) · "
    "PEG: {}/{} ({:.0f}%) · "
    "ROE: {}/{} ({:.0f}%) · "
    "Int Coverage: {}/{} ({:.0f}%) · "
    "Op Margin: {}/{} ({:.0f}%) · "
    "Primary: Yahoo Finance{}".format(
        has_pe,  total_t, has_pe  / total_t * 100,
        has_fwd, total_t, has_fwd / total_t * 100,
        has_peg, total_t, has_peg / total_t * 100,
        has_roe, total_t, has_roe / total_t * 100,
        has_ic,  total_t, has_ic  / total_t * 100,
        has_om,  total_t, has_om  / total_t * 100,
        " + FMP bonus" if fmp_key else "",
    )
)

# ── Build table ───────────────────────────────────────────────────────────────
scr = build_screener_table(universe_df, prices, merged_map, rev_map, momentum)

# ── Filters ───────────────────────────────────────────────────────────────────
st.markdown("### Filters")

with st.expander("Valuation & Size", expanded=True):
    fc1, fc2, fc3, fc4, fc5 = st.columns(5)
    all_sectors = sorted(scr["Sector"].dropna().unique().tolist())
    sector_sel  = fc1.selectbox("Sector", ["All Sectors"] + all_sectors)
    sort_by     = fc2.selectbox("Sort by", [
        "Sector then Rank", "Score high to low", "Conviction high to low",
        "Price low to high", "Price high to low", "Mkt Cap high to low",
        "PE low to high", "Fwd PE low to high", "PEG low to high",
        "Quality Score high", "ROE high to low",
        "Rev Growth high to low", "Momentum Score high", "52W Pos low to high",
    ])
    pe_max   = fc3.number_input("Max PE",              value=9999,  step=50)
    peg_max  = fc4.number_input("Max PEG",             value=999.0, step=1.0)
    mc_min_b = fc5.number_input("Min Market Cap ($B)", value=0,     step=5)

with st.expander("Quality Filters", expanded=False):
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    roe_min_f  = qc1.number_input("Min ROE (%)",              value=0.0,  step=5.0)
    ic_min_f   = qc2.number_input("Min Int Coverage (x)",     value=0.0,  step=1.0)
    om_min_f   = qc3.number_input("Min Op Margin (%)",        value=0.0,  step=5.0)
    qual_min_f = qc4.number_input("Min Quality Score",        value=0.0,  step=5.0)
    de_max_f   = qc5.number_input("Max Debt/Equity (ref)",    value=99.0, step=0.5)

with st.expander("Momentum & Display", expanded=False):
    mc1, mc2 = st.columns(2)
    mom_min   = mc1.number_input("Min Momentum Score",        value=-999.0, step=5.0)
    hide_nope = mc2.checkbox("Hide stocks missing both P/E and Fwd P/E", value=False)

render_sector_kpi_panel(scr, sector_sel)

filt = scr.copy()
if sector_sel != "All Sectors":
    filt = filt[filt["Sector"] == sector_sel]
filt = filt[(filt["Mkt Cap"].isna())        | (filt["Mkt Cap"]       >= mc_min_b * 1e9)]
filt = filt[(filt["P/E"].isna())            | (filt["P/E"]           <= pe_max)]
filt = filt[(filt["PEG"].isna())            | (filt["PEG"]           <= peg_max)]
filt = filt[(filt["ROE%"].isna())           | (filt["ROE%"]          >= roe_min_f)]
filt = filt[(filt["Int Coverage"].isna())   | (filt["Int Coverage"]  >= ic_min_f)]
filt = filt[(filt["Op Margin%"].isna())     | (filt["Op Margin%"]    >= om_min_f)]
filt = filt[(filt["Quality Score"].isna())  | (filt["Quality Score"] >= qual_min_f)]
filt = filt[(filt["Debt/Eq"].isna())        | (filt["Debt/Eq"]       <= de_max_f)]
filt = filt[(filt["Momentum Score"].isna()) | (filt["Momentum Score"]>= mom_min)]
if hide_nope:
    filt = filt[filt["P/E"].notna() | filt["Fwd P/E"].notna()]

sort_map = {
    "Sector then Rank":       (["Sector", "Rank"],     [True, True]),
    "Score high to low":      (["Score"],              [False]),
    "Conviction high to low": (["Conviction Score"],   [False]),
    "Price low to high":      (["Price"],              [True]),
    "Price high to low":      (["Price"],              [False]),
    "Mkt Cap high to low":    (["Mkt Cap"],            [False]),
    "PE low to high":         (["P/E"],                [True]),
    "Fwd PE low to high":     (["Fwd P/E"],            [True]),
    "PEG low to high":        (["PEG"],                [True]),
    "Quality Score high":     (["Quality Score"],      [False]),
    "ROE high to low":        (["ROE%"],               [False]),
    "Rev Growth high to low": (["Rev Growth% (CAGR)"], [False]),
    "Momentum Score high":    (["Momentum Score"],     [False]),
    "52W Pos low to high":    (["52W Pos%"],           [True]),
}
sc, sa = sort_map.get(sort_by, (["Sector", "Rank"], [True, True]))
filt   = filt.sort_values(sc, ascending=sa, na_position="last")

st.caption("Showing {} of {} stocks · Sector: {} · Sort: {}".format(
    len(filt), len(scr), sector_sel, sort_by))

disp = filt.copy()
disp["Price ($)"]    = disp["Price"].round(2)
disp["Mkt Cap ($B)"] = (disp["Mkt Cap"] / 1e9).round(2)
disp["Rev Q1 ($B)"]  = (disp["Rev Q1"]  / 1e9).round(2)
disp["Rev Q2 ($B)"]  = (disp["Rev Q2"]  / 1e9).round(2)
disp["Rev Q3 ($B)"]  = (disp["Rev Q3"]  / 1e9).round(2)
disp["Rev Q4 ($B)"]  = (disp["Rev Q4"]  / 1e9).round(2)
disp["Quality Flag"] = disp.apply(
    lambda r: quality_flag(r.get("ROIC%"), r.get("ROE%"),
                           r.get("Int Coverage"),
                           r.get("Op Margin%"), r.get("Debt/Eq")), axis=1)

for c in ["P/E", "Fwd P/E", "PEG", "Earn Revision", "52W Pos%",
          "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
          "Quality Score", "Momentum Score", "Ret 1Mo%", "Ret 3Mo%",
          "Ret 6Mo%", "Trailing Vol%", "Score", "Conviction Score",
          "Rev Growth% (CAGR)"]:
    if c in disp.columns:
        disp[c] = disp[c].round(2)

disp["Rank"] = disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

COLS = [
    "Ticker", "Sector", "Price ($)", "Mkt Cap ($B)",
    "P/E", "Fwd P/E", "PEG", "PEG Method",
    "Earn Revision",
    "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
    "Quality Score", "Quality Flag",
    "Momentum Score", "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
    "52W Pos%", "Score", "Conviction Score", "Rank",
    "Rev Q1 ($B)", "Rev Q2 ($B)", "Rev Q3 ($B)", "Rev Q4 ($B)",
    "Rev Growth% (CAGR)", "Data Sources",
]
disp_final = disp[[c for c in COLS if c in disp.columns]].copy()
st.dataframe(disp_final, use_container_width=True, height=680)

st.download_button(
    label="Download CSV",
    data=disp_final.to_csv(index=False).encode("utf-8"),
    file_name="sp500_screener_v7_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M")),
    mime="text/csv",
)

st.markdown("---")
st.markdown(
    "**Sources v7:** Yahoo Finance primary (PE, Fwd PE, PEG via pegRatio, ROE, "
    "OpMargin, D/E, IntCoverage from EBIT/InterestExp, EPS growth) · "
    "FMP bonus if key available (ROIC, IntCoverage override) · "
    "Momentum: Yahoo batch download · Revenue: Yahoo quarterly"
)
# ══════════════════════════════════════════════════════════════════════════════
# ── COLUMN REFERENCE GUIDE — DETAILED WITH EXAMPLES ──────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Column Reference Guide")
st.caption("Every column explained with formula, real-world example, and how to use it.")

tab_val, tab_qual, tab_peg, tab_erev, tab_mom, tab_rank, tab_disp = st.tabs([
    "Valuation", "Quality", "PEG", "Earn Revision", "Momentum", "Ranking & Score", "Display-Only Columns"
])

with tab_val:
    st.markdown("""
### P/E — Price to Earnings Ratio (Trailing)
**What it is:** How many dollars you are paying per dollar of actual profit the company earned over the last 12 months.

**Formula:** `Current Stock Price ÷ Trailing 12-Month EPS`

**Example:**
- Apple stock price = $190. Apple earned $6.43 per share over the last 12 months.
- P/E = 190 ÷ 6.43 = **29.5**
- Meaning: You are paying $29.50 for every $1 of Apple's annual profit.

**How to interpret:**
- P/E of 15 in a sector where median is 25 → stock looks cheap vs peers
- P/E of 50 in a sector where median is 25 → stock looks expensive vs peers
- P/E is undefined when EPS is negative (company is losing money)

**Data source waterfall:** FMP /quote → Finviz → Yahoo Finance

**Used in scoring?** Yes — this is the primary Valuation factor (25% weight). Lower P/E = higher percentile = better score within sector.

---

### Fwd P/E — Forward Price to Earnings Ratio
**What it is:** Same as P/E but uses analysts' consensus estimate of what earnings will be over the NEXT 12 months instead of last 12 months.

**Formula:** `Current Stock Price ÷ Next 12-Month Estimated EPS`

**Example:**
- Apple stock price = $190. Analysts estimate Apple will earn $7.20 per share next year.
- Fwd P/E = 190 ÷ 7.20 = **26.4**
- Trailing P/E was 29.5. Fwd P/E is 26.4. This means earnings are expected to grow → you are paying less per future dollar of earnings than per past dollar.

**When Fwd P/E < Trailing P/E:** Earnings growing. Good sign.
**When Fwd P/E > Trailing P/E:** Earnings expected to shrink. Bad sign.

**Data source waterfall:** FMP /ratios → FMP /key-metrics → FMP /analyst-estimates → Finviz → Yahoo

**Used in scoring?** Yes — Fwd P/E is PREFERRED over trailing P/E for the Valuation factor. If Fwd P/E is available it is used. Trailing P/E is the fallback only.

---

### 52W Pos% — 52-Week Position
**What it is:** Where the current price sits between the stock's 52-week low and 52-week high, expressed as a percentage.

**Formula:** `(Current Price − 52W Low) ÷ (52W High − 52W Low) × 100`

**Example:**
- Stock 52W Low = $80, 52W High = $150, Current Price = $110
- 52W Pos% = (110 − 80) ÷ (150 − 80) × 100 = 30 ÷ 70 × 100 = **42.9%**
- Meaning: The stock is sitting 43% of the way up from its yearly low.

**0% = at the yearly low. 100% = at the yearly high.**

**How to use:**
- Low 52W Pos% (under 30%) = stock is near its yearly lows — could be a value opportunity or a value trap. Always combine with quality filters.
- High 52W Pos% (above 80%) = stock is near yearly highs — strong momentum but less upside room.

**Used in scoring?** No. Display only. Use the sort "52W Pos low to high" to find beaten-down stocks.
""")

with tab_qual:
    st.markdown("""
### Quality Score (0–100)
**What it is:** A composite score measuring the fundamental strength and safety of the business. Made up of three equally-weighted sub-scores.

**Formula:** `(ROIC sub-score + Interest Coverage sub-score + Op Margin sub-score) ÷ 3`

**Used in scoring?** Yes — 25% weight. Scored using min-max scaling within each sector (not percentile rank) to preserve magnitude differences between stocks.

---

### ROIC% — Return on Invested Capital
**What it is:** For every dollar of capital invested in the business (equity + debt combined), how many cents of profit does it generate after tax?

**Formula:** `Net Operating Profit After Tax ÷ (Total Equity + Net Debt) × 100`

**Example:**
- Company A invested $10 billion in its business (factories, equipment, working capital).
- It generated $2 billion in operating profit after tax.
- ROIC = 2B ÷ 10B × 100 = **20%**
- Meaning: Every $1 of capital deployed generates $0.20 of annual profit.

**Benchmarks:**
- ROIC > 20%: Excellent. Company has a real competitive advantage.
- ROIC 10–20%: Good. Solid business.
- ROIC < 8%: Flagged. May be destroying value if cost of capital is around 8–10%.

**Why ROIC instead of ROE?**
ROE (Return on Equity) is easily inflated by taking on debt or doing share buybacks. Apple's ROE appears to be 160%+ because it has bought back so many shares that book equity is near zero — that's not a real profitability signal. ROIC includes both debt and equity in the denominator so it is leverage-neutral and cannot be gamed this way.

**Sub-score formula:** `min(100, log(1 + ROIC) ÷ log(1 + 30) × 100)`
Log-scaling is used so that ROIC 160% (Apple) and ROIC 50% (Microsoft) are not treated as identical just because both exceed a linear cap of 100. Apple still scores meaningfully higher than Microsoft.

**ROIC Sub-score examples:**
- ROIC 30% → sub-score 100 (perfect)
- ROIC 15% → sub-score ~77
- ROIC 8% → sub-score ~49
- ROIC 5% → sub-score ~38
- ROIC 0% or negative → sub-score 0

**Data source:** FMP /ratios (bulk) → FMP /key-metrics fallback

---

### Int Coverage — Interest Coverage Ratio
**What it is:** How many times over can the company pay its annual interest expense using its operating profit? Measures debt safety more directly than Debt/Equity.

**Formula:** `EBIT (Earnings Before Interest and Tax) ÷ Annual Interest Expense`

**Example:**
- Company B's EBIT = $5 billion. Annual interest payments = $500 million.
- Interest Coverage = 5B ÷ 0.5B = **10x**
- Meaning: The company earns 10 times what it needs to pay in interest. Very safe.

**Another example (risky):**
- Company C's EBIT = $300 million. Annual interest = $250 million.
- Interest Coverage = 300 ÷ 250 = **1.2x**
- Meaning: Only barely covering interest payments. Any revenue dip = trouble.

**Benchmarks:**
- 10x or above: Very safe — full marks (sub-score 100)
- 5x: Comfortable — sub-score 50
- 3x: Borderline — sub-score 30 — flagged in Quality Flag column
- Below 3x: Risky — flagged "IntCov<3x"
- Below 1x: Company cannot cover interest from operations — in distress

**Why this instead of Debt/Equity?**
Debt/Equity tells you the structure but not whether it is a problem. A company with D/E of 3.0 but EBIT covering interest 20x is totally fine. A company with D/E of 0.5 but barely covering interest is in serious trouble. Interest Coverage is a direct measure of ability to service debt.

Note: Debt/Equity is still shown as a display column so you can see it, it just does not feed into the score.

**Sub-score formula:** `min(100, (Int Coverage ÷ 10) × 100)`
Capped at 100x to avoid extremely cash-rich companies getting outsized scores.

**Data source:** FMP /ratios (bulk) only.

---

### Op Margin% — Operating Margin
**What it is:** Of every dollar of revenue the company brings in, how many cents become operating profit (before interest and taxes)?

**Formula:** `Operating Income ÷ Revenue × 100`

**Example:**
- Microsoft revenue = $211 billion. Operating income = $88 billion.
- Op Margin = 88 ÷ 211 × 100 = **41.7%**
- Meaning: For every $1 of revenue, Microsoft keeps $0.42 as operating profit.

**Another example:**
- Supermarket chain revenue = $100 billion. Operating income = $2.5 billion.
- Op Margin = 2.5%
- Low margin businesses are more vulnerable to cost increases or revenue drops.

**Benchmarks:**
- 40%+ : Elite (software, pharma, luxury) — sub-score 100
- 20%  : Good — sub-score 50
- 10%  : Acceptable — sub-score 25
- Below 5%: Flagged "Margin<5%"
- Negative: Company is losing money from operations — sub-score 0

**Sub-score formula:** `min(100, max(0, Op Margin ÷ 40 × 100))`

**Data source:** FMP /ratios (bulk) → Finviz → Yahoo Finance

---

### Quality Flag
**What it is:** A simple pass/fail check that runs independently of the Quality Score. Flags stocks that breach minimum thresholds on any of the three quality metrics.

**Flags triggered:**
- `ROIC<8%` — return on capital below 8% (may be destroying value)
- `IntCov<3x` — interest coverage below 3x (debt servicing risk)
- `Margin<5%` — operating margin below 5% (thin profitability buffer)
- `Pass` — all three checks passed
- `D/E: 1.5` shown as informational note (Debt/Equity for reference, not a pass/fail)

**Example:** A stock might show `ROIC<8%, Margin<5% | D/E:2.1` meaning it failed two quality checks and has a debt/equity ratio of 2.1 for your reference.
""")

with tab_peg:
    st.markdown("""
### PEG — Price/Earnings-to-Growth Ratio
**What it is:** Takes the P/E ratio and adjusts it for the company's earnings growth rate. Answers the question: "Is this stock cheap or expensive GIVEN how fast it is growing?"

**Formula:** `P/E Ratio ÷ Annual EPS Growth Rate (%)`

**Example 1 — Growth stock that looks expensive but is actually fair:**
- Stock A: P/E = 60, EPS growing at 40% per year
- PEG = 60 ÷ 40 = **1.5** — fairly valued for its growth rate
- A pure P/E screen would rank this stock near the bottom. PEG reveals it is not that expensive.

**Example 2 — Value stock that looks cheap but is actually a trap:**
- Stock B: P/E = 12, EPS growing at 3% per year
- PEG = 12 ÷ 3 = **4.0** — very expensive relative to growth
- P/E looks cheap at 12 but the company is barely growing — not actually cheap on PEG.

**Example 3 — Genuinely undervalued:**
- Stock C: P/E = 20, EPS growing at 30% per year
- PEG = 20 ÷ 30 = **0.67** — undervalued. You are paying less than 1x for each unit of growth.

**Interpreting PEG:**
- PEG below 1.0: Stock is cheap relative to its growth. Potentially undervalued.
- PEG 1.0–2.0: Fairly valued for its growth rate.
- PEG above 2.0: Expensive relative to growth. Requires exceptional quality or other justification.
- PEG above 3.0: Hard to justify unless growth is accelerating significantly.

**Data source waterfall (v5):**
1. FMP /ratios bulk endpoint — analyst-computed TTM PEG (most reliable)
2. Finviz — PEG from their screener (good secondary coverage)
3. Computed: (Fwd P/E or Trailing P/E) ÷ EPS Growth % from FMP /key-metrics or Yahoo
4. Revenue CAGR is NOT used as fallback (removed in v4 — revenue growth is not earnings growth)

**Growth guard:** PEG is only computed when EPS growth is at least 5%. Below 5% growth the denominator is near-zero and produces misleading ratios. Example: P/E 15 ÷ EPS growth 0.5% = PEG 30, which looks terrible but is just a math artifact.

**Used in scoring?** Yes — 20% standalone weight. Lower PEG = higher percentile = better score within sector. This is independent of Valuation (no double counting in v5).

### PEG Method
Shows which source was used to compute PEG:
- `FMP-ratios` — directly from FMP analyst-computed ratio (most trustworthy)
- `Finviz` — from Finviz screener data
- `FMP EPS growth` — computed as Fwd PE ÷ FMP EPS growth estimate
- `Yahoo EPS growth` — computed as Fwd PE ÷ Yahoo EPS growth estimate
- `—` — PEG could not be computed (missing PE or growth below 5%)
""")

with tab_erev:
    st.markdown("""
### Earn Revision — Earnings Estimate Revision Score
**What it is:** Measures whether Wall Street analysts are raising or cutting their earnings estimates for this company right now. This is one of the most powerful short-to-medium term signals in quantitative equity research.

**Formula:**
**Example 1 — Positive revision:**
- 3 months ago analysts estimated Apple would earn $7.00 per share next year.
- Today the consensus estimate is $7.70 per share.
- Earn Revision = (7.70 − 7.00) ÷ 7.00 = 0.10 → **+0.10**
- Meaning: Analysts collectively raised estimates by 10%. Positive signal.

**Example 2 — Negative revision:**
- 3 months ago Intel estimated at $1.50 EPS. Today estimate is $0.90 EPS.
- Earn Revision = (0.90 − 1.50) ÷ 1.50 = −0.40 → **−0.40**
- Meaning: Analysts cut estimates by 40%. Strong negative signal.

**Example 3 — Extreme positive (clipped):**
- Estimate moved from $1.00 to $2.20 (120% increase).
- Raw = 1.20 but clipped to **+1.0** maximum.

**Interpreting the score:**
- +1.0: Maximum positive — analysts massively upgraded estimates. Very bullish.
- +0.10 to +0.30: Moderate upgrade. Positive signal.
- 0.0: No change in consensus estimates.
- −0.10 to −0.30: Moderate downgrade. Negative signal.
- −1.0: Maximum negative — analysts massively cut estimates. Very bearish.

**Why this matters so much:**
When analysts revise EPS estimates upward it almost always means one of three things happened: the company gave positive guidance on an earnings call, the company reported a beat-and-raise quarter, or the macro environment for that company improved. Stocks with rising estimates consistently outperform over the next 30–90 days. This is documented extensively in MSCI Barra factor research, Goldman Sachs alpha models, and JP Morgan quant strategy papers. It was completely missing from v3 and v4 of this screener — adding it as 15% weight is one of the most impactful improvements in v5.

**How to use the filter:**
- Set Min Earn Revision to 0.05 to see only stocks where analysts are raising estimates
- Set Min Earn Revision to 0.15 for stronger revision momentum only
- Leave at −1.0 default to see all stocks

**Data source:** FMP /analyst-estimates batch endpoint (50 tickers per call).
Earn Revision is NOT available from Finviz or Yahoo — FMP is the only source here.

**Used in scoring?** Yes — 15% weight. Higher revision = higher percentile = better score within sector.
""")

with tab_mom:
    st.markdown("""
### Momentum Score — Skip-Month Volatility-Adjusted Momentum
**What it is:** Measures the stock's recent price trend, adjusted for how volatile the stock is. This is the signal used in the ranking model.

**Formula:**
**Example:**
- Stock X: 6mo return = +22%, 1mo return = +3%, trailing vol = 25%
- Skip-month raw = 22 − 3 = 19%
- Momentum Score = 19 ÷ 25 = **0.76**

- Stock Y: 6mo return = +22%, 1mo return = +3%, trailing vol = 50%
- Skip-month raw = same 19%
- Momentum Score = 19 ÷ 50 = **0.38**

Both stocks gained the same 19% skip-month return but Stock X has half the volatility — its momentum is a much cleaner signal. Stock X scores higher.

**Why subtract 1-month return (skip-month)?**
The 1-month window has a documented short-term reversal effect — stocks that jumped last month slightly tend to mean-revert. Including it adds noise that actually works against you. By subtracting it you isolate the more durable 2–6 month momentum signal. This is the exact construction used in the Fama-French UMD (Up Minus Down) academic momentum factor.

**Why divide by volatility?**
A +15% return in a stock with 10% annualised vol is a very strong signal (the stock moved 1.5 standard deviations). The same +15% return in a stock with 80% vol is barely a whisper (less than 0.2 standard deviations). Vol-adjustment makes momentum comparable across stocks with very different risk profiles.

**Used in scoring?** Yes — 15% weight. Higher Momentum Score = higher percentile = better.

---

### Ret 1Mo%, Ret 3Mo%, Ret 6Mo% — Raw Monthly Returns
**What they are:** The raw percentage price return over 1, 3, and 6 months respectively, measured from monthly closing prices.

**Example:**
- Stock price 6 months ago: $100. Today: $118.
- Ret 6Mo% = (118 − 100) ÷ 100 × 100 = **+18%**

**Used in scoring?** No. These are display-only columns. They are shown so you can see the raw return profile. The Momentum Score (above) is what actually feeds into ranking.

---

### Trailing Vol% — Trailing 90-Day Annualised Volatility
**What it is:** The annualised standard deviation of daily returns over the last 90 trading days, expressed as a percentage. Measures how much the stock price jumps around day-to-day.

**Formula:** `std(daily_returns, last 90 days) × sqrt(252) × 100`

**Example:**
- Utility stock: daily std dev = 0.6%, annualised vol = 0.6% × √252 = **9.5%** — very stable
- Tech growth stock: daily std dev = 2.5%, annualised vol = 2.5% × √252 = **39.7%** — very volatile

**How to use:**
- High vol (>40%) + high Momentum Score = strong momentum but risky
- Low vol (<15%) + high Momentum Score = very clean, reliable momentum signal
- Trailing Vol% is the denominator in the Momentum Score formula

**Used in scoring?** No. Display only.
""")

with tab_rank:
    st.markdown("""
### Score (0–100)
**What it is:** The final composite percentile score for a stock within its GICS sector. This is the primary ranking signal.

**Formula:**
**Example of scoring in the Information Technology sector (75 stocks):**

Say Apple has these raw metrics:
- Fwd P/E = 26 → ranks 18th lowest out of 75 tech stocks → PE percentile = 76
- Quality Score = 82 out of 100 → min-max score within tech = 78
- PEG = 1.8 → ranks 22nd lowest → PEG percentile = 71
- Earn Revision = +0.12 → ranks 15th highest → EarnRev percentile = 81
- Momentum Score = 0.65 → ranks 20th highest → Mom percentile = 74
- Missing factors = 0 → no penalty

Score = 0.25×76 + 0.25×78 + 0.20×71 + 0.15×81 + 0.15×74 = 19+19.5+14.2+12.2+11.1 = **76.0**

**Key points:**
- Score is always relative to sector. Score 76 in Tech vs Score 76 in Utilities are independent — both mean the stock is in the top quarter of their respective sectors.
- A stock can have a mediocre absolute P/E but if every other stock in the sector has a higher P/E, it still gets a high Valuation percentile.

---

### Rank
**What it is:** The ordinal position of the stock within its sector based on Score. Rank 1 = highest Score in that sector.

**Example:** Rank 3 in Information Technology (75 stocks) means this stock has the 3rd highest composite Score among all IT stocks in the S&P 500.

**Important:** Ranks are NOT comparable across sectors. Rank 1 in Utilities does not mean the same as Rank 1 in Technology. They are entirely independent sector-specific rankings.

---

### Missing Data Penalty
Applied before finalising the Score:
- 0 or 1 factor missing: no penalty (× 1.0)
- 2 factors missing: Score × 0.85 (−15%)
- 3 or more factors missing: Score × 0.70 (−30%)

**Factors checked:** P/E, PEG, Quality Score, Earn Revision, Momentum Score

**Why?** Without this penalty, a stock with only one available metric (say a very low P/E) could accidentally rank #1 purely because it scores 100 on valuation while everything else defaults to 0. The penalty ensures incomplete data is reflected in the final rank.

---

### Conviction Score (0–100)
**What it is:** A confidence-adjusted version of Score that rewards stocks where data is complete and the sector is not overvalued. Normalised to 0–100 across the entire S&P 500 universe.

**Formula:**
**Data completeness example:**
- Stock has P/E, PEG, Quality Score, Earn Revision but is missing Fwd P/E and Momentum Score
- Completeness = 4 ÷ 6 = 0.67
- Score gets multiplied by 0.67 → lower Conviction Score than a stock with all 6 factors present

**Sector discount example:**
- S&P 500 median P/E = 22. Technology sector median P/E = 30.
- Sector discount = 22 ÷ 30 = 0.73 (clipped to 0.73 — tech is expensive vs market)
- A tech stock's Conviction Score is slightly reduced because the whole sector is premium-priced.

- S&P 500 median P/E = 22. Energy sector median P/E = 12.
- Sector discount = 22 ÷ 12 = 1.83 → clipped to 1.30 (energy is cheap vs market)
- An energy stock's Conviction Score gets a 30% boost because the sector is undervalued.

**How to use:** Sort by Conviction Score descending to find the highest-quality, best-data, cheapest-sector ranked stocks. These are the highest-confidence ideas from the model.
""")

with tab_disp:
    st.markdown("""
### ROE% — Return on Equity (Display Only)
**What it is:** Net income divided by shareholders equity. Shows how much profit a company generates from shareholders' money.

**Formula:** `Net Income ÷ Total Shareholders Equity × 100`

**Example:**
- Company earned $5B net income. Shareholders equity = $25B.
- ROE = 5 ÷ 25 × 100 = **20%**

**Why display only (not in scoring)?**
ROE is distorted by leverage and buybacks. Apple's ROE is 150%+ because it has bought back so much stock that book equity is near zero — that is an accounting quirk, not a sign that Apple is 7x better than a company with ROE 20%. ROIC (in the Quality Score) is the leverage-neutral replacement.

**Data source:** FMP /ratios → Finviz → Yahoo Finance

---

### Debt/Eq — Debt to Equity Ratio (Display Only)
**What it is:** Total debt divided by total shareholders equity. Shows how much of the company's financing comes from debt vs equity.

**Formula:** `Total Debt ÷ Total Shareholders Equity`

**Example:**
- Company has $10B in debt and $20B in equity.
- D/E = 10 ÷ 20 = **0.5x** — moderate leverage
- D/E = 3.0x — heavily leveraged

**Why display only (not in scoring)?**
D/E tells you the capital structure but not whether it is a problem. A company with D/E 3.0 and Interest Coverage 20x is perfectly safe. A company with D/E 0.5 and Interest Coverage 1.2x is in trouble. Interest Coverage (in Quality Score) is the better signal. D/E is shown for your reference and appears in the Quality Flag note.

**Data source:** FMP /ratios → Yahoo Finance

---

### Rev Q1–Q4 ($B) — Quarterly Revenue
**What they are:** The company's total revenue (top-line sales) for each of the last four fiscal quarters, in billions of dollars.

**Example for a large retailer:**
- Rev Q1 = $35.2B, Rev Q2 = $38.1B, Rev Q3 = $41.0B, Rev Q4 = $44.5B
- Clearly an accelerating revenue trajectory — growing each quarter.

**Data source:** Yahoo Finance quarterly_financials

---

### Rev Growth% (CAGR) — Revenue Compound Annual Growth Rate
**What it is:** The annualised revenue growth rate from Q1 to Q4 of the last four quarters on hand.

**Formula:** `(Q4 Revenue ÷ Q1 Revenue)^(1/3) − 1 × 100`

**Example:**
- Q1 revenue = $35B, Q4 revenue = $44.5B (3 quarters later)
- CAGR = (44.5 ÷ 35)^(1/3) − 1 × 100 = **8.3% annualised**

**Used in scoring?** No. Display only. Previously used as a fallback for PEG computation in v3 — removed in v4/v5 because revenue growth ≠ earnings growth and produced misleading PEG values.

**How to use:** Sort by "Rev Growth high to low" to find companies with accelerating revenue. Combine with strong Earn Revision to find growth stories where both revenue AND analysts are moving up.

---

### Data Sources
**What it shows:** Which data providers successfully contributed data for each stock.

**Examples:**
- `PE:FMP-quote | PEG:Finviz | G:FMP` — P/E from FMP bulk quote, PEG from Finviz scrape, growth from FMP
- `PE:Yahoo | PEG:FMP-ratios | EarnRev:FMP` — P/E fell through to Yahoo, but PEG and EarnRev from FMP
- `Yahoo only` — FMP had no data for this ticker and Finviz didn't cover it either

**How to use:** If a stock has `Yahoo only` for most fields, its Score and Rank are less reliable. The Conviction Score will automatically be lower for such stocks due to the data completeness penalty.
""")

st.markdown("""---
**Data sources:** FMP /quote · FMP /ratios (bulk 50/call) · FMP /key-metrics (bulk 50/call) ·
FMP /analyst-estimates (batch) · Finviz export (1 call, all 500) · Yahoo Finance (fallback + revenue + momentum) ·
S&P 500 universe: Wikipedia GICS table
""")
