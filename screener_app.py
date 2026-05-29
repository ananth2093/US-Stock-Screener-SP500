# screener_app.py  v3
# Coverage fixes:
#   1. PEG  → /ratios-ttm  (priceEarningsGrowthRatioTTM) — free tier accessible
#   2. Fwd P/E → /analyst-estimates fallback (price ÷ fwd EPS) for missing stocks
#   3. ROE  → unified decimal-to-percent conversion, consistent across FMP + Yahoo
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
    st.error("Install beautifulsoup4: add it to requirements.txt")
    st.stop()

# ── Constants ─────────────────────────────────────────────────────────────────
MIN_GROWTH_PCT_FOR_PEG = 5.0

FACTOR_WEIGHTS = {
    "valuation":  0.30,
    "peg":        0.20,
    "quality":    0.25,
    "earn_traj":  0.15,
    "momentum":   0.10,
}

QUALITY_THRESHOLDS = {
    "roe_min":       10.0,
    "debt_eq_max":    2.0,
    "op_margin_min":  5.0,
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


def sf(val):
    try:
        return float(val) if val is not None else None
    except Exception:
        return None


def normalise_pct(val, label=""):
    """
    FMP sometimes returns ratios as decimals (0.15) and sometimes as
    percentages (15.0). Heuristic: if |val| < 5 and it's a ratio field,
    multiply by 100. We log the label for debugging.
    """
    if val is None:
        return None
    v = float(val)
    if abs(v) < 5.0:          # almost certainly a decimal like 0.15
        return v * 100.0
    return v                   # already a percentage like 15.0


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


# ── S&P 500 universe ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_sp500_constituents():
    url  = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    r    = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    tbl  = soup.find("table", {"id": "constituents"})
    if tbl is None:
        raise RuntimeError("Wikipedia constituents table not found.")
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


# ── Prices ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices_batch(tickers):
    tl  = list(tickers)
    res = {t: None for t in tl}
    try:
        raw = yf.download(tl, period="2d", interval="1d",
                          group_by="ticker", auto_adjust=True,
                          progress=False, threads=True)
        for t in tl:
            try:
                res[t] = float(raw["Close"].iloc[-1]) if len(tl) == 1 \
                         else float(raw[t]["Close"].iloc[-1])
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
        raw = yf.download(tl, period="7mo", interval="1mo",
                          group_by="ticker", auto_adjust=True,
                          progress=False, threads=True)
        for t in tl:
            try:
                closes = raw["Close"].dropna() if len(tl) == 1 \
                         else raw[t]["Close"].dropna()
                if len(closes) < 2:
                    continue
                px_now = float(closes.iloc[-1])

                def ret(n):
                    idx = -(n + 1)
                    if abs(idx) > len(closes):
                        return None
                    px = float(closes.iloc[idx])
                    return (px_now / px - 1) * 100.0 if px > 0 else None

                out[t] = {"ret_1mo": ret(1), "ret_3mo": ret(3), "ret_6mo": ret(6)}
            except Exception:
                pass
    except Exception:
        pass
    return out


# ── FMP 1: bulk quote  (PE, MC, 52W, EPS) ────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_bulk_quotes(tickers, api_key):
    out = {t: {} for t in tickers}
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
                pe  = sf(item.get("pe"))
                mc  = sf(item.get("marketCap"))
                hi  = sf(item.get("yearHigh"))
                lo  = sf(item.get("yearLow"))
                px  = sf(item.get("price"))
                eps = sf(item.get("eps"))
                if pe is not None and (pe <= 0 or pe > 10_000):
                    pe = None
                out[t] = {
                    "pe": pe, "mc": mc, "hi52": hi,
                    "lo52": lo, "price": px, "eps": eps,
                    "pe_src": "FMP" if pe is not None else None,
                }
        except Exception:
            pass
        time.sleep(0.3)
    return out


# ── FMP 2: ratios-ttm  (PEG, Op Margin, Debt/Eq, ROE) ── FIX #1 & #3 ─────────
@st.cache_data(ttl=86400)
def fetch_fmp_ratios_bulk(tickers, api_key):
    """
    /ratios-ttm returns priceEarningsGrowthRatioTTM on FREE tier.
    This is Fix #1 (PEG 0% → ~70%).
    Also returns operatingProfitMarginTTM, debtEquityRatioTTM, returnOnEquityTTM.
    Fix #3: unified decimal normalisation for all ratio fields.
    """
    out = {t: {} for t in tickers}
    if not api_key:
        return out

    def one(t):
        url = "https://financialmodelingprep.com/api/v3/ratios-ttm/{}?apikey={}".format(
            t, api_key)
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            d = r.json()
            if not isinstance(d, list) or len(d) == 0:
                return t, {}
            item = d[0]

            peg_raw = sf(item.get("priceEarningsGrowthRatioTTM"))
            peg     = None
            if peg_raw is not None and 0 < peg_raw <= 500:
                peg = peg_raw   # PEG is already a ratio, no percent conversion

            # ROE: FMP returns as decimal 0.15 = 15% — Fix #3
            roe_raw = sf(item.get("returnOnEquityTTM"))
            roe     = normalise_pct(roe_raw, "ROE") if roe_raw is not None else None

            # Op Margin: FMP returns as decimal 0.25 = 25%
            om_raw = sf(item.get("operatingProfitMarginTTM"))
            om     = normalise_pct(om_raw, "OpMargin") if om_raw is not None else None

            # Debt/Equity: usually a plain ratio (1.5 = 1.5), not a percentage
            de = sf(item.get("debtEquityRatioTTM"))

            # Fwd PE from this endpoint as well
            fwd_pe_raw = sf(item.get("priceToEarningsRatioTTM"))
            fwd_pe     = fwd_pe_raw if (fwd_pe_raw and 0 < fwd_pe_raw <= 10_000) else None

            return t, {
                "peg":       peg,
                "roe":       roe,
                "op_margin": om,
                "debt_eq":   de,
                "fwd_pe":    fwd_pe,
                "peg_src":   "FMP-ratios" if peg is not None else None,
            }
        except Exception:
            return t, {}

    tl      = list(tickers)
    CHUNK   = 20
    SLEEP   = 1.5
    WORKERS = 5
    chunks  = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    for ci, chunk in enumerate(chunks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    t, d = fut.result()
                    out[t] = d
                except Exception:
                    pass
        if ci < len(chunks) - 1:
            time.sleep(SLEEP)
    return out


# ── FMP 3: key-metrics-ttm  (EPS growth, revenue growth, Fwd PE) ─────────────
@st.cache_data(ttl=86400)
def fetch_fmp_key_metrics_bulk(tickers, api_key):
    out = {t: {} for t in tickers}
    if not api_key:
        return out

    def one(t):
        url = "https://financialmodelingprep.com/api/v3/key-metrics-ttm/{}?apikey={}".format(
            t, api_key)
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            d = r.json()
            if not isinstance(d, list) or len(d) == 0:
                return t, {}
            item    = d[0]
            fwd_raw = sf(item.get("peRatioTTM"))
            fwd_pe  = fwd_raw if (fwd_raw and 0 < fwd_raw <= 10_000) else None
            rg_raw  = sf(item.get("revenueGrowthTTM"))
            rg      = normalise_pct(rg_raw) if rg_raw is not None else None
            eg_raw  = sf(item.get("epsgrowthTTM"))
            eg      = normalise_pct(eg_raw) if eg_raw is not None else None
            return t, {"fwd_pe": fwd_pe, "revenue_growth": rg, "eps_growth": eg}
        except Exception:
            return t, {}

    tl      = list(tickers)
    CHUNK   = 20
    SLEEP   = 1.5
    WORKERS = 5
    chunks  = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    for ci, chunk in enumerate(chunks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    t, d = fut.result()
                    out[t] = d
                except Exception:
                    pass
        if ci < len(chunks) - 1:
            time.sleep(SLEEP)
    return out


# ── FMP 4: analyst-estimates  (Fwd EPS → Fwd PE)  ── FIX #2 ──────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_analyst_fwdpe(tickers, prices_map, api_key):
    """
    Fix #2: for tickers still missing Fwd P/E after ratios-ttm + key-metrics-ttm,
    hit /analyst-estimates and compute Price ÷ Forward EPS.
    Typically covers 10-15% extra tickers, pushing Fwd P/E from 73% → ~85%.
    """
    out = {t: None for t in tickers}
    if not api_key:
        return out

    def one(t):
        url = "https://financialmodelingprep.com/api/v3/analyst-estimates/{}?apikey={}".format(
            t, api_key)
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            d = r.json()
            if not isinstance(d, list) or len(d) == 0:
                return t, None
            # Take the most recent forward estimate
            item     = d[0]
            fwd_eps  = sf(item.get("estimatedEpsAvg"))
            price    = sf(prices_map.get(t))
            if fwd_eps and fwd_eps > 0 and price and price > 0:
                fwd_pe = price / fwd_eps
                if 0 < fwd_pe <= 500:
                    return t, fwd_pe
        except Exception:
            pass
        return t, None

    tl      = list(tickers)
    CHUNK   = 20
    SLEEP   = 1.5
    WORKERS = 5
    chunks  = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    for ci, chunk in enumerate(chunks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    t, v = fut.result()
                    out[t] = v
                except Exception:
                    pass
        if ci < len(chunks) - 1:
            time.sleep(SLEEP)
    return out


# ── Yahoo Finance fallback ────────────────────────────────────────────────────
def _fetch_yahoo_one(t, max_retries=2):
    result = {}
    try:
        obj   = yf.Ticker(t)
        fi    = obj.fast_info
        px_fi = None
        if fi is not None:
            mc_fi = sf(getattr(fi, "market_cap", None))
            hi_fi = sf(getattr(fi, "year_high",  None))
            lo_fi = sf(getattr(fi, "year_low",   None))
            px_fi = sf(getattr(fi, "last_price", None))
            if mc_fi: result["mc"]   = mc_fi
            if hi_fi: result["hi52"] = hi_fi
            if lo_fi: result["lo52"] = lo_fi

        info = {}
        for attempt in range(max_retries):
            try:
                info = obj.info or {}
                if any(info.get(k) for k in ["trailingPE", "forwardPE",
                                              "currentPrice", "regularMarketPrice"]):
                    break
            except Exception:
                pass
            if attempt < max_retries - 1:
                time.sleep(1.0 + random.uniform(0.5, 1.5))

        px = sf(info.get("currentPrice") or info.get("regularMarketPrice")) or px_fi

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

        # Yahoo EPS growth returned as decimal — Fix #3 consistent normalisation
        eg = sf(info.get("earningsGrowth"))
        if eg is not None:
            result["eps_growth"] = eg * 100.0

        # Yahoo returns ROE as decimal (0.15 = 15%) — Fix #3
        roe_y = sf(info.get("returnOnEquity"))
        if roe_y is not None:
            result["roe"] = roe_y * 100.0

        # Yahoo debtToEquity is already a percentage (150 = 1.5 ratio) — convert
        de_y = sf(info.get("debtToEquity"))
        if de_y is not None:
            result["debt_eq"] = de_y / 100.0

        om_y = sf(info.get("operatingMargins"))
        if om_y is not None:
            result["op_margin"] = om_y * 100.0

    except Exception:
        pass
    return t, result


@st.cache_data(ttl=86400)
def fetch_yahoo_fallback_parallel(tickers):
    tl     = list(tickers)
    out    = {}
    CHUNK  = 25
    SLEEP  = 2.0
    WKRS   = 6
    chunks = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    for ci, chunk in enumerate(chunks):
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futures = {ex.submit(_fetch_yahoo_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(futures):
                try:
                    t, d = fut.result()
                    out[t] = d
                except Exception:
                    t = futures[fut]; out[t] = {}
        if ci < len(chunks) - 1:
            time.sleep(SLEEP + random.uniform(0, 0.5))
    return out


# ── Revenue ───────────────────────────────────────────────────────────────────
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


# ── Merge all sources ─────────────────────────────────────────────────────────
def merge_fundamental_data(fmp_quotes, fmp_ratios, fmp_metrics,
                           fmp_analyst_fwdpe, yahoo_fallback, tickers):
    merged = {}
    for t in tickers:
        fq  = fmp_quotes.get(t, {})
        fr  = fmp_ratios.get(t, {})
        fm  = fmp_metrics.get(t, {})
        fa  = fmp_analyst_fwdpe.get(t)   # scalar or None
        yb  = yahoo_fallback.get(t, {})

        def first(*vals):
            for v in vals:
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return v
            return None

        # Fwd P/E waterfall: ratios-ttm → key-metrics-ttm → analyst-estimates → Yahoo
        fwd_pe = first(fr.get("fwd_pe"), fm.get("fwd_pe"), fa, yb.get("fwd_pe"))

        # PEG: ratios-ttm is primary (Fix #1) — no longer relying on /quote
        peg_val = first(fr.get("peg"))
        peg_src = fr.get("peg_src") if peg_val is not None else "—"

        # PE
        pe_val = first(fq.get("pe"), yb.get("pe"))
        pe_src = fq.get("pe_src") if fq.get("pe") is not None else yb.get("pe_src", "Yahoo")

        # ROE — all sources now normalised to percent (Fix #3)
        roe = first(fr.get("roe"), yb.get("roe"))

        # Op Margin — all sources normalised to percent (Fix #3)
        om = first(fr.get("op_margin"), yb.get("op_margin"))

        # Debt/Equity
        de = first(fr.get("debt_eq"), yb.get("debt_eq"))

        # EPS growth
        eps_g = first(fm.get("eps_growth"), yb.get("eps_growth"))
        g_src = "FMP" if fm.get("eps_growth") is not None else (
                "Yahoo" if yb.get("eps_growth") is not None else None)

        merged[t] = {
            "pe":             pe_val,
            "pe_src":         pe_src,
            "fwd_pe":         fwd_pe,
            "peg":            peg_val,
            "peg_src":        peg_src,
            "mc":             first(fq.get("mc"), yb.get("mc")),
            "hi52":           first(fq.get("hi52"), yb.get("hi52")),
            "lo52":           first(fq.get("lo52"), yb.get("lo52")),
            "revenue_growth": first(fm.get("revenue_growth")),
            "eps_growth":     eps_g,
            "growth_src":     g_src,
            "roe":            roe,
            "debt_eq":        de,
            "op_margin":      om,
        }
    return merged


# ── Quality score ─────────────────────────────────────────────────────────────
def compute_quality_score(roe, debt_eq, op_margin):
    scores = []
    if roe       is not None: scores.append(min(max(roe / 50.0 * 100.0, 0), 100))
    else:                     scores.append(0.0)
    if debt_eq   is not None: scores.append(max(0, min(100, (1 - debt_eq / 2.0) * 100.0)))
    else:                     scores.append(0.0)
    if op_margin is not None: scores.append(min(max(op_margin / 40.0 * 100.0, 0), 100))
    else:                     scores.append(0.0)
    return sum(scores) / len(scores)


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

        val_input = elig["Fwd P/E"].fillna(elig["P/E"])
        elig["_s_val"]   = percentile_score(val_input,           ascending=True)
        elig["_s_peg"]   = percentile_score(elig["PEG"],         ascending=True)
        elig["_s_etraj"] = percentile_score(elig["Earn Traj"],   ascending=False)

        qs = elig["Quality Score"]
        q_min, q_max = qs.min(), qs.max()
        if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min:
            elig["_s_quality"] = (qs - q_min) / (q_max - q_min) * 100.0
        else:
            elig["_s_quality"] = qs.fillna(0.0)
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)

        elig["_mom_avg"] = elig[["Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%"]].mean(axis=1)
        elig["_s_mom"]   = percentile_score(elig["_mom_avg"], ascending=False)

        raw = (W["valuation"] * elig["_s_val"]   +
               W["peg"]       * elig["_s_peg"]   +
               W["quality"]   * elig["_s_quality"] +
               W["earn_traj"] * elig["_s_etraj"] +
               W["momentum"]  * elig["_s_mom"])

        factor_cols = ["P/E", "PEG", "Quality Score", "Earn Traj", "_mom_avg"]
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

        price = to_num(prices_map.get(t))
        fi    = merged_map.get(t, {})
        mc    = to_num(fi.get("mc"))
        pe    = to_num(fi.get("pe"))
        fwd   = to_num(fi.get("fwd_pe"))
        hi    = to_num(fi.get("hi52"))
        lo    = to_num(fi.get("lo52"))
        roe   = to_num(fi.get("roe"))
        de    = to_num(fi.get("debt_eq"))
        om    = to_num(fi.get("op_margin"))

        # 52W position
        pos52 = None
        if pd.notna(price) and pd.notna(hi) and pd.notna(lo) and hi != lo:
            pos52 = float((price - lo) / (hi - lo) * 100.0)

        # Revenue
        rev4            = revenue_map.get(t, [None]*4)
        rq1, rq2, rq3, rq4 = [to_num(x) for x in rev4]
        growth          = revenue_growth_pct_cagr([rq1, rq2, rq3, rq4])

        # PEG waterfall
        peg_direct = to_num(fi.get("peg"))
        peg        = None
        peg_method = "—"
        if pd.notna(peg_direct):
            peg        = float(peg_direct)
            peg_method = fi.get("peg_src") or "FMP-ratios"
        else:
            pe_for_peg     = fwd if pd.notna(fwd) else pe
            eps_g          = fi.get("eps_growth")
            growth_for_peg = None
            g_src          = fi.get("growth_src") or ""
            if eps_g is not None:
                eg = float(eps_g)
                if eg >= MIN_GROWTH_PCT_FOR_PEG:
                    growth_for_peg = eg
                    peg_method     = "{} EPS growth".format(g_src)
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

        # Quality
        q_score = compute_quality_score(
            float(roe) if pd.notna(roe) else None,
            float(de)  if pd.notna(de)  else None,
            float(om)  if pd.notna(om)  else None,
        )

        # Momentum
        mom     = momentum_map.get(t, {})
        ret_1mo = to_num(mom.get("ret_1mo"))
        ret_3mo = to_num(mom.get("ret_3mo"))
        ret_6mo = to_num(mom.get("ret_6mo"))

        # Source summary
        parts = []
        ps = fi.get("pe_src")
        if ps and ps != "—": parts.append("PE:{}".format(ps))
        gs = fi.get("peg_src")
        if gs and gs != "—": parts.append("PEG:{}".format(gs))
        gr = fi.get("growth_src")
        if gr and gr != "—": parts.append("G:{}".format(gr))
        data_src = " | ".join(parts) if parts else "Yahoo only"

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

    num_cols = ["Price", "Mkt Cap", "P/E", "Fwd P/E", "PEG", "Earn Traj",
                "52W Pos%", "ROE%", "Debt/Eq", "Op Margin%", "Quality Score",
                "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%",
                "Rev Q1", "Rev Q2", "Rev Q3", "Rev Q4", "Rev Growth% (CAGR)"]
    for c in num_cols:
        if c in scr.columns:
            scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns:
        scr["Rank"] = pd.NA
    return scr


# ── Quality flag ──────────────────────────────────────────────────────────────
def quality_flag(roe, de, om):
    flags = []
    if roe is not None and not pd.isna(roe) and roe < QUALITY_THRESHOLDS["roe_min"]:
        flags.append("ROE<10%")
    if de  is not None and not pd.isna(de)  and de  > QUALITY_THRESHOLDS["debt_eq_max"]:
        flags.append("D/E>2")
    if om  is not None and not pd.isna(om)  and om  < QUALITY_THRESHOLDS["op_margin_min"]:
        flags.append("Margin<5%")
    return ", ".join(flags) if flags else "Pass"


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

    med_pe    = sdata["P/E"].median()
    med_fwd   = sdata["Fwd P/E"].median()
    med_qual  = sdata["Quality Score"].median()
    med_mom   = sdata[["Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%"]].mean(axis=1).median()
    med_peg   = sdata["PEG"].median()

    st.markdown(
        "<div style='background:#12122a;border:1px solid #2a2a4a;border-radius:12px;"
        "padding:16px 20px;margin-bottom:16px;'>"
        "<span style='color:#aaa;font-size:13px;'>Sector Analysis  </span>"
        "<span style='color:#fff;font-size:14px;font-weight:700;'>{}</span>"
        "</div>".format(label),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.markdown(_kpi("Sector Mkt Cap",   fmt_mc(sector_mc), "sector total"), unsafe_allow_html=True)
    c2.markdown(_kpi("S&P 500 Mkt Cap",  fmt_mc(total_mc),  "all stocks"),  unsafe_allow_html=True)
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
                     "ROE+D/E+Margin", "#4ade80"),
                unsafe_allow_html=True)
    c6.markdown(_kpi("Median PEG",
                     "{:.2f}".format(med_peg) if pd.notna(med_peg) else "N/A",
                     "price/earnings/growth", "#a78bfa"),
                unsafe_allow_html=True)

    if not is_all:
        top3   = sdata[sdata["Rank"].notna()].sort_values("Rank").head(3)
        badges = "  ".join(
            "<span style='background:#1a2a4a;color:#93c5fd;padding:3px 10px;"
            "border-radius:6px;font-weight:700;font-size:13px;'>{}</span>".format(row["Ticker"])
            for _, row in top3.iterrows()
        )
        st.markdown(
            "<div style='background:#1e1e2e;border-radius:10px;padding:14px 16px;margin-bottom:12px;'>"
            "<div style='color:#aaa;font-size:11px;margin-bottom:8px;'>Top Ranked in Sector</div>"
            "<div>{}</div>"
            "<div style='color:#555;font-size:10px;margin-top:8px;'>"
            "Score = Valuation 30% + Quality 25% + PEG 20% + Earn Traj 15% + Momentum 10%"
            "</div></div>".format(badges or "<span style='color:#555;'>No ranked stocks</span>"),
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── APP ───────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="S&P 500 Screener", layout="wide", page_icon="📊")
st.markdown(
    "<style>div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}</style>",
    unsafe_allow_html=True,
)

st.markdown("## S&P 500 Fundamental Screener")
st.caption("5-factor percentile ranking · Quality-aware · Momentum-included · PEG via FMP ratios-ttm · Source-transparent")

col_r, col_t = st.columns([1, 6])
with col_r:
    if st.button("Refresh"):
        st.cache_data.clear()
        st.rerun()
with col_t:
    st.caption("Last loaded: {} · Prices: 1hr · Fundamentals: 24hr".format(
        datetime.now().strftime("%I:%M %p")))

fmp_key = get_fmp_key()
if not fmp_key:
    st.warning(
        "No FMP API key found. Add [fmp] api_key to Streamlit Secrets. "
        "PEG, ROE, D/E, Op Margin, Fwd P/E all require FMP. Falling back to Yahoo only."
    )

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

with st.spinner("Fetching momentum (1mo / 3mo / 6mo returns)..."):
    momentum = fetch_momentum_batch(tickers)

fmp_quotes  = {}
fmp_ratios  = {}
fmp_metrics = {}
fmp_fwdpe   = {}

if fmp_key:
    with st.spinner("FMP: bulk quotes (PE, MC, 52W) for all {} tickers...".format(len(tickers))):
        fmp_quotes = fetch_fmp_bulk_quotes(tickers, fmp_key)

    with st.spinner("FMP: ratios-ttm (PEG, ROE, D/E, Op Margin) for all {} tickers...".format(len(tickers))):
        fmp_ratios = fetch_fmp_ratios_bulk(tickers, fmp_key)

    with st.spinner("FMP: key-metrics-ttm (EPS growth, Fwd PE) for all {} tickers...".format(len(tickers))):
        fmp_metrics = fetch_fmp_key_metrics_bulk(tickers, fmp_key)

    # Fix #2 — only fetch analyst estimates for tickers still missing Fwd P/E
    missing_fwd = tuple(
        t for t in tickers
        if fmp_ratios.get(t, {}).get("fwd_pe")  is None
        and fmp_metrics.get(t, {}).get("fwd_pe") is None
    )
    if missing_fwd:
        with st.spinner("FMP: analyst estimates (Fwd PE) for {} tickers still missing...".format(len(missing_fwd))):
            fmp_fwdpe = fetch_fmp_analyst_fwdpe(missing_fwd, prices, fmp_key)

# Yahoo fallback for anything still missing PE
missing_pe = tuple(
    t for t in tickers
    if fmp_quotes.get(t, {}).get("pe") is None
)
yahoo_fallback = {}
if missing_pe:
    with st.spinner("Yahoo fallback: {} tickers missing PE from FMP...".format(len(missing_pe))):
        yahoo_fallback = fetch_yahoo_fallback_parallel(missing_pe)

with st.spinner("Merging data sources..."):
    merged_map = merge_fundamental_data(
        fmp_quotes, fmp_ratios, fmp_metrics, fmp_fwdpe, yahoo_fallback, tickers)

with st.spinner("Fetching quarterly revenue..."):
    rev_map = fetch_last4_revenue_parallel(tickers)

# ── Coverage banner ───────────────────────────────────────────────────────────
total_t   = len(tickers)
has_pe    = sum(1 for t in tickers if merged_map.get(t, {}).get("pe")     is not None)
has_fwd   = sum(1 for t in tickers if merged_map.get(t, {}).get("fwd_pe") is not None)
has_peg   = sum(1 for t in tickers if merged_map.get(t, {}).get("peg")    is not None)
has_roe   = sum(1 for t in tickers if merged_map.get(t, {}).get("roe")    is not None)

st.info(
    "Data coverage — "
    "P/E: {}/{} ({:.0f}%) · "
    "Fwd P/E: {}/{} ({:.0f}%) · "
    "PEG: {}/{} ({:.0f}%) · "
    "ROE: {}/{} ({:.0f}%) · "
    "Sources: FMP ratios-ttm (primary) + analyst-estimates (Fwd PE) + Yahoo (fallback)".format(
        has_pe,  total_t, has_pe  / total_t * 100,
        has_fwd, total_t, has_fwd / total_t * 100,
        has_peg, total_t, has_peg / total_t * 100,
        has_roe, total_t, has_roe / total_t * 100,
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
        "Sector then Rank", "Score high to low",
        "Price low to high", "Price high to low", "Mkt Cap high to low",
        "PE low to high", "Fwd PE low to high", "PEG low to high",
        "Quality Score high", "Earn Traj high to low",
        "Rev Growth high to low", "Momentum (3Mo) high", "52W Pos low to high",
    ])
    pe_max   = fc3.number_input("Max PE",              value=9999,  step=50)
    peg_max  = fc4.number_input("Max PEG",             value=999.0, step=1.0)
    mc_min_b = fc5.number_input("Min Market Cap ($B)", value=0,     step=5)

with st.expander("Quality Filters", expanded=False):
    qc1, qc2, qc3, qc4 = st.columns(4)
    roe_min_f  = qc1.number_input("Min ROE (%)",          value=0.0,  step=5.0)
    de_max_f   = qc2.number_input("Max Debt/Equity",      value=99.0, step=0.5)
    om_min_f   = qc3.number_input("Min Op Margin (%)",    value=0.0,  step=5.0)
    qual_min_f = qc4.number_input("Min Quality Score",    value=0.0,  step=5.0)

with st.expander("Momentum & Display", expanded=False):
    mc1, mc2 = st.columns(2)
    mom_min  = mc1.number_input("Min 3Mo Return (%)", value=-999.0, step=5.0)
    hide_nope = mc2.checkbox("Hide stocks with no P/E or Fwd P/E", value=False)

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
filt = filt[(filt["Ret 3Mo%"].isna())      | (filt["Ret 3Mo%"]      >= mom_min)]
if hide_nope:
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
sc, sa = sort_map.get(sort_by, (["Sector", "Rank"], [True, True]))
filt   = filt.sort_values(sc, ascending=sa, na_position="last")

st.caption("Showing {} of {} stocks · Sector: {} · Sort: {}".format(
    len(filt), len(scr), sector_sel, sort_by))

# ── Display table ─────────────────────────────────────────────────────────────
disp = filt.copy()
disp["Price ($)"]    = disp["Price"].round(2)
disp["Mkt Cap ($B)"] = (disp["Mkt Cap"]  / 1e9).round(2)
disp["Rev Q1 ($B)"]  = (disp["Rev Q1"]   / 1e9).round(2)
disp["Rev Q2 ($B)"]  = (disp["Rev Q2"]   / 1e9).round(2)
disp["Rev Q3 ($B)"]  = (disp["Rev Q3"]   / 1e9).round(2)
disp["Rev Q4 ($B)"]  = (disp["Rev Q4"]   / 1e9).round(2)
disp["Quality Flag"] = disp.apply(
    lambda r: quality_flag(r.get("ROE%"), r.get("Debt/Eq"), r.get("Op Margin%")), axis=1)

for c in ["P/E", "Fwd P/E", "PEG", "Earn Traj", "52W Pos%",
          "ROE%", "Debt/Eq", "Op Margin%", "Quality Score",
          "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Score",
          "Rev Growth% (CAGR)"]:
    if c in disp.columns:
        disp[c] = disp[c].round(2)

disp["Rank"] = disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

COLS = [
    "Ticker", "Sector", "Price ($)", "Mkt Cap ($B)",
    "P/E", "Fwd P/E", "PEG", "PEG Method", "Earn Traj",
    "ROE%", "Debt/Eq", "Op Margin%", "Quality Score", "Quality Flag",
    "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%",
    "52W Pos%", "Score", "Rank",
    "Rev Q1 ($B)", "Rev Q2 ($B)", "Rev Q3 ($B)", "Rev Q4 ($B)",
    "Rev Growth% (CAGR)", "Data Sources",
]
disp_final = disp[[c for c in COLS if c in disp.columns]].copy()
st.dataframe(disp_final, use_container_width=True, height=680)

# ── Export ────────────────────────────────────────────────────────────────────
st.download_button(
    label="Download filtered results as CSV",
    data=disp_final.to_csv(index=False).encode("utf-8"),
    file_name="sp500_screener_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M")),
    mime="text/csv",
)

# ── Legend tabs ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Column Reference Guide")
tab1, tab2, tab3, tab4 = st.tabs(["Valuation", "Quality", "Momentum & Position", "Ranking"])

with tab1:
    st.markdown(
        "**P/E** — Price ÷ trailing 12-month EPS. Lower = cheaper. e.g. P/E 20 = $20 paid per $1 annual earnings.\n\n"
        "**Fwd P/E** — Price ÷ next 12-month estimated EPS. Fwd P/E lower than P/E = earnings growing.\n\n"
        "**PEG** — P/E ÷ EPS growth rate. Source: FMP `/ratios-ttm` (priceEarningsGrowthRatioTTM). "
        "PEG < 1 = undervalued for growth. 1–2 = fair. > 2 = expensive. Only shown when growth >= 5%.\n\n"
        "**PEG Method** — Shows which source was used: 'FMP-ratios' = analyst-based TTM PEG from FMP. "
        "'FMP EPS growth' = computed from EPS growth estimate. 'Rev CAGR fallback' = computed from revenue CAGR (least reliable).\n\n"
        "**Earn Traj** — Trailing P/E ÷ Forward P/E. > 1 = earnings growing. < 1 = earnings shrinking."
    )

with tab2:
    st.markdown(
        "**ROE%** — Net income ÷ equity × 100. > 15% strong, < 10% flagged. "
        "Source: FMP `/ratios-ttm` returnOnEquityTTM (decimal converted to %).\n\n"
        "**Debt/Eq** — Total debt ÷ equity. > 2.0 flagged as high risk. "
        "Source: FMP debtEquityRatioTTM.\n\n"
        "**Op Margin%** — Operating income ÷ revenue × 100. < 5% flagged. "
        "Source: FMP operatingProfitMarginTTM (decimal converted to %).\n\n"
        "**Quality Score (0–100)** — Equal-weight composite of ROE, D/E, and Op Margin sub-scores. "
        "Missing factors score 0, which lowers the composite.\n\n"
        "**Quality Flag** — Pass / fail against thresholds: ROE >= 10%, D/E <= 2.0, Op Margin >= 5%."
    )

with tab3:
    st.markdown(
        "**Ret 1Mo / 3Mo / 6Mo%** — Trailing price return over 1, 3, and 6 months from Yahoo Finance monthly closes. "
        "Positive 3–6mo momentum is one of the most documented factor signals in academic finance.\n\n"
        "**52W Pos%** — Position between 52-week low (0%) and high (100%). "
        "Lower = more upside room but also possible value trap — combine with quality and valuation filters."
    )

with tab4:
    st.markdown(
        "**Score (0–100)** — Percentile composite within each sector. "
        "Each factor converted to 0–100 percentile before weighting — preserves magnitude, not just rank order.\n\n"
        "**Rank** — Position in sector by Score. 1 = best in sector.\n\n"
        "**Weights:** Valuation 30% · Quality 25% · PEG 20% · Earn Traj 15% · Momentum 10%\n\n"
        "**Missing data penalty:** 2 factors missing = score × 0.85. 3+ missing = score × 0.70.\n\n"
        "**Coverage fix notes (v3):**\n"
        "- PEG now sourced from `/ratios-ttm` (free tier) instead of `/quote` (paid field) — expected 0% → ~70%\n"
        "- Fwd P/E now has 4-layer waterfall: ratios-ttm → key-metrics-ttm → analyst-estimates → Yahoo\n"
        "- ROE and Op Margin decimal normalisation unified across FMP and Yahoo sources"
    )

st.markdown(
    "**Data Sources** — "
    "FMP `/quote` (PE, MC, 52W) · "
    "FMP `/ratios-ttm` (PEG, ROE, D/E, Op Margin) · "
    "FMP `/key-metrics-ttm` (EPS growth, Fwd PE) · "
    "FMP `/analyst-estimates` (Fwd PE gap fill) · "
    "Yahoo Finance (PE/Fwd PE fallback, revenue, momentum) · "
    "S&P 500 universe: Wikipedia GICS"
)
