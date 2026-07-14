# screener_app.py  v10
# ─────────────────────────────────────────────────────────────────────────────
# v10 CHANGES from v9:
#   1. Page 1 filters: simplified to 5 controls in ONE row
#      (Sector, Sort By, Min Mkt Cap $B, Max PE, Min Mkt Cap — clean single row)
#   2. Page 2: render_reference_guide() fully populated with all metric
#      explanations, formulas, numeric examples, and sector benchmarks
#   3. All v9 logic (sector-adaptive weights, ROIC computation, Earn Traj,
#      Financials ROE override, FMP bonus layer) preserved verbatim
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

# ── Yahoo Finance per-ticker fundamentals ─────────────────────────────────────
def _fetch_yahoo_fundamentals_one(t):
    result = {
        "pe": None, "pe_src": None,
        "fwd_pe": None,
        "peg": None, "peg_src": None,
        "roe": None,
        "roic": None,
        "op_margin": None,
        "debt_eq": None,
        "eps_growth": None,
        "int_coverage": None,
        "earn_traj": None,
        "mc": None,
        "hi52": None,
        "lo52": None,
    }
    try:
        obj = yf.Ticker(t)

        try:
            fi = obj.fast_info
            if fi is not None:
                mc_fi = sf(getattr(fi, "market_cap", None))
                hi_fi = sf(getattr(fi, "year_high",  None))
                lo_fi = sf(getattr(fi, "year_low",   None))
                if mc_fi: result["mc"]   = mc_fi
                if hi_fi: result["hi52"] = hi_fi
                if lo_fi: result["lo52"] = lo_fi
        except Exception:
            pass

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

        t_pe  = sf(info.get("trailingPE"))
        t_eps = sf(info.get("trailingEps"))
        if t_pe and 0 < t_pe <= 10_000:
            result["pe"]     = t_pe
            result["pe_src"] = "Yahoo"
        elif t_eps and t_eps > 0 and px and px > 0:
            result["pe"]     = px / t_eps
            result["pe_src"] = "Yahoo(calc)"

        f_pe  = sf(info.get("forwardPE"))
        f_eps = sf(info.get("forwardEps"))
        if f_pe and 0 < f_pe <= 10_000:
            result["fwd_pe"] = f_pe
        elif f_eps and f_eps > 0 and px and px > 0:
            result["fwd_pe"] = px / f_eps

        peg_y = sf(info.get("pegRatio"))
        if peg_y and 0 < peg_y <= 500:
            result["peg"]     = peg_y
            result["peg_src"] = "Yahoo"

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
        if fwd_eps_val is not None and trail_eps_val is not None and abs(trail_eps_val) > 0.01:
            earn_traj_raw       = (fwd_eps_val - trail_eps_val) / abs(trail_eps_val)
            result["earn_traj"] = max(-1.0, min(1.0, earn_traj_raw))

        if result["mc"] is None:
            mc_y = sf(info.get("marketCap"))
            if mc_y: result["mc"] = mc_y
        if result["hi52"] is None:
            h52 = sf(info.get("fiftyTwoWeekHigh"))
            if h52: result["hi52"] = h52
        if result["lo52"] is None:
            l52 = sf(info.get("fiftyTwoWeekLow"))
            if l52: result["lo52"] = l52

        try:
            qfin = obj.quarterly_financials
            if qfin is not None and not qfin.empty:
                ebit_row = None
                for nm in ["EBIT", "Operating Income", "Ebit"]:
                    if nm in qfin.index:
                        ebit_row = nm
                        break
                int_row = None
                for nm in ["Interest Expense", "Interest Expense Non Operating", "Net Interest Income"]:
                    if nm in qfin.index:
                        int_row = nm
                        break
                if ebit_row and int_row:
                    ebit_ttm = qfin.loc[ebit_row].dropna().head(4).sum()
                    int_ttm  = abs(qfin.loc[int_row].dropna().head(4).sum())
                    if int_ttm > 0 and ebit_ttm > 0:
                        result["int_coverage"] = min(float(ebit_ttm / int_ttm), 100.0)
        except Exception:
            pass

        try:
            qfin = obj.quarterly_financials
            bs   = obj.quarterly_balance_sheet
            if qfin is not None and not qfin.empty and bs is not None and not bs.empty:
                op_inc_row = None
                for nm in ["Operating Income", "EBIT", "Ebit"]:
                    if nm in qfin.index:
                        op_inc_row = nm
                        break
                tax_row = None
                for nm in ["Tax Provision", "Income Tax Expense", "Tax Expense"]:
                    if nm in qfin.index:
                        tax_row = nm
                        break
                pretax_row = None
                for nm in ["Pretax Income", "Income Before Tax", "EBT"]:
                    if nm in qfin.index:
                        pretax_row = nm
                        break
                if op_inc_row:
                    op_inc_ttm   = float(qfin.loc[op_inc_row].dropna().head(4).sum())
                    eff_tax_rate = 0.21
                    if tax_row and pretax_row:
                        tax_ttm    = float(qfin.loc[tax_row].dropna().head(4).sum())
                        pretax_ttm = float(qfin.loc[pretax_row].dropna().head(4).sum())
                        if pretax_ttm > 0 and tax_ttm >= 0:
                            computed_rate = tax_ttm / pretax_ttm
                            if 0 < computed_rate < 0.6:
                                eff_tax_rate = computed_rate
                    nopat      = op_inc_ttm * (1 - eff_tax_rate)
                    equity_val = None
                    for nm in ["Total Stockholders Equity", "Stockholders Equity",
                               "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                        if nm in bs.index:
                            eq_s = bs.loc[nm].dropna()
                            if len(eq_s) > 0:
                                equity_val = float(eq_s.iloc[0])
                                break
                    debt_val = None
                    for nm in ["Total Debt", "Net Debt", "Long Term Debt",
                               "Long Term Debt And Capital Lease Obligation"]:
                        if nm in bs.index:
                            d_s = bs.loc[nm].dropna()
                            if len(d_s) > 0:
                                debt_val = float(d_s.iloc[0])
                                break
                    cash_val = None
                    for nm in ["Cash And Cash Equivalents",
                               "Cash Cash Equivalents And Short Term Investments",
                               "Cash Financial", "Cash And Short Term Investments"]:
                        if nm in bs.index:
                            c_s = bs.loc[nm].dropna()
                            if len(c_s) > 0:
                                cash_val = float(c_s.iloc[0])
                                break
                    if equity_val is not None and debt_val is not None:
                        cash_use         = cash_val if cash_val is not None else 0
                        invested_capital = equity_val + debt_val - cash_use
                        if invested_capital > 0 and nopat != 0:
                            roic_computed = (nopat / invested_capital) * 100.0
                            if -100 < roic_computed < 200:
                                result["roic"] = roic_computed
        except Exception:
            pass

    except Exception:
        pass
    return t, result


@st.cache_data(ttl=86400)
def fetch_yahoo_fundamentals_all(tickers):
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

# ── FMP /quote bulk ───────────────────────────────────────────────────────────
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

# ── FMP /ratios-ttm ───────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_ratios_if_available(tickers, api_key):
    out = {}
    if not api_key:
        return out
    test_url = "https://financialmodelingprep.com/api/v3/ratios-ttm/AAPL?apikey={}".format(api_key)
    try:
        r    = requests.get(test_url, timeout=10)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            st.caption("FMP /ratios-ttm: not available on your tier.")
            return out
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
            item     = d[0]
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
                    if d: out[t] = d
                except Exception:
                    pass
        if ci < len(chunks) - 1:
            time.sleep(SLEEP)
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

# ── Merge sources ─────────────────────────────────────────────────────────────
def merge_all_sources(yahoo_data, fmp_quotes, fmp_ratios, tickers):
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

        pe_val  = first(fq.get("pe"),      yb.get("pe"))
        pe_src  = ("FMP-quote" if fq.get("pe") is not None else yb.get("pe_src", "Yahoo"))
        fwd_pe  = first(fr.get("fwd_pe"),  yb.get("fwd_pe"))
        peg_val = first(fr.get("peg"),     yb.get("peg"))
        peg_src = ("FMP-ratios" if fr.get("peg") is not None else
                   yb.get("peg_src", "Yahoo") if yb.get("peg") is not None else "—")
        roic    = first(fr.get("roic"),    yb.get("roic"))
        roe     = first(fr.get("roe"),     yb.get("roe"))
        ic      = first(fr.get("int_coverage"), yb.get("int_coverage"))
        om      = first(fr.get("op_margin"),    yb.get("op_margin"))
        de      = first(fr.get("debt_eq"),      yb.get("debt_eq"))
        eps_g   = yb.get("eps_growth")
        g_src   = "Yahoo" if eps_g is not None else None
        earn_traj = yb.get("earn_traj")
        mc      = first(fq.get("mc"),   yb.get("mc"))
        hi52    = first(fq.get("hi52"), yb.get("hi52"))
        lo52    = first(fq.get("lo52"), yb.get("lo52"))

        merged[t] = {
            "pe": pe_val, "pe_src": pe_src, "fwd_pe": fwd_pe,
            "peg": peg_val, "peg_src": peg_src,
            "roic": roic, "roe": roe, "int_coverage": ic,
            "op_margin": om, "debt_eq": de,
            "eps_growth": eps_g, "growth_src": g_src,
            "earn_traj": earn_traj,
            "mc": mc, "hi52": hi52, "lo52": lo52,
        }
    return merged

# ── Quality Score ─────────────────────────────────────────────────────────────
def compute_quality_score(roic, roe, int_coverage, op_margin, sector=None):
    scores = []
    if sector in ROE_PRIMARY_SECTORS:
        profitability = roe
    else:
        profitability = roic if roic is not None else roe

    if profitability is not None and not pd.isna(profitability):
        pf = float(profitability)
        scores.append(min(100.0, np.log1p(pf) / np.log1p(30.0) * 100.0) if pf > 0 else 0.0)
    else:
        scores.append(0.0)

    if int_coverage is not None and not pd.isna(int_coverage):
        scores.append(min(100.0, max(0.0, float(int_coverage) / 10.0 * 100.0)))
    else:
        scores.append(0.0)

    if sector not in ROE_PRIMARY_SECTORS:
        if op_margin is not None and not pd.isna(op_margin):
            scores.append(min(100.0, max(0.0, float(op_margin) / 40.0 * 100.0)))
        else:
            scores.append(0.0)

    return sum(scores) / len(scores)

# ── Quality flag ──────────────────────────────────────────────────────────────
def quality_flag(roic, roe, ic, om, de, sector=None):
    flags = []
    if sector in ROE_PRIMARY_SECTORS:
        profitability = roe
        prof_label    = "ROE"
    else:
        profitability = roic if (roic is not None and not pd.isna(roic)) else roe
        prof_label    = "ROIC" if (roic is not None and not pd.isna(roic)) else "ROE"

    if profitability is not None and not pd.isna(profitability) \
            and profitability < QUALITY_THRESHOLDS["roic_min"]:
        flags.append("{}<8%".format(prof_label))
    if ic is not None and not pd.isna(ic) and ic < QUALITY_THRESHOLDS["int_coverage_min"]:
        flags.append("IntCov<3x")
    if sector not in ROE_PRIMARY_SECTORS:
        if om is not None and not pd.isna(om) and om < QUALITY_THRESHOLDS["op_margin_min"]:
            flags.append("Margin<5%")
    de_note = " | D/E:{:.1f}".format(de) if (de is not None and not pd.isna(de)) else ""
    return (", ".join(flags) if flags else "Pass") + de_note

# ── Conviction Score ──────────────────────────────────────────────────────────
def compute_conviction_scores(scr):
    KEY_FACTORS = ["P/E", "Fwd P/E", "PEG", "Quality Score", "Momentum Score", "Earn Traj"]
    n_factors   = len(KEY_FACTORS)
    scr         = scr.copy()

    def completeness(row):
        present = sum(1 for c in KEY_FACTORS if c in row.index and pd.notna(row[c]))
        return present / n_factors

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

    for sector in scr["Sector"].dropna().unique():
        g    = scr[scr["Sector"] == sector].copy()
        elig = g[g["Eligible"]].copy()
        if elig.empty:
            continue

        W = SECTOR_FACTOR_WEIGHTS.get(sector, DEFAULT_FACTOR_WEIGHTS)

        pe_input          = elig["Fwd P/E"].fillna(elig["P/E"])
        elig["_s_val"]   = percentile_score(pe_input,               ascending=True)
        elig["_s_peg"]   = percentile_score(elig["PEG"],             ascending=True)
        elig["_s_mom"]   = percentile_score(elig["Momentum Score"],  ascending=False)
        elig["_s_etraj"] = percentile_score(elig["Earn Traj"],       ascending=False)

        qs    = elig["Quality Score"]
        q_min = qs.min(); q_max = qs.max()
        if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min:
            elig["_s_quality"] = (qs - q_min) / (q_max - q_min) * 100.0
        else:
            elig["_s_quality"] = qs.fillna(0.0)
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)

        raw = (W["valuation"] * elig["_s_val"]      +
               W["quality"]   * elig["_s_quality"]  +
               W["peg"]       * elig["_s_peg"]      +
               W["earn_traj"] * elig["_s_etraj"]    +
               W["momentum"]  * elig["_s_mom"])

        factor_cols = ["P/E", "PEG", "Quality Score", "Earn Traj", "Momentum Score"]
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

        px_info   = prices_map.get(t, {})
        price     = to_num(px_info.get("price"))
        fi        = merged_map.get(t, {})
        mc        = to_num(fi.get("mc"))
        pe        = to_num(fi.get("pe"))
        fwd       = to_num(fi.get("fwd_pe"))
        hi        = to_num(fi.get("hi52"))
        lo        = to_num(fi.get("lo52"))
        roic      = to_num(fi.get("roic"))
        roe       = to_num(fi.get("roe"))
        ic        = to_num(fi.get("int_coverage"))
        om        = to_num(fi.get("op_margin"))
        de        = to_num(fi.get("debt_eq"))
        earn_traj = to_num(fi.get("earn_traj"))

        pos52 = None
        if pd.notna(price) and pd.notna(hi) and pd.notna(lo) and hi != lo:
            pos52 = float((price - lo) / (hi - lo) * 100.0)

        rev4               = revenue_map.get(t, [None]*4)
        rq1, rq2, rq3, rq4 = [to_num(x) for x in rev4]
        growth             = revenue_growth_pct_cagr([rq1, rq2, rq3, rq4])

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
            sector=sec,
        )

        mom       = momentum_map.get(t, {})
        ret_1mo   = to_num(mom.get("ret_1mo"))
        ret_3mo   = to_num(mom.get("ret_3mo"))
        ret_6mo   = to_num(mom.get("ret_6mo"))
        mom_score = to_num(mom.get("momentum_score"))
        t_vol     = to_num(mom.get("trailing_vol"))

        parts = []
        ps = fi.get("pe_src")
        if ps: parts.append("PE:{}".format(ps))
        pg = fi.get("peg_src")
        if pg and pg != "—": parts.append("PEG:{}".format(pg))
        if roic is not None and not pd.isna(roic): parts.append("ROIC:Yahoo")
        if ic   is not None and not pd.isna(ic):   parts.append("IC:Yahoo")
        if earn_traj is not None and not pd.isna(earn_traj): parts.append("ET:Yahoo")
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

    total_sp500_mc = scr["Mkt Cap"].sum()
    if total_sp500_mc > 0:
        scr["MC% of S&P500"] = (scr["Mkt Cap"] / total_sp500_mc * 100.0)
    else:
        scr["MC% of S&P500"] = None

    num_cols = ["Price", "Mkt Cap", "P/E", "Fwd P/E", "PEG", "52W Pos%",
                "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
                "Quality Score", "Earn Traj", "Momentum Score",
                "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
                "MC% of S&P500",
                "Rev Q1", "Rev Q2", "Rev Q3", "Rev Q4", "Rev Growth% (CAGR)"]
    for c in num_cols:
        if c in scr.columns:
            scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns:
        scr["Rank"] = pd.NA
    scr = compute_conviction_scores(scr)
    return scr

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
    c1.markdown(_kpi("Sector Mkt Cap",  fmt_mc(sector_mc), "sector total"),  unsafe_allow_html=True)
    c2.markdown(_kpi("S&P 500 Mkt Cap", fmt_mc(total_mc),  "all 503 stocks"), unsafe_allow_html=True)
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
                     "ROIC+IntCov+Margin", "#4ade80"),
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
            "margin-bottom:8px;'>"
            "<div style='color:#aaa;font-size:11px;margin-bottom:8px;'>"
            "Top Ranked in Sector</div>"
            "<div>{}</div></div>".format(
                badges or "<span style='color:#555;'>No ranked stocks</span>"
            ),
            unsafe_allow_html=True,
        )

        W = SECTOR_FACTOR_WEIGHTS.get(sector_sel, DEFAULT_FACTOR_WEIGHTS)
        weight_text = "  |  ".join(
            "{}: {:.0f}%".format(k.replace("_", " ").title(), v * 100)
            for k, v in W.items()
        )
        st.markdown(
            "<div style='background:#0a0a1a;border:1px solid #1e3a5f;border-radius:8px;"
            "padding:10px 16px;margin-bottom:12px;'>"
            "<span style='color:#475569;font-size:11px;font-weight:700;"
            "text-transform:uppercase;letter-spacing:0.05em;'>"
            "Active weights — {}: </span>"
            "<span style='color:#93c5fd;font-size:12px;'>{}</span></div>".format(
                sector_sel, weight_text
            ),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── REFERENCE GUIDE (fully populated)
# ══════════════════════════════════════════════════════════════════════════════
def render_reference_guide():
    st.markdown("## Column Reference Guide")
    st.caption(
        "Every metric explained with formula, real-world numeric example, "
        "sector benchmarks, and how it is used in scoring."
    )

    tab_val, tab_qual, tab_peg, tab_etraj, tab_mom, tab_rank, tab_disp = st.tabs([
        "📐 Valuation",
        "🏆 Quality",
        "📈 PEG",
        "🎯 Earn Trajectory",
        "⚡ Momentum",
        "🏅 Ranking & Score",
        "📋 Display-Only",
    ])

    # ── TAB 1: VALUATION ─────────────────────────────────────────────────────
    with tab_val:
        st.markdown("""
### P/E — Price to Earnings Ratio (Trailing)
**What it is:** How many dollars you pay per dollar of actual profit the company earned over the last 12 months.

**Formula:** `Current Stock Price / Trailing 12-Month EPS`

**Numeric Example (S&P 500 context):**

- Apple price = $210. Apple earned $6.57 per share TTM.
  - P/E = 210 / 6.57 = **32.0**
  - Meaning: you pay $32 for every $1 of Apple's annual profit.
- ExxonMobil price = $115. ExxonMobil earned $9.60 per share TTM.
  - P/E = 115 / 9.60 = **12.0** — energy sector commands a discount vs tech.

**Sector context matters:**

- P/E of 15 when sector median = 25 → cheap vs peers → high Valuation percentile → **boosts Score**
- P/E of 50 when sector median = 25 → expensive → low percentile → **hurts Score**

**Sector median P/E benchmarks (typical US S&P 500 ranges):**

| Sector | Typical Median P/E |
|---|---|
| Information Technology | 28–40 |
| Consumer Staples (FMCG) | 22–32 |
| Financials / Banking | 12–18 |
| Energy / Oil & Gas | 10–16 |
| Industrials | 22–30 |
| Health Care / Pharma | 22–35 |
| Materials / Metals | 14–20 |
| Utilities | 16–22 |
| Consumer Discretionary | 20–35 |
| Real Estate (REITs) | 30–50 |
| Communication Services | 18–28 |

**Used in scoring?** Yes — primary Valuation factor. Lower P/E = better percentile within sector.

---

### Fwd P/E — Forward Price to Earnings
**What it is:** Same as P/E but uses analysts' consensus EPS estimate for the **next 12 months**.

**Formula:** `Current Stock Price / Next 12-Month Estimated EPS`

**Numeric Example:**

- Microsoft price = $415, trailing EPS = $11.45, forward EPS estimate = $13.20
  - Trailing P/E = 36.2, Fwd P/E = 415 / 13.20 = **31.4**
  - Fwd P/E < Trailing P/E → earnings expected to grow ~15% → **positive signal**

- Declining earnings example: a retailer price = $50, trailing EPS = $4.00, forward EPS = $2.50
  - Trailing P/E = 12.5, Fwd P/E = 50 / 2.50 = **20.0**
  - Fwd P/E > Trailing P/E → earnings expected to fall 37.5% → **bad signal** even though trailing P/E looks cheap

**Used in scoring?** Yes — preferred over trailing P/E for the Valuation factor.

---

### MC% of S&P 500 — Market Cap as % of Index Total
**What it is:** What fraction of the entire S&P 500's combined market capitalisation this stock represents.

**Formula:** `Stock Market Cap / Sum of All S&P 500 Market Caps × 100`

**Numeric Examples (approximate 2024/25):**

- Apple (~$3.3T in a ~$45T index) → MC% ≈ **7.3%** — single largest weight
- Microsoft (~$3.1T) → MC% ≈ **6.9%**
- A mid-cap S&P 500 stock at $50B → MC% ≈ **0.11%**

**Why it matters:** S&P 500 is market-cap weighted. Stocks with high MC% dominate index returns.
If Apple falls 10%, the S&P 500 loses ~0.73% from Apple alone.

**Used in scoring?** No. Display and filter only.

---

### 52W Pos% — 52-Week Position
**What it is:** Where the current price sits as a percentage between the 52-week low and high.

**Formula:** `(Current Price − 52W Low) / (52W High − 52W Low) × 100`

**Numeric Example:**

- 52W Low = $140, 52W High = $220, Current Price = $175
  - 52W Pos% = (175 − 140) / (220 − 140) × 100 = 35 / 80 × 100 = **43.8%**
- 0% = at yearly low. 100% = at yearly high.

**Use cases:**
- Sort "52W Pos low to high" → find stocks near 52-week lows (potential value)
- Sort "52W Pos high to low" → find momentum leaders near 52-week highs

**Used in scoring?** No. Sort/filter only.
""")

    # ── TAB 2: QUALITY ────────────────────────────────────────────────────────
    with tab_qual:
        st.markdown("""
### Quality Score (0–100)
**What it is:** Composite score measuring fundamental business strength across three equally-weighted components.

**Formula:** `(ROIC sub-score + Interest Coverage sub-score + Op Margin sub-score) / 3`

**Numeric Example:**

- Stock A: ROIC = 20% → sub-score 77, Int Coverage = 8x → sub-score 80, Op Margin = 25% → sub-score 63
  - Quality Score = (77 + 80 + 63) / 3 = **73.3 / 100**

**Why this structure?**
- ROIC measures capital efficiency (are you generating returns above cost of capital?)
- Interest Coverage measures financial safety (can you service your debt?)
- Op Margin measures pricing power (how much profit per dollar of revenue?)

> **v9 note — Financials sector:** For banks, insurance, and diversified financials, the Quality Score
> uses **ROE** as the primary profitability metric instead of ROIC, and **Op Margin is excluded**
> from the composite. Bank ROA/ROIC is structurally 1–2% by design — not a weakness.
> Net interest margin is the correct metric for banks but is not surfaced by Yahoo Finance.

**Used in scoring?** Yes — 25% weight (sector-adaptive, see Ranking tab).

---

### ROIC% — Return on Invested Capital
**What it is:** For every dollar of capital deployed in the business, how many cents of profit does it generate?

> **Note on data source:** True ROIC = NOPAT / Invested Capital. This screener computes it from
> Yahoo financials: NOPAT = Operating Income × (1 − effective tax rate),
> Invested Capital = Equity + Debt − Cash. Where financials are unavailable, ROA (Net Income / Total
> Assets) is used as a fallback proxy. ROA understates true ROIC for capital-light businesses (IT,
> pharma) but is a reasonable approximation.

**Numeric Example:**

- Apple: NOPAT TTM ≈ $95B, Invested Capital ≈ $165B → ROIC ≈ **57.6%** — exceptional asset-light model
- General Motors: NOPAT ≈ $9B, Invested Capital ≈ $85B → ROIC ≈ **10.6%** — capital-intensive, acceptable

| ROIC | Assessment | Quality sub-score |
|---|---|---|
| 25%+ | Best-in-class | ~90–100 |
| 15% | Excellent | ~65 |
| 10% | Good | ~55 |
| 8% | Minimum threshold | ~49 |
| 5% | Borderline — flagged | ~38 |
| Below 0% | Losing money on assets | 0 |

---

### Int Coverage — Interest Coverage Ratio
**What it is:** How many times over can the company pay its annual interest expense from operating profit?

**Formula:** `EBIT / |Interest Expense|` (from annual income statement, TTM)

**Numeric Examples:**

- Microsoft: EBIT ≈ $100B, Interest Expense ≈ $1.5B → Coverage = **66.7x** — fortress balance sheet
- A leveraged industrial: EBIT = $500M, Interest = $450M → Coverage = **1.11x** — very risky
- S&P 500 median for non-financials: typically 8–15x

| Coverage | Assessment | Quality sub-score |
|---|---|---|
| 10x+ | Very safe | 100 |
| 5x | Comfortable | 50 |
| 3x | Minimum threshold | 30 |
| 1x | Barely covering interest | 10 |
| Below 1x | In distress | 0 |

> **Note for Banks and NBFCs:** Interest Coverage is **not meaningful** for financial companies
> whose core business is borrowing and lending. The screener excludes Int Coverage from the
> Quality Score for the Financials sector.

---

### Op Margin% — Operating Profit Margin
**What it is:** Of every $1 of revenue, how many cents become operating profit.

**Formula:** `Operating Income / Revenue × 100`

**Numeric Examples:**

- Microsoft: Revenue = $236B, Operating Income = $109B → **46.2%** — elite software margins
- Apple: Revenue = $383B, Op Income = $114B → **29.8%** — consumer hardware + services blend
- Ford: Revenue = $185B, Op Income = $7.2B → **3.9%** → flagged Margin<5%

| Op Margin | Quality sub-score |
|---|---|
| 40%+ | 100 (elite — software, pharma) |
| 25% | 62 |
| 15% | 37 |
| 5% | 12 (minimum threshold) |
| Below 5% | 0 + Margin<5% flag |

---

### Quality Flag
Pass/fail check shown next to Quality Score:

- **ROIC<8%** — capital return below minimum threshold
- **ROE<8%** — shown for Financials sector when ROE is the primary metric
- **IntCov<3x** — debt-servicing risk; below 3x interest coverage
- **Margin<5%** — thin profitability; operating margin below 5%
- **Pass** — all checks cleared
- **D/E: 1.5** — informational Debt/Equity appended to every row regardless of pass/fail

**Example full flag:** `ROIC<8%, Margin<5% | D/E:2.1`
Means: failed ROIC threshold (e.g. ROA = 5.2%) AND operating margin is thin (e.g. 3.8%), with D/E of 2.1 shown for context.
""")

    # ── TAB 3: PEG ───────────────────────────────────────────────────────────
    with tab_peg:
        st.markdown("""
### PEG — Price/Earnings-to-Growth Ratio
**What it is:** P/E adjusted for earnings growth rate. Answers the key question:
*"Is this stock cheap or expensive GIVEN how fast it is growing?"*

**Formula:** `P/E Ratio / Annual EPS Growth Rate (%)`

**Numeric Examples (S&P 500 stocks):**

| Stock | P/E | EPS Growth | PEG | Verdict |
|---|---|---|---|---|
| Nvidia | 35 | 40%/yr | 35/40 = **0.88** | Potentially undervalued for hypergrowth |
| JPMorgan Chase | 12 | 12%/yr | 12/12 = **1.00** | Perfectly fairly valued |
| Procter & Gamble | 26 | 4%/yr | PEG not computed | Growth below 5% floor |
| Tesla | 60 | 25%/yr | 60/25 = **2.40** | Expensive vs growth rate |
| Exxon Mobil | 12 | 1%/yr | PEG not computed | Growth below 5% floor |
| Alphabet | 22 | 20%/yr | 22/20 = **1.10** | Growth well-priced in |

**Interpreting PEG:**

| PEG | Signal |
|---|---|
| Below 1.0 | Potentially undervalued — paying less than 1× the growth rate |
| 1.0–2.0 | Fairly valued for its growth rate |
| 2.0–3.0 | Expensive — growth partially justifies premium |
| Above 3.0 | Very hard to justify valuation from growth alone |

**Growth guard — why PEG is only computed when EPS growth ≥ 5%:**

- ExxonMobil P/E = 12, EPS growth = 1% → PEG = 12. Mathematically large but not meaningful.
- Utility and commodity stocks with stable but minimal growth are fine investments —
  their PEG just isn't a useful signal.
- The 5% floor removes noise from near-zero growth stocks distorting the PEG ranking.

**Data source waterfall:**

1. Yahoo Finance `pegRatio` field — direct, most reliable, ~70% coverage
2. FMP `/ratios-ttm` → `priceEarningsGrowthRatioTTM` (if FMP key provided)
3. Calculated: (Fwd P/E or Trailing P/E) / EPS growth % — fallback when both sources are null

**Used in scoring?** Yes — weight varies by sector (20–25% for growth sectors, 5–10% for Utilities/REIT).
Lower PEG = better percentile within sector.
""")

    # ── TAB 4: EARN TRAJECTORY ────────────────────────────────────────────────
    with tab_etraj:
        st.markdown("""
### Earn Traj — Earnings Trajectory
**What it is:** Direction and magnitude of expected earnings change, derived from the gap between
Forward EPS and Trailing EPS.

**Formula:** `(Forward EPS − Trailing EPS) / |Trailing EPS|` — clipped to range [−1.0, +1.0]

The raw ratio is divided by 2 before clipping so that a 100% earnings jump = +1.0 rather than
requiring a 200% jump. This preserves nuance for moderate changes.

**Numeric Examples (S&P 500 stocks):**

| Situation | Trail EPS | Fwd EPS | Raw | Earn Traj | Signal |
|---|---|---|---|---|---|
| Microsoft growing | $11.45 | $13.20 | +0.15 | +0.08 | Mild positive — steady growth |
| Pharma guidance cut | $8.20 | $7.10 | −0.13 | −0.07 | Mild negative — earnings under pressure |
| Turnaround story | $1.50 | $4.00 | +1.67/2 | **+0.83** (clipped) | Strong recovery expected |
| Cyclical trough | $6.00 | $3.50 | −0.42/2 | **−0.21** | Significant earnings decline |
| Stable utility | $3.20 | $3.35 | +0.05 | +0.02 | Flat — predictable earnings |

**Reading the score:**

| Range | Signal |
|---|---|
| +0.5 to +1.0 | Strong earnings recovery or high analyst confidence in growth |
| +0.1 to +0.3 | Moderate growth — healthy company, earnings expected to expand |
| Near 0 | Flat earnings — stable but no catalyst |
| −0.1 to −0.3 | Earnings under pressure — guidance cuts, margin compression |
| −0.5 to −1.0 | Significant earnings deterioration expected — avoid or investigate |

**Practical filter uses:**

- Min Earn Traj = 0.10 → only stocks where analysts forecast ≥10% EPS growth
- Min Earn Traj = 0.25 → high-conviction earnings growth stories
- Min Earn Traj = −0.05 → exclude stocks with flat or declining earnings

**Data source:** Yahoo Finance `forwardEps` and `trailingEps` from `.info`

**Used in scoring?** Yes — 15–22% weight (higher for Real Estate where earnings trajectory is key).
Higher Earn Traj = better percentile within sector.
""")

    # ── TAB 5: MOMENTUM ───────────────────────────────────────────────────────
    with tab_mom:
        st.markdown("""
### Momentum Score — Skip-Month Volatility-Adjusted Momentum
**What it is:** Medium-term price trend adjusted for how noisy/volatile the stock is.
A clean, sustained move scores higher than a volatile spike of the same raw magnitude.

**Formula:** `(6-month return − 1-month return) / Trailing 90-day Annualised Volatility`

**Why subtract the 1-month return (skip-month technique)?**
The most recent month exhibits a documented short-term reversal effect — stocks that surged last
month tend to mean-revert slightly in the near term. Removing it isolates the durable 2–6 month
trend (based on Fama-French momentum factor construction).

**Numeric Example (S&P 500 stocks):**

**JPMorgan Chase (steady, low volatility):**
- 6mo return = +18%, 1mo return = +3%, trailing vol = 16%
- Skip-month raw = 18 − 3 = 15%
- Momentum Score = 15 / 16 = **0.94** — strong signal per unit of risk

**Tesla (high volatility):**
- 6mo return = +40%, 1mo return = +10%, trailing vol = 55%
- Skip-month raw = 40 − 10 = 30%
- Momentum Score = 30 / 55 = **0.55** — lower score despite higher raw return

JPMorgan's +18% in a 16% vol stock is far more signal-rich than Tesla's +40% in a 55% vol stock.

**Interpreting the score:**

| Score | Signal |
|---|---|
| Above +1.0 | Exceptionally strong momentum |
| +0.3 to +1.0 | Healthy uptrend |
| −0.3 to +0.3 | Neutral — no clear trend |
| Below −0.3 | Downtrend — negative momentum |

**Used in scoring?** Yes — 12–25% weight (higher for Energy and Materials where price momentum
is a key signal for commodity cycles).

---

### Ret 1Mo%, Ret 3Mo%, Ret 6Mo%
Raw percentage price returns over 1, 3, 6 months calculated from monthly closing prices.

**Example:** Price 6 months ago = $100, today = $122 → Ret 6Mo% = **+22.0%**

Display only — these feed into Momentum Score but are not directly used for ranking.

---

### Trailing Vol%
Annualised standard deviation of daily price returns over last 90 calendar days.

**Formula:** `Daily Return Std Dev × √252 × 100`

**Numeric Examples:**
- JPMorgan: daily std dev = 1.0% → annualised = 1.0% × 15.87 = **15.9%** — stable large-cap bank
- Tesla: daily std dev = 3.5% → annualised = 3.5% × 15.87 = **55.5%** — volatile growth stock
- S&P 500 index: typically 12–18% annualised volatility in calm markets

Display only — it is the denominator in Momentum Score.
""")

    # ── TAB 6: RANKING & SCORE ────────────────────────────────────────────────
    with tab_rank:
        st.markdown("""
### Score (0–100)
**What it is:** Final weighted composite percentile score, computed within each GICS sector independently
using **sector-adaptive weights** (v9).

**Full formula:** Score = W_val × Valuation_pct + W_qual × Quality_pct + W_peg × PEG_pct
+ W_etraj × EarnTraj_pct + W_mom × Momentum_pct
Then multiplied by Missing Factor Penalty (see below).

**Numeric Example — Information Technology sector (weights: Val 20%, Qual 25%, PEG 25%, Earn 15%, Mom 15%):**

| Factor | Value | Percentile within IT | Weight | Contribution |
|---|---|---|---|---|
| Fwd P/E | 28.5 | 62nd (lower = better) | 20% | 12.4 |
| Quality Score | 74/100 | 68th | 25% | 17.0 |
| PEG | 1.35 | 70th | 25% | 17.5 |
| Earn Traj | +0.12 | 58th | 15% | 8.7 |
| Momentum | 0.88 | 74th | 15% | 11.1 |
| **Raw Score** | | | | **66.7** |
| Missing factors | 0 | Penalty = ×1.0 | | |
| **Final Score** | | | | **66.7** |

> **Key point:** Score of 67 in IT and Score of 67 in Utilities both mean "67th percentile in their
> sector." They are **NOT directly comparable** — sector rankings are independent.

---

### Sector-Adaptive Weights

| Sector | Val | Quality | PEG | Earn | Mom |
|---|---|---|---|---|---|
| Information Technology | 20% | 25% | **25%** | 15% | 15% |
| Consumer Discretionary | 20% | 20% | 22% | 18% | **20%** |
| Communication Services | 22% | 23% | 22% | 18% | 15% |
| Health Care | 25% | **30%** | 18% | 15% | 12% |
| Industrials | 25% | **28%** | 18% | 17% | 12% |
| Consumer Staples | 28% | **32%** | 10% | 15% | 15% |
| Financials | **30%** | 25% | 18% | 17% | 10% |
| Energy | **30%** | 18% | 12% | 15% | **25%** |
| Materials | 28% | 20% | 12% | 15% | **25%** |
| Real Estate | **30%** | 18% | 10% | **22%** | 20% |
| Utilities | **38%** | 27% | 5% | 15% | 15% |

Select a sector in the Screener to see the active weights bar in the KPI panel.

---

### Missing Factor Penalty

| Missing factors | Multiplier | Effect | Why |
|---|---|---|---|
| 0 or 1 | ×1.00 | None | High data confidence |
| 2 | ×0.85 | −15% | Moderate uncertainty |
| 3 or more | ×0.70 | −30% | Low confidence |

A PSU stock with only P/E available (PEG = None, Quality = None, Earn Traj = None, Momentum = None)
has 4 missing factors → Score × 0.70. This prevents a stock with a single very-low P/E from
falsely ranking #1 when we have no other data to validate it.

---

### Rank
Ordinal position within sector by Score. **Rank 1 = best-scoring stock in that sector.**

- Rank 2 in Information Technology = 2nd highest Score among all IT stocks
- Rank 2 in Utilities is a completely independent ranking with no relation to IT Rank 2
- Stocks with insufficient data (all factors missing) receive no Rank

---

### Conviction Score (0–100)
Score further adjusted for: (1) how complete the data is, and (2) whether the stock's sector
trades at a premium or discount to the overall index median P/E.

**Formula:** `Score × data_completeness_ratio × sector_discount_factor` → normalised 0–100

**Data completeness example:**
- Stock has P/E, Quality, Earn Traj but missing PEG and Momentum → completeness = 3/5 = 0.60
- Raw Score 72 × 0.60 = 43.2 → lower Conviction despite decent Score

**Sector discount factor:**
- Index median P/E = 22. Consumer Staples median = 28 (premium sector).
  - Discount = 22/28 = 0.79 → clipped to 0.79 → Staples scores gently penalised
- Index median P/E = 22. Energy median = 12 (discount sector).
  - Discount = 22/12 = 1.83 → clipped to **1.30** → Energy stocks get up to 30% boost

**Practical use:** Sort by Conviction Score to find stocks that combine strong fundamentals,
good data availability, AND reasonable sector valuation all at once.
""")

    # ── TAB 7: DISPLAY-ONLY ───────────────────────────────────────────────────
    with tab_disp:
        st.markdown("""
### ROE% — Return on Equity (Display Only)
**Formula:** `Net Income / Shareholders Equity × 100`

**Numeric Examples:**

- Apple: Net Income = $97B, Equity = $57B → ROE = **170%** — extreme due to buybacks reducing equity base
- JPMorgan: Net Income = $49B, Equity = $327B → ROE = **15.0%** — healthy for a major bank
- ExxonMobil: Net Income = $36B, Equity = $165B → ROE = **21.8%** — capital-heavy but efficient

**Why display only (not scored)?** ROE is distorted by leverage and share buybacks.
A company that borrows heavily or buys back all its shares can show an astronomically high ROE
without being a better business. ROIC adjusts for this. ROE is shown as a reference point.

---

### Debt/Eq — Debt to Equity Ratio (Display Only)
**Formula:** `Total Debt / Total Shareholders Equity`

**Numeric Examples:**

- Apple: D/E ≈ 1.80 — moderate; offset by massive cash position
- Microsoft: D/E ≈ 0.35 — conservative balance sheet
- AT&T: D/E ≈ 1.10 — legacy telecom infrastructure debt
- A leveraged REIT: D/E = 3.5 — project-finance model, structurally high

**Why display only?** D/E tells you the capital structure but not safety. D/E 3.5 with Int Coverage
10x (long-term contracted cash flows) is safer than D/E 0.8 with Int Coverage 1.3x (struggling
retailer). Int Coverage is the scoring metric; D/E is context.

---

### Rev Q1–Q4 ($B)
Last four fiscal quarters of total revenue, newest first.

**Unit:** USD billions

**Examples:**

- Apple Q1 (most recent) = $94.9B
- An accelerating pattern (positive): Q4 = $14B → Q3 = $15.8B → Q2 = $17.1B → Q1 = $19.3B
- A decelerating pattern (negative): Q4 = $22B → Q3 = $20.5B → Q2 = $19.1B → Q1 = $18.0B

---

### Rev Growth% (CAGR)
**Formula:** `(Newest quarter revenue / Revenue 4 quarters ago)^(1/3) − 1 × 100`

Compares the most recent quarter to the same quarter one year prior — removes seasonal effects.

**Numeric Example:**

- Nvidia: Q (current) = $26B, Q (1 year ago) = $13.5B → YoY growth = **+92.6%**
- A commodity company: Q = $12B, Q (year ago) = $16B → **−25.0%** — revenue declined

Display only — not used in PEG or scoring (revenue growth ≠ earnings growth).

---

### PEG Method
Indicates how PEG was derived for this row:

| Value | Meaning |
|---|---|
| `Yahoo` | Directly from Yahoo Finance `pegRatio` field — most reliable |
| `FMP-ratios` | From FMP `/ratios-ttm` endpoint (if FMP key configured) |
| `Yahoo EPS growth` | Calculated as Fwd P/E (or P/E) ÷ Yahoo `earningsGrowth` |
| `—` | PEG not available (growth < 5% or insufficient data) |

---

### Data Sources
Pipe-separated log of which source provided each data point for the row.

Example: `PE:Yahoo | PEG:FMP-ratios | ROIC:Yahoo | IC:Yahoo | ET:Yahoo`

---

### Data Coverage
Yahoo Finance is the primary source. Coverage varies across the S&P 500:

| Metric | Typical Coverage |
|---|---|
| Price, MC, 52W | ~99% of tickers |
| Trailing P/E | ~90% |
| Forward P/E | ~78% |
| PEG Ratio | ~70% |
| ROE, Op Margin | ~88% |
| Int Coverage | ~65% (computed from financials) |
| ROIC (computed) | ~60% (requires full quarterly financials) |
| Earn Traj | ~82% |
| Momentum | ~97% (requires only price history) |
| Quarterly Revenue | ~72% |

Stocks with **fewer than 3 factors** populated receive a −30% Score penalty (Missing Factor Penalty).
""")

    st.markdown("---")
    st.markdown(
        "**Data sources v10:** Yahoo Finance (primary) · FMP bonus if key available · "
        "ROIC computed from Yahoo quarterly financials · Earn Traj from Yahoo FwdEPS/TrailEPS · "
        "MC% computed across all S&P 500 constituents · Sector-adaptive 5-factor scoring. "
        "_Nothing here is financial advice — all metrics are educational references._"
    )


# ══════════════════════════════════════════════════════════════════════════════
# ── APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="S&P 500 Screener v10", layout="wide", page_icon="📊")
st.markdown(
    "<style>"
    "div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}"
    "</style>",
    unsafe_allow_html=True,
)

st.markdown("## S&P 500 Fundamental Screener v10")
st.caption(
    "Yahoo-first · ROIC computed from financials · Earn Traj from FwdEPS/TrailEPS · "
    "MC% of S&P500 · Sector-Adaptive 5-factor scoring · FMP bonus layer if key available"
)

page_screener, page_reference = st.tabs(["📊 Screener", "📖 Column Reference Guide"])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════
with page_screener:

    # ── Refresh row ──────────────────────────────────────────────────────────
    col_r, col_t = st.columns([1, 6])
    with col_r:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
    with col_t:
        st.caption("Last loaded: {} · Prices: 1hr cache · Fundamentals: 24hr cache".format(
            datetime.now().strftime("%I:%M %p")))

    # ── FMP key status ────────────────────────────────────────────────────────
    fmp_key = get_fmp_key()
    if fmp_key:
        st.success("FMP API key found — will use as bonus layer for PE override and ROIC/IntCov.")
    else:
        st.info("No FMP key. Running on Yahoo Finance only. Add [fmp] api_key to Streamlit Secrets for FMP overrides.")

    # ── Load data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading S&P 500 universe..."):
        sp500 = fetch_sp500_constituents()
    if sp500.empty:
        st.error("Failed to load S&P 500 universe.")
        st.stop()

    universe_df = sp500.copy().reset_index(drop=True)
    tickers     = tuple(universe_df["Ticker"].tolist())

    with st.spinner("Fetching prices ({} tickers)...".format(len(tickers))):
        prices = fetch_prices_batch(tickers)

    with st.spinner("Fetching momentum (skip-month vol-adjusted)..."):
        momentum = fetch_momentum_batch(tickers)

    with st.spinner("Fetching Yahoo fundamentals for all {} tickers...".format(len(tickers))):
        yahoo_fundamentals = fetch_yahoo_fundamentals_all(tickers)

    fmp_quotes = {}
    fmp_ratios = {}
    if fmp_key:
        with st.spinner("FMP bonus: bulk /quote..."):
            fmp_quotes = fetch_fmp_quotes_if_available(tickers, fmp_key)
        with st.spinner("FMP bonus: /ratios-ttm..."):
            fmp_ratios = fetch_fmp_ratios_if_available(tickers, fmp_key)

    with st.spinner("Merging data sources..."):
        merged_map = merge_all_sources(yahoo_fundamentals, fmp_quotes, fmp_ratios, tickers)

    with st.spinner("Fetching quarterly revenue..."):
        rev_map = fetch_last4_revenue_parallel(tickers)

    # ── Coverage banner ───────────────────────────────────────────────────────
    total_t  = len(tickers)
    has_pe   = sum(1 for t in tickers if merged_map.get(t, {}).get("pe")           is not None)
    has_fwd  = sum(1 for t in tickers if merged_map.get(t, {}).get("fwd_pe")       is not None)
    has_peg  = sum(1 for t in tickers if merged_map.get(t, {}).get("peg")          is not None)
    has_roe  = sum(1 for t in tickers if merged_map.get(t, {}).get("roe")          is not None)
    has_roic = sum(1 for t in tickers if merged_map.get(t, {}).get("roic")         is not None)
    has_ic   = sum(1 for t in tickers if merged_map.get(t, {}).get("int_coverage") is not None)
    has_om   = sum(1 for t in tickers if merged_map.get(t, {}).get("op_margin")    is not None)
    has_et   = sum(1 for t in tickers if merged_map.get(t, {}).get("earn_traj")    is not None)

    st.info(
        "Data coverage — "
        "P/E: {}/{} ({:.0f}%) · Fwd P/E: {}/{} ({:.0f}%) · PEG: {}/{} ({:.0f}%) · "
        "ROIC: {}/{} ({:.0f}%) · ROE: {}/{} ({:.0f}%) · Int Coverage: {}/{} ({:.0f}%) · "
        "Op Margin: {}/{} ({:.0f}%) · Earn Traj: {}/{} ({:.0f}%) · Primary: Yahoo Finance{}".format(
            has_pe,   total_t, has_pe   / total_t * 100,
            has_fwd,  total_t, has_fwd  / total_t * 100,
            has_peg,  total_t, has_peg  / total_t * 100,
            has_roic, total_t, has_roic / total_t * 100,
            has_roe,  total_t, has_roe  / total_t * 100,
            has_ic,   total_t, has_ic   / total_t * 100,
            has_om,   total_t, has_om   / total_t * 100,
            has_et,   total_t, has_et   / total_t * 100,
            " + FMP bonus" if fmp_key else "",
        )
    )

    # ── Build table ───────────────────────────────────────────────────────────
    scr = build_screener_table(universe_df, prices, merged_map, rev_map, momentum)

    # ══════════════════════════════════════════════════════════════════════════
    # v10 FILTERS — 5 controls in ONE row
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("### Filters")

    all_sectors = sorted(scr["Sector"].dropna().unique().tolist())

    f1, f2, f3, f4, f5 = st.columns(5)

    sector_sel = f1.selectbox(
        "Sector",
        ["All Sectors"] + all_sectors,
        help="Filter to one GICS sector or view all 503 S&P 500 stocks.",
    )

    sort_by = f2.selectbox(
        "Sort by",
        [
            "Sector then Rank",
            "Score high to low",
            "Conviction high to low",
            "MC% of S&P500 high to low",
            "Price low to high",
            "Price high to low",
            "Mkt Cap high to low",
            "PE low to high",
            "Fwd PE low to high",
            "PEG low to high",
            "Quality Score high",
            "ROIC high to low",
            "ROE high to low",
            "Earn Traj high to low",
            "Rev Growth high to low",
            "Momentum Score high",
            "52W Pos low to high",
        ],
        help="Choose the primary sort column for the results table.",
    )

    mc_min_b = f3.number_input(
        "Min Mkt Cap ($B)",
        value=0,
        step=10,
        min_value=0,
        help="Only show stocks with market cap above this threshold (in USD billions).",
    )

    pe_max = f4.number_input(
        "Max P/E",
        value=9999,
        step=50,
        min_value=0,
        help="Exclude stocks with trailing P/E above this value. Stocks with no P/E data are always shown.",
    )

    qual_min_f = f5.number_input(
        "Min Quality Score",
        value=0.0,
        step=5.0,
        min_value=0.0,
        max_value=100.0,
        help="Only show stocks with Quality Score at or above this value (0–100).",
    )

    # ── KPI panel ─────────────────────────────────────────────────────────────
    render_sector_kpi_panel(scr, sector_sel)

    # ── Apply filters ─────────────────────────────────────────────────────────
    filt = scr.copy()
    if sector_sel != "All Sectors":
        filt = filt[filt["Sector"] == sector_sel]
    filt = filt[(filt["Mkt Cap"].isna())       | (filt["Mkt Cap"]       >= mc_min_b * 1e9)]
    filt = filt[(filt["P/E"].isna())           | (filt["P/E"]           <= pe_max)]
    filt = filt[(filt["Quality Score"].isna()) | (filt["Quality Score"] >= qual_min_f)]

    # ── Sort ──────────────────────────────────────────────────────────────────
    sort_map = {
        "Sector then Rank":          (["Sector", "Rank"],     [True, True]),
        "Score high to low":         (["Score"],              [False]),
        "Conviction high to low":    (["Conviction Score"],   [False]),
        "MC% of S&P500 high to low": (["MC% of S&P500"],     [False]),
        "Price low to high":         (["Price"],              [True]),
        "Price high to low":         (["Price"],              [False]),
        "Mkt Cap high to low":       (["Mkt Cap"],            [False]),
        "PE low to high":            (["P/E"],                [True]),
        "Fwd PE low to high":        (["Fwd P/E"],            [True]),
        "PEG low to high":           (["PEG"],                [True]),
        "Quality Score high":        (["Quality Score"],      [False]),
        "ROIC high to low":          (["ROIC%"],              [False]),
        "ROE high to low":           (["ROE%"],               [False]),
        "Earn Traj high to low":     (["Earn Traj"],          [False]),
        "Rev Growth high to low":    (["Rev Growth% (CAGR)"], [False]),
        "Momentum Score high":       (["Momentum Score"],     [False]),
        "52W Pos low to high":       (["52W Pos%"],           [True]),
    }
    sc, sa = sort_map.get(sort_by, (["Sector", "Rank"], [True, True]))
    filt   = filt.sort_values(sc, ascending=sa, na_position="last")

    st.caption("Showing **{}** of **{}** stocks · Sector: {} · Sort: {}".format(
        len(filt), len(scr), sector_sel, sort_by))

    # ── Display ───────────────────────────────────────────────────────────────
    disp = filt.copy()
    disp["Price ($)"]    = disp["Price"].round(2)
    disp["Mkt Cap ($B)"] = (disp["Mkt Cap"] / 1e9).round(2)
    disp["MC% of S&P500"] = disp["MC% of S&P500"].round(4)
    disp["Rev Q1 ($B)"]  = (disp["Rev Q1"] / 1e9).round(2)
    disp["Rev Q2 ($B)"]  = (disp["Rev Q2"] / 1e9).round(2)
    disp["Rev Q3 ($B)"]  = (disp["Rev Q3"] / 1e9).round(2)
    disp["Rev Q4 ($B)"]  = (disp["Rev Q4"] / 1e9).round(2)

    disp["Quality Flag"] = disp.apply(
        lambda r: quality_flag(
            r.get("ROIC%"), r.get("ROE%"),
            r.get("Int Coverage"), r.get("Op Margin%"), r.get("Debt/Eq"),
            sector=r.get("Sector"),
        ),
        axis=1,
    )

    for c in ["P/E", "Fwd P/E", "PEG", "Earn Traj", "52W Pos%",
              "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
              "Quality Score", "Momentum Score", "Ret 1Mo%", "Ret 3Mo%",
              "Ret 6Mo%", "Trailing Vol%", "Score", "Conviction Score",
              "Rev Growth% (CAGR)"]:
        if c in disp.columns:
            disp[c] = disp[c].round(2)

    disp["Rank"] = disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

    COLS = [
        "Ticker", "Sector", "Price ($)", "Mkt Cap ($B)", "MC% of S&P500",
        "P/E", "Fwd P/E", "PEG", "PEG Method",
        "Earn Traj",
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
        label="⬇️ Download CSV",
        data=disp_final.to_csv(index=False).encode("utf-8"),
        file_name="sp500_screener_v10_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M")),
        mime="text/csv",
    )

    # ── Metric legend under table ─────────────────────────────────────────────
    st.markdown("""
**PEG:** Price/Earnings-to-Growth. PEG < 1.0 = potentially undervalued for its growth rate.
Only computed when EPS growth ≥ 5% to avoid meaningless ratios for slow-growth stocks.

**Earn Traj:** (Forward EPS − Trailing EPS) / |Trailing EPS|, clipped to [−1.0, +1.0].
Positive = analysts expect earnings growth. +0.20 = analysts forecast ~20% EPS improvement ahead.

**MC% of S&P500:** This stock's market cap as a % of total S&P 500 market cap.
Apple at ~7% means a 10% Apple drop drags the index ~0.7% on its own.

**Score:** Sector-adaptive weights — select a sector to see the active weights bar in the KPI panel above.
""")

    st.markdown("---")
    st.caption(
        "Sources v10: Yahoo Finance primary · FMP bonus if key available · "
        "ROIC from Yahoo quarterly financials · Earn Traj from Yahoo FwdEPS/TrailEPS · "
        "MC% computed across all S&P 500 constituents · Sector-adaptive scoring"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COLUMN REFERENCE GUIDE
# ══════════════════════════════════════════════════════════════════════════════
with page_reference:
    render_reference_guide()
