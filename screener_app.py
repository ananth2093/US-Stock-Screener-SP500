# screener_app.py  v8
# ─────────────────────────────────────────────────────────────────────────────
# v8 CHANGES from v7:
#   1. ROIC% computed from Yahoo quarterly_financials + balance_sheet
#      Formula: NOPAT / Invested Capital (TTM)
#      NOPAT = Operating Income × (1 - effective tax rate)
#      Invested Capital = Total Equity + Total Debt - Cash
#
#   2. Earn Revision → renamed "Earn Trajectory" (Earn Traj)
#      Proxy: (forwardEps - trailingEps) / abs(trailingEps)
#      Captures whether analysts expect earnings to grow or shrink
#      Available from Yahoo .info for ~85% of S&P 500
#
#   3. Weights redistributed — no dead-weight factors:
#      Valuation: 25% → 25% (unchanged)
#      Quality:   25% → 25% (unchanged)
#      PEG:       20% → 20% (unchanged)
#      Earn Traj: 15% → 15% (now POPULATED from Yahoo)
#      Momentum:  15% → 15% (unchanged)
#
#   4. ROIC column now populated for display AND used in Quality Score
#      (ROE remains as fallback if ROIC computation fails)
#
# ARCHITECTURE (same as v7):
#   LAYER 1 (PRIMARY): Yahoo Finance
#   LAYER 2 (BONUS): FMP /quote bulk — if API key exists
#   LAYER 3 (BONUS): FMP /ratios-ttm — if key exists AND tier supports it
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
    "earn_traj":     0.15,
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
# v8: Now also computes ROIC from financials + balance_sheet
#     and Earn Trajectory from forwardEps vs trailingEps
# ══════════════════════════════════════════════════════════════════════════════
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

        # ── .info: PE, FwdPE, PEG, quality metrics, Earn Traj ────────────
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

        # ── Earn Trajectory (v8 NEW) ─────────────────────────────────────
        # Proxy for earnings revision: (forwardEps - trailingEps) / |trailingEps|
        # Positive = analysts expect growth, Negative = expect decline
        # Clipped to [-1.0, +1.0] range
        fwd_eps_val = sf(info.get("forwardEps"))
        trail_eps_val = sf(info.get("trailingEps"))
        if fwd_eps_val is not None and trail_eps_val is not None and abs(trail_eps_val) > 0.01:
            earn_traj_raw = (fwd_eps_val - trail_eps_val) / abs(trail_eps_val)
            result["earn_traj"] = max(-1.0, min(1.0, earn_traj_raw))

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
        try:
            qfin = obj.quarterly_financials
            if qfin is not None and not qfin.empty:
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

        # ── ROIC from financials + balance_sheet (v8 NEW) ─────────────────
        # ROIC = NOPAT / Invested Capital
        # NOPAT = Operating Income (TTM) × (1 - effective tax rate)
        # Invested Capital = Total Stockholders Equity + Total Debt - Cash
        try:
            qfin = obj.quarterly_financials
            bs   = obj.quarterly_balance_sheet

            if qfin is not None and not qfin.empty and bs is not None and not bs.empty:
                # Get Operating Income TTM (sum of last 4 quarters)
                op_inc_row = None
                for nm in ["Operating Income", "EBIT", "Ebit"]:
                    if nm in qfin.index:
                        op_inc_row = nm
                        break

                # Get Tax Provision and Pre-Tax Income for effective tax rate
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
                    op_inc_ttm = float(qfin.loc[op_inc_row].dropna().head(4).sum())

                    # Compute effective tax rate
                    eff_tax_rate = 0.21  # default to US corporate rate
                    if tax_row and pretax_row:
                        tax_ttm    = float(qfin.loc[tax_row].dropna().head(4).sum())
                        pretax_ttm = float(qfin.loc[pretax_row].dropna().head(4).sum())
                        if pretax_ttm > 0 and tax_ttm >= 0:
                            computed_rate = tax_ttm / pretax_ttm
                            if 0 < computed_rate < 0.6:  # sanity check
                                eff_tax_rate = computed_rate

                    nopat = op_inc_ttm * (1 - eff_tax_rate)

                    # Get most recent balance sheet data
                    # Total Stockholders Equity
                    equity_val = None
                    for nm in ["Total Stockholders Equity", "Stockholders Equity",
                               "Common Stock Equity", "Total Equity Gross Minority Interest"]:
                        if nm in bs.index:
                            eq_series = bs.loc[nm].dropna()
                            if len(eq_series) > 0:
                                equity_val = float(eq_series.iloc[0])
                                break

                    # Total Debt
                    debt_val = None
                    for nm in ["Total Debt", "Net Debt", "Long Term Debt",
                               "Long Term Debt And Capital Lease Obligation"]:
                        if nm in bs.index:
                            d_series = bs.loc[nm].dropna()
                            if len(d_series) > 0:
                                debt_val = float(d_series.iloc[0])
                                break

                    # Cash
                    cash_val = None
                    for nm in ["Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments",
                               "Cash Financial", "Cash And Short Term Investments"]:
                        if nm in bs.index:
                            c_series = bs.loc[nm].dropna()
                            if len(c_series) > 0:
                                cash_val = float(c_series.iloc[0])
                                break

                    # Compute Invested Capital
                    if equity_val is not None and debt_val is not None:
                        cash_use = cash_val if cash_val is not None else 0
                        invested_capital = equity_val + debt_val - cash_use

                        if invested_capital > 0 and nopat != 0:
                            roic_computed = (nopat / invested_capital) * 100.0
                            # Sanity check: ROIC between -100% and +200%
                            if -100 < roic_computed < 200:
                                result["roic"] = roic_computed
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
            st.caption("FMP /ratios-ttm: not available on your tier. Using Yahoo-computed ROIC and quality metrics.")
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

        # ROIC: FMP-ratios > Yahoo-computed (v8: Yahoo now computes ROIC!)
        roic    = first(fr.get("roic"),    yb.get("roic"))

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

        # Earn Trajectory (v8 NEW): from Yahoo forwardEps vs trailingEps
        earn_traj = yb.get("earn_traj")

        # MC, 52W: FMP-quote > Yahoo
        mc      = first(fq.get("mc"),  yb.get("mc"))
        hi52    = first(fq.get("hi52"), yb.get("hi52"))
        lo52    = first(fq.get("lo52"), yb.get("lo52"))

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
            "earn_traj":      earn_traj,
            "mc":             mc,
            "hi52":           hi52,
            "lo52":           lo52,
        }
    return merged

# ── Quality Score ─────────────────────────────────────────────────────────────
def compute_quality_score(roic, roe, int_coverage, op_margin):
    """
    v8: ROIC now populated from Yahoo financials.
    If ROIC missing, use ROE as proxy (both measure capital efficiency).
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
    KEY_FACTORS = ["P/E", "Fwd P/E", "PEG", "Quality Score", "Momentum Score", "Earn Traj"]
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
        elig["_s_etraj"]= percentile_score(elig["Earn Traj"],  ascending=False)

        qs    = elig["Quality Score"]
        q_min = qs.min(); q_max = qs.max()
        if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min:
            elig["_s_quality"] = (qs - q_min) / (q_max - q_min) * 100.0
        else:
            elig["_s_quality"] = qs.fillna(0.0)
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)

        raw = (W["valuation"]  * elig["_s_val"]     +
               W["quality"]    * elig["_s_quality"] +
               W["peg"]        * elig["_s_peg"]     +
               W["earn_traj"]  * elig["_s_etraj"]   +
               W["momentum"]   * elig["_s_mom"])

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
        earn_traj = to_num(fi.get("earn_traj"))

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

    num_cols = ["Price", "Mkt Cap", "P/E", "Fwd P/E", "PEG", "52W Pos%",
                "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
                "Quality Score", "Earn Traj", "Momentum Score",
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
        flags.append("ROIC<8%" if (roic is not None and not pd.isna(roic)) else "ROE<8%")
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
            "margin-bottom:12px;'>"
            "<div style='color:#aaa;font-size:11px;margin-bottom:8px;'>Top Ranked in Sector</div>"
            "<div>{}</div>"
            "<div style='color:#555;font-size:10px;margin-top:8px;'>"
            "Score = Valuation 25% + Quality 25% + PEG 20% + Earn Traj 15% + Momentum 15%"
            "</div></div>".format(badges or "<span style='color:#555;'>No ranked stocks</span>"),
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── APP ───────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="S&P 500 Screener v8", layout="wide", page_icon="📊")
st.markdown(
    "<style>div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}</style>",
    unsafe_allow_html=True,
)

st.markdown("## S&P 500 Fundamental Screener v8")
st.caption(
    "Yahoo-first architecture · ROIC computed from financials · "
    "Earn Trajectory from FwdEPS vs TrailEPS · "
    "Full 5-factor model with all factors populated · "
    "FMP bonus layer if key available"
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
    st.success("FMP API key found — will use as bonus layer for PE override and ROIC/IntCov if tier supports /ratios-ttm.")
else:
    st.info("No FMP key configured. Running on Yahoo Finance only (ROIC computed from financials, Earn Traj from EPS estimates). Add [fmp] api_key to Streamlit Secrets for additional overrides.")

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

# PRIMARY: Yahoo fundamentals for ALL tickers (now includes ROIC + Earn Traj)
with st.spinner("Fetching Yahoo fundamentals (PE, PEG, ROE, ROIC, OpMargin, D/E, IntCoverage, EarnTraj) for all {} tickers...".format(len(tickers))):
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
has_roic = sum(1 for t in tickers if merged_map.get(t, {}).get("roic")         is not None)
has_ic   = sum(1 for t in tickers if merged_map.get(t, {}).get("int_coverage") is not None)
has_om   = sum(1 for t in tickers if merged_map.get(t, {}).get("op_margin")    is not None)
has_et   = sum(1 for t in tickers if merged_map.get(t, {}).get("earn_traj")    is not None)

st.info(
    "Data coverage — "
    "P/E: {}/{} ({:.0f}%) · "
    "Fwd P/E: {}/{} ({:.0f}%) · "
    "PEG: {}/{} ({:.0f}%) · "
    "ROIC: {}/{} ({:.0f}%) · "
    "ROE: {}/{} ({:.0f}%) · "
    "Int Coverage: {}/{} ({:.0f}%) · "
    "Op Margin: {}/{} ({:.0f}%) · "
    "Earn Traj: {}/{} ({:.0f}%) · "
    "Primary: Yahoo Finance{}".format(
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
        "Quality Score high", "ROIC high to low", "ROE high to low",
        "Earn Traj high to low",
        "Rev Growth high to low", "Momentum Score high", "52W Pos low to high",
    ])
    pe_max   = fc3.number_input("Max PE",              value=9999,  step=50)
    peg_max  = fc4.number_input("Max PEG",             value=999.0, step=1.0)
    mc_min_b = fc5.number_input("Min Market Cap ($B)", value=0,     step=5)

with st.expander("Quality Filters", expanded=False):
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    roic_min_f = qc1.number_input("Min ROIC (%)",             value=0.0,  step=5.0)
    ic_min_f   = qc2.number_input("Min Int Coverage (x)",     value=0.0,  step=1.0)
    om_min_f   = qc3.number_input("Min Op Margin (%)",        value=0.0,  step=5.0)
    qual_min_f = qc4.number_input("Min Quality Score",        value=0.0,  step=5.0)
    de_max_f   = qc5.number_input("Max Debt/Equity (ref)",    value=99.0, step=0.5)

with st.expander("Momentum & Earnings", expanded=False):
    mc1, mc2, mc3 = st.columns(3)
    mom_min   = mc1.number_input("Min Momentum Score",        value=-999.0, step=5.0)
    et_min    = mc2.number_input("Min Earn Traj",             value=-1.0,   step=0.1)
    hide_nope = mc3.checkbox("Hide stocks missing both P/E and Fwd P/E", value=False)

render_sector_kpi_panel(scr, sector_sel)

filt = scr.copy()
if sector_sel != "All Sectors":
    filt = filt[filt["Sector"] == sector_sel]
filt = filt[(filt["Mkt Cap"].isna())        | (filt["Mkt Cap"]       >= mc_min_b * 1e9)]
filt = filt[(filt["P/E"].isna())            | (filt["P/E"]           <= pe_max)]
filt = filt[(filt["PEG"].isna())            | (filt["PEG"]           <= peg_max)]
filt = filt[(filt["ROIC%"].isna())          | (filt["ROIC%"]         >= roic_min_f)]
filt = filt[(filt["Int Coverage"].isna())   | (filt["Int Coverage"]  >= ic_min_f)]
filt = filt[(filt["Op Margin%"].isna())     | (filt["Op Margin%"]    >= om_min_f)]
filt = filt[(filt["Quality Score"].isna())  | (filt["Quality Score"] >= qual_min_f)]
filt = filt[(filt["Debt/Eq"].isna())        | (filt["Debt/Eq"]       <= de_max_f)]
filt = filt[(filt["Momentum Score"].isna()) | (filt["Momentum Score"]>= mom_min)]
filt = filt[(filt["Earn Traj"].isna())      | (filt["Earn Traj"]     >= et_min)]
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
    "ROIC high to low":       (["ROIC%"],              [False]),
    "ROE high to low":        (["ROE%"],               [False]),
    "Earn Traj high to low":  (["Earn Traj"],          [False]),
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

for c in ["P/E", "Fwd P/E", "PEG", "Earn Traj", "52W Pos%",
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
    label="Download CSV",
    data=disp_final.to_csv(index=False).encode("utf-8"),
    file_name="sp500_screener_v8_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M")),
    mime="text/csv",
)

# ── Explanatory text under table ──────────────────────────────────────────────
st.markdown(
    """
    **PEG (Price/Earnings-to-Growth):** Measures valuation relative to growth. 
    PEG < 1.0 = potentially undervalued for its growth rate. PEG > 2.0 = expensive relative to growth. 
    Only computed when EPS growth ≥ 5% (avoids math artifacts from near-zero denominators).
    
    **Earn Traj (Earnings Trajectory):** Proxy for analyst consensus revision direction.
    Computed as `(Forward EPS − Trailing EPS) / |Trailing EPS|`. 
    Positive = analysts expect earnings growth ahead. Negative = expect decline. 
    Range clipped to [-1.0, +1.0]. A value of +0.20 means 20% expected EPS improvement.
    """
)

st.markdown("---")
st.markdown(
    "**Sources v8:** Yahoo Finance primary (PE, Fwd PE, PEG via pegRatio, ROE, ROIC from financials, "
    "OpMargin, D/E, IntCoverage from EBIT/InterestExp, EPS growth, Earn Traj from FwdEPS/TrailEPS) · "
    "FMP bonus if key available (PE override, ROIC/IntCoverage override if tier supports /ratios-ttm) · "
    "Momentum: Yahoo batch download · Revenue: Yahoo quarterly"
)

# ══════════════════════════════════════════════════════════════════════════════
# ── COLUMN REFERENCE GUIDE ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("## Column Reference Guide")
st.caption("Every column explained with formula, real-world example, and how to use it.")

tab_val, tab_qual, tab_peg, tab_etraj, tab_mom, tab_rank, tab_disp = st.tabs([
    "Valuation", "Quality", "PEG", "Earn Trajectory", "Momentum", "Ranking & Score", "Display-Only Columns"
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

**Data source:** FMP /quote (if key available) → Yahoo Finance

**Used in scoring?** Yes — this is the primary Valuation factor (25% weight). Lower P/E = higher percentile = better score within sector. Fwd P/E is preferred when available.

---

### Fwd P/E — Forward Price to Earnings Ratio
**What it is:** Same as P/E but uses analysts' consensus estimate of what earnings will be over the NEXT 12 months.

**Formula:** `Current Stock Price ÷ Next 12-Month Estimated EPS`

**When Fwd P/E < Trailing P/E:** Earnings growing. Good sign.
**When Fwd P/E > Trailing P/E:** Earnings expected to shrink. Bad sign.

**Data source:** Yahoo Finance (.info forwardPE / forwardEps)

**Used in scoring?** Yes — Fwd P/E is PREFERRED over trailing P/E for the Valuation factor.

---

### 52W Pos% — 52-Week Position
**What it is:** Where the current price sits between the stock's 52-week low and 52-week high.

**Formula:** `(Current Price − 52W Low) ÷ (52W High − 52W Low) × 100`

**0% = at the yearly low. 100% = at the yearly high.**

**Used in scoring?** No. Display only.
""")

with tab_qual:
    st.markdown("""
### Quality Score (0–100)
**What it is:** A composite score measuring fundamental strength. Made up of three equally-weighted sub-scores.

**Formula:** `(ROIC sub-score + Interest Coverage sub-score + Op Margin sub-score) ÷ 3`

**Used in scoring?** Yes — 25% weight.

---

### ROIC% — Return on Invested Capital (v8: Yahoo-computed)
**What it is:** For every dollar of capital invested in the business (equity + debt − cash), how many cents of after-tax operating profit does it generate?

**Formula (v8):**
```
NOPAT = Operating Income (TTM) × (1 − effective tax rate)
Invested Capital = Total Equity + Total Debt − Cash
ROIC = NOPAT ÷ Invested Capital × 100
```

**Data source (v8):** Computed from Yahoo `quarterly_financials` + `quarterly_balance_sheet`. FMP `/ratios-ttm` overrides if your tier supports it.

**Benchmarks:**
- ROIC > 20%: Excellent competitive advantage
- ROIC 10–20%: Good solid business
- ROIC < 8%: Flagged — may be destroying value

**Sub-score formula:** `min(100, log(1 + ROIC) ÷ log(1 + 30) × 100)`

---

### Int Coverage — Interest Coverage Ratio
**What it is:** How many times over can the company pay its annual interest expense from operating profit.

**Formula:** `EBIT (TTM) ÷ |Interest Expense (TTM)|`

**Benchmarks:** 10x+ = safe (score 100), 5x = comfortable, <3x = flagged

**Data source:** Yahoo `quarterly_financials` (EBIT / Interest Expense)

---

### Op Margin% — Operating Margin
**What it is:** Of every dollar of revenue, how many cents become operating profit.

**Formula:** `Operating Income ÷ Revenue × 100`

**Data source:** Yahoo `.info` (operatingMargins)

---

### Quality Flag
- `ROIC<8%` — return on capital below threshold
- `ROE<8%` — shown when ROIC unavailable and ROE used as proxy
- `IntCov<3x` — interest coverage risky
- `Margin<5%` — thin profitability buffer
- `Pass` — all checks passed
- `D/E: X.X` — informational reference
""")

with tab_peg:
    st.markdown("""
### PEG — Price/Earnings-to-Growth Ratio
**What it is:** P/E adjusted for earnings growth rate.

**Formula:** `P/E Ratio ÷ Annual EPS Growth Rate (%)`

**Interpreting PEG:**
- PEG below 1.0: Cheap relative to growth — potentially undervalued
- PEG 1.0–2.0: Fairly valued for growth rate
- PEG above 2.0: Expensive relative to growth

**Data source waterfall:**
1. Yahoo `.info` pegRatio (direct — most common)
2. FMP `/ratios-ttm` if tier supports it (override)
3. Computed: (Fwd P/E or Trailing P/E) ÷ EPS Growth % from Yahoo

**Growth guard:** PEG only computed when EPS growth ≥ 5%.

**Used in scoring?** Yes — 20% weight. Lower PEG = better.
""")

with tab_etraj:
    st.markdown("""
### Earn Traj — Earnings Trajectory (v8 NEW)
**What it is:** Measures whether analysts expect earnings to grow or shrink relative to current trailing earnings. A proxy for the direction of consensus estimate revisions.

**Formula:** `(Forward EPS − Trailing EPS) ÷ |Trailing EPS|`

**Clipped to range:** [-1.0, +1.0]

**Example 1 — Positive trajectory:**
- Apple trailing EPS = $6.43, forward EPS estimate = $7.70
- Earn Traj = (7.70 − 6.43) ÷ 6.43 = **+0.20**
- Meaning: Analysts expect 20% earnings growth ahead. Positive signal.

**Example 2 — Negative trajectory:**
- Intel trailing EPS = $1.50, forward EPS estimate = $0.90
- Earn Traj = (0.90 − 1.50) ÷ 1.50 = **−0.40**
- Meaning: Analysts expect a 40% earnings decline. Bearish signal.

**Interpreting Earn Traj:**
- +0.5 to +1.0: Strong expected growth (fast-growing company)
- +0.1 to +0.3: Moderate growth expected
- 0.0: Flat — no change expected
- −0.1 to −0.3: Moderate decline expected
- −0.5 to −1.0: Significant decline expected

**Why this instead of true Earn Revision?**
True earnings revision (change in consensus estimates over time) requires tracking estimate history from services like FMP /analyst-estimates, which is behind a paid tier wall. This proxy captures the SAME directional signal — when forward EPS > trailing EPS, it means analysts see growth ahead, which correlates strongly with positive recent revisions.

**Data source:** Yahoo `.info` (forwardEps, trailingEps)
Expected coverage: ~85% of S&P 500 (excludes companies with negative trailing EPS where ratio is meaningless).

**Used in scoring?** Yes — 15% weight. Higher Earn Traj = higher percentile = better score within sector.
""")

with tab_mom:
    st.markdown("""
### Momentum Score — Skip-Month Volatility-Adjusted Momentum
**What it is:** Recent price trend adjusted for volatility.

**Formula:** `(6-month return − 1-month return) ÷ Trailing Volatility`

**Used in scoring?** Yes — 15% weight. Higher = better.

---

### Ret 1Mo%, Ret 3Mo%, Ret 6Mo% — Raw Monthly Returns
Display only. Shows raw return profile.

---

### Trailing Vol% — 90-Day Annualised Volatility
**Formula:** `std(daily_returns, last 90 days) × √252 × 100`

Display only — used as denominator in Momentum Score.
""")

with tab_rank:
    st.markdown("""
### Score (0–100)
**What it is:** Final composite percentile score within GICS sector.

**Formula:**
```
Score = 0.25 × Valuation_pctile + 0.25 × Quality_minmax + 
        0.20 × PEG_pctile + 0.15 × EarnTraj_pctile + 
        0.15 × Momentum_pctile
```
Then multiplied by missing-factor penalty (×0.85 if 2 missing, ×0.70 if 3+).

### Rank
Ordinal position within sector. Rank 1 = highest Score in that sector.

### Conviction Score (0–100)
Score × data completeness × sector discount, normalised across entire S&P 500.
""")

with tab_disp:
    st.markdown("""
### ROE% — Return on Equity (Display Only)
Net Income ÷ Shareholders Equity × 100. Distorted by leverage/buybacks. ROIC is the scoring metric.

### Debt/Eq — Debt to Equity Ratio (Display Only)
Total Debt ÷ Total Equity. Shown for reference. Interest Coverage is the scoring metric.

### Rev Q1–Q4 ($B) — Quarterly Revenue
Last four fiscal quarters of total revenue.

### Rev Growth% (CAGR)
`(Q4 Revenue ÷ Q1 Revenue)^(1/3) − 1 × 100`. Display only.

### Data Sources
Shows which providers contributed data for each stock (PE:Yahoo, ROIC:Yahoo, ET:Yahoo, etc.)
""")

st.markdown("""---
**Data sources v8:** Yahoo Finance (primary: PE, Fwd PE, PEG, ROE, ROIC from financials, OpMargin, D/E, 
IntCoverage, EPS growth, Earn Traj) · FMP /quote (bonus: PE, MC, 52W override) · 
FMP /ratios-ttm (bonus: ROIC, IntCov override if tier available) · 
Momentum: Yahoo batch · Revenue: Yahoo quarterly · Universe: Wikipedia S&P 500 GICS table
""")
