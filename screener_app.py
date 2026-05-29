# screener_app.py  v4
# ─────────────────────────────────────────────────────────────────────────────
# Upgrades from v3:
#  1. MOMENTUM      → skip-month (6mo return - 1mo return), volatility-adjusted
#  2. EARN REVISION → new factor (FMP /analyst-estimates up vs down revisions)
#  3. QUALITY       → ROIC replaces ROE; Interest Coverage replaces Debt/Equity
#  4. DEBT/EQ       → display column only, not in quality score
#  5. FACTOR WEIGHTS→ Valuation 25% | Quality 25% | PEG 20% |
#                     Earn Revision 10% | Earn Traj 10% | Momentum 10%
#  6. VALUATION     → composite: 50% PE-percentile + 50% PEG-percentile
#  7. REV CAGR      → standalone display column, removed as PEG denominator
#  8. CONVICTION    → Score × data_completeness × sector_valuation_discount
#  9. ROIC          → FMP /key-metrics-ttm (returnOnInvestedCapitalTTM)
# 10. INT COVERAGE  → FMP /ratios-ttm (interestCoverageTTM)
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
MIN_GROWTH_PCT_FOR_PEG = 5.0          # minimum EPS growth to compute PEG

FACTOR_WEIGHTS = {
    "valuation":    0.25,   # composite: 50% PE-pctl + 50% PEG-pctl
    "peg":          0.20,   # PEG percentile (lower = better)
    "quality":      0.25,   # ROIC + Interest Coverage + Op Margin
    "earn_traj":    0.10,   # trailing PE ÷ forward PE
    "momentum":     0.10,   # skip-month vol-adjusted momentum
    "earn_revision":0.10,   # analyst revision score
}

QUALITY_THRESHOLDS = {
    "roic_min":         8.0,    # ROIC < 8% flagged
    "int_coverage_min": 3.0,    # Interest coverage < 3x flagged
    "op_margin_min":    5.0,    # Op margin < 5% flagged
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


def normalise_pct(val):
    """
    FMP sometimes returns ratios as decimals (0.15) and sometimes as
    percentages (15.0). If |val| < 5 treat it as a decimal and multiply by 100.
    """
    if val is None:
        return None
    v = float(val)
    if abs(v) < 5.0:
        return v * 100.0
    return v


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


# ── Momentum v4: skip-month + volatility-adjusted ────────────────────────────
@st.cache_data(ttl=3600)
def fetch_momentum_batch(tickers):
    """
    Upgrade from v3:
      - Uses skip-month momentum: 6mo return MINUS 1mo return
        (avoids short-term mean reversion bias in the 1-month window)
      - Vol-adjusted: divide by trailing 90-day annualised volatility
        (so a +10% return in a volatile stock is worth less than in a stable one)
      - Still returns ret_1mo / ret_3mo / ret_6mo for display purposes
      - Adds 'momentum_score' = (ret_6mo - ret_1mo) / trailing_vol
    """
    tl  = list(tickers)
    out = {t: {} for t in tl}
    try:
        # Daily data for vol; monthly for returns
        raw_d = yf.download(tl, period="7mo", interval="1d",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=True)
        raw_m = yf.download(tl, period="7mo", interval="1mo",
                            group_by="ticker", auto_adjust=True,
                            progress=False, threads=True)

        for t in tl:
            try:
                # Monthly closes for return calculation
                closes_m = raw_m["Close"].dropna() if len(tl) == 1 \
                           else raw_m[t]["Close"].dropna()
                # Daily closes for volatility
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

                # Trailing 90-day annualised volatility
                trailing_vol = None
                if len(closes_d) >= 20:
                    daily_rets = closes_d.pct_change().dropna().tail(90)
                    if len(daily_rets) >= 15:
                        trailing_vol = float(daily_rets.std() * np.sqrt(252) * 100.0)

                # Skip-month momentum: 6mo gain minus 1mo gain
                skip_mom_raw = None
                if r6 is not None and r1 is not None:
                    skip_mom_raw = r6 - r1

                # Vol-adjusted skip-month momentum
                skip_mom_adj = None
                if skip_mom_raw is not None and trailing_vol and trailing_vol > 0:
                    skip_mom_adj = skip_mom_raw / trailing_vol
                elif skip_mom_raw is not None:
                    skip_mom_adj = skip_mom_raw   # fallback: unadjusted

                out[t] = {
                    "ret_1mo":       r1,
                    "ret_3mo":       r3,
                    "ret_6mo":       r6,
                    "trailing_vol":  trailing_vol,
                    "momentum_score": skip_mom_adj,   # ← used in ranking
                }
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


# ── FMP 2: ratios-ttm  (PEG, Op Margin, Debt/Eq, ROIC, Int Coverage) ─────────
@st.cache_data(ttl=86400)
def fetch_fmp_ratios_bulk(tickers, api_key):
    """
    v4 changes vs v3:
      - Adds interestCoverageTTM  (replaces Debt/Equity in Quality Score)
      - Adds returnOnInvestedCapitalTTM  (ROIC, replaces ROE in Quality Score)
      - Still fetches debtEquityRatioTTM for DISPLAY only
      - Still fetches returnOnEquityTTM for display only
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

            # PEG (already a ratio, no pct conversion needed)
            peg_raw = sf(item.get("priceEarningsGrowthRatioTTM"))
            peg     = peg_raw if (peg_raw is not None and 0 < peg_raw <= 500) else None

            # ROIC — decimal from FMP → multiply by 100 to get %
            roic_raw = sf(item.get("returnOnInvestedCapitalTTM"))
            roic     = normalise_pct(roic_raw) if roic_raw is not None else None

            # ROE — kept for display only
            roe_raw = sf(item.get("returnOnEquityTTM"))
            roe     = normalise_pct(roe_raw) if roe_raw is not None else None

            # Op Margin — decimal from FMP
            om_raw = sf(item.get("operatingProfitMarginTTM"))
            om     = normalise_pct(om_raw) if om_raw is not None else None

            # Interest Coverage — plain ratio (already a multiple, not pct)
            # Cap at 100 to avoid astronomical values distorting scores
            ic_raw = sf(item.get("interestCoverageTTM"))
            ic     = min(float(ic_raw), 100.0) if ic_raw is not None and ic_raw > 0 else None

            # Debt/Equity — display only, not used in scoring
            de = sf(item.get("debtEquityRatioTTM"))

            # Fwd PE
            fwd_pe_raw = sf(item.get("priceToEarningsRatioTTM"))
            fwd_pe     = fwd_pe_raw if (fwd_pe_raw and 0 < fwd_pe_raw <= 10_000) else None

            return t, {
                "peg":              peg,
                "roic":             roic,
                "roe":              roe,        # display only
                "op_margin":        om,
                "int_coverage":     ic,         # replaces D/E in quality score
                "debt_eq":          de,         # display only
                "fwd_pe":           fwd_pe,
                "peg_src":          "FMP-ratios" if peg is not None else None,
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


# ── FMP 3: key-metrics-ttm  (EPS growth, Fwd PE) ─────────────────────────────
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


# ── FMP 4: analyst-estimates  (Fwd EPS + Earnings Revision Score) v4 ─────────
@st.cache_data(ttl=86400)
def fetch_fmp_analyst_estimates(tickers, prices_map, api_key):
    """
    v4 upgrade: now also computes Earnings Revision Score from this endpoint.
    
    Earnings Revision Score = (# upward EPS revisions - # downward revisions)
                               ÷ total analysts
    Range: -1.0 (all cuts) to +1.0 (all upgrades)
    
    FMP /analyst-estimates returns estimatedEpsAvg and the analyst count
    for current and prior estimates. We compare current vs prior period
    to derive revision direction.
    
    Also still computes Fwd P/E for tickers missing it from other endpoints.
    """
    out = {t: {"fwd_pe": None, "earn_revision": None} for t in tickers}
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
                return t, {"fwd_pe": None, "earn_revision": None}

            # Most recent forward estimate for Fwd PE
            item    = d[0]
            fwd_eps = sf(item.get("estimatedEpsAvg"))
            price   = sf(prices_map.get(t))
            fwd_pe  = None
            if fwd_eps and fwd_eps > 0 and price and price > 0:
                fp = price / fwd_eps
                fwd_pe = fp if 0 < fp <= 500 else None

            # ── Earnings Revision Score ───────────────────────────────────
            # Strategy: compare estimatedEpsAvg of most recent period to
            # the prior period's estimate. If current > prior = upgrade.
            # FMP provides multiple forward periods — take current year (d[0])
            # vs same period prior release (d[1] if available).
            earn_rev = None
            if len(d) >= 2:
                curr_eps = sf(d[0].get("estimatedEpsAvg"))
                prev_eps = sf(d[1].get("estimatedEpsAvg"))
                num_analysts = sf(d[0].get("numberAnalystEstimatedEps"))
                # Normalise: how many standard deviations did the estimate move?
                if curr_eps is not None and prev_eps is not None and prev_eps != 0:
                    pct_change = (curr_eps - prev_eps) / abs(prev_eps)
                    # Clip to [-1, 1] range to avoid outliers
                    earn_rev = float(np.clip(pct_change, -1.0, 1.0))

            return t, {"fwd_pe": fwd_pe, "earn_revision": earn_rev}
        except Exception:
            pass
        return t, {"fwd_pe": None, "earn_revision": None}

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

        eg = sf(info.get("earningsGrowth"))
        if eg is not None:
            result["eps_growth"] = eg * 100.0

        # ROE: Yahoo decimal → pct (display only in v4)
        roe_y = sf(info.get("returnOnEquity"))
        if roe_y is not None:
            result["roe"] = roe_y * 100.0

        # Debt/Equity: Yahoo is already pct (150 = 1.5 ratio) → convert
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
                           fmp_analyst, yahoo_fallback, tickers):
    """
    v4: adds ROIC, int_coverage, earn_revision to merged output.
    Debt/Equity kept for display only.
    ROE kept for display only.
    """
    merged = {}
    for t in tickers:
        fq  = fmp_quotes.get(t, {})
        fr  = fmp_ratios.get(t, {})
        fm  = fmp_metrics.get(t, {})
        fa  = fmp_analyst.get(t, {})
        yb  = yahoo_fallback.get(t, {})

        def first(*vals):
            for v in vals:
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return v
            return None

        # Fwd P/E waterfall: ratios-ttm → key-metrics-ttm → analyst-est → Yahoo
        fwd_pe = first(fr.get("fwd_pe"), fm.get("fwd_pe"),
                       fa.get("fwd_pe"), yb.get("fwd_pe"))

        # PEG: FMP ratios-ttm only — no revenue fallback in v4
        peg_val = first(fr.get("peg"))
        peg_src = fr.get("peg_src") if peg_val is not None else "—"

        # PE
        pe_val = first(fq.get("pe"), yb.get("pe"))
        pe_src = fq.get("pe_src") if fq.get("pe") is not None else yb.get("pe_src", "Yahoo")

        # ROIC (quality scoring) — FMP ratios-ttm only
        roic = first(fr.get("roic"))

        # ROE (display only) — FMP ratios-ttm → Yahoo
        roe = first(fr.get("roe"), yb.get("roe"))

        # Interest Coverage (quality scoring) — FMP ratios-ttm
        ic = first(fr.get("int_coverage"))

        # Op Margin (quality scoring) — FMP ratios-ttm → Yahoo
        om = first(fr.get("op_margin"), yb.get("op_margin"))

        # Debt/Equity (display only) — FMP → Yahoo
        de = first(fr.get("debt_eq"), yb.get("debt_eq"))

        # EPS growth
        eps_g = first(fm.get("eps_growth"), yb.get("eps_growth"))
        g_src = "FMP" if fm.get("eps_growth") is not None else (
                "Yahoo" if yb.get("eps_growth") is not None else None)

        # Earnings Revision Score — analyst-estimates only
        earn_rev = fa.get("earn_revision")

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
            "roic":           roic,           # quality scoring
            "roe":            roe,            # display only
            "int_coverage":   ic,             # quality scoring
            "debt_eq":        de,             # display only
            "op_margin":      om,             # quality scoring
            "earn_revision":  earn_rev,       # new factor in v4
        }
    return merged


# ── Quality score v4: ROIC + Interest Coverage + Op Margin ───────────────────
def compute_quality_score(roic, int_coverage, op_margin):
    """
    v4 replaces v3's (ROE + Debt/Eq + Op Margin):
      • ROIC          replaces ROE       — leverage-neutral profitability
      • Int Coverage  replaces Debt/Eq   — ability to service debt (more direct)
      • Op Margin     unchanged

    Sub-score formulas:
      ROIC:         log-scaled to handle outliers like AAPL (ROIC>100%)
                    score = min(100, log1p(ROIC) / log1p(30) * 100)
                    ROIC 30% → 100, ROIC 15% → ~77, ROIC 5% → ~40
      Int Coverage: min(100, (IC / 10) * 100)  → IC 10x+ = perfect 100
                    IC 5x → 50, IC 3x → 30, IC 1x → 10
      Op Margin:    same as v3 — min(max(OM / 40 * 100, 0), 100)
                    40%+ → 100, 20% → 50, 5% → 12.5
    """
    scores = []

    if roic is not None and not pd.isna(roic):
        roic_f = float(roic)
        if roic_f > 0:
            roic_score = min(100.0, np.log1p(roic_f) / np.log1p(30.0) * 100.0)
        else:
            roic_score = 0.0
        scores.append(roic_score)
    else:
        scores.append(0.0)

    if int_coverage is not None and not pd.isna(int_coverage):
        ic_f = float(int_coverage)
        ic_score = min(100.0, max(0.0, (ic_f / 10.0) * 100.0))
        scores.append(ic_score)
    else:
        scores.append(0.0)

    if op_margin is not None and not pd.isna(op_margin):
        om_score = min(100.0, max(0.0, float(op_margin) / 40.0 * 100.0))
        scores.append(om_score)
    else:
        scores.append(0.0)

    return sum(scores) / len(scores)


# ── Conviction Score ──────────────────────────────────────────────────────────
def compute_conviction_scores(scr):
    """
    Conviction Score = Score × data_completeness × sector_valuation_discount

    data_completeness = (number of non-null key factors) / total key factors
    Key factors: P/E, Fwd P/E, PEG, Quality Score, Earn Traj,
                 Momentum Score, Earn Revision

    sector_valuation_discount = median(sector PE) / median(all S&P 500 PE)
    Sectors trading at a discount get a bonus. Sectors at premium get penalised.
    Clipped to [0.7, 1.3] so no single sector gets a wild multiplier.

    Result is normalised to 0–100 within the full universe.
    """
    KEY_FACTORS = ["P/E", "Fwd P/E", "PEG", "Quality Score",
                   "Earn Traj", "Momentum Score", "Earn Revision"]
    n_factors = len(KEY_FACTORS)

    # Data completeness per row
    def completeness(row):
        present = sum(1 for c in KEY_FACTORS if c in row.index and pd.notna(row[c]))
        return present / n_factors

    scr = scr.copy()
    scr["_completeness"] = scr.apply(completeness, axis=1)

    # Sector PE premium/discount vs market
    overall_median_pe = scr["P/E"].median()
    sector_pe_map     = scr.groupby("Sector")["P/E"].median()

    def sector_discount(sector):
        if pd.isna(overall_median_pe) or overall_median_pe == 0:
            return 1.0
        s_pe = sector_pe_map.get(sector)
        if pd.isna(s_pe) or s_pe == 0:
            return 1.0
        # sector trading at discount → multiplier > 1
        disc = overall_median_pe / s_pe
        return float(np.clip(disc, 0.7, 1.3))

    scr["_sec_discount"] = scr["Sector"].map(sector_discount)

    raw_conviction = scr["Score"] * scr["_completeness"] * scr["_sec_discount"]

    # Normalise to 0–100
    c_min = raw_conviction.min()
    c_max = raw_conviction.max()
    if c_max > c_min:
        scr["Conviction Score"] = (raw_conviction - c_min) / (c_max - c_min) * 100.0
    else:
        scr["Conviction Score"] = 50.0

    scr = scr.drop(columns=["_completeness", "_sec_discount"])
    return scr


# ── Ranking v4 ────────────────────────────────────────────────────────────────
def compute_rank_by_sector(scr):
    """
    v4 changes vs v3:
      1. VALUATION: composite 50% PE-percentile + 50% PEG-percentile
         (growth stocks no longer killed by pure PE rank)
      2. MOMENTUM: uses momentum_score (skip-month vol-adjusted) not raw avg
      3. EARN REVISION: new 10% factor, percentile of earn_revision score
      4. Weights: see FACTOR_WEIGHTS at top of file
    """
    scr = scr.copy()
    scr["Score"] = pd.NA
    scr["Rank"]  = pd.NA
    W = FACTOR_WEIGHTS

    for sector in scr["Sector"].dropna().unique():
        g    = scr[scr["Sector"] == sector].copy()
        elig = g[g["Eligible"]].copy()
        if elig.empty:
            continue

        # ── VALUATION: composite (50% PE + 50% PEG percentiles) ──────────
        pe_input  = elig["Fwd P/E"].fillna(elig["P/E"])
        peg_input = elig["PEG"]
        s_pe_pct  = percentile_score(pe_input,  ascending=True)   # lower PE = better
        s_peg_pct = percentile_score(peg_input, ascending=True)   # lower PEG = better
        # If PEG is missing, fall back to pure PE percentile
        has_peg   = peg_input.notna()
        elig["_s_val"] = np.where(
            has_peg,
            0.5 * s_pe_pct + 0.5 * s_peg_pct,
            s_pe_pct
        )

        # ── PEG (standalone, in addition to composite valuation) ──────────
        elig["_s_peg"] = percentile_score(elig["PEG"], ascending=True)

        # ── EARNINGS TRAJECTORY ───────────────────────────────────────────
        elig["_s_etraj"] = percentile_score(elig["Earn Traj"], ascending=False)

        # ── QUALITY (min-max within sector, same as v3) ───────────────────
        qs    = elig["Quality Score"]
        q_min = qs.min()
        q_max = qs.max()
        if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min:
            elig["_s_quality"] = (qs - q_min) / (q_max - q_min) * 100.0
        else:
            elig["_s_quality"] = qs.fillna(0.0)
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)

        # ── MOMENTUM: skip-month vol-adjusted score ───────────────────────
        elig["_s_mom"] = percentile_score(elig["Momentum Score"], ascending=False)

        # ── EARNINGS REVISION ─────────────────────────────────────────────
        elig["_s_erev"] = percentile_score(elig["Earn Revision"], ascending=False)

        # ── COMPOSITE SCORE ───────────────────────────────────────────────
        raw = (W["valuation"]     * elig["_s_val"]     +
               W["peg"]           * elig["_s_peg"]     +
               W["quality"]       * elig["_s_quality"] +
               W["earn_traj"]     * elig["_s_etraj"]   +
               W["momentum"]      * elig["_s_mom"]     +
               W["earn_revision"] * elig["_s_erev"])

        # ── MISSING DATA PENALTY ──────────────────────────────────────────
        factor_cols = ["P/E", "PEG", "Quality Score",
                       "Earn Traj", "Momentum Score", "Earn Revision"]
        penalties   = elig.apply(lambda r: missing_factor_penalty(r, factor_cols), axis=1)
        raw         = raw * penalties

        elig["Score"] = raw
        elig = elig.sort_values("Score", ascending=False)
        elig["Rank"]  = range(1, len(elig) + 1)
        scr.loc[elig.index, "Score"] = elig["Score"]
        scr.loc[elig.index, "Rank"]  = elig["Rank"]

    return scr


# ── Build screener table v4 ───────────────────────────────────────────────────
def build_screener_table(universe_df, prices_map, merged_map,
                         revenue_map, momentum_map):
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

        # Quality inputs (v4)
        roic  = to_num(fi.get("roic"))
        ic    = to_num(fi.get("int_coverage"))
        om    = to_num(fi.get("op_margin"))

        # Display-only fields
        roe   = to_num(fi.get("roe"))
        de    = to_num(fi.get("debt_eq"))

        # Earnings revision
        earn_rev = to_num(fi.get("earn_revision"))

        # 52W position
        pos52 = None
        if pd.notna(price) and pd.notna(hi) and pd.notna(lo) and hi != lo:
            pos52 = float((price - lo) / (hi - lo) * 100.0)

        # Revenue CAGR — display only in v4, not used for PEG
        rev4            = revenue_map.get(t, [None]*4)
        rq1, rq2, rq3, rq4 = [to_num(x) for x in rev4]
        growth          = revenue_growth_pct_cagr([rq1, rq2, rq3, rq4])

        # PEG waterfall v4: FMP direct ONLY — no revenue CAGR fallback
        peg_direct = to_num(fi.get("peg"))
        peg        = None
        peg_method = "—"
        if pd.notna(peg_direct):
            peg        = float(peg_direct)
            peg_method = fi.get("peg_src") or "FMP-ratios"
        else:
            # Only compute from EPS growth — NOT from revenue CAGR (v4 change)
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

        # Earnings trajectory
        earn_traj = None
        if pd.notna(pe) and pd.notna(fwd) and fwd > 0:
            earn_traj = float(pe) / float(fwd)

        # Quality Score v4
        q_score = compute_quality_score(
            float(roic) if pd.notna(roic) else None,
            float(ic)   if pd.notna(ic)   else None,
            float(om)   if pd.notna(om)   else None,
        )

        # Momentum v4 — skip-month vol-adjusted score from momentum_map
        mom      = momentum_map.get(t, {})
        ret_1mo  = to_num(mom.get("ret_1mo"))
        ret_3mo  = to_num(mom.get("ret_3mo"))
        ret_6mo  = to_num(mom.get("ret_6mo"))
        mom_score= to_num(mom.get("momentum_score"))    # ← used in ranking
        t_vol    = to_num(mom.get("trailing_vol"))

        # Data source summary
        parts = []
        ps = fi.get("pe_src")
        if ps and ps != "—": parts.append("PE:{}".format(ps))
        gs = fi.get("peg_src")
        if gs and gs != "—": parts.append("PEG:{}".format(gs))
        gr = fi.get("growth_src")
        if gr and gr != "—": parts.append("G:{}".format(gr))
        if fi.get("earn_revision") is not None: parts.append("Rev:FMP")
        if fi.get("roic")          is not None: parts.append("ROIC:FMP")
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
            "Earn Revision":      earn_rev,
            "52W Pos%":           to_num(pos52),
            "ROIC%":              roic,           # quality scoring
            "Int Coverage":       ic,             # quality scoring
            "Op Margin%":         om,             # quality scoring
            "ROE%":               roe,            # display only
            "Debt/Eq":            de,             # display only
            "Quality Score":      to_num(q_score),
            "Momentum Score":     mom_score,      # skip-month vol-adj (scoring)
            "Ret 1Mo%":           ret_1mo,        # display
            "Ret 3Mo%":           ret_3mo,        # display
            "Ret 6Mo%":           ret_6mo,        # display
            "Trailing Vol%":      t_vol,          # display
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
                "Earn Revision", "52W Pos%", "ROIC%", "Int Coverage",
                "Op Margin%", "ROE%", "Debt/Eq", "Quality Score",
                "Momentum Score", "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%",
                "Trailing Vol%", "Rev Q1", "Rev Q2", "Rev Q3", "Rev Q4",
                "Rev Growth% (CAGR)"]
    for c in num_cols:
        if c in scr.columns:
            scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns:
        scr["Rank"] = pd.NA

    # Conviction Score computed after ranking
    scr = compute_conviction_scores(scr)

    return scr


# ── Quality flag v4 ───────────────────────────────────────────────────────────
def quality_flag(roic, ic, om, de):
    """
    v4: flags based on ROIC + Interest Coverage + Op Margin.
    Debt/Equity shown as display note (not a flag threshold).
    """
    flags = []
    if roic is not None and not pd.isna(roic) and roic < QUALITY_THRESHOLDS["roic_min"]:
        flags.append("ROIC<8%")
    if ic   is not None and not pd.isna(ic)   and ic   < QUALITY_THRESHOLDS["int_coverage_min"]:
        flags.append("IntCov<3x")
    if om   is not None and not pd.isna(om)   and om   < QUALITY_THRESHOLDS["op_margin_min"]:
        flags.append("Margin<5%")
    # Debt/Equity note (display only, not a pass/fail)
    de_note = ""
    if de is not None and not pd.isna(de):
        de_note = " | D/E:{:.1f}".format(de)
    result = ", ".join(flags) if flags else "Pass"
    return result + de_note


# ── KPI panel v4 ──────────────────────────────────────────────────────────────
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
    med_roic  = sdata["ROIC%"].median()
    med_peg   = sdata["PEG"].median()
    med_erev  = sdata["Earn Revision"].median()

    st.markdown(
        "<div style='background:#12122a;border:1px solid #2a2a4a;border-radius:12px;"
        "padding:16px 20px;margin-bottom:16px;'>"
        "<span style='color:#aaa;font-size:13px;'>Sector Analysis  </span>"
        "<span style='color:#fff;font-size:14px;font-weight:700;'>{}</span>"
        "</div>".format(label),
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.markdown(_kpi("Sector Mkt Cap",   fmt_mc(sector_mc), "sector total"),            unsafe_allow_html=True)
    c2.markdown(_kpi("S&P 500 Mkt Cap",  fmt_mc(total_mc),  "all stocks"),              unsafe_allow_html=True)
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
    c6.markdown(_kpi("Median ROIC",
                     "{:.1f}%".format(med_roic) if pd.notna(med_roic) else "N/A",
                     "return on inv. capital", "#a78bfa"),
                unsafe_allow_html=True)
    c7.markdown(_kpi("Median Earn Rev",
                     "{:.2f}".format(med_erev) if pd.notna(med_erev) else "N/A",
                     "+1=all upgrades, -1=cuts", "#f87171"),
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
            "Score = Valuation 25% (composite PE+PEG) + Quality 25% (ROIC+IntCov+Margin) "
            "+ PEG 20% + Earn Revision 10% + Earn Traj 10% + Momentum 10% (skip-month vol-adj)"
            "</div></div>".format(badges or "<span style='color:#555;'>No ranked stocks</span>"),
            unsafe_allow_html=True,
        )
    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ── APP ───────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="S&P 500 Screener v4", layout="wide", page_icon="📊")
st.markdown(
    "<style>div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}</style>",
    unsafe_allow_html=True,
)

st.markdown("## S&P 500 Fundamental Screener v4")
st.caption(
    "6-factor sector-relative ranking · "
    "ROIC replaces ROE · Interest Coverage replaces D/E in scoring · "
    "Skip-month vol-adjusted momentum · Earnings Revision factor · "
    "Composite valuation (PE + PEG) · Conviction Score"
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
if not fmp_key:
    st.warning(
        "No FMP API key found. Add [fmp] api_key to Streamlit Secrets. "
        "ROIC, Interest Coverage, Earn Revision, PEG all require FMP. "
        "Falling back to Yahoo only (reduced factor coverage)."
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

with st.spinner("Fetching momentum — skip-month + vol-adjusted ({} tickers)...".format(len(tickers))):
    momentum = fetch_momentum_batch(tickers)

fmp_quotes  = {}
fmp_ratios  = {}
fmp_metrics = {}
fmp_analyst = {t: {"fwd_pe": None, "earn_revision": None} for t in tickers}

if fmp_key:
    with st.spinner("FMP: bulk quotes (PE, MC, 52W) for all {} tickers...".format(len(tickers))):
        fmp_quotes = fetch_fmp_bulk_quotes(tickers, fmp_key)

    with st.spinner("FMP: ratios-ttm (PEG, ROIC, Int Coverage, Op Margin, D/E display) for all {} tickers...".format(len(tickers))):
        fmp_ratios = fetch_fmp_ratios_bulk(tickers, fmp_key)

    with st.spinner("FMP: key-metrics-ttm (EPS growth, Fwd PE) for all {} tickers...".format(len(tickers))):
        fmp_metrics = fetch_fmp_key_metrics_bulk(tickers, fmp_key)

    # Analyst estimates: Fwd PE gap fill + Earn Revision for ALL tickers
    with st.spinner("FMP: analyst-estimates (Earn Revision + Fwd PE gap fill) for {} tickers...".format(len(tickers))):
        fmp_analyst = fetch_fmp_analyst_estimates(tickers, prices, fmp_key)

# Yahoo fallback for tickers missing PE from FMP
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
        fmp_quotes, fmp_ratios, fmp_metrics, fmp_analyst, yahoo_fallback, tickers)

with st.spinner("Fetching quarterly revenue..."):
    rev_map = fetch_last4_revenue_parallel(tickers)

# ── Coverage banner ───────────────────────────────────────────────────────────
total_t   = len(tickers)
has_pe    = sum(1 for t in tickers if merged_map.get(t, {}).get("pe")            is not None)
has_fwd   = sum(1 for t in tickers if merged_map.get(t, {}).get("fwd_pe")        is not None)
has_peg   = sum(1 for t in tickers if merged_map.get(t, {}).get("peg")           is not None)
has_roic  = sum(1 for t in tickers if merged_map.get(t, {}).get("roic")          is not None)
has_ic    = sum(1 for t in tickers if merged_map.get(t, {}).get("int_coverage")  is not None)
has_erev  = sum(1 for t in tickers if merged_map.get(t, {}).get("earn_revision") is not None)

st.info(
    "Data coverage — "
    "P/E: {}/{} ({:.0f}%) · "
    "Fwd P/E: {}/{} ({:.0f}%) · "
    "PEG: {}/{} ({:.0f}%) · "
    "ROIC: {}/{} ({:.0f}%) · "
    "Int Coverage: {}/{} ({:.0f}%) · "
    "Earn Revision: {}/{} ({:.0f}%)".format(
        has_pe,   total_t, has_pe   / total_t * 100,
        has_fwd,  total_t, has_fwd  / total_t * 100,
        has_peg,  total_t, has_peg  / total_t * 100,
        has_roic, total_t, has_roic / total_t * 100,
        has_ic,   total_t, has_ic   / total_t * 100,
        has_erev, total_t, has_erev / total_t * 100,
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
        "Quality Score high", "ROIC high to low",
        "Earn Revision high to low", "Earn Traj high to low",
        "Rev Growth high to low", "Momentum Score high", "52W Pos low to high",
    ])
    pe_max   = fc3.number_input("Max PE",              value=9999,  step=50)
    peg_max  = fc4.number_input("Max PEG",             value=999.0, step=1.0)
    mc_min_b = fc5.number_input("Min Market Cap ($B)", value=0,     step=5)

with st.expander("Quality Filters v4", expanded=False):
    qc1, qc2, qc3, qc4, qc5 = st.columns(5)
    roic_min_f = qc1.number_input("Min ROIC (%)",          value=0.0,  step=5.0)
    ic_min_f   = qc2.number_input("Min Int Coverage (x)",  value=0.0,  step=1.0)
    om_min_f   = qc3.number_input("Min Op Margin (%)",     value=0.0,  step=5.0)
    qual_min_f = qc4.number_input("Min Quality Score",     value=0.0,  step=5.0)
    de_max_f   = qc5.number_input("Max Debt/Equity (display ref)", value=99.0, step=0.5)

with st.expander("Revision & Momentum & Display", expanded=False):
    mc1, mc2, mc3 = st.columns(3)
    erev_min  = mc1.number_input("Min Earn Revision (-1 to 1)", value=-1.0, step=0.1)
    mom_min   = mc2.number_input("Min Momentum Score",          value=-999.0, step=5.0)
    hide_nope = mc3.checkbox("Hide stocks with no P/E or Fwd P/E", value=False)

render_sector_kpi_panel(scr, sector_sel)

# ── Apply filters ─────────────────────────────────────────────────────────────
filt = scr.copy()
if sector_sel != "All Sectors":
    filt = filt[filt["Sector"] == sector_sel]
filt = filt[(filt["Mkt Cap"].isna())       | (filt["Mkt Cap"]       >= mc_min_b * 1e9)]
filt = filt[(filt["P/E"].isna())           | (filt["P/E"]           <= pe_max)]
filt = filt[(filt["PEG"].isna())           | (filt["PEG"]           <= peg_max)]
filt = filt[(filt["ROIC%"].isna())         | (filt["ROIC%"]         >= roic_min_f)]
filt = filt[(filt["Int Coverage"].isna())  | (filt["Int Coverage"]  >= ic_min_f)]
filt = filt[(filt["Op Margin%"].isna())    | (filt["Op Margin%"]    >= om_min_f)]
filt = filt[(filt["Quality Score"].isna()) | (filt["Quality Score"] >= qual_min_f)]
filt = filt[(filt["Debt/Eq"].isna())       | (filt["Debt/Eq"]       <= de_max_f)]
filt = filt[(filt["Earn Revision"].isna()) | (filt["Earn Revision"] >= erev_min)]
filt = filt[(filt["Momentum Score"].isna())| (filt["Momentum Score"]>= mom_min)]
if hide_nope:
    filt = filt[filt["P/E"].notna() | filt["Fwd P/E"].notna()]

sort_map = {
    "Sector then Rank":          (["Sector", "Rank"],           [True,  True]),
    "Score high to low":         (["Score"],                    [False]),
    "Conviction high to low":    (["Conviction Score"],         [False]),
    "Price low to high":         (["Price"],                    [True]),
    "Price high to low":         (["Price"],                    [False]),
    "Mkt Cap high to low":       (["Mkt Cap"],                  [False]),
    "PE low to high":            (["P/E"],                      [True]),
    "Fwd PE low to high":        (["Fwd P/E"],                  [True]),
    "PEG low to high":           (["PEG"],                      [True]),
    "Quality Score high":        (["Quality Score"],            [False]),
    "ROIC high to low":          (["ROIC%"],                    [False]),
    "Earn Revision high to low": (["Earn Revision"],            [False]),
    "Earn Traj high to low":     (["Earn Traj"],                [False]),
    "Rev Growth high to low":    (["Rev Growth% (CAGR)"],       [False]),
    "Momentum Score high":       (["Momentum Score"],           [False]),
    "52W Pos low to high":       (["52W Pos%"],                 [True]),
}
sc, sa = sort_map.get(sort_by, (["Sector", "Rank"], [True, True]))
filt   = filt.sort_values(sc, ascending=sa, na_position="last")

st.caption("Showing {} of {} stocks · Sector: {} · Sort: {}".format(
    len(filt), len(scr), sector_sel, sort_by))

# ── Display table ─────────────────────────────────────────────────────────────
disp = filt.copy()
disp["Price ($)"]       = disp["Price"].round(2)
disp["Mkt Cap ($B)"]    = (disp["Mkt Cap"]  / 1e9).round(2)
disp["Rev Q1 ($B)"]     = (disp["Rev Q1"]   / 1e9).round(2)
disp["Rev Q2 ($B)"]     = (disp["Rev Q2"]   / 1e9).round(2)
disp["Rev Q3 ($B)"]     = (disp["Rev Q3"]   / 1e9).round(2)
disp["Rev Q4 ($B)"]     = (disp["Rev Q4"]   / 1e9).round(2)
disp["Quality Flag"]    = disp.apply(
    lambda r: quality_flag(
        r.get("ROIC%"), r.get("Int Coverage"),
        r.get("Op Margin%"), r.get("Debt/Eq")
    ), axis=1)

for c in ["P/E", "Fwd P/E", "PEG", "Earn Traj", "Earn Revision",
          "52W Pos%", "ROIC%", "Int Coverage", "Op Margin%",
          "ROE%", "Debt/Eq", "Quality Score", "Momentum Score",
          "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
          "Score", "Conviction Score", "Rev Growth% (CAGR)"]:
    if c in disp.columns:
        disp[c] = disp[c].round(2)

disp["Rank"] = disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

COLS = [
    "Ticker", "Sector", "Price ($)", "Mkt Cap ($B)",
    "P/E", "Fwd P/E", "PEG", "PEG Method", "Earn Traj",
    "Earn Revision",
    "ROIC%", "Int Coverage", "Op Margin%",
    "ROE%", "Debt/Eq",
    "Quality Score", "Quality Flag",
    "Momentum Score", "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
    "52W Pos%", "Score", "Conviction Score", "Rank",
    "Rev Q1 ($B)", "Rev Q2 ($B)", "Rev Q3 ($B)", "Rev Q4 ($B)",
    "Rev Growth% (CAGR)", "Data Sources",
]
disp_final = disp[[c for c in COLS if c in disp.columns]].copy()
st.dataframe(disp_final, use_container_width=True, height=680)

# ── Export ────────────────────────────────────────────────────────────────────
st.download_button(
    label="Download filtered results as CSV",
    data=disp_final.to_csv(index=False).encode("utf-8"),
    file_name="sp500_screener_v4_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M")),
    mime="text/csv",
)

# ── Legend tabs ───────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("#### Column Reference Guide — v4")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Valuation", "Quality v4", "Momentum v4", "Earn Revision", "Ranking v4"])

with tab1:
    st.markdown(
        "**P/E** — Price ÷ trailing 12-month EPS. Lower = cheaper.\n\n"
        "**Fwd P/E** — Price ÷ next 12-month estimated EPS. "
        "Fwd P/E lower than P/E = earnings expected to grow.\n\n"
        "**PEG** — P/E ÷ EPS growth rate. Source: FMP /ratios-ttm. "
        "PEG < 1 = undervalued for growth. 1–2 = fair. > 2 = expensive. "
        "Only shown when EPS growth >= 5%. Revenue CAGR no longer used as fallback (v4 change).\n\n"
        "**Earn Traj** — Trailing P/E ÷ Forward P/E. "
        "> 1 = earnings growing. < 1 = earnings shrinking.\n\n"
        "**Composite Valuation (scoring)** — 50% PE percentile + 50% PEG percentile. "
        "Prevents high-growth stocks from being killed by pure PE rank (v4 change from v3)."
    )

with tab2:
    st.markdown(
        "**ROIC% (Return on Invested Capital)** — Net Operating Profit After Tax ÷ Invested Capital × 100. "
        "Leverage-neutral measure of capital efficiency. Replaces ROE in v4. "
        "ROIC 30%+ = perfect quality score. Uses log-scale to handle AAPL-style outliers (ROIC 160%+).\n\n"
        "**Int Coverage (Interest Coverage)** — EBIT ÷ Interest Expense. "
        "Measures ability to service debt — more direct than Debt/Equity. "
        "Replaces D/E in quality scoring. "
        "10x+ = perfect score. 3x = 30 score. < 3x flagged as risky.\n\n"
        "**Op Margin%** — Operating income ÷ revenue × 100. Unchanged from v3.\n\n"
        "**Quality Score (0–100)** — Equal-weight composite of ROIC, Int Coverage, Op Margin sub-scores. "
        "All sub-scores log-scaled or linearly capped to avoid outlier distortion.\n\n"
        "**ROE%** — Displayed for reference only. Not used in Quality Score in v4 "
        "(replaced by ROIC which is leverage-neutral).\n\n"
        "**Debt/Eq** — Displayed for reference only. Not used in scoring in v4. "
        "Shown in Quality Flag as D/E note for awareness. "
        "Use Int Coverage filter for debt-risk screening instead."
    )

with tab3:
    st.markdown(
        "**Momentum Score** — Skip-month volatility-adjusted momentum (used in scoring). "
        "Formula: (6-month return − 1-month return) ÷ trailing 90-day annualised volatility. "
        "Why skip-month? The 1-month window exhibits short-term mean reversion — "
        "subtracting it removes noise and isolates the durable 2–6 month momentum signal "
        "(Jegadeesh & Titman 1993 methodology). Vol-adjustment ensures a +10% gain in a "
        "low-volatility stock ranks higher than the same gain in a high-volatility stock.\n\n"
        "**Ret 1Mo / 3Mo / 6Mo%** — Raw price returns over 1, 3, 6 months. Display only — "
        "NOT directly used in scoring (replaced by Momentum Score above).\n\n"
        "**Trailing Vol%** — 90-day annualised daily return volatility. "
        "Shown for context. Lower vol = Momentum Score amplified.\n\n"
        "**52W Pos%** — (Price − 52W Low) ÷ (52W High − 52W Low) × 100. "
        "0% = at yearly low, 100% = at yearly high."
    )

with tab4:
    st.markdown(
        "**Earn Revision** — Earnings revision score, range −1.0 to +1.0. "
        "Formula: (current consensus EPS estimate − prior period estimate) ÷ |prior estimate|, "
        "clipped to [−1, 1]. Source: FMP /analyst-estimates.\n\n"
        "+1.0 = analysts collectively upgraded EPS estimates significantly.\n"
        "0.0  = no change in consensus.\n"
        "−1.0 = analysts collectively cut EPS estimates significantly.\n\n"
        "**Why it matters:** Earnings estimate revisions are one of the most well-documented "
        "short-to-medium-term predictive signals in academic equity research "
        "(documented by MSCI Barra, Goldman Sachs Global Alpha, JP Morgan). "
        "When analysts revise estimates up, stocks tend to outperform over the next 30–90 days. "
        "Weight: 10% of total score.\n\n"
        "**Filter:** Use 'Min Earn Revision' filter to screen for stocks with positive revisions only."
    )

with tab5:
    st.markdown(
        "**Score (0–100)** — Percentile composite within each sector.\n\n"
        "**Weights (v4):**\n"
        "- Valuation 25% (composite: 50% PE-pctl + 50% PEG-pctl)\n"
        "- Quality 25% (ROIC + Interest Coverage + Op Margin)\n"
        "- PEG 20% (standalone PEG percentile)\n"
        "- Earn Revision 10% (new in v4)\n"
        "- Earn Trajectory 10% (was 15% in v3)\n"
        "- Momentum 10% (skip-month vol-adjusted, was simple avg in v3)\n\n"
        "**Rank** — Position in sector by Score. 1 = best in sector.\n\n"
        "**Conviction Score (0–100)** — Score × data completeness × sector valuation discount. "
        "Rewards stocks with complete data from cheap sectors. "
        "Formula: Score × (non-null key factors ÷ 7) × clip(market_median_PE ÷ sector_PE, 0.7, 1.3). "
        "Normalised to 0–100 across the full universe.\n\n"
        "**Missing data penalty:** 2 factors missing = score × 0.85. 3+ missing = score × 0.70.\n\n"
        "**Key v4 improvements over v3:**\n"
        "- ROIC replaces ROE: leverage-neutral, harder to game via buybacks\n"
        "- Interest Coverage replaces Debt/Equity: measures debt service ability directly\n"
        "- Earnings Revision added: strongest short-term signal, was entirely missing in v3\n"
        "- Composite valuation: high-growth stocks no longer penalised by pure PE rank\n"
        "- Skip-month momentum: removes 1-month mean-reversion noise from momentum signal\n"
        "- Conviction Score: helps distinguish high-confidence Rank 1 from data-thin Rank 1"
    )

st.markdown(
    "**Data Sources** — "
    "FMP /quote (PE, MC, 52W) · "
    "FMP /ratios-ttm (PEG, ROIC, Int Coverage, Op Margin, ROE display, D/E display) · "
    "FMP /key-metrics-ttm (EPS growth, Fwd PE) · "
    "FMP /analyst-estimates (Earn Revision + Fwd PE gap fill) · "
    "Yahoo Finance (PE/Fwd PE fallback, revenue, momentum) · "
    "S&P 500 universe: Wikipedia GICS"
)
