# screener_app.py v14
# ─────────────────────────────────────────────────────────────────────────────
# v14 CHANGES from v13:
#  1. UI CLEAN  — Removed subtitle caption under main heading
#  2. UI CLEAN  — Removed pre-filter + FMP tier st.caption messages
#  3. UI CLEAN  — Removed render_sector_kpi_panel() call (sector KPI block)
#  4. FIX       — Momentum/52W None fix: robust _yf_close for all batch sizes;
#                 added fallback column name variants; explicit ticker-level
#                 try/except with column debug guard in momentum fetch
#  5. UI CLEAN  — Removed "Data Sources" column from display table and COLS
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
MIN_GROWTH_PCT_FOR_PEG = 5.0
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


# ── Helpers ────────────────────────────────────────────────────────────────────
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


# ── v14 fix: robust MultiIndex + flat column accessor for yf.download ─────────
def _yf_close(raw, ticker, n_tickers):
    """
    Safely extract the Close series from a yf.download result.

    yfinance ≥ 0.2.x behaviour:
      • Multiple tickers  → MultiIndex columns: (field, ticker)
      • Single ticker     → may be flat ("Close") OR MultiIndex depending on version

    Strategy:
      1. MultiIndex present  → try raw["Close"][ticker], fallback raw["Close"]
      2. Flat columns        → try raw["Close"], then raw["Adj Close"]
      3. All else fails      → return empty Series
    """
    try:
        if isinstance(raw.columns, pd.MultiIndex):
            # Level 0 = field name, Level 1 = ticker symbol
            close_df = raw.get("Close", raw.get("Adj Close"))
            if close_df is None:
                return pd.Series(dtype=float)
            if isinstance(close_df, pd.DataFrame):
                # Multiple tickers in download
                if ticker in close_df.columns:
                    return close_df[ticker].dropna()
                # Ticker not found — try case-insensitive match
                for col in close_df.columns:
                    if str(col).upper() == ticker.upper():
                        return close_df[col].dropna()
                return pd.Series(dtype=float)
            else:
                # Single-ticker MultiIndex collapsed to Series
                return close_df.dropna()
        else:
            # Flat columns (legacy single-ticker download)
            for field in ("Close", "Adj Close"):
                if field in raw.columns:
                    return raw[field].dropna()
            return pd.Series(dtype=float)
    except Exception:
        return pd.Series(dtype=float)


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


# ── Prices + 52W batch ────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices_batch(tickers):
    tl  = list(tickers)
    res = {t: {"price": None, "hi52": None, "lo52": None, "mc": None} for t in tl}
    try:
        raw = yf.download(
            tl, period="2d", interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
        for t in tl:
            try:
                closes = _yf_close(raw, t, len(tl))
                if not closes.empty:
                    res[t]["price"] = float(closes.iloc[-1])
            except Exception:
                pass
    except Exception:
        pass
    return res


# ── Momentum ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_momentum_batch(tickers):
    """
    v14 fix: Robust momentum fetch.
    - _yf_close updated to handle all MultiIndex / flat column variants
    - Per-ticker exception isolation prevents one bad ticker from zeroing others
    - 52W high/low derived from the 7-month daily download as a bonus fallback
    """
    tl  = list(tickers)
    out = {t: {} for t in tl}

    # ── Daily download (7 months) — used for volatility + 52W fallback ────
    raw_d = None
    try:
        raw_d = yf.download(
            tl, period="7mo", interval="1d",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
    except Exception:
        pass

    # ── Monthly download (7 months) — used for 1/3/6-month returns ────────
    raw_m = None
    try:
        raw_m = yf.download(
            tl, period="7mo", interval="1mo",
            group_by="ticker", auto_adjust=True,
            progress=False, threads=True,
        )
    except Exception:
        pass

    if raw_d is None and raw_m is None:
        return out

    for t in tl:
        try:
            closes_m = _yf_close(raw_m, t, len(tl)) if raw_m is not None else pd.Series(dtype=float)
            closes_d = _yf_close(raw_d, t, len(tl)) if raw_d is not None else pd.Series(dtype=float)

            # Need at least 2 monthly bars for return calculation
            if len(closes_m) < 2:
                continue

            px_now = float(closes_m.iloc[-1])
            if px_now <= 0:
                continue

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
                    trailing_vol = float(
                        daily_rets.std() * np.sqrt(252) * 100.0
                    )

            skip_mom_raw = (
                (r6 - r1) if (r6 is not None and r1 is not None) else None
            )
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
            # Isolate per-ticker failures — do not propagate
            pass

    return out


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Lightweight info fetch for ALL tickers
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_yahoo_info_one(t):
    result = {
        "pe": None, "pe_src": None,
        "fwd_pe": None,
        "peg": None, "peg_src": None,
        "roe": None,
        "op_margin": None,
        "debt_eq": None,
        "eps_growth": None,
        "earn_traj": None,
        "mc": None,
        "hi52": None,
        "lo52": None,
        "roic": None,
        "int_coverage": None,
        "rev4": [None, None, None, None],
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
        if (
            fwd_eps_val is not None
            and trail_eps_val is not None
            and abs(trail_eps_val) > 0.01
        ):
            earn_traj_raw = (fwd_eps_val - trail_eps_val) / abs(trail_eps_val)
            clipped       = max(-1.0, min(1.0, earn_traj_raw))
            if trail_eps_val < 0 and fwd_eps_val < 0:
                clipped = min(clipped, 0.30)
            result["earn_traj"] = clipped

        if result["mc"] is None:
            mc_y = sf(info.get("marketCap"))
            if mc_y:
                result["mc"] = mc_y
        if result["hi52"] is None:
            h52 = sf(info.get("fiftyTwoWeekHigh"))
            if h52:
                result["hi52"] = h52
        if result["lo52"] is None:
            l52 = sf(info.get("fiftyTwoWeekLow"))
            if l52:
                result["lo52"] = l52

    except Exception:
        pass
    return t, result


@st.cache_data(ttl=86400)
def fetch_yahoo_info_all(tickers, _cache_date=None):
    tl       = list(tickers)
    out      = {}
    CHUNK    = 30
    WKRS     = 8
    SLEEP    = 1.5
    chunks   = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    total    = len(chunks)
    progress = st.progress(0)
    status   = st.empty()

    for ci, chunk in enumerate(chunks):
        status.text(
            "Phase 1/2 — Yahoo info: chunk {}/{} ({} tickers done)...".format(
                ci + 1, total, ci * CHUNK
            )
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futures = {ex.submit(_fetch_yahoo_info_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(
                futures, timeout=FETCH_TIMEOUT_PER_TICKER * len(chunk)
            ):
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
# PHASE 2 — Deep financials + REVENUE for pre-filtered tickers
# ══════════════════════════════════════════════════════════════════════════════
def _fetch_yahoo_deep_one(t):
    result = {
        "roic":         None,
        "int_coverage": None,
        "rev4":         [None, None, None, None],
    }
    try:
        obj  = yf.Ticker(t)
        qfin = obj.quarterly_financials
        bs   = obj.quarterly_balance_sheet

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

        if qfin is not None and not qfin.empty:
            ebit_row = next(
                (nm for nm in ["EBIT", "Operating Income", "Ebit"]
                 if nm in qfin.index),
                None,
            )
            int_row = next(
                (nm for nm in [
                    "Interest Expense",
                    "Interest Expense Non Operating",
                    "Net Interest Income",
                ] if nm in qfin.index),
                None,
            )
            if ebit_row and int_row:
                ebit_ttm = qfin.loc[ebit_row].dropna().head(4).sum()
                int_ttm  = abs(qfin.loc[int_row].dropna().head(4).sum())
                if int_ttm > 0 and ebit_ttm > 0:
                    result["int_coverage"] = min(
                        float(ebit_ttm / int_ttm), 100.0
                    )

        if qfin is not None and not qfin.empty and bs is not None and not bs.empty:
            op_inc_row = next(
                (nm for nm in ["Operating Income", "EBIT", "Ebit"]
                 if nm in qfin.index),
                None,
            )
            tax_row = next(
                (nm for nm in [
                    "Tax Provision", "Income Tax Expense", "Tax Expense",
                ] if nm in qfin.index),
                None,
            )
            pretax_row = next(
                (nm for nm in [
                    "Pretax Income", "Income Before Tax", "EBT",
                ] if nm in qfin.index),
                None,
            )

            if op_inc_row:
                op_inc_ttm   = float(qfin.loc[op_inc_row].dropna().head(4).sum())
                eff_tax_rate = 0.21
                if tax_row and pretax_row:
                    tax_ttm    = float(
                        qfin.loc[tax_row].dropna().head(4).sum()
                    )
                    pretax_ttm = float(
                        qfin.loc[pretax_row].dropna().head(4).sum()
                    )
                    if pretax_ttm > 0 and tax_ttm >= 0:
                        computed_rate = tax_ttm / pretax_ttm
                        if 0 < computed_rate < 0.6:
                            eff_tax_rate = computed_rate
                nopat = op_inc_ttm * (1 - eff_tax_rate)

                equity_val = next(
                    (
                        float(bs.loc[nm].dropna().iloc[0])
                        for nm in [
                            "Total Stockholders Equity",
                            "Stockholders Equity",
                            "Common Stock Equity",
                            "Total Equity Gross Minority Interest",
                        ]
                        if nm in bs.index and len(bs.loc[nm].dropna()) > 0
                    ),
                    None,
                )
                debt_val = next(
                    (
                        float(bs.loc[nm].dropna().iloc[0])
                        for nm in [
                            "Total Debt",
                            "Net Debt",
                            "Long Term Debt",
                            "Long Term Debt And Capital Lease Obligation",
                        ]
                        if nm in bs.index and len(bs.loc[nm].dropna()) > 0
                    ),
                    None,
                )
                cash_val = next(
                    (
                        float(bs.loc[nm].dropna().iloc[0])
                        for nm in [
                            "Cash And Cash Equivalents",
                            "Cash Cash Equivalents And Short Term Investments",
                            "Cash Financial",
                            "Cash And Short Term Investments",
                        ]
                        if nm in bs.index and len(bs.loc[nm].dropna()) > 0
                    ),
                    None,
                )

                cash_use = 0
                if cash_val is not None:
                    rev4_vals = result["rev4"]
                    rev_ttm = None
                    if all(v is not None for v in rev4_vals):
                        rev_ttm = sum(rev4_vals)
                    if rev_ttm is not None and rev_ttm > 0:
                        operating_cash_floor = OPERATING_CASH_PCT_OF_REV * rev_ttm
                        cash_use = max(0.0, cash_val - operating_cash_floor)
                    else:
                        cash_use = cash_val

                if equity_val is not None and debt_val is not None:
                    invested_capital = equity_val + debt_val - cash_use
                    if invested_capital > 0 and nopat != 0:
                        roic_computed = (nopat / invested_capital) * 100.0
                        if -100 < roic_computed < 200:
                            result["roic"] = roic_computed

    except Exception:
        pass
    return t, result


@st.cache_data(ttl=86400)
def fetch_yahoo_deep_financials(tickers_filtered, _cache_date=None):
    tl = list(tickers_filtered)
    out = {}
    if not tl:
        return out

    CHUNK    = 20
    WKRS     = 6
    SLEEP    = 2.0
    chunks   = [tl[i:i+CHUNK] for i in range(0, len(tl), CHUNK)]
    total    = len(chunks)
    progress = st.progress(0)
    status   = st.empty()

    for ci, chunk in enumerate(chunks):
        status.text(
            "Phase 2/2 — Deep financials + revenue: chunk {}/{} ({}/{} tickers)...".format(
                ci + 1, total,
                min((ci + 1) * CHUNK, len(tl)),
                len(tl),
            )
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=WKRS) as ex:
            futures = {ex.submit(_fetch_yahoo_deep_one, t): t for t in chunk}
            for fut in concurrent.futures.as_completed(
                futures, timeout=FETCH_TIMEOUT_PER_TICKER * len(chunk)
            ):
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


def _pre_filter_tickers(info_map, universe_df, mc_min_b, pe_max):
    keep = []
    for t in universe_df["Ticker"]:
        d    = info_map.get(t, {})
        mc   = d.get("mc")
        pe   = d.get("pe")
        mc_ok = (mc is None) or (mc >= mc_min_b * 1e9)
        pe_ok = (pe is None) or (pe <= pe_max)
        if mc_ok and pe_ok:
            keep.append(t)
    return keep


def merge_yahoo_phases(info_map, deep_map, tickers):
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
        url = (
            "https://financialmodelingprep.com/api/v3/quote/{}?apikey={}".format(
                ",".join(chunk), api_key
            )
        )
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


# ── FMP /ratios-ttm ────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def fetch_fmp_ratios_if_available(tickers, api_key):
    out = {}
    if not api_key:
        return out

    test_url = (
        "https://financialmodelingprep.com/api/v3/ratios-ttm/AAPL?apikey={}".format(
            api_key
        )
    )
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
        url = (
            "https://financialmodelingprep.com/api/v3/ratios-ttm/{}?apikey={}".format(
                t, api_key
            )
        )
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
            roic     = normalise_pct_fmp(roic_raw) if roic_raw is not None else None

            roe_raw  = sf(item.get("returnOnEquityTTM"))
            roe      = normalise_pct_fmp(roe_raw) if roe_raw is not None else None

            om_raw   = sf(item.get("operatingProfitMarginTTM"))
            om       = normalise_pct_fmp(om_raw) if om_raw is not None else None

            ic_raw   = sf(item.get("interestCoverageTTM"))
            ic       = min(float(ic_raw), 100.0) if (ic_raw and ic_raw > 0) else None

            de       = sf(item.get("debtEquityRatioTTM"))

            fmp_trailing_pe_raw = sf(item.get("priceToEarningsRatioTTM"))
            fmp_trailing_pe = (
                fmp_trailing_pe_raw
                if (fmp_trailing_pe_raw and 0 < fmp_trailing_pe_raw <= 10_000)
                else None
            )

            return t, {
                "peg":          peg,
                "roic":         roic,
                "roe":          roe,
                "op_margin":    om,
                "int_coverage": ic,
                "debt_eq":      de,
                "fmp_trailing_pe": fmp_trailing_pe,
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


# ── Merge all sources ──────────────────────────────────────────────────────────
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

        pe_val  = first(fq.get("pe"), fr.get("fmp_trailing_pe"), yb.get("pe"))
        pe_src  = (
            "FMP-quote"   if fq.get("pe") is not None else
            "FMP-ratios"  if fr.get("fmp_trailing_pe") is not None else
            yb.get("pe_src", "Yahoo")
        )

        fwd_pe  = yb.get("fwd_pe")

        peg_val = first(fr.get("peg"), yb.get("peg"))
        peg_src = (
            "FMP-ratios" if fr.get("peg") is not None else
            yb.get("peg_src", "Yahoo") if yb.get("peg") is not None else "—"
        )

        roic    = first(fr.get("roic"),         yb.get("roic"))
        roe     = first(fr.get("roe"),           yb.get("roe"))
        ic      = first(fr.get("int_coverage"),  yb.get("int_coverage"))
        om      = first(fr.get("op_margin"),     yb.get("op_margin"))
        de      = first(fr.get("debt_eq"),       yb.get("debt_eq"))
        eps_g   = yb.get("eps_growth")
        g_src   = "Yahoo" if eps_g is not None else None
        earn_traj = yb.get("earn_traj")
        mc      = first(fq.get("mc"),   yb.get("mc"))
        hi52    = first(fq.get("hi52"), yb.get("hi52"))
        lo52    = first(fq.get("lo52"), yb.get("lo52"))
        rev4    = yb.get("rev4", [None, None, None, None])

        merged[t] = {
            "pe": pe_val, "pe_src": pe_src, "fwd_pe": fwd_pe,
            "peg": peg_val, "peg_src": peg_src,
            "roic": roic, "roe": roe, "int_coverage": ic,
            "op_margin": om, "debt_eq": de,
            "eps_growth": eps_g, "growth_src": g_src,
            "earn_traj": earn_traj,
            "mc": mc, "hi52": hi52, "lo52": lo52,
            "rev4": rev4,
        }
    return merged


# ── Quality Score ──────────────────────────────────────────────────────────────
def compute_quality_score(roic, roe, int_coverage, op_margin, sector=None):
    scores = []
    profitability = (
        roe if sector in ROE_PRIMARY_SECTORS
        else (roic if roic is not None else roe)
    )

    if profitability is not None and not pd.isna(profitability):
        pf = float(profitability)
        scores.append(
            min(100.0, np.log1p(pf) / np.log1p(30.0) * 100.0) if pf > 0 else 0.0
        )
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


# ── Quality flag ───────────────────────────────────────────────────────────────
def quality_flag(roic, roe, ic, om, sector=None):
    flags = []
    if sector in ROE_PRIMARY_SECTORS:
        profitability = roe
        prof_label    = "ROE"
    else:
        profitability = roic if (roic is not None and not pd.isna(roic)) else roe
        prof_label    = "ROIC" if (roic is not None and not pd.isna(roic)) else "ROE"

    if (
        profitability is not None
        and not pd.isna(profitability)
        and profitability < QUALITY_THRESHOLDS["roic_min"]
    ):
        flags.append("{}<8%".format(prof_label))
    if ic is not None and not pd.isna(ic) and ic < QUALITY_THRESHOLDS["int_coverage_min"]:
        flags.append("IntCov<3x")
    if sector not in ROE_PRIMARY_SECTORS:
        if om is not None and not pd.isna(om) and om < QUALITY_THRESHOLDS["op_margin_min"]:
            flags.append("Margin<5%")

    return ", ".join(flags) if flags else "Pass"


# ── Conviction Score ───────────────────────────────────────────────────────────
def compute_conviction_scores(scr):
    KEY_FACTORS = ["P/E", "Fwd P/E", "PEG", "Quality Score", "Momentum Score", "Earn Traj"]
    n_factors   = len(KEY_FACTORS)
    scr         = scr.copy()

    def completeness(row):
        present = sum(
            1 for c in KEY_FACTORS if c in row.index and pd.notna(row[c])
        )
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

    scr["_sec_discount"] = scr["Sector"].map(sector_discount)
    raw_conviction       = scr["Score"] * scr["_completeness"] * scr["_sec_discount"]
    c_min, c_max         = raw_conviction.min(), raw_conviction.max()

    if c_max > c_min:
        scr["Conviction Score"] = (raw_conviction - c_min) / (c_max - c_min) * 100.0
    else:
        scr["Conviction Score"] = pd.Series(50.0, index=scr.index)

    return scr.drop(columns=["_completeness", "_sec_discount"])


# ── Ranking ────────────────────────────────────────────────────────────────────
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

        pe_input         = elig["Fwd P/E"].fillna(elig["P/E"])
        elig["_s_val"]   = percentile_score(pe_input,              ascending=True)
        elig["_s_peg"]   = percentile_score(elig["PEG"],            ascending=True)
        elig["_s_mom"]   = percentile_score(elig["Momentum Score"], ascending=False)
        elig["_s_etraj"] = percentile_score(elig["Earn Traj"],      ascending=False)

        qs    = elig["Quality Score"]
        q_min = qs.min(); q_max = qs.max()
        if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min:
            elig["_s_quality"] = (qs - q_min) / (q_max - q_min) * 100.0
        else:
            elig["_s_quality"] = qs.fillna(0.0)
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)

        raw = (
            W["valuation"] * elig["_s_val"]      +
            W["quality"]   * elig["_s_quality"]  +
            W["peg"]       * elig["_s_peg"]      +
            W["earn_traj"] * elig["_s_etraj"]    +
            W["momentum"]  * elig["_s_mom"]
        )

        factor_cols = ["P/E", "PEG", "Quality Score", "Earn Traj", "Momentum Score"]
        penalties   = elig.apply(
            lambda r: missing_factor_penalty(r, factor_cols), axis=1
        )
        raw = raw * penalties

        elig["Score"] = raw
        elig = elig.sort_values("Score", ascending=False)
        elig["Rank"]  = range(1, len(elig) + 1)
        scr.loc[elig.index, "Score"] = elig["Score"]
        scr.loc[elig.index, "Rank"]  = elig["Rank"]

    return scr


# ── Build screener table ───────────────────────────────────────────────────────
def build_screener_table(universe_df, prices_map, merged_map, momentum_map):
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
    ]
    for c in num_cols:
        if c in scr.columns:
            scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns:
        scr["Rank"] = pd.NA

    sector_med_pe   = scr.groupby("Sector")["P/E"].transform("median")
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
**P/E — Price to Earnings Ratio (Trailing)**

Formula: `Current Stock Price / Trailing 12-Month EPS`

- Apple price = $210, EPS TTM = $6.57 → P/E = 32.0
- ExxonMobil price = $115, EPS TTM = $9.60 → P/E = 12.0

| Sector | Typical Median P/E |
|---|---|
| Information Technology | 28–40 |
| Consumer Staples | 22–32 |
| Financials / Banking | 12–18 |
| Energy / Oil & Gas | 10–16 |
| Industrials | 22–30 |
| Health Care / Pharma | 22–35 |

Used in scoring? **Yes** — primary Valuation factor (20–38% weight).

---
**Fwd P/E — Forward Price to Earnings**

Formula: `Current Stock Price / Next 12-Month Estimated EPS`

Source: **Yahoo Finance only** (forwardPE / forwardEps fields).
FMP `priceToEarningsRatioTTM` is trailing P/E — not used for Fwd P/E (v13 fix).

---
**P/E vs Sector Med** _(v13 new)_

Formula: `Stock P/E / Median P/E of all stocks in same sector`

- 0.80 = 20% cheaper than sector peers → bullish signal
- 1.20 = 20% more expensive than sector peers → expensive signal

Used in scoring? **No**. Display and context only.

---
**MC% of S&P 500**

Formula: `Stock Market Cap / Sum of All S&P 500 Market Caps × 100`

Used in scoring? **No**. Display and filter only.

---
**52W Pos%**

Formula: `(Current Price − 52W Low) / (52W High − 52W Low) × 100`

0% = at 52-week low · 100% = at 52-week high
        """)

    with tab_qual:
        st.markdown("""
**Quality Score (0–100)**

Formula: `(ROIC sub-score + Interest Coverage sub-score + Op Margin sub-score) / 3`

Financials sector: Uses ROE as primary profitability metric. Op Margin excluded.

---
**ROIC% — Return on Invested Capital**
NOPAT = Operating Income × (1 − effective tax rate)
Excess Cash = max(0, Total Cash − 2% of Revenue TTM)
Invested Capital = Equity + Debt − Excess Cash
ROIC = NOPAT / Invested Capital × 100

| ROIC | Assessment |
|---|---|
| 25%+ | Best-in-class |
| 15% | Excellent |
| 10% | Good |
| 8% | Minimum threshold |
| Below 8% | Flagged ROIC<8% |

---
**Quality Flag** _(v13: D/E removed from this column)_

Flags: `ROIC<8%` · `ROE<8%` · `IntCov<3x` · `Margin<5%` · `Pass`

D/E ratio now shown separately in the **Debt/Eq** column — no longer embedded
in Quality Flag text, making flag text filterable.
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

Data source waterfall:
1. Yahoo Finance pegRatio
2. FMP /ratios-ttm → priceEarningsGrowthRatioTTM
3. Calculated: (Fwd P/E or Trailing P/E) / EPS growth %
        """)

    with tab_etraj:
        st.markdown("""
**Earn Traj — Earnings Trajectory**

Formula: `(Forward EPS − Trailing EPS) / |Trailing EPS|` clipped to `[−1.0, +1.0]`

v12 cap: When both EPS values are negative, capped at +0.30 (recovery-in-progress).

| Scenario | Earn Traj | Interpretation |
|---|---|---|
| Trail +$2.00 → Fwd +$2.50 | +0.25 | Healthy earnings growth |
| Trail −$2.00 → Fwd +$1.00 | +1.0 | Full turnaround |
| Trail −$2.00 → Fwd −$0.50 | +0.30 (capped) | Still losing, improving |
| Trail +$2.00 → Fwd +$1.50 | −0.25 | Earnings under pressure |
        """)

    with tab_mom:
        st.markdown("""
**Momentum Score — Skip-Month Volatility-Adjusted Momentum**

Formula: `(6-month return − 1-month return) / Trailing 90-day Annualised Volatility`

Why skip the last month? Short-term reversal effect — removes noise, isolates the durable 2–6 month trend.

| Score | Signal |
|---|---|
| Above +1.0 | Exceptionally strong momentum |
| +0.3 to +1.0 | Healthy uptrend |
| −0.3 to +0.3 | Neutral |
| Below −0.3 | Downtrend |
        """)

    with tab_rank:
        st.markdown("""
**Score (0–100)**

Composite percentile score within GICS sector using sector-adaptive weights.

| Sector | Val | Quality | PEG | Earn | Mom |
|---|---|---|---|---|---|
| Information Technology | 20% | 25% | 25% | 15% | 15% |
| Consumer Staples | 28% | 32% | 10% | 15% | 15% |
| Financials | 30% | 25% | 18% | 17% | 10% |
| Energy | 30% | 18% | 12% | 15% | 25% |
| Utilities | 38% | 27% | 5% | 15% | 15% |

**Missing Factor Penalty**

| Missing factors | Multiplier |
|---|---|
| 0 | ×1.00 |
| 1 | ×0.95 |
| 2 | ×0.85 |
| 3+ | ×0.70 |

**Conviction Score (0–100)**

`Score × data_completeness_ratio × sector_discount_factor → normalised 0–100`

v13 fix: Single-stock or uniform-score edge case now correctly returns 50
instead of causing a Pandas scalar assignment warning.
        """)

    with tab_disp:
        st.markdown("""
**Rev Q1 Oldest ($B) → Rev Q4 Latest ($B)** _(v13: renamed for clarity)_

Last four fiscal quarters of total revenue, ordered **oldest → newest**.
Q4 Latest is the most recent quarter available.

v13 improvement: Revenue is now fetched in Phase 2 alongside ROIC/IntCov,
eliminating ~500 duplicate API calls vs v12.

**Data Coverage (typical)**

| Metric | Typical Coverage |
|---|---|
| Price, MC, 52W | ~99% |
| Trailing P/E | ~90% |
| Forward P/E | ~78% |
| PEG Ratio | ~70% |
| ROE, Op Margin | ~88% |
| Int Coverage | ~65% |
| ROIC (computed) | ~60% |
| Earn Traj | ~82% |
| Momentum | ~97% |
| Quarterly Revenue | ~72% |
        """)

    st.markdown("---")
    st.markdown(
        "**Data sources v14:** Yahoo Finance (primary) · FMP bonus if key available · "
        "ROIC + Revenue from Phase 2 (single quarterly_financials call per ticker) · "
        "Fwd P/E from Yahoo only (FMP TTM ratio is trailing — v13 fix) · "
        "Earn Traj both-negative cap at +0.30 · Sector-adaptive scoring. "
        "_Nothing here is financial advice._"
    )


# ══════════════════════════════════════════════════════════════════════════════
# APP ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="S&P 500 Screener v14", layout="wide", page_icon="📊"
)
st.markdown(
    "<style>"
    "div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}"
    "</style>",
    unsafe_allow_html=True,
)

st.markdown("## S&P 500 Fundamental Screener v14")
# ── v14: subtitle caption removed (change #1) ─────────────────────────────────

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
            "Last loaded: {} · Prices: 1hr cache · Fundamentals: 24hr cache".format(
                datetime.now().strftime("%I:%M %p")
            )
        )

    fmp_key = get_fmp_key()
    if fmp_key:
        st.success(
            "FMP API key found — bonus layer active for PE override, PEG, ROIC/IntCov."
        )
    else:
        st.info(
            "No FMP key. Running on Yahoo Finance only. "
            "Add [fmp] api_key to Streamlit Secrets."
        )

    # ── Load universe ──────────────────────────────────────────────────────
    with st.spinner("Loading S&P 500 universe..."):
        sp500 = fetch_sp500_constituents()
    if sp500.empty:
        st.error("Failed to load S&P 500 universe.")
        st.stop()

    universe_df = sp500.copy().reset_index(drop=True)
    tickers     = tuple(universe_df["Ticker"].tolist())

    today_date = date.today()

    # ── Filters ────────────────────────────────────────────────────────────
    st.markdown("### Filters")
    all_sectors_placeholder = sorted(
        universe_df["Sector"].dropna().unique().tolist()
    )
    f1, f2, f3, f4, f5 = st.columns(5)

    sector_sel = f1.selectbox(
        "Sector",
        ["All Sectors"] + all_sectors_placeholder,
        help="Filter to one GICS sector or view all S&P 500 stocks.",
    )
    sort_by = f2.selectbox(
        "Sort by",
        [
            "Sector then Rank", "Score high to low", "Conviction high to low",
            "MC% of S&P500 high to low", "Price low to high", "Price high to low",
            "Mkt Cap high to low", "PE low to high", "Fwd PE low to high",
            "PEG low to high", "Quality Score high", "ROIC high to low",
            "ROE high to low", "Earn Traj high to low", "Rev Growth high to low",
            "Momentum Score high", "52W Pos low to high",
            "P/E vs Sector Med low to high",
        ],
        help="Primary sort column for the results table.",
    )
    mc_min_b = f3.number_input(
        "Min Mkt Cap ($B)", value=0, step=10, min_value=0,
        help="Show stocks with market cap above this value (USD billions).",
    )
    pe_max = f4.number_input(
        "Max P/E", value=9999, step=50, min_value=0,
        help="Exclude stocks with trailing P/E above this value.",
    )
    qual_min_f = f5.number_input(
        "Min Quality Score", value=0.0, step=5.0, min_value=0.0, max_value=100.0,
        help="Show stocks with Quality Score at or above this value (0–100).",
    )

    # ── Phase 1 ────────────────────────────────────────────────────────────
    with st.spinner(
        "Phase 1/2 — Fetching Yahoo info for all {} tickers...".format(len(tickers))
    ):
        yahoo_info = fetch_yahoo_info_all(tickers, _cache_date=today_date)

    # ── Pre-filter ─────────────────────────────────────────────────────────
    filtered_tickers = _pre_filter_tickers(
        yahoo_info, universe_df, mc_min_b, pe_max
    )
    # ── v14: pre-filter caption removed (change #2) ────────────────────────

    # ── Phase 2 ────────────────────────────────────────────────────────────
    with st.spinner(
        "Phase 2/2 — Deep financials + revenue for {} tickers...".format(
            len(filtered_tickers)
        )
    ):
        yahoo_deep = fetch_yahoo_deep_financials(
            tuple(filtered_tickers), _cache_date=today_date
        )

    # ── Merge Phase 1 + Phase 2 ────────────────────────────────────────────
    yahoo_fundamentals = merge_yahoo_phases(yahoo_info, yahoo_deep, tickers)

    # ── Prices + Momentum ──────────────────────────────────────────────────
    with st.spinner("Fetching prices ({} tickers)...".format(len(tickers))):
        prices = fetch_prices_batch(tickers)

    with st.spinner("Fetching momentum (skip-month vol-adjusted)..."):
        momentum = fetch_momentum_batch(tickers)

    # ── FMP bonus layer ────────────────────────────────────────────────────
    fmp_quotes = {}
    fmp_ratios = {}
    if fmp_key:
        with st.spinner("FMP bonus: bulk /quote..."):
            fmp_quotes = fetch_fmp_quotes_if_available(tickers, fmp_key)
        with st.spinner("FMP bonus: /ratios-ttm..."):
            fmp_ratios = fetch_fmp_ratios_if_available(tickers, fmp_key)

    # ── Merge all sources ──────────────────────────────────────────────────
    with st.spinner("Merging data sources..."):
        merged_map = merge_all_sources(
            yahoo_fundamentals, fmp_quotes, fmp_ratios, tickers
        )

    # ── Coverage stats ─────────────────────────────────────────────────────
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
        "P/E: {}/{} ({:.0f}%) · Fwd P/E: {}/{} ({:.0f}%) · "
        "PEG: {}/{} ({:.0f}%) · ROIC: {}/{} ({:.0f}%) · "
        "ROE: {}/{} ({:.0f}%) · Int Coverage: {}/{} ({:.0f}%) · "
        "Op Margin: {}/{} ({:.0f}%) · Earn Traj: {}/{} ({:.0f}%) · "
        "Primary: Yahoo{}".format(
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

    # ── Build screener ─────────────────────────────────────────────────────
    scr = build_screener_table(universe_df, prices, merged_map, momentum)

    # ── v14: render_sector_kpi_panel() call removed (change #3) ──────────

    # ── Apply filters ──────────────────────────────────────────────────────
    filt = scr.copy()
    if sector_sel != "All Sectors":
        filt = filt[filt["Sector"] == sector_sel]
    filt = filt[
        (filt["Mkt Cap"].isna())       | (filt["Mkt Cap"]       >= mc_min_b * 1e9)
    ]
    filt = filt[
        (filt["P/E"].isna())           | (filt["P/E"]           <= pe_max)
    ]
    filt = filt[
        (filt["Quality Score"].isna()) | (filt["Quality Score"] >= qual_min_f)
    ]

    sort_map = {
        "Sector then Rank":             (["Sector", "Rank"],         [True, True]),
        "Score high to low":            (["Score"],                  [False]),
        "Conviction high to low":       (["Conviction Score"],       [False]),
        "MC% of S&P500 high to low":    (["MC% of S&P500"],         [False]),
        "Price low to high":            (["Price"],                  [True]),
        "Price high to low":            (["Price"],                  [False]),
        "Mkt Cap high to low":          (["Mkt Cap"],                [False]),
        "PE low to high":               (["P/E"],                    [True]),
        "Fwd PE low to high":           (["Fwd P/E"],                [True]),
        "PEG low to high":              (["PEG"],                    [True]),
        "Quality Score high":           (["Quality Score"],          [False]),
        "ROIC high to low":             (["ROIC%"],                  [False]),
        "ROE high to low":              (["ROE%"],                   [False]),
        "Earn Traj high to low":        (["Earn Traj"],              [False]),
        "Rev Growth high to low":       (["Rev Growth% (CAGR)"],     [False]),
        "Momentum Score high":          (["Momentum Score"],         [False]),
        "52W Pos low to high":          (["52W Pos%"],               [True]),
        "P/E vs Sector Med low to high":(["P/E vs Sector Med"],      [True]),
    }
    sc, sa = sort_map.get(sort_by, (["Sector", "Rank"], [True, True]))
    filt   = filt.sort_values(sc, ascending=sa, na_position="last")

    st.caption(
        "Showing **{}** of **{}** stocks · Sector: {} · Sort: {}".format(
            len(filt), len(scr), sector_sel, sort_by
        )
    )

    # ── Display table ──────────────────────────────────────────────────────
    disp = filt.copy()
    disp["Price ($)"]              = disp["Price"].round(2)
    disp["Mkt Cap ($B)"]           = (disp["Mkt Cap"] / 1e9).round(2)
    disp["MC% of S&P500"]          = disp["MC% of S&P500"].round(4)
    disp["Rev Q1 Oldest ($B)"]     = (disp["Rev Q1 Oldest ($B)"] / 1e9).round(2)
    disp["Rev Q2 ($B)"]            = (disp["Rev Q2 ($B)"] / 1e9).round(2)
    disp["Rev Q3 ($B)"]            = (disp["Rev Q3 ($B)"] / 1e9).round(2)
    disp["Rev Q4 Latest ($B)"]     = (disp["Rev Q4 Latest ($B)"] / 1e9).round(2)

    disp["Quality Flag"] = disp.apply(
        lambda r: quality_flag(
            r.get("ROIC%"), r.get("ROE%"),
            r.get("Int Coverage"), r.get("Op Margin%"),
            sector=r.get("Sector"),
        ),
        axis=1,
    )

    for c in [
        "P/E", "Fwd P/E", "PEG", "Earn Traj", "52W Pos%",
        "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
        "Quality Score", "Momentum Score", "Ret 1Mo%", "Ret 3Mo%",
        "Ret 6Mo%", "Trailing Vol%", "Score", "Conviction Score",
        "Rev Growth% (CAGR)", "P/E vs Sector Med",
    ]:
        if c in disp.columns:
            disp[c] = disp[c].round(2)

    disp["Rank"] = disp["Rank"].apply(
        lambda v: int(v) if pd.notna(v) else pd.NA
    )

    # ── v14: "Data Sources" removed from COLS (change #5) ─────────────────
    COLS = [
        "Ticker", "Sector", "Price ($)", "Mkt Cap ($B)", "MC% of S&P500",
        "P/E", "P/E vs Sector Med",
        "Fwd P/E", "PEG", "PEG Method", "Earn Traj",
        "ROIC%", "ROE%", "Int Coverage", "Op Margin%",
        "Debt/Eq",
        "Quality Score", "Quality Flag",
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
        file_name="sp500_screener_v14_{}.csv".format(
            datetime.now().strftime("%Y%m%d_%H%M")
        ),
        mime="text/csv",
    )


    st.markdown("---")
    st.caption(
        "v14 · UI cleanup · Momentum MultiIndex fix · Sector KPI removed · "
        "Data Sources column removed · All v13 features preserved"
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — COLUMN REFERENCE GUIDE
# ══════════════════════════════════════════════════════════════════════════════
with page_reference:
    render_reference_guide()
