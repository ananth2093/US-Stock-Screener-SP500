# app.py  (Nifty 50 Screener v9 — sector-adaptive factor weights)
# Built on v7 base (column name consistency fix)
# v9 changes:
#   1. SECTOR_FACTOR_WEIGHTS + DEFAULT_FACTOR_WEIGHTS replace FACTOR_WEIGHTS
#   2. ROE_PRIMARY_SECTORS constant — Financials uses ROE, not ROA
#   3. compute_quality_score() gains sector= param (Financials skips Op Margin)
#   4. compute_rank_by_sector() pulls per-sector weights (one line)
#   5. render_sector_kpi_panel() shows active weights bar per sector
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import time
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

MIN_GROWTH_PCT_FOR_PEG = 5.0

# ── CHANGE 1 ─────────────────────────────────────────────────────────────────
# Replaced single FACTOR_WEIGHTS dict with SECTOR_FACTOR_WEIGHTS (11 sectors)
# + DEFAULT_FACTOR_WEIGHTS fallback. All rows sum to 1.0.
# Rationale: each sector has structurally different value drivers.
# ─────────────────────────────────────────────────────────────────────────────

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


# ── CHANGE 2 ─────────────────────────────────────────────────────────────────
# ROE_PRIMARY_SECTORS: these sectors use ROE (not ROA) as primary quality
# metric, and Op Margin is excluded from their quality composite.
# Reason: banks' ROA is structurally 1-2% (excellent for a bank but our
# scoring penalises it); Op Margin is meaningless for interest businesses.
# ─────────────────────────────────────────────────────────────────────────────

ROE_PRIMARY_SECTORS = {"Financials"}


QUALITY_THRESHOLDS = {
    "roic_min":         8.0,
    "int_coverage_min": 3.0,
    "op_margin_min":    5.0,
}

SECTOR_MAP = {
    "Financial Services":                "Financials",
    "Banking":                           "Financials",
    "Insurance":                         "Financials",
    "Diversified Financials":            "Financials",
    "Information Technology":            "Information Technology",
    "IT":                                "Information Technology",
    "Oil Gas & Consumable Fuels":        "Energy",
    "Oil & Gas":                         "Energy",
    "Energy":                            "Energy",
    "Power":                             "Utilities",
    "Utilities":                         "Utilities",
    "Fast Moving Consumer Goods":        "Consumer Staples",
    "FMCG":                              "Consumer Staples",
    "Consumer Goods":                   "Consumer Staples",
    "Tobacco":                           "Consumer Staples",
    "Automobile":                        "Consumer Discretionary",
    "Automobile And Auto Components":    "Consumer Discretionary",
    "Consumer Durables":                "Consumer Discretionary",
    "Retailing":                         "Consumer Discretionary",
    "Construction":                      "Industrials",
    "Capital Goods":                     "Industrials",
    "Services":                          "Industrials",
    "Industrial Manufacturing":          "Industrials",
    "Infrastructure":                    "Industrials",
    "Ports & Shipping":                  "Industrials",
    "Metals & Mining":                   "Materials",
    "Metals":                            "Materials",
    "Mining":                            "Materials",
    "Cement & Cement Products":          "Materials",
    "Cement":                            "Materials",
    "Steel":                             "Materials",
    "Construction Materials":            "Materials",
    "Pharmaceuticals":                  "Health Care",
    "Healthcare":                        "Health Care",
    "Pharma":                            "Health Care",
    "Hospital & Diagnostic Centres":     "Health Care",
    "Telecommunication":                "Communication Services",
    "Telecom":                           "Communication Services",
    "Media Entertainment & Publication": "Communication Services",
    "Real Estate":                       "Real Estate",
    "Realty":                            "Real Estate",
}

# ── Canonical column names ── defined once, used everywhere ──────────────────
COL_MC  = "Mkt Cap (LCr)"
COL_RQ1 = "Rev Q1 (1000Cr)"
COL_RQ2 = "Rev Q2 (1000Cr)"
COL_RQ3 = "Rev Q3 (1000Cr)"
COL_RQ4 = "Rev Q4 (1000Cr)"

# ─── Helpers ──────────────────────────────────────────────────────────────────
def to_num(x):
    return pd.to_numeric(x, errors="coerce")

def sf(val):
    try:
        return float(val) if val is not None else None
    except Exception:
        return None

def fmt_mc_inr(val):
    """Format raw market cap (INR) as Rs Lakh Cr for the KPI panel."""
    if val is None or (isinstance(val, float) and pd.isna(val)) or val == 0:
        return "N/A"
    return "Rs.{:.2f}L Cr".format(val / 1e12)

def percentile_score(series, ascending=True):
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
    if missing >= 3:
        return 0.70
    if missing == 2:
        return 0.85
    return 1.0

def revenue_growth_yoy(rev4):
    try:
        if rev4 is None or len(rev4) != 4:
            return None
        q_newest = rev4[0]
        q_oldest = rev4[3]
        if q_newest is None or q_oldest is None:
            return None
        q_newest = float(q_newest)
        q_oldest = float(q_oldest)
        if q_newest <= 0 or q_oldest <= 0:
            return None
        return (q_newest / q_oldest - 1) * 100.0
    except Exception:
        return None

def decimal_to_pct(val):
    if val is None:
        return None
    v = float(val)
    if abs(v) <= 20.0:
        return v * 100.0
    return v

def safe_float(obj):
    if obj is None:
        return None
    if isinstance(obj, pd.Series):
        obj = obj.dropna()
        if obj.empty:
            return None
        obj = obj.iloc[0]
    try:
        f = float(obj)
        return None if np.isnan(f) else f
    except Exception:
        return None

def _extract_scalar(info, *keys, default=None):
    for k in keys:
        v = info.get(k)
        if v is not None:
            try:
                f = float(v)
                if not np.isnan(f):
                    return f
            except Exception:
                pass
    return default

# ─── Universe ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400)
def get_nifty50_universe():
    try:
        r = requests.get(
            "https://en.wikipedia.org/wiki/NIFTY_50",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )
        r.raise_for_status()
        soup  = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table", {"id": "constituents"})
        if table is None:
            for tbl in soup.find_all("table", {"class": "wikitable"}):
                hdrs = [th.get_text(strip=True).lower() for th in tbl.find_all("th")]
                if any("symbol" in h or "ticker" in h for h in hdrs):
                    table = tbl
                    break
        if table is None:
            table = soup.find("table", {"class": "wikitable sortable"})
        if table is None:
            for tbl in soup.find_all("table", {"class": "wikitable"}):
                if len(tbl.find_all("tr")) >= 30:
                    table = tbl
                    break
        if table is None:
            raise RuntimeError("Table not found")

        header_row = table.find("tr")
        headers = (
            [th.get_text(strip=True).lower() for th in header_row.find_all(["th", "td"])]
            if header_row else []
        )
        ticker_col = next(
            (i for i, h in enumerate(headers)
             if any(k in h for k in ["symbol", "ticker", "nse"])), 2)
        sector_col = next(
            (i for i, h in enumerate(headers)
             if any(k in h for k in ["sector", "industry", "gics"])), 1)

        data = []
        for row in table.find_all("tr")[1:]:
            cols  = row.find_all(["td", "th"])
            if len(cols) <= max(ticker_col, sector_col):
                continue
            raw_t = re.sub(r"$.*?$", "", cols[ticker_col].get_text(strip=True)).strip()
            raw_t = re.sub(r"[^A-Za-z0-9&\-]", "", raw_t).upper()
            raw_s = re.sub(r"$.*?$", "", cols[sector_col].get_text(strip=True)).strip()
            if not raw_t or len(raw_t) < 2:
                continue
            gics = SECTOR_MAP.get(raw_s)
            if gics is None:
                for nse_name, gics_name in SECTOR_MAP.items():
                    if nse_name.lower() in raw_s.lower() or raw_s.lower() in nse_name.lower():
                        gics = gics_name
                        break
            data.append({
                "Base":       raw_t,
                "Ticker":     raw_t + ".NS",
                "Sector":     gics or raw_s,
                "NSE Sector": raw_s,
            })

        if len(data) < 30:
            raise RuntimeError("Only {} rows".format(len(data)))

        df = pd.DataFrame(data).drop_duplicates(subset=["Ticker"])
        st.success("Universe: {} stocks from Wikipedia".format(len(df)))
        return df

    except Exception as e:
        st.warning("Wikipedia failed: {}. Using fallback list.".format(e))
        fallback = [
            ("RELIANCE",   "Energy"),
            ("TCS",        "Information Technology"),
            ("HDFCBANK",   "Financials"),
            ("INFY",       "Information Technology"),
            ("ICICIBANK",  "Financials"),
            ("HINDUNILVR", "Consumer Staples"),
            ("ITC",        "Consumer Staples"),
            ("SBIN",       "Financials"),
            ("BHARTIARTL", "Communication Services"),
            ("LT",         "Industrials"),
            ("KOTAKBANK",  "Financials"),
            ("AXISBANK",   "Financials"),
            ("WIPRO",      "Information Technology"),
            ("HCLTECH",    "Information Technology"),
            ("ASIANPAINT", "Materials"),
            ("MARUTI",     "Consumer Discretionary"),
            ("BAJFINANCE", "Financials"),
            ("TITAN",      "Consumer Discretionary"),
            ("SUNPHARMA",  "Health Care"),
            ("ULTRACEMCO", "Materials"),
        ]
        return pd.DataFrame([
            {"Base": b, "Ticker": b + ".NS", "Sector": s, "NSE Sector": s}
            for b, s in fallback
        ])

# ─── Quarterly financial helpers ──────────────────────────────────────────────
def _quarterly_revenues(ticker_obj):
    for attr in ("quarterly_income_stmt", "quarterly_financials"):
        try:
            df = getattr(ticker_obj, attr)
            if df is None or df.empty:
                continue
            for label in ["Total Revenue", "Revenue", "Net Revenue", "Operating Revenue"]:
                matches = [r for r in df.index if label.lower() in str(r).lower()]
                if matches:
                    row  = df.loc[matches[0]]
                    cols = sorted(row.index, reverse=True)[:4]
                    vals = [safe_float(row[c]) for c in cols]
                    while len(vals) < 4:
                        vals.append(None)
                    return vals[:4]
        except Exception:
            pass
    return [None, None, None, None]

def _quarterly_eps(ticker_obj):
    for attr in ("quarterly_income_stmt", "quarterly_financials"):
        try:
            df = getattr(ticker_obj, attr)
            if df is None or df.empty:
                continue
            for label in ["Basic EPS", "Diluted EPS", "EPS"]:
                matches = [r for r in df.index if label.lower() in str(r).lower()]
                if matches:
                    row  = df.loc[matches[0]]
                    cols = sorted(row.index, reverse=True)
                    eps_r = safe_float(row[cols[0]]) if len(cols) > 0 else None
                    eps_o = safe_float(row[cols[3]]) if len(cols) > 3 else None
                    return eps_r, eps_o
        except Exception:
            pass
    return None, None

def _interest_coverage_from_financials(ticker_obj):
    for attr in ("income_stmt", "financials"):
        try:
            df = getattr(ticker_obj, attr)
            if df is None or df.empty:
                continue
            ebit_row = None
            int_row  = None
            for label in ["EBIT", "Operating Income"]:
                matches = [r for r in df.index if label.lower() in str(r).lower()]
                if matches:
                    ebit_row = df.loc[matches[0]]
                    break
            for label in ["Interest Expense", "Interest Expense Non Operating",
                           "Net Interest Income", "Interest And Debt Expense"]:
                matches = [r for r in df.index if label.lower() in str(r).lower()]
                if matches:
                    int_row = df.loc[matches[0]]
                    break
            if ebit_row is not None and int_row is not None:
                cols = sorted(ebit_row.index, reverse=True)
                for col in cols[:2]:
                    ebit = safe_float(ebit_row[col])
                    iexp = safe_float(int_row[col])
                    if ebit is not None and iexp is not None and iexp != 0:
                        ic = ebit / abs(iexp)
                        if ic > 0:
                            return min(ic, 100.0)
        except Exception:
            pass
    return None

# ─── yfinance fundamentals ────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_yf_fundamentals(tickers):
    out = {t: {} for t in tickers}

    def one(t):
        try:
            ticker_obj = yf.Ticker(t)
            info       = ticker_obj.info or {}

            price  = _extract_scalar(info, "currentPrice", "regularMarketPrice", "previousClose")
            mc     = _extract_scalar(info, "marketCap")
            hi52   = _extract_scalar(info, "fiftyTwoWeekHigh")
            lo52   = _extract_scalar(info, "fiftyTwoWeekLow")
            pe     = _extract_scalar(info, "trailingPE")
            fwd_pe = _extract_scalar(info, "forwardPE")
            peg_yf = _extract_scalar(info, "pegRatio")

            roe_raw  = _extract_scalar(info, "returnOnEquity")
            roic_raw = _extract_scalar(info, "returnOnAssets")
            om_raw   = _extract_scalar(info, "operatingMargins")
            de_raw   = _extract_scalar(info, "debtToEquity")

            roe  = decimal_to_pct(roe_raw)
            roic = decimal_to_pct(roic_raw)
            om   = decimal_to_pct(om_raw)

            ic = _interest_coverage_from_financials(ticker_obj)
            if ic is None:
                ebitda  = _extract_scalar(info, "ebitda")
                int_exp = _extract_scalar(info, "interestExpense")
                if ebitda is not None and int_exp is not None and int_exp != 0:
                    ic = min(abs(ebitda / int_exp), 100.0)

            eps_r, eps_o = _quarterly_eps(ticker_obj)
            earn_traj    = None
            eps_growth   = None
            if eps_r is not None and eps_o is not None and abs(eps_o) > 0.001:
                raw        = (eps_r - eps_o) / abs(eps_o)
                earn_traj  = max(-1.0, min(1.0, raw / 2.0))
                eps_growth = raw * 100.0
            if earn_traj is None:
                eps_curr = _extract_scalar(info, "trailingEps")
                eps_fwd  = _extract_scalar(info, "forwardEps")
                if eps_curr is not None and eps_fwd is not None and abs(eps_curr) > 0.001:
                    raw        = (eps_fwd - eps_curr) / abs(eps_curr)
                    earn_traj  = max(-1.0, min(1.0, raw / 2.0))
                    eps_growth = raw * 100.0

            rev4 = _quarterly_revenues(ticker_obj)

            peg        = None
            peg_method = "N/A"
            if peg_yf is not None and 0 < peg_yf <= 500:
                peg        = peg_yf
                peg_method = "yfinance"
            else:
                pe_for_peg = fwd_pe if fwd_pe is not None else pe
                if eps_growth is not None and pe_for_peg is not None:
                    eg = float(eps_growth)
                    if eg >= MIN_GROWTH_PCT_FOR_PEG:
                        peg        = pe_for_peg / eg
                        peg_method = "Calc"
            if peg is not None and (peg <= 0 or peg > 500):
                peg = None

            return t, {
                "price":        price,
                "mc":           mc,
                "hi52":         hi52,
                "lo52":         lo52,
                "pe":           pe     if (pe     is not None and 0 < pe     <= 10000) else None,
                "fwd_pe":       fwd_pe if (fwd_pe is not None and 0 < fwd_pe <= 10000) else None,
                "peg":          peg,
                "peg_method":   peg_method,
                "roe":          roe,
                "roic":         roic,
                "op_margin":    om,
                "int_coverage": ic,
                "debt_eq":      de_raw,
                "earn_traj":    earn_traj,
                "eps_growth":   eps_growth,
                "rev4":         rev4,
            }
        except Exception:
            return t, {}

    CHUNK  = 10
    SLEEP  = 0.5
    tl     = list(tickers)
    chunks = [tl[i:i + CHUNK] for i in range(0, len(tl), CHUNK)]
    prog   = st.progress(0)
    stat   = st.empty()
    for ci, chunk in enumerate(chunks):
        stat.text("yfinance fundamentals: {}/{} tickers...".format(
            min(ci * CHUNK, len(tl)), len(tl)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for t, d in ex.map(one, chunk):
                out[t] = d
        prog.progress((ci + 1) / len(chunks))
        if ci < len(chunks) - 1:
            time.sleep(SLEEP)
    prog.empty()
    stat.empty()
    return out

# ─── Momentum ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_momentum_batch(tickers):
    tl  = list(tickers)
    out = {t: {} for t in tl}

    def _process_single(t):
        try:
            raw_d = yf.download(t, period="7mo", interval="1d",
                                auto_adjust=True, progress=False)
            raw_m = yf.download(t, period="7mo", interval="1mo",
                                auto_adjust=True, progress=False)

            def _get_close(df):
                if df is None or df.empty:
                    return pd.Series(dtype=float)
                if isinstance(df.columns, pd.MultiIndex):
                    try:
                        col = df["Close"].iloc[:, 0]
                    except Exception:
                        return pd.Series(dtype=float)
                else:
                    col = df.get("Close", pd.Series(dtype=float))
                if isinstance(col, pd.DataFrame):
                    col = col.iloc[:, 0]
                return pd.to_numeric(col, errors="coerce").dropna()

            closes_d = _get_close(raw_d)
            closes_m = _get_close(raw_m)

            if len(closes_m) < 2:
                return t, {}

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
                dr = closes_d.pct_change().dropna().tail(90)
                if len(dr) >= 15:
                    trailing_vol = float(dr.std() * np.sqrt(252) * 100.0)

            skip = (r6 - r1) if (r6 is not None and r1 is not None) else None
            mom  = None
            if skip is not None and trailing_vol and trailing_vol > 0:
                mom = skip / trailing_vol
            elif skip is not None:
                mom = skip

            return t, {
                "ret_1mo": r1, "ret_3mo": r3, "ret_6mo": r6,
                "trailing_vol": trailing_vol, "momentum_score": mom,
            }
        except Exception:
            return t, {}

    CHUNK  = 10
    SLEEP  = 0.5
    chunks = [tl[i:i + CHUNK] for i in range(0, len(tl), CHUNK)]
    prog   = st.progress(0)
    stat   = st.empty()
    for ci, chunk in enumerate(chunks):
        stat.text("yfinance momentum: {}/{} tickers...".format(
            min(ci * CHUNK, len(tl)), len(tl)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            for t, d in ex.map(_process_single, chunk):
                out[t] = d
        prog.progress((ci + 1) / len(chunks))
        if ci < len(chunks) - 1:
            time.sleep(SLEEP)
    prog.empty()
    stat.empty()
    return out

# ─── Quality ──────────────────────────────────────────────────────────────────
# CHANGE 3: sector= parameter added.
#   Financials  → primary metric = ROE (not ROA proxy), Op Margin excluded.
#   All others  → unchanged v7 logic (ROIC/ROA primary, Op Margin included).

def compute_quality_score(roic, roe, int_coverage, op_margin, sector=None):

    scores  = []
    weights = []


    # v9: Financials use ROE; all other sectors use ROIC/ROA then fall back to ROE


    if sector in ROE_PRIMARY_SECTORS:


        primary = roe


    else:


        primary = roic if roic is not None else roe


    if primary is not None and not pd.isna(primary):
        pf = float(primary)
        scores.append(
            min(100.0, np.log1p(max(pf, 0)) / np.log1p(30.0) * 100.0) if pf > 0 else 0.0
        )
        weights.append(1.0)
    if int_coverage is not None and not pd.isna(int_coverage):
        scores.append(min(100.0, max(0.0, float(int_coverage) / 10.0 * 100.0)))
        weights.append(1.0)

    # v9: skip Op Margin for Financials — structurally not meaningful for banks


    if sector not in ROE_PRIMARY_SECTORS:

        if op_margin is not None and not pd.isna(op_margin):
            scores.append(min(100.0, max(0.0, float(op_margin) / 40.0 * 100.0)))
            weights.append(1.0)
    if not scores:
        return None
    return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

def quality_flag(roic, roe, ic, om, de, sector=None):
    flags = []

    if sector in ROE_PRIMARY_SECTORS:


        prof  = roe


        label = "ROE"


    else:


        prof  = roic if (roic is not None and not pd.isna(roic)) else roe


        label = "ROIC" if (roic is not None and not pd.isna(roic)) else "ROE"

    if prof is not None and not pd.isna(prof) and prof < QUALITY_THRESHOLDS["roic_min"]:
        flags.append("{}<8%".format(label))
    if ic is not None and not pd.isna(ic) and ic < QUALITY_THRESHOLDS["int_coverage_min"]:
        flags.append("IntCov<3x")

    if sector not in ROE_PRIMARY_SECTORS:

        if om is not None and not pd.isna(om) and om < QUALITY_THRESHOLDS["op_margin_min"]:
            flags.append("Margin<5%")
    de_note = " | D/E:{:.1f}".format(de) if (de is not None and not pd.isna(de)) else ""
    return (", ".join(flags) if flags else "Pass") + de_note

# ─── Ranking ──────────────────────────────────────────────────────────────────
def compute_rank_by_sector(scr):
    scr = scr.copy()
    scr["Score"] = pd.NA
    scr["Rank"]  = pd.NA
    for sector in scr["Sector"].dropna().unique():
        elig = scr[(scr["Sector"] == sector) & scr["Eligible"]].copy()
        if elig.empty:
            continue

        # CHANGE 4: pull per-sector weights; fall back to DEFAULT if unmapped


        W = SECTOR_FACTOR_WEIGHTS.get(sector, DEFAULT_FACTOR_WEIGHTS)

        elig["_s_val"]   = percentile_score(elig["P/E"],            ascending=True)
        elig["_s_peg"]   = percentile_score(elig["PEG"],            ascending=True)
        elig["_s_mom"]   = percentile_score(elig["Momentum Score"], ascending=False)
        elig["_s_etraj"] = percentile_score(elig["Earn Traj"],      ascending=False)
        qs               = elig["Quality Score"]
        q_min, q_max     = qs.min(), qs.max()
        elig["_s_quality"] = (
            (qs - q_min) / (q_max - q_min) * 100.0
            if pd.notna(q_min) and pd.notna(q_max) and q_max > q_min
            else qs.fillna(0.0)
        )
        elig["_s_quality"] = elig["_s_quality"].fillna(0.0)
        raw = (
            W["valuation"] * elig["_s_val"]
            + W["quality"]   * elig["_s_quality"]
            + W["peg"]       * elig["_s_peg"]
            + W["earn_traj"] * elig["_s_etraj"]
            + W["momentum"]  * elig["_s_mom"]
        )
        pen = elig.apply(
            lambda r: missing_factor_penalty(
                r, ["P/E", "PEG", "Quality Score", "Earn Traj", "Momentum Score"]
            ), axis=1
        )
        elig["Score"] = raw * pen
        elig          = elig.sort_values("Score", ascending=False)
        elig["Rank"]  = range(1, len(elig) + 1)
        scr.loc[elig.index, "Score"] = elig["Score"]
        scr.loc[elig.index, "Rank"]  = elig["Rank"]
    return scr

def compute_conviction_scores(scr):
    KEY  = ["P/E", "PEG", "Quality Score", "Momentum Score", "Earn Traj"]
    scr  = scr.copy()
    scr["_comp"] = scr.apply(
        lambda r: sum(1 for c in KEY if c in r.index and pd.notna(r[c])) / len(KEY),
        axis=1,
    )
    med_pe  = scr["P/E"].median()
    sec_map = scr.groupby("Sector")["P/E"].median()

    def sec_disc(s):
        if pd.isna(med_pe) or med_pe == 0:
            return 1.0
        sp = sec_map.get(s)
        if pd.isna(sp) or sp == 0:
            return 1.0
        return float(np.clip(med_pe / sp, 0.7, 1.3))

    scr["_disc"] = scr["Sector"].map(sec_disc)
    raw          = scr["Score"] * scr["_comp"] * scr["_disc"]
    cmin, cmax   = raw.min(), raw.max()
    scr["Conviction Score"] = (
        (raw - cmin) / (cmax - cmin) * 100.0 if cmax > cmin else 50.0
    )
    return scr.drop(columns=["_comp", "_disc"])

# ─── Build Table ──────────────────────────────────────────────────────────────
def build_screener_table(universe_df, yf_fundamentals, momentum_map):
    rows = []
    for _, r in universe_df.iterrows():
        t    = r["Ticker"]
        base = r["Base"]
        sec  = r["Sector"]

        fd  = yf_fundamentals.get(t, {})
        mom = momentum_map.get(t, {})

        price  = to_num(fd.get("price"))
        mc     = to_num(fd.get("mc"))
        hi52   = to_num(fd.get("hi52"))
        lo52   = to_num(fd.get("lo52"))
        pe     = to_num(fd.get("pe"))
        fwd_pe = to_num(fd.get("fwd_pe"))
        peg    = to_num(fd.get("peg"))
        roic   = to_num(fd.get("roic"))
        roe    = to_num(fd.get("roe"))
        ic     = to_num(fd.get("int_coverage"))
        om     = to_num(fd.get("op_margin"))
        de     = to_num(fd.get("debt_eq"))
        earn_traj  = to_num(fd.get("earn_traj"))
        eps_growth = fd.get("eps_growth")

        pos52 = None
        if pd.notna(price) and pd.notna(hi52) and pd.notna(lo52) and hi52 != lo52:
            pos52 = float(np.clip((price - lo52) / (hi52 - lo52) * 100.0, 0.0, 105.0))

        rev4               = fd.get("rev4", [None] * 4)
        rq1, rq2, rq3, rq4 = [to_num(x) for x in rev4]
        growth             = revenue_growth_yoy([rq1, rq2, rq3, rq4])


        # CHANGE 3 (call site): pass sector= to quality functions


        q_score = compute_quality_score(


            float(roic) if pd.notna(roic) else None,


            float(roe)  if pd.notna(roe)  else None,


            float(ic)   if pd.notna(ic)   else None,


            float(om)   if pd.notna(om)   else None,


            sector=sec,


        )


        ret_1mo   = to_num(mom.get("ret_1mo"))
        ret_3mo   = to_num(mom.get("ret_3mo"))
        ret_6mo   = to_num(mom.get("ret_6mo"))
        mom_score = to_num(mom.get("momentum_score"))
        t_vol     = to_num(mom.get("trailing_vol"))

        def to_lcr(v):
            return float(v) / 1e12 if (v is not None and pd.notna(v)) else None

        def to_tcr(v):
            return float(v) / 1e11 if (v is not None and pd.notna(v)) else None

        rows.append({
            "Ticker":            base,
            "YF Ticker":         t,
            "Sector":            sec,
            "Price (Rs)":        price,
            COL_MC:               to_lcr(mc),
            "Mkt Cap Raw":       mc,
            "P/E":               pe,
            "Fwd P/E":           fwd_pe,
            "PEG":               peg,
            "Earn Traj":         earn_traj,
            "52W Pos%":          to_num(pos52),
            "ROIC% (ROA)":       roic,
            "ROE%":              roe,
            "Int Coverage":      ic,
            "Op Margin%":        om,
            "Debt/Eq":           de,
            "Quality Score":     to_num(q_score) if q_score is not None else None,
            "Momentum Score":    mom_score,
            "Ret 1Mo%":          ret_1mo,
            "Ret 3Mo%":          ret_3mo,
            "Ret 6Mo%":          ret_6mo,
            "Trailing Vol%":     t_vol,
            "Eligible":          True,
            COL_RQ1:              to_tcr(rq1),
            COL_RQ2:              to_tcr(rq2),
            COL_RQ3:              to_tcr(rq3),
            COL_RQ4:              to_tcr(rq4),
            "Rev Growth% (YoY)": to_num(growth),
        })

    scr = pd.DataFrame(rows)
    if scr.empty:
        return scr

    total_mc = scr["Mkt Cap Raw"].sum()
    scr["MC% of Nifty50"] = scr["Mkt Cap Raw"] / total_mc * 100.0 if total_mc > 0 else None

    num_cols = [
        "Price (Rs)", COL_MC, "P/E", "Fwd P/E", "PEG", "52W Pos%",
        "ROIC% (ROA)", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
        "Quality Score", "Earn Traj", "Momentum Score",
        "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%", "MC% of Nifty50",
        COL_RQ1, COL_RQ2, COL_RQ3, COL_RQ4,
        "Rev Growth% (YoY)",
    ]
    for c in num_cols:
        if c in scr.columns:
            scr[c] = to_num(scr[c])

    scr = compute_rank_by_sector(scr)
    if "Rank" not in scr.columns:
        scr["Rank"] = pd.NA
    scr = compute_conviction_scores(scr)
    return scr

# ─── KPI Panel ────────────────────────────────────────────────────────────────
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

    is_all   = (sector_sel == "All Sectors")
    label    = "All Sectors (Nifty 50)" if is_all else sector_sel
    total_mc = scr["Mkt Cap Raw"].sum()
    sdata    = scr.copy() if is_all else scr[scr["Sector"] == sector_sel]
    sec_mc   = sdata["Mkt Cap Raw"].sum()
    pct      = 100.0 if is_all else (sec_mc / total_mc * 100.0 if total_mc > 0 else 0.0)
    med_pe   = sdata["P/E"].median()
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
    c1.markdown(_kpi("Sector Mkt Cap",   fmt_mc_inr(sec_mc),   "Rs Lakh Cr"),         unsafe_allow_html=True)
    c2.markdown(_kpi("Nifty 50 Mkt Cap", fmt_mc_inr(total_mc), "Rs Lakh Cr"),         unsafe_allow_html=True)
    c3.markdown(_kpi("Sector Share",     "{:.1f}%".format(pct), "{} stocks".format(len(sdata))), unsafe_allow_html=True)
    c4.markdown(_kpi("Median P/E",
                     "{:.1f}".format(med_pe) if pd.notna(med_pe) else "N/A",
                     "trailing twelve months", "#facc15"), unsafe_allow_html=True)
    c5.markdown(_kpi("Median Quality",
                     "{:.0f}/100".format(med_qual) if pd.notna(med_qual) else "N/A",
                     "ROE+IntCov+Margin", "#4ade80"), unsafe_allow_html=True)
    c6.markdown(_kpi("Median PEG",
                     "{:.2f}".format(med_peg) if pd.notna(med_peg) else "N/A",
                     "price/earnings/growth", "#a78bfa"), unsafe_allow_html=True)

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

        # CHANGE 5: show active scoring weights for the selected sector


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
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Nifty 50 Screener",
    layout="wide",
    page_icon="IN",
    initial_sidebar_state="collapsed",
)
st.markdown(
    "<style>"
    "div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}"
    "</style>",
    unsafe_allow_html=True,
)

st.markdown("## Nifty 50 Fundamental Screener v9")
st.caption("100% yfinance · No paid API required · Wikipedia universe · Sector-Adaptive 5-factor scoring · INR")

page_screener, page_about, page_debug = st.tabs(["Screener", "About", "Debug"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SCREENER
# ══════════════════════════════════════════════════════════════════════════════
with page_screener:
    col_r, col_t = st.columns([1, 6])
    with col_r:
        if st.button("Refresh"):
            st.cache_data.clear()
            st.rerun()
    with col_t:
        st.caption("Last loaded: {} · Prices+fundamentals: 1hr cache · Universe: 24hr cache".format(
            datetime.now().strftime("%I:%M %p")))

    with st.spinner("Loading universe from Wikipedia..."):
        universe_df = get_nifty50_universe()
    tickers = tuple(universe_df["Ticker"].tolist())

    with st.spinner("Fetching fundamentals from yfinance..."):
        yf_fundamentals = fetch_yf_fundamentals(tickers)

    with st.spinner("Fetching momentum data from yfinance..."):
        momentum = fetch_momentum_batch(tickers)

    total_t   = len(tickers)
    has_price = sum(1 for t in tickers if yf_fundamentals.get(t, {}).get("price")        is not None)
    has_pe    = sum(1 for t in tickers if yf_fundamentals.get(t, {}).get("pe")            is not None)
    has_roe   = sum(1 for t in tickers if yf_fundamentals.get(t, {}).get("roe")           is not None)
    has_ic    = sum(1 for t in tickers if yf_fundamentals.get(t, {}).get("int_coverage")  is not None)
    has_et    = sum(1 for t in tickers if yf_fundamentals.get(t, {}).get("earn_traj")     is not None)
    has_mom   = sum(1 for t in tickers if momentum.get(t, {}).get("momentum_score")       is not None)

    coverage_color = "info" if has_price >= total_t * 0.7 else "warning"
    getattr(st, coverage_color)(
        "Data coverage — "
        "Price: {}/{} ({:.0f}%) · P/E: {}/{} ({:.0f}%) · "
        "ROE: {}/{} ({:.0f}%) · Int Coverage: {}/{} ({:.0f}%) · "
        "Earn Traj: {}/{} ({:.0f}%) · Momentum: {}/{} ({:.0f}%)".format(
            has_price, total_t, has_price / total_t * 100,
            has_pe,    total_t, has_pe    / total_t * 100,
            has_roe,   total_t, has_roe   / total_t * 100,
            has_ic,    total_t, has_ic    / total_t * 100,
            has_et,    total_t, has_et    / total_t * 100,
            has_mom,   total_t, has_mom   / total_t * 100,
        )
    )

    with st.spinner("Building screener table..."):
        scr = build_screener_table(universe_df, yf_fundamentals, momentum)

    if scr.empty:
        st.error("No data returned. Check the Debug tab.")
        st.stop()

    st.markdown("### Filters")
    with st.expander("Valuation and Size", expanded=True):
        fc1, fc2, fc3, fc4, fc5 = st.columns(5)
        all_sectors = sorted(scr["Sector"].dropna().unique().tolist())
        sector_sel  = fc1.selectbox("Sector", ["All Sectors"] + all_sectors)
        sort_by     = fc2.selectbox("Sort by", [
            "Sector then Rank", "Score high to low", "Conviction high to low",
            "MC% of Nifty50 high to low", "Price low to high", "Price high to low",
            "Mkt Cap high to low", "PE low to high", "Fwd PE low to high",
            "PEG low to high", "Quality Score high", "ROE high to low",
            "Earn Traj high to low", "Momentum Score high",
            "52W Pos low to high", "Rev Growth high to low",
        ])
        pe_max   = fc3.number_input("Max PE",             value=9999,  step=10)
        peg_max  = fc4.number_input("Max PEG",            value=999.0, step=1.0)
        mc_min_l = fc5.number_input("Min Mkt Cap (LCr)",  value=0.0,   step=1.0)

    with st.expander("Quality Filters", expanded=False):
        qc1, qc2, qc3, qc4 = st.columns(4)
        roe_min_f  = qc1.number_input("Min ROE (%)",          value=0.0, step=5.0)
        ic_min_f   = qc2.number_input("Min Int Coverage (x)", value=0.0, step=1.0)
        om_min_f   = qc3.number_input("Min Op Margin (%)",    value=0.0, step=5.0)
        qual_min_f = qc4.number_input("Min Quality Score",    value=0.0, step=5.0)

    with st.expander("Momentum and Earnings", expanded=False):
        mc1, mc2 = st.columns(2)
        mom_min = mc1.number_input("Min Momentum Score", value=-999.0, step=5.0)
        et_min  = mc2.number_input("Min Earn Traj",      value=-1.0,   step=0.1)

    render_sector_kpi_panel(scr, sector_sel)

    filt = scr.copy()
    if sector_sel != "All Sectors":
        filt = filt[filt["Sector"] == sector_sel]
    filt = filt[(filt[COL_MC].isna())           | (filt[COL_MC]           >= mc_min_l)]
    filt = filt[(filt["P/E"].isna())          | (filt["P/E"]          <= pe_max)]
    filt = filt[(filt["PEG"].isna())          | (filt["PEG"]          <= peg_max)]
    filt = filt[(filt["ROE%"].isna())         | (filt["ROE%"]         >= roe_min_f)]
    filt = filt[(filt["Int Coverage"].isna()) | (filt["Int Coverage"] >= ic_min_f)]
    filt = filt[(filt["Op Margin%"].isna())   | (filt["Op Margin%"]   >= om_min_f)]
    filt = filt[(filt["Quality Score"].isna())| (filt["Quality Score"]>= qual_min_f)]
    filt = filt[(filt["Momentum Score"].isna())|(filt["Momentum Score"]>= mom_min)]
    filt = filt[(filt["Earn Traj"].isna())    | (filt["Earn Traj"]    >= et_min)]

    sort_map = {
        "Sector then Rank":           (["Sector", "Rank"],  [True,  True]),
        "Score high to low":          (["Score"],           [False]),
        "Conviction high to low":     (["Conviction Score"],[False]),
        "MC% of Nifty50 high to low": (["MC% of Nifty50"], [False]),
        "Price low to high":          (["Price (Rs)"],      [True]),
        "Price high to low":          (["Price (Rs)"],      [False]),
        "Mkt Cap high to low":        ([COL_MC],            [False]),
        "PE low to high":             (["P/E"],             [True]),
        "Fwd PE low to high":         (["Fwd P/E"],         [True]),
        "PEG low to high":            (["PEG"],             [True]),
        "Quality Score high":         (["Quality Score"],   [False]),
        "ROE high to low":            (["ROE%"],            [False]),
        "Earn Traj high to low":      (["Earn Traj"],       [False]),
        "Momentum Score high":        (["Momentum Score"],  [False]),
        "52W Pos low to high":        (["52W Pos%"],        [True]),
        "Rev Growth high to low":     (["Rev Growth% (YoY)"],[False]),
    }
    sc, sa = sort_map.get(sort_by, (["Sector", "Rank"], [True, True]))
    filt   = filt.sort_values(sc, ascending=sa, na_position="last")

    st.caption("Showing {} of {} stocks · Sector: {} · Sort: {}".format(
        len(filt), len(scr), sector_sel, sort_by))

    disp = filt.copy()
    round_cols = [
        "P/E", "Fwd P/E", "PEG", "Earn Traj", "52W Pos%",
        "ROIC% (ROA)", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
        "Quality Score", "Momentum Score", "Ret 1Mo%", "Ret 3Mo%",
        "Ret 6Mo%", "Trailing Vol%", "Score", "Conviction Score",
        "Rev Growth% (YoY)", "MC% of Nifty50", "Price (Rs)",
        COL_MC, COL_RQ1, COL_RQ2, COL_RQ3, COL_RQ4,
    ]
    for c in round_cols:
        if c in disp.columns:
            disp[c] = disp[c].round(2)

    disp["Quality Flag"] = disp.apply(
        lambda r: quality_flag(
            r.get("ROIC% (ROA)"), r.get("ROE%"),
            r.get("Int Coverage"), r.get("Op Margin%"), r.get("Debt/Eq"),

            sector=r.get("Sector"),   # CHANGE 3 (call site): pass sector to flag

        ), axis=1,
    )
    disp["Rank"] = disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

    COLS = [
        "Ticker", "Sector",
        "Price (Rs)", "52W Pos%", COL_MC, "MC% of Nifty50",
        "P/E", "Fwd P/E", "PEG", "Earn Traj",
        "ROIC% (ROA)", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
        "Quality Score", "Quality Flag",
        "Momentum Score", "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
        "Score", "Conviction Score", "Rank",
        COL_RQ1, COL_RQ2, COL_RQ3, COL_RQ4,
        "Rev Growth% (YoY)",
    ]
    disp_final = disp[[c for c in COLS if c in disp.columns]].copy()
    st.dataframe(disp_final, use_container_width=True, height=680)

    st.download_button(
        label="Download CSV",
        data=disp_final.to_csv(index=False).encode("utf-8"),
        file_name="nifty50_screener_{}.csv".format(datetime.now().strftime("%Y%m%d_%H%M")),
        mime="text/csv",
    )

    st.markdown("---")
    st.markdown("**Column Glossary**")
    st.markdown(
        "- **{mc}**: Market cap in Rs Lakh Crore (1 LCr = Rs 1,00,000 Cr = 10^12).\n"
        "- **{rq1} to {rq4}**: Quarterly revenue in Rs Thousand Crore (1 = Rs 1,000 Cr = 10^11).\n"
        "- **P/E**: Trailing twelve months P/E from yfinance trailingPE.\n"
        "- **Fwd P/E**: Forward P/E from yfinance forwardPE (analyst consensus).\n"
        "- **PEG**: From yfinance pegRatio if available; else Fwd P/E / EPS growth (>= 5% only).\n"
        "- **Earn Traj**: YoY EPS change from quarterly income statement, clamped to [-1, +1].\n"
        "- **ROIC% (ROA)**: Return on Assets as ROIC proxy (decimal to %).\n"
        "- **ROE%**: Return on Equity (decimal to %).\n"
        "- **Int Coverage**: EBIT / Interest Expense from annual income statement. Capped at 100x.\n"
        "- **Op Margin%**: Operating margin (decimal to %).\n"
        "- **Debt/Eq**: Debt-to-equity from yfinance debtToEquity.\n"
        "- **Quality Score**: 0-100 composite. Financials: ROE + IntCov. Others: ROIC/ROA + IntCov + OpMargin.\n"
        "- **Momentum Score**: (6M return - 1M return) / annualised volatility.\n"
        "- **Score**: Sector-adaptive weights — see active weights bar when sector is selected.\n"
        "- **Rank**: Within-sector rank by Score (1 = best).\n".format(
            mc=COL_MC, rq1=COL_RQ1, rq2=COL_RQ2, rq3=COL_RQ3, rq4=COL_RQ4)
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with page_about:
    st.markdown("## About - Nifty 50 Screener v9")
    st.markdown("### Data Sources")
    st.markdown(
        "| Field | yfinance source | Unit |\n"
        "|---|---|---|\n"
        "| Price | currentPrice / regularMarketPrice | Rs |\n"
        "| Market Cap | marketCap | Rs Lakh Cr |\n"
        "| Trailing P/E | trailingPE | x |\n"
        "| Forward P/E | forwardPE | x |\n"
        "| PEG | pegRatio or calculated | x |\n"
        "| ROE | returnOnEquity | % |\n"
        "| ROIC proxy | returnOnAssets | % |\n"
        "| Op Margin | operatingMargins | % |\n"
        "| Int Coverage | EBIT/InterestExpense from income_stmt | x |\n"
        "| Debt/Equity | debtToEquity | ratio |\n"
        "| Revenue (quarterly) | quarterly_income_stmt | Rs Thousand Cr |\n"
        "| Momentum | yf.download() price history | score |\n"
        "| Universe | Wikipedia NIFTY_50 | cached 24h |\n"
    )
    st.markdown("### v9 — Sector-Adaptive Factor Weights")
    st.info(
        "v9 replaces the single global FACTOR_WEIGHTS dict with per-sector weights. "
        "Financials quality scoring now uses ROE as primary input (not ROA). "
        "Op Margin excluded from Financials quality score."
    )
    st.markdown(
        "| Sector | Val | Quality | PEG | Earn | Mom | Primary Driver |\n"
        "|---|---|---|---|---|---|---|\n"
        "| Information Technology | 20% | 25% | **25%** | 15% | 15% | PEG — growth premium |\n"
        "| Consumer Discretionary | 20% | 20% | 22% | 18% | **20%** | Consumer cycle + growth |\n"
        "| Communication Services | 22% | 23% | 22% | 18% | 15% | Balanced telco+media |\n"
        "| Health Care | 25% | **30%** | 18% | 15% | 12% | Quality: moat durability |\n"
        "| Industrials | 25% | **28%** | 18% | 17% | 12% | Quality: execution |\n"
        "| Consumer Staples | 28% | **32%** | 10% | 15% | 15% | Quality: compounders |\n"
        "| Financials | **30%** | 25% | 18% | 17% | 10% | Valuation + ROE (not ROA) |\n"
        "| Energy | **30%** | 18% | 12% | 15% | **25%** | Commodity cycle momentum |\n"
        "| Materials | 28% | 20% | 12% | 15% | **25%** | Commodity cycle momentum |\n"
        "| Real Estate | **30%** | 18% | 10% | **22%** | 20% | Pre-sales cycle trajectory |\n"
        "| Utilities | **38%** | 27% | 5% | 15% | 15% | Bond proxy — valuation only |\n"
        "| DEFAULT | 25% | 25% | 20% | 15% | 15% | Fallback for unmapped sectors |\n"
    )
    st.markdown("### Scoring Model")
    st.code(
        "Score = Val%  * Valuation (P/E percentile, lower = better)\n"
        "      + Qual% * Quality   (ROE/ROIC + Int Coverage + Op Margin)\n"
        "      + PEG%  * PEG       (lower = better, only when EPS growth >= 5%)\n"
        "      + Earn% * Earn Traj (YoY EPS direction, -1 to +1)\n"
        "      + Mom%  * Momentum  (6M-1M skip return / trailing volatility)\n\n"
        "Weights are sector-specific — see table above.\n"
        "Missing data penalty: -15% for 2 missing factors, -30% for 3+.\n\n"
        "Financials: Quality = ROE + IntCov (Op Margin excluded).",
        language=None,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DEBUG  (unchanged from v7)
# ══════════════════════════════════════════════════════════════════════════════
with page_debug:
    st.markdown("## Debug - yfinance Diagnostics")
    st.info("No API key required. All data from yfinance (.NS tickers).")

    test_base = st.text_input("Base symbol (no suffix)", value="RELIANCE")
    test_yf   = test_base.upper().strip() + ".NS"
    st.markdown("**Testing ticker:** `{}`".format(test_yf))

    if st.button("Run diagnostic"):
        with st.spinner("Testing {}...".format(test_yf)):

            st.markdown("### 1. yfinance .info")
            try:
                t_obj = yf.Ticker(test_yf)
                info  = t_obj.info or {}
                if info.get("regularMarketPrice") or info.get("currentPrice"):
                    st.success("yfinance .info OK")
                    st.json({k: info.get(k) for k in [
                        "shortName", "currentPrice", "marketCap",
                        "trailingPE", "forwardPE", "pegRatio",
                        "returnOnEquity", "returnOnAssets",
                        "operatingMargins", "debtToEquity",
                        "ebitda", "interestExpense",
                        "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
                        "trailingEps", "forwardEps",
                    ]})
                else:
                    st.error("No price returned for {}".format(test_yf))
            except Exception as ex:
                st.error("yfinance .info error: {}".format(ex))

            st.markdown("### 2. Annual income statement (interest coverage)")
            try:
                t_obj = yf.Ticker(test_yf)
                for attr in ("income_stmt", "financials"):
                    df = getattr(t_obj, attr, None)
                    if df is not None and not df.empty:
                        st.success("{} OK — {} rows".format(attr, len(df)))
                        st.dataframe(df.head(8))
                        ic = _interest_coverage_from_financials(t_obj)
                        st.json({"Int Coverage (computed)": ic})
                        break
                else:
                    st.warning("No annual income statement found")
            except Exception as ex:
                st.error("Annual income statement error: {}".format(ex))

            st.markdown("### 3. Quarterly income statement (EPS, revenue)")
            try:
                t_obj = yf.Ticker(test_yf)
                for attr in ("quarterly_income_stmt", "quarterly_financials"):
                    df = getattr(t_obj, attr, None)
                    if df is not None and not df.empty:
                        st.success("{} OK — {} rows x {} quarters".format(
                            attr, len(df), len(df.columns)))
                        st.dataframe(df.head(6))
                        rev4         = _quarterly_revenues(t_obj)
                        eps_r, eps_o = _quarterly_eps(t_obj)
                        et = None
                        if eps_r and eps_o and abs(eps_o) > 0.001:
                            raw_et = (eps_r - eps_o) / abs(eps_o)
                            et     = max(-1.0, min(1.0, raw_et / 2.0))
                        rg = revenue_growth_yoy(rev4)
                        st.json({
                            "Rev Q1 raw (INR)":   rev4[0],
                            "Rev Q4 raw (INR)":   rev4[3],
                            "Rev Q1 (1000Cr)":    (rev4[0] / 1e11) if rev4[0] else None,
                            "Rev YoY Growth%":    rg,
                            "EPS recent":         eps_r,
                            "EPS 1yr ago":        eps_o,
                            "Earn Traj":          et,
                        })
                        break
                else:
                    st.warning("No quarterly income statement found")
            except Exception as ex:
                st.error("Quarterly income statement error: {}".format(ex))

            st.markdown("### 4. Price history (momentum)")
            try:
                hist = yf.download(test_yf, period="7mo", interval="1d",
                                   auto_adjust=True, progress=False)
                if not hist.empty:
                    close_col = hist["Close"]
                    if isinstance(close_col, pd.DataFrame):
                        close_col = close_col.iloc[:, 0]
                    closes = pd.to_numeric(close_col, errors="coerce").dropna()
                    latest = float(closes.iloc[-1])
                    st.success("Price history OK — {} rows, latest: {:.2f}".format(
                        len(hist), latest))
                    hist_m  = yf.download(test_yf, period="7mo", interval="1mo",
                                          auto_adjust=True, progress=False)
                    close_m = hist_m["Close"]
                    if isinstance(close_m, pd.DataFrame):
                        close_m = close_m.iloc[:, 0]
                    closes_m = pd.to_numeric(close_m, errors="coerce").dropna()
                    px_now   = float(closes_m.iloc[-1])

                    def ret_mo(n):
                        idx = -(n + 1)
                        if abs(idx) > len(closes_m):
                            return None
                        px = float(closes_m.iloc[idx])
                        return round((px_now / px - 1) * 100.0, 2) if px > 0 else None

                    dr  = closes.pct_change().dropna().tail(90)
                    vol = round(float(dr.std() * np.sqrt(252) * 100.0), 2) if len(dr) >= 15 else None
                    r1, r3, r6 = ret_mo(1), ret_mo(3), ret_mo(6)
                    skip = (r6 - r1) if (r6 and r1) else None
                    mom  = round(skip / vol, 3) if (skip and vol and vol > 0) else skip
                    st.json({
                        "Ret 1Mo%": r1, "Ret 3Mo%": r3, "Ret 6Mo%": r6,
                        "Trailing Vol%": vol, "Momentum Score": mom,
                    })
                else:
                    st.warning("Price history empty for {}".format(test_yf))
            except Exception as ex:
                st.error("Price history error: {}".format(ex))
