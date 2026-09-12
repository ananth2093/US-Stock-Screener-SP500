# screener_app.py
"""Fast snapshot dashboard for the saved screener output.

Run `python update_data.py` to refresh `data/latest_screener.parquet`.
"""

from datetime import datetime
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════════
# Copied helpers / constants from screener_app.py
# ═══════════════════════════════════════════════════════════════════════════════
SLOAN_ACCRUALS_THRESHOLD = 0.08

ROE_PRIMARY_SECTORS = {"Financials"}

QUALITY_THRESHOLDS = {
    "roic_min": 8.0,
    "int_coverage_min": 3.0,
    "op_margin_min": 5.0,
}

SECTOR_QUALITY_THRESHOLDS = {
    "Information Technology": {"roic_min": 12.0, "int_coverage_min": 5.0, "op_margin_min": 15.0},
    "Communication Services": {"roic_min":  8.0, "int_coverage_min": 3.0, "op_margin_min": 12.0},
    "Consumer Discretionary": {"roic_min":  8.0, "int_coverage_min": 3.0, "op_margin_min":  8.0},
    "Consumer Staples":       {"roic_min":  8.0, "int_coverage_min": 4.0, "op_margin_min":  6.0},
    "Energy":                 {"roic_min":  6.0, "int_coverage_min": 3.0, "op_margin_min": 10.0},
    "Financials":             {"roic_min":  8.0, "int_coverage_min": 3.0, "op_margin_min": 15.0},
    "Health Care":            {"roic_min": 10.0, "int_coverage_min": 5.0, "op_margin_min": 12.0},
    "Industrials":            {"roic_min":  9.0, "int_coverage_min": 4.0, "op_margin_min":  8.0},
    "Materials":              {"roic_min":  7.0, "int_coverage_min": 3.0, "op_margin_min":  8.0},
    "Real Estate":            {"roic_min":  5.0, "int_coverage_min": 2.0, "op_margin_min": 25.0},
    "Utilities":              {"roic_min":  4.0, "int_coverage_min": 2.0, "op_margin_min": 12.0},
}

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

METRIC_DIRECTION = {
    "Score": "green",
    "Quality Score": "green",
    "Momentum Score": "red",
}


def safe_round(series: pd.Series, decimals=2) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").round(decimals)


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def _blend_color(ratio: float, low_hex: str, high_hex: str) -> str:
    low = _hex_to_rgb(low_hex)
    high = _hex_to_rgb(high_hex)
    rgb = tuple(int(low[i] + (high[i] - low[i]) * ratio) for i in range(3))
    return f"background-color: rgb({rgb[0]},{rgb[1]},{rgb[2]}); color: #1f2937;"


def _column_gradient(series: pd.Series, low_hex: str, high_hex: str) -> list:
    s = pd.to_numeric(series, errors="coerce")
    s_min, s_max = s.min(), s.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max == s_min:
        return [""] * len(s)
    ratios = ((s - s_min) / (s_max - s_min)).tolist()
    return [_blend_color(r, low_hex, high_hex) if pd.notna(r) else "" for r in ratios]


def _green_grad(s: pd.Series) -> list:
    return _column_gradient(s, "#d4f7d4", "#6bc46b")


def _red_grad(s: pd.Series) -> list:
    return _column_gradient(s, "#ffd9d9", "#e07a7a")


def _sector_quality_thresholds(sector):
    return SECTOR_QUALITY_THRESHOLDS.get(sector, QUALITY_THRESHOLDS)


def quality_flag(roic, roe, ic, om, sloan_ratio=None, sector=None):
    EPSILON = 1e-9
    flags = []
    th = _sector_quality_thresholds(sector)
    prof = (roe if sector in ROE_PRIMARY_SECTORS
            else (roic if (roic is not None and not pd.isna(roic)) else roe))
    lbl = ("ROE" if sector in ROE_PRIMARY_SECTORS
           else ("ROIC" if (roic is not None and not pd.isna(roic)) else "ROE"))
    roic_min = th["roic_min"]
    if prof is not None and not pd.isna(prof) and float(prof) < roic_min - EPSILON:
        flags.append("{}<{:.0f}%".format(lbl, roic_min))
    if ic is not None and not pd.isna(ic) and float(ic) < th["int_coverage_min"]:
        flags.append("IntCov<{:.0f}x".format(th["int_coverage_min"]))
    if sector not in ROE_PRIMARY_SECTORS:
        if om is not None and not pd.isna(om) and float(om) < th["op_margin_min"]:
            flags.append("Margin<{:.0f}%".format(th["op_margin_min"]))
    if sloan_ratio is not None and not pd.isna(sloan_ratio) and float(sloan_ratio) > SLOAN_ACCRUALS_THRESHOLD:
        flags.append("HighAccruals")
    return ", ".join(flags) if flags else "Pass"


def render_reference_guide():
    st.markdown("## Column Reference Guide")
    tabs = st.tabs([
        "Valuation", "Quality", "PEG", "Earn Trajectory",
        "Momentum", "Earnings Surprise", "Ranking & Score", "Coverage v19.1",
    ])

    with tabs[0]:
        st.markdown("""
### What the valuation columns mean

| Column | Definition | Calculation / Source |
|---|---|---|
| **P/E** | Trailing 12-month price-to-earnings | Market Cap / Net Income TTM. Filled from FMP quote/key-metrics/ratios; Yahoo `.info` fallback. | FMP quote, key-metrics, ratios; Yahoo `.info` |
| **Fwd P/E** | Forward price-to-earnings | Market Cap / Forward EPS (next fiscal year). | FMP quote; Yahoo `.info` |
| **EV/EBITDA** | Enterprise value over EBITDA | Enterprise Value / EBITDA TTM. | FMP key-metrics; Yahoo `.info` |
| **FCF Yield%** | Free cash flow as a % of market cap | FCF TTM / Market Cap × 100. | FMP key-metrics / cash-flow; computed from FCF TTM |
| **EV/Sales** | Enterprise value over revenue | EV / Total Revenue. | FMP key-metrics; Yahoo `.info` |
| **Div Yield%** | Dividend yield | Most recent dividend yield. | Yahoo `.info` |
| **P/E vs Sector Med** | P/E relative to sector median | `P/E / sector_median(P/E)`. 1.0 = exactly median; <1 = cheaper; >1 = pricier. | Computed inside the app |

### How the Valuation **sub-score** (0-100) is built
Inside each sector ranking, every eligible stock receives a valuation score using **robust percentile scoring** (MAD Z-score) rather than raw numbers, so extreme outliers don't dominate:

1. **FCF Yield%** — higher is better → weight **40%**
2. **EV/EBITDA** — lower is better → weight **35%**
3. **P/E** (using Fwd P/E if available, else P/E) — lower is better → weight **25%**

**Example** — Suppose a Communications stock has FCF Yield 9.6%, EV/EBITDA 8.4, Fwd P/E 9.6. The app compares those values to all other Communications stocks, converts each into a 0-100 score, then blends them 40/35/25. A stock cheaper than most peers scores near 100; an expensive one near 0.

### Why robust scoring?
A single stock with P/E = 500 would wreck a simple average. We winsorise values and use the median absolute deviation (MAD) to keep outliers from distorting the ranking.
        """)

    with tabs[1]:
        st.markdown("""
### Quality columns

| Column | What it measures | How it is computed | Typical "good" |
|---|---|---|---|
| **ROIC%** | Return on invested capital | `Net Income TTM / (Equity + Debt - Cash) × 100`; also filled from FMP `roicTTM` / `returnOnCapitalEmployedTTM` | >8% |
| **ROE%** | Return on equity | Net Income TTM / Shareholders' Equity × 100 | >10% |
| **Int Coverage** | Ability to pay interest | EBIT TTM / Interest Expense TTM | >3x |
| **Op Margin%** | Operating profit per dollar of sales | Operating Income / Revenue × 100 | >5% |
| **Debt/Eq** | Leverage | Total Debt / Shareholders' Equity | <2.0 |
| **Quality Score** | Composite 0-100 quality score | See weights below | >60 |
| **Piotroski F** | 9-point fundamental health score | Sum of 9 binary accounting flags (0-9) | ≥5 |
| **Sloan Ratio** | Accrual / earnings quality check | `(Net Income - Operating Cash Flow) / Average Total Assets` | close to 0; >0.08 flagged |
| **Quality Flag** | Human-readable failure labels | Labels the specific thresholds a stock missed | "Pass" = all thresholds met |

### Quality Score formula (0-100)
1. **Profitability** (ROIC, or ROE for Financials) — **25%**<br>
   `score = log1p(ROIC) / log1p(30) * 100`; negative = 0
2. **Interest Coverage** — **15%**<br>
   `score = min(IC/10 * 100, 100)`
3. **Operating Margin** — **15%** (excluded for Financials)<br>
   `score = min(OpMargin/40 * 100, 100)`
4. **Gross Margin** — **20%**<br>
   `score = min(GM/60 * 100, 100)`; +10% if GM improved vs prior year
5. **Piotroski F-score** — **15%**<br>
   `score = F/9 * 100`
6. **Sloan Ratio** — **10%**<br>
   Best near 0; penalised as it moves toward +0.08 or beyond

**Example** — A stock with ROIC 15%, IC 5x, Op Margin 12%, GM 45%, Piotroski 7, Sloan 0.02 gets roughly:
- Profitability: log(16)/log(31) ≈ 76
- IC: 5/10 = 50
- Op Margin: 12/40 = 30
- GM: 45/60 = 75
- Piotroski: 7/9 = 78
- Sloan: near 80
- Weighted total ≈ **65**

### Quality Flag thresholds
```
ROIC < 8%        → "ROIC<8%"   (or "ROE<8%" for Financials)
Interest Coverage < 3 → "IntCov<3x"
Op Margin < 5%   → "Margin<5%"
Sloan Ratio > 0.08 → "HighAccruals"
```
If none are triggered, the flag is **"Pass"**.
        """)

    with tabs[2]:
        st.markdown("""
### PEG Ratio (Price/Earnings-to-Growth)
PEG tells you how much you are paying per unit of earnings growth. Lower is generally better.

```
PEG = (Forward P/E or Trailing P/E) / EPS Growth %
```

### 3-tier cascade (v19.1)
Because PEG is not always directly available, the app tries three sources in order:

| Tier | Source | When it is used |
|---|---|---|
| 1 | **Direct FMP** (`pegRatioTTM` from key-metrics or ratios) | First choice; most accurate |
| 2 | **Computed from EPS growth** | If Tier 1 missing but `EPS Growth %` is available from FMP income statement or Yahoo |
| 3 | **Earn Trajectory proxy** | If no EPS growth, but the stock has a positive `Earn Traj` (forward EPS > trailing EPS) |

PEG is only shown when the growth input is **≥5%** (otherwise a tiny growth number would create a meaningless PEG).

**Example**
- Fwd P/E = 15.0
- EPS Growth = 10%
- **PEG = 15 / 10 = 1.5**

A PEG of 1.5 means you pay 1.5x the earnings growth rate. The "PEG Method" column tells you which tier was used (e.g., "FMP-km", "FMP-IS-2yr", "Yahoo-fwd", "EarnTraj-proxy").
        """)

    with tabs[3]:
        st.markdown("""
### Earn Traj (Earnings Trajectory)
Earn Traj measures the **direction and magnitude of expected earnings change** from trailing to forward EPS.

```
Earn Traj = (Forward EPS - Trailing EPS) / |Trailing EPS|
Result is clipped to the range [-1, +1]
```

| Value | Meaning |
|---|---|
| +1.0 | Forward EPS is much higher than trailing EPS (strong expected growth) |
| 0.0 | No expected change |
| -1.0 | Forward EPS is much lower (expected decline) |

**Why clip?** If a company swung from a tiny loss to a big profit, the raw percentage could be thousands of percent. Capping it at ±1 keeps the metric stable and comparable across all 500 stocks.

**Example**
- Trailing EPS = $2.00
- Forward EPS = $2.74
- Earn Traj = (2.74 - 2.00) / 2.00 = **+0.37**

This company is expected to grow earnings by ~37%.

If trailing EPS is negative and forward EPS is still negative, the positive clip is limited to **+0.30** to avoid rewarding a "less bad" loss too much.
        """)

    with tabs[4]:
        st.markdown("""
### Momentum columns

| Column | Definition | Calculation |
|---|---|---|
| **Ret 1Mo%** | 1-month price return | `(Price today / Price 1 month ago - 1) × 100` |
| **Ret 3Mo%** | 3-month price return | `(Price today / Price 3 months ago - 1) × 100` |
| **Ret 6Mo%** | 6-month price return | `(Price today / Price 6 months ago - 1) × 100` |
| **Trailing Vol%** | Annualised volatility | `std(daily returns) × sqrt(252) × 100` |
| **52W Pos%** | Where price sits in its 52-week range | `(Price - 52W Low) / (52W High - 52W Low) × 100` |
| **Momentum Score** | Composite 0-100 momentum score | Blend of 4 signals (see below) |

### Momentum Score components
| Signal | What it measures | Weight |
|---|---|---|
| **Skip Mo** | 6-month return minus 1-month return, normalised by volatility | 40% |
| **52W Proximity** | How close price is to its 52-week high | 25% |
| **vs MA200** | Price relative to its 200-day moving average (or 50-day if <200 days) | 20% |
| **Rel Str SPY** | 3-month return vs SPY 3-month return | 15% |

**Example — Skip Month**
- 6-month return = +20%
- 1-month return = +5%
- Trailing vol = 25%
- Skip Mo raw = (20 - 5) / 25 = 0.6 → clipped to [-1, +1] → **+0.6**

**Example — 52W Proximity**
- Price = $95, 52W High = $100
- Proximity = 95/100 = 0.95 → clipped to [0, 1] → **0.95**

The four signals are each converted to a 0-100 score and blended with the weights above. A stock with strong recent returns, near its highs, above its moving average, and beating SPY will score near 100.
        """)

    with tabs[5]:
        st.markdown("""
### Earnings surprise columns

| Column | Definition | Calculation |
|---|---|---|
| **EPS Surp Avg%** | Average earnings surprise over the last 4 quarters | `mean((Actual EPS - Estimate) / |Estimate| × 100)` |
| **EPS Beat Rate** | % of recent quarters that beat estimates | `quarters with positive surprise / total quarters` |
| **EPS Surp Trend** | Direction of surprises | +1 if the last 2 quarters' average surprise is higher than the previous 2; -1 if lower |
| **Revision Mom** | Analyst estimate revision momentum | Net change in strong buy/buy vs sell/strong-sell ratings between the two latest months, scaled to [-1,+1] |

**Example — EPS Surprise**
Recent 4 quarters:
- Q1: Actual $1.10, Estimate $1.00 → +10%
- Q2: Actual $1.05, Estimate $1.00 → +5%
- Q3: Actual $0.98, Estimate $1.00 → -2%
- Q4: Actual $1.12, Estimate $1.00 → +12%

- Avg surprise = (10 + 5 - 2 + 12) / 4 = **6.25%**
- Beat rate = 3/4 = **0.75**
- Trend: last 2 avg = (12 - 2)/2 = 5; previous 2 avg = (10 + 5)/2 = 7.5 → **-1** (trending down)

A high positive Revision Mom means analysts are upgrading the stock recently.
        """)

    with tabs[6]:
        sector_rows = "\n".join(
            "| {} | {:.0%} | {:.0%} | {:.0%} | {:.0%} | {:.0%} |".format(
                sector, w["valuation"], w["quality"], w["peg"], w["earn_traj"], w["momentum"]
            )
            for sector, w in SECTOR_FACTOR_WEIGHTS.items()
        )
        default_row = "| **Default** | {:.0%} | {:.0%} | {:.0%} | {:.0%} | {:.0%} |".format(
            DEFAULT_FACTOR_WEIGHTS["valuation"], DEFAULT_FACTOR_WEIGHTS["quality"],
            DEFAULT_FACTOR_WEIGHTS["peg"], DEFAULT_FACTOR_WEIGHTS["earn_traj"],
            DEFAULT_FACTOR_WEIGHTS["momentum"]
        )
        st.markdown(f"""
### Ranking & score columns

| Column | Definition | How it is computed |
|---|---|---|
| **Score** | Sector-relative composite score | Blend of Valuation, Quality, PEG, Earn Traj, and Momentum within each sector |
| **Rank** | Rank within sector | Position after sorting by Score descending |
| **Conviction Score** | Adjusted confidence in the Score | Score x completeness x signal agreement x anomaly penalty, then rescaled 0-100 |
| **CS Score** | Cross-sectional score | Same five factors, but scored across **all** S&P 500 stocks instead of within a sector |
| **Score Delta** | Change in Score since the previous run | `Current Score - Previous Score` from saved history |
| **MC% of S&P500** | Market-cap weight | `Stock Market Cap / Total S&P 500 Market Cap x 100` |

### How the main **Score** is built (per sector)
Each sector has its own factor weights because some factors are more predictive than others for that industry.

#### Sector-specific weights
| Sector | Valuation | Quality | PEG | Earn Traj | Momentum |
|---|---|---|---|---|---|
{sector_rows}
{default_row}

#### Example — Information Technology
```
Score = 0.20 x Valuation + 0.25 x Quality + 0.25 x PEG + 0.15 x Earn Traj + 0.15 x Momentum
```

All five sub-scores are already 0-100. The composite is then penalised for missing data:
- 1 missing factor -> x0.95
- 2 missing factors -> x0.85
- 3+ missing factors -> x0.70

### Conviction Score adjustment
The raw Score is adjusted to reflect how **complete and consistent** the signals are:

1. **Completeness multiplier** — more data, higher multiplier
   - `0.5 + 0.5 x (available factors / 6)`
2. **Signal agreement** — do P/E, momentum, and earnings trajectory agree?
   - High agreement -> multiplier up to 1.0
   - Mixed signals -> multiplier as low as 0.0
3. **Anomaly multiplier** — penalise red flags
   - Piotroski F <= 2 -> x0.70
   - Sloan Ratio > 0.08 -> x0.85

After multiplying, the result is min-max scaled to **0-100**.

### Cross-sectional (CS) Score
The CS Score ignores sectors and compares every stock to the whole S&P 500:
```
CS = 0.25 x Valuation + 0.25 x Quality + 0.20 x PEG + 0.15 x Earn Traj + 0.15 x Momentum
```
It is useful for finding the cheapest / highest-quality names across the entire market, regardless of sector.

### Score Delta
Score Delta shows how a stock's Score has changed since the **previous run**. It is persisted to `score_history.csv` so it survives refreshes and new sessions.

**Example**
- Previous Score: 72.5
- Current Score: 78.3
- **Score Delta = +5.8**
        """)

    with tabs[7]:
        st.markdown("""
### Expected data coverage v19.1

| Metric | Primary source | Fallback | Target coverage |
|---|---|---|---|
| **P/E** | FMP quote / key-metrics / ratios | Yahoo `.info` | 85%+ |
| **Fwd P/E** | FMP quote | Yahoo `.info` | 86%+ |
| **EV/EBITDA** | FMP key-metrics | Yahoo `.info` | 82%+ |
| **FCF / FCF Yield** | FMP key-metrics / cash-flow | Yahoo quarterly cash-flow | 94%+ |
| **PEG** | FMP direct | EPS growth, Earn Traj proxy | 90%+ |
| **EPS Surprise** | FMP earnings-surprises | Yahoo earnings history | 87%+ |
| **Revision Mom** | Yahoo recommendations summary | — | ~60% |
| **Momentum** | Yahoo bulk price download (`yf.download`) | Per-ticker fallback | 95%+ |

### Data-source priority
The app uses **FMP as primary** and Yahoo as a fallback. The coverage line at the bottom of the screener shows exactly how many tickers were filled for each metric.

**Why does coverage vary?**
- Not every company reports all metrics (e.g., some have no dividends).
- Yahoo rate-limits heavy `.info` calls, so Yahoo-fill coverage can fluctuate.
- FMP free-tier rate limits can reduce coverage if too many requests are fired too quickly; the app chunks requests and sleeps between chunks to stay under limits.

### Sources key
- **FMP** = Financial Modeling Prep API
- **Yahoo** = Yahoo Finance via `yfinance`
        """)

    st.caption("v19.1: momentum bulk download fixed · cached build_momentum_map · uppercase key normalisation · score history persisted to disk.")


# ═══════════════════════════════════════════════════════════════════════════════
# App entry point
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="S&P 500 Screener v19.1", layout="wide")

DATA_DIR = Path(__file__).parent / "data"

try:
    scr = pd.read_parquet(DATA_DIR / "latest_screener.parquet")
except Exception:
    try:
        scr = pd.read_csv(DATA_DIR / "latest_screener.csv")
    except Exception:
        st.markdown("## S&P 500 Fundamental Screener v19.1")
        st.info("No snapshot found. Run `python update_data.py` first.")
        st.stop()
st.markdown(
    "<style>"
    "div[data-testid='stDataFrame'] table{font-size:13px;}"
    ".stDataFrame thead th{background:#1a1a2e;color:#93c5fd;font-weight:700;}"
    "</style>",
    unsafe_allow_html=True,
)
st.markdown("## S&P 500 Fundamental Screener v19.1")

_sel_col, _refresh_col = st.columns([5, 1])
with _sel_col:
    _selected_page = st.segmented_control(
        "View", ["Screener", "Column Reference Guide"],
        default="Screener", label_visibility="collapsed"
    )
with _refresh_col:
    if st.button("Refresh", key="refresh_main"):
        st.rerun()

if _selected_page == "Column Reference Guide":
    render_reference_guide()
    st.stop()

# ── Sidebar filters ──────────────────────────────────────────────────────────
all_sectors = sorted(scr["Sector"].dropna().unique().tolist())
with st.sidebar:
    st.markdown("### Filters")
    sector_sel = st.selectbox("Sector", ["All Sectors"] + all_sectors)
    sort_by = st.selectbox("Sort by", [
        "Sector then Rank", "Overall Score high to low", "Score high to low",
        "Conviction high to low", "CS Score high to low", "MC% of S&P500 high to low",
        "Price low to high", "Price high to low", "Mkt Cap high to low",
        "PE low to high", "Fwd PE low to high", "PEG low to high",
        "Quality Score high", "ROIC high to low", "ROE high to low",
        "Earn Traj high to low", "Rev Growth high to low",
        "Momentum Score high", "52W Pos low to high",
        "P/E vs Sector Med low to high", "Piotroski F high",
        "FCF Yield high", "EV/EBITDA low", "EPS Beat Rate high",
        "Revision Mom high", "Score Delta high",
    ])
    mc_min_b = st.number_input("Min Mkt Cap ($B)", value=0, step=10, min_value=0)
    pe_max = st.number_input("Max P/E", value=9999, step=50, min_value=0)
    qual_min_f = st.number_input("Min Quality Score", value=0.0, step=5.0,
                                  min_value=0.0, max_value=100.0)

# ── Apply filters & sort ─────────────────────────────────────────────────────
filt = scr.copy()
if sector_sel != "All Sectors":
    filt = filt[filt["Sector"] == sector_sel]
filt = filt[(filt["Mkt Cap"].isna()) | (filt["Mkt Cap"] >= mc_min_b * 1e9)]
filt = filt[(filt["P/E"].isna()) | (filt["P/E"] <= pe_max)]
filt = filt[(filt["Quality Score"].isna()) | (filt["Quality Score"] >= qual_min_f)]

sort_map = {
    "Sector then Rank": (["Sector", "Rank"], [True, True]),
    "Overall Score high to low": (["Overall Score"], [False]),
    "Score high to low": (["Score"], [False]),
    "Conviction high to low": (["Conviction Score"], [False]),
    "CS Score high to low": (["CS Score"], [False]),
    "MC% of S&P500 high to low": (["MC% of S&P500"], [False]),
    "Price low to high": (["Price"], [True]),
    "Price high to low": (["Price"], [False]),
    "Mkt Cap high to low": (["Mkt Cap"], [False]),
    "PE low to high": (["P/E"], [True]),
    "Fwd PE low to high": (["Fwd P/E"], [True]),
    "PEG low to high": (["PEG"], [True]),
    "Quality Score high": (["Quality Score"], [False]),
    "ROIC high to low": (["ROIC%"], [False]),
    "ROE high to low": (["ROE%"], [False]),
    "Earn Traj high to low": (["Earn Traj"], [False]),
    "Rev Growth high to low": (["Rev Growth% (CAGR)"], [False]),
    "Momentum Score high": (["Momentum Score"], [False]),
    "52W Pos low to high": (["52W Pos%"], [True]),
    "P/E vs Sector Med low to high": (["P/E vs Sector Med"], [True]),
    "Piotroski F high": (["Piotroski F"], [False]),
    "FCF Yield high": (["FCF Yield%"], [False]),
    "EV/EBITDA low": (["EV/EBITDA"], [True]),
    "EPS Beat Rate high": (["EPS Beat Rate"], [False]),
    "Revision Mom high": (["Revision Mom"], [False]),
    "Score Delta high": (["Score Delta"], [False]),
}
sc, sa = sort_map.get(sort_by, (["Sector", "Rank"], [True, True]))
filt = filt.sort_values(sc, ascending=sa, na_position="last")

# ── Summary cards ────────────────────────────────────────────────────────────
if not filt.empty:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Passing stocks", len(filt))
    m2.metric("Avg Quality", f"{filt['Quality Score'].mean():.2f}")
    m3.metric("Avg Momentum", f"{filt['Momentum Score'].mean():.2f}")
    m4.metric("Avg Score", f"{filt['Score'].mean():.2f}")
    best_sector = (filt.groupby("Sector")["Score"].mean().idxmax()
                   if "Sector" in filt.columns else "—")
    m5.metric("Best sector", best_sector)
else:
    st.warning("No stocks match the current filters.")

st.caption("Showing **{}** of **{}** · Sector: {} · Sort: {}".format(
    len(filt), len(scr), sector_sel, sort_by))

# ── Build display table ────────────────────────────────────────────────────────
disp = filt.copy()
disp["Price ($)"] = safe_round(disp["Price"], 2)
disp["Mkt Cap ($B)"] = safe_round(disp["Mkt Cap"] / 1e9, 2)
disp["MC% of S&P500"] = safe_round(disp["MC% of S&P500"], 2)
disp["Rev Q1 Oldest ($B)"] = safe_round(disp["Rev Q1 Oldest ($B)"] / 1e9, 2)
disp["Rev Q2 ($B)"] = safe_round(disp["Rev Q2 ($B)"] / 1e9, 2)
disp["Rev Q3 ($B)"] = safe_round(disp["Rev Q3 ($B)"] / 1e9, 2)
disp["Rev Q4 Latest ($B)"] = safe_round(disp["Rev Q4 Latest ($B)"] / 1e9, 2)

disp["Quality Flag"] = disp.apply(
    lambda r: quality_flag(
        r.get("ROIC%"), r.get("ROE%"), r.get("Int Coverage"),
        r.get("Op Margin%"), sloan_ratio=r.get("Sloan Ratio"),
        sector=r.get("Sector"),
    ), axis=1,
)

ROUND_COLS = [
    "P/E", "Fwd P/E", "PEG", "Earn Traj", "52W Pos%",
    "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
    "Quality Score", "Momentum Score", "Overall Score", "Ret 1Mo%", "Ret 3Mo%",
    "Ret 6Mo%", "Trailing Vol%", "Score", "Conviction Score", "CS Score",
    "Rev Growth% (CAGR)", "P/E vs Sector Med",
    "EV/EBITDA", "FCF Yield%", "EV/Sales", "Div Yield%", "Sloan Ratio",
    "EPS Surp Avg%", "EPS Beat Rate", "EPS Surp Trend", "Revision Mom",
    "Skip Mo", "52W Prox", "vs MA200", "Rel Str SPY", "Score Delta",
]
for c in ROUND_COLS:
    if c in disp.columns:
        disp[c] = safe_round(disp[c], 2)

disp["Rank"] = pd.to_numeric(disp["Rank"], errors="coerce")
disp["Rank"] = disp["Rank"].apply(lambda v: int(v) if pd.notna(v) else pd.NA)

COLS = [
    "Ticker", "Sector", "Price ($)", "Mkt Cap ($B)", "MC% of S&P500",
    "Overall Score", "P/E", "P/E vs Sector Med", "Fwd P/E",
    "EV/EBITDA", "FCF Yield%", "EV/Sales", "Div Yield%",
    "PEG", "PEG Method", "Earn Traj",
    "EPS Surp Avg%", "EPS Beat Rate", "EPS Surp Trend", "Revision Mom",
    "ROIC%", "ROE%", "Int Coverage", "Op Margin%", "Debt/Eq",
    "Quality Score", "Quality Flag", "Piotroski F", "Sloan Ratio",
    "Momentum Score", "Skip Mo", "52W Prox", "vs MA200", "Rel Str SPY",
    "Ret 1Mo%", "Ret 3Mo%", "Ret 6Mo%", "Trailing Vol%",
    "52W Pos%", "Score", "Score Delta", "Conviction Score", "CS Score", "Rank",
    "Rev Q1 Oldest ($B)", "Rev Q2 ($B)", "Rev Q3 ($B)", "Rev Q4 Latest ($B)",
    "Rev Growth% (CAGR)",
]
disp_final = disp[[c for c in COLS if c in disp.columns]].copy()

# ── Top 10 overall ─────────────────────────────────────────────────────────────
if not scr.empty:
    with st.expander("Top 10 overall (across all sectors)", expanded=False):
        top10 = scr.nlargest(10, "Overall Score").copy()
        top10_cols = ["Ticker", "Sector", "Overall Score", "Score", "CS Score",
                      "Conviction Score", "Quality Score", "Momentum Score",
                      "P/E", "PEG", "Earn Traj"]
        top10_cols = [c for c in top10_cols if c in top10.columns]
        for c in top10_cols:
            if c not in ("Ticker", "Sector"):
                top10[c] = safe_round(top10[c], 2)
        st.dataframe(top10[top10_cols], use_container_width=True, height=380)

# ── Sector distribution ──────────────────────────────────────────────────────
if not filt.empty:
    with st.expander("Sector distribution", expanded=False):
        sector_df = (filt.groupby("Sector")
                          .agg(Count=("Ticker", "count"),
                               MC_Sum=("Mkt Cap", "sum"))
                          .reset_index())
        total_mc = filt["Mkt Cap"].sum()
        sector_df["MC%"] = (sector_df["MC_Sum"] / total_mc * 100.0).round(2)
        sector_df = sector_df.sort_values("Count", ascending=False)

        x_sort = alt.EncodingSortField(field="Count", op="sum", order="descending")
        bar = (alt.Chart(sector_df)
                  .mark_bar(color="#3b82f6")
                  .encode(x=alt.X("Sector:N", sort=x_sort,
                                    axis=alt.Axis(labelAngle=-45)),
                          y=alt.Y("Count:Q", title="Number of stocks")))
        line = (alt.Chart(sector_df)
                   .mark_line(color="#f97316", point=alt.OverlayMarkDef(color="#f97316"))
                   .encode(x=alt.X("Sector:N", sort=x_sort,
                                    axis=alt.Axis(labelAngle=-45)),
                           y=alt.Y("MC%:Q", title="% of total market cap")))
        chart = (alt.layer(bar, line)
                      .resolve_scale(y="independent")
                      .properties(height=320))
        st.altair_chart(chart, use_container_width=True)

# ── Main table ─────────────────────────────────────────────────────────────────
fmt_cols = [c for c in disp_final.columns
            if c not in ("Ticker", "Sector", "Quality Flag", "PEG Method")]
styled = disp_final.style.format("{:.2f}", subset=fmt_cols, na_rep="")

green_cols = [c for c in disp_final.columns if METRIC_DIRECTION.get(c) == "green"]
red_cols = [c for c in disp_final.columns if METRIC_DIRECTION.get(c) == "red"]
if green_cols:
    styled = styled.apply(_green_grad, axis=0, subset=green_cols)
if red_cols:
    styled = styled.apply(_red_grad, axis=0, subset=red_cols)
st.dataframe(styled, use_container_width=True, height=680)

st.download_button(
    label="Download CSV",
    data=disp_final.to_csv(index=False).encode("utf-8"),
    file_name="sp500_screener_v19_1_{}.csv".format(
        datetime.now().strftime("%Y%m%d_%H%M")),
    mime="text/csv",
)
