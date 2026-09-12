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
    st.caption(
        "Definitions, formulas, scoring logic, and a full glossary for every column in the dashboard. "
        "v19.2"
    )

    tabs = st.tabs([
        "Overview", "Valuation", "Quality", "PEG", "Earn Trajectory",
        "Earnings Surprise", "Momentum", "Ranking & Score", "Full Column Glossary", "Data Sources & Coverage", "Worked Examples",
    ])

    with tabs[0]:
        st.markdown("""
### What this screener does
This dashboard ranks S&P 500 stocks using five factors:
1. **Valuation** — how cheap or expensive a stock is relative to earnings, cash flow, and sales.
2. **Quality** — profitability, financial health, and earnings quality.
3. **PEG** — price/earnings relative to expected growth.
4. **Earnings Trajectory** — whether forward earnings are expected to rise or fall.
5. **Momentum** — price trend strength, adjusted for volatility and relative to the market.

Most scoring is done **within each sector**, because a bank and a tech company should not be compared with the same valuation yardstick.

### Key jargon
- **Market cap (Mkt Cap)** — total value of all outstanding shares.
- **Enterprise Value (EV)** — market cap + debt − cash; what it would cost to buy the whole company.
- **Free Cash Flow (FCF)** — cash generated after capital expenditures.
- **Sector-relative score** — percentile ranking using only the stock's sector peers.
- **Cross-sectional score** — percentile ranking across the entire S&P 500.
- **Risk-adjusted return** — return divided by volatility; a 20% gain with 5% volatility is better than a 20% gain with 50% volatility.
""")

    with tabs[1]:
        st.markdown("""
### Valuation metrics
| Column | Definition | Why it matters |
|---|---|---|
| **P/E** | Trailing 12-month price-to-earnings: Market Cap / Net Income TTM. | Lower often means cheaper; compare within sector. |
| **Fwd P/E** | Forward price-to-earnings: Market Cap / Forward EPS. | Looks at expected future earnings. |
| **P/E vs Sector Med** | Stock P/E ÷ sector median P/E. | 1.0 = sector median; <1 cheaper than peers; >1 pricier. |
| **EV/EBITDA** | Enterprise Value / EBITDA TTM. | Enterprise-wide valuation; lower is generally cheaper. |
| **FCF Yield%** | Free Cash Flow TTM / Market Cap × 100. | Higher = more cash returned relative to price. |
| **EV/Sales** | Enterprise Value / Total Revenue. | Useful for companies without earnings. |
| **Div Yield%** | Most recent dividend yield. | Income component of total return. |
| **Mkt Cap ($B)** | Market value of all shares, in billions of USD. | Size classification. |
| **MC% of S&P500** | Stock market cap ÷ total S&P 500 market cap × 100. | Index weight / influence. |

### How the Valuation sub-score (0-100) is built
Inside each sector, every eligible stock receives a valuation score using robust percentile scoring:
1. **FCF Yield%** — higher is better → 40%
2. **EV/EBITDA** — lower is better → 35%
3. **P/E** (Fwd P/E if available, else P/E) — lower is better → 25%

Extreme outliers are winsorised so a single P/E of 500 does not wreck the ranking.
""")

    with tabs[2]:
        st.markdown("""
### Quality metrics
| Column | Definition | Why it matters |
|---|---|---|
| **ROIC%** | Return on Invested Capital: Net Income TTM / (Equity + Debt − Cash) × 100. | Measures efficiency of capital use. Higher = better. |
| **ROE%** | Return on Equity: Net Income / Shareholders' Equity × 100. | Profit per unit of shareholder money. |
| **Int Coverage** | EBIT TTM / Interest Expense TTM. | Ability to pay interest; >3x generally safe. |
| **Op Margin%** | Operating Income / Revenue × 100. | Profit per dollar of sales; higher = pricing power. |
| **Debt/Eq** | Total Debt / Shareholders' Equity. | Leverage; lower is generally safer. |
| **Piotroski F** | 9-point fundamental health score. | ≥5 is considered healthy. |
| **Sloan Ratio** | `(Net Income − Operating Cash Flow) / Average Total Assets`. | Accrual/earnings-quality check; >0.08 flagged. |
| **Quality Score** | Composite 0–100 quality score. | One-number summary of fundamental health. |
| **Quality Flag** | Human-readable failure labels. | Pass = all thresholds met. |

### Quality Score formula
1. **Profitability** (ROIC, or ROE for Financials) — 25%
2. **Interest Coverage** — 15%
3. **Operating Margin** — 15% (excluded for Financials)
4. **Gross Margin** — 20%
5. **Piotroski F-score** — 15%
6. **Sloan Ratio** — 10%

### Quality Flag thresholds
- ROIC < 8% → flagged (ROE < 8% for Financials)
- Interest Coverage < 3 → flagged
- Op Margin < 5% → flagged
- Sloan Ratio > 0.08 → flagged
""")

    with tabs[3]:
        st.markdown("""
### PEG Ratio
**Formula:** `P/E / Annual EPS Growth %`

| PEG | Signal |
|---|---|
| Below 1.0 | Potentially undervalued |
| 1.0–2.0 | Fairly valued |
| Above 2.0 | Expensive |

- Only computed when growth input is ≥ 5%.
- The **PEG Method** column shows which source was used: direct FMP, computed from EPS growth, or an Earn Traj proxy.

**Example:** Fwd P/E = 15.0, EPS Growth = 10% → PEG = 1.5.
""")

    with tabs[4]:
        st.markdown("""
### Earn Trajectory
**Formula:** `(Forward EPS − Trailing EPS) / |Trailing EPS|` clipped to [−1, +1]

| Value | Meaning |
|---|---|
| +1.0 | Forward EPS much higher than trailing EPS (strong expected growth) |
| 0.0 | No expected change |
| −1.0 | Forward EPS much lower (expected decline) |

- Clipped to keep extreme swings from distorting the score.
- If trailing EPS is negative and forward EPS is still negative, the positive clip is limited to +0.30.
""")

    with tabs[5]:
        st.markdown("""
### Earnings Surprise metrics
| Column | Definition | Why it matters |
|---|---|---|
| **EPS Surp Avg%** | Mean earnings surprise over the last 4 quarters: `(Actual − Estimate) / |Estimate| × 100`. | Consistent positive surprises often precede estimate upgrades. |
| **EPS Beat Rate** | % of recent quarters with positive surprise. | Frequency of beating expectations. |
| **EPS Surp Trend** | +1 if the last 2 quarters' average surprise is higher than the previous 2; −1 if lower. | Direction of surprise momentum. |
| **Revision Mom** | Net change in strong buy/buy vs sell/strong-sell ratings between the two latest months, scaled to [−1, +1]. | Analyst estimate revision momentum. |

High positive Revision Mom means analysts are upgrading the stock recently.
""")

    with tabs[6]:
        st.markdown("""
### Momentum metrics
| Column | Definition | Why it matters |
|---|---|---|
| **Ret 1Mo%** | 1-month price return. | Short-term trend. |
| **Ret 3Mo%** | 3-month price return. | Medium-term trend. |
| **Ret 6Mo%** | 6-month price return. | Longer-term trend. |
| **Trailing Vol%** | Annualised standard deviation of daily returns: `std(daily returns) × sqrt(252) × 100`. | Risk measure. |
| **52W Pos%** | `(Price − 52W Low) / (52W High − 52W Low) × 100`. | 0% = 52-week low; 100% = 52-week high. |
| **Skip Mo** | 6-month return minus 1-month return, normalised by volatility. | Removes short-term reversal noise. |
| **52W Prox** | How close price is to its 52-week high. | Trend persistence signal. |
| **vs MA200** | Price relative to its 200-day moving average. | Long-term trend direction. |
| **Rel Str SPY** | 3-month return vs SPY 3-month return. | Market-relative strength. |
| **Momentum Score** | Composite 0–100 momentum score. | Blend of Skip Mo, 52W Prox, vs MA200, Rel Str SPY. |

### Momentum Score components
| Signal | Weight |
|---|---|
| Skip Mo | 40% |
| 52W Proximity | 25% |
| vs MA200 | 20% |
| Rel Str SPY | 15% |

### Why momentum matters in stock screening
- **Price follows fundamentals, but with a lag.** Momentum shows which stocks the market is already rewarding.
- **Avoids value traps.** A cheap stock can stay cheap for years; positive momentum indicates buyers are stepping in.
- **Risk-adjusted view.** Normalising return by volatility highlights smooth, durable trends rather than erratic spikes.
- **Best combined with quality and valuation.** High momentum + reasonable valuation + strong fundamentals = higher-conviction idea.
""")

    with tabs[7]:
        st.markdown("""
### Ranking & Score columns
| Column | Definition | Why it matters |
|---|---|---|
| **Score** | Sector-relative composite 0–100. | Main ranking signal within a sector. |
| **Overall Score** | Same as Score. | Main ranking signal. |
| **Rank** | Rank within sector after sorting by Score descending. | 1 = best in sector. |
| **Conviction Score** | Adjusted confidence in the Score. Penalises missing data and mixed signals, then rescaled 0–100. | Higher = more reliable. |
| **CS Score** | Cross-sectional score: same five factors scored across all S&P 500 stocks. | Finds cheapest/highest-quality names market-wide. |
| **Score Delta** | Change in Score since the previous run. | Shows momentum in the ranking itself. |
| **MC% of S&P500** | Market-cap weight in the S&P 500. | Index influence. |

### How the main Score is built (per sector)
Each sector has its own factor weights.

| Sector | Valuation | Quality | PEG | Earn Traj | Momentum |
|---|---|---|---|---|---|
| Information Technology | 20% | 25% | 25% | 15% | 15% |
| Consumer Discretionary | 20% | 20% | 22% | 18% | 20% |
| Communication Services | 22% | 23% | 22% | 18% | 15% |
| Health Care | 25% | 30% | 18% | 15% | 12% |
| Industrials | 25% | 28% | 18% | 17% | 12% |
| Consumer Staples | 28% | 32% | 10% | 15% | 15% |
| Financials | 30% | 25% | 18% | 17% | 10% |
| Energy | 30% | 18% | 12% | 15% | 25% |
| Materials | 28% | 20% | 12% | 15% | 25% |
| Real Estate | 30% | 18% | 10% | 22% | 20% |
| Utilities | 38% | 27% | 5% | 15% | 15% |

### Missing factor penalty
| Missing factors | Score multiplier |
|---|---|
| 0 | ×1.00 |
| 1 | ×0.95 |
| 2 | ×0.85 |
| 3+ | ×0.70 |

### Conviction Score adjustment
1. **Completeness multiplier** — more data, higher multiplier.
2. **Signal agreement** — do P/E, momentum, and earnings trajectory agree?
3. **Anomaly multiplier** — penalise Piotroski F ≤ 2 and Sloan Ratio > 0.08.
""")

    with tabs[8]:
        st.markdown("""
### Full column glossary
| Column | Definition | Interpretation |
|---|---|---|
| **Ticker** | Stock symbol. | e.g. AAPL, MSFT. |
| **Sector** | GICS sector classification. | Used for peer-relative scoring. |
| **Price ($)** | Latest closing price in USD. | |
| **Mkt Cap ($B)** | Market cap in billions of USD. | Size classification. |
| **MC% of S&P500** | Stock weight in the S&P 500 index. | Sum = 100%. |
| **Overall Score / Score** | Sector-relative composite 0–100. | Higher = better ranking. |
| **P/E** | Trailing price-to-earnings. | Lower can be cheaper; compare within sector. |
| **P/E vs Sector Med** | Stock P/E ÷ sector median P/E. | <1 cheaper than peers. |
| **Fwd P/E** | Forward price-to-earnings. | Based on estimated future earnings. |
| **EV/EBITDA** | Enterprise Value / EBITDA. | Lower = cheaper on an enterprise basis. |
| **FCF Yield%** | Free cash flow / market cap × 100. | Higher = more cash yield. |
| **EV/Sales** | Enterprise Value / Revenue. | Lower = cheaper per dollar of sales. |
| **Div Yield%** | Dividend yield. | Income return. |
| **PEG** | P/E / EPS growth. | <1 potentially undervalued. |
| **PEG Method** | Source used for PEG. | Audit trail. |
| **Earn Traj** | Normalised earnings direction. | +1 strong growth, −1 decline. |
| **EPS Surp Avg%** | Average EPS surprise over last 4 quarters. | Positive = consistent beats. |
| **EPS Beat Rate** | % of quarters beating estimates. | Higher = more reliable outperformance. |
| **EPS Surp Trend** | Direction of surprise momentum. | +1 improving, −1 deteriorating. |
| **Revision Mom** | Analyst rating/revision momentum. | Higher = more upgrades. |
| **ROIC%** | Return on invested capital. | Efficiency; >8% target. |
| **ROE%** | Return on equity. | Profit per unit of equity. |
| **Int Coverage** | EBIT / interest expense. | >3x preferred. |
| **Op Margin%** | Operating profit margin. | >5% target. |
| **Debt/Eq** | Debt-to-equity. | Lower generally safer. |
| **Quality Score** | Composite 0–100 quality. | Higher = stronger fundamentals. |
| **Quality Flag** | Threshold failure labels. | Pass = all good. |
| **Piotroski F** | 9-point fundamental health score. | ≥5 considered healthy. |
| **Sloan Ratio** | Accrual/earnings quality metric. | Close to 0 best; >0.08 flagged. |
| **Momentum Score** | Composite 0–100 momentum. | Higher = stronger risk-adjusted trend. |
| **Skip Mo** | Skip-month momentum signal. | Higher = stronger intermediate trend. |
| **52W Prox** | Proximity to 52-week high. | Higher = closer to high. |
| **vs MA200** | Price vs 200-day moving average. | >0 = above average. |
| **Rel Str SPY** | 3-month return vs SPY. | >0 = beating the market. |
| **Ret 1Mo% / 3Mo% / 6Mo%** | Price returns. | Trend measures. |
| **Trailing Vol%** | Annualised volatility. | Risk measure. |
| **52W Pos%** | Position in 52-week range. | 100% = 52-week high. |
| **Score Delta** | Change in Score since previous run. | Shows ranking momentum. |
| **Conviction Score** | Confidence-adjusted Score. | Higher = more reliable. |
| **CS Score** | Cross-sectional composite score. | Ranks vs entire S&P 500. |
| **Rank** | Sector rank by Score. | 1 = best in sector. |
| **Rev Q1 Oldest ($B)** ... **Rev Q4 Latest ($B)** | Quarterly revenue in $B. | Growth inputs. |
| **Rev Growth% (CAGR)** | Revenue CAGR across 4 quarters. | Growth measure. |
| **Data Sources** | Audit of source per metric. | FMP, Yahoo, Computed. |
""")

    with tabs[9]:
        st.markdown("""
### Data sources & coverage
The app uses **FMP as primary** and Yahoo as a fallback. The coverage banner at the bottom of the screener shows exactly how many tickers were filled for each metric.

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

### Sources key
- **FMP** = Financial Modeling Prep API
- **Yahoo** = Yahoo Finance via `yfinance`

### Why some cells are empty
- Not every company reports all metrics (e.g., some have no dividends).
- Yahoo rate-limits heavy `.info` calls, so Yahoo-fill coverage can fluctuate.
- FMP free-tier rate limits can reduce coverage; the app chunks requests and sleeps between chunks.
- **A blank is not a zero.** It means "data unavailable" and the score is penalised for missing that factor.
""")

    with tabs[10]:
        st.markdown("""
### Worked examples — how the numbers are calculated

#### P/E (Price-to-Earnings)
**Formula:** `P/E = Stock Price / Trailing 12-Month EPS`

- If **Apple** stock price = **$200** and trailing 12-month EPS = **$8**, then:
  - P/E = 200 / 8 = **25**
- A P/E of 25 means investors are paying $25 for every $1 of past-year earnings.
- Compare to the sector median. If Tech median P/E = 28, then P/E vs Sector Med = 25 / 28 = **0.89** → cheaper than peers.

#### Fwd P/E (Forward P/E)
**Formula:** `Fwd P/E = Stock Price / Expected Next-12-Month EPS`

- If stock price = $200 and expected EPS = $10, then:
  - Fwd P/E = 200 / 10 = **20**
- Lower than trailing P/E suggests expected earnings growth.

#### EV/EBITDA
**Formula:** `EV/EBITDA = Enterprise Value / EBITDA`

- Market Cap = $1,000B, Debt = $200B, Cash = $50B
  - Enterprise Value = 1,000 + 200 − 50 = **$1,150B**
- EBITDA = $115B
  - EV/EBITDA = 1,150 / 115 = **10.0x**
- Lower often means cheaper on an enterprise basis.

#### FCF Yield%
**Formula:** `FCF Yield = Free Cash Flow TTM / Market Cap × 100`

- FCF TTM = $50B, Market Cap = $1,000B
  - FCF Yield = 50 / 1,000 × 100 = **5%**
- Higher yield means more cash returned relative to the price.

#### PEG
**Formula:** `PEG = P/E / Annual EPS Growth %`

- P/E = 25, EPS growth = 15%
  - PEG = 25 / 15 = **1.67**
- <1.0 attractive; 1.0–2.0 fair; >2.0 expensive.

#### ROIC%
**Formula:** `ROIC = Net Operating Profit After Tax / (Equity + Debt − Cash) × 100`

- NOPAT = $20B, Equity = $100B, Debt = $30B, Cash = $10B
  - Invested Capital = 100 + 30 − 10 = **$120B**
  - ROIC = 20 / 120 × 100 = **16.7%**
- >15% excellent; <8% flagged.

#### ROE%
**Formula:** `ROE = Net Income / Shareholders' Equity × 100`

- Net Income = $15B, Equity = $75B
  - ROE = 15 / 75 × 100 = **20%**
- For Financials, ROE is the primary quality metric.

#### Int Coverage
**Formula:** `Int Coverage = EBIT / Interest Expense`

- EBIT = $30B, Interest Expense = $5B
  - Int Coverage = 30 / 5 = **6x**
- >3x safe; <3x flagged.

#### Op Margin%
**Formula:** `Op Margin = Operating Income / Revenue × 100`

- Operating Income = $40B, Revenue = $200B
  - Op Margin = 40 / 200 × 100 = **20%**
- Higher = better pricing power.

#### Piotroski F
A 9-point score based on profitability, leverage, and efficiency.
- Positive net income, positive operating cash flow, improving ROA, lower debt, improving current ratio, no new share issuance, higher gross margin, higher asset turnover, etc.
- **Score ≥ 5** considered healthy.

#### Sloan Ratio
**Formula:** `Sloan Ratio = (Net Income − Operating Cash Flow) / Average Total Assets`

- Net Income = $10B, Operating Cash Flow = $6B, Average Total Assets = $100B
  - Sloan Ratio = (10 − 6) / 100 = **0.04**
- >0.08 → accruals are high, earnings quality may be lower.

#### Earn Traj
**Formula:** `(Forward EPS − Trailing EPS) / |Trailing EPS|`, clipped to [−1, +1]

- Trailing EPS = $8, Forward EPS = $10
  - Earn Traj = (10 − 8) / 8 = **+0.25**
- Positive = expected growth; negative = expected decline.

#### EPS Surp Avg%
**Formula:** `(Actual EPS − Estimate EPS) / |Estimate EPS| × 100`, averaged over last 4 quarters

- Beat estimates by 2%, 3%, 1%, 4% over 4 quarters
  - Avg = (2 + 3 + 1 + 4) / 4 = **2.5%**

#### EPS Beat Rate
- 3 of last 4 quarters beat estimates → **75%**

#### Revision Mom
- Strong buy/buy ratings increased from 12 to 15 while sell/strong-sell stayed at 2 → positive Revision Mom.

#### Skip Mo (Momentum signal)
**Formula:** `(6-month return − 1-month return) / Trailing Volatility`

- 6-month return = 18%, 1-month return = 5%, Trailing Vol = 25%
  - Skip Mo = (18 − 5) / 25 = **0.52**

#### 52W Prox
**Formula:** `Current Price / 52W High`

- Price = $95, 52W High = $100 → **0.95** (near highs)

#### vs MA200
**Formula:** `(Current Price − 200-day MA) / 200-day MA`

- Price = $110, MA200 = $100 → **+10%** (above long-term average)

#### Rel Str SPY
**Formula:** `(Stock 3Mo Return − SPY 3Mo Return)`

- Stock +12%, SPY +8% → **+4%** (outperforming market)

#### Momentum Score
A 0–100 composite blending Skip Mo (40%), 52W Prox (25%), vs MA200 (20%), Rel Str SPY (15%).
- Strong scores across all four components → Momentum Score near **100**.

#### Score / Overall Score
A sector-relative composite 0–100 using sector-adaptive weights for Valuation, Quality, PEG, Earn Traj, and Momentum.
- A stock scoring 85 is better than 85% of peers in its sector on the combined factors.

#### CS Score
Same five factors scored across **all S&P 500 stocks** instead of within a sector.
- Useful for finding the cheapest/highest-quality names market-wide.

#### Score Delta
**Formula:** `Current Score − Previous Run Score`

- Last run Score = 70, current Score = 78 → **Score Delta = +8**
- Positive means the stock's ranking improved since the last update.

#### Rev Growth% (CAGR)
**Formula:** `(Newest Quarter Revenue / Revenue 4 Quarters Ago)^(1/3) − 1 × 100`

- Q4 = $120B, Q1 = $100B
  - CAGR = (120 / 100)^(1/3) − 1 = **6.3%**

#### MC% of S&P500
**Formula:** `Stock Market Cap / Total S&P 500 Market Cap × 100`

- Apple market cap = $3,000B, total S&P 500 market cap = $40,000B
  - MC% = 3,000 / 40,000 × 100 = **7.5%**
""")

    st.caption("v19.2: full column glossary added · momentum bulk download · score history persisted to disk.")

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
