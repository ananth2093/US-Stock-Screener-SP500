# evaluate_predictions.py
"""Evaluate how predictive the screener scores are using the saved archive.

Joins each archived run with forward price returns from `price_history.parquet`,
then computes factor Information Coefficients (IC) and a simple top-N backtest.
Outputs are saved under `data/evaluation/`.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_DIR = Path(__file__).parent / "data"
EVAL_DIR = DATA_DIR / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

ARCHIVE_PATH = DATA_DIR / "prediction_archive.parquet"
PRICE_PATH = DATA_DIR / "price_history.parquet"

HORIZONS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "1y": 252,
}

FACTORS = [
    "Overall Score",
    "Score",
    "CS Score",
    "Conviction Score",
    "Quality Score",
    "Momentum Score",
    "P/E",
    "PEG",
    "Earn Traj",
    "ROIC%",
    "ROE%",
    "FCF Yield%",
    "Piotroski F",
]


def load_data():
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Archive not found: {ARCHIVE_PATH}. Run update_data.py first.")
    if not PRICE_PATH.exists():
        raise FileNotFoundError(f"Price history not found: {PRICE_PATH}. Run update_data.py first.")

    archive = pd.read_parquet(ARCHIVE_PATH)
    archive["run_timestamp"] = pd.to_datetime(archive["run_timestamp"], utc=True)
    archive["run_date"] = archive["run_timestamp"].dt.date
    # Keep only the last run of each day for cleaner time-series evaluation
    archive = archive.sort_values(["run_timestamp"]).drop_duplicates(
        subset=["run_date", "Ticker"], keep="last"
    )

    prices = pd.read_parquet(PRICE_PATH)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.sort_values(["ticker", "date"]).drop_duplicates(
        subset=["ticker", "date"], keep="last"
    )

    return archive, prices


def compute_forward_returns(archive: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Add forward-return columns to the archive using available closing prices."""
    price_series = {
        ticker: group.set_index("date")["close"].sort_index()
        for ticker, group in prices.groupby("ticker")
    }

    result_frames = []
    for ticker, group in archive.groupby("Ticker"):
        if ticker not in price_series:
            continue
        s = price_series[ticker]
        if s.empty:
            continue

        dates = np.asarray(group["run_date"])
        idx = s.index
        # position of the latest close on or before each run_date
        pos = idx.searchsorted(dates, side="right") - 1
        pos = np.clip(pos, 0, len(idx) - 1)
        now_prices = s.iloc[pos].values

        for label, h in HORIZONS.items():
            fwd_pos = pos + h
            valid = fwd_pos < len(idx)
            rets = np.full(len(group), np.nan, dtype=float)
            rets[valid] = s.iloc[fwd_pos[valid]].values / now_prices[valid] - 1.0
            group = group.copy()
            group[f"ret_{label}"] = rets

        result_frames.append(group)

    if not result_frames:
        raise RuntimeError("No forward returns could be computed; price history may be empty.")

    archive = pd.concat(result_frames, ignore_index=True)

    # SPY benchmark returns for each run
    if "SPY" in price_series:
        spy_s = price_series["SPY"]
        idx = spy_s.index
        spy_dates = np.asarray(archive["run_date"])
        spy_pos = idx.searchsorted(spy_dates, side="right") - 1
        spy_pos = np.clip(spy_pos, 0, len(idx) - 1)
        spy_now = spy_s.iloc[spy_pos].values
        for label, h in HORIZONS.items():
            spy_fwd_pos = spy_pos + h
            valid = spy_fwd_pos < len(idx)
            spy_rets = np.full(len(archive), np.nan, dtype=float)
            spy_rets[valid] = spy_s.iloc[spy_fwd_pos[valid]].values / spy_now[valid] - 1.0
            archive[f"spy_ret_{label}"] = spy_rets
    else:
        for label in HORIZONS:
            archive[f"spy_ret_{label}"] = np.nan

    return archive


def compute_ic(archive: pd.DataFrame) -> pd.DataFrame:
    """Mean Spearman IC for each factor and horizon (pandas rank-correlation)."""
    rows = []
    for label in HORIZONS:
        ret_col = f"ret_{label}"
        row = {"horizon": label}
        for factor in FACTORS:
            if factor not in archive.columns:
                continue
            ic_values = []
            for _, run_df in archive.groupby(["run_date", "run_id"]):
                run_df = run_df[[factor, ret_col]].dropna()
                if len(run_df) < 5:
                    continue
                try:
                    ic = run_df[factor].rank().corr(run_df[ret_col].rank())
                    if not np.isnan(ic):
                        ic_values.append(ic)
                except Exception:
                    pass
            row[f"{factor}_IC"] = float(np.nanmean(ic_values)) if ic_values else np.nan
        rows.append(row)

    ic_df = pd.DataFrame(rows).set_index("horizon")
    return ic_df


def top_n_backtest(archive: pd.DataFrame, factor: str, n: int = 20) -> pd.DataFrame:
    """Equal-weight forward returns of the top-N stocks by `factor` each run."""
    records = []
    for (run_date, run_id), run_df in archive.groupby(["run_date", "run_id"]):
        if factor not in run_df.columns:
            continue
        top = run_df.dropna(subset=[factor]).nlargest(n, factor)
        if top.empty:
            continue
        for label in HORIZONS:
            ret_col = f"ret_{label}"
            spy_col = f"spy_ret_{label}"
            if ret_col not in top.columns:
                continue
            mean_ret = top[ret_col].mean()
            spy_ret = run_df[spy_col].dropna().iloc[0] if spy_col in run_df.columns else np.nan
            records.append({
                "run_date": run_date,
                "run_id": run_id,
                "factor": factor,
                "horizon": label,
                "top_n_return": mean_ret,
                "spy_return": spy_ret,
                "active_return": mean_ret - spy_ret if not np.isnan(spy_ret) else np.nan,
                "n_stocks": len(top.dropna(subset=[ret_col])),
            })
    return pd.DataFrame(records)


def main():
    print("Loading archive and price history...")
    archive, prices = load_data()
    print(f"Archive rows: {len(archive):,}  |  Price rows: {len(prices):,}")

    print("Computing forward returns...")
    archive = compute_forward_returns(archive, prices)

    out_path = EVAL_DIR / "archive_with_returns.parquet"
    archive.to_parquet(out_path, index=False)
    print(f"Saved: {out_path}")

    print("\nInformation Coefficients (mean Spearman rank correlation factor -> future return)")
    ic_df = compute_ic(archive)
    ic_df.to_csv(EVAL_DIR / "ic_summary.csv")
    print(ic_df.round(3).to_string())

    print("\nTop-20 equal-weight backtest (active return vs SPY)")
    backtest_records = []
    for factor in ["Overall Score", "Score", "CS Score", "Quality Score", "Momentum Score"]:
        if factor not in archive.columns:
            continue
        bt = top_n_backtest(archive, factor, n=20)
        if bt.empty:
            continue
        bt.to_csv(EVAL_DIR / f"backtest_top20_{factor.lower().replace(' ', '_')}.csv", index=False)
        summary = bt.groupby("horizon").agg(
            mean_active_return=("active_return", "mean"),
            mean_top_return=("top_n_return", "mean"),
            mean_spy_return=("spy_return", "mean"),
            n_runs=("run_date", "nunique"),
        ).reindex(list(HORIZONS.keys()))
        print(f"\n--- Factor: {factor} ---")
        print(summary.round(4).to_string())
        backtest_records.append(bt)

    if backtest_records:
        all_bt = pd.concat(backtest_records, ignore_index=True)
        all_bt.to_csv(EVAL_DIR / "backtest_top20_all_factors.csv", index=False)

    # Save a flat ML-ready dataset (features + targets)
    feature_cols = [c for c in archive.columns if c not in
                    ["run_timestamp", "run_id", "run_date"] + [f"ret_{h}" for h in HORIZONS] +
                    [f"spy_ret_{h}" for h in HORIZONS]]
    ml_cols = feature_cols + [f"ret_{h}" for h in HORIZONS] + [f"spy_ret_{h}" for h in HORIZONS]
    ml_dataset = archive[ml_cols].copy()
    ml_path = EVAL_DIR / "ml_dataset.parquet"
    ml_dataset.to_parquet(ml_path, index=False)
    print(f"\nML dataset saved: {ml_path} ({len(ml_dataset):,} rows)")


if __name__ == "__main__":
    main()
