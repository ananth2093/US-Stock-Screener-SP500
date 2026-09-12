# evaluate_predictions.py
"""Evaluate how predictive the S&P 500 screener scores are and train a first ML model.

Joins each archived run with forward price returns from `price_history.parquet`,
computes factor Information Coefficients (IC), runs top-N backtests, and trains an
XGBoost regression model to predict forward returns. Outputs are saved under
`data/evaluation/`.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

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

# Horizons for which we train ML models.
ML_HORIZONS = ["1w", "1m", "3m"]

# Exclude from features to prevent leakage or encoding noise.
EXCLUDE_COLS = {
    "run_timestamp",
    "run_id",
    "run_date",
    "Ticker",
    "YF Ticker",
    "Sector",
    "Data Sources",
    "PEG Method",
    "Quality Flag",
}


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
            spy_vals = run_df[spy_col].dropna()
            spy_ret = spy_vals.iloc[0] if spy_col in run_df.columns and len(spy_vals) > 0 else np.nan
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


def _numeric_features(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that are safe to use as model features."""
    cols = []
    for c in df.columns:
        if c in EXCLUDE_COLS:
            continue
        if c.startswith("ret_") or c.startswith("spy_ret_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            # Require at least some non-NaN values across the dataset.
            if df[c].notna().sum() > 50:
                cols.append(c)
    return cols


def _build_feature_matrix(
    df: pd.DataFrame,
    numeric_cols: list[str],
    encoder: OneHotEncoder | None,
    fit_encoder: bool = False,
) -> tuple[pd.DataFrame, OneHotEncoder]:
    """Build numeric + one-hot sector features for the supplied rows."""
    X = df[numeric_cols].copy()

    if "Sector" in df.columns:
        sectors = df[["Sector"]].fillna("Unknown").astype(str)
        if fit_encoder:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            sector_arr = encoder.fit_transform(sectors)
        else:
            sector_arr = encoder.transform(sectors)
        sector_df = pd.DataFrame(
            sector_arr,
            columns=encoder.get_feature_names_out(["Sector"]),
            index=df.index,
        )
        X = pd.concat([X, sector_df], axis=1)

    return X, encoder


def _date_split(dates: pd.Series, train_frac: float = 0.6, val_frac: float = 0.2):
    """Return boolean masks for train / validation / test sorted by date."""
    unique_dates = np.sort(dates.unique())
    n = len(unique_dates)
    if n < 3:
        # Not enough distinct dates for a time split; use simple fractional split.
        idx = np.arange(len(dates))
        n_train = int(len(idx) * train_frac)
        n_val = int(len(idx) * val_frac)
        return (
            idx[:n_train],
            idx[n_train : n_train + n_val],
            idx[n_train + n_val :],
        )

    train_cut = unique_dates[int(n * train_frac)]
    val_cut = unique_dates[int(n * (train_frac + val_frac))]

    train_mask = dates <= train_cut
    val_mask = (dates > train_cut) & (dates <= val_cut)
    test_mask = dates > val_cut

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]
    test_idx = np.where(test_mask)[0]
    return train_idx, val_idx, test_idx


def _ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Spearman rank IC."""
    df = pd.DataFrame({"y": y_true, "p": y_pred}).dropna()
    if len(df) < 5:
        return np.nan
    return df["y"].rank().corr(df["p"].rank())


def _quintile_spread(
    predictions: pd.Series, returns: pd.Series, n_bins: int = 5
) -> float:
    """Mean top-bin return minus bottom-bin return across rows grouped by run_date."""
    df = pd.DataFrame({"pred": predictions, "ret": returns}).dropna()
    if len(df) < n_bins * 2:
        return np.nan
    df["bin"] = pd.qcut(df["pred"], q=n_bins, labels=False, duplicates="drop")
    top = df[df["bin"] == df["bin"].max()]["ret"].mean()
    bot = df[df["bin"] == df["bin"].min()]["ret"].mean()
    return float(top - bot)


def _annualisation_factor(horizon: str) -> float:
    return {
        "1w": np.sqrt(52),
        "1m": np.sqrt(12),
        "3m": np.sqrt(4),
        "6m": np.sqrt(2),
    }.get(horizon, 1.0)


def train_ml_model(archive: pd.DataFrame, horizon: str):
    """Train an XGBoost regressor to predict `ret_{horizon}` and evaluate it."""
    ret_col = f"ret_{horizon}"
    bench_col = f"spy_ret_{horizon}"

    if ret_col not in archive.columns:
        print(f"  [{horizon}] Target column {ret_col} not found; skipping.")
        return None

    df = archive.dropna(subset=[ret_col]).copy()
    if len(df) < 200:
        print(f"  [{horizon}] Only {len(df)} rows with target; skipping ML training.")
        return None

    numeric_cols = _numeric_features(df)
    if len(numeric_cols) < 3:
        print(f"  [{horizon}] Not enough numeric features; skipping.")
        return None

    dates = df["run_date"]
    train_idx, val_idx, test_idx = _date_split(dates)

    if len(test_idx) < 20:
        print(f"  [{horizon}] Test set too small; skipping ML training.")
        return None

    encoder = None
    X_train, encoder = _build_feature_matrix(df.iloc[train_idx], numeric_cols, encoder, fit_encoder=True)
    X_val, _ = _build_feature_matrix(df.iloc[val_idx], numeric_cols, encoder, fit_encoder=False)
    X_test, _ = _build_feature_matrix(df.iloc[test_idx], numeric_cols, encoder, fit_encoder=False)

    y_train = df.iloc[train_idx][ret_col].values
    y_val = df.iloc[val_idx][ret_col].values
    y_test = df.iloc[test_idx][ret_col].values

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.7,
        colsample_bytree=0.7,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_test)

    test_df = df.iloc[test_idx].copy()
    test_df["predicted_return"] = y_pred
    test_df["actual_return"] = y_test

    # Per-period evaluation metrics
    ic = _ic(pd.Series(y_test, index=test_df.index), pd.Series(y_pred, index=test_df.index))
    score_ic = _ic(test_df["actual_return"], test_df["Score"]) if "Score" in test_df.columns else np.nan

    spread = _quintile_spread(test_df["predicted_return"], test_df["actual_return"], n_bins=5)
    score_spread = _quintile_spread(test_df["Score"], test_df["actual_return"], n_bins=5) if "Score" in test_df.columns else np.nan

    # Top-20 equal-weight portfolio each run_date
    portfolio_records = []
    for run_date, run in test_df.groupby("run_date"):
        if len(run) < 20:
            continue
        top20 = run.nlargest(20, "predicted_return")
        port_ret = top20["actual_return"].mean()
        bench_vals = run[bench_col].dropna()
        bench_ret = bench_vals.iloc[0] if bench_col in run.columns and len(bench_vals) > 0 else np.nan
        portfolio_records.append({
            "run_date": run_date,
            "portfolio_return": port_ret,
            "benchmark_return": bench_ret,
            "active_return": port_ret - bench_ret if not pd.isna(bench_ret) else np.nan,
            "n_stocks": len(top20.dropna(subset=["actual_return"])),
        })
    port_df = pd.DataFrame(portfolio_records)

    # Cumulative performance
    if not port_df.empty:
        port_df = port_df.sort_values("run_date")
        port_df["cum_portfolio"] = (1 + port_df["portfolio_return"].fillna(0)).cumprod()
        port_df["cum_benchmark"] = (1 + port_df["benchmark_return"].fillna(0)).cumprod()
        port_df["cum_active"] = (1 + port_df["active_return"].fillna(0)).cumprod()
        mean_active = port_df["active_return"].mean()
        std_active = port_df["active_return"].std()
        sharpe = (mean_active / std_active * _annualisation_factor(horizon)) if std_active and std_active > 0 else np.nan
    else:
        mean_active = np.nan
        sharpe = np.nan
        port_df = pd.DataFrame()

    feature_importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Save artifacts
    model_path = EVAL_DIR / f"ml_model_{horizon}.joblib"
    joblib.dump({"model": model, "encoder": encoder, "numeric_cols": numeric_cols}, model_path)

    pred_path = EVAL_DIR / f"ml_predictions_{horizon}.parquet"
    test_df[["run_date", "Ticker", "Sector", "Score", "predicted_return", "actual_return", bench_col]].to_parquet(
        pred_path, index=False
    )

    if not port_df.empty:
        port_df.to_csv(EVAL_DIR / f"ml_portfolio_{horizon}.csv", index=False)

    metrics = {
        "horizon": horizon,
        "test_samples": len(test_idx),
        "features_used": numeric_cols,
        "test_ic": float(ic),
        "score_ic": float(score_ic),
        "top_bottom_quintile_spread": float(spread),
        "score_quintile_spread": float(score_spread),
        "top20_mean_active_return": float(mean_active),
        "top20_active_sharpe": float(sharpe),
        "model_path": str(model_path),
        "feature_importance": feature_importance.head(15).to_dict(orient="records"),
    }

    print(f"\n--- ML Model: {horizon} forward returns ---")
    print(f"Test IC (model)      : {ic:.4f}")
    if not np.isnan(score_ic):
        print(f"Test IC (Score)      : {score_ic:.4f}")
    print(f"Top-bottom Q spread  : {spread:.4f}")
    if not port_df.empty:
        print(f"Top-20 active return : {mean_active:.4f}")
        print(f"Top-20 active Sharpe : {sharpe:.4f}")
    print("Top features:")
    print(feature_importance.head(10).to_string(index=False))

    return metrics


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

    # Train ML models for selected horizons
    print("\nTraining ML models...")
    ml_metrics = {}
    for horizon in ML_HORIZONS:
        metrics = train_ml_model(archive, horizon)
        if metrics:
            ml_metrics[horizon] = metrics

    if ml_metrics:
        with open(EVAL_DIR / "ml_metrics.json", "w", encoding="utf-8") as f:
            json.dump(ml_metrics, f, indent=2, default=str)
        print(f"\nML metrics saved: {EVAL_DIR / 'ml_metrics.json'}")


if __name__ == "__main__":
    main()
