# update_data.py
"""Headless data updater for the screener.

Runs the legacy screener pipeline with a minimal Streamlit stub, then persists
`scr` and `prices_map` to the `data/` folder for instant loading by the dashboard.
"""

import importlib.util
import os
import sys
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf


# ── Market-open throttle for the Pi scheduler ────────────────────────────────
def _is_us_market_open(dt: datetime | None = None) -> bool:
    """Rough US equity market-open check (EST/EDT, no holidays)."""
    from datetime import time as _time
    from zoneinfo import ZoneInfo

    if dt is None:
        dt = datetime.now(ZoneInfo("US/Eastern"))
    if dt.weekday() >= 5:
        return False
    return _time(9, 30) <= dt.time() < _time(16, 0)


def _should_update(data_dir: Path) -> bool:
    """Return True if we should run a full update now.

    Logic:
      - Always run if no snapshot exists yet.
      - Run every hour while the US market is open.
      - Run at most every 4 hours outside market hours.
    """
    last_update_file = data_dir / "last_update.json"
    now = datetime.now(timezone.utc)
    if not (data_dir / "latest_screener.parquet").exists():
        return True
    if not last_update_file.exists():
        return True
    try:
        last_update = datetime.fromisoformat(last_update_file.read_text().strip())
    except Exception:
        return True
    elapsed = (now - last_update).total_seconds()
    if _is_us_market_open():
        return elapsed >= 3500  # ~1 hour
    return elapsed >= 4 * 3600 - 100  # ~4 hours


# ── Minimal Streamlit stub ───────────────────────────────────────────────────
class _NoOp:
    """A do-nothing object that is also a context manager and callable."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, *args, **kwargs):
        return _NoOp()

    def __getattr__(self, name):
        return _NoOp()

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __getitem__(self, key):
        return _NoOp()


def _cache_data_decorator(fn):
    return fn


_cache_data_decorator.clear = lambda: None


def _make_columns(n, *args, **kwargs):
    count = len(n) if isinstance(n, (list, tuple)) else int(n)
    return tuple(_NoOp() for _ in range(count))


def _make_tabs(labels, *args, **kwargs):
    return [_NoOp() for _ in labels]


def _selectbox(label, options, *args, **kwargs):
    return (options[0] if options else None)


def _number_input(label, *args, **kwargs):
    return kwargs.get("value")


def _segmented_control(label, options, *args, **kwargs):
    return kwargs.get("default") or (options[0] if options else None)


def _button(*args, **kwargs):
    return False


def _stop():
    raise SystemExit(0)


def _build_streamlit_stub():
    st = types.ModuleType("streamlit")
    st.__dict__.update({
        "set_page_config": lambda *a, **k: None,
        "markdown": lambda *a, **k: None,
        "columns": _make_columns,
        "tabs": _make_tabs,
        "sidebar": _NoOp(),
        "spinner": lambda *a, **k: _NoOp(),
        "expander": lambda *a, **k: _NoOp(),
        "button": _button,
        "selectbox": _selectbox,
        "number_input": _number_input,
        "segmented_control": _segmented_control,
        "metric": lambda *a, **k: None,
        "info": lambda *a, **k: None,
        "warning": lambda *a, **k: None,
        "error": lambda *a, **k: None,
        "caption": lambda *a, **k: None,
        "success": lambda *a, **k: None,
        "dataframe": lambda *a, **k: None,
        "bar_chart": lambda *a, **k: None,
        "altair_chart": lambda *a, **k: None,
        "download_button": lambda *a, **k: None,
        "cache_data": lambda *a, **k: _cache_data_decorator,
        "rerun": lambda *a, **k: None,
        "stop": _stop,
        "session_state": {},
        "secrets": {},
        "empty": lambda *a, **k: _NoOp(),
        "progress": lambda *a, **k: _NoOp(),
    })

    def __getattr__(name):
        return _NoOp()

    st.__getattr__ = __getattr__
    return st


# ── Throttle check (Pi scheduler runs us every hour) ─────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
SCREENER_PATH = SCRIPT_DIR / "screener_app_legacy.py"

data_dir = SCRIPT_DIR / "data"
data_dir.mkdir(exist_ok=True)

if not _should_update(data_dir):
    print("Update skipped: outside market hours and recent update exists.")
    raise SystemExit(0)


# ── Run screener_app_legacy.py headlessly ──────────────────────────────────

sys.modules["streamlit"] = _build_streamlit_stub()

spec = importlib.util.spec_from_file_location("screener_app", SCREENER_PATH)
screener_mod = importlib.util.module_from_spec(spec)

try:
    spec.loader.exec_module(screener_mod)
except SystemExit:
    pass

scr = getattr(screener_mod, "scr", None)
prices_map = getattr(screener_mod, "prices_map", None)

if scr is None or prices_map is None:
    raise RuntimeError(
        "screener_app_legacy.py did not produce `scr` and/or `prices_map`. "
        "Check the headless run output for errors."
    )

# ── Save latest screener snapshot ─────────────────────────────────────────────
parquet_path = data_dir / "latest_screener.parquet"
csv_path = data_dir / "latest_screener.csv"

scr.to_parquet(parquet_path, index=False)
scr.to_csv(csv_path, index=False)

# ── Append to prediction archive ──────────────────────────────────────────────
archive_path = data_dir / "prediction_archive.parquet"

run_timestamp = datetime.now(timezone.utc).isoformat()
run_id = screener_mod.st.session_state.get("run_id") or str(uuid.uuid4())[:8]

archive_cols = [c for c in scr.columns if c != "Eligible"]
archive_row = scr[archive_cols].copy()
archive_row["run_timestamp"] = run_timestamp
archive_row["run_id"] = run_id

if archive_path.exists():
    existing = pd.read_parquet(archive_path)
    archive_row = pd.concat([existing, archive_row], ignore_index=True)

archive_row = archive_row.drop_duplicates(subset=["run_id", "Ticker"])
archive_row.to_parquet(archive_path, index=False)

# ── Append to price history ───────────────────────────────────────────────────
price_history_path = data_dir / "price_history.parquet"

price_rows = []
for ticker, series in prices_map.items():
    series = series.dropna()
    if series.empty:
        continue
    price_rows.append({
        "date": pd.Timestamp(series.index[-1]).date(),
        "ticker": str(ticker).upper(),
        "close": float(series.iloc[-1]),
    })

# Add latest SPY close
try:
    spy_hist = yf.Ticker("SPY").history(period="5d", auto_adjust=True)
    if spy_hist is not None and not spy_hist.empty and "Close" in spy_hist.columns:
        spy_close = float(spy_hist["Close"].dropna().iloc[-1])
        spy_date = pd.Timestamp(spy_hist.index[-1]).date()
        price_rows.append({"date": spy_date, "ticker": "SPY", "close": spy_close})
except Exception:
    pass

price_history = pd.DataFrame(price_rows)
if price_history_path.exists():
    existing_ph = pd.read_parquet(price_history_path)
    price_history = pd.concat([existing_ph, price_history], ignore_index=True)

price_history["date"] = pd.to_datetime(price_history["date"]).dt.date
price_history = price_history.drop_duplicates(subset=["date", "ticker"])
price_history.to_parquet(price_history_path, index=False)

# ── Mark successful update time ────────────────────────────────────────────────
(data_dir / "last_update.json").write_text(datetime.now(timezone.utc).isoformat())

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"Tickers processed: {len(prices_map)}")
print(f"Screener rows:     {len(scr)}")
print(f"Archive shape:     {archive_row.shape}")
print(f"Price rows:        {len(price_history)}")
print(f"  {parquet_path}")
print(f"  {csv_path}")
print(f"  {archive_path}")
print(f"  {price_history_path}")
