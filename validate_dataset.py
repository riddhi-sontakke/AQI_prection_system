"""Validate a local AQI dataset before training."""

from __future__ import annotations
import argparse
import json
import os
import sys
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))

from data.city_data import available_cities, canonical_city, resolve_city_dataset
from data.data_loader import REQUIRED_COLUMNS, RealDataLoader


def validate(
    filepath: str = None,
    city: str = "Bengaluru",
    column_map: dict = None,
    timestamp_format: str = None,
):
    """Load and validate one local dataset."""
    city = canonical_city(city)
    filepath = resolve_city_dataset(city, filepath)

    print("\n" + "=" * 65)
    print(f"DATASET VALIDATOR - {city}")
    print(f"Dataset: {filepath}")
    print("=" * 65)

    loader = RealDataLoader(city=city)
    df = loader.load(
        filepath,
        column_map=column_map,
        timestamp_format=timestamp_format,
    )

    checks = []
    freq = loader.freq_detected
    min_req = 24 * 90 if freq == "H" else 180
    checks.append(("Minimum row count", len(df) >= min_req, f"{len(df):,} rows; need {min_req:,}"))

    days = (df.index[-1] - df.index[0]).days
    checks.append(("Minimum duration", days >= 180, f"{days} days"))

    required_cols = [col for col, meta in REQUIRED_COLUMNS.items() if meta["required"]]
    for col in required_cols:
        checks.append((f"Required column: {col}", col in df.columns, "present" if col in df.columns else "missing"))

    aqi_pct = df["AQI"].notna().mean() * 100
    checks.append(("AQI coverage", aqi_pct >= 50, f"{aqi_pct:.1f}% valid values"))

    for col in ["PM2.5", "PM10", "NO2", "Temperature", "Humidity", "WindSpeed"]:
        if col in df.columns:
            pct_missing = df[col].isna().mean() * 100
            checks.append((f"Missing rate: {col}", pct_missing < 30, f"{pct_missing:.1f}% missing"))

    aqi_min, aqi_max = df["AQI"].min(), df["AQI"].max()
    checks.append(("AQI range 0-500", 0 <= aqi_min and aqi_max <= 500, f"min={aqi_min:.1f}, max={aqi_max:.1f}"))

    passed = sum(1 for _, ok, _ in checks if ok)
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"{status:<5} {name:<28} {detail}")

    print("-" * 65)
    print(f"RESULT: {passed}/{len(checks)} checks passed")
    print("\nSample rows after loading:")
    display_cols = [col for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity", "WindSpeed", "AQI"] if col in df.columns]
    print(df[display_cols].head(3).to_string())

    print("\nAQI category distribution:")
    bins = [0, 50, 100, 200, 300, 400, 500]
    labels = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
    cats = pd.cut(df["AQI"], bins=bins, labels=labels, right=True)
    for cat, count in cats.value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"{cat:<15} {count:>6,} ({pct:>5.1f}%)")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate local AQI dataset before training")
    parser.add_argument("--city", default="Bengaluru", choices=available_cities())
    parser.add_argument("--data", default=None, help="Optional local CSV path")
    parser.add_argument("--column-map", default=None, help="JSON dict of column renames")
    parser.add_argument("--timestamp-format", default=None)
    args = parser.parse_args()

    col_map = json.loads(args.column_map) if args.column_map else None
    validate(args.data, args.city, col_map, args.timestamp_format)
