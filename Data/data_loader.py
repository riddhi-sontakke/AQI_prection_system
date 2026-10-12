"""
=============================================================
AQI PREDICTION SYSTEM v2 - REAL DATA LOADER
=============================================================
Module  : data/data_loader.py
Version : 2.0 (Real Dataset Integration)

CHANGES FROM v1:
  [REMOVED] generate_synthetic_aqi_dataset() function
  [REMOVED] CITY_PROFILES dict (synthetic noise parameters)
  [REMOVED] All numpy random simulation logic
  [ADDED]   RealDataLoader class — loads your CSV/DataFrame
  [ADDED]   Column auto-mapper (flexible column name matching)
  [ADDED]   Data validation with detailed error reporting
  [ADDED]   Frequency detection (hourly vs daily)
  [ADDED]   AQI auto-calculation if AQI column is missing
  [STRICT]  Optional columns are skipped if absent; no defaults are invented.

ASSUMPTIONS ABOUT YOUR DATASET:
  1. File is CSV (comma-separated) or Excel (.xlsx)
  2. One row = one time observation (hourly or daily)
  3. A timestamp/datetime column exists (any name)
  4. At least PM2.5 and PM10 are present
  5. The dataset is sorted by time (or will be sorted here)
  6. Missing values are NaN, blank, or common placeholders
     like -999, 9999, 'NA', 'N/A', '--'
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Union


# ─────────────────────────────────────────────────────────────
# REQUIRED COLUMNS SPECIFICATION
# (used for validation + auto-mapping)
# ─────────────────────────────────────────────────────────────

# Exact internal names the preprocessor expects.
# Your CSV may use different names — the auto-mapper handles that.
REQUIRED_COLUMNS = {
    # Pollutants
    "PM2.5":       {"dtype": float, "unit": "µg/m³",  "required": True},
    "PM10":        {"dtype": float, "unit": "µg/m³",  "required": True},
    "NO2":         {"dtype": float, "unit": "µg/m³",  "required": False},
    "SO2":         {"dtype": float, "unit": "µg/m³",  "required": False},
    "CO":          {"dtype": float, "unit": "mg/m³",  "required": False},
    "O3":          {"dtype": float, "unit": "µg/m³",  "required": False},
    # Meteorological
    "Temperature": {"dtype": float, "unit": "°C",     "required": False},
    "Humidity":    {"dtype": float, "unit": "%",       "required": False},
    "WindSpeed":   {"dtype": float, "unit": "km/h",   "required": False},
    "WindDirection":{"dtype": float,"unit": "degrees", "required": False},
    "Rainfall":    {"dtype": float, "unit": "mm",      "required": False},
    # Target
    "AQI":         {"dtype": float, "unit": "index",   "required": False},
}

# Auto-mapper: alternate names your dataset might use → internal name
COLUMN_ALIASES = {
    # Timestamp variants
    "timestamp":   "timestamp", "Timestamp": "timestamp",
    "date":        "timestamp", "Date": "timestamp",
    "datetime":    "timestamp", "DateTime": "timestamp",
    "Date/Time":   "timestamp", "date_time": "timestamp",
    "time":        "timestamp", "Time": "timestamp",
    "DATE":        "timestamp", "TIMESTAMP": "timestamp",

    # PM2.5 variants
    "PM2.5":       "PM2.5",  "pm2.5":  "PM2.5", "pm25":   "PM2.5",
    "PM25":        "PM2.5",  "PM_2_5": "PM2.5", "pm_2_5": "PM2.5",
    "pm2_5":       "PM2.5",  "fine_particulate": "PM2.5",
    "Fine Particulate Matter": "PM2.5",

    # PM10 variants
    "PM10":        "PM10",   "pm10":   "PM10",  "pm_10":  "PM10",
    "PM_10":       "PM10",   "coarse_particulate": "PM10",

    # NO2 variants
    "NO2":         "NO2",    "no2":    "NO2",   "nitrogen_dioxide": "NO2",
    "Nitrogen Dioxide": "NO2",

    # SO2 variants
    "SO2":         "SO2",    "so2":    "SO2",   "sulfur_dioxide": "SO2",
    "sulphur_dioxide": "SO2",

    # CO variants
    "CO":          "CO",     "co":     "CO",    "carbon_monoxide": "CO",
    "Carbon Monoxide": "CO",

    # O3 / Ozone variants
    "O3":          "O3",     "o3":     "O3",    "ozone": "O3",
    "Ozone":       "O3",

    # Temperature variants
    "Temperature": "Temperature", "temperature": "Temperature",
    "Temp":        "Temperature", "temp": "Temperature",
    "temp_c":      "Temperature", "temperature_c": "Temperature",
    "TEMP":        "Temperature", "AT": "Temperature",

    # Humidity variants
    "Humidity":    "Humidity", "humidity": "Humidity",
    "RH":          "Humidity", "rh": "Humidity",
    "relative_humidity": "Humidity", "Relative Humidity": "Humidity",

    # Wind speed variants
    "WindSpeed":   "WindSpeed", "wind_speed": "WindSpeed",
    "Windspeed":   "WindSpeed", "windspeed": "WindSpeed",
    "WS":          "WindSpeed", "ws": "WindSpeed",
    "Wind Speed":  "WindSpeed", "WSPD": "WindSpeed",
    "wind":        "WindSpeed",

    # Wind direction variants
    "WindDirection": "WindDirection", "wind_direction": "WindDirection",
    "WD":          "WindDirection", "wd": "WindDirection",
    "Wind Direction": "WindDirection",

    # Rainfall variants
    "Rainfall":    "Rainfall", "rainfall": "Rainfall",
    "Rain":        "Rainfall", "rain": "Rainfall",
    "precipitation": "Rainfall", "Precipitation": "Rainfall",
    "PRCP":        "Rainfall", "RF": "Rainfall",

    # AQI variants
    "AQI":         "AQI", "aqi": "AQI",
    "Air Quality Index": "AQI",
    "AQI_VALUE":   "AQI", "aqi_value": "AQI",
}

# Common bad-value placeholders that should become NaN
BAD_VALUE_PLACEHOLDERS = {
    -999, -9999, 999, 9999, -1, 999.0, 9999.0, -999.0,
    "NA", "N/A", "n/a", "--", "---", "NaN", "nan",
    "null", "NULL", "None", "MISSING", "missing",
}


# ─────────────────────────────────────────────────────────────
# REAL DATA LOADER CLASS
# ─────────────────────────────────────────────────────────────
class RealDataLoader:
    """
    Loads and validates your real-world AQI dataset.

    Supports:
      - CSV files (.csv)
      - Excel files (.xlsx, .xls)
      - Pandas DataFrames (already in memory)

    Usage:
        loader = RealDataLoader()
        df = loader.load("path/to/your_data.csv")

        # Or with column mapping if auto-mapper doesn't catch everything:
        df = loader.load("data.csv", column_map={"Temp_C": "Temperature"})
    """

    def __init__(self, city: str = "Unknown"):
        self.city         = city
        self.freq_detected = None   # Set after load: 'H', 'D', etc.
        self.column_report = {}     # Filled during load for diagnostics

    # ─────────────────────────────────────────────────────────
    # MAIN LOAD METHOD
    # ─────────────────────────────────────────────────────────
    def load(
        self,
        source: Union[str, pd.DataFrame],
        column_map: Optional[dict] = None,
        timestamp_col: Optional[str] = None,
        timestamp_format: Optional[str] = None,
        city: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load, validate, and standardize a real AQI dataset.

        Parameters
        ----------
        source           : File path (CSV/Excel) OR a pandas DataFrame
        column_map       : Manual overrides for column mapping.
                           E.g. {"My_Temp_Col": "Temperature"}
                           Applied on top of the auto-mapper.
        timestamp_col    : Force a specific column to use as timestamp.
                           If None, auto-detected.
        timestamp_format : strftime format string for parsing timestamps.
                           E.g. "%d/%m/%Y %H:%M"  or  "%Y-%m-%d"
                           If None, pandas will infer it automatically.
        city             : Overrides self.city

        Returns
        -------
        pd.DataFrame
            Cleaned DataFrame with:
            - DatetimeIndex named 'timestamp'
            - Standard column names (PM2.5, PM10, NO2, ...)
            - Numeric dtypes for all feature columns
            - Sentinel bad values replaced with NaN
            - AQI column added if missing (calculated from pollutants)
        """
        if city:
            self.city = city

        print("\n" + "═" * 60)
        print(f"  📂 REAL DATA LOADER v2.0 — {self.city}")
        print("═" * 60)

        # Step A: Read raw data
        df_raw = self._read_source(source)
        print(f"  Raw shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")
        print(f"  Columns found: {list(df_raw.columns)}")

        # Step B: Apply auto column mapping
        df_mapped = self._map_columns(df_raw, extra_map=column_map)

        # Step C: Parse and set timestamp index
        df_ts = self._parse_timestamp(df_mapped, timestamp_col, timestamp_format)

        # Step D: Replace bad placeholders with NaN
        df_clean = self._replace_bad_values(df_ts)

        # Step E: Convert all feature columns to numeric
        df_numeric = self._enforce_numeric(df_clean)

        # Step F: Detect sampling frequency
        self._detect_frequency(df_numeric)

        # Step G: Add AQI if missing
        df_final = self._ensure_aqi(df_numeric)

        # Step H: Report optional columns only. Missing optional data is not
        # invented; the model trains only on columns present in the CSV.
        self._report_missing_optional_cols(df_final)

        # Step I: Validate
        self._validate(df_final)

        # Summary
        self._print_summary(df_final)
        return df_final

    # ─────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ─────────────────────────────────────────────────────────
    def _read_source(self, source) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            print("  Source: in-memory DataFrame")
            return source.copy()
        elif isinstance(source, str):
            ext = os.path.splitext(source)[1].lower()
            if not os.path.exists(source):
                raise FileNotFoundError(
                    f"Dataset not found: {source}\n"
                    f"Please ensure the file exists at this path."
                )
            print(f"  Source: {source}")
            if ext == ".csv":
                # Try common encodings
                for enc in ["utf-8", "latin-1", "cp1252"]:
                    try:
                        return pd.read_csv(source, encoding=enc, low_memory=False)
                    except UnicodeDecodeError:
                        continue
                raise ValueError("Could not decode CSV. Try saving as UTF-8.")
            elif ext in [".xlsx", ".xls"]:
                return pd.read_excel(source)
            else:
                raise ValueError(f"Unsupported file type: {ext}. Use .csv or .xlsx")
        else:
            raise TypeError(f"source must be a file path (str) or DataFrame, got {type(source)}")

    def _map_columns(self, df: pd.DataFrame, extra_map: Optional[dict]) -> pd.DataFrame:
        """Apply auto-mapper + user overrides to standardize column names."""
        rename = {}
        # First apply extra_map (user-specified, highest priority)
        if extra_map:
            rename.update(extra_map)

        # Then auto-mapper for anything not already mapped
        for col in df.columns:
            if col not in rename:
                mapped = COLUMN_ALIASES.get(col)
                if mapped:
                    rename[col] = mapped

        if rename:
            print(f"\n  Column mapping applied:")
            for old, new in rename.items():
                if old in df.columns:
                    print(f"    '{old}' → '{new}'")
            df = df.rename(columns=rename)

        # Track which standard columns were found
        self.column_report = {
            col: (col in df.columns) for col in list(REQUIRED_COLUMNS.keys()) + ["timestamp"]
        }
        return df

    def _parse_timestamp(
        self,
        df: pd.DataFrame,
        ts_col: Optional[str],
        ts_fmt: Optional[str]
    ) -> pd.DataFrame:
        """Find, parse, and set the timestamp as DatetimeIndex."""
        # Determine which column to use
        if ts_col:
            col = ts_col
        elif "timestamp" in df.columns:
            col = "timestamp"
        else:
            # Heuristic: find any column that looks like a date
            date_candidates = [
                c for c in df.columns
                if any(kw in c.lower() for kw in ["date", "time", "dt", "ts"])
            ]
            if not date_candidates:
                raise ValueError(
                    "No timestamp column found.\n"
                    "Your dataset must have a date/time column.\n"
                    f"Available columns: {list(df.columns)}\n"
                    "Hint: pass timestamp_col='your_column_name'"
                )
            col = date_candidates[0]
            print(f"\n  Auto-detected timestamp column: '{col}'")

        # Parse dates
        try:
            if ts_fmt:
                df[col] = pd.to_datetime(df[col], format=ts_fmt)
            else:
                df[col] = pd.to_datetime(df[col])
        except Exception as e:
            raise ValueError(
                f"Could not parse timestamp column '{col}'.\n"
                f"Error: {e}\n"
                "Hint: pass timestamp_format='%d/%m/%Y %H:%M' (adjust to match your data)"
            )

        df = df.set_index(col)
        df.index.name = "timestamp"
        df = df.sort_index()

        # Drop duplicate timestamps (keep last reading)
        n_dups = df.index.duplicated().sum()
        if n_dups > 0:
            print(f"\n  ⚠️  {n_dups} duplicate timestamps found — keeping last occurrence")
            df = df[~df.index.duplicated(keep="last")]

        return df

    def _replace_bad_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace known sensor-error placeholder values with NaN."""
        n_replaced = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mask = df[col].isin(BAD_VALUE_PLACEHOLDERS)
            n_replaced += mask.sum()
            df.loc[mask, col] = np.nan

        # Also handle string-encoded bad values
        str_cols = df.select_dtypes(include=["object"]).columns
        for col in str_cols:
            mask = df[col].astype(str).isin(BAD_VALUE_PLACEHOLDERS)
            n_replaced += mask.sum()
            df.loc[mask, col] = np.nan

        if n_replaced:
            print(f"\n  Replaced {n_replaced:,} bad-value placeholders with NaN")
        return df

    def _enforce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert feature columns to float, coercing errors to NaN."""
        feature_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
        for col in feature_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _detect_frequency(self, df: pd.DataFrame):
        """Infer sampling frequency from the timestamp index."""
        if len(df) < 2:
            self.freq_detected = "unknown"
            return

        # Compute median time delta
        deltas = pd.Series(df.index).diff().dropna()
        median_delta = deltas.median()

        if median_delta <= pd.Timedelta("1.5H"):
            self.freq_detected = "H"
            label = "Hourly (optimal)"
        elif median_delta <= pd.Timedelta("1.5D"):
            self.freq_detected = "D"
            label = "Daily (acceptable — see data quantity guide)"
        else:
            self.freq_detected = "other"
            label = f"Non-standard ({median_delta})"

        print(f"\n  Detected frequency: {label}")

        if self.freq_detected == "D":
            warnings.warn(
                "Daily data detected. LSTM window_size will be reduced to 30 days "
                "automatically. For best results, use hourly data.",
                UserWarning
            )

    def _ensure_aqi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate AQI using India's CPCB sub-index method if AQI column is absent.
        Requires at least PM2.5 and PM10.
        """
        if "AQI" in df.columns and df["AQI"].notna().sum() > len(df) * 0.5:
            print("\n  AQI column found and used as-is.")
            return df

        print("\n  AQI column missing or mostly empty — calculating from pollutants...")

        def pm25_subindex(x):
            if pd.isna(x): return np.nan
            if x <= 30:    return x * 50 / 30
            elif x <= 60:  return 50 + (x - 30) * 50 / 30
            elif x <= 90:  return 100 + (x - 60) * 100 / 30
            elif x <= 120: return 200 + (x - 90) * 100 / 30
            elif x <= 250: return 300 + (x - 120) * 100 / 130
            else:          return min(500, 400 + (x - 250) * 100 / 130)

        def pm10_subindex(x):
            if pd.isna(x): return np.nan
            if x <= 50:    return x
            elif x <= 100: return x
            elif x <= 250: return 100 + (x - 100) * 100 / 150
            elif x <= 350: return 200 + (x - 250)
            elif x <= 430: return 300 + (x - 350) * 100 / 80
            else:          return min(500, 400 + (x - 430) * 100 / 80)

        sub_indices = pd.DataFrame()

        if "PM2.5" in df.columns:
            sub_indices["si_pm25"] = df["PM2.5"].apply(pm25_subindex)
        if "PM10" in df.columns:
            sub_indices["si_pm10"] = df["PM10"].apply(pm10_subindex)
        if "NO2" in df.columns:
            sub_indices["si_no2"] = (df["NO2"] * 4).clip(0, 400)
        if "SO2" in df.columns:
            sub_indices["si_so2"] = (df["SO2"] * 12).clip(0, 400)
        if "CO" in df.columns:
            sub_indices["si_co"] = (df["CO"] * 30).clip(0, 400)
        if "O3" in df.columns:
            sub_indices["si_o3"] = (df["O3"] * 6).clip(0, 400)

        if sub_indices.empty:
            raise ValueError(
                "Cannot calculate AQI: no pollutant columns found.\n"
                "Dataset must have at least PM2.5 or PM10."
            )

        df["AQI"] = sub_indices.max(axis=1).clip(0, 500).round(1)
        n_valid = df["AQI"].notna().sum()
        print(f"  AQI calculated for {n_valid:,} rows "
              f"(mean={df['AQI'].mean():.1f}, max={df['AQI'].max():.1f})")
        return df

    def _report_missing_optional_cols(self, df: pd.DataFrame):
        """Print optional columns that are absent without modifying the data."""
        missing_optional = [
            col for col, meta in REQUIRED_COLUMNS.items()
            if not meta["required"] and col not in df.columns
        ]
        if missing_optional:
            print(
                "\n  Optional columns not present in CSV and skipped: "
                + ", ".join(missing_optional)
            )

    def _validate(self, df: pd.DataFrame):
        """Check required columns exist and raise informative errors."""
        missing_required = [
            col for col, meta in REQUIRED_COLUMNS.items()
            if meta["required"] and col not in df.columns
        ]
        if missing_required:
            raise ValueError(
                f"Required columns missing after mapping: {missing_required}\n"
                "Your dataset MUST contain PM2.5 and PM10 columns.\n"
                "If they exist under different names, pass column_map:\n"
                "  loader.load('data.csv', column_map={'My_PM25_Col': 'PM2.5'})"
            )

        # Check minimum rows
        min_rows = 24 * 30  # 30 days of hourly data minimum
        if len(df) < min_rows:
            warnings.warn(
                f"Dataset has only {len(df):,} rows. "
                f"Minimum recommended: {min_rows:,} rows (30 days hourly). "
                "Model accuracy may be reduced.",
                UserWarning
            )

        # Check AQI coverage
        aqi_coverage = df["AQI"].notna().mean()
        if aqi_coverage < 0.7:
            warnings.warn(
                f"AQI column has only {aqi_coverage:.0%} valid values. "
                "Consider checking your data or using the auto-calculator.",
                UserWarning
            )

    def _print_summary(self, df: pd.DataFrame):
        """Print a structured data quality summary."""
        print("\n" + "─" * 60)
        print("  DATA SUMMARY")
        print("─" * 60)
        print(f"  Total records  : {len(df):,}")
        print(f"  Date range     : {df.index[0]} → {df.index[-1]}")
        days = (df.index[-1] - df.index[0]).days
        print(f"  Duration       : {days} days ({days/365:.1f} years)")
        print(f"  Frequency      : {self.freq_detected}")
        print(f"\n  {'Column':<18} {'Present':>8} {'Valid%':>8} {'Min':>8} {'Max':>8} {'Mean':>8}")
        print(f"  {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

        all_cols = list(REQUIRED_COLUMNS.keys())
        for col in all_cols:
            if col in df.columns:
                s = df[col].dropna()
                pct = len(s) / len(df) * 100
                status = "✅" if pct > 90 else ("⚠️ " if pct > 50 else "❌")
                print(f"  {status} {col:<16} {'Yes':>8} {pct:>7.1f}% "
                      f"{s.min():>8.1f} {s.max():>8.1f} {s.mean():>8.1f}")
            else:
                print(f"  ➖ {col:<16} {'No':>8} {'—':>8}% {'—':>8} {'—':>8} {'—':>8}")
        print("═" * 60)


# ─────────────────────────────────────────────────────────────
# CONVENIENCE FUNCTION (drop-in replacement for old generator)
# ─────────────────────────────────────────────────────────────
def load_real_dataset(
    filepath: str,
    city: str = "Unknown",
    column_map: Optional[dict] = None,
    timestamp_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    One-line interface to load your real dataset.

    Parameters
    ----------
    filepath         : Path to your CSV or Excel file
    city             : City name (for labeling only)
    column_map       : Dict of {your_col_name: standard_name}
                       Only needed if auto-mapper misses something.
                       Example: {"T_celsius": "Temperature", "WS_kmh": "WindSpeed"}
    timestamp_format : Datetime parse format if auto-detection fails.
                       Example: "%d-%m-%Y %H:%M:%S"

    Returns
    -------
    pd.DataFrame with DatetimeIndex and standardized columns
    """
    loader = RealDataLoader(city=city)
    return loader.load(
        filepath,
        column_map=column_map,
        timestamp_format=timestamp_format,
    )
