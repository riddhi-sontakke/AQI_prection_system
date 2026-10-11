"""
=============================================================
AQI PREDICTION SYSTEM v2 - PREPROCESSOR (Real Data Ready)
=============================================================
Module  : data/preprocessor.py
Version : 2.0 (Real Dataset Integration)

CHANGES FROM v1:
  [UNCHANGED] All preprocessing logic (missing values, outliers,
              feature engineering, scaling, LSTM windowing)
  [CHANGED]   window_size and forecast_horizon now auto-adjust
              based on detected data frequency (hourly vs daily)
  [CHANGED]   fit_transform() now accepts a pre-loaded DataFrame
              instead of city name / generator call
  [ADDED]     frequency-aware defaults

Everything else is IDENTICAL to v1.
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import warnings

POLLUTANT_COLS = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
METEO_COLS     = ["Temperature", "Humidity", "WindSpeed",
                  "WindDirection", "Rainfall"]
BASE_FEATURES  = POLLUTANT_COLS + METEO_COLS
TARGET_COL     = "AQI"


class AQIPreprocessor:
    """
    Complete preprocessing pipeline — identical logic to v1,
    updated to accept a real DataFrame directly.
    """

    def __init__(
        self,
        window_size: int = None,         # None = auto from frequency
        forecast_horizon: int = None,    # None = auto from frequency
        lag_hours: list = None,
        rolling_windows: list = None,
        data_frequency: str = "H",       # 'H' = hourly, 'D' = daily
    ):
        self.data_frequency   = data_frequency
        self._set_defaults(window_size, forecast_horizon)
        self.lag_hours        = lag_hours or self._default_lags()
        self.rolling_windows  = rolling_windows or self._default_rolls()
        self.scaler_X         = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y         = MinMaxScaler(feature_range=(0, 1))
        self.feature_names    = []
        self._fitted          = False

    def _set_defaults(self, ws, fh):
        """Auto-set window/horizon based on data frequency."""
        if self.data_frequency == "D":
            # Daily data: 30-day lookback, 3-day forecast
            self.window_size      = ws or 30
            self.forecast_horizon = fh or 3
        else:
            # Hourly data: 24-hour lookback, 6-hour forecast
            self.window_size      = ws or 24
            self.forecast_horizon = fh or 6

    def _default_lags(self):
        if self.data_frequency == "D":
            return [1, 2, 3, 7, 14, 30]   # Days
        return [1, 3, 6, 12, 24, 48]       # Hours

    def _default_rolls(self):
        if self.data_frequency == "D":
            return [3, 7, 14, 30]          # Days
        return [3, 6, 12, 24]             # Hours

    # ── All preprocessing steps unchanged from v1 ────────────

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        print("  [1/5] Handling missing values...")
        df = df.copy()
        missing_before = df.isnull().sum().sum()
        feature_cols = [c for c in BASE_FEATURES if c in df.columns]

        if feature_cols:
            df[feature_cols] = (
                df[feature_cols]
                .ffill(limit=3)
                .bfill(limit=1)
            )
            df[feature_cols] = df[feature_cols].interpolate(
                method="linear", limit_direction="both"
            )

        for col in feature_cols + [TARGET_COL]:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())

        missing_after = df.isnull().sum().sum()
        print(f"     Missing values: {missing_before:,} → {missing_after:,}")
        return df

    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        print("  [2/5] Removing outliers...")
        df = df.copy()
        physical_limits = {
            "PM2.5": (0, 500), "PM10": (0, 600), "NO2": (0, 400),
            "SO2": (0, 500), "CO": (0, 50), "O3": (0, 300),
            "Temperature": (0, 55), "Humidity": (0, 100),
            "WindSpeed": (0, 120), "WindDirection": (0, 360),
            "Rainfall": (0, 300), "AQI": (0, 500),
        }
        outliers_clipped = 0
        for col, (lo, hi) in physical_limits.items():
            if col in df.columns:
                out = ((df[col] < lo) | (df[col] > hi)).sum()
                df[col] = df[col].clip(lo, hi)
                outliers_clipped += out
        for col in POLLUTANT_COLS:
            if col in df.columns:
                Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                IQR = Q3 - Q1
                df[col] = df[col].clip(Q1 - 3 * IQR, Q3 + 3 * IQR)
        print(f"     Outliers clipped: {outliers_clipped:,}")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("  [3/5] Engineering features...")
        df = df.copy()

        # Cyclic time encodings
        df["hour_sin"]    = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"]    = np.cos(2 * np.pi * df.index.hour / 24)
        df["month_sin"]   = np.sin(2 * np.pi * df.index.month / 12)
        df["month_cos"]   = np.cos(2 * np.pi * df.index.month / 12)
        df["weekday_sin"] = np.sin(2 * np.pi * df.index.weekday / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df.index.weekday / 7)

        # Lag features
        for lag in self.lag_hours:
            df[f"AQI_lag_{lag}"]  = df[TARGET_COL].shift(lag)
            df[f"PM25_lag_{lag}"] = df["PM2.5"].shift(lag)

        # Rolling statistics
        for w in self.rolling_windows:
            df[f"AQI_roll_mean_{w}"]  = df[TARGET_COL].shift(1).rolling(w).mean()
            df[f"AQI_roll_std_{w}"]   = df[TARGET_COL].shift(1).rolling(w).std()
            df[f"PM25_roll_mean_{w}"] = df["PM2.5"].shift(1).rolling(w).mean()

        # Interaction features
        if {"PM2.5", "Humidity"}.issubset(df.columns):
            df["PM25_x_Humidity"] = df["PM2.5"] * df["Humidity"] / 100
        if {"WindSpeed", "Temperature"}.issubset(df.columns):
            df["Ventilation_Index"] = df["WindSpeed"] * df["Temperature"]
        if {"PM2.5", "PM10"}.issubset(df.columns):
            df["PM_ratio"] = df["PM2.5"] / (df["PM10"] + 1)

        df = df.dropna()
        print(f"     Features: {len(df.columns)} | Rows after dropna: {len(df):,}")
        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> tuple:
        print("  [4/5] Scaling features...")
        if fit:
            feature_cols = [c for c in df.columns if c != TARGET_COL]
            self.feature_names = feature_cols
        else:
            if not self._fitted:
                raise RuntimeError("Call fit_transform first!")
            feature_cols = list(self.feature_names)
            missing = [c for c in feature_cols if c not in df.columns]
            if missing:
                raise ValueError(
                    "Prediction data is missing trained feature columns: "
                    + ", ".join(missing)
                )
        X = df[feature_cols].values
        if TARGET_COL in df.columns:
            y = df[TARGET_COL].values.reshape(-1, 1)
        else:
            y = np.zeros((len(df), 1))
        if fit:
            X_scaled = self.scaler_X.fit_transform(X)
            y_scaled = self.scaler_y.fit_transform(y)
            self._fitted = True
        else:
            X_scaled = self.scaler_X.transform(X)
            y_scaled = self.scaler_y.transform(y)
        print(f"     X: {X_scaled.shape} | y: {y_scaled.shape}")
        return X_scaled, y_scaled.ravel(), df.index

    def create_lstm_windows(self, X_scaled, y_scaled) -> tuple:
        print("  [5/5] Creating LSTM sliding windows...")
        X_win, y_win = [], []
        total = len(X_scaled)
        for i in range(total - self.window_size - self.forecast_horizon + 1):
            X_win.append(X_scaled[i : i + self.window_size])
            y_win.append(y_scaled[i + self.window_size + self.forecast_horizon - 1])
        X_win = np.array(X_win)
        y_win = np.array(y_win)
        print(f"     LSTM X: {X_win.shape} | y: {y_win.shape}")
        return X_win, y_win

    def fit_transform(self, df: pd.DataFrame) -> dict:
        """
        Run complete preprocessing on your loaded real DataFrame.

        Parameters
        ----------
        df : pd.DataFrame with DatetimeIndex and standardized columns
             (output of RealDataLoader.load())

        Returns
        -------
        dict with X_lstm, X_flat, y_scaled, y_original, indices
        """
        print("\n📊 Starting preprocessing pipeline...")

        # Auto-detect frequency from index if not set
        if len(df) > 1:
            delta = pd.Series(df.index).diff().dropna().median()
            if delta <= pd.Timedelta("1.5H"):
                self.data_frequency = "H"
            else:
                self.data_frequency = "D"
                self._set_defaults(None, None)
                self.lag_hours = self._default_lags()
                self.rolling_windows = self._default_rolls()

        df = self.handle_missing_values(df)
        df = self.remove_outliers(df)
        df = self.engineer_features(df)
        X_scaled, y_scaled, idx = self.scale_features(df, fit=True)
        X_lstm, y_win = self.create_lstm_windows(X_scaled, y_scaled)
        X_flat = X_lstm[:, -1, :]
        y_original = self.scaler_y.inverse_transform(
            y_win.reshape(-1, 1)
        ).ravel()

        print(f"\n✅ Preprocessing complete!")
        print(f"   LSTM: {X_lstm.shape} | XGBoost: {X_flat.shape} | y: {y_win.shape}")
        return {
            "X_lstm":     X_lstm,
            "X_flat":     X_flat,
            "y_scaled":   y_win,
            "y_original": y_original,
            "indices":    idx[-(len(y_win)):],
        }

    def save(self, path: str = "models/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"💾 Preprocessor saved → {path}")

    @staticmethod
    def load(path: str = "models/preprocessor.pkl") -> "AQIPreprocessor":
        prep = joblib.load(path)
        print(f"📂 Preprocessor loaded ← {path}")
        return prep


def train_test_split_timeseries(data: dict, train_ratio=0.70, val_ratio=0.15) -> dict:
    """Time-series aware 70/15/15 split — no shuffling."""
    n = len(data["y_scaled"])
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    splits = {}
    for key in ["X_lstm", "X_flat", "y_scaled", "y_original"]:
        splits[f"{key}_train"] = data[key][:train_end]
        splits[f"{key}_val"]   = data[key][train_end:val_end]
        splits[f"{key}_test"]  = data[key][val_end:]
    print(f"\n📂 Split: Train={train_end:,} | Val={val_end-train_end:,} | Test={n-val_end:,}")
    return splits
