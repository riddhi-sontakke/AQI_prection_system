"""
Local-file AQI prediction pipeline.

This module never calls external APIs and never creates mock readings. It uses
only the selected city CSV from data/ plus trained model artifacts.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import sys
import time
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.city_data import canonical_city, resolve_city_dataset
from data.data_loader import RealDataLoader
from utils.aqi_advisor import AQIAdvisor, AQIResult


class RealTimePipeline:
    """
    Prediction pipeline backed by local CSV files only.

    Data flow:
      1. Resolve city -> bundled CSV, unless an explicit local CSV is supplied.
      2. Load and standardize that CSV with RealDataLoader.
      3. Keep the latest rows as the prediction context.
      4. Run the trained preprocessor and hybrid model.
      5. Return only model predictions and source row values to the frontend.
    """

    def __init__(
        self,
        city: str = "Bengaluru",
        preprocessor=None,
        lstm_model=None,
        xgb_model=None,
        hybrid_model=None,
        window_size: int = 24,
        training_csv: Optional[str] = None,
        buffer_rows: int = 240,
    ):
        self.city = canonical_city(city)
        self.preprocessor = preprocessor
        self.lstm_model = lstm_model
        self.xgb_model = xgb_model
        self.hybrid_model = hybrid_model
        self.window_size = window_size
        self.training_csv = resolve_city_dataset(self.city, training_csv)
        self.buffer_rows = buffer_rows
        self.advisor = AQIAdvisor()

        self.history_df = pd.DataFrame()
        self._initialize_buffer()

    def _initialize_buffer(self):
        """Seed the prediction context from the latest rows of the local CSV."""
        loader = RealDataLoader(city=self.city)
        df = loader.load(self.training_csv)
        self.history_df = df.tail(self.buffer_rows).copy()
        print(
            f"History buffer seeded with {len(self.history_df)} local rows "
            f"from {os.path.basename(self.training_csv)}"
        )

    def _buffer_to_dataframe(self) -> pd.DataFrame:
        if self.history_df.empty:
            raise RuntimeError(
                f"No local data loaded for {self.city}. Check {self.training_csv}."
            )
        return self.history_df.copy()

    def predict_now(self, verbose: bool = True) -> AQIResult:
        """Run one prediction from the local CSV context."""
        if self.hybrid_model is None or self.preprocessor is None:
            raise RuntimeError(
                "Models not loaded. Train first with `python train.py --city "
                f"{self.city}`."
            )

        predicted_aqi, confidence, _ = self._run_model_prediction()
        now = _dt.datetime.now()
        result = self.advisor.generate_advice(
            aqi=predicted_aqi,
            confidence=confidence,
            city=self.city,
            hour=now.hour,
            month=now.month,
        )

        if verbose:
            result.print_report()
        return result

    def _processed_context(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Preprocess local context and return processed rows plus scaled X."""
        df = self._buffer_to_dataframe()
        df_proc = self.preprocessor.handle_missing_values(df)
        df_proc = self.preprocessor.remove_outliers(df_proc)
        df_proc = self.preprocessor.engineer_features(df_proc)
        X_scaled, _, _ = self.preprocessor.scale_features(df_proc, fit=False)
        return df_proc, X_scaled

    def _run_model_prediction(self) -> tuple[float, float, dict]:
        """Preprocess the local buffer and run a single hybrid inference."""
        _, X_scaled = self._processed_context()
        window_size = self.preprocessor.window_size

        if len(X_scaled) < window_size:
            raise RuntimeError(
                f"Only {len(X_scaled)} processed rows available; need at least "
                f"{window_size}. Use a larger local dataset."
            )

        window = X_scaled[-window_size:]
        X_window = window.reshape(1, window_size, window.shape[1])
        X_flat = window[-1:].reshape(1, -1)

        preds = self.hybrid_model.predict(
            X_window,
            X_flat,
            scaler_y=self.preprocessor.scaler_y,
        )
        current_data = self._latest_source_row()
        return (
            float(np.clip(preds["hybrid"][0], 0, 500)),
            float(np.clip(preds["confidence"][0], 0, 1)),
            current_data,
        )

    def _latest_source_row(self) -> dict:
        row = self.history_df.iloc[-1].to_dict()
        clean = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, np.integer)):
                clean[key] = float(value)
            elif pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        clean["city"] = self.city
        clean["timestamp"] = self.history_df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
        clean["source"] = os.path.basename(self.training_csv)
        return clean

    def predict_with_forecast(self, steps: int = 6) -> dict:
        """
        Return the current AQI prediction plus recursive model forecasts.

        The forecast uses the trained model recursively. It does not fetch API
        values and does not generate random readings.
        """
        if self.hybrid_model is None or self.preprocessor is None:
            return {
                "current_aqi": 0,
                "forecast": [],
                "confidence": 0.0,
                "source": "no_model",
                "current_data": {},
            }

        df_proc, X_scaled = self._processed_context()
        window_size = self.preprocessor.window_size
        if len(X_scaled) < window_size:
            raise RuntimeError(
                f"Only {len(X_scaled)} processed rows available; need at least "
                f"{window_size}."
            )

        window = X_scaled[-window_size:].copy()
        X_window = window.reshape(1, window_size, window.shape[1])
        X_flat = window[-1:].reshape(1, -1)
        preds = self.hybrid_model.predict(
            X_window,
            X_flat,
            scaler_y=self.preprocessor.scaler_y,
        )
        current_aqi = float(np.clip(preds["hybrid"][0], 0, 500))
        confidence = float(np.clip(preds["confidence"][0], 0, 1))

        current_data = self._latest_source_row()
        current_data["AQI"] = current_aqi

        expected = list(self.preprocessor.feature_names)
        aqi_lag_idx = next((i for i, name in enumerate(expected) if name == "AQI_lag_1"), None)
        now = _dt.datetime.now()
        forecast = []
        last_aqi = current_aqi

        for step in range(1, steps + 1):
            pred_scaled = float(self.preprocessor.scaler_y.transform([[last_aqi]])[0, 0])
            next_row = window[-1].copy()
            if aqi_lag_idx is not None:
                next_row[aqi_lag_idx] = pred_scaled
            window = np.vstack([window[1:], next_row])

            X_window = window.reshape(1, window_size, window.shape[1])
            X_flat = window[-1:].reshape(1, -1)
            step_preds = self.hybrid_model.predict(
                X_window,
                X_flat,
                scaler_y=self.preprocessor.scaler_y,
            )
            last_aqi = float(np.clip(step_preds["hybrid"][0], 0, 500))
            step_time = now + _dt.timedelta(hours=step)
            forecast.append(
                {
                    "time": step_time.strftime("%H:%M"),
                    "label": f"t+{step}h",
                    "aqi": round(last_aqi, 1),
                }
            )

        return {
            "current_aqi": round(current_aqi, 1),
            "forecast": forecast,
            "confidence": round(confidence * 100, 1),
            "source": "model",
            "current_data": current_data,
        }

    def run_scheduler(
        self,
        interval_minutes: int = 15,
        max_iterations: int = None,
        output_file: str = None,
    ):
        """Run repeated local-file predictions at a fixed interval."""
        print(f"\nPrediction scheduler started: every {interval_minutes} min")
        print(f"City: {self.city} | Dataset: {self.training_csv}\n")

        iteration = 0
        results_log = []
        while True:
            try:
                iteration += 1
                print(
                    f"\n[{_dt.datetime.now().strftime('%H:%M:%S')}] "
                    f"Prediction #{iteration}"
                )
                result = self.predict_now(verbose=True)
                results_log.append(result.to_dict())

                if output_file:
                    with open(output_file, "w") as f:
                        json.dump(results_log[-50:], f, indent=2, default=str)

                if max_iterations and iteration >= max_iterations:
                    break
                time.sleep(interval_minutes * 60)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"Prediction error: {exc}")
                time.sleep(60)

        return results_log


def batch_predict(hybrid_model, preprocessed_data: dict, preprocessor) -> dict:
    """Generate predictions for an existing preprocessed test set."""
    X_lstm = preprocessed_data["X_lstm_test"]
    X_flat = preprocessed_data["X_flat_test"]
    y_true = preprocessed_data["y_original_test"]

    preds = hybrid_model.predict(
        X_lstm,
        X_flat,
        scaler_y=preprocessor.scaler_y,
    )
    return {
        "y_true": y_true,
        "y_hybrid": preds["hybrid"],
        "y_lstm": preds["lstm"],
        "y_xgb": preds["xgb"],
    }


if __name__ == "__main__":
    for city in ["Bengaluru", "Delhi", "Mumbai"]:
        pipeline = RealTimePipeline(city=city)
        result = pipeline.predict_with_forecast(steps=6)
        print(f"{city}: source={result['source']}")
