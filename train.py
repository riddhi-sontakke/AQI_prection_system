"""
=============================================================
AQI PREDICTION SYSTEM v2 - MAIN TRAINING SCRIPT
=============================================================
Module  : train.py
Version : 2.0 (Real Dataset Integration)

CHANGES FROM v1:
  [REMOVED] --start-date, --end-date arguments (not needed)
  [REMOVED] generate_synthetic_aqi_dataset() calls
  [REMOVED] Synthetic data fallback paths
  [ADDED]   --data argument: path to your real CSV/Excel file
  [ADDED]   --column-map argument: JSON string for column renames
  [ADDED]   --timestamp-format argument: custom datetime format
  [ADDED]   --city-col argument: city column if multi-city dataset

Usage:
    python train.py --data my_aqi_data.csv --city Pune

    # If your columns have non-standard names:
    python train.py --data data.csv --city Delhi \
        --column-map '{"Temp_C": "Temperature", "Wind_kmh": "WindSpeed"}'

    # If timestamp format needs specifying:
    python train.py --data data.csv --timestamp-format "%d/%m/%Y %H:%M"

    # XGBoost only (faster, no TensorFlow needed):
    python train.py --data data.csv --no-lstm
=============================================================
"""

import argparse
import os
import sys
import json
import time
import warnings
import numpy as np
warnings.filterwarnings("ignore")

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, os.path.dirname(__file__))

# [v2] Use RealDataLoader instead of synthetic generator
from data.data_loader import load_real_dataset
from data.preprocessor import AQIPreprocessor, train_test_split_timeseries
from data.city_data import artifact_dir_for_city, canonical_city, city_key, resolve_city_dataset
from models.hybrid_model import HybridAQIModel, evaluate_models
from utils.aqi_advisor import AQIAdvisor

try:
    from models.lstm_model import build_lstm_model, train_lstm, predict_lstm
    TF_AVAILABLE = True
    TF_IMPORT_ERROR = None
except Exception as exc:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = exc
    print(f"TensorFlow/LSTM import failed: {exc}")

try:
    from models.xgboost_model import AQIXGBoostModel
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not available — using sklearn GradientBoosting")


def run_training_pipeline(args) -> dict:
    """Full end-to-end training pipeline using real dataset."""
    args.city = canonical_city(args.city)
    data_path = resolve_city_dataset(args.city, args.data)
    artifact_dir = artifact_dir_for_city(args.city)
    preprocessor_path = os.path.join(artifact_dir, "preprocessor.pkl")
    lstm_path = os.path.join(artifact_dir, "lstm_model.weights.h5")
    xgb_path = os.path.join(artifact_dir, "xgboost_model.pkl")
    hybrid_path = os.path.join(artifact_dir, "hybrid_meta.pkl")

    print("\n" + "═" * 65)
    print(f"  🌬️  AQI PREDICTION SYSTEM v2.0 — REAL DATA TRAINING")
    print(f"  Dataset: {data_path}")
    print(f"  City:    {args.city} | Epochs: {args.epochs}")
    print("═" * 65)

    start_time = time.time()
    os.makedirs(artifact_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # ── STEP 1: LOAD REAL DATASET ─────────────────────────────
    # [v2 CHANGE] load_real_dataset() replaces generate_synthetic_aqi_dataset()
    print("\n📊 STEP 1: LOADING REAL DATASET")
    print("─" * 40)

    column_map = None
    if args.column_map:
        try:
            column_map = json.loads(args.column_map)
            print(f"  Custom column map: {column_map}")
        except json.JSONDecodeError:
            print(f"⚠️  Invalid --column-map JSON, ignoring: {args.column_map}")

    df = load_real_dataset(
        filepath=data_path,
        city=args.city,
        column_map=column_map,
        timestamp_format=args.timestamp_format if args.timestamp_format else None,
    )

    # If multi-city dataset, filter to requested city
    if "city" in df.columns and args.city != "Unknown":
        mask = df["city"].str.lower() == args.city.lower()
        if mask.sum() > 100:
            df = df[mask].drop(columns=["city"])
            print(f"\n  Filtered to city='{args.city}': {len(df):,} rows")

    print(f"\n  Dataset ready: {len(df):,} rows × {df.shape[1]} cols")
    print(f"  Date range: {df.index[0]} → {df.index[-1]}")

    # ── STEP 2: PREPROCESSING ─────────────────────────────────
    print("\n\n🔧 STEP 2: DATA PREPROCESSING")
    print("─" * 40)

    preprocessor = AQIPreprocessor(
        window_size=args.window_size if args.window_size > 0 else None,
        forecast_horizon=args.forecast_horizon if args.forecast_horizon > 0 else None,
    )

    # [v2 CHANGE] fit_transform() now takes the real DataFrame directly
    data = preprocessor.fit_transform(df)
    splits = train_test_split_timeseries(data)
    preprocessor.save(preprocessor_path)

    # ── STEP 3: LSTM MODEL ────────────────────────────────────
    lstm_model = None
    if TF_AVAILABLE and not args.no_lstm:
        print("\n\n🧠 STEP 3: LSTM MODEL TRAINING")
        print("─" * 40)
        input_shape = (preprocessor.window_size, splits["X_lstm_train"].shape[2])
        lstm_model = build_lstm_model(
            input_shape=input_shape,
            lstm_units=[256, 128],
            dropout_rate=0.25,
        )
        train_lstm(
            lstm_model,
            splits["X_lstm_train"], splits["y_scaled_train"],
            splits["X_lstm_val"],   splits["y_scaled_val"],
            epochs=args.epochs,
            batch_size=args.batch_size,
            model_save_path=lstm_path,
        )
    else:
        reason = "--no-lstm was passed" if args.no_lstm else f"TensorFlow/LSTM import failed: {TF_IMPORT_ERROR}"
        print(f"\nSTEP 3: LSTM skipped - {reason}. XGBoost-only model will be served")
        lstm_model = None

    # ── STEP 4: XGBOOST MODEL ────────────────────────────────
    xgb_model = None
    if XGB_AVAILABLE:
        print("\n\n🌲 STEP 4: XGBOOST MODEL TRAINING")
        print("─" * 40)
        xgb_model = AQIXGBoostModel()
        if args.tune_xgb:
            xgb_model.tune_hyperparameters(
                splits["X_flat_train"], splits["y_original_train"],
                method="random", n_iter=20,
            )
        xgb_model.train(
            splits["X_flat_train"], splits["y_original_train"],
            splits["X_flat_val"],   splits["y_original_val"],
            feature_names=preprocessor.feature_names,
        )
        xgb_model.get_top_features(top_n=10)
        xgb_model.save(xgb_path)
    else:
        print("\n🌲 STEP 4: XGBoost — sklearn fallback")
        from models.xgboost_model import GradientBoostingFallback
        xgb_model = GradientBoostingFallback()
        xgb_model.train(splits["X_flat_train"], splits["y_original_train"])

    # ── STEP 5: HYBRID MODEL ──────────────────────────────────
    print("\n\n🔗 STEP 5: HYBRID MODEL")
    print("─" * 40)
    hybrid = HybridAQIModel(
        lstm_model=lstm_model,
        xgb_model=xgb_model,
        blend_method=args.blend_method,
    )
    if lstm_model is not None:
        hybrid.fit_meta(
            splits["X_lstm_val"], splits["X_flat_val"],
            splits["y_original_val"],
            scaler_y=preprocessor.scaler_y,
        )
    else:
        hybrid.lstm_weight = 0.0
        hybrid.xgb_weight  = 1.0
        hybrid._meta_trained = True
    hybrid.save(hybrid_path)

    # ── STEP 6: EVALUATION ───────────────────────────────────
    print("\n\n📊 STEP 6: MODEL EVALUATION")
    print("─" * 40)
    all_predictions = {}
    y_test = splits["y_original_test"]

    if xgb_model:
        all_predictions["XGBoost"] = np.clip(xgb_model.predict(splits["X_flat_test"]), 0, 500)

    if lstm_model is not None:
        raw = lstm_model.predict(splits["X_lstm_test"], batch_size=256, verbose=0)
        all_predictions["LSTM"] = np.clip(
            preprocessor.scaler_y.inverse_transform(raw).ravel(), 0, 500
        )
        hybrid_preds = hybrid.predict(
            splits["X_lstm_test"], splits["X_flat_test"],
            scaler_y=preprocessor.scaler_y,
        )
        all_predictions["Hybrid"] = hybrid_preds["hybrid"]
    else:
        if "XGBoost" in all_predictions:
            all_predictions["Hybrid"] = all_predictions["XGBoost"]

    eval_results = evaluate_models(all_predictions, y_test)

    # ── SUMMARY ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'═'*65}")
    print(f"  ✅ TRAINING COMPLETE in {elapsed:.1f}s")
    print(f"  Models saved to: {artifact_dir}")
    print(f"  🚀 Predict: python predict.py --city {args.city}")
    print(f"  🌐 Dashboard: streamlit run deployment/streamlit_app.py")
    print("═" * 65)

    summary = {
        "city":          args.city,
        "dataset":       data_path,
        "artifact_dir":  artifact_dir,
        "lstm_enabled":  lstm_model is not None,
        "training_time": round(elapsed, 1),
        "dataset_rows":  len(df),
        "eval_results":  eval_results,
        "timestamp":     time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = os.path.join("logs", f"{city_key(args.city)}_training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(artifact_dir, "training_config.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open("logs/training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train AQI Prediction System v2 — Real Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # [v2 ADDED] --data replaces --start-date/--end-date
    parser.add_argument("--data", default=None,
                        help="Optional local dataset path. If omitted, the bundled CSV for --city is used.")
    parser.add_argument("--city",          default="Bengaluru",
                        help="City name for labeling")
    # [v2 ADDED] Column mapping and timestamp format
    parser.add_argument("--column-map",    default=None,
                        help='JSON dict of column renames e.g. \'{"T": "Temperature"}\'')
    parser.add_argument("--timestamp-format", default=None,
                        help='Datetime parse format e.g. "%%d/%%m/%%Y %%H:%%M"')

    parser.add_argument("--epochs",        type=int, default=50)
    parser.add_argument("--batch-size",    type=int, default=64)
    # 0 = auto-detect from frequency
    parser.add_argument("--window-size",   type=int, default=0,
                        help="LSTM lookback window. 0 = auto (24h or 30d)")
    parser.add_argument("--forecast-horizon", type=int, default=0,
                        help="Steps ahead to predict. 0 = auto (6h or 3d)")
    parser.add_argument("--blend-method", default="stacking",
                        choices=["stacking", "weighted"])
    parser.add_argument("--tune-xgb",     action="store_true")
    parser.add_argument("--no-lstm",      action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_training_pipeline(args)
