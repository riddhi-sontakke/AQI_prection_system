"""Load city-specific trained model artifacts into a prediction pipeline."""

from __future__ import annotations

import os
from typing import Optional

from data.city_data import artifact_dir_for_city, canonical_city, resolve_city_dataset
from data.preprocessor import AQIPreprocessor
from models.hybrid_model import HybridAQIModel
from models.xgboost_model import AQIXGBoostModel
from pipeline.realtime_pipeline import RealTimePipeline


def _load_lstm_if_available(preprocessor: AQIPreprocessor, weights_path: str):
    if not os.path.exists(weights_path):
        return None

    from models.lstm_model import build_lstm_model

    model = build_lstm_model(
        input_shape=(preprocessor.window_size, len(preprocessor.feature_names)),
        lstm_units=[256, 128],
        dropout_rate=0.25,
    )
    model.load_weights(weights_path)
    return model


def load_pipeline_for_city(
    city: str,
    data_path: Optional[str] = None,
    models_root: str = "models",
) -> RealTimePipeline:
    """Return a RealTimePipeline with local CSV and trained city artifacts loaded."""
    display_city = canonical_city(city)
    dataset = resolve_city_dataset(display_city, data_path)
    artifact_dir = artifact_dir_for_city(display_city, root=os.path.abspath(models_root))

    preprocessor_path = os.path.join(artifact_dir, "preprocessor.pkl")
    xgb_path = os.path.join(artifact_dir, "xgboost_model.pkl")
    hybrid_path = os.path.join(artifact_dir, "hybrid_meta.pkl")
    lstm_weights_path = os.path.join(artifact_dir, "lstm_model.weights.h5")

    missing = [
        path for path in [preprocessor_path, xgb_path, hybrid_path]
        if not os.path.exists(path)
    ]
    if missing:
        names = ", ".join(os.path.basename(path) for path in missing)
        raise FileNotFoundError(
            f"Missing trained artifacts for {display_city}: {names}. "
            f"Run `python train.py --city {display_city}` first."
        )

    preprocessor = AQIPreprocessor.load(preprocessor_path)
    xgb_model = AQIXGBoostModel.load(xgb_path)
    lstm_model = _load_lstm_if_available(preprocessor, lstm_weights_path)

    hybrid_model = HybridAQIModel(lstm_model=lstm_model, xgb_model=xgb_model)
    hybrid_model.load_meta(hybrid_path)

    pipeline = RealTimePipeline(
        city=display_city,
        preprocessor=preprocessor,
        lstm_model=lstm_model,
        xgb_model=xgb_model,
        hybrid_model=hybrid_model,
        window_size=preprocessor.window_size,
        training_csv=dataset,
    )
    return pipeline
