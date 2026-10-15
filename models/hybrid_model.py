

import numpy as np
import joblib
import os

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class HybridAQIModel:
    """
    Hybrid LSTM + XGBoost model with a meta-learner (Level-2 model).

    Ensemble Strategy: Stacking
    ─────────────────────────────────────────────────────────
    Level 1 Models  : LSTM + XGBoost (base predictors)
    Level 2 Model   : Ridge Regression (meta-learner)

    The meta-learner learns the optimal combination weights
    from validation data. It can down-weight a model that
    performs poorly in certain AQI ranges.

    Alternative: Weighted Averaging
      hybrid = α × lstm_pred + (1-α) × xgb_pred
      α is optimized on validation MAE.
    """

    def __init__(
        self,
        lstm_model=None,
        xgb_model=None,
        blend_method: str = "stacking",  # 'stacking' | 'weighted'
        lstm_weight: float = 0.5,         # Used only for 'weighted' method
    ):
        self.lstm_model   = lstm_model
        self.xgb_model    = xgb_model
        self.blend_method = blend_method
        self.lstm_weight  = lstm_weight
        self.xgb_weight   = 1 - lstm_weight
        self.meta_model   = Ridge(alpha=1.0)
        self._meta_trained = False

    # ─────────────────────────────────────────────────────────
    # GET BASE MODEL PREDICTIONS
    # ─────────────────────────────────────────────────────────
    def _get_base_predictions(
        self,
        X_lstm: np.ndarray,
        X_flat: np.ndarray,
        scaler_y,
    ) -> np.ndarray:
        """
        Get predictions from both base models.

        Returns:
            np.ndarray shape (n_samples, 2) → [lstm_pred, xgb_pred]
        """
        # LSTM predictions (scaled → inverse transform)
        if self.lstm_model is None:
            lstm_preds = None
        else:
            try:
                lstm_raw = self.lstm_model.predict(X_lstm, batch_size=256, verbose=0)
            except TypeError:
                lstm_raw = self.lstm_model.predict(X_lstm)
            lstm_preds = scaler_y.inverse_transform(
                np.asarray(lstm_raw).reshape(-1, 1)
            ).ravel()
            lstm_preds = np.clip(lstm_preds, 0, 500)

        # XGBoost predictions (already on original scale)
        if XGB_AVAILABLE and hasattr(self.xgb_model, 'model'):
            import xgboost as xgb
            #dmatrix = xgb.DMatrix(X_flat)
            dmatrix = xgb.DMatrix(
                X_flat,
                feature_names=self.xgb_model.feature_names
            )
            xgb_preds = self.xgb_model.model.predict(dmatrix)
        else:
            xgb_preds = self.xgb_model.predict(X_flat)

        xgb_preds = np.clip(xgb_preds, 0, 500)
        if lstm_preds is None:
            lstm_preds = xgb_preds.copy()

        return np.column_stack([lstm_preds, xgb_preds])

    # ─────────────────────────────────────────────────────────
    # TRAIN META-LEARNER
    # ─────────────────────────────────────────────────────────
    def fit_meta(
        self,
        X_lstm_val: np.ndarray,
        X_flat_val: np.ndarray,
        y_val: np.ndarray,
        scaler_y,
    ):
        
        print("\n🔗 Training Hybrid Meta-Learner...")

        base_preds = self._get_base_predictions(X_lstm_val, X_flat_val, scaler_y)
        lstm_preds = base_preds[:, 0]
        xgb_preds  = base_preds[:, 1]

        if self.blend_method == "weighted":
            # Grid-search optimal weight α
            best_mae = float("inf")
            best_alpha = 0.5

            for alpha in np.linspace(0, 1, 21):
                preds = alpha * lstm_preds + (1 - alpha) * xgb_preds
                mae = mean_absolute_error(y_val, preds)
                if mae < best_mae:
                    best_mae = mae
                    best_alpha = alpha

            self.lstm_weight = best_alpha
            self.xgb_weight  = 1 - best_alpha
            self._meta_trained = True

            print(f"   Optimal weights → LSTM: {best_alpha:.2f}, "
                  f"XGBoost: {1-best_alpha:.2f}")
            print(f"   Best val MAE: {best_mae:.2f}")

        elif self.blend_method == "stacking":
            # Train Ridge meta-learner
            # Extra feature: |lstm - xgb| signals disagreement
            disagreement = np.abs(lstm_preds - xgb_preds).reshape(-1, 1)
            meta_X = np.column_stack([base_preds, disagreement])

            self.meta_model.fit(meta_X, y_val)
            self._meta_trained = True

            meta_preds = self.meta_model.predict(meta_X)
            meta_mae = mean_absolute_error(y_val, meta_preds)
            print(f"   Meta-learner val MAE: {meta_mae:.2f}")

        print("✅ Hybrid model ready!")

    # ─────────────────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────────────────
    def predict(
        self,
        X_lstm: np.ndarray,
        X_flat: np.ndarray,
        scaler_y,
    ) -> dict:
        """
        Generate hybrid predictions.

        Returns dict with:
            hybrid    : Final blended prediction
            lstm      : LSTM-only prediction
            xgb       : XGBoost-only prediction
            confidence: Inverse of LSTM/XGBoost disagreement
        """
        base_preds = self._get_base_predictions(X_lstm, X_flat, scaler_y)
        lstm_preds = base_preds[:, 0]
        xgb_preds  = base_preds[:, 1]

        if self.blend_method == "weighted":
            hybrid = (self.lstm_weight * lstm_preds
                      + self.xgb_weight * xgb_preds)

        elif self.blend_method == "stacking" and self._meta_trained:
            disagreement = np.abs(lstm_preds - xgb_preds).reshape(-1, 1)
            meta_X = np.column_stack([base_preds, disagreement])
            hybrid = self.meta_model.predict(meta_X)
        else:
            # Fallback: simple average
            hybrid = 0.5 * lstm_preds + 0.5 * xgb_preds

        hybrid = np.clip(hybrid, 0, 500)

        # Confidence: lower disagreement = higher confidence
        disagreement = np.abs(lstm_preds - xgb_preds)
        confidence = 1 / (1 + disagreement / 50)  # Normalize to [0,1]

        return {
            "hybrid":     hybrid,
            "lstm":       lstm_preds,
            "xgb":        xgb_preds,
            "confidence": confidence,
        }

    # ─────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────
    def save(self, path: str = "models/hybrid_meta.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save meta-model and weights (not base models separately)
        state = {
            "blend_method":  self.blend_method,
            "lstm_weight":   self.lstm_weight,
            "xgb_weight":    self.xgb_weight,
            "meta_model":    self.meta_model,
            "_meta_trained": self._meta_trained,
        }
        joblib.dump(state, path)
        print(f"💾 Hybrid meta-model saved → {path}")

    def load_meta(self, path: str = "models/hybrid_meta.pkl"):
        state = joblib.load(path)
        self.blend_method  = state["blend_method"]
        self.lstm_weight   = state["lstm_weight"]
        self.xgb_weight    = state["xgb_weight"]
        self.meta_model    = state["meta_model"]
        self._meta_trained = state["_meta_trained"]
        print(f"📂 Hybrid meta-model loaded ← {path}")


# ─────────────────────────────────────────────────────────────
# MODEL EVALUATION UTILITIES
# ─────────────────────────────────────────────────────────────
def evaluate_models(
    predictions: dict,
    y_true: np.ndarray,
    model_names: list = None,
) -> dict:
    """
    Evaluate all models using MAE, RMSE, and R² Score.

    Metrics explained:
    ─────────────────
    MAE  (Mean Absolute Error) :
        Average absolute difference. Same unit as AQI.
        MAE=10 means predictions are off by ~10 AQI on average.
        Lower is better.

    RMSE (Root Mean Squared Error):
        Penalizes large errors more than MAE.
        Useful for detecting if model occasionally makes big mistakes.
        Lower is better.

    R²  (Coefficient of Determination):
        How much variance the model explains.
        R²=1.0: perfect. R²=0: same as predicting the mean.
        Higher is better (max = 1.0).
    """
    if model_names is None:
        model_names = list(predictions.keys())

    results = {}
    print("\n" + "═" * 60)
    print("📊 MODEL EVALUATION COMPARISON")
    print("═" * 60)
    print(f"{'Model':<20} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("─" * 60)

    for name in model_names:
        if name not in predictions:
            continue
        preds = predictions[name]
        mae  = mean_absolute_error(y_true, preds)
        rmse = np.sqrt(mean_squared_error(y_true, preds))
        r2   = r2_score(y_true, preds)

        results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
        print(f"{name:<20} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f}")

    print("═" * 60)

    # Find best model
    best = min(results, key=lambda k: results[k]["MAE"])
    print(f"\n🏆 Best Model (lowest MAE): {best}")
    print(f"   MAE={results[best]['MAE']:.2f} | "
          f"RMSE={results[best]['RMSE']:.2f} | "
          f"R²={results[best]['R2']:.4f}")
    return results
