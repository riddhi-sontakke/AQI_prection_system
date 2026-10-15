"""
=============================================================
AQI PREDICTION SYSTEM - XGBOOST MODEL
=============================================================
Module: xgboost_model.py
Purpose: XGBoost regressor for AQI prediction with
         hyperparameter tuning and feature importance analysis
Why XGBoost?
  - Handles tabular data extremely well
  - Non-linear feature interactions automatically captured
  - Built-in feature importance reveals what drives AQI
  - Robust to outliers with tree-based splits
  - Gradient boosting corrects previous trees' errors
  - Fast inference for real-time prediction
=============================================================
"""

import numpy as np
import pandas as pd
import joblib
import os
import json

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not installed. Install: pip install xgboost")

try:
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# DEFAULT HYPERPARAMETERS (tuned for AQI prediction)
# ─────────────────────────────────────────────────────────────
DEFAULT_XGB_PARAMS = {
    # Tree structure
    "n_estimators":     500,    # Number of boosting rounds
    "max_depth":        6,      # Max tree depth (deeper = more overfit risk)
    "min_child_weight": 3,      # Min samples in leaf (regularization)

    # Learning
    "learning_rate":    0.05,   # Step size (lower = more robust, slower)
    "subsample":        0.8,    # Row sampling per tree (prevents overfit)
    "colsample_bytree": 0.8,    # Feature sampling per tree

    # Regularization
    "reg_alpha":        0.1,    # L1 (lasso) regularization
    "reg_lambda":       1.0,    # L2 (ridge) regularization
    "gamma":            0.1,    # Min loss reduction to split a node

    # Objective & evaluation
    "objective":        "reg:squarederror",
    "eval_metric":      "mae",

    # Performance
    "n_jobs":           -1,     # Use all CPU cores
    "random_state":     42,
    "verbosity":        1,
}

# Hyperparameter search space for tuning
PARAM_SEARCH_SPACE = {
    "n_estimators":     [200, 500, 800],
    "max_depth":        [4, 6, 8],
    "learning_rate":    [0.01, 0.05, 0.1],
    "subsample":        [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "min_child_weight": [1, 3, 5],
    "reg_alpha":        [0, 0.1, 0.5],
    "reg_lambda":       [0.5, 1.0, 2.0],
}


class AQIXGBoostModel:
    """
    XGBoost model for AQI prediction with:
    - Hyperparameter tuning
    - Feature importance analysis
    - Indian city-specific optimizations
    """

    def __init__(self, params: dict = None):
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost required: pip install xgboost")

        self.params = params or DEFAULT_XGB_PARAMS.copy()
        self.model  = None
        self.feature_names = None
        self.feature_importances_ = None

    # ─────────────────────────────────────────────────────────
    # TRAINING
    # ─────────────────────────────────────────────────────────
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: list = None,
        early_stopping_rounds: int = 30,
    ) -> dict:
        """
        Train XGBoost with early stopping.

        Early stopping:
        - Trains up to n_estimators rounds
        - Stops if val_mae doesn't improve for 30 rounds
        - Uses the best model at that checkpoint

        Parameters:
            X_train, y_train : Training data
            X_val, y_val     : Validation data
            feature_names    : Column names for importance plots
            early_stopping_rounds: Patience for early stopping

        Returns:
            Training metrics dict
        """
        self.feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        # Build DMatrix (XGBoost's optimized data structure)
        dtrain = xgb.DMatrix(
            X_train, label=y_train,
            feature_names=self.feature_names
        )
        dval = xgb.DMatrix(
            X_val, label=y_val,
            feature_names=self.feature_names
        )

        print(f"\n🌲 Training XGBoost | Trees={self.params['n_estimators']}")
        print(f"   Train: {X_train.shape} | Val: {X_val.shape}")

        # Train with eval monitoring
        evals_result = {}
        params = {k: v for k, v in self.params.items()
                  if k not in ["n_estimators"]}  # n_estimators = num_boost_round

        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.params["n_estimators"],
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=50
        )

        # Feature importance
        self.feature_importances_ = self.model.get_score(
            importance_type="gain"  # Gain = most meaningful importance type
        )

        best_round = self.model.best_iteration
        best_score = self.model.best_score

        print(f"\n✅ XGBoost Training complete!")
        print(f"   Best round    : {best_round}")
        print(f"   Best val_mae  : {best_score:.4f}")

        return {
            "evals_result": evals_result,
            "best_iteration": best_round,
            "best_score": best_score,
        }

    # ─────────────────────────────────────────────────────────
    # HYPERPARAMETER TUNING
    # ─────────────────────────────────────────────────────────
    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        method: str = "random",
        n_iter: int = 30,
        cv: int = 3,
    ) -> dict:
        """
        Hyperparameter tuning via Random Search (fast) or Grid Search.

        Why Random Search?
        - Grid search tests ALL combinations: with 8 params × 3 values = 6,561 fits
        - Random search samples n_iter combinations: finds near-optimal much faster
        - Studies show random search is usually 90%+ as good as grid search

        Returns:
            Best hyperparameters
        """
        if not SKLEARN_AVAILABLE:
            print("sklearn required for tuning")
            return self.params

        print(f"\n🔍 Hyperparameter tuning | method={method} | n_iter={n_iter}")

        base_xgb = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=42,
            verbosity=0
        )

        if method == "random":
            searcher = RandomizedSearchCV(
                base_xgb,
                PARAM_SEARCH_SPACE,
                n_iter=n_iter,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
        else:
            # Grid search (expensive)
            searcher = GridSearchCV(
                base_xgb,
                PARAM_SEARCH_SPACE,
                cv=cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                verbose=1,
            )

        searcher.fit(X_train, y_train)

        best = searcher.best_params_
        print(f"\n✅ Best params found: {json.dumps(best, indent=2)}")
        self.params.update(best)
        return best

    # ─────────────────────────────────────────────────────────
    # PREDICTION
    # ─────────────────────────────────────────────────────────
    def predict(
        self,
        X: np.ndarray,
        feature_names: list = None
    ) -> np.ndarray:
        """
        Generate AQI predictions.

        Note: XGBoost predictions are on the ORIGINAL AQI scale
        (not scaled like LSTM), because XGBoost is tree-based
        and doesn't require normalized inputs.
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        fn = feature_names or self.feature_names
        dmatrix = xgb.DMatrix(X, feature_names=fn)
        preds = self.model.predict(dmatrix)
        return np.clip(preds, 0, 500)

    # ─────────────────────────────────────────────────────────
    # FEATURE IMPORTANCE
    # ─────────────────────────────────────────────────────────
    def get_top_features(self, top_n: int = 15) -> pd.DataFrame:
        """
        Returns the top-N most important features by gain.

        Gain = average improvement in loss when a feature is used
        in a tree split. Higher = more important.
        """
        if self.feature_importances_ is None:
            return pd.DataFrame()

        imp_df = pd.DataFrame(
            list(self.feature_importances_.items()),
            columns=["feature", "importance_gain"]
        ).sort_values("importance_gain", ascending=False).head(top_n)

        print(f"\n📊 Top {top_n} Features (by gain):")
        print(imp_df.to_string(index=False))
        return imp_df

    # ─────────────────────────────────────────────────────────
    # SAVE / LOAD
    # ─────────────────────────────────────────────────────────
    def save(self, path: str = "models/xgboost_model.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"💾 XGBoost model saved → {path}")

    @staticmethod
    def load(path: str = "models/xgboost_model.pkl") -> "AQIXGBoostModel":
        model = joblib.load(path)
        print(f"📂 XGBoost model loaded ← {path}")
        return model


# ─────────────────────────────────────────────────────────────
# SKLEARN FALLBACK (when xgboost unavailable)
# ─────────────────────────────────────────────────────────────
class GradientBoostingFallback:
    """
    Uses sklearn's GradientBoostingRegressor as fallback.
    Performance is similar for small datasets.
    """
    def __init__(self):
        from sklearn.ensemble import GradientBoostingRegressor
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        self.feature_names = None

    def train(self, X_train, y_train, **kwargs):
        print("🌲 Training GradientBoosting (sklearn fallback)...")
        self.model.fit(X_train, y_train)
        return {}

    def predict(self, X, **kwargs):
        return np.clip(self.model.predict(X), 0, 500)
