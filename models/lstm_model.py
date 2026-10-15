"""
=============================================================
AQI PREDICTION SYSTEM - LSTM MODEL
=============================================================
Module: lstm_model.py
Purpose: Bidirectional LSTM with Attention for AQI prediction
Why LSTM?
  - AQI is sequential: today's pollution depends on yesterday's
  - LSTM solves the vanishing gradient problem of vanilla RNNs
  - Long-term dependencies (e.g., seasonal trends) are captured
  - Bidirectional LSTM looks at future context too
  - Attention weights highlight which timesteps matter most
=============================================================
"""

import numpy as np
import os

# ── TensorFlow / Keras imports (with graceful fallback) ───────
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, callbacks
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not installed. Install: pip install tensorflow")


def build_lstm_model(
    input_shape: tuple,
    lstm_units: list = None,
    dropout_rate: float = 0.25,
    learning_rate: float = 0.001,
    use_attention: bool = True,
    use_bidirectional: bool = True,
) -> "keras.Model":
    """
    Build a Bidirectional LSTM model with optional Attention.

    Architecture:
        Input (batch, timesteps, features)
            ↓
        Bidirectional LSTM (256 units) + return sequences
            ↓
        Dropout (0.25)
            ↓
        Bidirectional LSTM (128 units) + return sequences
            ↓
        Dropout (0.25)
            ↓
        [Attention Layer]  ← weights important timesteps
            ↓
        Dense (64, ReLU)
            ↓
        Dropout (0.15)
            ↓
        Dense (1, Linear)  ← final AQI prediction

    Parameters:
        input_shape  : (timesteps, n_features)
        lstm_units   : [256, 128] → units per LSTM layer
        dropout_rate : Regularization between LSTM layers
        learning_rate: Adam optimizer rate
        use_attention: Apply scaled dot-product attention
        use_bidirectional: Wrap LSTM in Bidirectional wrapper

    Returns:
        Compiled Keras model
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required. pip install tensorflow")

    lstm_units = lstm_units or [256, 128]

    # ── Input ────────────────────────────────────────────────
    inputs = layers.Input(shape=input_shape, name="aqi_input")
    x = inputs

    # ── Stack LSTM layers ─────────────────────────────────────
    for i, units in enumerate(lstm_units):
        return_sequences = True  # All layers return sequences (for attention)
        lstm_layer = layers.LSTM(
            units,
            return_sequences=return_sequences,
            dropout=0.1,          # Input dropout
            recurrent_dropout=0.1, # Recurrent dropout
            name=f"lstm_{i+1}"
        )

        if use_bidirectional:
            # Bidirectional: processes sequence forward AND backward
            # Forward pass: captures trends leading to current state
            # Backward pass: captures how current state evolves
            x = layers.Bidirectional(lstm_layer, name=f"bilstm_{i+1}")(x)
        else:
            x = lstm_layer(x)

        x = layers.BatchNormalization(name=f"bn_{i+1}")(x)
        x = layers.Dropout(dropout_rate, name=f"dropout_{i+1}")(x)

    # ── Attention Layer ───────────────────────────────────────
    # Why Attention?
    # LSTM treats all timesteps somewhat equally. Attention learns
    # to focus on the most relevant time steps (e.g., rush-hour
    # spikes 6 hours ago matter more than light traffic 20h ago).
    if use_attention:
        # Self-attention: each timestep attends to all others
        attention_scores = layers.Dense(1, activation="tanh",
                                        name="att_dense")(x)
        attention_weights = layers.Softmax(axis=1,
                                           name="att_softmax")(attention_scores)
        # Weighted sum of LSTM outputs
        x = layers.Multiply(name="att_apply")([x, attention_weights])
        x = layers.Lambda(
            lambda t: tf.reduce_sum(t, axis=1),
            name="att_reduce"
        )(x)
    else:
        # Without attention: just take the last timestep output
        x = layers.Lambda(lambda t: t[:, -1, :], name="last_step")(x)

    # ── Dense prediction head ─────────────────────────────────
    x = layers.Dense(64, activation="relu", name="dense_1")(x)
    x = layers.Dropout(0.15, name="dropout_out")(x)
    x = layers.Dense(32, activation="relu", name="dense_2")(x)

    # Linear activation for regression (no upper bound on AQI)
    outputs = layers.Dense(1, activation="linear", name="aqi_output")(x)

    model = Model(inputs=inputs, outputs=outputs, name="AQI_LSTM")

    # ── Compile ───────────────────────────────────────────────
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0  # Gradient clipping for stability
        ),
        loss="huber",      # Huber loss: robust to outliers (vs MSE)
        metrics=["mae"]
    )

    # Print summary
    print("\n🧠 LSTM Model Architecture:")
    model.summary()
    return model


def train_lstm(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    model_save_path: str = "models/lstm_model.weights.h5",
) -> dict:
    """
    Train the LSTM model with callbacks for best-model saving
    and learning rate scheduling.

    Callbacks used:
    1. ModelCheckpoint: save only when val_loss improves
    2. EarlyStopping:   stop if no improvement for 10 epochs
    3. ReduceLROnPlateau: halve LR when stuck for 5 epochs
    4. TensorBoard:     visualize training curves

    Returns:
        dict with history and best val_loss
    """
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required")

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    cb_list = [
        # Save best model (by validation loss)
        callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        # Stop early if no improvement
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        # Reduce learning rate on plateau
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
    ]

    print(f"\n🏋️  Training LSTM | Epochs={epochs} | Batch={batch_size}")
    print(f"   Train: {X_train.shape} | Val: {X_val.shape}")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1,
        shuffle=False,   # CRITICAL: do NOT shuffle time-series data
    )

    best_val_loss = min(history.history["val_loss"])
    best_val_mae  = min(history.history["val_mae"])

    print(f"\n✅ LSTM Training complete!")
    print(f"   Best val_loss: {best_val_loss:.4f}")
    print(f"   Best val_mae : {best_val_mae:.4f}")
    print(f"   Model saved  : {model_save_path}")

    return {
        "history": history.history,
        "best_val_loss": best_val_loss,
        "best_val_mae": best_val_mae,
        "epochs_trained": len(history.history["loss"])
    }


def predict_lstm(
    model,
    X: np.ndarray,
    scaler_y,
    batch_size: int = 256
) -> np.ndarray:
    """
    Generate AQI predictions using trained LSTM model.

    Parameters:
        model    : Trained Keras model
        X        : Scaled LSTM input (samples, window, features)
        scaler_y : MinMaxScaler to inverse-transform predictions
        batch_size: Inference batch size

    Returns:
        np.ndarray of predicted AQI values (original scale)
    """
    y_pred_scaled = model.predict(X, batch_size=batch_size, verbose=0)
    y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
    return np.clip(y_pred, 0, 500)  # AQI ∈ [0, 500]


def load_lstm_model(path: str = "models/lstm_model.keras"):
    """Load a saved LSTM model from disk."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required")
    model = keras.models.load_model(path)
    print(f"📂 LSTM model loaded ← {path}")
    return model


# ── Fallback: Lightweight LSTM for environments without GPU ──
class SimpleLSTMNumpy:
    """
    Pure NumPy lightweight LSTM for demo when TensorFlow unavailable.
    Not suitable for production — use Keras version.
    Uses sklearn LinearRegression as stand-in.
    """
    def __init__(self):
        from sklearn.linear_model import Ridge
        self.model = Ridge(alpha=1.0)
        self._name = "SimpleLSTM_Fallback"

    def fit(self, X_3d: np.ndarray, y: np.ndarray):
        # Flatten 3D window to 2D
        X_2d = X_3d.reshape(X_3d.shape[0], -1)
        self.model.fit(X_2d, y)
        return self

    def predict(self, X_3d: np.ndarray) -> np.ndarray:
        X_2d = X_3d.reshape(X_3d.shape[0], -1)
        return self.model.predict(X_2d)
