"""
LSTM Forecasting Module
Deep Learning recurrent neural network designed to capture long- and short-term
temporal dependencies in Kubernetes cluster workload time-series metrics.
"""

import os
# Suppress non-critical TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MODELS_DIR,
    WINDOW_SIZE,
    LSTM_EPOCHS,
    LSTM_BATCH_SIZE,
    LSTM_UNITS,
    LSTM_DROPOUT
)


class LSTMForecaster:
    def __init__(self, window_size=WINDOW_SIZE, num_features=4):
        self.window_size = window_size
        self.num_features = num_features
        self.model = None
        self.model_path = MODELS_DIR / "lstm_model.keras"

    def build_model(self):
        """Construct stacked LSTM architecture with dropout regularizers."""
        inputs = layers.Input(shape=(self.window_size, self.num_features))
        
        # Layer 1: Recurrent temporal representation
        x = layers.LSTM(LSTM_UNITS[0], return_sequences=True, name="lstm_layer_1")(inputs)
        x = layers.Dropout(LSTM_DROPOUT, name="dropout_1")(x)
        
        # Layer 2: Higher abstraction of sequence patterns
        x = layers.LSTM(LSTM_UNITS[1], return_sequences=False, name="lstm_layer_2")(x)
        x = layers.Dropout(LSTM_DROPOUT, name="dropout_2")(x)
        
        # Dense Feed-Forward layers
        x = layers.Dense(16, activation="relu", name="dense_1")(x)
        outputs = layers.Dense(1, activation="linear", name="output_layer")(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name="Kubernetes_Workload_LSTM")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"]
        )
        self.model = model
        return model

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=LSTM_EPOCHS, batch_size=LSTM_BATCH_SIZE):
        """Train the LSTM network with early stopping."""
        if self.model is None:
            self.num_features = X_train.shape[2]
            self.build_model()

        validation_data = (X_val, y_val) if X_val is not None else None
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if validation_data else "loss",
                patience=5,
                restore_best_weights=True
            )
        ]
        
        print(f"[LSTM] Training model on {len(X_train)} samples across {epochs} epochs...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        self.save_model()
        return history

    def predict(self, X):
        """Generate forecasts for input sequence tensor."""
        if self.model is None:
            self.load_model()
        preds = self.model.predict(X, verbose=0)
        return preds.flatten()

    def evaluate(self, X_test, y_test):
        """Evaluate model performance using RMSE, MAE, and R2-score."""
        preds = self.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        metrics = {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
        print(f"[LSTM Evaluation] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        return metrics, preds

    def save_model(self):
        """Save Keras model weights and topology."""
        self.model.save(self.model_path)
        print(f"[LSTM] Model saved to {self.model_path}")

    def load_model(self):
        """Load trained model from disk."""
        if self.model_path.exists():
            self.model = keras.models.load_model(self.model_path)
            print(f"[LSTM] Loaded model successfully from {self.model_path}")
            return True
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train model first.")


if __name__ == "__main__":
    import pandas as pd
    from config.settings import DATA_DIR
    from preprocessing.preprocessor import WorkloadPreprocessor
    
    df_train = pd.read_csv(DATA_DIR / "workload_train.csv")
    df_test = pd.read_csv(DATA_DIR / "workload_test.csv")
    
    prep = WorkloadPreprocessor()
    prep.fit(df_train)
    
    X_train, y_train = prep.prepare_lstm_data(df_train)
    X_test, y_test = prep.prepare_lstm_data(df_test)
    
    lstm = LSTMForecaster(num_features=X_train.shape[2])
    lstm.train(X_train, y_train, X_test, y_test, epochs=10)
    metrics, _ = lstm.evaluate(X_test, y_test)
