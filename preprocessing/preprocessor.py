"""
Data Preprocessing and Feature Engineering Module
Prepares time-series data for both LSTM (3D sliding windows) and
XGBoost (engineered tabular features with lags, rolling statistics, rate-of-change, and cyclic time).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MODELS_DIR,
    WINDOW_SIZE,
    FEATURE_COLUMNS,
    TARGET_COLUMN
)


class WorkloadPreprocessor:
    def __init__(self, window_size=WINDOW_SIZE):
        self.window_size = window_size
        self.scaler_features = MinMaxScaler(feature_range=(0, 1))
        self.scaler_target = MinMaxScaler(feature_range=(0, 1))
        self.feature_columns = FEATURE_COLUMNS
        self.target_column = TARGET_COLUMN
        self.is_fitted = False
        self.xgb_feature_names = []

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values, timestamps, and duplicates."""
        df = df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            
        # Impute forward/backward for missing values
        df = df.ffill().bfill()
        return df

    def extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract cyclical hour and minute sine/cosine features."""
        df = df.copy()
        if "timestamp" in df.columns:
            hours = df["timestamp"].dt.hour
            minutes = df["timestamp"].dt.minute
            df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
            df["min_sin"] = np.sin(2 * np.pi * minutes / 60.0)
            df["min_cos"] = np.cos(2 * np.pi * minutes / 60.0)
        else:
            df["hour_sin"] = 0.0
            df["hour_cos"] = 0.0
            df["min_sin"] = 0.0
            df["min_cos"] = 0.0
        return df

    def engineer_xgb_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create tabular features: lags, rolling stats, deltas for XGBoost."""
        df = self.clean_data(df)
        df = self.extract_time_features(df)
        
        target = self.target_column
        
        # 1. Lag features for target (CPU usage)
        for lag in [1, 2, 3, 5, 10]:
            df[f"{target}_lag_{lag}"] = df[target].shift(lag)
            
        # 2. Lag features for request rate
        for lag in [1, 2, 3, 5]:
            df[f"request_rate_lag_{lag}"] = df["request_rate"].shift(lag)
            
        # 3. Rolling statistics
        df[f"{target}_roll_mean_5"] = df[target].shift(1).rolling(window=5).mean()
        df[f"{target}_roll_std_5"] = df[target].shift(1).rolling(window=5).std().fillna(0)
        df[f"{target}_roll_mean_15"] = df[target].shift(1).rolling(window=15).mean()
        
        # 4. Rate of change (First difference)
        df[f"{target}_delta_1"] = df[target].shift(1) - df[target].shift(2)
        df["req_delta_1"] = df["request_rate"].shift(1) - df["request_rate"].shift(2)
        
        # Drop rows with NaN due to shifting
        df = df.dropna().reset_index(drop=True)
        return df

    def fit(self, df: pd.DataFrame):
        """Fit scalers on training data."""
        df_clean = self.clean_data(df)
        
        features_data = df_clean[self.feature_columns].values
        target_data = df_clean[[self.target_column]].values
        
        self.scaler_features.fit(features_data)
        self.scaler_target.fit(target_data)
        self.is_fitted = True
        
        # Save scalers for live inference
        joblib.dump(self.scaler_features, MODELS_DIR / "scaler_features.joblib")
        joblib.dump(self.scaler_target, MODELS_DIR / "scaler_target.joblib")
        print("[Preprocessor] Scalers fitted and saved successfully.")

    def load_scalers(self):
        """Load fitted scalers from disk."""
        features_path = MODELS_DIR / "scaler_features.joblib"
        target_path = MODELS_DIR / "scaler_target.joblib"
        if features_path.exists() and target_path.exists():
            self.scaler_features = joblib.load(features_path)
            self.scaler_target = joblib.load(target_path)
            self.is_fitted = True
            return True
        return False

    def prepare_lstm_data(self, df: pd.DataFrame):
        """Transform dataframe into 3D sliding window tensors (samples, window_size, features)."""
        if not self.is_fitted:
            loaded = self.load_scalers()
            if not loaded:
                raise ValueError("Preprocessor must be fitted before preparing LSTM data.")
            
        df_clean = self.clean_data(df)
        scaled_features = self.scaler_features.transform(df_clean[self.feature_columns].values)
        scaled_target = self.scaler_target.transform(df_clean[[self.target_column]].values)
        
        X, y = [], []
        if len(scaled_features) == self.window_size:
            X.append(scaled_features)
            y.append(scaled_target[-1, 0])
        else:
            for i in range(self.window_size, len(scaled_features)):
                X.append(scaled_features[i - self.window_size:i])
                y.append(scaled_target[i, 0])
            
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def prepare_xgb_data(self, df: pd.DataFrame):
        """Prepare tabular feature matrix X and target y for XGBoost."""
        df_feats = self.engineer_xgb_features(df)
        
        # Target is current CPU usage
        y = df_feats[self.target_column].values
        
        # Exclude raw target, timestamp from feature matrix
        exclude_cols = [self.target_column, "timestamp"]
        feature_cols = [c for c in df_feats.columns if c not in exclude_cols]
        self.xgb_feature_names = feature_cols
        
        X = df_feats[feature_cols].values
        return X, y, feature_cols

    def inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
        """Convert scaled target predictions back to raw CPU cores."""
        if not self.is_fitted:
            self.load_scalers()
        reshaped = np.array(scaled_values).reshape(-1, 1)
        return self.scaler_target.inverse_transform(reshaped).flatten()


if __name__ == "__main__":
    from config.settings import DATA_DIR
    df = pd.read_csv(DATA_DIR / "workload_train.csv")
    prep = WorkloadPreprocessor()
    prep.fit(df)
    X_lstm, y_lstm = prep.prepare_lstm_data(df)
    X_xgb, y_xgb, cols = prep.prepare_xgb_data(df)
    print(f"LSTM input shape: {X_lstm.shape}, y shape: {y_lstm.shape}")
    print(f"XGB input shape:  {X_xgb.shape}, y shape: {y_xgb.shape}")
    print(f"XGB engineered {len(cols)} features: {cols[:5]}...")
