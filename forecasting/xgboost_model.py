"""
XGBoost Forecasting Module
Extreme Gradient Boosting (XGBoost) tree regressor trained on engineered tabular
features (lags, rolling stats, delta changes, time-of-day) to capture non-linear,
non-sequential patterns and sudden burst signals that complement the LSTM model.
"""

import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MODELS_DIR,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE
)


class XGBoostForecaster:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.model_path = MODELS_DIR / "xgboost_model.json"
        self.meta_path = MODELS_DIR / "xgboost_meta.json"

    def build_model(self):
        """Instantiate XGBoost Regressor."""
        self.model = xgb.XGBRegressor(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None):
        """Train XGBoost model with early stopping on validation set."""
        if self.model is None:
            self.build_model()
            
        if feature_names:
            self.feature_names = feature_names

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        
        print(f"[XGBoost] Training tree ensemble on {len(X_train)} samples with {X_train.shape[1]} features...")
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.save_model()
        return self.model

    def predict(self, X):
        """Generate forecasts using trained XGBoost trees."""
        if self.model is None:
            self.load_model()
        preds = self.model.predict(X)
        return np.array(preds).flatten()

    def evaluate(self, X_test, y_test):
        """Evaluate forecasting accuracy."""
        preds = self.predict(X_test)
        rmse = root_mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        metrics = {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
        print(f"[XGBoost Evaluation] RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        return metrics, preds

    def get_feature_importance(self):
        """Return relative importance of engineered features."""
        if self.model is None:
            self.load_model()
        importances = self.model.feature_importances_
        if self.feature_names and len(self.feature_names) == len(importances):
            return dict(zip(self.feature_names, [float(x) for x in importances]))
        return {f"f{i}": float(v) for i, v in enumerate(importances)}

    def save_model(self):
        """Save model to JSON format."""
        self.model.save_model(str(self.model_path))
        with open(self.meta_path, "w") as f:
            json.dump({"feature_names": self.feature_names}, f)
        print(f"[XGBoost] Model saved to {self.model_path}")

    def load_model(self):
        """Load trained XGBoost model from disk."""
        if self.model_path.exists():
            self.model = xgb.XGBRegressor()
            self.model.load_model(str(self.model_path))
            if self.meta_path.exists():
                with open(self.meta_path, "r") as f:
                    meta = json.load(f)
                    self.feature_names = meta.get("feature_names", [])
            print(f"[XGBoost] Loaded model successfully from {self.model_path}")
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
    
    X_train, y_train, feat_cols = prep.prepare_xgb_data(df_train)
    X_test, y_test, _ = prep.prepare_xgb_data(df_test)
    
    xgb_forecaster = XGBoostForecaster()
    xgb_forecaster.train(X_train, y_train, X_test, y_test, feature_names=feat_cols)
    metrics, _ = xgb_forecaster.evaluate(X_test, y_test)
    
    importances = xgb_forecaster.get_feature_importance()
    top_feats = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"Top 5 most influential features in XGBoost: {top_feats}")
