"""
LSTM-XGBoost Ensemble Fusion Layer
Combines predictions from both LSTM (temporal trend specialist) and
XGBoost (burst and non-linear feature specialist) through an adaptive
inverse-error weighted fusion layer to generate robust workload predictions.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
import json
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    MODELS_DIR,
    ENSEMBLE_WEIGHT_LSTM,
    ENSEMBLE_WEIGHT_XGB,
    WINDOW_SIZE
)
from forecasting.lstm_model import LSTMForecaster
from forecasting.xgboost_model import XGBoostForecaster
from preprocessing.preprocessor import WorkloadPreprocessor


class EnsembleForecaster:
    def __init__(self, weight_lstm=ENSEMBLE_WEIGHT_LSTM, weight_xgb=ENSEMBLE_WEIGHT_XGB):
        self.weight_lstm = weight_lstm
        self.weight_xgb = weight_xgb
        self.lstm = LSTMForecaster()
        self.xgb = XGBoostForecaster()
        self.preprocessor = WorkloadPreprocessor()
        self.weights_file = MODELS_DIR / "ensemble_weights.json"

    def optimize_weights_by_inverse_error(self, y_val, preds_lstm, preds_xgb):
        """
        Dynamically calculate optimal Bayesian/inverse-RMSE weights:
        w_i = (1 / RMSE_i) / sum(1 / RMSE_k)
        """
        rmse_lstm = root_mean_squared_error(y_val, preds_lstm)
        rmse_xgb = root_mean_squared_error(y_val, preds_xgb)
        
        inv_lstm = 1.0 / max(rmse_lstm, 1e-6)
        inv_xgb = 1.0 / max(rmse_xgb, 1e-6)
        total_inv = inv_lstm + inv_xgb
        
        self.weight_lstm = float(inv_lstm / total_inv)
        self.weight_xgb = float(inv_xgb / total_inv)
        
        print(f"[Ensemble] Dynamic Weights Calculated:")
        print(f"  - LSTM  (RMSE={rmse_lstm:.4f}) Weight: {self.weight_lstm:.3f}")
        print(f"  - XGB   (RMSE={rmse_xgb:.4f})  Weight: {self.weight_xgb:.3f}")
        
        with open(self.weights_file, "w") as f:
            json.dump({
                "weight_lstm": self.weight_lstm,
                "weight_xgb": self.weight_xgb,
                "rmse_lstm": float(rmse_lstm),
                "rmse_xgb": float(rmse_xgb)
            }, f, indent=2)

    def load_models_and_weights(self):
        """Load trained sub-models and fusion weights."""
        self.preprocessor.load_scalers()
        self.lstm.load_model()
        self.xgb.load_model()
        if self.weights_file.exists():
            with open(self.weights_file, "r") as f:
                data = json.load(f)
                self.weight_lstm = data.get("weight_lstm", 0.5)
                self.weight_xgb = data.get("weight_xgb", 0.5)
        print(f"[Ensemble] Ready with weights: LSTM={self.weight_lstm:.2f}, XGB={self.weight_xgb:.2f}")

    def fuse_predictions(self, preds_lstm, preds_xgb):
        """Weighted fusion of LSTM and XGBoost predictions."""
        # Align lengths if sequence shift caused slight difference
        min_len = min(len(preds_lstm), len(preds_xgb))
        p_lstm = preds_lstm[-min_len:]
        p_xgb = preds_xgb[-min_len:]
        
        fused = (self.weight_lstm * p_lstm) + (self.weight_xgb * p_xgb)
        return fused

    def predict_from_dataframe(self, df: pd.DataFrame):
        """
        End-to-end inference from raw dataframe of recent metrics.
        Returns:
            fused_prediction_cores: Unscaled CPU prediction
            dict_details: Individual model outputs and confidence
        """
        # 1. Prepare LSTM tensor
        X_lstm, _ = self.preprocessor.prepare_lstm_data(df)
        preds_lstm_scaled = self.lstm.predict(X_lstm[-1:])  # Latest window
        
        # Invert scale for LSTM
        pred_lstm_cores = float(self.preprocessor.inverse_transform_target(preds_lstm_scaled)[0])
        
        # 2. Prepare XGBoost feature vector
        X_xgb, _, _ = self.preprocessor.prepare_xgb_data(df)
        preds_xgb = self.xgb.predict(X_xgb[-1:])  # Latest feature vector
        pred_xgb_cores = float(preds_xgb[0])
        
        # 3. Ensemble Fusion
        fused_cores = (self.weight_lstm * pred_lstm_cores) + (self.weight_xgb * pred_xgb_cores)
        fused_cores = max(0.1, float(fused_cores))
        
        details = {
            "lstm_pred_cores": round(pred_lstm_cores, 4),
            "xgb_pred_cores": round(pred_xgb_cores, 4),
            "ensemble_pred_cores": round(fused_cores, 4),
            "weight_lstm": round(self.weight_lstm, 3),
            "weight_xgb": round(self.weight_xgb, 3)
        }
        return fused_cores, details

    def evaluate_ensemble(self, df_test: pd.DataFrame):
        """Evaluate single models vs ensemble on held-out test dataset."""
        print("\n================ ENSEMBLE EVALUATION BENCHMARK ================")
        
        # LSTM test evaluation
        X_lstm, y_lstm_scaled = self.preprocessor.prepare_lstm_data(df_test)
        preds_lstm_scaled = self.lstm.predict(X_lstm)
        preds_lstm_cores = self.preprocessor.inverse_transform_target(preds_lstm_scaled)
        y_true_lstm_cores = self.preprocessor.inverse_transform_target(y_lstm_scaled)
        
        # XGBoost test evaluation
        X_xgb, y_xgb_cores, _ = self.preprocessor.prepare_xgb_data(df_test)
        preds_xgb_cores = self.xgb.predict(X_xgb)
        
        # Align lengths
        min_len = min(len(preds_lstm_cores), len(preds_xgb_cores))
        y_true = y_xgb_cores[-min_len:]
        p_lstm = preds_lstm_cores[-min_len:]
        p_xgb = preds_xgb_cores[-min_len:]
        
        p_ensemble = (self.weight_lstm * p_lstm) + (self.weight_xgb * p_xgb)
        
        # Compute metrics
        results = {}
        for name, preds in [("LSTM Only", p_lstm), ("XGBoost Only", p_xgb), ("LSTM-XGB Ensemble", p_ensemble)]:
            rmse = root_mean_squared_error(y_true, preds)
            mae = mean_absolute_error(y_true, preds)
            r2 = r2_score(y_true, preds)
            results[name] = {"RMSE": float(rmse), "MAE": float(mae), "R2": float(r2)}
            print(f"[{name.ljust(18)}] RMSE: {rmse:.4f} cores | MAE: {mae:.4f} cores | R2: {r2:.4f}")
            
        print("===============================================================\n")
        return results, y_true, p_lstm, p_xgb, p_ensemble


if __name__ == "__main__":
    from config.settings import DATA_DIR
    df_train = pd.read_csv(DATA_DIR / "workload_train.csv")
    df_test = pd.read_csv(DATA_DIR / "workload_test.csv")
    
    ensemble = EnsembleForecaster()
    ensemble.load_models_and_weights()
    results, _, _, _, _ = ensemble.evaluate_ensemble(df_test)
