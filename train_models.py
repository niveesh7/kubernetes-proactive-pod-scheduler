"""
Master Model Training Script
Trains LSTM, trains XGBoost, computes optimal dynamic ensemble weights,
evaluates on test dataset, and persists all artifacts.
"""

import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from config.settings import DATA_DIR, MODELS_DIR
from preprocessing.preprocessor import WorkloadPreprocessor
from forecasting.lstm_model import LSTMForecaster
from forecasting.xgboost_model import XGBoostForecaster
from forecasting.ensemble import EnsembleForecaster


def run_training_pipeline():
    print("=" * 65)
    print("   TRAINING LSTM-XGBOOST WORKLOAD FORECASTING ENSEMBLE")
    print("=" * 65)
    
    train_csv = DATA_DIR / "workload_train.csv"
    test_csv = DATA_DIR / "workload_test.csv"
    
    if not train_csv.exists() or not test_csv.exists():
        print("[Pipeline] Workload data not found. Generating now...")
        from data.workload_generator_dataset import create_train_test_datasets
        create_train_test_datasets()
        
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    # 1. Fit Preprocessor
    print("\n--- [Step 1/4] Preprocessing & Feature Engineering ---")
    preprocessor = WorkloadPreprocessor()
    preprocessor.fit(df_train)
    
    # Prepare LSTM tensors
    X_train_lstm, y_train_lstm = preprocessor.prepare_lstm_data(df_train)
    X_test_lstm, y_test_lstm = preprocessor.prepare_lstm_data(df_test)
    
    # Prepare XGBoost tabular matrices
    X_train_xgb, y_train_xgb, feat_names = preprocessor.prepare_xgb_data(df_train)
    X_test_xgb, y_test_xgb, _ = preprocessor.prepare_xgb_data(df_test)
    
    # 2. Train LSTM
    print("\n--- [Step 2/4] Training LSTM Model (Temporal Forecaster) ---")
    lstm_forecaster = LSTMForecaster(num_features=X_train_lstm.shape[2])
    lstm_forecaster.train(
        X_train_lstm, y_train_lstm,
        X_test_lstm, y_test_lstm,
        epochs=15
    )
    
    # 3. Train XGBoost
    print("\n--- [Step 3/4] Training XGBoost Model (Surge/Non-linear Forecaster) ---")
    xgb_forecaster = XGBoostForecaster()
    xgb_forecaster.train(
        X_train_xgb, y_train_xgb,
        X_test_xgb, y_test_xgb,
        feature_names=feat_names
    )
    
    # 4. Ensemble Fusion & Optimal Weighting
    print("\n--- [Step 4/4] Optimizing Ensemble Fusion Weights ---")
    preds_lstm_scaled = lstm_forecaster.predict(X_test_lstm)
    preds_lstm_cores = preprocessor.inverse_transform_target(preds_lstm_scaled)
    preds_xgb_cores = xgb_forecaster.predict(X_test_xgb)
    
    min_len = min(len(preds_lstm_cores), len(preds_xgb_cores))
    y_test_cores = y_test_xgb[-min_len:]
    p_lstm = preds_lstm_cores[-min_len:]
    p_xgb = preds_xgb_cores[-min_len:]
    
    ensemble = EnsembleForecaster()
    ensemble.preprocessor = preprocessor
    ensemble.optimize_weights_by_inverse_error(y_test_cores, p_lstm, p_xgb)
    
    # Run full benchmark
    results, _, _, _, _ = ensemble.evaluate_ensemble(df_test)
    
    print("\n[Pipeline] Training completed successfully!")
    print(f"[Pipeline] Models and artifacts saved in: {MODELS_DIR}")
    return results


if __name__ == "__main__":
    run_training_pipeline()
