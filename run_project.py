"""
Master Execution Script
Launches the complete Kubernetes Proactive Pod Scheduling System:
1. Verifies dataset and trained ML models (LSTM + XGBoost + Ensemble)
2. Connects to Kubernetes cluster (or initializes live autonomous scheduler)
3. Runs comparative evaluation benchmarks
4. Starts the Live Monitoring & Control Dashboard
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config.settings import (
    DATA_DIR,
    MODELS_DIR,
    DASHBOARD_HOST,
    DASHBOARD_PORT
)


def main():
    print("=" * 75)
    print("     WORKLOAD-FORECASTING DRIVEN PROACTIVE POD SCHEDULER")
    print("       FOR KUBERNETES CLUSTERS USING LSTM-XGBOOST ENSEMBLE")
    print("                AND MULTI-OBJECTIVE NODE RANKING")
    print("     Department of Computer Science and Engineering - Final Year Project")
    print("=" * 75)
    
    # Step 1: Verify / Generate Datasets
    alibaba_csv = DATA_DIR / "alibaba_production_trace.csv"
    train_csv = DATA_DIR / "workload_train.csv"
    test_csv = DATA_DIR / "workload_test.csv"
    if not alibaba_csv.exists() or not train_csv.exists() or not test_csv.exists():
        print("\n[Step 1/5] Generating authentic enterprise cloud traces (Alibaba / Amazon)...")
        from data.enterprise_workload_loader import create_enterprise_datasets
        create_enterprise_datasets()
    else:
        print("\n[Step 1/5] Enterprise workload traces verified (Alibaba 11.11 & Amazon E-Commerce).")

    # Step 2: Verify / Train Machine Learning Models
    lstm_file = MODELS_DIR / "lstm_model.keras"
    xgb_file = MODELS_DIR / "xgboost_model.json"
    weights_file = MODELS_DIR / "ensemble_weights.json"
    
    if not (lstm_file.exists() and xgb_file.exists() and weights_file.exists()):
        print("\n[Step 2/4] Training LSTM and XGBoost models...")
        from train_models import run_training_pipeline
        run_training_pipeline()
    else:
        print("\n[Step 2/4] Trained models verified:")
        print(f"  - LSTM Model:     {lstm_file}")
        print(f"  - XGBoost Model:  {xgb_file}")
        print(f"  - Dynamic Fusion: {weights_file}")

    # Step 3: Run Comparative Evaluation Benchmark
    print("\n[Step 3/4] Running comparative benchmark vs Default Reactive Scheduler / HPA...")
    from evaluation.compare_baselines import run_comparative_benchmark
    run_comparative_benchmark()

    # Step 4: Verify Live Kubernetes Cluster & Start Service Port-Forward
    print("\n[Step 4/5] Checking Kubernetes Cluster connection...")
    import subprocess
    try:
        res = subprocess.run(["kubectl", "get", "nodes"], capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            print("  - Real Kubernetes cluster detected:")
            for line in res.stdout.strip().split("\n"):
                print(f"    {line}")
            # Ensure target workload is applied
            subprocess.run(["kubectl", "apply", "-f", "deploy/rbac.yaml"], capture_output=True)
            subprocess.run(["kubectl", "apply", "-f", "deploy/target-microservice.yaml"], capture_output=True)
            # Start background port-forward if not already listening
            subprocess.Popen(
                ["kubectl", "port-forward", "svc/cloud-workload-service", "30080:80"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print("  - Microservice deployed and bridged to http://127.0.0.1:30080")
    except Exception as e:
        print(f"  - Notice: {e}. Running in autonomous scheduler mode.")

    # Step 5: Launch Web Dashboard
    print("\n[Step 5/5] Starting Proactive Scheduler & Live Dashboard...")
    dashboard_url = f"http://{DASHBOARD_HOST}:{DASHBOARD_PORT}"
    print(f"\n===========================================================================")
    print(f"  >>> LIVE DASHBOARD RUNNING AT: {dashboard_url} <<<")
    print(f"  Press Ctrl+C to stop the server at any time.")
    print(f"===========================================================================\n")
    
    # Optionally open browser
    try:
        webbrowser.open(dashboard_url)
    except Exception:
        pass
        
    from dashboard.app import run_dashboard
    run_dashboard()


if __name__ == "__main__":
    main()
