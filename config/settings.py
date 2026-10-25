"""
Configuration and Hyperparameters
Project: Workload-Forecasting Driven Proactive Pod Scheduling for Kubernetes Clusters
         using LSTM-XGBoost Ensemble and Multi-Objective Node Ranking
Department of Computer Science and Engineering, Anjalai Ammal Mahalingam Engineering College
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "evaluation"
DASHBOARD_DIR = BASE_DIR / "dashboard"

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
EVALUATION_DIR.mkdir(exist_ok=True)

# Kubernetes Configuration
KUBECONFIG_PATH = os.path.expanduser(os.getenv("KUBECONFIG", str(Path.home() / ".kube" / "config")))
SCHEDULER_NAME = "proactive-scheduler"
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
METRICS_SCRAPE_INTERVAL_SECONDS = 5

# Forecasting Parameters
WINDOW_SIZE = 20          # Number of historical time-steps for LSTM sequence
FORECAST_HORIZON = 5      # Steps into the future to predict (e.g. 5 steps = 25s ahead)
FEATURE_COLUMNS = ["request_rate", "cpu_usage_cores", "memory_usage_mb", "latency_ms"]
TARGET_COLUMN = "cpu_usage_cores"

# Model Hyperparameters
LSTM_EPOCHS = 25
LSTM_BATCH_SIZE = 32
LSTM_UNITS = [64, 32]
LSTM_DROPOUT = 0.2

XGB_N_ESTIMATORS = 120
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.05

# Ensemble Fusion Weights (Initial default; dynamically adapted at runtime)
ENSEMBLE_WEIGHT_LSTM = 0.50
ENSEMBLE_WEIGHT_XGB = 0.50

# Multi-Objective Node Ranking Weights
# S_i = w_cpu * Headroom_cpu + w_mem * Headroom_mem + w_balance * Balance + w_latency * Latency + w_efficiency * Efficiency
# Must sum to 1.0
WEIGHT_CPU_HEADROOM = 0.30
WEIGHT_MEM_HEADROOM = 0.25
WEIGHT_LOAD_BALANCE = 0.20
WEIGHT_LATENCY = 0.15
WEIGHT_EFFICIENCY = 0.10

# SLO Requirements
SLO_MAX_LATENCY_MS = 200.0   # SLA: 95% requests < 200ms
SLO_MAX_CPU_UTIL_PCT = 80.0  # Safe threshold before performance degrades
SLO_MAX_MEM_UTIL_PCT = 85.0

# Service Ports
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
SCHEDULER_HOST = os.getenv("SCHEDULER_HOST", "0.0.0.0")
SCHEDULER_PORT = int(os.getenv("SCHEDULER_PORT", "8000"))
