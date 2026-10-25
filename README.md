# Workload-Forecasting Driven Proactive Pod Scheduling for Kubernetes Clusters using LSTM-XGBoost Ensemble and Multi-Objective Node Ranking

**Department of Computer Science and Engineering**  
**Anjalai Ammal Mahalingam Engineering College**  

### Team Members:
1. **Niveesh R** (820423104058)
2. **Shanmugapriyan K S** (820423104072)
3. **Mohamed Anwar S** (820423104050)

**Guide Name:** Mr. P Manikanda Prabhu (AP/CSE)

---

## 1. Problem Definition & Research Gap

In modern cloud computing, containerized microservices managed by Kubernetes experience highly dynamic, fluctuating traffic patterns (daily diurnal peaks and sudden burst surges). The default Kubernetes scheduling architecture relies on:
1. **Default `kube-scheduler`**: Placed pods using instantaneous static metrics and filtering/scoring without considering future demand.
2. **Horizontal Pod Autoscaler (HPA)**: Purely reactive threshold-based scaling (e.g. scales only after average CPU utilization crosses 80%).

**The Limitation:**
Because scaling and placement are reactive, there is an inherent delay (60–90 seconds) between the onset of a traffic surge and the availability of new pods/nodes to absorb it. During this lag period, applications suffer severe latency spikes, packet drops, and **Service Level Objective (SLO) violations**.

**The Research Gap:**
Existing literature (Amirullah & Saikhu 2025, Lipari et al. 2025, Toka et al. 2020, Matsiievskyi et al. 2025) either uses single forecasting models (LSTM alone, which lags during sudden non-temporal jumps) or focuses only on autoscaling replica counts without solving **multi-criteria node placement**. 

---

## 2. Proposed System Architecture

This project implements a complete, end-to-end **Proactive Kubernetes Pod Scheduling Framework**:
1. **Metrics Collection Module**: Real-time Prometheus metrics scraper and Kubernetes API telemetry client.
2. **Data Preprocessing & Feature Engineering Module**: 3D sliding window tensors for LSTM and lag/rolling/cyclic features for XGBoost.
3. **LSTM Forecasting Module**: Deep recurrent network capturing long- and short-term sequential temporal dependencies.
4. **XGBoost Forecasting Module**: Extreme Gradient Boosting regressor capturing sharp sudden traffic bursts and non-linear feature interactions.
5. **Ensemble Fusion Layer**: Adaptive inverse-error weighted fusion combining LSTM and XGBoost predictions into a unified near-future demand projection.
6. **Multi-Objective Node Ranking (MONR) Engine**: Scores all candidate worker nodes across 5 distinct objectives:
   - Predicted CPU Headroom
   - Predicted Memory Headroom
   - Cluster Load Balance (Variance Minimization)
   - Network & Inter-Node Latency
   - Resource Packing Efficiency
7. **Proactive Pod Scheduler**: A real Kubernetes scheduler and extender service (`schedulerName: proactive-scheduler`) that creates Kubernetes `Binding` objects via the Kubernetes API to bind pods onto the optimal worker node *before* the surge hits.
8. **Interactive Monitoring Dashboard**: Live web-based visualization of nodes, forecasting trajectories, ranking criteria, and comparative benchmark results.

---

## 3. Project Directory Structure

```
FINALYEAR PROJECT/
├── config/
│   ├── settings.py                 # Hyperparameters, paths, weights, and SLO targets
│   └── __init__.py
├── data/
│   ├── workload_generator_dataset.py # Realistic diurnal & burst traffic generator
│   ├── workload_train.csv          # Training trace (2160 time steps)
│   ├── workload_test.csv           # Test trace (720 time steps)
│   └── workload_full.csv           # Full 24h trace
├── preprocessing/
│   ├── preprocessor.py             # Feature engineering, scaling, and sequence windowing
│   └── __init__.py
├── forecasting/
│   ├── lstm_model.py               # Deep Learning LSTM model (Keras)
│   ├── xgboost_model.py            # XGBoost tree regressor
│   ├── ensemble.py                 # Adaptive weighted ensemble fusion
│   └── __init__.py
├── node_ranking/
│   ├── multi_objective_ranker.py   # Multi-Criteria Decision Analysis ranking engine
│   └── __init__.py
├── scheduler/
│   ├── proactive_scheduler.py      # Kubernetes custom scheduler & extender API
│   └── __init__.py
├── load_generator/
│   ├── traffic_simulator.py        # Real-time HTTP workload & surge injector
│   └── __init__.py
├── evaluation/
│   ├── compare_baselines.py        # Head-to-head benchmark against HPA & LSTM-only
│   ├── evaluation_results.png      # 4-panel publication-grade evaluation charts
│   └── metrics_comparison.csv      # Quantitative comparison metrics table
├── dashboard/
│   ├── app.py                      # FastAPI server for real-time monitoring
│   └── templates/
│       └── index.html              # High-tech responsive dashboard (Chart.js + Tailwind)
├── deploy/
│   ├── k8s-multi-node-cluster.yaml # Multi-node cluster spec (1 master + 3 workers)
│   ├── target-microservice.yaml    # Test microservice deployment with proactive-scheduler
│   ├── scheduler-extender-config.yaml # K8s scheduler extender webhook policy
│   ├── rbac.yaml                   # ClusterRole, RoleBinding, ServiceAccount
│   └── prometheus.yaml             # Prometheus deployment and scraping config
├── models/
│   ├── lstm_model.keras            # Trained LSTM model
│   ├── xgboost_model.json          # Trained XGBoost model
│   ├── ensemble_weights.json       # Dynamic ensemble weights
│   ├── scaler_features.joblib      # Fitted feature scaler
│   └── scaler_target.joblib        # Fitted target scaler
├── run_project.py                  # Master entry point launcher
├── train_models.py                 # Pipeline to retrain all models from scratch
├── requirements.txt                # Python package dependencies
├── run.bat                         # Double-click Windows launcher
└── README.md                       # Comprehensive documentation
```

---

## 4. Software & Tools Setup Guide (For Running with Real Kubernetes)

The project includes an **autonomous live cluster telemetry engine** that lets you run and demonstrate the entire system immediately. To run against a **live physical or local multi-node Kubernetes cluster**, install the following:

### Step 1: Install WSL 2 (Windows Subsystem for Linux)
Open PowerShell as Administrator and run:
```powershell
wsl --install
```
*(Restart your computer if prompted).*

### Step 2: Install Docker Desktop for Windows
1. Download Docker Desktop from: [https://www.docker.com/products/docker-desktop/](https://www.docker.com/products/docker-desktop/)  
   *(Or install via winget: `winget install Docker.DockerDesktop`)*
2. During installation, ensure **"Use WSL 2 instead of Hyper-V"** is checked.
3. Start Docker Desktop. Once running, go to **Settings (Gear Icon) -> Kubernetes -> Check "Enable Kubernetes" -> Click "Apply & Restart"**.

### Step 3: Install Kubectl (Kubernetes Command-Line Tool)
Run in PowerShell:
```powershell
winget install Kubernetes.kubectl
```
Verify with:
```powershell
kubectl version --client
kubectl get nodes
```

### Step 4 (Alternative): Minikube Multi-Node Cluster
If you prefer Minikube to create a multi-worker node cluster:
```powershell
winget install Kubernetes.minikube
minikube start --nodes 3 --driver=docker
kubectl get nodes
```
This gives you 3 real worker nodes (`minikube`, `minikube-m02`, `minikube-m03`) running on your machine.

---

## 5. How to Run the Project

### Option A: One-Click Execution
Simply double-click `run.bat` in the project folder.

### Option B: Command Line Execution
Open PowerShell or Command Prompt in the project folder:
```powershell
python run_project.py
```
This will:
1. Verify the realistic workload dataset (`data/workload_train.csv`, `data/workload_test.csv`).
2. Verify / load the trained **LSTM** and **XGBoost** models.
3. Execute the comparative benchmarking suite and generate results.
4. Launch the **Live Web Dashboard** at **`http://127.0.0.1:8080`** and open your browser automatically!

### Option C: 1-Click Free Cloud Execution (GitHub Codespaces - 15 Real K3s Nodes)
Run the entire project in the cloud with **zero local setup, zero RAM usage, and 15 real Kubernetes nodes**:
1. Push this repository to GitHub or open your Codespace.
2. In the Codespaces terminal, run:
   ```bash
   chmod +x *.sh
   ./setup_codespace.sh 15   # Launches 15 K3d nodes (1 server + 14 agents)
   ./start_dashboard.sh      # Starts web dashboard on port 8080
   ```
3. If you have made updates to the repository, pull the latest code inside Codespaces:
   ```bash
   git fetch origin
   git reset --hard origin/main
   pkill -f "dashboard/app.py" || true
   ./start_dashboard.sh
   ```
4. Open the forwarded port 8080 URL in your browser and press `Ctrl + Shift + R` (Hard Refresh) to view the live 15-node dashboard from anywhere!

---

## 6. Testing & Demonstrating Features on the Dashboard

Once the dashboard opens at `http://127.0.0.1:8080` (or your Codespaces public URL):

1. **Interactive Traffic Stream Controls (`▶ Start Alibaba Traffic Stream`)**:
   - The system starts in an **IDLE** state (0.0 req/s) so examiners and evaluators can inspect the baseline cluster before generating requests.
   - Click **"Start Alibaba Traffic Stream"** to begin replaying the real-world Alibaba Cloud 11.11 Global Shopping Festival workload trace.
   - Click **"Pause Traffic Stream"** at any time to freeze the incoming flow.

2. **Autonomous Proactive Lifecycle Storyboard (5-Phase Demonstration)**:
   - Click **"▶ Run Live 5-Phase Scenario Demonstration"** to watch the end-to-end autonomous lifecycle play out step-by-step:
     - **Phase 1: Baseline Normal State**: Steady 35 req/s baseline traffic on primary node.
     - **Phase 2: AI Lookahead Warning**: LSTM + XGBoost ensemble detects an impending +380% flash crowd spike $T-30s$ in advance.
     - **Phase 3: Multi-Objective Node Ranking (MONR)**: Evaluates all 15 nodes across 5 criteria and selects the optimal target.
     - **Phase 4: Pre-Emptive K8s Binding**: Pre-warms and binds the pod 15 seconds *before* traffic reaches the cluster.
     - **Phase 5: Surge Absorbed with Zero SLO Violations**: Flash crowd hits; 100% of requests handled smoothly under 30ms latency.

3. **Multi-Objective Node Ranking (MONR) Decision Engine**:
   - Displays the exact mathematical formula:
     $$S_i = \sum_{k=1}^5 w_k \cdot C_{i,k}$$
   - Shows the 5 weighted sub-scores for every node:
     - CPU Headroom ($w_1 = 30\%$)
     - RAM Headroom ($w_2 = 25\%$)
     - Cluster Balance ($w_3 = 20\%$)
     - Network Latency ($w_4 = 15\%$)
     - Packing Efficiency ($w_5 = 10\%$)
   - Displays a glowing **`★ TARGET: Optimal node chosen by proactive-scheduler`** banner on Rank #1.

4. **Trigger Flash Surge (+120 req/s)**:
   - Manually inject an instantaneous burst to test the ensemble forecaster's XGBoost layer.

5. **Schedule New Pod**:
   - Manually trigger a proactive scheduling cycle to bind a pod via the Kubernetes API.

6. **View Benchmark**:
   - Review the head-to-head performance against Default Reactive Kubernetes HPA.

---

## 7. Comparative Benchmark Results

| Scheduling Strategy | Avg Latency (ms) | P95 Latency (ms) | P99 Latency (ms) | SLO Violations (%) | Cluster Load Variance ($\sigma^2$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Reactive HPA (Default K8s)** | 28.78 ms | 33.68 ms | 35.83 ms | High during spikes | 0.0204 (Imbalanced) |
| **LSTM-Only Proactive** | 27.53 ms | 31.98 ms | 33.41 ms | Low | 0.0055 |
| **Proposed System (Ensemble + MONR)** | **25.06 ms** | **28.50 ms** | **30.18 ms** | **0.0% (Zero Violations)** | **0.0000 (Perfect Balance)** |

High-resolution charts are generated in: `evaluation/evaluation_results.png`.

---

## 8. Deploying to Real Kubernetes Cluster

To deploy the target microservice and RBAC to your live Kubernetes cluster:
```powershell
# 1. Apply RBAC permissions
kubectl apply -f deploy/rbac.yaml

# 2. Deploy Prometheus
kubectl apply -f deploy/prometheus.yaml

# 3. Deploy target microservice (scheduled via proactive-scheduler)
kubectl apply -f deploy/target-microservice.yaml

# 4. Inspect pods and nodes
kubectl get pods -o wide
```
You will observe the pods automatically bound to worker nodes according to the Multi-Objective Node Ranking scores!
