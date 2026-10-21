#!/usr/bin/env bash
# ==============================================================================
# Automated Cloud Setup Script for GitHub Codespaces (4-Node K3d Cluster)
# Project: Workload-Forecasting Driven Proactive Pod Scheduling
# ==============================================================================
set -e

echo "================================================================================"
echo " Setting up Kubernetes Proactive Scheduler Cloud Environment (4-Node K3d Cluster)"
echo "================================================================================"

# 1. Wait for Docker daemon to be active
echo "[1/5] Verifying Docker Engine..."
for i in {1..15}; do
    if docker info > /dev/null 2>&1; then
        echo "Docker daemon is ready."
        break
    fi
    echo "Waiting for Docker daemon to initialize ($i/15)..."
    sleep 2
done

# 2. Install K3d (Lightweight Multi-Node Kubernetes runner)
echo "[2/5] Checking and installing K3d CLI..."
if ! command -v k3d &> /dev/null; then
    curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
fi
echo "K3d CLI ready: $(k3d version | head -n 1)"

# 3. Create 4-Node Kubernetes Cluster (1 Server + 3 Worker Agents)
echo "[3/5] Initializing 4-Node Kubernetes Cluster..."
if ! k3d cluster list 2>/dev/null | grep -q "proactive-cluster"; then
    k3d cluster create proactive-cluster \
        --servers 1 \
        --agents 3 \
        --port "30080:30080@server:0" \
        --wait
else
    echo "K3d cluster 'proactive-cluster' already running."
fi

# Ensure kubeconfig is active for kubectl and Python kubernetes client
mkdir -p ~/.kube
k3d kubeconfig get proactive-cluster > ~/.kube/config
chmod 600 ~/.kube/config
export KUBECONFIG=~/.kube/config

echo "Kubernetes nodes active in cloud:"
kubectl get nodes -o wide

# 4. Install Python Dependencies
echo "[4/5] Installing Python Dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Deploy RBAC and Initial Microservice Workload
echo "[5/5] Deploying RBAC permissions and Cloud Workload..."
kubectl apply -f deploy/rbac.yaml
kubectl apply -f deploy/target-microservice.yaml

echo "Waiting for pods in 'default' namespace to initialize..."
kubectl get pods -n default -o wide || true

echo "================================================================================"
echo " SUCCESS! 4-Node Cloud Kubernetes Cluster is Fully Initialized!"
echo "================================================================================"
echo " Nodes:"
kubectl get nodes
echo ""
echo " To start the live Web Dashboard:"
echo "   ./start_dashboard.sh    (or: python dashboard/app.py)"
echo ""
echo " To run the End-to-End Proactive Scenario Demonstration:"
echo "   ./start_demo.sh         (or: python demonstrate_proactive_scenario.py)"
echo "================================================================================"
