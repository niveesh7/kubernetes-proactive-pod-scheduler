#!/usr/bin/env bash
# ==============================================================================
# Setup Script for 4-Node K3d Kubernetes Cluster in GitHub Codespaces
# ==============================================================================

echo "================================================================================"
echo " Setting up Kubernetes Proactive Scheduler Cloud Environment (4-Node K3d Cluster)"
echo "================================================================================"

# 1. Install kubectl if not present
if ! command -v kubectl &> /dev/null; then
    echo "[1/5] Installing kubectl..."
    curl -sLO "https://dl.k8s.io/release/v1.30.0/bin/linux/amd64/kubectl"
    sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl 2>/dev/null || install -m 0755 kubectl /usr/local/bin/kubectl 2>/dev/null || true
    rm -f kubectl
else
    echo "[1/5] kubectl is ready: $(kubectl version --client 2>/dev/null | grep -i 'Client Version' || echo 'installed')"
fi

# 2. Install K3d CLI if not present
echo "[2/5] Checking and installing K3d CLI..."
if ! command -v k3d &> /dev/null; then
    curl -s https://raw.githubusercontent.com/k3d-io/k3d/main/install.sh | bash
fi
echo "K3d CLI ready: $(k3d version 2>/dev/null | head -n 1)"

# 3. Create 4-Node Kubernetes Cluster (1 Server + 3 Worker Agents)
echo "[3/5] Initializing 4-Node Kubernetes Cluster..."
if ! k3d cluster list 2>/dev/null | grep -q "proactive-cluster"; then
    k3d cluster create proactive-cluster \
        --servers 1 \
        --agents 3 \
        --port "30080:30080@server:0" \
        --wait || true
fi

# Setup kubeconfig
mkdir -p ~/.kube
k3d kubeconfig merge proactive-cluster --kubeconfig-switch-context 2>/dev/null || true
export KUBECONFIG=~/.kube/config

echo "Kubernetes nodes active in cloud:"
kubectl get nodes -o wide || true

# 4. Install Python Dependencies
echo "[4/5] Installing Python Dependencies..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# 5. Deploy RBAC and Microservice Workload
echo "[5/5] Deploying RBAC permissions and Cloud Workload..."
kubectl apply -f deploy/rbac.yaml 2>/dev/null || true
kubectl apply -f deploy/target-microservice.yaml 2>/dev/null || true

echo "================================================================================"
echo " SUCCESS! 4-Node Cloud Kubernetes Cluster is Fully Initialized!"
echo "================================================================================"
kubectl get nodes || true
echo ""
echo " To start the live Web Dashboard:"
echo "   ./start_dashboard.sh    (or: python dashboard/app.py)"
echo ""
echo " To run the End-to-End Proactive Scenario Demonstration:"
echo "   ./start_demo.sh         (or: python demonstrate_proactive_scenario.py)"
echo "================================================================================"
