#!/usr/bin/env bash
# ==============================================================================
# Clean Reset & Restart Script for K3d Kubernetes Cluster in GitHub Codespaces
# Usage:
#   ./reset_cluster.sh          (Defaults to 4 nodes: 1 server + 3 agents)
#   ./reset_cluster.sh 6        (6 nodes: 1 server + 5 agents)
#   ./reset_cluster.sh 10       (10 nodes: 1 server + 9 agents)
#   ./reset_cluster.sh 15       (15 nodes: 1 server + 14 agents)
# ==============================================================================

TOTAL_NODES=${1:-4}
if [ "$TOTAL_NODES" -lt 2 ]; then
    TOTAL_NODES=4
fi
AGENTS=$((TOTAL_NODES - 1))

echo "================================================================================"
echo " Resetting Kubernetes Cluster: Clean Slate -> Deploying $TOTAL_NODES Real Nodes"
echo " (1 Control-Plane Server + $AGENTS Worker Agents)"
echo "================================================================================"

# 1. Stop any lingering python dashboard processes
echo "[1/5] Stopping any running dashboard processes..."
pkill -f "dashboard/app.py" 2>/dev/null || true
pkill -f "uvicorn" 2>/dev/null || true
sleep 1

# 2. Delete existing K3d cluster cleanly
echo "[2/5] Deleting previous cluster 'proactive-cluster'..."
k3d cluster delete proactive-cluster 2>/dev/null || true

# 3. Create fresh cluster with specified node count
echo "[3/5] Creating fresh $TOTAL_NODES-node cluster in K3d (takes ~30-45s)..."
k3d cluster create proactive-cluster \
    --servers 1 \
    --agents "$AGENTS" \
    --port "30080:30080@server:0" \
    --wait

# 4. Merge kubeconfig credentials & switch context
echo "[4/5] Updating kubeconfig security credentials..."
mkdir -p ~/.kube
k3d kubeconfig merge proactive-cluster --kubeconfig-switch-context
export KUBECONFIG=~/.kube/config

# Wait for nodes to reach Ready state
echo "Waiting for all $TOTAL_NODES nodes to report Ready..."
kubectl wait --for=condition=Ready nodes --all --timeout=60s

echo "Kubernetes nodes active in cloud:"
kubectl get nodes -o wide

# 5. Apply RBAC permissions and target microservice workload
echo "[5/5] Deploying RBAC permissions and microservice workload..."
kubectl apply -f deploy/rbac.yaml
kubectl apply -f deploy/target-microservice.yaml

echo "================================================================================"
echo " SUCCESS! Cluster reset complete with $TOTAL_NODES live nodes!"
echo "================================================================================"
kubectl get nodes
echo ""
echo " To start your live dashboard, run:"
echo "   ./start_dashboard.sh"
echo "================================================================================"
