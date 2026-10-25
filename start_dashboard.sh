#!/usr/bin/env bash
export KUBECONFIG="$HOME/.kube/config"
export DASHBOARD_HOST=0.0.0.0
export DASHBOARD_PORT=8080

# Auto-sync k3d cluster kubeconfig if cluster exists
if command -v k3d &>/dev/null && k3d cluster list 2>/dev/null | grep -q "proactive-cluster"; then
    mkdir -p "$HOME/.kube"
    k3d kubeconfig merge proactive-cluster --kubeconfig-switch-context 2>/dev/null || true
fi

echo "================================================================================"
echo " Starting Proactive Scheduler Web Dashboard on http://0.0.0.0:8080 ..."
if command -v kubectl &>/dev/null && kubectl get nodes &>/dev/null; then
    NODE_COUNT=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
    echo " [K8s Connected] Found $NODE_COUNT live Kubernetes nodes!"
else
    echo " [K8s Notice] Live Kubernetes cluster initializing or in autonomous telemetry mode."
fi
echo "================================================================================"
python dashboard/app.py

