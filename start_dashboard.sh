#!/usr/bin/env bash
# Start the Proactive Pod Scheduler Live Web Dashboard in GitHub Codespaces
export KUBECONFIG=~/.kube/config
export DASHBOARD_HOST=0.0.0.0
export DASHBOARD_PORT=8080

echo "Starting Proactive Scheduler Web Dashboard on port 8080..."
python dashboard/app.py
