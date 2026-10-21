#!/usr/bin/env bash
export KUBECONFIG=~/.kube/config
export DASHBOARD_HOST=0.0.0.0
export DASHBOARD_PORT=8080

echo "Starting Proactive Scheduler Web Dashboard on http://0.0.0.0:8080 ..."
python dashboard/app.py
