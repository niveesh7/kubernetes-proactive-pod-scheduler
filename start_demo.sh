#!/usr/bin/env bash
export KUBECONFIG=~/.kube/config

echo "Launching Proactive Scheduling Scenario Demonstration on 4-Node Cloud Cluster..."
python demonstrate_proactive_scenario.py
