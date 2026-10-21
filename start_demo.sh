#!/usr/bin/env bash
# Run the 5-Phase Autonomous Proactive Scheduling Scenario in GitHub Codespaces
export KUBECONFIG=~/.kube/config

echo "Launching Proactive Scheduling Scenario Demonstration on 4-Node Cloud Cluster..."
python demonstrate_proactive_scenario.py
