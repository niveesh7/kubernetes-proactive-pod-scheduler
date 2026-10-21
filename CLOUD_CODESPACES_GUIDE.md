# 1-Click Free Cloud Deployment Guide (GitHub Codespaces + 4-Node Kubernetes)

This guide shows how to run your final year project on a **100% Free Cloud Environment** (GitHub Codespaces) with a **4-Node Real Kubernetes Cluster** (`1 server + 3 worker nodes`).

- **Cost:** ₹0.00 / $0.00 (100% Free)
- **Credit Card Required:** **NO** (Zero credit card needed)
- **Free Quota:** GitHub gives 60 hours/month of free Linux cloud VM (2 vCPU, 8 GB RAM) to every GitHub account.
- **Access:** Accessible from **any browser, college computer lab, or mobile phone** with a public HTTPS link!

---

## Step 1: Push Your Project to GitHub

If you haven't already pushed your project folder to GitHub:

1. Open your browser and go to [GitHub](https://github.com).
2. Log in (or create a free account if you don't have one).
3. Click the **"+"** icon in the top-right corner and select **"New repository"**.
4. Name your repository (e.g., `kubernetes-proactive-pod-scheduler`).
5. Choose **Public** or **Private** (either works!). Do **not** check "Add a README file" (we already have one).
6. Click **"Create repository"**.
7. Now on your laptop, open Command Prompt or PowerShell in this project folder (`FINALYEAR PROJECT`) and run:
   ```bash
   git init
   git add .
   git commit -m "feat: complete proactive pod scheduler with 4-node cloud support"
   git branch -M main
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY_NAME>.git
   git push -u origin main
   ```
   *(Replace `<YOUR_GITHUB_USERNAME>` and `<YOUR_REPOSITORY_NAME>` with your actual GitHub username and repository name).*

---

## Step 2: Launch GitHub Codespaces in 1 Click

1. Go to your repository page on GitHub.
2. Click the green **"<> Code"** button.
3. Click on the **"Codespaces"** tab.
4. Click **"Create codespace on main"**.

GitHub will now automatically spin up a dedicated Linux Cloud VM with 8 GB RAM and Docker.
Our pre-configured `.devcontainer/devcontainer.json` will automatically trigger `setup_codespace.sh`, which will:
- Install the lightweight K3d Kubernetes engine.
- Spin up a **4-Node Real Kubernetes Cluster** (`1 control-plane server + 3 worker agent nodes`).
- Install all Python machine learning dependencies (`tensorflow`, `xgboost`, `scikit-learn`, `fastapi`, `uvicorn`).
- Deploy RBAC permissions and target microservices.

*(This initial cloud setup takes about 2 to 3 minutes).*

---

## Step 3: Verify the 4-Node Kubernetes Cluster

In the built-in terminal at the bottom of Codespaces, run:
```bash
kubectl get nodes -o wide
```
You will see all 4 real nodes active in the cloud:
```
NAME                             STATUS   ROLES                  AGE   VERSION        INTERNAL-IP
k3d-proactive-cluster-server-0   Ready    control-plane,master   1m    v1.30.x+k3s1   ...
k3d-proactive-cluster-agent-0    Ready    <none>                 1m    v1.30.x+k3s1   ...
k3d-proactive-cluster-agent-1    Ready    <none>                 1m    v1.30.x+k3s1   ...
k3d-proactive-cluster-agent-2    Ready    <none>                 1m    v1.30.x+k3s1   ...
```

---

## Step 4: Open the Live Web Dashboard

In the Codespaces terminal, run:
```bash
./start_dashboard.sh
```
*(or: `python dashboard/app.py`)*

VS Code will automatically show a popup in the bottom right corner:
> **"Your application running on port 8080 is available. Open in Browser"**

Click **"Open in Browser"**!

A new browser tab will open with a public HTTPS link (e.g., `https://<codespace-id>-8080.app.github.dev`).
You will see:
1. **Cluster Node Load Balance Meter** tracking all 4 nodes.
2. **Multi-Objective Node Ranking Table** ranking `server-0`, `agent-0`, `agent-1`, and `agent-2`.
3. **Live Cluster Pods Explorer Grid** showing pods distributed across the 4 nodes.
4. **"Scale +7 Pods (Flash Surge)"** button to trigger instant multi-node distribution!

---

## Step 5: Run the End-to-End Viva Demonstration

Open a second terminal in Codespaces (`Terminal -> New Terminal`) and run:
```bash
./start_demo.sh
```
*(or: `python demonstrate_proactive_scenario.py`)*

You will see the 5-phase live demonstration execute:
- **Phase 1:** Baseline diurnal morning e-commerce traffic.
- **Phase 2:** LSTM-XGBoost ensemble forecasts the upcoming +380% flash-sale surge 30 seconds ahead.
- **Phase 3:** Autonomous Proactive Scheduler binds and warms up a new pod on the optimal agent node *before* the surge hits.
- **Phase 4:** Flash crowd hits the cluster and is absorbed with **0% SLO violations** and < 30ms latency.
- **Phase 5:** Comparison table against Reactive Kubernetes HPA printed and saved to `evaluation/metrics_comparison.csv`.

---

## Step 6: How to Stop or Re-open Codespaces

- When you are done demonstrating, click the bottom-left gear icon in VS Code -> **Codespaces: Stop Current Codespace** (or simply close your browser tab; GitHub auto-sleeps the Codespace after 30 minutes of inactivity to save your 60 free hours).
- Whenever you need to present again (e.g. at college or during viva), simply go to [github.com/codespaces](https://github.com/codespaces) and click your codespace. It boots back up in seconds!
