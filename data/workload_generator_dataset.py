"""
Workload Dataset Generator
Generates realistic cloud microservice workload time-series traces:
- Diurnal cyclical traffic (morning peaks, evening streaming surges)
- Stochastic Gaussian noise
- Sudden flash-crowd bursts / traffic surges
- Realistic CPU, memory, request rate, and latency dynamics
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ensure config is discoverable
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import DATA_DIR


def generate_cloud_workload_traces(num_points=2880, step_seconds=30, random_seed=42):
    """
    Generate synthetic production-like Kubernetes cluster workload traces.
    2880 points at 30-sec intervals = 24 hours of cluster telemetry.
    """
    np.random.seed(random_seed)
    
    start_time = datetime(2026, 10, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(seconds=i * step_seconds) for i in range(num_points)]
    
    # 1. Base Diurnal Wave: 24h periodic wave with morning (10 AM) and evening (8 PM) peaks
    t_hours = np.array([ts.hour + ts.minute / 60.0 + ts.second / 3600.0 for ts in timestamps])
    
    # Multi-frequency diurnal cycle
    diurnal = (
        35.0
        + 25.0 * np.sin(2 * np.pi * (t_hours - 6) / 24)
        + 10.0 * np.sin(4 * np.pi * (t_hours - 12) / 24)
    )
    diurnal = np.clip(diurnal, 10.0, 75.0)
    
    # 2. Stochastic noise
    noise = np.random.normal(0, 3.5, num_points)
    
    # 3. Sudden Bursts / Flash Crowds (Traffic spikes)
    bursts = np.zeros(num_points)
    
    # Inject 8 distinct traffic spikes (surges lasting 10 to 30 minutes)
    spike_locations = [300, 720, 1100, 1450, 1800, 2150, 2400, 2700]
    for loc in spike_locations:
        duration = np.random.randint(15, 35)
        magnitude = np.random.uniform(30.0, 55.0)
        for d in range(duration):
            if loc + d < num_points:
                # Shape like an exponential rise and decay
                shape = np.sin(np.pi * d / duration)
                bursts[loc + d] += magnitude * shape
                
    # Combine for total Request Rate (requests / second)
    request_rate = diurnal + noise + bursts
    request_rate = np.clip(request_rate, 5.0, 150.0)
    
    # Derived CPU Usage (cores): Base idle + load proportional + slight non-linear saturation
    # Assuming a microservice running across 4 allocated cores
    base_cpu = 0.4  # idle 400m
    cpu_usage_cores = base_cpu + (request_rate / 150.0) * 3.2 + np.random.normal(0, 0.05, num_points)
    cpu_usage_cores = np.clip(cpu_usage_cores, 0.2, 4.0)
    
    # Derived Memory Usage (MB): Gradual memory caching + active request buffers
    mem_trend = np.linspace(400, 650, num_points)
    memory_usage_mb = mem_trend + (request_rate * 4.5) + np.random.normal(0, 12, num_points)
    memory_usage_mb = np.clip(memory_usage_mb, 300, 1800)
    
    # Network Throughput (MB/s)
    network_rx_mbps = (request_rate * 0.18) + np.random.normal(0, 0.5, num_points)
    network_rx_mbps = np.clip(network_rx_mbps, 0.5, 35.0)
    
    # Response Time / Latency (ms): Under low load ~ 25-45ms, spikes exponentially when CPU > 75%
    utilization_ratio = cpu_usage_cores / 4.0
    latency_ms = 30.0 + 20.0 * utilization_ratio + np.random.normal(0, 3, num_points)
    
    # Exponential degradation when CPU > 75%
    overload_idx = np.where(utilization_ratio > 0.75)[0]
    for idx in overload_idx:
        excess = (utilization_ratio[idx] - 0.75) / 0.25
        latency_ms[idx] += 180.0 * (excess ** 2) + np.random.uniform(20, 60)
        
    latency_ms = np.clip(latency_ms, 15.0, 600.0)
    
    df = pd.DataFrame({
        "timestamp": [ts.strftime("%Y-%m-%d %H:%M:%S") for ts in timestamps],
        "request_rate": np.round(request_rate, 2),
        "cpu_usage_cores": np.round(cpu_usage_cores, 4),
        "memory_usage_mb": np.round(memory_usage_mb, 2),
        "network_rx_mbps": np.round(network_rx_mbps, 2),
        "latency_ms": np.round(latency_ms, 2)
    })
    
    return df


def create_train_test_datasets():
    """Create train (75%) and test (25%) workload datasets."""
    print("[DataGenerator] Generating realistic Kubernetes workload traces...")
    df = generate_cloud_workload_traces(num_points=2880, step_seconds=30)
    
    split_idx = int(len(df) * 0.75)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    train_path = DATA_DIR / "workload_train.csv"
    test_path = DATA_DIR / "workload_test.csv"
    full_path = DATA_DIR / "workload_full.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    df.to_csv(full_path, index=False)
    
    print(f"[DataGenerator] Successfully saved datasets:")
    print(f"  - Full trace:  {full_path} ({len(df)} samples)")
    print(f"  - Train set:   {train_path} ({len(train_df)} samples)")
    print(f"  - Test set:    {test_path} ({len(test_df)} samples)")
    return train_path, test_path


if __name__ == "__main__":
    create_train_test_datasets()
