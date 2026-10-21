"""
Multi-Objective Node Ranking Module
Evaluates candidate Kubernetes worker nodes using a multi-criteria decision function:
1. Predicted CPU Headroom
2. Predicted Memory Headroom
3. Cluster Workload Balance (Variance minimization)
4. Network / Latency Penalty
5. Resource Packing Efficiency (Preventing stranded resources)

Computes composite ranking score S_i and outputs ranked candidate nodes for pod placement.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config.settings import (
    WEIGHT_CPU_HEADROOM,
    WEIGHT_MEM_HEADROOM,
    WEIGHT_LOAD_BALANCE,
    WEIGHT_LATENCY,
    WEIGHT_EFFICIENCY
)


class MultiObjectiveNodeRanker:
    def __init__(
        self,
        w_cpu: float = WEIGHT_CPU_HEADROOM,
        w_mem: float = WEIGHT_MEM_HEADROOM,
        w_balance: float = WEIGHT_LOAD_BALANCE,
        w_latency: float = WEIGHT_LATENCY,
        w_efficiency: float = WEIGHT_EFFICIENCY
    ):
        self.w_cpu = w_cpu
        self.w_mem = w_mem
        self.w_balance = w_balance
        self.w_latency = w_latency
        self.w_efficiency = w_efficiency
        
        # Normalize weights to guarantee sum == 1.0
        total_w = w_cpu + w_mem + w_balance + w_latency + w_efficiency
        self.w_cpu /= total_w
        self.w_mem /= total_w
        self.w_balance /= total_w
        self.w_latency /= total_w
        self.w_efficiency /= total_w

    def rank_nodes(
        self,
        nodes_metrics: Dict[str, Dict[str, float]],
        pod_request_cpu: float = 0.5,
        pod_request_mem_mb: float = 512.0,
        forecasted_surge_cpu: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Rank candidate nodes based on real-time metrics and forecasted upcoming workload.
        
        Args:
            nodes_metrics: Dictionary of node metrics keyed by node_name
            pod_request_cpu: CPU requirement of incoming pod(s)
            pod_request_mem_mb: Memory requirement of incoming pod(s)
            forecasted_surge_cpu: Forecasted demand increase from LSTM-XGBoost ensemble
            
        Returns:
            Sorted list of node rankings with sub-scores and composite score.
        """
        candidate_scores = []
        node_names = list(nodes_metrics.keys())
        if not node_names:
            return []

        # Current cluster CPU utilizations
        current_cpu_utils = [
            nodes_metrics[n]["cpu_used_cores"] / max(nodes_metrics[n]["cpu_capacity_cores"], 0.1)
            for n in node_names
        ]
        
        # Max latency observed for normalization
        latencies = [nodes_metrics[n].get("latency_ms", 15.0) for n in node_names]
        max_lat = max(latencies) if max(latencies) > 0 else 1.0
        min_lat = min(latencies)

        # Distribute predicted surge proportionally or evaluate impact per node
        surge_per_node = forecasted_surge_cpu / len(node_names)

        projected_cluster_variances = []
        for i, name in enumerate(node_names):
            m = nodes_metrics[name]
            cap_cpu = max(m["cpu_capacity_cores"], 0.1)
            new_cpu = m["cpu_used_cores"] + pod_request_cpu + surge_per_node
            new_util = new_cpu / cap_cpu
            
            # Simulated cluster utilization vector if placed on node i
            sim_utils = current_cpu_utils.copy()
            sim_utils[i] = new_util
            projected_cluster_variances.append(float(np.var(sim_utils)))

        max_var = max(projected_cluster_variances) if max(projected_cluster_variances) > 0 else 1.0
        min_var = min(projected_cluster_variances)

        for i, name in enumerate(node_names):
            m = nodes_metrics[name]
            cap_cpu = max(m["cpu_capacity_cores"], 0.1)
            cap_mem = max(m["mem_capacity_mb"], 100.0)
            
            used_cpu = m["cpu_used_cores"]
            used_mem = m["mem_used_mb"]

            # Projected load on this node
            proj_cpu = used_cpu + pod_request_cpu + surge_per_node
            proj_mem = used_mem + pod_request_mem_mb

            # 1. Predicted CPU Headroom: (Capacity - Projected) / Capacity
            cpu_headroom = max(0.0, (cap_cpu - proj_cpu) / cap_cpu)
            
            # 2. Predicted Memory Headroom
            mem_headroom = max(0.0, (cap_mem - proj_mem) / cap_mem)

            # 3. Workload Balance Score (Lower variance = Higher balance score)
            if max_var == min_var:
                balance_score = 1.0
            else:
                balance_score = 1.0 - ((projected_cluster_variances[i] - min_var) / (max_var - min_var + 1e-6))

            # 4. Latency Score (Lower network latency = Higher score)
            lat = m.get("latency_ms", 15.0)
            if max_lat == min_lat:
                lat_score = 1.0
            else:
                lat_score = 1.0 - ((lat - min_lat) / (max_lat - min_lat + 1e-6))

            # 5. Resource Packing Efficiency (Minimizes CPU/Mem imbalance)
            proj_cpu_util = min(1.0, proj_cpu / cap_cpu)
            proj_mem_util = min(1.0, proj_mem / cap_mem)
            efficiency_score = 1.0 - abs(proj_cpu_util - proj_mem_util)

            # Composite Multi-Objective Score S_i
            composite_score = (
                (self.w_cpu * cpu_headroom) +
                (self.w_mem * mem_headroom) +
                (self.w_balance * balance_score) +
                (self.w_latency * lat_score) +
                (self.w_efficiency * efficiency_score)
            )

            # Feasibility check: Can node accommodate pod?
            is_feasible = (proj_cpu <= cap_cpu) and (proj_mem <= cap_mem)

            candidate_scores.append({
                "node_name": name,
                "composite_score": round(float(composite_score), 4),
                "is_feasible": is_feasible,
                "sub_scores": {
                    "cpu_headroom": round(float(cpu_headroom), 4),
                    "mem_headroom": round(float(mem_headroom), 4),
                    "workload_balance": round(float(balance_score), 4),
                    "latency_score": round(float(lat_score), 4),
                    "resource_efficiency": round(float(efficiency_score), 4)
                },
                "current_util_pct": {
                    "cpu": m["cpu_util_pct"],
                    "mem": m["mem_util_pct"]
                },
                "projected_util_pct": {
                    "cpu": round(float(proj_cpu_util * 100), 2),
                    "mem": round(float(proj_mem_util * 100), 2)
                }
            })

        # Rank candidate nodes in descending order of composite score
        # Feasible nodes first, then highest composite score
        candidate_scores.sort(key=lambda x: (x["is_feasible"], x["composite_score"]), reverse=True)
        
        # Assign rank index (1 = Best)
        for rank_idx, item in enumerate(candidate_scores, 1):
            item["rank"] = rank_idx

        return candidate_scores

    def select_best_node(self, nodes_metrics: Dict[str, Dict[str, float]], **kwargs) -> Tuple[str, Dict[str, Any]]:
        """Select and return top-ranked node name and its diagnostic ranking."""
        rankings = self.rank_nodes(nodes_metrics, **kwargs)
        if not rankings:
            raise ValueError("No candidate worker nodes available for ranking.")
        best = rankings[0]
        return best["node_name"], best


if __name__ == "__main__":
    test_metrics = {
        "worker-node-1": {"cpu_used_cores": 1.2, "cpu_capacity_cores": 4.0, "cpu_util_pct": 30.0, "mem_used_mb": 2400, "mem_capacity_mb": 8192, "mem_util_pct": 29.3, "latency_ms": 12.0},
        "worker-node-2": {"cpu_used_cores": 3.2, "cpu_capacity_cores": 4.0, "cpu_util_pct": 80.0, "mem_used_mb": 6500, "mem_capacity_mb": 8192, "mem_util_pct": 79.3, "latency_ms": 25.0},
        "worker-node-3": {"cpu_used_cores": 2.0, "cpu_capacity_cores": 8.0, "cpu_util_pct": 25.0, "mem_used_mb": 4500, "mem_capacity_mb": 16384, "mem_util_pct": 27.5, "latency_ms": 14.0}
    }
    ranker = MultiObjectiveNodeRanker()
    ranked = ranker.rank_nodes(test_metrics, pod_request_cpu=1.0, pod_request_mem_mb=1024.0, forecasted_surge_cpu=1.5)
    print("Multi-Objective Node Ranking Results:")
    for r in ranked:
        print(f"Rank {r['rank']}: {r['node_name']} -> Score: {r['composite_score']} (Feasible: {r['is_feasible']}) | Sub-scores: {r['sub_scores']}")
