import asyncio
import os
import json
from cdf.config import CDFConfig
from cdf.agents.synthetic import SyntheticAgent
from cdf.evaluation.runner import ExperimentRunner
from cdf.parallel.worker_pool import WorkerPool
from cdf.evaluation.voting import Voting
from cdf.metrics.voting_bounds import VotingBounds
from cdf.metrics.drift import DriftMetrics
from cdf.utils.serialization import NumpyEncoder

class E2Experiment:
    def __init__(self, config: CDFConfig):
        self.config = config
        
    async def run_all(self):
        p0 = 0.95
        beta = 0.02
        k_values = [1, 3, 5, 9]
        
        agent = SyntheticAgent(p0=p0, beta=beta)
        runner = ExperimentRunner(agent, self.config)
        
        jobs = []
        for h in self.config.experiment.horizons:
            for i in range(self.config.experiment.n_per_horizon):
                for r_id in range(max(k_values)):
                    jobs.append({
                        "horizon": h,
                        "task_index": i,
                        "rollout_id": r_id
                    })
                    
        pool = WorkerPool(concurrency=self.config.parallel.concurrency)
        raw_results = await pool.map_async(runner.run_trajectory, jobs)
        
        # Group by (horizon, task_index)
        grouped = {}
        for r in raw_results:
            if "error" in r:
                continue
            key = (r["horizon"], r["task_id"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)
            
        summary = {}
        
        for k in k_values:
            print(f"Processing voting for k={k}")
            h_successes = {h: 0 for h in self.config.experiment.horizons}
            h_counts = {h: 0 for h in self.config.experiment.horizons}
            
            for (h, task_id), rollouts in grouped.items():
                # Sort by rollout_id and take first k
                valid_rollouts = sorted(rollouts, key=lambda x: x["rollout_id"])[:k]
                
                parsed_trajectories = [r["parsed_values"] for r in valid_rollouts]
                gt = valid_rollouts[0]["ground_truth"]
                
                voted_traj = Voting.aggregate_plurality_vote(parsed_trajectories, h)
                
                # Check if voted trajectory matches ground truth
                success = len(voted_traj) == len(gt) and all(v == g for v, g in zip(voted_traj, gt))
                
                h_counts[h] += 1
                if success:
                    h_successes[h] += 1
                    
            horizons = sorted(list(h_successes.keys()))
            s_hat_list = [h_successes[h]/h_counts[h] if h_counts[h] > 0 else 0 for h in horizons]
            h_50 = DriftMetrics.calculate_empirical_half_life(horizons, s_hat_list)
            
            # Theoretical bounds based on k=1 single-rollout success rate
            bounds_maj = []
            bounds_plur = []
            
            if k == 1:
                p_single = {h: s for h, s in zip(horizons, s_hat_list)}
                summary["single_rollout_p"] = p_single
            else:
                p_single = summary.get("single_rollout_p", {})
                
            for h, s_hat in zip(horizons, s_hat_list):
                p = p_single.get(h, 0.0)
                bounds_maj.append(VotingBounds.majority_bound(k, p))
                bounds_plur.append(VotingBounds.plurality_bound(k, p))
                
            summary[f"k_{k}"] = {
                "horizons": horizons,
                "s_h": [h_successes[h] for h in horizons],
                "n_h": [h_counts[h] for h in horizons],
                "s_hat": s_hat_list,
                "H_hat_50": h_50,
                "majority_bound": bounds_maj,
                "plurality_bound": bounds_plur
            }
            
        os.makedirs("experiments/outputs/aggregate", exist_ok=True)
        with open("experiments/outputs/aggregate/e2_voting_summary.json", "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
            
        return summary
