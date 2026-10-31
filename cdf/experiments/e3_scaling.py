import asyncio
import os
import time
import json
import numpy as np
from cdf.config import CDFConfig
from cdf.agents.synthetic import SyntheticAgent
from cdf.evaluation.runner import ExperimentRunner
from cdf.parallel.worker_pool import WorkerPool
from cdf.utils.serialization import NumpyEncoder

class E3Experiment:
    def __init__(self, config: CDFConfig):
        self.config = config
        
    async def run_scaling(self):
        horizons = [2, 4, 8, 16]
        n_per_horizon = 24
        concurrency_levels = [1, 2, 4, 8, 16, 32, 64, 96]
        latency_ms = 50
        
        agent = SyntheticAgent(p0=1.0, beta=0.0, latency_ms=latency_ms)
        runner = ExperimentRunner(agent, self.config)
        
        jobs = []
        for h in horizons:
            for i in range(n_per_horizon):
                jobs.append({
                    "horizon": h,
                    "task_index": i,
                    "rollout_id": 0
                })
                
        results = {}
        T1 = None
        
        for c in concurrency_levels:
            print(f"Running scaling experiment with concurrency c={c}...")
            pool = WorkerPool(concurrency=c)
            
            start_time = time.time()
            await pool.map_async(runner.run_trajectory, jobs)
            end_time = time.time()
            
            wall_clock = end_time - start_time
            if c == 1:
                T1 = wall_clock
                
            speedup = T1 / wall_clock if T1 else 1.0
            efficiency = speedup / c
            
            results[c] = {
                "wall_clock": wall_clock,
                "speedup": speedup,
                "efficiency": efficiency
            }
            print(f"c={c}: time={wall_clock:.2f}s, speedup={speedup:.2f}x, efficiency={efficiency:.2f}")
            
        # Fit Amdahl's Law: speedup = 1 / (f + (1-f)/c)
        # 1/speedup = f + (1-f)/c = f(1 - 1/c) + 1/c
        # (1/speedup - 1/c) = f * (1 - 1/c)
        # f = (1/speedup - 1/c) / (1 - 1/c) for c > 1
        
        f_estimates = []
        for c, res in results.items():
            if c > 1:
                speedup = res["speedup"]
                f_est = (1.0 / speedup - 1.0 / c) / (1.0 - 1.0 / c)
                f_estimates.append(max(0.0, f_est)) # clamp to 0 if noise causes < 0
                
        f_mean = float(np.mean(f_estimates)) if f_estimates else 0.0
        
        summary = {
            "total_jobs": len(jobs),
            "mean_horizon": 7.5,
            "latency_ms": latency_ms,
            "results": results,
            "amdahl_f": f_mean
        }
        
        os.makedirs("experiments/outputs/aggregate", exist_ok=True)
        with open("experiments/outputs/aggregate/e3_scaling_summary.json", "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
            
        return summary
