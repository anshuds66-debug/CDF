import asyncio
import os
import json
from cdf.config import CDFConfig
from cdf.agents.synthetic import SyntheticAgent
from cdf.evaluation.runner import ExperimentRunner
from cdf.parallel.worker_pool import WorkerPool
from cdf.parallel.map_reduce import MapReduce
from cdf.metrics.drift import DriftMetrics
from cdf.metrics.bootstrap import BootstrapMetrics
from cdf.utils.serialization import NumpyEncoder

class E1Experiment:
    def __init__(self, config: CDFConfig):
        self.config = config
        
    async def run_agent_config(self, p0: float, beta: float, name: str) -> dict:
        print(f"Running E1 for {name} (p0={p0}, beta={beta})")
        agent = SyntheticAgent(p0=p0, beta=beta)
        runner = ExperimentRunner(agent, self.config)
        
        jobs = []
        for h in self.config.experiment.horizons:
            for i in range(self.config.experiment.n_per_horizon):
                jobs.append({
                    "horizon": h,
                    "task_index": i,
                    "rollout_id": 0
                })
                
        pool = WorkerPool(concurrency=self.config.parallel.concurrency)
        raw_results = await pool.map_async(runner.run_trajectory, jobs)
        
        # Save raw results
        os.makedirs("experiments/outputs/raw", exist_ok=True)
        with open(f"experiments/outputs/raw/e1_{name}.jsonl", "w") as f:
            for r in raw_results:
                f.write(json.dumps(r, cls=NumpyEncoder) + "\n")
                
        # Map-Reduce
        partials = [MapReduce.map_trajectory(r) for r in raw_results if "error" not in r]
        reduced = MapReduce.reduce_all(partials)
        
        # Calculate Metrics
        horizons = sorted(list(reduced.keys()))
        n_h_list = [reduced[h]["n"] for h in horizons]
        s_h_list = [reduced[h]["s"] for h in horizons]
        s_hat_list = [s_h/n_h if n_h > 0 else 0 for s_h, n_h in zip(s_h_list, n_h_list)]
        
        fit = DriftMetrics.fit_exponential_drift(horizons, n_h_list, s_h_list)
        ci = BootstrapMetrics.parametric_bootstrap_ci(horizons, n_h_list, s_h_list, B=self.config.experiment.bootstrap_samples)
        h_50 = DriftMetrics.calculate_empirical_half_life(horizons, s_hat_list)
        delta = DriftMetrics.calculate_excess_drift(horizons, n_h_list, s_h_list)
        
        summary = {
            "name": name,
            "p0": p0,
            "beta": beta,
            "horizons": horizons,
            "n_h": n_h_list,
            "s_h": s_h_list,
            "s_hat": s_hat_list,
            "lambda": fit["lambda"],
            "A": fit["A"],
            "lambda_ci": ci["lambda"],
            "H_1_2": fit["H_1_2"],
            "H_hat_50": h_50,
            "delta_h": delta,
            "first_failure_histograms": {h: reduced[h]["ff_hist"] for h in horizons}
        }
        
        os.makedirs("experiments/outputs/aggregate", exist_ok=True)
        with open(f"experiments/outputs/aggregate/e1_{name}_summary.json", "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
            
        return summary
        
    async def run_all(self):
        configs = [
            (0.98, 0.0, "memoryless_0.98"),
            (0.95, 0.0, "memoryless_0.95"),
            (0.90, 0.0, "memoryless_0.90"),
            (0.95, 0.03, "degrading_0.95_0.03"),
            (0.98, 0.05, "degrading_0.98_0.05")
        ]
        
        all_summaries = {}
        for p0, beta, name in configs:
            all_summaries[name] = await self.run_agent_config(p0, beta, name)
            
        with open("experiments/outputs/aggregate/e1_all_summary.json", "w") as f:
            json.dump(all_summaries, f, indent=2, cls=NumpyEncoder)
            
        return all_summaries
