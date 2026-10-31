import asyncio
import os
import json
from cdf.config import CDFConfig
from cdf.agents.llm import LLMAgent
from cdf.evaluation.runner import ExperimentRunner
from cdf.parallel.worker_pool import WorkerPool
from cdf.parallel.map_reduce import MapReduce
from cdf.utils.serialization import NumpyEncoder
from cdf.metrics.drift import DriftMetrics

async def main():
    config = CDFConfig()
    
    # Check for API key
    api_key = os.environ.get("CDF_API_KEY")
    if not api_key:
        print("ERROR: CDF_API_KEY environment variable not set.")
        print("Please set it to run real model experiments.")
        return
        
    model = os.environ.get("CDF_MODEL", "gpt-3.5-turbo")
    base_url = os.environ.get("CDF_BASE_URL")
    
    print(f"Running Real LLM Experiment with model={model}")
    
    agent = LLMAgent(
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=config.experiment.temperature
    )
    
    runner = ExperimentRunner(agent, config)
    
    # We use a smaller default for real models unless overridden
    horizons = [1, 2, 4, 8, 16, 32]
    n_per_horizon = 10
    
    jobs = []
    for h in horizons:
        for i in range(n_per_horizon):
            jobs.append({
                "horizon": h,
                "task_index": i,
                "rollout_id": 0
            })
            
    pool = WorkerPool(concurrency=config.parallel.concurrency)
    raw_results = await pool.map_async(runner.run_trajectory, jobs)

    print(f"Number of raw results: {len(raw_results)}")
    print("First raw result:")
    if raw_results:
        print(raw_results[0])
    else:
        print("NO RESULTS RETURNED")
    
    os.makedirs("experiments/outputs/raw", exist_ok=True)
    with open(f"experiments/outputs/raw/real_{model.replace('/', '_').replace(':', '_')}.jsonl", "w") as f:
        for r in raw_results:
            f.write(json.dumps(r, cls=NumpyEncoder) + "\n")
            
    partials = [MapReduce.map_trajectory(r) for r in raw_results if "error" not in r]
    reduced = MapReduce.reduce_all(partials)
    
    horizons_sorted = sorted(list(reduced.keys()))
    n_h_list = [reduced[h]["n"] for h in horizons_sorted]
    s_h_list = [reduced[h]["s"] for h in horizons_sorted]
    
    fit = DriftMetrics.fit_exponential_drift(horizons_sorted, n_h_list, s_h_list)
    
    summary = {
        "model": model,
        "horizons": horizons_sorted,
        "n_h": n_h_list,
        "s_h": s_h_list,
        "lambda": fit["lambda"],
        "H_1_2": fit["H_1_2"]
    }
    
    os.makedirs("experiments/outputs/aggregate", exist_ok=True)
    with open(f"experiments/outputs/aggregate/real_{model.replace('/', '_').replace(':', '_')}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
    print("Real model experiment complete.")
    print(f"Lambda: {fit['lambda']:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
