import asyncio
from cdf.agents.synthetic import SyntheticAgent
from cdf.evaluation.runner import ExperimentRunner
from cdf.config import CDFConfig
from cdf.metrics.drift import DriftMetrics

async def main():
    config = CDFConfig()
    config.experiment.horizons = [1, 2, 4, 8]
    config.experiment.n_per_horizon = 20
    
    agent = SyntheticAgent(p0=0.95, beta=0.0)
    runner = ExperimentRunner(agent, config)
    
    results = {}
    
    for h in config.experiment.horizons:
        n_h = config.experiment.n_per_horizon
        s_h = 0
        for i in range(n_h):
            res = await runner.run_trajectory(h, i)
            if res["success"]:
                s_h += 1
        results[h] = {"n_h": n_h, "s_h": s_h, "S_hat": s_h / n_h}
        print(f"Horizon {h}: {s_h}/{n_h} successes (S_hat = {s_h/n_h:.2f})")
        
    h_list = list(results.keys())
    n_h_list = [results[h]["n_h"] for h in h_list]
    s_h_list = [results[h]["s_h"] for h in h_list]
    
    fit = DriftMetrics.fit_exponential_drift(h_list, n_h_list, s_h_list)
    print(f"Fit: lambda={fit['lambda']:.4f}, H_1/2={fit['H_1_2']:.2f}")

if __name__ == "__main__":
    asyncio.run(main())
