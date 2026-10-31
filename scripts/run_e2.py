import asyncio
import os
from cdf.config import CDFConfig
from cdf.experiments.e2_voting import E2Experiment

async def main():
    config = CDFConfig()
    config.experiment.horizons = [1, 2, 4, 8, 16, 32]
    config.experiment.n_per_horizon = 400
    
    experiment = E2Experiment(config)
    await experiment.run_all()
    print("E2 complete. Check experiments/outputs for results.")

if __name__ == "__main__":
    asyncio.run(main())
