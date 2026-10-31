import asyncio
import os
from cdf.config import CDFConfig
from cdf.experiments.e1_validation import E1Experiment

async def main():
    config = CDFConfig()
    # E1 uses H = {1,2,4,8,16,32} and n=300 per horizon
    config.experiment.horizons = [1, 2, 4, 8, 16, 32]
    config.experiment.n_per_horizon = 300
    
    experiment = E1Experiment(config)
    await experiment.run_all()
    print("E1 complete. Check experiments/outputs for results.")

if __name__ == "__main__":
    asyncio.run(main())
