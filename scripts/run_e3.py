import asyncio
import os
from cdf.config import CDFConfig
from cdf.experiments.e3_scaling import E3Experiment

async def main():
    config = CDFConfig()
    experiment = E3Experiment(config)
    await experiment.run_scaling()
    print("E3 complete. Check experiments/outputs for results.")

if __name__ == "__main__":
    asyncio.run(main())
