import asyncio
import os
import json
from cdf.config import CDFConfig
from cdf.experiments.e1_validation import E1Experiment
from cdf.experiments.e2_voting import E2Experiment
from cdf.experiments.e3_scaling import E3Experiment
from cdf.visualization.plots import Plotter

async def main():
    config = CDFConfig()
    
    print("Running E1 Validation...")
    config.experiment.horizons = [1, 2, 4, 8, 16, 32]
    config.experiment.n_per_horizon = 300
    e1 = E1Experiment(config)
    e1_summary = await e1.run_all()
    
    print("Running E2 Voting...")
    config.experiment.n_per_horizon = 400
    e2 = E2Experiment(config)
    e2_summary = await e2.run_all()
    
    print("Running E3 Scaling...")
    e3 = E3Experiment(config)
    e3_summary = await e3.run_scaling()
    
    plotter = Plotter()
    plotter.plot_reliability_curves(e1_summary)
    plotter.plot_excess_drift(e1_summary)
    plotter.plot_e1_half_life(e1_summary)
    plotter.plot_voting(e2_summary)
    plotter.plot_e2_half_life(e2_summary)
    plotter.plot_scaling(e3_summary)
    
    print("Generating Dashboard and Report...")
    import subprocess
    subprocess.run(["python", "scripts/generate_dashboard.py"], check=True)
    
    print("All tasks complete. See experiments/outputs/dashboard/index.html")

if __name__ == "__main__":
    asyncio.run(main())
