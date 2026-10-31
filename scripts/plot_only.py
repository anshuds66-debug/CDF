import json
from cdf.visualization.plots import Plotter

with open("experiments/outputs/aggregate/e1_all_summary.json", "r") as f:
    e1_summary = json.load(f)
    
with open("experiments/outputs/aggregate/e2_voting_summary.json", "r") as f:
    e2_summary = json.load(f)
    
with open("experiments/outputs/aggregate/e3_scaling_summary.json", "r") as f:
    e3_summary = json.load(f)

plotter = Plotter()
plotter.plot_reliability_curves(e1_summary)
plotter.plot_excess_drift(e1_summary)
plotter.plot_e1_half_life(e1_summary)
plotter.plot_voting(e2_summary)
plotter.plot_e2_half_life(e2_summary)
plotter.plot_scaling(e3_summary)

import subprocess
print("Generating Dashboard and Report...")
subprocess.run(["python", "scripts/generate_dashboard.py"], check=True)

print("Plots, report, and dashboard generated!")
