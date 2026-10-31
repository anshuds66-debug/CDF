import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import math
from matplotlib.ticker import ScalarFormatter

class Plotter:
    def __init__(self):
        sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
        os.makedirs("experiments/outputs/figures", exist_ok=True)
        
    def _format_x_axis(self, horizons):
        ax = plt.gca()
        ax.set_xticks(horizons)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        
    def plot_reliability_curves(self, e1_summary: dict):
        plt.figure(figsize=(8, 6))
        
        horizons_for_ticks = []
        for name, res in e1_summary.items():
            horizons = res["horizons"]
            if not horizons_for_ticks:
                horizons_for_ticks = horizons
            s_hat = res["s_hat"]
            plt.plot(horizons, s_hat, marker='o', label=f"{name} ($\lambda$={res['lambda']:.4f})")
            
            # Plot fitted curve if lambda is valid
            if not math.isnan(res['lambda']) and not math.isnan(res.get('A', 1.0)):
                h_fine = np.linspace(min(horizons), max(horizons), 100)
                A = res.get('A', 1.0)
                fitted = A * np.exp(-res['lambda'] * h_fine)
                plt.plot(h_fine, fitted, linestyle='--', alpha=0.5)
            
        plt.xscale('log', base=2)
        self._format_x_axis(horizons_for_ticks)
        plt.xlabel("Horizon $h$")
        plt.ylabel("Reliability $S(h)$")
        plt.title("E1: Horizon-Dependent Reliability")
        plt.legend()
        plt.tight_layout()
        plt.savefig("experiments/outputs/figures/fig1_e1_reliability.pdf")
        plt.savefig("experiments/outputs/figures/fig1_e1_reliability.png", dpi=300)
        plt.close()
        
    def plot_excess_drift(self, e1_summary: dict):
        plt.figure(figsize=(8, 6))
        horizons_for_ticks = []
        for name, res in e1_summary.items():
            delta_h = res.get("delta_h", {})
            if not delta_h:
                continue
            h_vals = sorted([int(k) for k in delta_h.keys()])
            if not horizons_for_ticks:
                horizons_for_ticks = h_vals
            actual_d_vals = []
            for h in h_vals:
                if h in delta_h:
                    actual_d_vals.append(delta_h[h])
                else:
                    actual_d_vals.append(delta_h[str(h)])
                    
            plt.plot(h_vals, actual_d_vals, marker='s', label=name)
            
        plt.axhline(0, color='black', linestyle=':', alpha=0.7)
        plt.xscale('log', base=2)
        if horizons_for_ticks:
            self._format_x_axis(horizons_for_ticks)
        plt.xlabel("Horizon $h$")
        plt.ylabel("Excess Drift $\delta(h)$")
        plt.title("E1: Excess Drift")
        plt.legend()
        plt.tight_layout()
        plt.savefig("experiments/outputs/figures/fig2_e1_excess_drift.pdf")
        plt.savefig("experiments/outputs/figures/fig2_e1_excess_drift.png", dpi=300)
        plt.close()
        
    def plot_e1_half_life(self, e1_summary: dict):
        plt.figure(figsize=(10, 6))
        names = list(e1_summary.keys())
        fitted_h50 = []
        empirical_h50 = []
        
        for name in names:
            res = e1_summary[name]
            fitted_h50.append(res["H_1_2"] if not math.isinf(res["H_1_2"]) else 0)
            emp = res["H_hat_50"]
            if isinstance(emp, str):
                empirical_h50.append(float(emp.replace('>', '').replace('<', '')))
            else:
                empirical_h50.append(emp)
                
        x = np.arange(len(names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, fitted_h50, width, label='Fitted Exponential $H_{1/2}$')
        rects2 = ax.bar(x + width/2, empirical_h50, width, label='Empirical $H^{50}$')
        
        ax.set_ylabel('Half-Life (Horizon)')
        ax.set_title('E1: Reliability Half-Life by Configuration')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()
        
        plt.tight_layout()
        plt.savefig("experiments/outputs/figures/fig3_e1_half_life.pdf")
        plt.savefig("experiments/outputs/figures/fig3_e1_half_life.png", dpi=300)
        plt.close()
        
    def plot_voting(self, e2_summary: dict):
        plt.figure(figsize=(8, 6))
        k_vals = [k.split("_")[1] for k in e2_summary.keys() if k.startswith("k_")]
        k_vals.sort(key=int)
        
        horizons_for_ticks = []
        for k_str in k_vals:
            k = int(k_str)
            res = e2_summary[f"k_{k}"]
            horizons = res["horizons"]
            if not horizons_for_ticks:
                horizons_for_ticks = horizons
            s_hat = res["s_hat"]
            plt.plot(horizons, s_hat, marker='o', label=f"Measured $k={k}$")
            
            if k > 1:
                plt.plot(horizons, res["majority_bound"], linestyle='--', alpha=0.5, label=f"Majority bound $k={k}$")
                plt.plot(horizons, res["plurality_bound"], linestyle=':', alpha=0.5, label=f"Plurality bound $k={k}$")
                
        plt.xscale('log', base=2)
        if horizons_for_ticks:
            self._format_x_axis(horizons_for_ticks)
        plt.xlabel("Horizon $h$")
        plt.ylabel("Voted Success $S_k(h)$")
        plt.title("E2: Parallel Rollout Voting Reliability")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig("experiments/outputs/figures/fig4_e2_voting_reliability.pdf")
        plt.savefig("experiments/outputs/figures/fig4_e2_voting_reliability.png", dpi=300)
        plt.close()
        
    def plot_e2_half_life(self, e2_summary: dict):
        plt.figure(figsize=(8, 6))
        k_vals = [int(k.split("_")[1]) for k in e2_summary.keys() if k.startswith("k_")]
        k_vals.sort()
        
        h50_vals = []
        for k in k_vals:
            emp = e2_summary[f"k_{k}"]["H_hat_50"]
            if isinstance(emp, str):
                h50_vals.append(float(emp.replace('>', '').replace('<', '')))
            else:
                h50_vals.append(emp)
                
        plt.plot(k_vals, h50_vals, marker='o', linestyle='-', color='indigo')
        plt.xlabel("Number of Rollouts ($k$)")
        plt.ylabel("Empirical Half-Life $H^{50}$")
        plt.title("E2: Half-Life vs Number of Rollouts")
        plt.xticks(k_vals)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig("experiments/outputs/figures/fig5_e2_half_life.pdf")
        plt.savefig("experiments/outputs/figures/fig5_e2_half_life.png", dpi=300)
        plt.close()
        
    def plot_scaling(self, e3_summary: dict):
        results = e3_summary["results"]
        c_vals = sorted([int(k) for k in results.keys()])
        def get_result(c):
            if c in results:
                return results[c]
            return results[str(c)]

        speedup = [get_result(c)["speedup"] for c in c_vals]
        efficiency = [get_result(c)["efficiency"] for c in c_vals]
        
        f = e3_summary["amdahl_f"]
        amdahl = [1 / (f + (1 - f) / c) for c in c_vals]
        
        plt.figure(figsize=(8, 6))
        plt.plot(c_vals, speedup, marker='o', label="Measured Speedup")
        plt.plot(c_vals, amdahl, linestyle='--', color='gray', label=f"Amdahl's Law (f={f:.4f})")
        plt.plot(c_vals, c_vals, linestyle=':', color='black', label="Ideal Linear Speedup")
        plt.xlabel("Concurrency $c$")
        plt.ylabel("Speedup")
        plt.title("E3: Parallel Harness Scaling (Speedup)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("experiments/outputs/figures/fig6_e3_speedup.pdf")
        plt.savefig("experiments/outputs/figures/fig6_e3_speedup.png", dpi=300)
        plt.close()
        
        plt.figure(figsize=(8, 6))
        plt.plot(c_vals, efficiency, marker='s', label="Measured Efficiency")
        plt.axhline(y=1.0, linestyle=':', color='black')
        plt.xlabel("Concurrency $c$")
        plt.ylabel("Efficiency")
        plt.title("E3: Parallel Efficiency")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.tight_layout()
        plt.savefig("experiments/outputs/figures/fig7_e3_efficiency.pdf")
        plt.savefig("experiments/outputs/figures/fig7_e3_efficiency.png", dpi=300)
        plt.close()
