# Capability Drift Framework (CDF)

CDF is a parallel framework for measuring horizon-dependent reliability degradation in LLM-based agents. It evaluates how an agent's end-to-end reliability changes as the number of dependent steps in a task (the "horizon", $h$) increases.

## Installation & Setup

1. Clone or copy this repository.
2. Ensure you have Python 3.9+ installed.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests
Run the comprehensive `pytest` suite covering operations, ground truth generation, grading, metrics computation (lambda, half-life, delta, bounds, bootstrap), and map-reduce associativity:
```bash
PYTHONPATH=. pytest tests/
```

## Running Experiments (Synthetic)

The framework reproduces the theoretical capability drift curves using a simulated degrading agent with reliability law $p_t = p_0 \exp(-\beta t)$.

- **E1 - Estimator Validation**:
  ```bash
  PYTHONPATH=. python scripts/run_e1.py
  ```
- **E2 - Parallel Rollout Voting**:
  ```bash
  PYTHONPATH=. python scripts/run_e2.py
  ```
- **E3 - Harness Scaling**:
  ```bash
  PYTHONPATH=. python scripts/run_e3.py
  ```
Or run all at once:
```bash
PYTHONPATH=. python scripts/run_all.py
```

### Output Locations
- **Raw JSONL Trajectories**: `experiments/outputs/raw/`
- **Aggregated Statistics**: `experiments/outputs/aggregate/`
- **Publication-Quality Plots**: `experiments/outputs/figures/`
- **Summary Report**: `experiments/outputs/report.md`

## Running Real-Model Experiments (OpenAI API)

To run CDF against a real LLM endpoint, use the `scripts/run_real_model.py` script. The framework strictly parses outputs formatted as `Step X: <value>`.

1. Set the required environment variables:
   ```bash
   export CDF_API_KEY="your-api-key"
   export CDF_BASE_URL="https://api.openai.com/v1"  # Optional, default shown
   export CDF_MODEL="gpt-4o"                        # Optional
   ```
2. Run the script:
   ```bash
   PYTHONPATH=. python scripts/run_real_model.py
   ```

## Configuration & Reproducibility
Central configuration is managed via `cdf/config.py`. Adjust horizons, sample sizes, bootstrap $B$, API timeout/retries, and worker concurrency here.

All ground truth generation uses deterministic seeding: `seed = 1000 * h + i`, making task contents entirely reproducible. Parametric bootstrapping and simulated agent rollouts use explicit random seeds.

## Interpretation of Metrics

- **$\lambda$ (Drift Coefficient)**: The rate of exponential degradation per step. $\lambda = 0$ implies a perfectly memoryless, non-degrading agent. Higher $\lambda$ implies faster capability decay as horizon increases.
- **$H_{1/2}$ (Exponential Half-Life)**: Theoretical horizon at which the success probability $S(h)$ halves. $H_{1/2} = \ln(2) / \lambda$.
- **$H^{50}$ (Empirical Half-Life)**: The observed horizon where empirical success crosses $0.5$ via log-linear interpolation.
- **$\delta(h)$ (Excess Drift)**: The deviation of measured log-reliability from the short-horizon exponential baseline. $\delta(h) > 0$ indicates degradation beyond standard memoryless compounding.
- **Voting Bounds**:
  - **Majority Bound $S_{maj}$**: Represents a pessimistic case where wrong rollouts correlate perfectly.
  - **Plurality Bound $S_{plur}$**: Represents an optimistic case where incorrect rollouts are uniformly random and do not collide. Real LLM voting often falls between these bounds.
