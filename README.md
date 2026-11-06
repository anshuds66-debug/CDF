# Capability Drift Framework (CDF)

CDF measures how an LLM-based agent’s end-to-end reliability changes as the number of dependent steps in a task—the horizon, \(h\)—increases. A trajectory succeeds only when every reported intermediate result is correct. CDF uses deterministic ground truth so that results can be graded without an LLM judge.

## Installation & Setup

1. Clone or copy this repository.
2. Ensure Python 3.9 or later is installed.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running Tests

Run the test suite covering task generation, parsing and grading, metric calculations, voting bounds, bootstrap intervals, and result aggregation:

```bash
PYTHONPATH=. pytest tests/
```

## Running Experiments (Synthetic)

The synthetic agent has per-step correctness probability

\[
p_t = p_0 \exp(-\beta t).
\]

Errors are sticky: after the first incorrect step, subsequent steps are also incorrect. When \(\beta=0\), the agent’s per-step reliability is constant. Positive \(\beta\) models reliability that declines with step position.

Run an experiment from the repository’s `cdf/` directory:

- **E1 — Estimator validation:** tests memoryless and degrading synthetic agents with known behavior. It estimates drift metrics and bootstrap confidence intervals.

  ```bash
  PYTHONPATH=. python scripts/run_e1.py
  ```

- **E2 — Rollout voting:** measures how voting across \(k \in \{1, 3, 5, 9\}\) rollouts changes whole-trajectory success by horizon. Results are compared with majority and plurality reference bounds.

  ```bash
  PYTHONPATH=. python scripts/run_e2.py
  ```

- **E3 — Harness scaling:** measures runtime, speedup, and efficiency across concurrency settings using simulated per-step latency. This experiment uses a synthetic endpoint; it is not a measurement of a real model’s API latency.

  ```bash
  PYTHONPATH=. python scripts/run_e3.py
  ```

Run all three synthetic experiments and generate the outputs:

```bash
PYTHONPATH=. python scripts/run_all.py
```

### Output Locations

- **Raw JSONL trajectories:** `experiments/outputs/raw/`
- **Aggregated statistics:** `experiments/outputs/aggregate/`
- **Figures:** `experiments/outputs/figures/`
- **Summary report:** `experiments/outputs/report.md`
- **Generated static dashboard:** `experiments/outputs/dashboard/index.html`

Running experiments can replace files with the same output names. Copy or archive results you want to keep before rerunning them.

## Running Real-Model Experiments

The command-line script `scripts/run_real_model.py` uses an OpenAI-compatible endpoint. It expects model responses with one line per step, formatted as `Step X: <value>`. Missing, malformed, or out-of-order steps are treated as failures beginning at the first unparsed step.

Set the environment variables before starting a run:

```bash
export CDF_API_KEY="your-api-key"
export CDF_BASE_URL="https://api.openai.com/v1"  # Optional
export CDF_MODEL="your-available-model-id"
```

Then run:

```bash
PYTHONPATH=. python scripts/run_real_model.py
```

The command-line script uses a smaller default sample size than the paper’s proposed protocol. Treat a small run as exploratory, and record its model, sample count, horizons, temperature, and concurrency when reporting results. Use a model ID currently available to your provider and account.

### Windows PowerShell

Set environment variables in the current PowerShell session with:

```powershell
$env:CDF_API_KEY = "your-api-key"
$env:CDF_BASE_URL = "https://api.openai.com/v1" # Optional
$env:CDF_MODEL = "your-available-model-id"
```

### Interactive Dashboard

Start the real-model dashboard from the repository’s `cdf/` directory:

```bash
python scripts/start_dashboard.py
```

Open `http://127.0.0.1:8000` in your browser. The dashboard lets you configure a provider and model, run experiments, view progress, and inspect saved experiment history. Stop the server with `Ctrl+C`.

The dashboard supports the native Google Gemini adapter when the `google-genai` dependency is installed and a Gemini API key is configured. Provider credentials are stored locally outside the repository and displayed in masked form.

The current real-model runner sends the full operation sequence in one prompt and asks the model to return all intermediate answers in one response. It measures end-to-end correctness across horizons, but it does not make one dependent API call for every step.

## Configuration & Reproducibility

Defaults are managed in `cdf/config.py`. Individual experiment scripts can also set their own horizons, sample sizes, rollout counts, or concurrency. Record the settings used for each run, including:

- Model and provider
- Horizons and samples per horizon
- Rollout count
- Temperature and concurrency
- Run date and any API errors

Ground-truth task generation uses the seed `1000 * h + i`, where \(h\) is the horizon and \(i\) is the task index. This makes task contents reproducible for the same horizon and task index. Preserve the generated results and run configuration when you need to reproduce or audit a measurement.

## Interpretation of Metrics

- **\(\lambda\) — Drift coefficient:** The fitted decline rate in \(S(h) \approx A e^{-\lambda h}\). A larger \(\lambda\) means faster decline across the tested horizons. For a memoryless agent with constant per-step success probability \(p\), \(\lambda=-\ln p\). Thus, constant per-step reliability can still produce declining whole-trajectory success as the task gets longer.
- **\(H_{1/2}\) — Fitted half-life:** The horizon at which the fitted curve reaches half its fitted intercept \(A\): \(H_{1/2}=\ln(2)/\lambda\).
- **\(H^{50}\) — Empirical half-life:** The horizon where the measured success curve crosses 0.5, estimated by log-linear interpolation. If the measured curve never crosses 0.5, the empirical half-life may be beyond the largest tested horizon.
- **\(\delta(h)\) — Excess drift:** The amount the measured log-reliability falls below the exponential baseline fitted on short horizons. Positive, increasing values indicate that observed success falls below that baseline; \(\delta(h)\) alone does not identify the cause.
- **Majority reference \(S_{\text{maj}}\):** A pessimistic voting case in which incorrect rollouts tend to agree with one another.
- **Plurality reference \(S_{\text{plur}}\):** An optimistic voting case in which incorrect rollouts do not collide. Measured voting performance can be compared with these reference assumptions; they are not guarantees for every error distribution.

Interpret fitted metrics together with the reliability curve, sample sizes, and confidence intervals. A fitted value beyond the largest measured horizon is an extrapolation.

## Scope and Limitations

- Synthetic experiments validate metric behavior against controlled ground truth; they do not establish performance for a particular real LLM.
- The reference arithmetic and string tasks keep individual operations simple and do not represent every task domain or difficulty level.
- Real providers may impose rate limits, return errors, or exhibit variable latency. Record failed requests and provider conditions when interpreting a run.
- A positive excess-drift value can be consistent with context effects, accumulated mistakes, formatting failures, or truncation. Further experiments are needed to distinguish those causes.
- CDF measures reliability as task horizon increases. It does not by itself establish broad generalization or general intelligence.