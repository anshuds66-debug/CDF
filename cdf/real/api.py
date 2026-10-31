import asyncio
import os
import json
import uuid
import time
import math

from glob import glob
from collections import defaultdict, deque
from typing import List, Optional, Tuple

from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import numpy as np

from cdf.real.credentials import CredentialManager
from cdf.real.providers import ProviderRegistry, ProviderConfig
from cdf.real.adapters import get_adapter, FakeAdapter
from cdf.real.runner import RealLLMRunner
from cdf.real.storage import JSONLStorage
from cdf.real.events import ExperimentStopped

from cdf.metrics.drift import DriftMetrics
from cdf.metrics.bootstrap import BootstrapMetrics


# ============================================================================
# App
# ============================================================================

app = FastAPI(
    title="CDF Real-LLM Dashboard"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# JSON safety
# ============================================================================

def _json_safe(value):
    """
    Convert non-JSON-safe values into JSON-serializable types:
      - numpy integer scalars -> Python int
      - numpy floating scalars -> Python float
      - non-finite floats (NaN, +inf, -inf) -> None
      - numpy bool_ -> Python bool
      - numpy arrays -> lists
      - dicts and lists/tuples are processed recursively.
    """

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        val = float(value)
        if not math.isfinite(val):
            return None
        return val

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]

    if isinstance(value, dict):
        return {
            str(key): _json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    return value


# ============================================================================
# Managers
# ============================================================================

creds = CredentialManager()
registry = ProviderRegistry()


# ============================================================================
# SSE state
# ============================================================================

# Each subscriber:
#
#     (queue, experiment_id)
#
# experiment_id == ""
#     -> general stream, receives all experiments
#
# experiment_id != ""
#     -> experiment-specific stream

subscribers: List[
    Tuple[asyncio.Queue, str]
] = []


# Recent event buffer per experiment.
# This allows newly connected dashboards to replay recent events.

event_buffer: dict[str, deque] = defaultdict(
    lambda: deque(maxlen=500)
)


# Active experiment cancellation events.

cancel_flags: dict[
    str,
    asyncio.Event
] = {}


# Experiment status.

experiment_status: dict[
    str,
    str
] = {}


# ============================================================================
# SSE broadcast
# ============================================================================

async def broadcast(
    data: str,
    experiment_id: str = "",
):
    """
    Store an event and broadcast it only to subscribers interested
    in this experiment.

    General subscribers receive all experiments.

    Experiment-specific subscribers receive only their experiment.
    """

    if experiment_id:
        event_buffer[
            experiment_id
        ].append(data)

    for queue, subscriber_experiment_id in list(
        subscribers
    ):
        # General subscriber receives everything.
        #
        # Experiment-specific subscriber receives only its
        # selected experiment.

        if (
            subscriber_experiment_id
            and subscriber_experiment_id != experiment_id
        ):
            continue

        try:
            queue.put_nowait(data)

        except asyncio.QueueFull:
            # Never block the experiment engine because of a
            # slow browser/client.
            pass


# ============================================================================
# Request models
# ============================================================================

class StartRequest(BaseModel):
    provider_id: str
    model: str
    horizons: List[int]
    samples_per_horizon: int
    rollouts: int
    concurrency: int
    temperature: float = 0.0


class ProviderUpdate(BaseModel):
    name: str
    models: str
    base_url: str = ""
    is_openai_compatible: bool = True
    api_key: str = ""


# ============================================================================
# Provider routes
# ============================================================================

@app.get("/api/providers")
def get_providers():

    providers = registry.get_all_providers()

    result = []

    # Always expose built-in fake provider.
    result.append(
        {
            "id": "fake",
            "name": "Fake (Test/Mock)",
            "models": ["fake-model"],
            "base_url": None,
            "is_openai_compatible": True,
            "has_key": True,
            "masked_key": "****",
            "is_fake": True,
        }
    )

    for p_conf in providers.values():

        masked_key = (
            creds.get_masked_api_key(
                p_conf.id
            )
        )

        result.append(
            {
                "id": p_conf.id,
                "name": p_conf.name,
                "models": p_conf.models,
                "base_url": p_conf.base_url,
                "is_openai_compatible":
                    p_conf.is_openai_compatible,
                "has_key":
                    masked_key is not None,
                "masked_key":
                    masked_key,
                "is_fake": False,
            }
        )

    return result


@app.post("/api/providers/{provider_id}")
def update_provider(
    provider_id: str,
    payload: ProviderUpdate,
):

    if provider_id == "fake":
        return {
            "status": "ok",
            "note":
                "Built-in fake provider cannot be overwritten",
        }

    models = [
        m.strip()
        for m in payload.models.split(",")
        if m.strip()
    ]

    if not models:
        return JSONResponse(
            {
                "status": "error",
                "message":
                    "At least one model is required",
            },
            status_code=400,
        )

    config = ProviderConfig(
        id=provider_id,
        name=payload.name,
        models=models,
        base_url=(
            payload.base_url
            if payload.base_url
            else None
        ),
        is_openai_compatible=
            payload.is_openai_compatible,
    )

    registry.add_or_update_provider(
        config
    )

    # Only update the stored key when a real key
    # was supplied.
    if (
        payload.api_key
        and not payload.api_key.startswith("*")
    ):
        creds.set_api_key(
            provider_id,
            payload.api_key,
        )

    return {
        "status": "ok"
    }


@app.delete("/api/providers/{provider_id}")
def delete_provider(
    provider_id: str
):

    if provider_id == "fake":
        return JSONResponse(
            {
                "status": "error",
                "message":
                    "Cannot delete built-in fake provider",
            },
            status_code=400,
        )

    registry.delete_provider(
        provider_id
    )

    creds.delete_api_key(
        provider_id
    )

    return {
        "status": "ok"
    }


# ============================================================================
# Provider test
# ============================================================================

@app.get("/api/providers/{provider_id}/test")
async def test_provider(
    provider_id: str
):
    """
    Quick provider connectivity test.

    Fake provider receives a valid CDF task.
    """

    # ------------------------------------------------------------------------
    # Fake provider
    # ------------------------------------------------------------------------

    if provider_id == "fake":

        adapter = FakeAdapter(
            model="fake-model",
            api_key="fake",
        )

        test_messages = [
            {
                "role": "system",
                "content":
                    "Execute the numerical operations exactly.",
            },
            {
                "role": "user",
                "content":
                    (
                        "Initial value: 12\n"
                        "Operations:\n"
                        "1. add(5)\n"
                        "2. multiply_mod(2)\n"
                    ),
            },
        ]

        resp = await adapter.generate(
            test_messages,
            seed=1,
        )

        ok = resp.error is None

        return {
            "ok": ok,
            "message":
                (
                    "Fake adapter is available"
                    if ok
                    else resp.error
                ),
            "model":
                resp.model,
        }

    # ------------------------------------------------------------------------
    # Real provider
    # ------------------------------------------------------------------------

    p_conf = registry.get_provider(
        provider_id
    )

    if not p_conf:
        return JSONResponse(
            {
                "ok": False,
                "message":
                    "Provider not found",
            },
            status_code=404,
        )

    api_key = creds.get_api_key(
        provider_id
    )

    if not api_key:
        return {
            "ok": False,
            "message":
                "No API key stored for this provider",
        }

    model = (
        p_conf.models[0]
        if p_conf.models
        else "gpt-3.5-turbo"
    )

    adapter = get_adapter(
        provider_id,
        is_openai_compatible=(
            p_conf.is_openai_compatible
        ),
        model=model,
        api_key=api_key,
        base_url=p_conf.base_url,
        max_retries=0,
        timeout=10,
    )

    # Simple connectivity request.
    resp = await adapter.generate(
        [
            {
                "role": "user",
                "content": "Reply with: OK",
            }
        ]
    )

    if resp.error:
        return {
            "ok": False,
            "message":
                f"API error: {resp.error}",
        }

    return {
        "ok": True,
        "message":
            (
                f"Connected. Model: {resp.model}, "
                f"tokens: {resp.total_tokens}"
            ),
    }


# ============================================================================
# Experiment execution
# ============================================================================

async def _run_experiment(
    req: StartRequest,
    experiment_id: str,
    cancel_ev: asyncio.Event,
):

    experiment_status[
        experiment_id
    ] = "running"

    try:

        # --------------------------------------------------------------------
        # Provider
        # --------------------------------------------------------------------

        p_conf = registry.get_provider(
            req.provider_id
        )

        is_fake = (
            req.provider_id == "fake"
        )

        if not p_conf and not is_fake:
            raise ValueError(
                f"Provider '{req.provider_id}' not found"
            )

        # --------------------------------------------------------------------
        # Credentials
        # --------------------------------------------------------------------

        api_key = (
            creds.get_api_key(
                req.provider_id
            )
            if not is_fake
            else "fake"
        )

        if (
            not api_key
            and not is_fake
        ):
            raise ValueError(
                f"No API key for provider "
                f"'{req.provider_id}'"
            )

        # --------------------------------------------------------------------
        # Adapter
        # --------------------------------------------------------------------

        adapter = get_adapter(
            req.provider_id,
            is_openai_compatible=(
                p_conf.is_openai_compatible
                if p_conf
                else True
            ),
            model=req.model,
            api_key=api_key,
            base_url=(
                p_conf.base_url
                if p_conf
                else None
            ),
            temperature=req.temperature,
        )

        # --------------------------------------------------------------------
        # Storage
        # --------------------------------------------------------------------

        storage = JSONLStorage(
            directory=
                "experiments/outputs/real",
            experiment_id=
                experiment_id,
        )

        # --------------------------------------------------------------------
        # Runner
        # --------------------------------------------------------------------

        runner = RealLLMRunner(
            adapter=adapter,
            storage=storage,
            experiment_id=
                experiment_id,
            concurrency=
                req.concurrency,
        )

        # --------------------------------------------------------------------
        # Stream runner events
        # --------------------------------------------------------------------

        stopped_emitted = False

        async for event in runner.run_experiment(
            horizons=req.horizons,
            n_per_horizon=
                req.samples_per_horizon,
            rollouts=req.rollouts,
            cancel_event=cancel_ev,
        ):

            payload = json.dumps(
                event.model_dump(),
                allow_nan=False,
            )

            await broadcast(
                payload,
                experiment_id,
            )

            # ----------------------------------------------------------------
            # Stop request
            # ----------------------------------------------------------------

            if cancel_ev.is_set():
                break

        if cancel_ev.is_set():

            experiment_status[
                experiment_id
            ] = "stopped"

            stopped_event = ExperimentStopped(
                timestamp=time.time(),
                experiment_id=experiment_id,
            )

            await storage.append_event(
                stopped_event
            )

            status_msg = json.dumps(
                stopped_event.model_dump(),
                allow_nan=False,
            )

            await broadcast(
                status_msg,
                experiment_id,
            )

            stopped_emitted = True

            return

        # --------------------------------------------------------------------
        # Normal completion
        # --------------------------------------------------------------------

        if not cancel_ev.is_set():
            experiment_status[
                experiment_id
            ] = "completed"

    except asyncio.CancelledError:

        experiment_status[
            experiment_id
        ] = "stopped"

        if not stopped_emitted:
            stopped_event = ExperimentStopped(
                timestamp=time.time(),
                experiment_id=experiment_id,
            )

            try:
                await storage.append_event(
                    stopped_event
                )
            except Exception:
                pass

            status_msg = json.dumps(
                stopped_event.model_dump(),
                allow_nan=False,
            )

            await broadcast(
                status_msg,
                experiment_id,
            )

            stopped_emitted = True

    except Exception as e:

        experiment_status[
            experiment_id
        ] = "failed"

        err_payload = json.dumps(
            {
                "event_type":
                    "ExperimentFailed",
                "experiment_id":
                    experiment_id,
                "error":
                    str(e),
                "timestamp":
                    time.time(),
            },
            allow_nan=False,
        )

        await broadcast(
            err_payload,
            experiment_id,
        )

    finally:

        cancel_flags.pop(
            experiment_id,
            None,
        )


# ============================================================================
# Start experiment
# ============================================================================

@app.post("/api/start")
async def start_experiment(
    req: StartRequest,
    bg_tasks: BackgroundTasks,
):

    # ------------------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------------------

    if not req.horizons:
        return JSONResponse(
            {
                "error":
                    "At least one horizon is required"
            },
            status_code=400,
        )

    if any(
        h <= 0
        for h in req.horizons
    ):
        return JSONResponse(
            {
                "error":
                    "Horizons must be positive integers"
            },
            status_code=400,
        )

    if req.samples_per_horizon <= 0:
        return JSONResponse(
            {
                "error":
                    "samples_per_horizon must be > 0"
            },
            status_code=400,
        )

    if req.rollouts <= 0:
        return JSONResponse(
            {
                "error":
                    "rollouts must be > 0"
            },
            status_code=400,
        )

    if req.concurrency <= 0:
        return JSONResponse(
            {
                "error":
                    "concurrency must be > 0"
            },
            status_code=400,
        )

    if req.temperature < 0.0:
        return JSONResponse(
            {
                "error":
                    "temperature must be >= 0.0"
            },
            status_code=400,
        )

    # ------------------------------------------------------------------------
    # Experiment ID
    # ------------------------------------------------------------------------

    experiment_id = (
        f"exp_{int(time.time())}_"
        f"{uuid.uuid4().hex[:6]}"
    )

    cancel_ev = asyncio.Event()

    cancel_flags[
        experiment_id
    ] = cancel_ev

    experiment_status[
        experiment_id
    ] = "starting"

    # ------------------------------------------------------------------------
    # Start background execution
    # ------------------------------------------------------------------------

    bg_tasks.add_task(
        _run_experiment,
        req,
        experiment_id,
        cancel_ev,
    )

    return {
        "experiment_id":
            experiment_id,
        "status":
            "started",
    }


# ============================================================================
# Stop experiment
# ============================================================================

@app.delete("/api/stop/{exp_id}")
def stop_experiment(
    exp_id: str
):

    ev = cancel_flags.get(
        exp_id
    )

    if ev:

        ev.set()

        experiment_status[
            exp_id
        ] = "stopping"

        return {
            "status":
                "stopping",
            "experiment_id":
                exp_id,
        }

    cur = experiment_status.get(
        exp_id,
        "unknown",
    )

    return {
        "status":
            cur,
        "experiment_id":
            exp_id,
    }


# ============================================================================
# Experiment status
# ============================================================================

@app.get("/api/status/{exp_id}")
def get_experiment_status(
    exp_id: str
):

    return {
        "experiment_id":
            exp_id,
        "status":
            experiment_status.get(
                exp_id,
                "unknown",
            ),
    }


# ============================================================================
# SSE stream
# ============================================================================

@app.get("/api/stream")
async def stream_events(
    experiment_id: str = "",
):

    """
    Server-Sent Events stream.

    No experiment_id:
        General stream.

    experiment_id:
        Experiment-specific stream with replay.
    """

    queue = asyncio.Queue(
        maxsize=1000
    )

    subscriber = (
        queue,
        experiment_id or "",
    )

    subscribers.append(
        subscriber
    )

    # ------------------------------------------------------------------------
    # Replay recent events
    # ------------------------------------------------------------------------

    if (
        experiment_id
        and experiment_id in event_buffer
    ):

        for buffered in list(
            event_buffer[
                experiment_id
            ]
        ):

            try:
                queue.put_nowait(
                    buffered
                )
            except asyncio.QueueFull:
                break

    # ------------------------------------------------------------------------
    # Generator
    # ------------------------------------------------------------------------

    async def generator():

        try:

            while True:

                data = await queue.get()

                yield {
                    "data": data
                }

        except asyncio.CancelledError:
            raise

        finally:

            try:
                subscribers.remove(
                    subscriber
                )
            except ValueError:
                pass

    return EventSourceResponse(
        generator()
    )


# ============================================================================
# CDF Metrics
# ============================================================================

@app.get("/api/metrics/{exp_id}")
def compute_metrics(
    exp_id: str
):

    """
    Compute CDF metrics from a JSONL experiment.

    Reliability definition:

        n_h = ALL attempted trajectories
        s_h = successful trajectories

    Therefore:
        TaskCompleted failures count in n_h.
        TaskFailed/API failures also count in n_h.

    If there are zero successful trajectories,
    exponential fitting is skipped and an explicit
    insufficient_data response is returned.
    """

    filepath = (
        f"experiments/outputs/real/"
        f"{exp_id}.jsonl"
    )

    if not os.path.exists(filepath):
        return JSONResponse(
            {
                "error":
                    "Experiment not found"
            },
            status_code=404,
        )

    # ------------------------------------------------------------------------
    # Aggregation structures
    # ------------------------------------------------------------------------

    counts: dict[int, dict] = {}

    failure_steps: dict[
        int,
        list
    ] = {}

    api_errors: list = []
    task_errors: list = []

    total_attempted = 0
    total_successes = 0
    total_failures = 0

    # ------------------------------------------------------------------------
    # Read JSONL
    # ------------------------------------------------------------------------

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                try:
                    ev = json.loads(
                        line
                    )
                except json.JSONDecodeError:
                    continue

                event_type = ev.get(
                    "event_type"
                )

                # ============================================================
                # TaskCompleted
                # ============================================================

                if event_type == "TaskCompleted":

                    h = ev.get(
                        "horizon"
                    )

                    if h is None:
                        continue

                    if h not in counts:

                        counts[h] = {
                            "n": 0,
                            "s": 0,
                            "api_errors": 0,
                            "task_failures": 0,
                        }

                        failure_steps[h] = []

                    # Every TaskCompleted is an attempt.
                    counts[h]["n"] += 1
                    total_attempted += 1

                    if ev.get("success"):

                        counts[h]["s"] += 1
                        total_successes += 1

                    else:

                        counts[h][
                            "task_failures"
                        ] += 1

                        total_failures += 1

                        fs = ev.get(
                            "first_failure_step"
                        )

                        if fs is not None:

                            failure_steps[
                                h
                            ].append(fs)

                # ============================================================
                # TaskFailed
                # ============================================================

                elif event_type == "TaskFailed":

                    h = ev.get(
                        "horizon"
                    )

                    if h is None:
                        continue

                    if h not in counts:

                        counts[h] = {
                            "n": 0,
                            "s": 0,
                            "api_errors": 0,
                            "task_failures": 0,
                        }

                        failure_steps[h] = []

                    # IMPORTANT:
                    # TaskFailed is still an attempted trajectory.
                    counts[h]["n"] += 1
                    total_attempted += 1
                    total_failures += 1

                    is_api_error = bool(
                        ev.get(
                            "is_api_error",
                            False,
                        )
                    )

                    if is_api_error:

                        counts[h][
                            "api_errors"
                        ] += 1

                        api_errors.append(
                            {
                                "task_id":
                                    ev.get(
                                        "task_id"
                                    ),
                                "horizon":
                                    h,
                                "rollout_id":
                                    ev.get(
                                        "rollout_id"
                                    ),
                                "error":
                                    ev.get(
                                        "error_message",
                                        "Unknown API error",
                                    ),
                            }
                        )

                    else:

                        counts[h][
                            "task_failures"
                        ] += 1

                        task_errors.append(
                            {
                                "task_id":
                                    ev.get(
                                        "task_id"
                                    ),
                                "horizon":
                                    h,
                                "rollout_id":
                                    ev.get(
                                        "rollout_id"
                                    ),
                                "error":
                                    ev.get(
                                        "error_message",
                                        "Unknown task error",
                                    ),
                            }
                        )

    except OSError as e:

        return JSONResponse(
            {
                "error":
                    f"Could not read experiment: {e}"
            },
            status_code=500,
        )

    # ------------------------------------------------------------------------
    # No trajectory data
    # ------------------------------------------------------------------------

    if not counts:

        return {
            "experiment_id":
                exp_id,
            "status":
                "insufficient_data",
            "reason":
                "No task results were found in this experiment.",
            "horizons": [],
            "n_h": [],
            "s_h": [],
            "s_hat": [],
            "lambda": None,
            "A": None,
            "r": None,
            "H_1_2": None,
            "H_hat_50": None,
            "lambda_ci":
                [None, None],
            "delta_h": {},
            "ff_histogram": {},
            "total_attempted": 0,
            "total_successes": 0,
            "total_failures": 0,
            "api_error_count": 0,
            "api_errors": [],
            "task_error_count": 0,
            "task_errors": [],
            "counts": {},
        }

    # ------------------------------------------------------------------------
    # Horizon arrays
    # ------------------------------------------------------------------------

    horizons = sorted(
        counts.keys()
    )

    n_h_list = [
        counts[h]["n"]
        for h in horizons
    ]

    s_h_list = [
        counts[h]["s"]
        for h in horizons
    ]

    s_hat_list = [
        (
            s / n
            if n > 0
            else 0.0
        )
        for s, n in zip(
            s_h_list,
            n_h_list,
        )
    ]

    # ------------------------------------------------------------------------
    # First-failure histogram helper
    # ------------------------------------------------------------------------

    def build_failure_histogram():

        histogram: dict[
            int,
            int
        ] = {}

        for h in horizons:

            for step in failure_steps.get(
                h,
                [],
            ):

                histogram[step] = (
                    histogram.get(
                        step,
                        0,
                    )
                    + 1
                )

        return histogram

    # ------------------------------------------------------------------------
    # Zero-success case
    # ------------------------------------------------------------------------

    if total_successes == 0:

        ff_hist = (
            build_failure_histogram()
        )

        return _json_safe(
            {
                "experiment_id":
                    exp_id,

                "status":
                    "insufficient_data",

                "reason":
                    (
                        "No successful trajectories are available "
                        "for the exponential reliability fit. "
                        "Task/API failures were retained in the "
                        "reliability denominator."
                    ),

                "horizons":
                    horizons,

                "n_h":
                    n_h_list,

                "s_h":
                    s_h_list,

                "s_hat":
                    s_hat_list,

                "lambda":
                    None,

                "A":
                    None,

                "r":
                    None,

                "H_1_2":
                    None,

                "H_hat_50":
                    None,

                "lambda_ci":
                    [None, None],

                "delta_h":
                    {},

                "ff_histogram":
                    {
                        str(k): v
                        for k, v in sorted(
                            ff_hist.items()
                        )
                    },

                "total_attempted":
                    total_attempted,

                "total_successes":
                    total_successes,

                "total_failures":
                    total_failures,

                "api_error_count":
                    len(api_errors),

                "api_errors":
                    api_errors,

                "task_error_count":
                    len(task_errors),

                "task_errors":
                    task_errors,

                "counts":
                    {
                        str(h): counts[h]
                        for h in horizons
                    },
            }
        )

    # ------------------------------------------------------------------------
    # Exponential fit
    # ------------------------------------------------------------------------

    fit = None

    try:

        fit = DriftMetrics.fit_exponential_drift(
            horizons,
            n_h_list,
            s_h_list,
        )

    except Exception:
        fit = None

    # ------------------------------------------------------------------------
    # Bootstrap CI
    # ------------------------------------------------------------------------

    lambda_ci = [
        None,
        None,
    ]

    try:

        ci = (
            BootstrapMetrics
            .parametric_bootstrap_ci(
                horizons,
                n_h_list,
                s_h_list,
                B=500,
            )
        )

        candidate_ci = ci.get(
            "lambda",
            [None, None],
        )

        if (
            isinstance(
                candidate_ci,
                (list, tuple),
            )
            and len(candidate_ci) == 2
        ):

            lambda_ci = list(
                candidate_ci
            )

    except Exception:

        lambda_ci = [
            None,
            None,
        ]

    # ------------------------------------------------------------------------
    # Empirical half-life
    # ------------------------------------------------------------------------

    try:

        h50 = (
            DriftMetrics
            .calculate_empirical_half_life(
                horizons,
                s_hat_list,
            )
        )

    except Exception:

        h50 = None

    # ------------------------------------------------------------------------
    # Excess drift
    # ------------------------------------------------------------------------

    try:

        delta_h = (
            DriftMetrics
            .calculate_excess_drift(
                horizons,
                n_h_list,
                s_h_list,
            )
        )

    except Exception:

        delta_h = {}

    # ------------------------------------------------------------------------
    # First-failure histogram
    # ------------------------------------------------------------------------

    ff_hist_all = (
        build_failure_histogram()
    )

    # ------------------------------------------------------------------------
    # Fit availability
    # ------------------------------------------------------------------------

    fit_available = (
        fit is not None
        and fit.get("lambda") is not None
    )

    # ------------------------------------------------------------------------
    # Final response
    # ------------------------------------------------------------------------

    result = {

        "experiment_id":
            exp_id,

        "status":
            (
                "ok"
                if fit_available
                else "partial"
            ),

        "reason":
            (
                None
                if fit_available
                else
                (
                    "Reliability data was collected, "
                    "but the exponential drift fit could "
                    "not be computed from the available data."
                )
            ),

        "horizons":
            horizons,

        "n_h":
            n_h_list,

        "s_h":
            s_h_list,

        "s_hat":
            s_hat_list,

        "lambda":
            (
                fit.get("lambda")
                if fit_available
                else None
            ),

        "A":
            (
                fit.get("A")
                if fit_available
                else None
            ),

        "r":
            (
                fit.get("r")
                if fit_available
                else None
            ),

        "H_1_2":
            (
                fit.get("H_1_2")
                if fit_available
                else None
            ),

        "H_hat_50":
            h50,

        "lambda_ci":
            lambda_ci,

        "delta_h":
            {
                str(k): v
                for k, v in delta_h.items()
            },

        "ff_histogram":
            {
                str(k): v
                for k, v in sorted(
                    ff_hist_all.items()
                )
            },

        "total_attempted":
            total_attempted,

        "total_successes":
            total_successes,

        "total_failures":
            total_failures,

        "api_error_count":
            len(api_errors),

        "api_errors":
            api_errors,

        "task_error_count":
            len(task_errors),

        "task_errors":
            task_errors,

        "counts":
            {
                str(h): counts[h]
                for h in horizons
            },
    }

    # ------------------------------------------------------------------------
    # CRITICAL FIX:
    # Never allow NaN / Infinity to reach FastAPI's JSON encoder.
    # ------------------------------------------------------------------------

    return _json_safe(result)


# ============================================================================
# History
# ============================================================================

@app.get("/api/history")
def get_history():

    os.makedirs(
        "experiments/outputs/real",
        exist_ok=True,
    )

    files = glob(
        "experiments/outputs/real/*.jsonl"
    )

    history = []

    for filepath in sorted(
        files,
        reverse=True,
    ):

        try:

            with open(
                filepath,
                "r",
                encoding="utf-8",
            ) as fh:

                lines = [
                    line
                    for line in fh.readlines()
                    if line.strip()
                ]

            if not lines:
                continue

            first = json.loads(
                lines[0]
            )

            last = json.loads(
                lines[-1]
            )

            exp_id = first.get(
                "experiment_id",
                os.path.splitext(
                    os.path.basename(
                        filepath
                    )
                )[0],
            )

            last_event_type = last.get(
                "event_type"
            )

            if (
                last_event_type
                == "ExperimentCompleted"
            ):

                status = "Completed"

            elif (
                last_event_type
                == "ExperimentStopped"
            ):

                status = "Stopped"

            elif (
                last_event_type
                == "ExperimentFailed"
            ):

                status = "Failed"

            else:

                status = experiment_status.get(
                    exp_id,
                    "Incomplete",
                )

            if (
                last_event_type
                == "ExperimentCompleted"
            ):

                total_completed = last.get(
                    "total_completed",
                    0,
                )

            else:

                total_completed = sum(
                    1
                    for line in lines
                    if '"event_type": "TaskCompleted"'
                    in line
                    or '"event_type":"TaskCompleted"'
                    in line
                )

            history.append(
                {
                    "experiment_id":
                        exp_id,

                    "provider":
                        first.get(
                            "provider",
                            "Unknown",
                        ),

                    "model":
                        first.get(
                            "model",
                            "Unknown",
                        ),

                    "timestamp":
                        first.get(
                            "timestamp",
                            0,
                        ),

                    "total_tasks":
                        first.get(
                            "total_tasks",
                            0,
                        ),

                    "status":
                        status,

                    "total_completed":
                        total_completed,
                }
            )

        except Exception:
            # A malformed historical file should not
            # prevent the dashboard from loading the
            # remaining experiments.
            continue

    return history


# ============================================================================
# Experiment detail
# ============================================================================

@app.get("/api/history/{exp_id}")
def get_experiment_detail(
    exp_id: str
):

    filepath = (
        f"experiments/outputs/real/"
        f"{exp_id}.jsonl"
    )

    if not os.path.exists(filepath):

        return JSONResponse(
            {
                "error":
                    "Not found"
            },
            status_code=404,
        )

    events = []

    try:

        with open(
            filepath,
            "r",
            encoding="utf-8",
        ) as f:

            for line in f:

                if line.strip():

                    try:

                        events.append(
                            json.loads(
                                line
                            )
                        )

                    except json.JSONDecodeError:
                        continue

    except OSError as e:

        return JSONResponse(
            {
                "error":
                    str(e)
            },
            status_code=500,
        )

    return events


# ============================================================================
# Static dashboard — MUST BE LAST
# ============================================================================

from pathlib import Path

_dashboard_dir = Path(__file__).resolve().parent / "dashboard"
if not _dashboard_dir.exists():
    _dashboard_dir = Path("cdf/real/dashboard")

app.mount(
    "/",
    StaticFiles(
        directory=str(_dashboard_dir),
        html=True,
    ),
    name="dashboard",
)
