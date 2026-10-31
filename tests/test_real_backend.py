import pytest
import os
import json
import asyncio
from pathlib import Path
from cdf.real.credentials import CredentialManager
from cdf.real.providers import ProviderRegistry, ProviderConfig
from cdf.real.adapters import get_adapter, FakeAdapter
from cdf.real.storage import JSONLStorage
from cdf.real.runner import RealLLMRunner
from cdf.real.events import TaskCompleted, ExperimentStarted, ExperimentProgress, ExperimentCompleted

@pytest.fixture
def temp_creds(tmp_path):
    path = tmp_path / "creds.json"
    mgr = CredentialManager(storage_path=str(path))
    yield mgr
    
def test_credential_manager(temp_creds):
    temp_creds.set_api_key("test_provider", "sk-1234567890abcdef1234567890abcdef")
    assert temp_creds.get_api_key("test_provider") == "sk-1234567890abcdef1234567890abcdef"
    
    masked = temp_creds.get_masked_api_key("test_provider")
    assert masked == "sk-1...cdef"
    
    temp_creds.delete_api_key("test_provider")
    assert temp_creds.get_api_key("test_provider") is None
    
    # Test short key
    temp_creds.set_api_key("short", "12345")
    assert temp_creds.get_masked_api_key("short") == "*****"

def test_provider_registry(tmp_path):
    path = tmp_path / "providers.json"
    registry = ProviderRegistry(config_path=str(path))
    
    all_p = registry.get_all_providers()
    assert "openai" in all_p
    assert "openrouter" in all_p
    
    custom = ProviderConfig(id="custom", name="Custom", models=["c1"])
    registry.add_or_update_provider(custom)
    
    assert registry.get_provider("custom").name == "Custom"
    registry.delete_provider("custom")
    assert registry.get_provider("custom") is None

def test_fake_adapter():
    async def run_test():
        adapter = get_adapter("fake", True, model="test-model", api_key="test")
        prompt = (
            "Solve the following arithmetic problem step by step.\n"
            "Initial value: 100\n\n"
            "Operations:\n"
            "1. add(45)\n"
            "2. multiply_mod(3)\n"
        )
        messages = [{"role": "user", "content": prompt}]
        res = await adapter.generate(messages)

        assert res.error is None
        assert res.model == "test-model"
        assert res.latency > 0
        assert res.total_tokens is not None
        assert "Step 1: 145" in res.text
        assert "Step 2: 435" in res.text
    asyncio.run(run_test())

def test_real_runner_streaming(tmp_path):
    async def run_test():
        adapter = FakeAdapter(model="fake-model", api_key="my-secret-key")
        storage = JSONLStorage(directory=str(tmp_path), experiment_id="test_exp")
        
        runner = RealLLMRunner(adapter, storage, "test_exp", concurrency=2)
        
        events = []
        async for event in runner.run_experiment(horizons=[1, 2], n_per_horizon=2, rollouts=1):
            events.append(event)
            
        assert isinstance(events[0], ExperimentStarted)
        assert isinstance(events[-1], ExperimentCompleted)
        
        task_events = [e for e in events if isinstance(e, TaskCompleted)]
        assert len(task_events) == 4 # 2 horizons * 2 samples
        
        # Verify file was written
        filepath = storage.get_filepath()
        assert os.path.exists(filepath)
        with open(filepath, "r") as f:
            lines = f.readlines()
            assert len(lines) > 4 # Start + 4 tasks + progress + end
            
        # Check that API keys are NOT in the raw results
        content = "".join(lines)
        assert "my-secret-key" not in content # The API key shouldn't be dumped
        assert "ExperimentStarted" in content
        assert "TaskCompleted" in content
    asyncio.run(run_test())


def test_numpy_json_safety():
    import numpy as np
    from cdf.real.api import _json_safe

    data = {
        "py_int": 42,
        "np_int32": np.int32(10),
        "np_int64": np.int64(999),
        "py_float": 3.14,
        "np_float32": np.float32(2.5),
        "np_float64": np.float64(1.234),
        "py_nan": float("nan"),
        "np_nan": np.float64(np.nan),
        "py_inf": float("inf"),
        "np_inf": np.float64(np.inf),
        "py_bool": True,
        "np_bool": np.bool_(False),
        "np_arr": np.array([1, 2, np.nan]),
        "nested": {"k": np.int64(7), "v": [np.float64(np.nan), np.int64(8)]},
    }
    safe = _json_safe(data)
    dumped = json.dumps(safe, allow_nan=False)
    assert dumped is not None
    assert type(safe["py_int"]) is int
    assert type(safe["np_int32"]) is int
    assert type(safe["np_int64"]) is int
    assert type(safe["py_bool"]) is bool
    assert type(safe["np_bool"]) is bool
    assert safe["py_nan"] is None
    assert safe["np_nan"] is None
    assert safe["py_inf"] is None
    assert safe["np_inf"] is None
    assert safe["np_arr"] == [1.0, 2.0, None]
    assert safe["nested"]["k"] == 7
    assert safe["nested"]["v"] == [None, 8]


def test_start_request_temperature():
    from cdf.real.api import StartRequest, start_experiment
    from fastapi import BackgroundTasks

    req = StartRequest(
        provider_id="fake",
        model="fake-model",
        horizons=[1, 2],
        samples_per_horizon=2,
        rollouts=1,
        concurrency=1,
        temperature=0.7,
    )
    assert req.temperature == 0.7

    req_default = StartRequest(
        provider_id="fake",
        model="fake-model",
        horizons=[1, 2],
        samples_per_horizon=2,
        rollouts=1,
        concurrency=1,
    )
    assert req_default.temperature == 0.0

    async def run_validation_test():
        bg = BackgroundTasks()
        req_invalid = StartRequest(
            provider_id="fake",
            model="fake-model",
            horizons=[1],
            samples_per_horizon=1,
            rollouts=1,
            concurrency=1,
            temperature=-0.5,
        )
        res = await start_experiment(req_invalid, bg)
        assert res.status_code == 400
        data = json.loads(res.body.decode())
        assert "temperature" in data["error"]

    asyncio.run(run_validation_test())


def test_cancellation_and_experiment_stopped_persistence(tmp_path):
    import time
    from cdf.real.events import ExperimentStopped

    async def run_test():
        adapter = FakeAdapter(model="fake-model", api_key="secret-key")
        storage = JSONLStorage(directory=str(tmp_path), experiment_id="cancel_exp")
        runner = RealLLMRunner(adapter, storage, "cancel_exp", concurrency=1)
        cancel_event = asyncio.Event()

        events = []
        async for event in runner.run_experiment(
            horizons=[1, 2, 4],
            n_per_horizon=5,
            rollouts=1,
            cancel_event=cancel_event,
        ):
            events.append(event)
            if len([e for e in events if isinstance(e, TaskCompleted)]) >= 2:
                cancel_event.set()

        # Runner must NOT emit ExperimentCompleted
        assert not any(isinstance(e, ExperimentCompleted) for e in events)

        # Simulate stop persistence
        stopped_evt = ExperimentStopped(
            timestamp=time.time(),
            experiment_id="cancel_exp",
        )
        await storage.append_event(stopped_evt)

        # Verify storage file ends with ExperimentStopped and has NO ExperimentCompleted
        filepath = storage.get_filepath()
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]

        last_event = lines[-1]
        assert last_event["event_type"] == "ExperimentStopped"
        assert not any(l.get("event_type") == "ExperimentCompleted" for l in lines)
        assert "secret-key" not in open(filepath).read()

    asyncio.run(run_test())


