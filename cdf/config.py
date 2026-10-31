import yaml
import os
from pydantic import BaseModel, Field
from typing import List, Optional

class ExperimentConfig(BaseModel):
    horizons: List[int] = [1, 2, 4, 8, 16, 32]
    n_per_horizon: int = 300
    rollouts: int = 1
    temperature: float = 0.7
    bootstrap_samples: int = 500

class SyntheticConfig(BaseModel):
    p0: float = 0.95
    beta: float = 0.0

class ParallelConfig(BaseModel):
    concurrency: int = 16

class ApiConfig(BaseModel):
    model: str = ""
    base_url: str = ""
    timeout: int = 60
    max_retries: int = 5

class CDFConfig(BaseModel):
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    synthetic: SyntheticConfig = Field(default_factory=SyntheticConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)

def load_config(path: str) -> CDFConfig:
    if not os.path.exists(path):
        return CDFConfig()
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    return CDFConfig(**data)
