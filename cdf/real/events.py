from typing import List, Optional
from pydantic import BaseModel


class BaseEvent(BaseModel):
    timestamp: float
    event_type: str
    experiment_id: str


class ExperimentStarted(BaseEvent):
    event_type: str = "ExperimentStarted"
    provider: str
    model: str
    horizons: List[int]
    total_tasks: int


class TaskStarted(BaseEvent):
    event_type: str = "TaskStarted"
    task_id: str
    horizon: int
    rollout_id: int


class TaskCompleted(BaseEvent):
    event_type: str = "TaskCompleted"
    task_id: str
    horizon: int
    rollout_id: int

    # True only when the model's trajectory matches the
    # generated ground truth.
    success: bool

    first_failure_step: Optional[int]

    ground_truth: List[int]
    parsed_output: List[int]
    raw_output: str

    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]

    latency: float

    # Normally None for a successfully processed model response.
    error_message: Optional[str]

    seed: int


class TaskFailed(BaseEvent):
    event_type: str = "TaskFailed"

    task_id: str
    horizon: int
    rollout_id: int

    error_message: str
    latency: float

    # IMPORTANT:
    # A failed/API-error task is still an attempted trajectory and
    # therefore belongs in the reliability denominator.
    #
    # This lets the metrics layer distinguish:
    #   - total attempted tasks
    #   - successful trajectories
    #   - API/transport failures
    is_api_error: bool = False

    # Preserve these when an adapter/API failure happens after the
    # deterministic task has already been generated.
    ground_truth: Optional[List[int]] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    seed: Optional[int] = None


class ExperimentProgress(BaseEvent):
    event_type: str = "ExperimentProgress"

    completed: int
    total: int

    # Successes / all attempted tasks.
    # API-error tasks therefore remain in the denominator.
    success_rate: float

    average_latency: float
    total_tokens: int


class ExperimentCompleted(BaseEvent):
    event_type: str = "ExperimentCompleted"

    total_completed: int
    total_failed: int

    output_path: str


class ExperimentStopped(BaseEvent):
    event_type: str = "ExperimentStopped"
