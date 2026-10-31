import asyncio
import time
from typing import AsyncGenerator, Dict, Any, List

from cdf.real.adapters import BaseLLMAdapter
from cdf.real.events import (
    BaseEvent,
    ExperimentStarted,
    TaskCompleted,
    TaskFailed,
    ExperimentProgress,
    ExperimentCompleted,
)
from cdf.real.streaming_pool import StreamingPool
from cdf.generator.task_generator import TaskGenerator
from cdf.evaluation.grader import OutputParser, Grader
from cdf.real.storage import JSONLStorage


class RealLLMRunner:
    """
    Executes real-LLM CDF experiments.

    Important:
    - Uses the existing deterministic TaskGenerator.
    - Uses the existing OutputParser and Grader.
    - API failures remain part of the attempted-task denominator.
    - Events are persisted immediately to JSONL.
    - Cancellation is handled through cancel_event.
    """

    def __init__(
        self,
        adapter: BaseLLMAdapter,
        storage: JSONLStorage,
        experiment_id: str,
        concurrency: int = 4,
    ):
        self.adapter = adapter
        self.storage = storage
        self.experiment_id = experiment_id
        self.concurrency = concurrency

        self.generator = TaskGenerator()

        self.system_prompt = (
            "You are an assistant executing a sequence of exact numerical operations.\n\n"

            "Operation definitions:\n"
            "- add(y): calculate x + y.\n"
            "- multiply_mod(m): calculate (x * m) % 10000.\n"
            "- reverse_digits(): reverse the digits of x.\n"
            "- rotate_digits(): move the last digit of x to the front.\n"
            "- sum_digits(): calculate the sum of the digits of x.\n\n"

            "Execute the operations sequentially. "
            "The result of each operation becomes the input to the next operation.\n\n"

            "Output requirements:\n"
            "For every operation, output exactly one line in this format:\n"
            "Step X: <value>\n"
            "Do not include explanations, reasoning, markdown, or any other text."
        )

    async def _execute_single_task(
        self,
        horizon: int,
        task_index: int,
        rollout_id: int,
    ) -> Dict[str, Any]:

        # ------------------------------------------------------------
        # Generate deterministic task
        # ------------------------------------------------------------

        task = self.generator.generate_task(
            horizon,
            task_index,
        )

        # ------------------------------------------------------------
        # Build user prompt
        # ------------------------------------------------------------

        prompt = (
            f"Initial value: {task['initial_value']}\n"
            "Operations:\n"
        )

        for i, op in enumerate(task["operations"]):
            prompt += f"{i + 1}. {op}\n"

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        # Make every rollout deterministic at the task level while
        # allowing different rollout IDs to receive different seeds.
        seed = int(task["seed"]) + rollout_id

        # ------------------------------------------------------------
        # Call model
        # ------------------------------------------------------------

        response = await self.adapter.generate(
            messages,
            seed=seed,
        )

        # ------------------------------------------------------------
        # API / transport / adapter error
        # ------------------------------------------------------------

        if response.error:

            return {
                "success": False,
                "is_api_error": True,

                "error_message":
                    response.error,

                "task_id":
                    task["task_id"],

                "horizon":
                    horizon,

                "rollout_id":
                    rollout_id,

                "latency":
                    response.latency,

                "ground_truth":
                    task["ground_truth"],

                "seed":
                    seed,

                "input_tokens":
                    response.input_tokens,

                "output_tokens":
                    response.output_tokens,

                "total_tokens":
                    response.total_tokens,
            }

        # ------------------------------------------------------------
        # Parse model response
        # ------------------------------------------------------------

        parsed = OutputParser.parse(
            response.text,
            horizon,
        )

        # ------------------------------------------------------------
        # Grade response
        # ------------------------------------------------------------

        grade = Grader.grade(
            parsed,
            task["ground_truth"],
        )

        return {
            "success": True,
            "is_api_error": False,

            "task_id":
                task["task_id"],

            "horizon":
                horizon,

            "rollout_id":
                rollout_id,

            "grade_success":
                grade["success"],

            "first_failure_step":
                grade["first_failure_step"],

            "ground_truth":
                task["ground_truth"],

            "parsed_output":
                grade["parsed_values"],

            "raw_output":
                response.text,

            "input_tokens":
                response.input_tokens,

            "output_tokens":
                response.output_tokens,

            "total_tokens":
                response.total_tokens,

            "latency":
                response.latency,

            "seed":
                seed,
        }

    async def run_experiment(
        self,
        horizons: List[int],
        n_per_horizon: int,
        rollouts: int = 1,
        cancel_event: asyncio.Event = None,
    ) -> AsyncGenerator[BaseEvent, None]:

        if cancel_event is None:
            cancel_event = asyncio.Event()

        # ------------------------------------------------------------
        # Experiment metadata
        # ------------------------------------------------------------

        timestamp = time.time()

        total_tasks = (
            len(horizons)
            * n_per_horizon
            * rollouts
        )

        start_event = ExperimentStarted(
            timestamp=timestamp,
            experiment_id=self.experiment_id,
            provider=self.adapter.__class__.__name__,
            model=self.adapter.model,
            horizons=horizons,
            total_tasks=total_tasks,
        )

        yield start_event

        await self.storage.append_event(
            start_event
        )

        # ------------------------------------------------------------
        # Build job list
        # ------------------------------------------------------------

        jobs = []

        for h in horizons:

            for i in range(n_per_horizon):

                for r in range(rollouts):

                    jobs.append(
                        {
                            "horizon": h,
                            "task_index": i,
                            "rollout_id": r,
                        }
                    )

        # ------------------------------------------------------------
        # Streaming worker pool
        # ------------------------------------------------------------

        pool = StreamingPool(
            concurrency=self.concurrency
        )

        completed_count = 0
        failed_count = 0
        success_grade_count = 0

        total_latency = 0.0
        total_tokens = 0

        # ------------------------------------------------------------
        # Execute jobs
        # ------------------------------------------------------------

        async for res in pool.map_stream(
            self._execute_single_task,
            jobs,
        ):

            # --------------------------------------------------------
            # Stop processing newly completed results once cancelled.
            # --------------------------------------------------------

            if cancel_event.is_set():
                break

            timestamp = time.time()

            # --------------------------------------------------------
            # Worker-level exception
            # --------------------------------------------------------

            if not res["success"]:

                err = res["error"]
                kwargs = res["kwargs"]

                failed_count += 1

                task_id = (
                    f"task_h{kwargs['horizon']}_"
                    f"{kwargs['task_index']}"
                )

                evt = TaskFailed(
                    timestamp=timestamp,
                    experiment_id=self.experiment_id,

                    task_id=task_id,

                    horizon=kwargs["horizon"],

                    rollout_id=kwargs["rollout_id"],

                    error_message=str(err),

                    latency=0.0,

                    # Explicitly identify this as a worker/task
                    # infrastructure failure rather than an API
                    # response failure.
                    is_api_error=False,

                    ground_truth=None,

                    input_tokens=None,
                    output_tokens=None,
                    total_tokens=None,

                    seed=None,
                )

            else:

                data = res["result"]

                # ----------------------------------------------------
                # API / adapter error
                # ----------------------------------------------------

                if data.get("is_api_error"):

                    failed_count += 1

                    total_latency += (
                        data.get("latency")
                        or 0.0
                    )

                    total_tokens += (
                        data.get("total_tokens")
                        or 0
                    )

                    evt = TaskFailed(
                        timestamp=timestamp,

                        experiment_id=
                            self.experiment_id,

                        task_id=
                            data["task_id"],

                        horizon=
                            data["horizon"],

                        rollout_id=
                            data["rollout_id"],

                        error_message=
                            data["error_message"],

                        latency=
                            data.get("latency")
                            or 0.0,

                        # IMPORTANT:
                        # Preserve API-error information so the
                        # dashboard and metrics endpoint can
                        # distinguish request failures from
                        # model-grade failures.
                        is_api_error=True,

                        ground_truth=
                            data.get(
                                "ground_truth"
                            ),

                        input_tokens=
                            data.get(
                                "input_tokens"
                            ),

                        output_tokens=
                            data.get(
                                "output_tokens"
                            ),

                        total_tokens=
                            data.get(
                                "total_tokens"
                            ),

                        seed=
                            data.get("seed"),
                    )

                # ----------------------------------------------------
                # Normal graded result
                # ----------------------------------------------------

                else:

                    completed_count += 1

                    if data["grade_success"]:
                        success_grade_count += 1

                    total_latency += (
                        data.get("latency")
                        or 0.0
                    )

                    total_tokens += (
                        data.get("total_tokens")
                        or 0
                    )

                    evt = TaskCompleted(
                        timestamp=timestamp,

                        experiment_id=
                            self.experiment_id,

                        task_id=
                            data["task_id"],

                        horizon=
                            data["horizon"],

                        rollout_id=
                            data["rollout_id"],

                        success=
                            data["grade_success"],

                        first_failure_step=
                            data[
                                "first_failure_step"
                            ],

                        ground_truth=
                            data[
                                "ground_truth"
                            ],

                        parsed_output=
                            data[
                                "parsed_output"
                            ],

                        raw_output=
                            data[
                                "raw_output"
                            ],

                        input_tokens=
                            data.get(
                                "input_tokens"
                            ),

                        output_tokens=
                            data.get(
                                "output_tokens"
                            ),

                        total_tokens=
                            data.get(
                                "total_tokens"
                            ),

                        latency=
                            data["latency"],

                        error_message=None,

                        seed=
                            data["seed"],
                    )

            # --------------------------------------------------------
            # Emit and persist task event immediately
            # --------------------------------------------------------

            yield evt

            await self.storage.append_event(
                evt
            )

            # --------------------------------------------------------
            # Progress
            # --------------------------------------------------------

            total_done = (
                completed_count
                + failed_count
            )

            progress_interval = max(
                1,
                total_tasks // 20,
            )

            if (
                total_done % progress_interval == 0
                or total_done == total_tasks
            ):

                # API-error tasks remain in the denominator.
                attempted = total_done

                progress_success_rate = (
                    success_grade_count / attempted
                    if attempted > 0
                    else 0.0
                )

                progress_average_latency = (
                    total_latency / attempted
                    if attempted > 0
                    else 0.0
                )

                prog = ExperimentProgress(
                    timestamp=time.time(),

                    experiment_id=
                        self.experiment_id,

                    completed=
                        total_done,

                    total=
                        total_tasks,

                    success_rate=
                        progress_success_rate,

                    average_latency=
                        progress_average_latency,

                    total_tokens=
                        total_tokens,
                )

                yield prog

                await self.storage.append_event(
                    prog
                )

        # ------------------------------------------------------------
        # Cancellation
        # ------------------------------------------------------------

        if cancel_event.is_set():

            # Do not emit ExperimentCompleted for a stopped run.
            # The API layer is responsible for emitting the
            # ExperimentStopped event.

            return

        # ------------------------------------------------------------
        # Normal completion
        # ------------------------------------------------------------

        comp = ExperimentCompleted(
            timestamp=time.time(),

            experiment_id=
                self.experiment_id,

            total_completed=
                completed_count,

            total_failed=
                failed_count,

            output_path=
                self.storage.get_filepath(),
        )

        yield comp

        await self.storage.append_event(
            comp
        )