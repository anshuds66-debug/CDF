from typing import Dict, Any, List
from cdf.agents.base import BaseAgent
from cdf.evaluation.grader import OutputParser, Grader
from cdf.generator.task_generator import TaskGenerator
import asyncio

class ExperimentRunner:
    def __init__(self, agent: BaseAgent, config: Any):
        self.agent = agent
        self.config = config
        self.generator = TaskGenerator()

    async def run_trajectory(self, horizon: int, task_index: int, rollout_id: int = 0) -> Dict[str, Any]:
        task = self.generator.generate_task(horizon, task_index)
        
        # In real LLM case, rollout_id should influence the API seed or temp if applicable
        raw_output = await self.agent.run(task, rollout_id=rollout_id)
        
        parsed = OutputParser.parse(raw_output, horizon)
        grade = Grader.grade(parsed, task["ground_truth"])
        
        return {
            "task_id": task["task_id"],
            "horizon": horizon,
            "rollout_id": rollout_id,
            "seed": task["seed"],
            "success": grade["success"],
            "first_failure_step": grade["first_failure_step"],
            "num_correct": grade["num_correct"],
            "parsed_values": grade["parsed_values"],
            "ground_truth": task["ground_truth"],
            "raw_output": raw_output
        }
