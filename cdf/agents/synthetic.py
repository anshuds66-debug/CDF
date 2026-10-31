import math
import random
from typing import Dict, Any, List
from .base import BaseAgent
import asyncio

class SyntheticAgent(BaseAgent):
    def __init__(self, p0: float, beta: float, latency_ms: int = 0):
        self.p0 = p0
        self.beta = beta
        self.latency_ms = latency_ms

    async def run(self, task: Dict[str, Any], rollout_id: int = 0) -> str:
        """
        Runs the synthetic agent.
        Simulates step-by-step evaluation where p_t = p0 * exp(-beta * t).
        Errors are sticky: if step t is wrong, step t+1 and onwards are also wrong.
        """
        horizon = task["horizon"]
        ground_truth = task["ground_truth"]
        
        # Unique seed for this specific rollout
        # Incorporate rollout_id so multiple rollouts for the same task are independent
        seed = hash((task["seed"], rollout_id, "synthetic"))
        rng = random.Random(seed)
        
        output_lines = []
        is_correct = True
        
        for t in range(horizon):
            if self.latency_ms > 0:
                # Log-normal latency with mean latency_ms
                mu = math.log(self.latency_ms) - (0.5 ** 2) / 2
                sigma = 0.5
                latency = rng.lognormvariate(mu, sigma)
                await asyncio.sleep(latency / 1000.0)
                
            if is_correct:
                p_t = self.p0 * math.exp(-self.beta * t)
                if rng.random() <= p_t:
                    # Correct
                    val = ground_truth[t]
                else:
                    # Error
                    is_correct = False
                    # Generate a wrong value
                    val = ground_truth[t] + rng.choice([-1, 1]) * rng.randint(1, 100)
            else:
                # Once wrong, always wrong (sticky errors)
                val = ground_truth[t] + rng.choice([-1, 1]) * rng.randint(1, 100)
                
            output_lines.append(f"Step {t+1}: {val}")
            
        return "\n".join(output_lines)
