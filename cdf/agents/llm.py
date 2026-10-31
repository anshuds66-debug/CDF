import os
import asyncio
from typing import Dict, Any
from .base import BaseAgent
from openai import AsyncOpenAI
import time

class LLMAgent(BaseAgent):
    def __init__(self, model: str, base_url: str = None, api_key: str = None, timeout: int = 60, max_retries: int = 5, temperature: float = 0.7):
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        
        api_key = api_key or os.environ.get("CDF_API_KEY")
        base_url = base_url or os.environ.get("CDF_BASE_URL")
        
        if not api_key:
            raise ValueError("API Key must be provided via CDF_API_KEY or constructor.")
            
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        
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

    async def run(self, task: Dict[str, Any], rollout_id: int = 0) -> str:
        prompt = f"Initial value: {task['initial_value']}\nOperations:\n"
        for i, op in enumerate(task['operations']):
            prompt += f"{i+1}. {op}\n"
            
        # Using task seed and rollout_id for reproducibility (OpenAI API seed parameter)
        seed = int(task["seed"]) + rollout_id
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        retries = 0
        backoff = 1.0
        
        while retries <= self.max_retries:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    seed=seed,
                )
                return response.choices[0].message.content
            except Exception as e:
                retries += 1
                if retries > self.max_retries:
                    raise e
                await asyncio.sleep(backoff)
                backoff *= 2.0
                
        return ""
