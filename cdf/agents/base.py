from abc import ABC, abstractmethod
from typing import Dict, Any, List

class BaseAgent(ABC):
    @abstractmethod
    async def run(self, task: Dict[str, Any]) -> str:
        """
        Run the agent on a task and return the raw output string.
        """
        pass
