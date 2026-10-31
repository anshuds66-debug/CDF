import asyncio
from typing import List, Dict, Any, Callable

class WorkerPool:
    def __init__(self, concurrency: int):
        self.concurrency = concurrency
        self.queue = asyncio.Queue()
        self.results = []

    async def _worker(self):
        while True:
            try:
                job = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
            func, kwargs = job
            try:
                res = await func(**kwargs)
                self.results.append(res)
            except Exception as e:
                # Need to return the exception so it doesn't get lost
                self.results.append({"error": str(e), **kwargs})
            finally:
                self.queue.task_done()

    async def map_async(self, func: Callable, jobs: List[Dict[str, Any]]) -> List[Any]:
        self.results = []
        for kwargs in jobs:
            self.queue.put_nowait((func, kwargs))
            
        workers = [asyncio.create_task(self._worker()) for _ in range(self.concurrency)]
        
        await asyncio.gather(*workers)
        
        return self.results
