import asyncio
from typing import AsyncGenerator, Callable, Any, List, Dict

class StreamingPool:
    """
    An asynchronous worker pool that yields results as soon as they complete.
    Unlike the existing WorkerPool, it does not wait for all jobs to finish.
    """
    def __init__(self, concurrency: int):
        self.concurrency = concurrency
        self.queue = asyncio.Queue()
        self.active_workers = 0

    async def _worker(self, result_queue: asyncio.Queue):
        while True:
            try:
                job = self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
            func, kwargs = job
            try:
                res = await func(**kwargs)
                await result_queue.put({"success": True, "result": res, "kwargs": kwargs})
            except Exception as e:
                await result_queue.put({"success": False, "error": e, "kwargs": kwargs})
            finally:
                self.queue.task_done()

    async def map_stream(self, func: Callable, jobs: List[Dict[str, Any]]) -> AsyncGenerator[Any, None]:
        """
        Yields results immediately as they finish.
        """
        for kwargs in jobs:
            self.queue.put_nowait((func, kwargs))
            
        result_queue = asyncio.Queue()
        
        workers = [asyncio.create_task(self._worker(result_queue)) for _ in range(self.concurrency)]
        
        # We need to yield from result_queue until all jobs are done
        total_jobs = len(jobs)
        completed = 0
        
        while completed < total_jobs:
            res = await result_queue.get()
            completed += 1
            yield res
            
        # Ensure workers are fully cleaned up
        await asyncio.gather(*workers)
