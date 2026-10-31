import pytest
import asyncio
from cdf.parallel.worker_pool import WorkerPool
from cdf.parallel.map_reduce import MapReduce

async def dummy_job(x):
    return {"x": x * 2}

def test_worker_pool():
    async def run():
        pool = WorkerPool(concurrency=2)
        jobs = [{"x": 1}, {"x": 2}, {"x": 3}]
        res = await pool.map_async(dummy_job, jobs)
        # Sort results because async can return in any order
        res_sorted = sorted(res, key=lambda d: d["x"])
        assert res_sorted == [{"x": 2}, {"x": 4}, {"x": 6}]
    
    asyncio.run(run())

def test_map_reduce_associativity():
    p1 = {"h": 4, "n": 1, "s": 1, "ff_hist": {"success": 1}}
    p2 = {"h": 4, "n": 1, "s": 0, "ff_hist": {2: 1}}
    p3 = {"h": 4, "n": 1, "s": 0, "ff_hist": {3: 1}}
    
    r_1_2 = MapReduce.reduce_partials(p1, p2)
    r_left = MapReduce.reduce_partials(r_1_2, p3)
    
    r_2_3 = MapReduce.reduce_partials(p2, p3)
    r_right = MapReduce.reduce_partials(p1, r_2_3)
    
    assert r_left == r_right
    assert r_left["n"] == 3
    assert r_left["s"] == 1
    assert r_left["ff_hist"] == {"success": 1, 2: 1, 3: 1}
