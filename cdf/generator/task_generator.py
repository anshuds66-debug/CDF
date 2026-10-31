import random
from typing import List, Dict, Any
from .operations import Operation, add, multiply_mod, reverse_digits, rotate_digits_right, sum_digits

class TaskGenerator:
    def __init__(self):
        pass

    def generate_task(self, horizon: int, task_index: int) -> Dict[str, Any]:
        seed = 1000 * horizon + task_index
        rng = random.Random(seed)
        
        initial_value = rng.randint(10, 999)
        
        ops_pool = [
            lambda: Operation("add", add, rng.randint(1, 100)),
            lambda: Operation("multiply_mod", multiply_mod, rng.randint(2, 20)),
            lambda: Operation("reverse_digits", reverse_digits),
            lambda: Operation("rotate_digits", rotate_digits_right),
            lambda: Operation("sum_digits", sum_digits)
        ]
        
        operations = []
        ground_truth = []
        
        current_value = initial_value
        for _ in range(horizon):
            op = rng.choice(ops_pool)()
            operations.append(op)
            current_value = op.apply(current_value)
            ground_truth.append(current_value)
            
        return {
            "task_id": f"task_h{horizon}_{task_index}",
            "horizon": horizon,
            "seed": seed,
            "initial_value": initial_value,
            "operations": [repr(op) for op in operations],
            "ground_truth": ground_truth
        }
