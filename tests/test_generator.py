import pytest
from cdf.generator.operations import add, multiply_mod, reverse_digits, rotate_digits_right, sum_digits
from cdf.generator.task_generator import TaskGenerator

def test_operations():
    assert add(10, 5) == 15
    assert multiply_mod(10, 5) == 50
    assert multiply_mod(2000, 10) == 20000 % 10000
    assert reverse_digits(123) == 321
    assert reverse_digits(-456) == -654
    assert rotate_digits_right(123) == 312
    assert rotate_digits_right(-456) == -645
    assert sum_digits(123) == 6
    assert sum_digits(-45) == -9

def test_task_generator_determinism():
    gen = TaskGenerator()
    task1 = gen.generate_task(horizon=4, task_index=1)
    task2 = gen.generate_task(horizon=4, task_index=1)
    
    assert task1["seed"] == task2["seed"]
    assert task1["initial_value"] == task2["initial_value"]
    assert task1["operations"] == task2["operations"]
    assert task1["ground_truth"] == task2["ground_truth"]

def test_task_generator_correctness():
    gen = TaskGenerator()
    task = gen.generate_task(horizon=2, task_index=10)
    assert len(task["ground_truth"]) == 2
    assert len(task["operations"]) == 2
