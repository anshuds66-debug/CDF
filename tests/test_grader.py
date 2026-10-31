import pytest
from cdf.evaluation.grader import OutputParser, Grader

def test_parser_valid():
    output = "Here is my reasoning. Step 1: 15\nStep 2: -30\nStep 3: 400"
    parsed = OutputParser.parse(output, horizon=3)
    assert parsed == [15, -30, 400]

def test_parser_partial():
    output = "Step 1: 15\nOops I forgot Step 2. Step 3: 400"
    parsed = OutputParser.parse(output, horizon=3)
    assert parsed == [15]

def test_parser_strict_order():
    output = "Step 1: 10\nStep 3: 20"
    parsed = OutputParser.parse(output, horizon=3)
    assert parsed == [10]

def test_grader_success():
    parsed = [10, 20, 30]
    gt = [10, 20, 30]
    result = Grader.grade(parsed, gt)
    assert result["success"] is True
    assert result["first_failure_step"] is None
    assert result["num_correct"] == 3

def test_grader_failure():
    parsed = [10, 25, 30]
    gt = [10, 20, 30]
    result = Grader.grade(parsed, gt)
    assert result["success"] is False
    assert result["first_failure_step"] == 2
    assert result["num_correct"] == 2
    
def test_grader_incomplete():
    parsed = [10]
    gt = [10, 20, 30]
    result = Grader.grade(parsed, gt)
    assert result["success"] is False
    assert result["first_failure_step"] == 2
    assert result["num_correct"] == 1
