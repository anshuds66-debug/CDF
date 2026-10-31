import re
from typing import List, Dict, Any, Tuple

class OutputParser:
    @staticmethod
    def parse(output: str, horizon: int) -> List[int]:
        """
        Parses the output of the agent strictly.
        Expected format:
        Step 1: <value>
        Step 2: <value>
        ...
        Returns a list of parsed integers. If parsing fails at any point,
        it stops and returns the parsed values up to that point.
        """
        parsed = []
        # Pattern to match "Step X: <value>" where value is an integer
        # Allow optional whitespace and optional negative sign
        pattern = re.compile(r'Step\s+(\d+):\s*(-?\d+)')
        
        # We can extract all step matches
        matches = pattern.findall(output)
        
        expected_step = 1
        for step_str, val_str in matches:
            step = int(step_str)
            if step == expected_step:
                parsed.append(int(val_str))
                expected_step += 1
                if expected_step > horizon:
                    break
            else:
                # If step out of order or missing, parsing fails from here on
                break
                
        return parsed

class Grader:
    @staticmethod
    def grade(parsed_values: List[int], ground_truth: List[int]) -> Dict[str, Any]:
        """
        Grades the parsed trajectory against the ground truth.
        """
        horizon = len(ground_truth)
        correct_steps = []
        first_failure_step = None
        
        for i in range(horizon):
            if i < len(parsed_values) and parsed_values[i] == ground_truth[i]:
                correct_steps.append(True)
            else:
                correct_steps.append(False)
                if first_failure_step is None:
                    first_failure_step = i + 1  # 1-indexed
                    
        success = (first_failure_step is None)
        
        return {
            "success": success,
            "first_failure_step": first_failure_step,
            "correct_steps": correct_steps,
            "num_correct": sum(correct_steps),
            "parsed_values": parsed_values
        }
