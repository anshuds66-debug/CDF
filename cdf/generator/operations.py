import math

def add(x: int, y: int) -> int:
    return x + y

def multiply_mod(x: int, m: int) -> int:
    return (x * m) % 10000

def reverse_digits(x: int) -> int:
    return int(str(abs(x))[::-1]) * (1 if x >= 0 else -1)

def rotate_digits_right(x: int) -> int:
    s = str(abs(x))
    if len(s) <= 1:
        return x
    res = int(s[-1] + s[:-1])
    return res * (1 if x >= 0 else -1)

def sum_digits(x: int) -> int:
    return sum(int(d) for d in str(abs(x))) * (1 if x >= 0 else -1)

# Operation definitions for generator
class Operation:
    def __init__(self, name: str, func, *args):
        self.name = name
        self.func = func
        self.args = args

    def apply(self, x: int) -> int:
        return self.func(x, *self.args)

    def __repr__(self):
        args_str = ", ".join(map(str, self.args))
        if args_str:
            return f"{self.name}({args_str})"
        return f"{self.name}()"
