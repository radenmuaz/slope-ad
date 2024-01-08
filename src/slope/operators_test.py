import unittest

import slope
from slope.core import Tensor
import numpy as np
from typing import NamedTuple


class Result(NamedTuple):
    y: Tensor
    y_dot: Tensor
    L: Tensor
    gL_y: Tensor


class TestGrad(unittest.TestCase):
    @staticmethod
    def run_ad_fns(f, *args, **kwargs):
        args_dot = [slope.ones_like(x) for x in args]
        y, f_lin = slope.linearize(f, *args, **kwargs)
        y_dot = f_lin(*args_dot)
        loss_fn = slope.value_and_grad(lambda *args,: f(*args).sum())
        L, gL_y = loss_fn(*args)
        res = Result(y, y_dot, L, gL_y)
        slope.dblog(f"{args=}", enable=slope.core.backend.LOG_JIT)
        slope.dblog(f"{res=}", enable=slope.core.backend.LOG_JIT)

    def test_maximum(self):
        def f(x, **kwargs):
            z = slope.zeros_like(x)
            out = x.maximum(z)
            return out

        res = self.run_ad_fns(f, slope.tensor([1, 0.5, -0.4, 0, -200]))


if __name__ == "__main__":
    unittest.main()
