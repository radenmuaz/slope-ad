import unittest

import slope
from slope.core import (Tensor,)
import numpy as np
import os
from typing import NamedTuple
from functools import partial

DEBUG = os.backendiron.get("SLOPE_DEBUG", 0)


class ADResult(NamedTuple):
    y: Tensor
    y_dot: Tensor
    loss: Tensor
    grads: Tensor


class TestGrad(unittest.TestCase):
    @staticmethod
    def run_ad_fns(f, *args):
        args_dot = [Tensor.ones_like(x) for x in args]
        y, f_lin = slope.ad.linearize(f, *args)
        y_dot = f_lin(*args_dot)
        loss_fn = slope.grad(lambda *args,: f(*args).sum())
        loss, grads = loss_fn(*args)
        if DEBUG:
            print(f"{args=}")
            print(f"{y=}")
            print(f"{y_dot=}")
            print(f"{loss=}")
            print(f"{grads=}")
        return ADResult(y, y_dot, loss, grads)

    def test_maximum(self):
        def f(x, **kwargs):
            z = Tensor.zeros_like(x)
            out = x.maximum(z)
            return out

        res = self.run_ad_fns(f, Tensor([1, 0.5, -0.4, 0, -200]))

    def test_slice(self):
        def _f(x, *, starts, limits, strides):
            out = x.slice(starts, limits, strides)
            return out

        # partial because slope ad funcs cannot accept kwargs
        f = partial(_f, starts=(0,), limits=(2,), strides=(1,))
        res = self.run_ad_fns(f, Tensor.arange(5))

    def test_pad(self):
        def _f(x, *, lo, hi, interior, val):
            out = x.pad(lo, hi, interior, val)
            return out

        f = partial(_f, lo=(1,), hi=(2,), interior=(0,), val=0)
        res = self.run_ad_fns(f, Tensor.arange(5))


if __name__ == "__main__":
    unittest.main()
