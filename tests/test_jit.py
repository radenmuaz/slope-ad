import unittest

import slope as sp
from slope import rt
import numpy as np
import os
from typing import NamedTuple
from functools import partial


DEBUG = os.environ.get("SLOPE_DEBUG", 0)


class TestJit(unittest.TestCase):
    def test_add(self):
        @rt.jit
        def f(xl, y):
            x = xl[0]
            print("tracing!")
            out = x + y
            out = out.sum(keepdims=True)
            # out = x + Array([4.0, 5.0, 6.0])
            return out

        # print(f"{f.get_jit_fn()=}")
        # a = rt.array([1.0, 2.0])
        # a = a.pad((0,), (1,), None, 0.0)
        # a = a.slice((0,), (2,), (1,))
        x = rt.array([1.0, 2.0, 3.0])
        y = rt.array([1.0, 2.0, 3.0])
        res = f((x,),y )
        print(res)
        res = f((x,), y)
        # print(f"{f.get_jit_fn()=}")
        print(res)

    def test_conv(self):
        # @rt.jit
        def f(x):
            print("tracing!")
            out = x + x
            out = out.sum(keepdims=True)
            # out = x + Array([4.0, 5.0, 6.0])
            return out

        # print(f"{f.get_jit_fn()=}")
        # a = rt.array([1.0, 2.0])
        # a = a.pad((0,), (1,), None, 0.0)
        # a = a.slice((0,), (2,), (1,))
        res = f(rt.array([1.0, 2.0, 3.0]))
        print(res)
        res = f(rt.array([4.0, 5.0, 6.0]))
        # print(f"{f.get_jit_fn()=}")
        print(res)


if __name__ == "__main__":
    unittest.main()
