import unittest

import slope
from slope import ad
from slope.base_array import BaseArray
from slope.array import Array
import numpy as np
import os
from typing import NamedTuple
from functools import partial

DEBUG = os.environ.get("SLOPE_DEBUG", 0)


class TestJit(unittest.TestCase):
    def test_add(self):
        @slope.ad.jit
        def f(x, **kwargs):
            print("tracing!")
            out = x + x
            out = x + Array([4.0, 5.0, 6.0])
            return out

        print(f"{f.get_jit_fn()=}")
        res = f(Array([1.0, 2.0, 3.0]))
        print(res)
        print(f"{f.get_jit_fn()=}")
        res = f(Array([2.0, 4.0, 6.0]))  # should not print 'tracing!'
        print(res)
        breakpoint()


if __name__ == "__main__":
    unittest.main()
