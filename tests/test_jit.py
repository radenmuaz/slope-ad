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
        def f(x, **kwargs):
            # breakpoint()
            out = x + x
            return out

        res = f(Array([1., 2., 3.]))

if __name__ == "__main__":
    unittest.main()
