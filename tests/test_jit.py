import unittest

import slope as sp
from slope import rt
import numpy as np
import os
from typing import NamedTuple
from functools import partial


DEBUG = os.backendiron.get("SLOPE_DEBUG", 0)


class TestJit(unittest.TestCase):
    def test_maximum_sum_grad(self):
        @machine.jit
        def loss(x):
            out = x
            out = out.maximum(machine.procedures.zeros_like(out))
            out = out.sum()
            return out

        # @machine.jit
        def f(x):
            return machine.grad(loss)(x)

        # print(f"{f.get_jit_fn()=}")
        x = machine.backend.tensor([1.0, 2.0, 3.0])
        res = f(x)
        print(res)
        res = f(x)
        print(res)
        # print(f"{f.get_jit_fn()=}")

    def test_sum_grad(self):
        @machine.jit
        def loss(x):
            out = x.sum()
            return out

        # @machine.jit
        def f(x):
            return machine.grad(loss)(x)

        # print(f"{f.get_jit_fn()=}")
        x = machine.backend.tensor([1.0, 2.0, 3.0])
        res = f(x)
        print(res)
        res = f(x)
        print(res)
        # print(f"{f.get_jit_fn()=}")

    def test_reshape(self):
        @machine.jit
        def f(x):
            print("tracing")
            out = x.reshape((-1,))
            return out

        # print(f"{f.get_jit_fn()=}")
        x = machine.backend.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        res = f(x)
        print(res)
        res = f(x)
        print(res)
        # print(f"{f.get_jit_fn()=}")

    def test_add_v1(self):
        @machine.jit
        def f(x, y):
            print("tracing")
            out = x + y
            out = out.sum(keepdims=True)
            return out

        # print(f"{f.get_jit_fn()=}")
        x = machine.backend.tensor([1.0, 2.0, 3.0])
        y = machine.backend.tensor([4.0, 5.0, 6.0])
        res = f(x, y)
        print(res)
        res = f(y, x + x)
        print(res)
        # print(f"{f.get_jit_fn()=}")

    def test_add(self):
        @machine.jit
        def f(args0, args1):
            x, y = args0[0][0], args0[0][1]
            z = args1["s"]
            out = x + y
            out = out + z
            out = out + 1.0
            out = out.sum(keepdims=True)
            # out = x + Tensor([4.0, 5.0, 6.0])
            return out

        # print(f"{f.get_jit_fn()=}")
        # a = machine.backend.tensor([1.0, 2.0])
        # a = a.pad((0,), (1,), None, 0.0)
        # a = a.slice((0,), (2,), (1,))
        x = machine.backend.tensor([1.0, 2.0, 3.0])
        y = machine.backend.tensor([4.0, 5.0, 6.0])
        args0 = ((x, y),)
        args1 = {"s": 1.0}
        res = f(args0, args1)
        print(res)
        res = f(args0, args1)
        # print(f"{f.get_jit_fn()=}")
        print(res)

    def test_conv(self):
        # @machine.jit
        def f(x):
            print("tracing!")
            out = x + x
            out = out.sum(keepdims=True)
            # out = x + Tensor([4.0, 5.0, 6.0])
            return out

        # print(f"{f.get_jit_fn()=}")
        # a = machine.backend.tensor([1.0, 2.0])
        # a = a.pad((0,), (1,), None, 0.0)
        # a = a.slice((0,), (2,), (1,))
        res = f(machine.backend.tensor([1.0, 2.0, 3.0]))
        print(res)
        res = f(machine.backend.tensor([4.0, 5.0, 6.0]))
        # print(f"{f.get_jit_fn()=}")
        print(res)


if __name__ == "__main__":
    unittest.main()
