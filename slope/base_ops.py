import slope
import numpy as np
from slope.array_shape import ArrayShape
from typing import List, Tuple, Sequence, Any, Callable, Optional, Union
from abc import ABC, abstractmethod
import math
import functools
from slope.base_array import Array
from slope import utils

from enum import Enum, auto


def raise_not_implemented(self, *args, **kwargs):
    raise NotImplementedError


class OpType(Enum):
    Unary = auto()
    Binary = auto()
    Reduce = auto()
    Shape = auto()
    Load = auto()
    Other = auto()


class Op:
    def __init__(self, name, op_type=OpType.Other):
        self.name = name
        self.op_type = op_type
        self.impls = dict()
        self.eval = raise_not_implemented
        self.jvp = raise_not_implemented
        self.vmap = raise_not_implemented
        self.T = raise_not_implemented
        self.shape_eval = raise_not_implemented

    def __call__(self, *args, **kwargs):
        return slope.RT.bind1(self, *args, **kwargs)[0]

    def set_eval(self, fn):
        self.eval = fn

    def set_jvp(self, fn):
        self.jvp = fn

    def set_vmap(self, fn):
        self.vmap = fn

    def set_T(self, fn):
        self.T = fn

    def set_shape_eval(self, fn):
        self.shape_eval = fn

    def add_impl(self, backend):
        def add_impl_(impl):
            self[backend] = impl

        return add_impl_

    @classmethod
    def unary(cls, name):
        op = cls(name, OpType.Unary)

        @op.set_vmap
        def fn(self, axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            return [self(x, **params)], [x_bdim]

        @op.set_shape_eval
        def fn(self, axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            return [self(x, **params)], [x_bdim]

        @op.set_jvp
        def fn(self, primals, tangents, **params):
            (x,), (x_dot,) = primals, tangents
            return [self(x, **params)], [self(x_dot, **params)]

        return op

    @classmethod
    def binary(cls, name):
        op = cls(name, OpType.Binary)

        @op.set_vmap
        def fn(self, axis_size, vals_in, dims_in, **params):
            (x, y), (x_bdim, y_bdim) = vals_in, dims_in
            if x_bdim != y_bdim:
                if x_bdim is None:
                    x = slope.ad.move_batch_axis(axis_size, x_bdim, y_bdim, x)
                    x_bdim = y_bdim
                else:
                    y = slope.ad.move_batch_axis(axis_size, y_bdim, x_bdim, y)
            return [self(x, y, **params)], [x_bdim]

        @op.set_shape_eval
        def fn(x: ArrayShape, y: ArrayShape, **params) -> List[ArrayShape]:
            # if not isinstance(x, ArrayShape) or not isinstance(y, ArrayShape):
            if not type(x) in (Array, ArrayShape) or not type(x) in (Array, ArrayShape):
                # breakpoint()
                raise TypeError
            if ArrayShape.like(x) != ArrayShape.like(y):
                raise TypeError(f"{x} != {y}")
            return [ArrayShape(x.shape, x.dtype)]

        @op.set_jvp
        def fn(self, primals, tangents, **params):
            (x,), (x_dot,) = primals, tangents
            return [self(x, **params)], [self(x_dot, **params)]

        return op

    @classmethod
    def reduce(cls, name):
        op = cls(name, OpType.Reduce)

        @op.set_vmap
        def fn(cls, axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            axes = list(params["axes"])
            axes = tuple(a + (x_bdim <= a) for a in axes)
            out_bdim = x_bdim - sum(a < x_bdim for a in axes)
            params["axes"] = tuple(axes)
            return [cls.do(x, **params)], [out_bdim]

        @op.set_shape_eval
        def fn(x: ArrayShape, **params) -> List[ArrayShape]:
            axes = params["axes"]
            axes = [a + len(x.shape) if a < 0 else a for a in axes]
            axes_ = set(axes)
            new_shape = [d for i, d in enumerate(x.shape) if i not in axes_]
            return [ArrayShape(tuple(new_shape), x.dtype)]

        return op

    @classmethod
    def shape(cls, name):
        op = cls(name, OpType.Shape)
        return op

    @classmethod
    def load(cls, name):
        op = cls(name, OpType.Load)

        @op.set_jvp
        def fn(self, *args, **kwargs):
            out = cls.load_fn(*args, **kwargs)
            out_jvp = Array.ones_like(out)
            return [out], [out_jvp]

        @op.set_T
        def fn(self, cts, *args, **kwargs):
            return [cts[0]]

        return op
