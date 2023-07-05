import math
from contextlib import contextmanager
import numpy as np
import itertools
from typing import (
    Any,
    Optional,
    Union,
)
import numpy as np
import functools
import slope
from slope import utils
from slope.array_shape import ValuedArrayShape
from slope.base_array import BaseArray
from slope.dtypes import dtypes
import numpy as np

# import ops


class Array(BaseArray):
    __array_priority__ = 2000
    default_dtype = dtypes.float32

    def __init__(
        self, val: Union[list, tuple, np.ndarray], dtype: Optional[Any] = None
    ):
        self.val = np.asarray(val, dtype)

    dtype = property(lambda self: self.val.dtype)
    shape = property(lambda self: self.val.shape)
    ndim = property(lambda self: self.val.ndim)

    def __repr__(self):
        return f"{self.__class__.__name__}: {repr(self.val)[6:-1]}"

    __str__ = __repr__

    @staticmethod
    def full(shape, fill_value, dtype=default_dtype, **kwargs):
        return slope.RT.backend.full(
            shape, fill_value=fill_value, dtype=dtype, **kwargs
        )

    @staticmethod
    def zeros(shape, dtype=default_dtype, **kwargs):
        return slope.RT.backend.full(shape, 0.0, dtype, **kwargs)

    @staticmethod
    def ones(shape, dtype=default_dtype, **kwargs):
        return slope.RT.backend.full(shape, 1.0, dtype, **kwargs)

    @staticmethod
    def full_like(other, fill_value, **kwargs):
        return slope.RT.backend.full(
            other.shape, fill_value, dtype=other.dtype, **kwargs
        )

    @staticmethod
    def zeros_like(other, **kwargs):
        return slope.RT.backend.zeros(other.shape, dtype=other.dtype, **kwargs)

    @staticmethod
    def ones_like(other, **kwargs):
        return slope.RT.backend.ones(other.shape, dtype=other.dtype, **kwargs)

    @staticmethod
    def empty(*shape, **kwargs):
        return slope.RT.backend.zeros(*shape, **kwargs)

    @staticmethod
    def eye(dim, **kwargs):
        return slope.RT.backend.eye(dim, **kwargs)

    @staticmethod
    def arange(stop, start=0, step=1, **kwargs):
        return slope.RT.backend.arange(start=start, stop=stop, step=step, **kwargs)

    # TODO: distill RNG code from jax

    _rng: np.random.Generator = np.random.default_rng()

    @staticmethod
    def manual_seed(seed=None):
        slope.RT.backend._rng = np.random.default_rng(seed=seed)

    @staticmethod
    def rand(*shape, **kwargs):
        return slope.RT.backend.rand(
            size=shape,
            dtype=kwargs.get("dtype", slope.RT.backend.default_dtype),
            **kwargs,
        )

    @staticmethod
    def randn(*shape, **kwargs):
        return slope.RT.backend.randn(
            size=shape,
            dtype=kwargs.get("dtype", slope.RT.backend.default_dtype) ** kwargs,
        )

    @staticmethod
    def uniform(*shape, **kwargs):
        return slope.RT.backend.rand(*shape, **kwargs) * 2 - 1

    @staticmethod
    def scaled_uniform(*shape, **kwargs):
        return slope.RT.backend.uniform(*shape, **kwargs).mul(math.prod(shape) ** -0.5)

    @staticmethod
    def glorot_uniform(*shape, **kwargs):
        return slope.RT.backend.uniform(*shape, **kwargs).mul(
            (6 / (shape[0] + math.prod(shape[1:]))) ** 0.5
        )

    def stop_gradient(self):
        return self.zeros_like(self)

    def __array__(self, dtype=None):
        return self.val

    convert = lambda self, dtype: self.__class__(self.val, dtype=dtype)
    astype = convert
    neg = lambda self: slope.RT.backend.neg(self)
    exp = lambda self: slope.RT.backend.exp(self)
    log = lambda self: slope.RT.backend.log(self)
    add = lambda self, other: slope.RT.backend.add(self, other)
    sub = lambda self, other: slope.RT.backend.subtract(self, other)
    mul = lambda self, other: slope.RT.backend.multiply(self, other)
    div = lambda self, other: slope.RT.backend.divide(self, other)
    equal = lambda self, other: slope.RT.backend.equal(self, other)
    not_equal = lambda self, other: slope.RT.backend.not_equal(self, other)
    maximum = lambda self, other: slope.RT.backend.maximum(self, other)

    def max(self, axes=None, keepdims=False):
        return slope.RT.backend.max(self.val, axis=axes, keepdims=keepdims)

    def sum(self, axes=None, keepdims=False):
        return slope.RT.backend.sum(self.val, axis=axes, keepdims=keepdims)

    # Shape
    reshape = lambda self, shape: slope.RT.backend.reshape(self.val, shape)
    transpose = lambda self, perm: slope.RT.backend.transpose(self.val, perm)
    expand_dims = lambda self, axes: slope.RT.backend.expand_dims(self.val, axes)
    swapaxes = lambda self, a1, a2: slope.RT.backend.swapaxes(self.val, a1, a2)
    broadcast_to = lambda self, shape: slope.RT.backend.broadcast_to(self.val, shape)

    def broadcast(self, shape, axes=None):
        if axes is not None:
            for a in sorted(axes):
                self = self.expand_dims(a)
        return self.broadcast_to(shape)

    def pad(self, lo, hi, interior=None, value=0):
        if interior is None:
            interior = [1] * len(lo)
        new_shape, slices = [], []
        for s, l, h, r in zip(self.shape, lo, hi, interior):
            stride = r + 1
            new_shape += [s * stride + l + h]
            slices += [slice(l, s * stride + l, stride)]
        padded = slope.RT.backend.full(new_shape, value, dtype=self.dtype)
        padded[tuple(slices)] = self.val
        return self.__class__(padded)

    def slice(self, starts, limits, strides):
        return self.__class__(
            self.val[tuple(slice(s, l, r) for s, l, r in zip(starts, limits, strides))]
        )

    def __getitem__(self, idx):
        if type(idx) in (tuple, list):
            return self.slice(slice(idx))
        raise NotImplementedError

    def __getitem__(self, idx, val):
        raise NotImplementedError

    # control flow
    choose = select = lambda self, *vals, idx: slope.RT.backendchoose(idx, *vals)
    where = lambda self, trueval, falseval: slope.RT.backendwhere(
        self, trueval, falseval
    )
