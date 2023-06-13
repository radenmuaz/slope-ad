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

import numpy as np


class Array(BaseArray):
    __array_priority__ = 2000
    default_dtype = np.float32

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

    @classmethod
    def full(cls, shape, fill_value, dtype=default_dtype, **kwargs):
        return cls(np.full(shape, fill_value=fill_value, dtype=dtype, **kwargs))

    @classmethod
    def zeros(cls, shape, dtype=default_dtype, **kwargs):
        return cls.full(shape, 0.0, dtype, **kwargs)

    @classmethod
    def ones(cls, shape, dtype=default_dtype, **kwargs):
        return cls.full(shape, 1.0, dtype, **kwargs)

    @classmethod
    def full_like(cls, other, fill_value, **kwargs):
        return cls.full(other.shape, fill_value, dtype=other.dtype, **kwargs)

    @classmethod
    def zeros_like(cls, other, **kwargs):
        return cls.zeros(other.shape, dtype=other.dtype, **kwargs)

    @classmethod
    def ones_like(cls, other, **kwargs):
        return cls.ones(other.shape, dtype=other.dtype, **kwargs)

    @classmethod
    def empty(cls, *shape, **kwargs):
        return cls.zeros(*shape, **kwargs)

    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim), **kwargs)

    @classmethod
    def arange(cls, stop, start=0, step=1, **kwargs):
        return cls(
            np.arange(start=start, stop=stop, step=step, dtype=np.float32), **kwargs
        )

    # TODO: distill RNG code from jax

    _rng: np.random.Generator = np.random.default_rng()

    @classmethod
    def manual_seed(cls, seed=None):
        cls._rng = np.random.default_rng(seed=seed)

    @classmethod
    def rand(cls, *shape, **kwargs):
        return cls(
            np.array(
                cls._rng.random(
                    size=shape, dtype=kwargs.get("dtype", cls.default_dtype)
                ),
            ),
            **kwargs,
        )

    @classmethod
    def randn(cls, *shape, **kwargs):
        return cls(
            np.array(
                cls._rng.standard_normal(
                    size=shape, dtype=kwargs.get("dtype", cls.default_dtype)
                ),
            ),
            **kwargs,
        )

    @classmethod
    def uniform(cls, *shape, **kwargs):
        return cls.rand(*shape, **kwargs) * 2 - 1

    @classmethod
    def scaled_uniform(cls, *shape, **kwargs):
        return cls.uniform(*shape, **kwargs).mul(math.prod(shape) ** -0.5)

    @classmethod
    def glorot_uniform(cls, *shape, **kwargs):
        return cls.uniform(*shape, **kwargs).mul(
            (6 / (shape[0] + math.prod(shape[1:]))) ** 0.5
        )

    def stop_gradient(self):
        return self.zeros_like(self)

    def __array__(self, dtype=None):
        return self.val

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        assert method == "__call__"
        assert ufunc in [
            np.negative,
            np.add,
            np.subtract,
            np.multiply,
            np.divide,
            np.exp,
            np.log,
            np.equal,
            np.not_equal,
            np.maximum,
        ]
        inputs = [i.val if type(i) is self.__class__ else i for i in inputs]
        ret = ufunc(*inputs, **kwargs)
        return self.__class__(ret)

    convert = lambda self, dtype: self.__class__(self.val, dtype=dtype)
    astype = convert
    neg = lambda self: np.negative(self)
    exp = lambda self: np.exp(self)
    log = lambda self: np.log(self)
    add = lambda self, other: np.add(self, other)
    sub = lambda self, other: np.subtract(self, other)
    mul = lambda self, other: np.multiply(self, other)
    div = lambda self, other: np.divide(self, other)
    pow = lambda self, other: np.power(self, other)
    equal = lambda self, other: np.equal(self, other)
    not_equal = lambda self, other: np.not_equal(self, other)
    maximum = lambda self, other: np.maximum(self, other)
    __neg__ = neg
    __add__ = add
    __radd__ = lambda self, other: self.__class__.add(other, self)
    __sub__ = sub
    __rsub__ = lambda self, other: self.__class__.sub(other, self)
    __mul__ = __rmul__ = mul
    __div__ = div
    __rdiv__ = lambda self, other: self.__class__.div(other, self)
    __truediv__ = __div__
    __truerdiv__ = __rdiv__
    __pow__ = pow
    __eq__ = lambda self, other: self.equal(other)
    __ne__ = lambda self, other: self.not_equal(other)
    __ge__ = lambda self, other: self.maximum(other).equal(self)
    __le__ = lambda self, other: self.minimum(other).equal(self)
    __gt__ = lambda self, other: 1.0 - (self <= other)
    __lt__ = lambda self, other: 1.0 - (self >= other)

    def max(self, axes=None, keepdims=False):
        return self.__class__(np.max(self.val, axis=axes, keepdims=keepdims))

    def sum(self, axes=None, keepdims=False):
        return self.__class__(np.sum(self.val, axis=axes, keepdims=keepdims))

    # Shape
    reshape = lambda self, shape: self.__class__(np.reshape(self.val, shape))
    transpose = lambda self, perm: self.__class__(np.transpose(self.val, perm))
    expand_dims = lambda self, axes: self.__class__(np.expand_dims(self.val, axes))
    swapaxes = lambda self, a1, a2: self.__class__(np.swapaxes(self.val, a1, a2))
    broadcast_to = lambda self, shape: self.__class__(np.broadcast_to(self.val, shape))
    def broadcast(self, shape, axes=None):
        if axes is not None:
            for a in sorted(axes):
                self = self.expand_dims(a)
        return (self.broadcast_to(shape))

    __getitem__ = lambda self, idx: self.__class__(self.val.__getitem__(idx))
    __setitem__ = lambda self, idx, val: self.__class__(self.val.__setitem__(idx, val))
    gather = lambda self, idx, axis: self.__class__(
        np.take_along_axis(self.val, idx, axis)
    )
    choose = select = lambda self, *vals, idx: self.__class__(np.choose(idx, *vals))

    def scatter(operand, indices, updates, shape):
        target = np.zeros(shape, dtype=updates.dtype)
        indices = tuple(indices.reshape(-1, indices.shape[-1]).T)
        updates = updates.ravel()
        np.add.at(target, indices, updates)
        return target

    where = lambda self, trueval, falseval: self.__class__(
        np.where(self, trueval, falseval)
    )

    # slice = lambda self, start, end, step: self.__class__(self.val.__getitem__(slice(start, end, step)))
    # def broadcast_to(self, shape):
    #     return self.__class__(np.broadcast_to(self.val, shape))
