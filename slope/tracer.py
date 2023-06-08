import math
from contextlib import contextmanager
from slope import utils
import numpy as np
import itertools
from typing import (
    Sequence,
    Callable,
    Tuple,
    List,
    Any,
    List,
    Tuple,
    Optional,
    Any,
    Union,
    Dict,
    Set,
    DefaultDict,
    Callable,
)
import numpy as np
from functools import lru_cache, reduce

import slope
from slope import ops
from slope.array_shape import ValuedArrayShape
from slope.array import Array
from slope.compound_ops import CompoundOpsMixin

# patch numpy


def binaryop_decor(op_fn):
    def wrapped_fn(x, y):
        if type(x) in [int, float]: # TODO: more elegant way handling python numbers
            x = y._trace.pure(x)
        elif type(y) in [int, float]:
            y = x._trace.pure(y)
        bx = list(range((max(x.ndim, y.ndim) - x.ndim)))
        by = list(range((max(x.ndim, y.ndim) - y.ndim)))
        shape_ret = tuple(max(sx, sy) for sx, sy in zip(x.shape, y.shape))
        x = x.broadcast(shape_ret, bx)
        y = y.broadcast(shape_ret, by)
        return op_fn(x, y)

    return wrapped_fn


def reduceop_decor(op_fn):
    def wrapped_fn(x, axes=None, keepdims=False):
        if axes is None:
            axes = tuple(range(x.ndim))
        elif isinstance(axes, int):
            axes = (axes,)
        axes = tuple(a if a >= 0 else a + len(x.shape) for a in axes)
        ret = op_fn(x, axes)

        if keepdims:
            if len(ret.shape) == 0:
                shape = (1,)
            else:
                shape = tuple(1 if i in axes else d for i, d in enumerate(x.shape))
            ret = ret.reshape(shape)
        return ret

    return wrapped_fn


class Tracer(CompoundOpsMixin):
    TYPES = {
        bool,
        int,
        float,
        # np.bool_,
        # np.int32,
        # np.int64,
        # np.float32,
        # np.float64,
        Array
    }
    __array_priority__ = 1000

    default_dtype = np.float32
    _trace: "Trace"

    def __init__(self):
        raise NotImplementedError

    @property
    def aval(self):
        return self.get_aval(self.val)

    dtype = property(lambda self: self.val.dtype)
    shape = property(lambda self: self.val.shape)
    ndim = property(lambda self: self.val.ndim)

    def full_lower(self):
        return self.val

    # def __repr__(self):
    #     return f"{self.__class__.__name__}: {repr(self.val)}"

    # def __str__(self):
    #     return repr(self)

    @property
    def ndim(self):
        return len(self.shape)

    # UnaryOps

    def identity(x):
        return ops.Identity.do(x)

    def stop_gradient(x):
        return ops.StopGradient.do(x)

    def convert(x, dtype):
        return ops.Convert.do(x, dtype=dtype)

    def exp(x):
        return ops.Exp.do(x)

    def log(x):
        return ops.Log.do(x)

    def neg(x):
        return ops.Neg.do(x)

    @binaryop_decor
    def add(self, other):
        return slope.RT.bind1(ops.Add, self, other)

    @binaryop_decor
    def sub(self, other):
        return slope.RT.bind1(ops.Sub, self, other)

    @binaryop_decor
    def mul(self, other):
        return ops.Mul.do(self, other)

    @binaryop_decor
    def div(self, other):
        return slope.RT.bind1(ops.Div, self, other)

    def pow(self, y):
        assert type(y) is int
        if y == 0:
            return self.ones_like(x)
        is_reciprocal = y < 0
        if is_reciprocal:
            y = -y
        acc = None
        while y > 0:
            if y & 1:
                acc = x if acc is None else acc * x
            y >>= 1
            if y > 0:
                x = x * x
        ret = acc
        if is_reciprocal:
            ret = self.ones_like(acc) / acc
        return ret

    @binaryop_decor
    def equal(self, other):
        return ops.Equal.do(self, other)

    @binaryop_decor
    def maximum(self, other):
        return ops.Maximum.do(self, other)

    @binaryop_decor
    def mininum(self, other):
        return -(-self.maximum(-other))

    def __neg__(self):
        return self.neg()

    def __add__(self, other):
        return self.add(other)

    def __radd__(self, other):
        return self.__class__.add(other, self)

    def __sub__(self, other):
        return self.sub(other)

    def __rsub__(self, other):
        return self.__class__.sub(other, self)

    def __mul__(self, other):
        return self.mul(other)

    def __rmul__(self, other):
        return self.__class__.mul(other, self)

    def __div__(self, other):
        return self.div(other)

    def __rdiv__(self, other):
        return self.__class__.div(other, self)

    def __truediv__(self, other):
        return self.div(other)

    def __rtruediv__(self, other):
        return self.__class__.div(other, self)

    def __pow__(self, other):
        return self.pow(other)

    def __bool__(self):
        return self.aval._bool(self)

    def __nonzero__(self):
        return self.aval._nonzero(self)

    def __ge__(self, x):
        return self.maximum(x).equal(self)

    def __le__(self, x):
        return self.maximum(x).equal(x)

    def __lt__(self, x):
        return 1.0 - (self >= x)

    def __gt__(self, x):
        return 1.0 - (self <= x)

    def __eq__(self, x):
        return self.equal(x)

    @reduceop_decor
    def sum(self, axes=None, keepdims=False):
        return slope.RT.bind1(ops.Sum, self, axes=axes, keepdims=keepdims)

    @reduceop_decor
    def max(self, axes=None, keepdim=False):
        return slope.RT.bind1(ops.Max, self, axes=axes)

    # Shape
    def broadcast(self, shape, axes=None):
        if axes is None:
            axes = tuple(range(self.ndim))
        elif isinstance(axes, int):
            axes = (axes,)
        axes = tuple(a if a >= 0 else a + len(self.shape) for a in axes)
        return ops.Broadcast.do(self, shape=shape, axes=axes)

    def reshape(self, shape):
        if -1 in shape:
            shape_ = shape
            others = math.prod([d for d in shape if d != -1])
            numel = math.prod(self.shape)
            shape = tuple(d if d != -1 else (numel // others) for d in shape)
        return ops.Reshape.do(self, shape=shape)

    def transpose(self, perm):
        return ops.Transpose.do(self, perm=perm)
    @staticmethod
    def get_aval(x):
        if isinstance(x, Tracer):
            return x.aval
        elif type(x) in Tracer.TYPES:
            # return ValuedArrayShape(np.asarray(x))
            return Array(np.asarray(x))
        else:
            breakpoint()
            raise TypeError(x)

    def full_lower(self):
        return self

    def __getattr__(self, name):
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
