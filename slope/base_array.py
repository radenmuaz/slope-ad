import math
from contextlib import contextmanager
from typing import (
    Callable,
    Tuple,
    List,
    Any,
    List,
    Tuple,
    Optional,
    Any,
    Union,
    Callable,
    NamedTuple,
    Final,
)
import functools
import slope
from slope import utils
import numpy as np
from abc import ABC, abstractmethod


class DType(NamedTuple):
    priority: int
    itemsize: int
    name: str
    np: type

    def __repr__(self):
        return f"dtypes.{self.name}"


class BaseArray:
    bool: Final[DType] = DType(0, 1, "bool", bool)
    float16: Final[DType] = DType(0, 2, "half", np.float16)
    float32: Final[DType] = DType(4, 4, "float", np.float32)
    int8: Final[DType] = DType(0, 1, "char", np.int8)
    int32: Final[DType] = DType(1, 4, "int", np.int32)
    int64: Final[DType] = DType(2, 8, "int64", np.int64)
    uint8: Final[DType] = DType(0, 1, "uchar", np.uint8)
    default_dtype = float32

    # def is_int(x: DType) -> bool:
    #     return x in (int8, uint8, int32, int64)

    # def is_float(x: DType) -> bool:
    #     return x in (float16, float32)

    # def is_unsigned(x: DType) -> bool:
    #     return x in (uint8)
    def notimplemented(self, *args, **kwargs):
        raise

    neg = notimplemented
    add = notimplemented
    sub = notimplemented
    mul = notimplemented
    div = notimplemented
    equal = notimplemented
    not_equal = notimplemented
    maximum = notimplemented

    __neg__ = lambda self: self.neg()
    __add__ = lambda self, other: self.add(other)
    __radd__ = lambda self, other: self.__class__.add(other, self)
    __sub__ = lambda self, other: self.sub(other)
    __rsub__ = lambda self, other: self.__class__.sub(other, self)
    __mul__ = __rmul__ = mul
    __div__ = div
    __rdiv__ = lambda self, other: self.__class__.div(other, self)
    __truediv__ = __div__
    __truerdiv__ = __rdiv__
    __eq__ = lambda self, other: self.equal(other)
    __ne__ = lambda self, other: self.not_equal(other)
    __ge__ = lambda self, other: self.maximum(other).equal(self)
    __le__ = lambda self, other: self.minimum(other).equal(self)
    __gt__ = lambda self, other: 1.0 - (self <= other)
    __lt__ = lambda self, other: 1.0 - (self >= other)

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
        return cls.full(other.shape, fill_value=1.0, dtype=other.dtype, **kwargs)

    def where(self, trueval, falseval):
        cond = self != 0.0
        cond = cond.convert(trueval.dtype)  # TODO: type promotion logic
        return cond * trueval + (1.0 - cond) * falseval

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

    def cross_entropy(x, y):
        return x * y.log()

    def mse(x, y):
        return pow((x - y), 2)

    def mean(self, axes=None, keepdims=False):
        out = self.sum(axes=axes, keepdim=keepdims)
        return out * (math.prod(out.shape) / math.prod(self.shape))

    def minimum(self, other):
        return -self.maximum(-self, -other)

    def min(self, axes=None, keepdims=False):
        return -((-self).max(self, axes, keepdims))

    def flatten(self, start_dim=0):
        return self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))

    @classmethod
    def glorot_uniform(cls, *shape, **kwargs):
        return cls.rand(*shape, **kwargs).mul(
            (6 / (shape[0] + math.prod(shape[1:]))) ** 0.5
        )

    @property
    def T(self):
        perm = list(range(self.ndim))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        return self.transpose(perm)

    def _softmax(self, axes):
        m = self - self.max(axes, keepdims=True)
        e = m.exp()
        return m, e, e.sum(axes, keepdims=True)

    def softmax(self, axes=-1):
        _, e, ss = self._softmax(axes)
        return e.div(ss)

    def log_softmax(self, axes=-1):
        m, _, ss = self._softmax(axes)
        return m - ss.log()

    def dot(self, w):
        x = self.reshape((*self.shape[0:-1], 1, self.shape[-1]))
        w = w.reshape((*w.shape[0:-2], 1, w.shape[-2], w.shape[-1])).T
        return (x * w).sum(-1).reshape((*x.shape[0:-2], -1))
    
    def square(self):
        return self * self

    def clip(self, min_, max_):
        return ((self - min_).relu() + min_) - (self - max_).relu()

    def abs(self):
        return self.relu() + (-self).relu()

    def sign(self):
        return self / (self.abs() + 1e-10)

    def reciprocal(self):
        return 1.0 / self



class ArrayBuffer:
    def __init__(self, val):
        self.val = val


class Array(BaseArray):
    __array_priority__ = 2000
    default_dtype = BaseArray.float32

    def __init__(
        self,
        val: Union[list, tuple, np.ndarray, ArrayBuffer] = None,
        dtype: Optional[Any] = None,
    ):
        # Decides whether point existing buffer or create new buffer
        self.buf = (
            val
            if isinstance(val, ArrayBuffer)
            else slope.backend.constant(val=val, dtype=dtype).buf
        )

    val = property(lambda self: self.buf.val)
    dtype = property(lambda self: self.buf.val.dtype)
    shape = property(lambda self: self.buf.val.shape)
    ndim = property(lambda self: self.buf.val.ndim)

    def __repr__(self):
        return f"{self.__class__.__name__}: {repr(self.val)[6:-1]}"

    __str__ = __repr__

    @classmethod
    def full(cls, shape, fill_value, dtype=default_dtype, **kwargs):
        return slope.RT.backend.full(
            shape, fill_value=fill_value, dtype=dtype, **kwargs
        )

    def __array__(self, dtype=None):
        return self.val

    convert = lambda self, dtype: slope.RT.backend.convert(self, dtype=dtype)
    astype = convert
    stop_gradient = lambda self: slope.RT.backend.stop_gradient(self)
    sqrt = lambda self: slope.RT.backend.sqrt(self)
    neg = lambda self: slope.RT.backend.neg(self)
    exp = lambda self: slope.RT.backend.exp(self)
    log = lambda self: slope.RT.backend.log(self)
    sin = lambda self: slope.RT.backend.sin(self)

    add = lambda self, other: slope.RT.backend.add(self, other)
    sub = lambda self, other: slope.RT.backend.subtract(self, other)
    mul = lambda self, other: slope.RT.backend.multiply(self, other)
    div = lambda self, other: slope.RT.backend.divide(self, other)
    equal = lambda self, other: slope.RT.backend.equal(self, other)
    not_equal = lambda self, other: slope.RT.backend.not_equal(self, other)
    maximum = lambda self, other: slope.RT.backend.maximum(self, other)
    max = lambda self, axes=None, keepdims=False: slope.RT.backend.max(
        self.val, axis=axes, keepdims=keepdims
    )
    sum = lambda self, axes=None, keepdims=False: slope.RT.backend.sum(
        self.val, axis=axes, keepdims=keepdims
    )

    constant = lambda self, val, dtype: slope.RT.backend.constant(
        self, val=val, dtype=dtype
    )
    full = lambda self, val, shape, dtype: slope.RT.backend.full(
        self, val=val, shape=shape, dtype=dtype
    )
    arange = lambda self, start, stop, stride, dtype: slope.RT.backend.arange(
        self, start, stop, stride, dtype=dtype
    )
    random_normal = lambda self, shape, dtype: slope.RT.backend.random_normal(
        self, shape, dtype=dtype
    )
    randn = random_normal
    random_uniform = lambda self, shape, dtype: slope.RT.backend.random_uniform(
        self, shape, dtype=dtype
    )
    rand = random_uniform

    # Shape
    reshape = lambda self, shape: slope.RT.backend.reshape(self, shape)
    transpose = lambda self, perm: slope.RT.backend.transpose(self, perm)
    broadcast = lambda self, shape, axes: slope.RT.backend.broadcast(self, shape, axes)
    pad = lambda self, lo, hi, interior, value: slope.RT.backend.pad(
        self, lo, hi, interior, value
    )
    slice = lambda self, starts, limits, strides: slope.RT.backend.slice(
        self, starts, limits, strides
    )
    flip = lambda self, axes: slope.RT.backend.flip(self, axes)
    concatenate = classmethod(
        lambda cls, xs, axes: slope.RT.backend.concatenate(xs, axes)
    )

    def __getitem__(self, idx):
        if type(idx) in (tuple, list):
            return self.slice(slice(idx))
        raise NotImplementedError

    def __setitem__(self, idx, val):
        raise NotImplementedError

    # control flow
    choose = select = lambda self, *vals, idx: slope.RT.backend.choose(idx, *vals)
    where = lambda self, trueval, falseval: slope.RT.backend.where(
        self, trueval, falseval
    )
