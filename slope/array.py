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
from slope.array_shape import ValuedArrayShape

# patch numpy


class Array:
    __array_priority__ = 1000
    default_dtype = np.float32

    def __init__(
        self, val: Union[list, tuple, np.ndarray], dtype: Optional[Any] = None
    ):
        self.val = np.asarray(val, dtype)

    dtype = property(lambda self: self.val.dtype)
    shape = property(lambda self: self.val.shape)
    ndim = property(lambda self: self.val.ndim)

    def __repr__(self):
        return f"{self.__class__.__name__}: {repr(self.val)}"

    def __str__(self):
        return repr(self)
    
    @classmethod
    def full(cls, shape, fill_value, dtype=default_dtype, **kwargs):
        return cls(np.full(shape, fill_value=fill_value, dtype=dtype, **kwargs))

    @classmethod
    def zeros(cls, shape, dtype=default_dtype, **kwargs):
        return cls.full(shape, 0., dtype, **kwargs)

    @classmethod
    def ones(cls, shape, dtype=default_dtype, **kwargs):
        return cls.full(shape, 1., dtype, **kwargs)

    @classmethod
    def zeros_like(cls, **kwargs):
        return cls.zeros(*cls.shape, **kwargs)

    @classmethod
    def empty(cls, *shape, **kwargs):
        return cls.zeros(*shape, **kwargs)

    @classmethod
    def eye(cls, dim, **kwargs):
        return (
            cls(np.eye(dim), **kwargs)
        )

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
            np.maximum,
            np.minimum,
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
    maximum = lambda self, other: np.maximum(self, other)
    minimum = lambda self, other: -self.maximum(-self, -other)
    __neg__ = neg
    __add__ = __radd__ = add
    __sub__ = sub
    __rsub__ = lambda self, other: self.__class__.sub(other, self)
    __mul__ = __rmul__ = mul
    __div__ = div
    __rdiv__ = lambda self, other: self.__class__.div(other, self)
    __truediv__ = __div__
    __truerdiv__ = __rdiv__
    __pow__ = pow
    __eq__ = lambda self, other: self.equal(other)
    __ge__ = lambda self, other: self.maximum(other).equal(self)
    __le__ = lambda self, other: self.minimum(other).equal(self)
    __gt__ = lambda self, other: 1.0 - (self <= other)
    __lt__ = lambda self, other: 1.0 - (self >= other)

    def max(self, axes=None, keepdims=False):
        return self.__class__(np.max(self.val, axis=axes, keepdims=keepdims))

    def sum(self, axes=None, keepdims=False):
        return self.__class__(np.sum(self.val, axis=axes, keepdims=keepdims))

    def mean(self, axes=None, keepdims=False):
        out = self.sum(axes=axes, keepdim=keepdims)
        return out * (math.prod(out.shape) / math.prod(self.shape))

    def min(self, axes=None, keepdims=False):
        return -((-self).max(self, axes, keepdims))

    # Shape
    reshape = lambda self, shape: self.__class__(np.reshape(self.val, shape))
    transpose = lambda self, perm: self.__class__(np.transpose(self.val, perm))
    expand_dims = lambda self, axes: self.__class__(np.expand_dims(self.val, axes))
    swapaxes = lambda self, a1, a2: self.__class__(np.swapaxes(self.val, a1, a2))
    broadcast_to = lambda self, shape: self.__class__(np.broadcast_to(self.val, shape))

    def flatten(self, start_dim=0):
        return self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))

    def broadcast(self, shape, axes=None):
        if axes is not None:
            for a in sorted(axes):
                self = self.expand_dims(self, a)
        return self.broadcast_to(shape)

    # TODO:

    def flip(self, axis, *args):
        return self.__class__(
            np.flip(
                self,
                axis=[x if x >= 0 else x + len(self.shape) for x in (axis, *args)],
            )
        )

    def pad(self, arg: Tuple[Tuple[int, int], ...]):
        return self.__class__(
            np.pad(self, arg=arg) if any(x != (0, 0) for x in arg) else self
        )

    def shrink(self, arg: Tuple[Tuple[int, int], ...]):
        return self.__class__(self.val[arg])

    # NOTE: using slice is discouraged and things should migrate to pad and shrink
    def slice(self, arg: Sequence[Optional[Tuple[int, int]]]):
        arg_ = tuple(a if a is not None else (0, s) for s, a in zip(self.shape, arg))
        padding = tuple(
            (max(0, -p[0]), max(0, p[1] - self.shape[i])) for i, p in enumerate(arg_)
        )
        return self.pad(padding).shrink(
            tuple(
                (p[0] + padding[i][0], p[1] + padding[i][0]) for i, p in enumerate(arg_)
            )
        )

    def __getitem__(self, val):
        def slcfix(i, sz, default):
            return (
                default if i is None else max(0, min(sz, sz + i if i < 0 else i))
            )  # Fix negative idxs, clamp to [0,N]

        new_slice, new_shape = [], []
        val = [val] if not isinstance(val, (list, tuple)) else val
        assert sum(s is not None for s in val) <= len(self.shape)
        assert all(s.step is None or s.step == 1 for s in val if isinstance(s, slice))
        for i, (sz, s) in enumerate(
            zip(self.shape, [v for v in val if v is not None])
        ):  # Slicing only depends on ints + slices
            if isinstance(s, int) and not (-sz <= s < sz):
                raise IndexError(
                    f"index {s} is out of bounds for dimension {i} with size {sz}"
                )
        new_slice.append(
            (s % sz, s % sz + 1)
            if isinstance(s, int)
            else (slcfix(s.start, sz, 0), slcfix(s.stop, sz, sz))
        )
        for s, sz in zip(
            val,
            [
                self.shape[i - 1]
                for i in itertools.accumulate([int(s is not None) for s in val])
            ],
        ):  # Shape depends on slices + positions of Nones
            if not isinstance(s, int):
                new_shape.append(
                    1 if s is None else slcfix(s.stop, sz, sz) - slcfix(s.start, sz, 0)
                )
        new_shape += [self.shape[i] for i in range(len(new_slice), len(self.shape))]
        new_slice += [
            (0, self.shape[i]) for i in range(len(new_slice), len(self.shape))
        ]
        return self.slice(new_slice).reshape(new_shape if len(new_shape) else (1,))

    def concatenate(self, *args, dim=0):
        dim = (dim + len(self.shape)) if dim < 0 else dim
        for y in args:
            assert len(y.shape) == len(self.shape) and all(
                y.shape[i] == s for i, s in enumerate(self.shape) if i != dim
            )
        catargs = [self] + list(args)
        shape_cumsum = [0, *itertools.accumulate([y.shape[dim] for y in catargs])]
        slc = [[(0, s) for s in self.shape] for _ in catargs]
        for s, k in zip(slc, shape_cumsum):
            s[dim] = (-k, shape_cumsum[-1] - k)
        return reduce(
            self.__class__.__add__, [arg.slice(s) for arg, s in zip(catargs, slc)]
        )

    # TODO: make this nicer with syntactic sugar in slice
    def split(self, num, dim):
        slice_params = [[(0, s) for s in self.shape] for _ in range(num)]
        for i, k in enumerate(range(0, self.shape[dim], self.shape[dim] // num)):
            slice_params[i][dim] = (k, min(self.shape[dim], k + self.shape[dim] // num))
        return [self.slice(p) for p in slice_params]

    # (padding_left, padding_right, padding_top, padding_bottom)
    def pad2d(self, padding: Union[List[int], Tuple[int, ...]]):
        return self.slice(
            (
                (0, self.shape[0]),
                (0, self.shape[1]),
                (-padding[2], self.shape[2] + padding[3]),
                (-padding[0], self.shape[3] + padding[1]),
            )
        )

    @property
    def T(self):
        perm = list(range(self.ndim))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        return self.transpose(perm)

    def _softmax(self, axis):
        m = self - self.max(axis=axis, keepdim=True)
        e = m.exp()
        return m, e, e.sum(axis=axis, keepdim=True)

    def softmax(self, axis=-1):
        _, e, ss = self._softmax(axis)
        return e.div(ss)

    def log_softmax(self, axis=-1):
        m, _, ss = self._softmax(axis)
        return m - ss.log()

    def dot(self, w):
        x = self.reshape(*self.shape[0:-1], 1, self.shape[-1])
        w = w.reshape(*w.shape[0:-2], 1, w.shape[-2], w.shape[-1]).T
        return (x * w).sum(-1).reshape(*x.shape[0:-2], -1)

    def sqrt(self):
        return self.pow(0.5)

    def rsqrt(self):
        return self.pow(-0.5)

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

    # # ***** activation functions (unary) *****

    # def relu(self):
    #     return ops.ReLU.do(self)

    def sigmoid(self):
        return (1.0 + (-self).exp()).reciprocal()

    def elu(self, alpha=1.0):
        return self.relu() - alpha * (1 - self.exp()).relu()

    def celu(self, alpha=1.0):
        return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)

    def swish(self):
        return self * self.sigmoid()

    def silu(self):
        return self.swish()  # The SiLU function is also known as the swish function.

    def relu6(self):
        return self.relu() - (self - 6).relu()

    def hardswish(self):
        return self * (self + 3).relu6() * (1 / 6)

    def tanh(self):
        return 2.0 * ((2.0 * self).sigmoid()) - 1.0

    def hardtanh(self, min_val=-1, max_val=1):
        return self.clip(min_val, max_val)

    def gelu(self):
        return (
            0.5
            * self
            * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
        )

    def quick_gelu(self):
        return self * (self * 1.702).sigmoid()

    def leakyrelu(self, neg_slope=0.01):
        return self.relu() - (-neg_slope * self).relu()

    def mish(self):
        return self * self.softplus().tanh()

    def softplus(self, beta=1):
        return (1 / beta) * (1 + (self * beta).exp()).log()

    def softsign(self):
        return self / (1 + self.abs())

    # ***** functional nn ops *****

    def linear(self, weight, bias=None):
        x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
        return x.add(bias) if bias is not None else x

    def serial(self, ll: List[Callable]):
        return reduce(lambda x, f: f(x), ll, self)

    def layernorm(self, axis=-1, eps: float = 1e-5):
        y = self - self.mean(axis, keepdim=True)
        return y.mul((y * y).mean(axis, keepdim=True).add(eps).rsqrt())

    def batchnorm(
        self,
        weight,
        bias,
        mean,
        invstd,
    ):
        x = self - mean.reshape(shape=[1, -1, 1, 1])
        if weight:
            x = x * weight.reshape(shape=[1, -1, 1, 1])
        ret = x.mul(
            invstd.reshape(shape=[1, -1, 1, 1]) if len(invstd.shape) == 1 else invstd
        )
        return (ret + bias.reshape(shape=[1, -1, 1, 1])) if bias else ret
