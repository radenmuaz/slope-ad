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

# patch numpy


def binaryop_decor(op_fn):
    def wrapped_fn(x, y):
        # x = Array(x) if type(x) in (int, float, bool) else x
        # y = Array(y) if type(y) in (int, float, bool) else x
        bx = list(range((max(x.ndim, y.ndim) - x.ndim)))
        by = list(range((max(x.ndim, y.ndim) - y.ndim)))
        shape_ret = tuple(max(sx, sy) for sx, sy in zip(x.shape, y.shape))
        x = x.broadcast(shape_ret, bx)
        y = y.broadcast(shape_ret, by)
        return op_fn(x, y)

    return wrapped_fn


def reduceop_decor(op_fn):
    def wrapped_fn(x, axes, keepdim=False):
        if axes is None:
            axes = tuple(range(x.ndim))
        elif isinstance(axes, int):
            axes = (axes,)
        axes = tuple(a if a >= 0 else a + len(x.shape) for a in axes)
        ret = op_fn(x, axes)

        if keepdim:
            if len(ret.shape) == 0:
                shape = (1,)
            else:
                shape = tuple(1 if i in axes else d for i, d in enumerate(x.shape))
            ret = ret.reshape(shape)
        return ret

    return wrapped_fn


class Tracer:
    TYPES = {
        bool,
        int,
        float,
        np.bool_,
        np.int32,
        np.int64,
        np.float32,
        np.float64,
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
    def sum(self, axes=None, keepdim=False):
        return slope.RT.bind1(ops.Sum, self, axes=axes)

    @reduceop_decor
    def max(self, axes=None, keepdim=False):
        return slope.RT.bind1(ops.Max, self, axes=axes)

    def mean(self, axes=None, keepdim=False):
        out = self.sum(axes=axes, keepdim=keepdim)
        return out * (math.prod(out.shape) / math.prod(self.shape))

    def min(self, axes=None, keepdim=False):
        return -((-self).max(self, axes, keepdim))

    # Shape
    def broadcast(self, shape, axes=None):
        if axes is None:
            axes = tuple(range(self.ndim))
        elif isinstance(axes, int):
            axes = (axes,)
        axes = tuple(a if a >= 0 else a + len(self.shape) for a in axes)
        return ops.Broadcast.do(self, shape=shape, axes=axes)

    def reshape(self, shape):
        return ops.Reshape.do(self, shape=shape)

    def transpose(self, perm):
        return ops.Transpose.do(self, perm=perm)

    def swapaxes(self, ax1=1, ax2=0):
        order = list(range(self.ndim))
        order[ax1], order[ax2] = order[ax2], order[ax1]
        return self.transpose(order)

    def flatten(self, start_dim=0):
        return self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))

    def expand_dims(self, axis):
        shape = list(self.shape)
        for a in axis:
            if a < 0:
                a = len(shape) + (a + 1)
            shape.insert(a, 1)
        return self.reshape(shape)

    # TODO:

    def flip(self, axis, *args):
        return ops.Flip.do(
            self,
            axis=[
                x if x >= 0 else x + len(self.shape) for x in utils.argfix(axis, *args)
            ],
        )

    def pad(self, arg: Tuple[Tuple[int, int], ...]):
        return ops.Pad.do(self, arg=arg) if any(x != (0, 0) for x in arg) else self

    def shrink(self, arg: Tuple[Tuple[int, int], ...]):
        return (
            ops.Shrink.do(self, arg=arg)
            if any(x != (0, s) for x, s in zip(arg, self.shape))
            else self
        )

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

    def relu(self):
        return ops.ReLU.do(self)

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

    @staticmethod
    def get_aval(x):
        if isinstance(x, Tracer):
            return x.aval
        elif type(x) in Tracer.TYPES:
            return ValuedArrayShape(np.asarray(x))
        else:
            raise TypeError(x)

    def full_lower(self):
        return self

    def __getattr__(self, name):
        try:
            return getattr(self.aval, name)
        except AttributeError:
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
