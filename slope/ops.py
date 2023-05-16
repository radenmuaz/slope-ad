import slope
import numpy as np
from slope.array_shape import ArrayShape
from slope.array import Array
from typing import List, Tuple, Sequence, Any, Callable, NamedTuple
from abc import ABC, abstractmethod
import math


class Op(ABC):
    @classmethod
    def do(cls, *args, **params):
        return slope.RT.bind1(cls, *args, **params)

    @staticmethod
    @abstractmethod
    def eval(*args, **params):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def vmap(*args, **params):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def jvp(*args, **params):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def shape_eval(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pprint(cls):
        return None

    @staticmethod
    @abstractmethod
    def stablehlo(*args, **params):
        raise NotImplementedError


class UnaryOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in, **params):
        (x,), (x_bdim,) = vals_in, dims_in
        return [cls.do(x, **params)], [x_bdim]

    @staticmethod
    def shape_eval(x: ArrayShape, **params) -> List[ArrayShape]:
        return [ArrayShape(x.shape, x.dtype)]

    @classmethod
    def identity_jvp(cls, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [cls.do(x, **params)], [cls.do(x_dot, **params)]

    # @classmethod
    # def identity_T(cls, t, x):
    #     (z,) = t
    #     assert type(x) is slope.ad.UndefPrimal
    #     return [cls.do(z)]

    # @classmethod
    # def zero_T(cls, t, x):
    #     (z,) = t
    #     assert type(x) is slope.ad.UndefPrimal
    #     return [z.zeros_like()]


class BinaryOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in, **params):
        (x, y), (x_bdim, y_bdim) = vals_in, dims_in
        if x_bdim != y_bdim:
            if x_bdim is None:
                x = slope.ad.move_batch_axis(axis_size, x_bdim, y_bdim, x)
                x_bdim = y_bdim
            else:
                y = slope.ad.move_batch_axis(axis_size, y_bdim, x_bdim, y)
        return [cls.do(x, y, **params)], [x_bdim]

    @staticmethod
    def shape_eval(x: ArrayShape, y: ArrayShape, **params) -> List[ArrayShape]:
        if not isinstance(x, ArrayShape) or not isinstance(y, ArrayShape):
            raise TypeError
        if ArrayShape.like(x) != ArrayShape.like(y):
            raise TypeError(f"{x} != {y}")
        return [ArrayShape(x.shape, x.dtype)]


class ReduceOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in, **params):
        (x,), (x_bdim,) = vals_in, dims_in
        axes = list(params["axes"])
        axes = tuple(a + (x_bdim <= a) for a in axes)
        out_bdim = x_bdim - sum(a < x_bdim for a in axes)
        params["axes"] = tuple(axes)
        return [cls.do(x, **params)], [out_bdim]

    @staticmethod
    def shape_eval(x: ArrayShape, **params) -> List[ArrayShape]:
        axes = params["axes"]
        axes = [a + len(x.shape) if a < 0 else a for a in axes]
        axes_ = set(axes)
        new_shape = [d for i, d in enumerate(x.shape) if i not in axes_]
        return [ArrayShape(tuple(new_shape), x.dtype)]


class ShapeOp(Op):
    pass


# -----------------------
# UnaryOps
# -----------------------


class Identity(UnaryOp):
    @staticmethod
    def eval(x):
        return [x]

    @staticmethod
    def jvp(cls, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [identity(x, **params)], [identity(x_dot, **params)]

    @staticmethod
    def T(t, x):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [identity(z)]


# class FullLike(UnaryOp):
#     @staticmethod
#     def eval(x, *, fill_value):
#         return [np.full(x.shape, fill_value=fill_value, dtype=x.dtype)]

#     @staticmethod
#     def jvp(cls, primals, tangents, *, fill_value):
#         (x,), (x_dot,) = primals, tangents
#         return [full_like(x, fill_value)], [zeros_like(x_dot)]

#     @staticmethod
#     def T(t, x, *, fill_value):
#         (z,) = t
#         assert type(x) is slope.ad.UndefPrimal
        # return [zeros_like(z)]


class StopGradient(UnaryOp):
    @staticmethod
    def eval(x):
        return [identity(x)]

    @staticmethod
    def jvp(primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [identity(x, **params)], [zeros_like(x)]

    @staticmethod
    def T(t, x):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [zeros_like(z)]


class Convert(UnaryOp):
    @staticmethod
    def eval(x, *, dtype):
        return [x.astype(dtype)]

    @staticmethod
    def jvp(primals, tangents, *, dtype):
        (x,), (x_dot,) = primals, tangents
        return [convert(x, dtype)], [convert(x_dot, dtype)]

    @staticmethod
    def T(t, x):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [convert(z, x.dtype)]


class Exp(UnaryOp):
    @staticmethod
    def eval(x):
        return [np.exp(x)]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [exp(x)], [x_dot * exp(x)]


class Log(UnaryOp):
    @staticmethod
    def eval(x):
        return [np.log(x)]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [log(x)], [x_dot / x]


class Neg(UnaryOp):
    @staticmethod
    def eval(x):
        return [-x]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [-x], [-x_dot]

    @staticmethod
    def T(t, x):
        (z,) = t
        return [-z]


# -----------------------
# BinaryOps
# -----------------------


class Add(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x + y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x + y], [x_dot + y_dot]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar, z_bar]


class Sub(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x - y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x - y], [x_dot - y_dot]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar, -z_bar]


class Mul(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x * y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        eval_out = x * y
        jvp_out = (x_dot * y) + (y_dot * x)
        # jvp_out = (y * x_dot) + (y_dot * x) # order problem, x*y_dot fails
        # check about __array_priority
        return [eval_out], [jvp_out]
    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        assert (type(x) is slope.ad.UndefPrimal) ^ (type(y) is slope.ad.UndefPrimal)
        if type(x) is slope.ad.UndefPrimal:
            return [z_bar * y, None]
        elif type(y) is slope.ad.UndefPrimal:
            return [None, x * z_bar]


class Div(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x / y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x / y], [
            (x_dot / y) + (-y_dot * x * (y**-2))
        ]  # bug: power returns float64

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar / y, None]


class Maximum(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [np.maximum(x, y)]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), _ = primals, tangents
        out_primal = max(x, y)
        return [out_primal], [zeros_like(out_primal)]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar, None]


class Equal(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [np.equal(x, y)]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), _ = primals, tangents
        out_primal = equal(x, y)
        return [out_primal], [np.zeros(out_primal.shape, out_primal.dtype)]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar, None]


# max_p: core.Primitive = standard_naryop([_any, _any], 'max')
# ad.defjvp2(max_p,
#            lambda g, ans, x, y: mul(g, _balanced_eq(x, ans, y)),
#            lambda g, ans, x, y: mul(g, _balanced_eq(y, ans, x)))
# mlir.register_lowering(max_p, partial(_nary_lower_hlo, mlir.max_hlo))


# def _balanced_eq(x, z, y):
#   return div(select(_eq_meet(x, z), _ones(z), _zeros(z)),
#              select(_eq_meet(y, z), _twos(z), _ones(z)))


# def _eq_meet(a, b):
#   a_dtype, b_dtype = _dtype(a), _dtype(b)
#   if a_dtype != b_dtype:
#     higher_dtype = dtypes.promote_types(a_dtype, b_dtype)
#     if higher_dtype == a_dtype:
#       a = convert_element_type(a, b_dtype)
#     else:
#       b = convert_element_type(b, a_dtype)
#   return eq(a, b)
# -----------------------
# ReduceOps
# -----------------------


class Max(ReduceOp):
    @staticmethod
    def eval(x, axes):
        return [x.max(axes)]

    @staticmethod
    def jvp(primals, tangents, axes):
        (x,), (x_dot,) = primals, tangents
        eval_out = max(x, axes)
        locs = equal(x, broadcast(eval_out, x.shape, axes))
        # locs = equal(x, eval_out)
        locs = convert(locs, x_dot.dtype)
        counts = sum(locs, axes)
        jvp_out = sum(x_dot * locs, axes)
        jvp_out = jvp_out / broadcast(counts, jvp_out.shape)

        return [eval_out], [jvp_out]

    @staticmethod
    def T(cts, x, *, axes):
        (z,) = cts
        return [broadcast(z, x.aval.shape, ())]


class Sum(ReduceOp):
    @staticmethod
    def eval(x, *, axes):
        return [x.sum(axes)]

    @staticmethod
    def jvp(primals, tangents, *, axes):
        (x,), (x_dot,) = primals, tangents
        eval_out = sum(x, axes)
        jvp_out = sum(x_dot, axes)
        return [eval_out], [jvp_out]

    @staticmethod
    def T(cts, x, *, axes):
        (z,) = cts
        out = z
        out = broadcast(z, x.aval.shape, axes)
        return [out]


# -----------------------
# ShapeOps
# -----------------------


class Broadcast(ShapeOp):
    @staticmethod
    def eval(x, *, shape, axes):
        if axes is not None:
            for a in sorted(axes):
                x = x.expand_dims(a)
        out = x.broadcast_to(shape)
        return [out]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, shape, axes):
        (x,), (x_bdim,) = vals_in, dims_in
        # x1s = [d for i,d in enumerate(x.shape) if i != x_bdim]
        shape_ = list(shape)
        axes_ = list(axes)
        shape = list(shape)
        axes = [a + int(a >= (x_bdim)) for a in axes]
        if all([a < x_bdim for a in axes]):
            x_bdim += 1

        shape = shape[:x_bdim] + [axis_size] + shape[x_bdim:]
        # if sum(int(a<x_bdim) for a in axes) != 0:
        #     breakpoint()

        return [broadcast(x, shape, axes)], [x_bdim]

    @staticmethod
    def jvp(primals, tangents, *, shape, axes):
        (x,), (x_dot,) = primals, tangents
        return (
            [broadcast(x, shape=shape, axes=axes)],
            [broadcast(x_dot, shape=shape, axes=axes)],
        )

    @staticmethod
    def shape_eval(x: ArrayShape, *, shape: Sequence[int], axes) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), x.dtype)]

    @staticmethod
    def T(cts, x, *, shape, axes):
        (z,) = cts
        out = z
        if axes is not None:
            out = sum(z, axes)
        else:
            eshape = list(shape)
            for a in axes:
                if a < 0:
                    a = len(shape) + (a + 1)
                eshape.insert(a, 1)
        return [out]


# class Crop(ShapeOp):
#     @staticmethod
#     def eval(x, slice):
#         return [x[slice]]


class Reshape(ShapeOp):
    @staticmethod
    def eval(x, *, shape):
        return [x.reshape(shape)]

    @staticmethod
    def jvp(primals, tangents, *, shape):
        (x,), (x_dot,) = primals, tangents
        return [reshape(x, shape)], [reshape(x_dot, shape)]

    @staticmethod
    def shape_eval(x: ArrayShape, *, shape: Sequence[int]) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), x.dtype)]

    @staticmethod
    def T(cts, x, *, shape):
        (z,) = cts
        return [reshape(z, x.aval.shape)]


class Transpose(ShapeOp):
    @staticmethod
    def eval(x, *, perm):
        return [np.transpose(x, perm)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, perm):
        (x,), (x_bdim,) = vals_in, dims_in
        perm_ = list(perm)
        x_bdim_ = int(x_bdim)
        assert x_bdim >= 0
        # perm = [d - int(i >= x_bdim) for i, d in enumerate(perm)]
        perm = perm[:x_bdim] + [x_bdim] + perm[x_bdim:]
        perm = [d + int(d >= x_bdim) if i != x_bdim else d for i, d in enumerate(perm)]
        assert len(set(perm)) == len(perm)
        # perm[:x_bdim] = perm[:x_bdim][::-1]
        # breakpoint()
        return [transpose(x, perm)], [x_bdim]

    @staticmethod
    def jvp(primals, tangents, *, perm):
        (x,), (x_dot,) = primals, tangents
        return [transpose(x, perm)], [transpose(x_dot, perm)]

    @staticmethod
    def shape_eval(x: ArrayShape, *, perm: Sequence[int]) -> List[ArrayShape]:
        shape = [x.shape[i] for i in perm]
        return [ArrayShape(shape, x.dtype)]

    @staticmethod
    def T(cts, x, *, perm):
        (z,) = cts
        return [transpose(z, perm)]


# UnaryOps
def identity(x):
    return slope.RT.bind1(Identity, x)


def full_like(x, fill_value):
    return slope.RT.bind1(FullLike, x, fill_value=fill_value)


def zeros_like(x):
    return full_like(x, 0)


def ones_like(x):
    return full_like(x, 1)


def stop_gradient(x):
    return slope.RT.bind1(StopGradient, x)


def convert(x, dtype):
    return slope.RT.bind1(Convert, x, dtype=dtype)


def exp(x):
    return slope.RT.bind1(Exp, x)


def log(x):
    return slope.RT.bind1(Log, x)


def neg(x):
    return slope.RT.bind1(Neg, x)


## Arithmetic


def add(x, y):
    return slope.RT.bind1(Add, x, y)


def sub(x, y):
    return slope.RT.bind1(Sub, x, y)


def mul(x, y):
    return slope.RT.bind1(Mul, x, y)


def div(x, y):
    return slope.RT.bind1(Div, x, y)


## Logic


def equal(x, y):
    return slope.RT.bind1(Equal, x, y)


def max(x, y):
    return slope.RT.bind1(Max, x, y)


def min(x, y):
    return -slope.RT.bind1(Max, -x, -y)


# ReduceOps
def sum(x, axes=None):
    return slope.RT.bind1(Sum, x, axes=axes)


def mean(x, axes=None):
    if axes is None:
        axes = tuple(range(len(x.shape)))
    N = math.prod([x.shape[a] for a in axes])
    return sum(x, axes) / np.float32(N)


def max(x, axes=None):
    return slope.RT.bind1(Max, x, axes=axes)


# ShapeOps
def broadcast(x, shape, axes=None):
    return slope.RT.bind1(Broadcast, x, shape=shape, axes=axes)


def reshape(x, shape):
    return slope.RT.bind1(Reshape, x, shape=shape)


def transpose(x, perm):
    return slope.RT.bind1(Transpose, x, perm=perm)


def expand_dims(x, axes):
    shape = list(x.shape)
    for a in axes:
        if a < 0:
            a = len(shape) + (a + 1)
        shape.insert(a, 1)
    x = reshape(x, shape)
    return x


# NN


def T(x):
    perm = list(range(len(x.shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return transpose(x, perm)


# def mm_old(x, y):
#     x1, x2 = x.shape[-2], x.shape[-1]
#     y1, y2 = y.shape[-2], y.shape[-1]
#     assert x2 == y1
#     y = T(y)
#     # br_shape = (*x.shape[:-3], *(d, a, b))
#     br_shape = (y2, x1, x2)
#     x = broadcast(x, br_shape, (-3,))
#     y = broadcast(y, br_shape, (-2,))
#     z = x * y
#     z = sum(z, (-1,))
#     breakpoint()
#     z = T(z)
#     return z


def mm(x, y):
    x1, x2 = x.shape[0], x.shape[1]
    y1, y2 = y.shape[0], y.shape[1]
    assert x2 == y1
    y = T(y)
    # br_shape = (*x.shape[:-3], *(d, a, b))
    br_shape = (y2, x1, x2)
    x = broadcast(x, br_shape, (0,))
    y = broadcast(y, br_shape, (1,))
    z = x * y
    z = sum(z, (2,))
    z = T(z)
    return z


def mm_noT(x, y):
    a, b = x.shape[-2], x.shape[-1]
    d, c = y.shape[-2], y.shape[-1]
    br_shape = (*x.shape[:-3], *(d, a, b))
    # breakpoint()
    x = broadcast(x, br_shape, (-3,))
    y = broadcast(y, br_shape, (-2,))
    z = x * y
    z = sum(z, (-1,))
    return z


def relu(x):
    return max(x, np.zeros(x.shape, x.dtype))


def softmax(x, axes):
    x_max = max(x, axes)
    x_max = broadcast(x_max, x.shape)

    e = exp(x - x_max)
    s_e = sum(e, axes)
    s_e = broadcast(s_e, e.shape)
    return e / s_e


def cross_entropy(x, y):
    return x * log(y)


def mse(x, y):
    return pow((x - y), 2)


def pow(x, y):
    assert type(y) is int
    if y == 0:
        return slope.ad.ones_like(x)
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
        ret = ones_like(acc) / acc
    return ret


def mean(x, axes=None):
    x_sum = sum(x, axes)
    if axes is None:
        axes = list(range(len(x.shape)))
    N = math.prod([x.shape[a] for a in axes])
    return x_sum / N


def log_softmax(x, axes=(-1,)):
    x_max = max(x, axes)
    x_max = broadcast(x_max, x.shape, (-1,))
    # x_s = x - stop_gradient(x_max)
    x_s = x - x_max
    x_s_se = sum(exp(x_s), axes)
    x_s_se = broadcast(x_s_se, x.shape, (-1,))
    x_s_lse = log(x_s_se)
    return x_s - x_s_lse
