import slope
import numpy as np
from slope.array_shape import ArrayShape
from typing import List, Tuple, Sequence, Any, Callable, NamedTuple
from abc import ABC, abstractmethod
import math


class Op(ABC):
    @classmethod
    def do(cls, *args, **params):
        slope.RT.bind1(cls, *args, **params)

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

    @classmethod
    def identity_T(cls, t, x):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [cls.do(z)]
    
    @classmethod
    def zero_T(cls, t, x):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [zeros_like(z)]


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
            raise TypeError(f"{x} is not same as {y}")
        return [ArrayShape(x.shape, x.dtype)]


class ReduceOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in, **params):
        (x,), (x_bdim,) = vals_in, dims_in
        params["axis"] = tuple(ax + (x_bdim <= ax) for ax in params["axis"])
        out_bdim = x_bdim - sum(ax < x_bdim for ax in params["axis"])
        return [cls.do(x, **params)], [out_bdim]

    @staticmethod
    def shape_eval(x: ArrayShape, **params) -> List[ArrayShape]:
        axis_ = set(params["axis"])
        new_shape = [d for i, d in enumerate(x.shape) if i not in axis_]
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



class FullLike(UnaryOp):
    @staticmethod
    def eval(x, *, fill_value):
        return [np.full(x.shape, fill_value=fill_value, dtype=x.dtype)]

    @staticmethod
    def jvp(cls, primals, tangents, *, fill_value):
        (x,), (x_dot,) = primals, tangents
        return [full_like(x, fill_value)], [zeros_like(x_dot)]

    @staticmethod
    def T(t, x, *, fill_value):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [zeros_like(z)]

class StopGradient(UnaryOp):
    @staticmethod
    def eval(x):
        return [identity(x)]

    @staticmethod
    def jvp(primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [identity(x, **params)], [zeros_like(x)]

    def T(t, x):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [zeros_like(z)]



class Convert(UnaryOp):
    @staticmethod
    def eval(x, *, dtype):
        return [x.astype(dtype)]

    jvp = staticmethod(UnaryOp.identity_jvp)
    T = staticmethod(UnaryOp.identity_T)


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
        return [x * y], [x_dot * y + x * y_dot]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        if type(x) is slope.ad.UndefPrimal:
            return [(mul(z_bar, y)), None]
        elif type(y) is slope.ad.UndefPrimal:
            return [None, x * z_bar]


class Div(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x / y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x / y], [(x_dot / y) + (-y_dot * x * (y**-2))]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar / y, None]


class Max(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [np.greater(x, y)]

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


class ReduceMax(ReduceOp):
    @staticmethod
    def eval(x, axis):
        return [x.max(axis)]

    @staticmethod
    def jvp(primals, tangents, axis):
        (x,), (x_dot,) = primals, tangents
        eval_out = reduce_max(x, axis)
        # eval_out = broadcast(eval_out, x.shape, (-1,))
        locs = equal(x, eval_out)
        locs = convert(locs, x_dot.dtype)
        counts = reduce_sum(locs, axis)
        # counts = broadcast(counts, x.shape, (-1,))
        jvp_out = reduce_sum(x_dot * locs, axis)
        jvp_out = jvp_out / counts

        return [eval_out], [jvp_out]

    @staticmethod
    def T(cts, x, *, axis):
        (z,) = cts
        return [broadcast(z, x.aval.shape, ())]


class ReduceSum(ReduceOp):
    @staticmethod
    def eval(x, *, axis):
        if type(axis) is not tuple:
            axis = tuple(axis)
            breakpoint()
        return [np.sum(x, axis)]

    @staticmethod
    def jvp(primals, tangents, *, axis):
        (x,), (x_dot,) = primals, tangents
        eval_out = reduce_sum(x, axis)
        jvp_out = reduce_sum(x_dot, axis)
        return [eval_out], [jvp_out]

    @staticmethod
    def T(cts, x, *, axis):
        (z,) = cts
        return [broadcast(z, x.aval.shape, axis)]


# -----------------------
# ShapeOps
# -----------------------


class Broadcast(ShapeOp):
    @staticmethod
    def eval(x, *, shape, axes):
        if axes is not None:
            for axis in sorted(axes):
                x = np.expand_dims(x, axis)
        return [np.broadcast_to(x, shape) if shape is not None else x]

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
        out = reduce_sum(z, axes)
        out = reshape(out, x.aval.shape)
        return [out]


# class Crop(ShapeOp):
#     @staticmethod
#     def eval(x, slice):
#         return [x[slice]]


class Reshape(ShapeOp):
    @staticmethod
    def eval(x, *, shape):
        return [np.reshape(x, shape)]

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
        return [x.transpose(perm)]

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


# BinaryOps

def binaryop_broadcast(op):
    def wrapped_op(x, y):
        axis = None
        diff = len(x.shape) - len(y.shape)
        x_shape, y_shape = x.shape, y.shape
        if diff > 0:
            axis = list(range(diff))
            y_shape = [1]*diff+list(y_shape)
        elif diff < 0:
            axis = list(range(-diff))
            x_shape = [1]*(-diff)+list(x_shape)
        for dim_x, dim_y in zip(x_shape[::-1], y_shape[::-1]):
            if dim_x != dim_y and not (dim_x == 1 or dim_y == 1):
                raise ValueError("Arrays could not be broadcast together.")
        if x.shape != x_shape:
            x = broadcast(x, y.shape, axis)
        elif y.shape != y_shape:
            y = broadcast(y, x.shape, axis)
        return op(x, y)
    return wrapped_op

## Arithmetic
@binaryop_broadcast
def add(x, y):
    return slope.RT.bind1(Add, x, y)


@binaryop_broadcast
def sub(x, y):
    return slope.RT.bind1(Sub, x, y)

@binaryop_broadcast
def mul(x, y):
    return slope.RT.bind1(Mul, x, y)


@binaryop_broadcast
def div(x, y):
    return slope.RT.bind1(Div, x, y)


## Logic
@binaryop_broadcast
def equal(x, y):
    return slope.RT.bind1(Equal, x, y)

@binaryop_broadcast
def max(x, y):
    return slope.RT.bind1(Max, x, y)

@binaryop_broadcast
def min(x, y):
    return -slope.RT.bind1(Max, -x, -y)


# ReduceOps
def reduce_sum(x, axis=None):
    return slope.RT.bind1(ReduceSum, x, axis=axis)


def reduce_max(x, axis=None):
    return slope.RT.bind1(ReduceMax, x, axis=axis)


# ShapeOps
def broadcast(x, shape, axes=None):
    return slope.RT.bind1(Broadcast, x, shape=shape, axes=axes)


def reshape(x, shape):
    return slope.RT.bind1(Reshape, x, shape=shape)


def transpose(x, perm):
    return slope.RT.bind1(Transpose, x, perm=perm)


# def expand_dims(x, axis):
#     shape = list(x.shape)
#     for a in axis:
#         if a < 0:
#             a = len(shape) + (a+1)
#         shape.insert(a, 1)
#     x = reshape(x, shape)
#     return x

# NN

def T(x):
    perm = list(range(len(x.shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return transpose(x, perm)


def dot(x, y):
    # x_is_vec, y_is_vec = False, False
    # if len(x.shape) == 1:
    #     x_is_vec = True
    #     x = reshape(x, [1]+list(x.shape))
    # if len(y.shape) == 1:
    #     y_is_vec = True
    #     y = reshape(y, list(y.shape)+[1])
    a, b = x.shape[-2], x.shape[-1]
    c, d = y.shape[-2], y.shape[-1]
    assert b == c
    y = T(y)
    br_shape = (*x.shape[:-3], *(d, a, b))
    # x = expand_dims(x, (-3,))
    x = broadcast(x, br_shape, (-3,))
    # y = expand_dims(y, (-2,))
    y = broadcast(y, br_shape, (-2,))
    z = x * y
    z = reduce_sum(z, (-1,))
    z = T(z)
    # if x_is_vec:
    #     z = reshape(z,z.shape[1:])
    # if x_is_vec:
    #     z = reshape(z,z.shape[1:]) 
    return z


def relu(x):
    return max(x, np.zeros(x.shape, x.dtype))


def softmax(x, axis):
    x_max = reduce_max(x, axis)
    x_max = broadcast(x_max, x.shape)

    e = exp(x - x_max)
    s_e = reduce_sum(e, axis)
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


def mean(x, axis=None):
    x_sum = sum(x, axis)
    if axis is None:
        axis = list(range(len(x.shape)))
    N = math.prod([x.shape[a] for a in axis])
    return x_sum / N


def log_softmax(x, axis = (-1,)):
  x_max = reduce_max(x, axis)
  shifted = x - stop_gradient(x_max)
  sumexp = reduce_sum(exp(shifted), axis)
  shifted_logsumexp = log(sumexp)
  return shifted - shifted_logsumexp
