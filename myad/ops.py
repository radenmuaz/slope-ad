import myad
import numpy as np
from myad.array_shape import ArrayShape
from typing import List, Tuple, Sequence, Any
from abc import ABC, abstractmethod

class Op(ABC):
    @staticmethod
    @abstractmethod
    def eval(*args):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def vmap(*args):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def jvp(*args):
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
    def mlir(cls):
        raise NotImplementedError

# class FnOp(Op):
#     def fn(self, *args):
#         raise NotImplementedError
#     def __call__(self, *args):
#         if myad.RT.trace_stack.trace_type == myad.core.JVPTrace:

    
#     eval = vmap = jvp = shape_eval = __call__


class UnaryOp(Op):
    @staticmethod
    def vmap(axis_size, vals_in, dims_in):
        (x,), (x_bdim,) = vals_in, dims_in
        return [myad.RT.bind1(x)], [x_bdim]

    @staticmethod
    def shape_eval(x: ArrayShape, **params) -> List[ArrayShape]:
        return [ArrayShape(x.shape, x.dtype)]

class BinaryOp(Op):
    
    @staticmethod
    def vmap(axis_size, vals_in, dims_in):
        (x, y), (x_bdim, y_bdim) = vals_in, dims_in
        if x_bdim != y_bdim:
            if x_bdim is None:
                x = myad.core.move_batch_axis(axis_size, x_bdim, y_bdim, x)
                x_bdim = y_bdim
            else:
                y = myad.core.move_batch_axis(axis_size, y_bdim, x_bdim, y)
        return [myad.RT.bind1(x, y)], [x_bdim]

    @staticmethod
    def shape_eval(x: ArrayShape, y: ArrayShape) -> List[ArrayShape]:
        if not isinstance(x, ArrayShape) or not isinstance(y, ArrayShape):
            raise TypeError
        if ArrayShape.like(x) != ArrayShape.like(y):
            
            raise TypeError
        return [ArrayShape(x.shape, x.dtype)]


class ReduceOp(Op):
    @staticmethod
    def vmap(axis_size, vals_in, dims_in, axis):
        (x,), (x_bdim,) = vals_in, dims_in
        new_axis = tuple(ax + (x_bdim <= ax) for ax in axis)
        out_bdim = x_bdim - sum(ax < x_bdim for ax in axis)
        return [myad.RT.bind1(x, axis=new_axis)], [out_bdim]

    @staticmethod
    def shape_eval(x: ArrayShape, axis: Tuple[int, ...]) -> List[ArrayShape]:
        axis_ = set(axis)
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
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [identity(x)], [identity(x_dot)]


class Convert(UnaryOp):
    @staticmethod
    def eval(x, *, dtype):
        return [x.astype(dtype)]
    
    @staticmethod
    def jvp(primals, tangents, *, dtype):
        (x,), (x_dot,) = primals, tangents
        return [convert(x, dtype)], [convert(x_dot, dtype)]
    
    @staticmethod
    def T(cts, x, **params):
        (y_bar,) = cts
        assert type(x) is myad.core.UndefPrimal
        return [convert(y_bar, x.dtype)]

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
        return [x*y], [x_dot*y + x*y_dot]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        if type(x) is myad.core.UndefPrimal:
            return [(mul(z_bar, y)), None] 
        elif type(y) is myad.core.UndefPrimal:
            return [None, x * z_bar]


class Div(BinaryOp):
    @staticmethod
    def eval(x, y):
        return [x / y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x / y], [(x_dot/y) + (-y_dot * x * (y**-2))]
    
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
        return [out_primal], [np.zeros(out_primal.shape, out_primal.dtype)]
    
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
        eval_out = broadcast(eval_out, x.shape, ())
        locs = equal(x, eval_out)
        locs = convert(locs, x_dot.dtype)
        counts = reduce_sum(locs, axis)
        jvp_out = reduce_sum(x_dot * locs, axis)
        
        jvp_out = jvp_out / counts
        
        return [eval_out], [jvp_out]
    
    @staticmethod
    def T(cts, x, *, axis):
        (y_bar,) = cts
        return [broadcast(y_bar, x.aval.shape, ())]

class ReduceSum(ReduceOp):
    @staticmethod
    def eval(x, *, axis):
        if len(axis) == 0:
            axis = None # empty tuple does not do sum
        return [np.sum(x, axis)]

    @staticmethod
    def jvp(primals, tangents, *, axis):
        (x,), (x_dot,) = primals, tangents
        eval_out = reduce_sum(x, axis)
        jvp_out = reduce_sum(x_dot, axis)
        return [eval_out], [jvp_out]

    @staticmethod
    def T(cts, x, *, axis):
        (y_bar,) = cts
        return [broadcast(y_bar, x.aval.shape, axis)]
    
# -----------------------
# ShapeOps
# -----------------------

class Broadcast(ShapeOp):
    @staticmethod
    def eval(x, *, shape, axes):
        for axis in sorted(axes):
            x = np.expand_dims(x, axis)
        return [np.broadcast_to(x, shape)]


    @staticmethod
    def jvp(primals, tangents, *, shape, axes):
        (x,), (x_dot,) = primals, tangents
        return ([broadcast(x, shape=shape, axes=axes)],
                [broadcast(x_dot, shape=shape, axes=axes)])

    @staticmethod
    def shape_eval(
        x: ArrayShape, *, shape: Sequence[int], axes
    ) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), x.dtype)]
    
    @staticmethod
    def T(cts, x, *, shape, axes):
        (y_bar,) = cts
        out = y_bar
        out = reduce_sum(y_bar, axes)
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
    def shape_eval(
        x: ArrayShape, *, shape: Sequence[int]
    ) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), x.dtype)]

    @staticmethod
    def T(cts, x, *, shape):
        (y_bar,) = cts
        return [reshape(y_bar, x.aval.shape)]


class Transpose(ShapeOp):
    @staticmethod
    def eval(x, *, perm):
        return [x.transpose(perm)]
    
    @staticmethod
    def jvp(primals, tangents, *, perm):
        (x,), (x_dot,) = primals, tangents
        return [transpose(x, perm)], [transpose(x_dot, perm)]

    @staticmethod
    def shape_eval(
        x: ArrayShape, *, perm: Sequence[int]
    ) -> List[ArrayShape]:
        shape = [x.shape[i] for i in perm]
        return [ArrayShape(shape, x.dtype)]
    
    @staticmethod
    def T(cts, x, *, perm):
        (y_bar,) = cts
        return [transpose(y_bar, perm)]

# UnaryOps
def identity(x):
    return myad.RT.bind1(Identity, x)
def convert(x, dtype):
    return myad.RT.bind1(Convert, x, dtype=dtype)
def exp(x):
    return myad.RT.bind1(Exp, x)
def log(x):
    return myad.RT.bind1(Log, x)
def neg(x):
    return myad.RT.bind1(Neg, x)

# BinaryOps

## Arithmetic
def add(x, y):
    return myad.RT.bind1(Add, x, y)
def sub(x, y):
    return myad.RT.bind1(Sub, x, y)
def mul(x, y):
    return myad.RT.bind1(Mul, x, y)
def div(x, y):
    return myad.RT.bind1(Div, x, y)

## Logic
def equal(x, y):
    return myad.RT.bind1(Equal, x, y)
def max(x, y):
    return myad.RT.bind1(Max, x, y)
def min(x, y):
    return -myad.RT.bind1(Max, -x, -y)


# ReduceOps
def reduce_sum(x, axis):
    return myad.RT.bind1(ReduceSum, x, axis=axis)
def reduce_max(x, axis):
    return myad.RT.bind1(ReduceMax, x, axis=axis)

# ShapeOps
def broadcast(x, shape, axes):
    return myad.RT.bind1(Broadcast, x, shape=shape, axes=axes)
def reshape(x, shape):
    return myad.RT.bind1(Reshape, x, shape=shape)

def transpose(x, perm):
    return myad.RT.bind1(Transpose, x, perm=perm)


# def expand_dims(x, axis):
#     shape = list(x.shape)
#     for a in axis:
#         if a < 0:
#             a = len(shape) + (a+1)
#         shape.insert(a, 1)
#     x = reshape(x, shape)
#     return x

def T(x):
    perm = list(range(len(x.shape)))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return transpose(x, perm)

def dot(x, y):
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
    return z

def relu(x):
    return max(x, myad.core.zeros_like(x))

def softmax(x, axis):
    x_max = reduce_max(x, axis)
    x_max = broadcast(x_max, x.shape, ())
    
    e = exp(x - x_max)
    # e = exp(x)
    s_e = reduce_sum(e, axis)
    s_e = broadcast(s_e, e.shape, ())
    return e / s_e

    
def cross_entropy(x, y):
    return x * log(y)

def mse(x, y):
    return pow((x - y), 2)

def pow(x, y):
    assert type(y) is int
    if y == 0:
        return myad.core.ones_like(x)
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
        ret = (myad.core.ones_like(acc) / acc)
    return ret