import myad
import numpy as np
from myad.array_shape import ArrayShape
from typing import List, Tuple, Sequence, Any
from abc import ABC, abstractmethod

class Op(ABC):
    @classmethod
    def do(cls, *args, **params):
        return myad.RT.bind1(cls, *args, **params)

    @classmethod
    @abstractmethod
    def eval(cls, *args):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def vmap(cls, *args):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def jvp(cls, *args):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def shape_eval(cls, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def pprint(cls):
        return None

    @classmethod
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
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in):
        (x,), (x_bdim,) = vals_in, dims_in
        return [cls.do(x)], [x_bdim]

    @classmethod
    def shape_eval(cls, x: ArrayShape) -> List[ArrayShape]:
        return [ArrayShape(x.shape, x.dtype)]


class BinaryOp(Op):
    
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in):
        (x, y), (x_bdim, y_bdim) = vals_in, dims_in
        if x_bdim != y_bdim:
            if x_bdim is None:
                x = myad.core.move_batch_axis(axis_size, x_bdim, y_bdim, x)
                x_bdim = y_bdim
            else:
                y = myad.core.move_batch_axis(axis_size, y_bdim, x_bdim, y)
        return [cls.do(x, y)], [x_bdim]

    @classmethod
    def shape_eval(cls, x: ArrayShape, y: ArrayShape) -> List[ArrayShape]:
        if not isinstance(x, ArrayShape) or not isinstance(y, ArrayShape):
            raise TypeError
        if ArrayShape.like(x) != ArrayShape.like(y):
            raise TypeError
        return [ArrayShape(x.shape, x.dtype)]


class ReduceOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in, axis):
        (x,), (x_bdim,) = vals_in, dims_in
        new_axis = tuple(ax + (x_bdim <= ax) for ax in axis)
        out_bdim = x_bdim - sum(ax < x_bdim for ax in axis)
        return [cls.do(x, new_axis)], [out_bdim]

    @classmethod
    def shape_eval(cls, x: ArrayShape, axis: Tuple[int, ...]) -> List[ArrayShape]:
        axis_ = set(axis)
        new_shape = [d for i, d in enumerate(x.shape) if i not in axis_]
        return [ArrayShape(tuple(new_shape), x.dtype)]


class ShapeOp(Op):
    pass


# -----------------------
# UnaryOps
# -----------------------


class Identity(UnaryOp):
    @classmethod
    def eval(cls, x):
        return [x]
    
    @classmethod
    def jvp(cls, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [cls.do(x)], [x_dot]

class Exp(UnaryOp):
    @classmethod
    def eval(cls, x):
        return [np.exp(x)]

    @classmethod
    def jvp(cls, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [cls.do(x)], [x_dot * cls.do(x)]


class Log(UnaryOp):
    @classmethod
    def eval(cls, x):
        return [np.log(x)]

    @classmethod
    def jvp(cls, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [cls.do(x)], [x_dot / x]


class Neg(UnaryOp):
    @classmethod
    def eval(cls, x):
        return [-x]

    @classmethod
    def jvp(cls, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [-x], [-x_dot]
    
    @classmethod
    def T(cls, t, x):
        (z,) = t
        return [-z]


# -----------------------
# BinaryOps
# -----------------------


class Add(BinaryOp):
    @classmethod
    def eval(cls, x, y):
        return [x + y]

    @classmethod
    def jvp(cls, primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x + y], [x_dot + y_dot]

    @classmethod
    def T(cls, cts, x, y):
        (z_bar,) = cts
        return [z_bar, z_bar]



class Sub(BinaryOp):
    @classmethod
    def eval(cls, x, y):
        return [x - y]

    @classmethod
    def jvp(cls, primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x - y], [x_dot - y_dot]

    @classmethod
    def T(cls, cts, x, y):
        (z_bar,) = cts
        return [z_bar, -z_bar]

class Mul(BinaryOp):
    @classmethod
    def eval(cls, x, y):
        return [x * y]

    @classmethod
    def jvp(cls, primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]

    @classmethod
    def T(cls, cts, x, y):
        (z_bar,) = cts
        if type(x) is myad.core.UndefPrimal:
            return [(z_bar * y), None] 
        elif type(y) is myad.core.UndefPrimal:
            return [None, (x * z_bar)]


class Div(BinaryOp):
    @classmethod
    def eval(cls, x, y):
        return [x / y]

    @classmethod
    def jvp(cls, primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x / y], [(x_dot / y) + (-y_dot * x * (y**-2))]
    
    @classmethod
    def T(cls, cts, x, y):
        (z_bar,) = cts
        return [z_bar / y, None]


class Pow(BinaryOp):
    @classmethod
    def eval(cls, x, y):
        return [x**y]

    @classmethod
    def jvp(cls, primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x * y], [x_dot * y + x * y_dot]
    
    @classmethod
    def T(cls, cts, x, y):
        (z_bar,) = cts
        return [z_bar / y, None]


class Max(BinaryOp):
    @classmethod
    def eval(cls, x, y):
        return [np.greater(x, y)]

    @classmethod
    def jvp(cls, primals, tangents):
        (x, y), _ = primals, tangents
        out_primal = cls.do(x, y)
        return [out_primal], [np.zeros(out_primal.shape, out_primal.dtype)]
    
    @classmethod
    def T(cls, cts, x, y):
        (z_bar,) = cts
        return [z_bar, None]


class Equal(BinaryOp):
    @classmethod
    def eval(cls, x, y):
        return [np.equal(x, y)]

    @classmethod
    def jvp(cls, primals, tangents):
        (x, y), _ = primals, tangents
        out_primal = cls.do(x, y)
        return [out_primal], [np.zeros(out_primal.shape, out_primal.dtype)]
    
    @classmethod
    def T(cls, cts, x, y):
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
    @classmethod
    def eval(cls, x, axis):
        return [x.sum(axis)]

    @classmethod
    def jvp(cls, primals, tangents, axis):
        eval_out = cls.do(*primals)
        eq_shape = [1 if i in axis else d 
                    for i, d in enumerate(primals.shape)]
        # we do equal with implicit broadcasting
        locs = Equal.do(primals, Reshape.do(eval_out, eq_shape))
        counts = ReduceSum.do(locs, axis)
        jvp_out = ReduceSum.do(tangents * locs, axis) / counts
        
        return [eval_out], [jvp_out]
    
    @classmethod
    def T(cls, cts, x):
        (y_bar, axis) = cts
        reshape_shape = list(y_bar.shape)
        reshape_shape.insert(axis, 1)
        y_bar = Reshape.do(x, reshape_shape)
        return [Broadcast(y_bar, x.aval.shape)]

class ReduceSum(ReduceOp):
    @classmethod
    def eval(cls, x, *, axis):
        return [np.sum(x, axis)]

    @classmethod
    def jvp(cls, primals, tangents, *, axis):
        (x,), (x_dot,) = primals, tangents
        eval_out = cls.do(x, axis=axis)
        jvp_out = cls.do(x_dot, axis=axis)
        return [eval_out], [jvp_out]

    @classmethod
    def T(cls, cts, x, *, axis):
        (y_bar,) = cts
        shape = [1 if i == axis[i] else k
                for i, k in enumerate(x.aval.shape)]
        y_bar = Reshape.do(y_bar, shape=shape)
        return [Broadcast.do(y_bar, shape=x.aval.shape)]
    

# def _reduce_sum_transpose_rule(cotangent, operand, *, axes):
#   assert ad.is_undefined_primal(operand)
#   input_shape = operand.aval.shape
#   broadcast_dimensions = tuple(np.delete(np.arange(len(input_shape)), axes))
#   result = broadcast_in_dim(cotangent, input_shape, broadcast_dimensions)
#   assert result.shape == input_shape
#   return [result]


# -----------------------
# ShapeOps
# -----------------------


class Broadcast(ShapeOp):
    @classmethod
    def eval(cls, x, *, shape):
        return [np.broadcast_to(x, shape)]

    @classmethod
    def jvp(cls, primals, tangents, *, shape):
        (x,), (x_dot,) = primals, tangents
        return [cls.do(x, shape=shape)], [cls.do(x_dot, shape=shape)]

    @classmethod
    def shape_eval(cls,
        x: ArrayShape, shape: Sequence[int]
    ) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), x.dtype)]
    
    @classmethod
    def T(cls, cts, x, *, shape):
        (y_bar,) = cts
        axis = []
        for idx, i, j in enumerate(zip(x.shape, shape)):
            if i == 1 and i < j:
                axis += [idx]
        if axis is None:
            raise ValueError
        return [ReduceSum.do(y_bar, axis=axis)]


# class Crop(ShapeOp):
#     @classmethod
#     def eval(cls, x, slice):
#         return [x[slice]]


class Reshape(ShapeOp):
    @classmethod
    def eval(cls, x, *, shape):
        return [np.reshape(x, shape)]
    
    @classmethod
    def jvp(cls, primals, tangents, *, shape):
        (x), (x_dot) = primals, tangents
        return [cls.do(x, shape)], [cls.do(x_dot, shape)]
    

    @classmethod
    def T(cls, cts, x, *, shape):
        (y_bar,) = cts
        return [Reshape.do(y_bar, shape=x.aval.shape)]


class Transpose(ShapeOp):
    @classmethod
    def eval(cls, x, *, perm):
        return [x.transpose(perm)]
    
    @classmethod
    def jvp(cls, primals, tangents, *, perm):
        (x), (x_dot) = primals, tangents
        return [cls.do(x, perm)], [cls.do(x_dot, perm)]
    
    @classmethod
    def T(cls, cts, x, *, perm):
        (y_bar,) = cts
        return [cls.do(y_bar, perm=perm)]