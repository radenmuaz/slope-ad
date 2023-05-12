import slope
import numpy as np
from slope.array_shape import ArrayShape
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
        a = list(params["axis"])
        (x,), (x_bdim,) = vals_in, dims_in
        axis_ = list(params["axis"])
        x_bdim_ = int(x_bdim)
        axis = list(params["axis"])
        x1s = [d for i, d in enumerate(x.shape) if i != x_bdim]
        # axis = [a+len(x1s) if a < 0 else a for a in axis]
        axis = tuple(ax + (x_bdim <= ax) for ax in axis)
        out_bdim = x_bdim - sum(ax < x_bdim for ax in axis)
        params["axis"] = tuple(axis)
        return [cls.do(x, **params)], [out_bdim]
        # return [cls.do(x, **params)], dims_in

    @staticmethod
    def shape_eval(x: ArrayShape, **params) -> List[ArrayShape]:
        axis = params["axis"]
        axis = [a + len(x.shape) if a < 0 else a for a in axis]
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
    def jvp(cls, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [x], [x_dot]

    @staticmethod
    def T(t, x):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [z]


class FullLike(UnaryOp):
    @staticmethod
    def eval(x, *, fill_value):
        return [np.full(x.shape, fill_value=fill_value, dtype=x.dtype)]

    @staticmethod
    def jvp(cls, primals, tangents, *, fill_value):
        (x,), (x_dot,) = primals, tangents
        return [x.full_like(fill_value)], [x_dot.zeros_like()]

    @staticmethod
    def T(t, x, *, fill_value):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [z.zeros_like()]


class StopGradient(UnaryOp):
    @staticmethod
    def eval(x):
        return [x]

    @staticmethod
    def jvp(primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [x], [x.zeros_like()]

    @staticmethod
    def T(t, x):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [z.zeros_like()]


class Convert(UnaryOp):
    @staticmethod
    def eval(x, *, dtype):
        return [x.astype(dtype)]

    @staticmethod
    def jvp(primals, tangents, *, dtype):
        (x,), (x_dot,) = primals, tangents
        return [x.convert(dtype)], [x_dot.convert(dtype)]

    @staticmethod
    def T(t, x):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [z.convert(x.dtype)]


class Exp(UnaryOp):
    @staticmethod
    def eval(x):
        return [np.exp(x)]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [x.exp()], [x_dot * x.exp()]
        


class Log(UnaryOp):
    @staticmethod
    def eval(x):
        return [np.log(x)]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [x.log()], [x_dot / x]


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


class ReLU(UnaryOp):
    @staticmethod
    def eval(x):
        return [np.max(x, 0)]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [x.relu()], [(x>0)*g]
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
        # breakpoint()
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
        (x, y), x_dot, y_dot = primals, tangents
        ans = x.maximum(y)
        x_jvp = x_dot * (x == ans) / ((y == ans) + 1)
        y_jvp = y_dot * (y == ans) / ((x == ans) + 1)
        return [ans], [x_jvp + y_jvp]

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
        ans = x == y
        return [ans], [ans.zeros_like()]

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
    def eval(x, axis):
        return [x.max(axis)]

    @staticmethod
    def jvp(primals, tangents, axis):
        (x,), (x_dot,) = primals, tangents
        ans = x.max(axis)
        matches = (x == ans).convert(x_dot.dtype)
        counts = matches.sum(axis)
        ans_jvp = (x_dot * matches).sum(axis)
        ans_jvp = ans_jvp / counts

        return [ans], [ans_jvp]

    @staticmethod
    def T(cts, x, *, axis):
        (z,) = cts
        return [z.broadcast(x.aval.shape, ())]


class Sum(ReduceOp):
    @staticmethod
    def eval(x, *, axis):
        return [np.sum(x, axis)]

    @staticmethod
    def jvp(primals, tangents, *, axis):
        (x,), (x_dot,) = primals, tangents
        ans, ans_jvp = x.sum(axis), x_dot.sum(axis)
        return [ans], [ans_jvp]

    @staticmethod
    def T(cts, x, *, axis):
        (z,) = cts
        out = z
        out = z.broadcast(x.aval.shape, axis)
        return [out]


# -----------------------
# ShapeOps
# -----------------------


class Broadcast(ShapeOp):
    @staticmethod
    def eval(x, *, shape, axes):
        if axes is not None:
            for axis in sorted(axes):
                x = np.expand_dims(x, axis)
        return [np.broadcast_to(x, shape)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, shape, axes):
        (x,), (x_bdim,) = vals_in, dims_in
        shape = list(shape)
        axes = [a + int(a >= (x_bdim)) for a in axes]
        if all([a < x_bdim for a in axes]):
            x_bdim += 1
        shape = shape[:x_bdim] + [axis_size] + shape[x_bdim:]
        return [x.broadcast(shape, axes)], [x_bdim]

    @staticmethod
    def jvp(primals, tangents, *, shape, axes):
        (x,), (x_dot,) = primals, tangents
        return (
            [x.broadcast(shape, axes)],
            [x_dot.broadcast(shape, axes)],
        )

    @staticmethod
    def shape_eval(x: ArrayShape, *, shape: Sequence[int], axes) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), x.dtype)]

    @staticmethod
    def T(cts, x, *, shape, axes):
        (z,) = cts
        out = z
        if axes is not None:
            out = z.sum(axes)
        else:
            eshape = list(shape)
            for a in axes:
                if a < 0:
                    a = len(shape) + (a + 1)
                eshape.insert(a, 1)
            breakpoint()
        return [out]

class Reshape(ShapeOp):
    @staticmethod
    def eval(x, *, shape):
        return [np.reshape(x, shape)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, shape):
        (x,), (x_bdim,) = vals_in, dims_in
        shape = list(shape)
        shape = shape[:x_bdim] + [axis_size] + shape[x_bdim:]
        return [x.reshape(shape)], [x_bdim]

    @staticmethod
    def jvp(primals, tangents, *, shape):
        (x,), (x_dot,) = primals, tangents
        return [x.reshape(shape)], [x_dot.reshape(shape)]

    @staticmethod
    def shape_eval(x: ArrayShape, *, shape: Sequence[int]) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), x.dtype)]

    @staticmethod
    def T(cts, x, *, shape):
        (z,) = cts
        return [z.reshape(x.aval.shape)]


class Transpose(ShapeOp):
    @staticmethod
    def eval(x, *, perm):
        return [x.transpose(perm)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, perm):
        (x,), (x_bdim,) = vals_in, dims_in
        assert x_bdim >= 0
        perm = perm[:x_bdim] + [x_bdim] + perm[x_bdim:]
        perm = [d + int(d >= x_bdim) if i != x_bdim else d for i, d in enumerate(perm)]
        assert len(set(perm)) == len(perm)
        return [x.transpose(perm)], [x_bdim]

    @staticmethod
    def jvp(primals, tangents, *, perm):
        (x,), (x_dot,) = primals, tangents
        return [x.transpose(perm)], [x_dot.ranspose(perm)]

    @staticmethod
    def shape_eval(x: ArrayShape, *, perm: Sequence[int]) -> List[ArrayShape]:
        shape = [x.shape[i] for i in perm]
        return [ArrayShape(shape, x.dtype)]

    @staticmethod
    def T(cts, x, *, perm):
        (z,) = cts
        return [z.transpose(perm)]

