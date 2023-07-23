import slope
import numpy as np
from slope.array_shape import ArrayShape
from typing import List, Tuple, Sequence, Any, Callable, Optional, Union
from abc import ABC, abstractmethod
import math
import functools
from slope.array import Array
from slope import utils
def raise_not_implemented(self, *args, **kwargs):
    raise NotImplementedError

class Op:
    def __init__(self, name):
        self.name = name
        self.eval = raise_not_implemented
        self.jvp = raise_not_implemented
        self.vmap = raise_not_implemented
        self.T = raise_not_implemented
        self.shape_eval = raise_not_implemented
    
    def __call__(self, *args, **kwargs):
        return slope.RT.bind1(self, *args, **kwargs)[0]
    
    def def_eval(self, fn):
        self.eval = fn

    def def_jvp(self, fn):
        self.jvp = fn

    def def_vmap(self, fn):
        self.vmap = fn

    def def_T(self, fn):
        self.T = fn
    
    def def_shape_eval(self, fn):
        self.shape_eval = fn
    
    @classmethod
    def unary(cls, name):
        op = cls(name)

        @op.def_vmap
        def fn(self, axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            return [self(x, **params)], [x_bdim]
        
        @op.def_shape_eval
        def fn(self, axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            return [self(x, **params)], [x_bdim]
        
        @op.def_jvp
        def fn(self, primals, tangents, **params):
            (x,), (x_dot,) = primals, tangents
            return [self(x, **params)], [self(x_dot, **params)]
        
        return op

    @classmethod
    def binary(cls, name):
        op = cls(name)

        @op.def_vmap
        def fn(self, axis_size, vals_in, dims_in, **params):
            (x, y), (x_bdim, y_bdim) = vals_in, dims_in
            if x_bdim != y_bdim:
                if x_bdim is None:
                    x = slope.ad.move_batch_axis(axis_size, x_bdim, y_bdim, x)
                    x_bdim = y_bdim
                else:
                    y = slope.ad.move_batch_axis(axis_size, y_bdim, x_bdim, y)
            return [self(x, y, **params)], [x_bdim]
        
        @op.def_shape_eval
        def fn(x: ArrayShape, y: ArrayShape, **params) -> List[ArrayShape]:
            # if not isinstance(x, ArrayShape) or not isinstance(y, ArrayShape):
            if not type(x) in (Array, ArrayShape) or not type(x) in (Array, ArrayShape):
                # breakpoint()
                raise TypeError
            if ArrayShape.like(x) != ArrayShape.like(y):
                raise TypeError(f"{x} != {y}")
            return [ArrayShape(x.shape, x.dtype)]
        
        @op.def_jvp
        def fn(self, primals, tangents, **params):
            (x,), (x_dot,) = primals, tangents
            return [self(x, **params)], [self(x_dot, **params)]
        
        return op
    
    @classmethod
    def reduce(cls, name):
        op = cls(name)
        
        @op.def_vmap
        def fn(cls, axis_size, vals_in, dims_in, **params):
            (x,), (x_bdim,) = vals_in, dims_in
            axes = list(params["axes"])
            axes = tuple(a + (x_bdim <= a) for a in axes)
            out_bdim = x_bdim - sum(a < x_bdim for a in axes)
            params["axes"] = tuple(axes)
            return [cls.do(x, **params)], [out_bdim]

        @op.def_shape_eval
        def fn(x: ArrayShape, **params) -> List[ArrayShape]:
            axes = params["axes"]
            axes = [a + len(x.shape) if a < 0 else a for a in axes]
            axes_ = set(axes)
            new_shape = [d for i, d in enumerate(x.shape) if i not in axes_]
            return [ArrayShape(tuple(new_shape), x.dtype)]
        return op
    
    @classmethod
    def shape(cls, name):
        op = cls(name)
        return op
    
    @classmethod
    def load(cls, name):
        op = cls(name)
        @op.def_jvp
        def fn(self, *args, **kwargs):
            out = cls.load_fn(*args, **kwargs)
            out_jvp = Array.ones_like(out)
            return [out], [out_jvp]
        
        @op.def_T
        def fn(self, cts, *args, **kwargs):
            return [cts[0]]
        return op

stop_gradient = Op.unary("stop_gradient")

@stop_gradient.def_eval
def fn(x):
    return [x]

@stop_gradient.def_jvp
def fn(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x], [Array.zeros_like(x_dot)]

@stop_gradient.def_T
def fn(cts, x):
    (z,) = cts
    assert type(x) is slope.ad.UndefPrimal
    return [Array.zeros_like(z)]

class Convert(UnaryOp):
    get_impl = lambda: slope.RT.backend.ConvertImpl

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


class Sqrt(UnaryOp):
    get_impl = lambda: slope.RT.backend.SqrtImpl

    @staticmethod
    def eval(x):
        return [x.sqrt()]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        ans = x.sqrt()
        return [ans], [x_dot * (0.5/ans)]

class Sin(UnaryOp):
    get_impl = lambda: slope.RT.backend.SinImpl

    @staticmethod
    def eval(x):
        return [x.sin()]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        ans = x.sin()
        return [ans], [x_dot * (1 - ans)]
    
    def T(cts, x):
        (z,) = cts
        return [-z * (1 - x.sin())]
    
class Exp(UnaryOp):
    get_impl = lambda: slope.RT.backend.ExpImpl

    @staticmethod
    def eval(x):
        return [x.exp()]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [x.exp()], [x_dot * x.exp()]
    
    def T(cts, x):
        (z,) = cts
        return [1 / z]


class Log(UnaryOp):
    get_impl = lambda: slope.RT.backend.LogImpl

    @staticmethod
    def eval(x):
        return [x.log()]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [x.log()], [x_dot / x]

    def T(cts, x):
        (z,) = cts
        return [1 / z]


class Neg(UnaryOp):
    get_impl = lambda: slope.RT.backend.NegImpl

    @staticmethod
    def eval(x):
        return [-x]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [-x], [-x_dot]

    @staticmethod
    def T(cts, x):
        (z,) = cts
        return [-z]


# -----------------------
# BinaryOps
# -----------------------


class Add(BinaryOp):
    get_impl = lambda: slope.RT.backend.AddImpl

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
    get_impl = lambda: slope.RT.backend.SubImpl

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
    get_impl = lambda: slope.RT.backend.MulImpl

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
    get_impl = lambda: slope.RT.backend.DivImpl

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
    get_impl = lambda: slope.RT.backend.MaximumImpl

    @staticmethod
    def eval(x, y):
        return [np.maximum(x, y)]

    @staticmethod
    def jvp(primals, tangents):
        def _balanced_eq(x, z, y):
            return ((x == z).where(Array.ones_like(z), Array.zeros_like(z))) / (
                (y == z).where(Array.full_like(z, 2), Array.ones_like(z))
            )

        (x, y), (x_dot, y_dot) = primals, tangents
        eval_out = x.maximum(y)
        jvp_out = x_dot * _balanced_eq(x, eval_out, y) + y_dot * _balanced_eq(
            y, eval_out, x
        )

        return [eval_out], [jvp_out]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar, None]


class Equal(BinaryOp):
    get_impl = lambda: slope.RT.backend.EqualImpl

    @staticmethod
    def eval(x, y):
        return [x.equal(y)]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), _ = primals, tangents
        out_primal = x.equal(y)
        return [out_primal], [np.zeros(out_primal.shape, out_primal.dtype)]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar, None]


# -----------------------
# ReduceOps
# -----------------------


class Max(ReduceOp):
    get_impl = lambda: slope.RT.backend.MaxImpl

    @staticmethod
    def eval(x, axes):
        return [x.max(axes)]

    @staticmethod
    def jvp(primals, tangents, axes):
        (x,), (x_dot,) = primals, tangents
        eval_out = x.max(axes)
        locs = x.equal(eval_out.broadcast(x.shape, axes))
        locs = locs.convert(x_dot.dtype)
        counts = locs.sum(axes)
        jvp_out = (x_dot * locs).sum(axes)
        jvp_out = jvp_out / counts.broadcast(jvp_out.shape)

        return [eval_out], [jvp_out]

    @staticmethod
    def T(cts, x, *, axes):
        (z,) = cts
        return [z.broadcast(x.aval.shape, ())]


class Sum(ReduceOp):
    get_impl = lambda: slope.RT.backend.SumImpl

    @staticmethod
    def eval(x, *, axes, keepdims):
        return [x.sum(axes, keepdims)]

    @staticmethod
    def jvp(primals, tangents, *, axes, keepdims):
        (x,), (x_dot,) = primals, tangents
        eval_out = x.sum(axes, keepdims)
        jvp_out = x_dot.sum(axes, keepdims)
        return [eval_out], [jvp_out]

    @staticmethod
    def T(cts, x, *, axes, keepdims):
        (z,) = cts
        out = z
        out = z.broadcast(x.aval.shape, axes)
        return [out]


# -----------------------
# ShapeOps
# -----------------------


class Broadcast(ShapeOp):
    get_impl = lambda: slope.RT.backend.BroadcastImpl

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

        return [x.broadcast(shape, axes)], [x_bdim]

    @staticmethod
    def jvp(primals, tangents, *, shape, axes):
        (x,), (x_dot,) = primals, tangents
        return (
            [x.broadcast(shape=shape, axes=axes)],
            [x_dot.broadcast(shape=shape, axes=axes)],
        )

    @staticmethod
    def shape_eval(x: ArrayShape, *, shape: Sequence[int], axes) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), x.dtype)]

    @staticmethod
    def T(cts, x, *, shape, axes):
        (z,) = cts
        out = z
        out = out.sum(axes, keepdims=False)
        more_axes = []
        for i, (dx, dz) in enumerate(zip(x.aval.shape, z.shape)):
            if dz > dx and i not in axes:
                more_axes += [i]
        out = out.sum(axes=tuple(more_axes), keepdims=True)
        # print(axes, z.shape, x.aval.shape, more_axes, out.shape)
        if out.shape != x.aval.shape:
            breakpoint()
        return [out]


class Reshape(ShapeOp):
    get_impl = lambda: slope.RT.backend.ReshapeImpl

    @staticmethod
    def eval(x, *, shape):
        return [x.reshape(shape)]

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
    get_impl = lambda: slope.RT.backend.TransposeImpl

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
        return [x.tranpose(perm)], [x_bdim]

    @staticmethod
    def jvp(primals, tangents, *, perm):
        (x,), (x_dot,) = primals, tangents
        return [x.transpose(perm)], [x_dot.transpose(perm)]

    @staticmethod
    def shape_eval(x: ArrayShape, *, perm: Sequence[int]) -> List[ArrayShape]:
        shape = [x.shape[i] for i in perm]
        return [ArrayShape(shape, x.dtype)]

    @staticmethod
    def T(cts, x, *, perm):
        (z,) = cts
        return [z.transpose(perm)]


class Pad(ShapeOp):
    get_impl = lambda: slope.RT.backend.PadImpl

    @staticmethod
    def eval(x, *, lo, hi, interior, value):
        return [x.pad(lo, hi, interior, value)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, perm):
        raise NotImplementedError
        operand, padding_value = batched_args
        operand_bdim, padding_value_bdim = batch_dims
        if operand_bdim is None:
            operand_bdim = 0
            operand = broadcast(operand, (padding_value.shape[padding_value_bdim],))

        padding_config = list(padding_config)
        padding_config.insert(operand_bdim, (0, 0, 0))
        if padding_value_bdim is None:
            return pad(operand, padding_value, padding_config), operand_bdim

        assert padding_value_bdim == 0, padding_value_bdim

        x = pad(operand, _zero(operand), padding_config)
        mask = pad(full_like(operand, True, np.bool_), False, padding_config)
        broadcasted_padding = broadcast_in_dim(padding_value, x.shape, (operand_bdim,))
        return select(mask, x, broadcasted_padding), operand_bdim

    @staticmethod
    def jvp(primals, tangents, *, lo, hi, interior, value):
        (x,), (x_dot,) = primals, tangents
        return [x.pad(lo, hi, interior, value)], [x_dot.pad(lo, hi, interior, value)]

    @staticmethod
    def shape_eval(x: ArrayShape, *, lo, hi, interior, value) -> List[ArrayShape]:
        op_shape = np.shape(x)

        def _dilate_dim(d, dilation):
            return 0 if d == 0 else 1 + dilation * (d - 1)

        shape = (
            sum([l, h, _dilate_dim(d, r + 1)])
            for l, h, r, d in zip(lo, hi, interior, op_shape)
        )
        res = ArrayShape(shape, x.dtype)
        if not all(d >= 0 for d in res.shape):
            raise ValueError(
                f"Dimension size after padding is not at least 0, "
                f"got result shape {res}, for {lo=} {hi=} {interior=} {value=}"
                f"{op_shape=}"
            )
        return [res]

    @staticmethod
    def T(cts, x, *, lo, hi, interior, value):
        (z,) = cts

        def t_op():
            unpadded = z.slice(
                lo,
                tuple(s - h for s, h in zip(z.shape, hi)),
                tuple([1] * len(interior)),
            )
            return unpadded.slice(
                tuple([0] * len(lo)), unpadded.shape, tuple(r + 1 for r in interior)
            )

        res = t_op() if isinstance(x, slope.ad.UndefPrimal) else None
        return [res]


class Slice(ShapeOp):
    get_impl = lambda: slope.RT.backend.SliceImpl

    @staticmethod
    def eval(x, *, starts, limits, strides):
        return [x.slice(starts, limits, strides)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, starts, limits, strides):
        raise NotImplementedError
        (x,) = vals_in
        (x_bdim,) = dims_in

        new_start_indices = list(starts)
        new_start_indices.insert(x_bdim, 0)

        new_limit_indices = list(limits)
        new_limit_indices.insert(x_bdim, x.shape[x_bdim])

        if strides is None:
            new_strides = None
        else:
            new_strides = list(strides)
            new_strides.insert(x_bdim, 1)

        out = x.slice(new_start_indices, new_limit_indices, new_strides)
        return out, x_bdim

    @staticmethod
    def jvp(primals, tangents, *, starts, limits, strides):
        (x,), (x_dot,) = primals, tangents
        return [x.slice(starts, limits, strides)], [
            x_dot.slice(starts, limits, strides)
        ]

    @staticmethod
    def shape_eval(
        x: ArrayShape, *, starts, limits, strides: Sequence[int]
    ) -> List[ArrayShape]:
        if strides is None or tuple(strides) == (1,) * len(x.shape):
            shape = [
                limit if type(start) is int and start == 0 else limit - start
                for start, limit in zip(starts, limits)
            ]
            return [ArrayShape(shape, x.dtype)]
        else:
            # TODO: compute strided shape without numpy
            x = np.zeros_like(x.shape)
            x = x[tuple(slice(s, l, r) for s, l, r in zip(starts, limits, strides))]
            return [ArrayShape(x.shape, x.dtype)]

    @staticmethod
    def T(cts, x, *, starts, limits, strides):
        # TODO: compute tuple arithmetic numpy
        (z,) = cts
        x_shape = x.aval.shape
        assert isinstance(x, slope.ad.UndefPrimal)
        if strides is None or np.all(np.equal(strides, 1)):
            lo, hi, interior = (
                starts,
                np.subtract(x.aval.shape, limits),
                (0,) * len(starts),
            )
        else:
            real_limits = np.add(
                starts,
                np.where(
                    np.array(x.shape) == 0,
                    0,
                    np.add(1, np.multiply(np.subtract(t.shape, 1), strides)),
                ),
            )
            lo, hi, interior = utils.list_zip(
                starts, np.subtract(x_shape, real_limits), np.subtract(strides, 1)
            )
        res = z.pad(lo, hi, interior)
        assert res.shape == x_shape, f"{res.shape=} {x_shape=}"
        return [res]


class Flip(ShapeOp):
    get_impl = lambda: slope.RT.backend.FlipImpl

    @staticmethod
    def eval(x, *, axes):
        return [x.flip(axes)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, perm):
        raise NotImplementedError

    @staticmethod
    def jvp(primals, tangents, *, axes):
        (x,), (x_dot,) = primals, tangents
        return [x.flip(axes)], [x_dot.flip(axes)]

    @staticmethod
    def shape_eval(x: ArrayShape, *, padding: Sequence[int]) -> List[ArrayShape]:
        shape = [x.shape[i] for i in padding]
        return [ArrayShape(shape, x.dtype)]

    @staticmethod
    def T(cts, *, axes):
        (z,) = cts
        return [z.flip(axes)]


class Concatenate(ShapeOp):
    get_impl = lambda: slope.RT.backend.ConcatenateImpl

    @staticmethod
    def eval(xs: Sequence[Any],*, axis):
        return [Array.concatenate(xs, axis=axis)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, perm):
        raise NotImplementedError

    @staticmethod
    def jvp(primals, tangents, *, axis):
        (xs,), (xs_dot,) = primals, tangents
        return [Array.concatenate(xs, axis=axis)], [
            Array.concatenate(xs_dot, axis=axis)
        ]

    @staticmethod
    def shape_eval(x: ArrayShape, idx, *, axis: Sequence[int]) -> List[ArrayShape]:
        shape = [x.shape[i] for i in axis]
        return [ArrayShape(shape, x.dtype)]

    @staticmethod
    def T(cts, xs, *, axis):
        (zs,) = cts
        return [zs.slice(x.shape) for x in xs]


#


class Constant(LoadOp):
    get_impl = lambda: slope.RT.backend.ConstantImpl

    @staticmethod
    def load_fn(*, val, dtype):
        return Array(val, dtype)

    @staticmethod
    def shape_eval(*, val, dtype) -> List[ArrayShape]:
        # TODO: not using numpy to extract shape
        return [ArrayShape(np.array(val).shape, dtype)]


class Full(LoadOp):
    get_impl = lambda: slope.RT.backend.FullImpl

    @staticmethod
    def load_fn(*, fill_value, shape, dtype):
        return Array.full(fill_value, shape, dtype)

    @staticmethod
    def shape_eval(*, fill_value, shape, dtype) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), dtype)]


class RandomUniform(LoadOp):
    get_impl = lambda: slope.RT.backend.RandomUniformImpl

    @staticmethod
    def load_fn(*, shape, dtype):
        return Array.random_uniform(shape, dtype)

    @staticmethod
    def shape_eval(*, shape, dtype) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), dtype)]


class RandomNormal(LoadOp):
    get_impl = lambda: slope.RT.backend.RandomNormalImpl

    @staticmethod
    def load_fn(*, shape, dtype):
        return Array.random_normal(shape, dtype)

    @staticmethod
    def shape_eval(*, shape, dtype) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), dtype)]


class Arange(LoadOp):
    get_impl = lambda: slope.RT.backend.ArangeImpl

    @staticmethod
    def load_fn(*, start, stop, stride, dtype):
        return Array.arange(start, stop, stride, dtype)

    @staticmethod
    def shape_eval(start, stop, stride, dtype) -> List[ArrayShape]:
        return [ArrayShape(len(tuple(slice(start, stop, stride))), dtype)]


class Jit(Op):
    @staticmethod
    def eval(*args, hashable_prog, hashable_consts):
        jit_fn = slope.RT.backend.callable(hashable_prog, hashable_consts)
        return [jit_fn(*args)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, x):
        raise NotImplementedError

    @staticmethod
    def jvp(*, x):
        raise NotImplementedError

    @staticmethod
    def shape_eval(*, x) -> List[ArrayShape]:
        raise NotImplementedError

    @staticmethod
    def T(cts, *, x):
        raise NotImplementedError

# Gather and Scatter


class Gather(ShapeOp):
    get_impl = lambda: slope.RT.backend.GatherImpl

    @staticmethod
    def eval(x, idx, *, axis):
        return [x.gather(idx)]

    
    @staticmethod
    def _gather_batching_rule(axis_size, vals_in, dims_in,
        # batched_args,
        # batch_dims,
        *,
        dimension_numbers,
        slice_sizes,
        unique_indices,
        indices_are_sorted,
        mode,
        fill_value,
    ):
        operand, indices = batched_args
        operand_bdim, indices_bdim = batch_dims

        if operand_bdim is not None and indices_bdim is None:
            operand = batching.moveaxis(operand, operand_bdim, 0)
            slice_sizes = (operand.shape[0],) + slice_sizes
            offset_dims = (0,) + tuple(np.add(1, dimension_numbers.offset_dims))
            collapsed_slice_dims = tuple(np.add(1, dimension_numbers.collapsed_slice_dims))
            start_index_map = tuple(np.add(1, dimension_numbers.start_index_map))
            dnums = GatherDimensionNumbers(
                offset_dims=offset_dims,
                collapsed_slice_dims=collapsed_slice_dims,
                start_index_map=start_index_map,
            )
            return (
                gather(
                    operand,
                    indices,
                    dimension_numbers=dnums,
                    slice_sizes=slice_sizes,
                    unique_indices=unique_indices,
                    indices_are_sorted=indices_are_sorted,
                    mode=mode,
                    fill_value=fill_value,
                ),
                0,
            )

        elif operand_bdim is None and indices_bdim is not None:
            indices = batching.moveaxis(indices, indices_bdim, 0)
            offset_dims = tuple(1 + d for d in dimension_numbers.offset_dims)
            dnums = GatherDimensionNumbers(
                offset_dims=offset_dims,
                collapsed_slice_dims=dimension_numbers.collapsed_slice_dims,
                start_index_map=dimension_numbers.start_index_map,
            )
            # If batching indexed accesses into the same array, the batched gather may
            # no longer have sorted or unique indices.
            return (
                gather(
                    operand,
                    indices,
                    dimension_numbers=dnums,
                    slice_sizes=slice_sizes,
                    unique_indices=False,
                    indices_are_sorted=False,
                    mode=mode,
                    fill_value=fill_value,
                ),
                0,
            )

        else:
            # move batch dimensions to the front to simplify logic
            operand = batching.moveaxis(operand, operand_bdim, 0)
            indices = batching.moveaxis(indices, indices_bdim, 0)

            # This slightly awkward special case is needed because the shape rule for
            # gather does not allow size-1 slices out of a size-0 dimension, even if
            # the number of slices is zero. Likely the best fix would be to change the
            # definition of gather() so it can be batched without the construction of
            # an explicit iota of size-1 slices.
            if core.symbolic_equal_dim(operand.shape[0], 0):
                output_shape = _gather_shape_rule(
                    core.ShapedArray(operand.shape[1:], operand.dtype),
                    core.ShapedArray(
                        indices.shape[1:], dtypes.canonicalize_dtype(indices.dtype)
                    ),
                    dimension_numbers=dimension_numbers,
                    slice_sizes=slice_sizes,
                    unique_indices=unique_indices,
                    indices_are_sorted=indices_are_sorted,
                    mode=mode,
                    fill_value=fill_value,
                )
                return lax.full((0,) + output_shape, lax._zero(operand)), 0

            # Example: user code had indices shape (3, 4, 5), and we have to deal with
            # indices shape (7, 3, 4, 5). We transform that to indices of shape
            # (7, 3, 4, 6) where we concatenated an iota that counts along our batch
            # dimension to the front of the ndindex.
            count_shape = list(indices.shape)
            count_shape[-1] = 1
            counts = lax.broadcasted_iota(indices.dtype, tuple(count_shape), 0)
            indices = lax.concatenate([counts, indices], len(count_shape) - 1)

            slice_sizes = (1,) + slice_sizes
            collapsed_slice_dims = (0,) + tuple(
                np.add(1, dimension_numbers.collapsed_slice_dims)
            )
            offset_dims = tuple(np.add(1, dimension_numbers.offset_dims))
            start_index_map = (0,) + tuple(np.add(1, dimension_numbers.start_index_map))

            dnums = GatherDimensionNumbers(
                offset_dims=offset_dims,
                collapsed_slice_dims=collapsed_slice_dims,
                start_index_map=start_index_map,
            )
            return (
                gather(
                    operand,
                    indices,
                    dimension_numbers=dnums,
                    slice_sizes=slice_sizes,
                    unique_indices=unique_indices,
                    indices_are_sorted=indices_are_sorted,
                    mode=mode,
                    fill_value=fill_value,
                ),
                0,
            )


    @staticmethod
    def jvp(primals, tangents, indices,
    # g,
    # operand,
    *,
    dimension_numbers,
    slice_sizes,
    unique_indices,
    indices_are_sorted,
    mode,
    fill_value,
    ):
        return gather(
        g,
        indices,
        dimension_numbers,
        slice_sizes,
        unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted,
        mode=mode,
        fill_value=0,
    )


    @staticmethod
    def shape_eval(x: ArrayShape, idx, *,  dimension_numbers,
                       slice_sizes, unique_indices, indices_are_sorted,
                       mode, fill_value) -> List[ArrayShape]:
        offset_dims = dimension_numbers.offset_dims
        collapsed_slice_dims = dimension_numbers.collapsed_slice_dims
        start_index_map = dimension_numbers.start_index_map

        # Note: in JAX, index_vector_dim is always computed as below, cf. the
        # documentation of the GatherDimensionNumbers class.
        index_vector_dim = _rank(indices) - 1

        # This case should never happen in JAX, due to the implicit construction of
        # index_vector_dim, but is included for completeness.
        if _rank(indices) < index_vector_dim or index_vector_dim < 0:
            raise TypeError(f"Gather index leaf dimension must be within [0, rank("
                    f"indices) + 1). rank(indices) is {_rank(indices)} and "
                    f"gather index leaf dimension is {index_vector_dim}.")

        expanded_indices_shape = list(indices.shape)

  # This case should never happen in JAX, due to the implicit construction of
  # index_vector_dim, but is included for completeness.
        if len(expanded_indices_shape) == index_vector_dim:
            expanded_indices_shape.append(1)

  # Start ValidateGatherDimensions
  # In the error messages output by XLA, "offset_dims" is called "Output window
  # dimensions" in error messages. For consistency's sake, our error messages
  # stick to "offset_dims".
        _is_sorted(offset_dims, "gather", "offset_dims")
        _no_duplicate_dims(offset_dims, "gather", "offset_dims")

        output_offset_dim_count = len(offset_dims)
        output_shape_rank = len(offset_dims) + _rank(indices) - 1

        for i in range(output_offset_dim_count):
            offset_dim = offset_dims[i]
            if offset_dim < 0 or offset_dim >= output_shape_rank:
                raise TypeError(f"Offset dimension {i} in gather op is out of bounds; "
                      f"got {offset_dim}, but should have been in "
                      f"[0, {output_shape_rank})")

            if len(start_index_map) != indices.shape[index_vector_dim]:
                raise TypeError(f"Gather op has {len(start_index_map)} elements in "
                    f"start_index_map and the bound of dimension "
                    f"{index_vector_dim=} of indices is "
                    f"{indices.shape[index_vector_dim]}. These two "
                    f"numbers must be equal.")

            for i in range(len(start_index_map)):
                operand_dim_for_start_index_i = start_index_map[i]
                if (operand_dim_for_start_index_i < 0 or
                    operand_dim_for_start_index_i >= _rank(operand)):
                    raise TypeError(f"Invalid start_index_map; domain is "
                      f"[0, {_rank(operand)}), got: "
                      f"{i}->{operand_dim_for_start_index_i}.")

            _no_duplicate_dims(start_index_map, "gather", "start_index_map")

  # _is_sorted and _sorted_dims_in_range are checked in the opposite order
  # compared to the XLA implementation. In cases when the input is not sorted
  # AND there are problematic collapsed_slice_dims, the error message will thus
  # be different.
        _is_sorted(collapsed_slice_dims, "gather", "collapsed_slice_dims")
        _sorted_dims_in_range(collapsed_slice_dims, _rank(operand), "gather",
                                "collapsed_slice_dims")
        _no_duplicate_dims(collapsed_slice_dims, "gather", "collapsed_slice_dims")
        # End ValidateGatherDimensions

        if _rank(operand) != len(slice_sizes):
            raise TypeExrror(f"Gather op must have one slice size for every input "
                    f"dimension; got: len(slice_sizes)={len(slice_sizes)}, "
                    f"input_shape.rank={_rank(operand)}")

        if len(slice_sizes) != len(offset_dims) + len(collapsed_slice_dims):
            raise TypeError(f"All components of the offset index in a gather op must "
                    f"either be a offset dimension or explicitly collapsed; "
                    f"got len(slice_sizes)={len(slice_sizes)}, "
                    f"output_slice_sizes={offset_dims}, collapsed_slice_dims="
                    f"{collapsed_slice_dims}.")

        for i in range(len(slice_sizes)):
            slice_size = slice_sizes[i]
            corresponding_input_size = operand.shape[i]

            if not (core.greater_equal_dim(slice_size, 0) and
            core.greater_equal_dim(corresponding_input_size, slice_size)):
                raise TypeError(f"Slice size at index {i} in gather op is out of range, "
                      f"must be within [0, {corresponding_input_size} + 1), "
                      f"got {slice_size}.")

        for i in range(len(collapsed_slice_dims)):
            bound = slice_sizes[collapsed_slice_dims[i]]
            if bound != 1:
                raise TypeError(f"Gather op can only collapse slice dims with bound 1, "
                      f"but bound is {bound} for index "
                      f"{collapsed_slice_dims[i]} at position {i}.")

        expanded_indices_shape.pop(index_vector_dim)
        indices_shape = iter(expanded_indices_shape)

        slice_sizes = (s for i, s in enumerate(slice_sizes)
                 if i not in collapsed_slice_dims)
        return tuple(next(slice_sizes) if i in offset_dims
               else next(indices_shape) for i in range(output_shape_rank))


    @staticmethod
    def T(
        cts,
        operand,
        indices,
        *,
        dimension_numbers,
        slice_sizes,
        unique_indices,
        indices_are_sorted,
        mode,
        fill_value,
    ):
        assert ad.is_undefined_primal(operand)
        operand_shape = operand.aval.shape
        if type(t) is ad_util.Zero:
            out = ad_util.Zero(operand.aval)
        else:
            zeros = lax.full(operand_shape, lax._zero(t))
            scatter_dnums = ScatterDimensionNumbers(
                update_window_dims=dimension_numbers.offset_dims,
                inserted_window_dims=dimension_numbers.collapsed_slice_dims,
                scatter_dims_to_operand_dims=dimension_numbers.start_index_map,
            )
            out = scatter_add(
                zeros,
                indices,
                t,
                scatter_dnums,
                unique_indices=unique_indices,
                indices_are_sorted=indices_are_sorted,
                mode=mode,
            )
        return [out, None]





class Scatter(ShapeOp):
    get_impl = lambda: slope.RT.backend.ScatterImpl

    @staticmethod
    def eval(x, idx, *, axis):
        return [x.gatter(idx)]

    @staticmethod
    def vmap(
            axis_size, vals_in, dims_in, 
        # scatter_op,
        # batched_args,
        # batch_dims,
        *,
        update_jaxpr,
        update_consts,
        dimension_numbers,
        indices_are_sorted,
        unique_indices,
        mode,
    ):
        operand, indices, updates = batched_args
        operand_bdim, indices_bdim, updates_bdim = batch_dims
        del update_jaxpr, update_consts  # Unused.

        # move the operand batch dim to the front if it is not None, otherwise create
        # it at the front (so that we can scatter into it)
        size = next(
            x.shape[ax] for x, ax in zip(batched_args, batch_dims) if ax is not None
        )
        operand = batching.bdim_at_front(operand, operand_bdim, size)
        operand_bdim = 0

        updates = batching.bdim_at_front(updates, updates_bdim, size)

        if indices_bdim is None:
            inserted_window_dims = tuple(np.add(1, dimension_numbers.inserted_window_dims))
            update_window_dims = (0,) + tuple(
                np.add(1, dimension_numbers.update_window_dims)
            )
            scatter_dims_to_operand_dims = tuple(
                np.add(1, dimension_numbers.scatter_dims_to_operand_dims)
            )
            dnums = ScatterDimensionNumbers(
                update_window_dims=update_window_dims,
                inserted_window_dims=inserted_window_dims,
                scatter_dims_to_operand_dims=scatter_dims_to_operand_dims,
            )
            return (
                scatter_op(
                    operand,
                    indices,
                    updates,
                    dnums,
                    indices_are_sorted=indices_are_sorted,
                    unique_indices=unique_indices,
                    mode=mode,
                ),
                0,
            )

        # see the third case in _gather_batching_rule for comparison and comments
        indices = batching.bdim_at_front(indices, indices_bdim, size)

        count_shape = list(indices.shape)
        count_shape[-1] = 1
        counts = lax.broadcasted_iota(indices.dtype, tuple(count_shape), 0)
        indices = lax.concatenate([counts, indices], len(count_shape) - 1)

        update_window_dims = tuple(np.add(1, dimension_numbers.update_window_dims))
        inserted_window_dims = (0,) + tuple(
            np.add(1, dimension_numbers.inserted_window_dims)
        )
        scatter_dims_to_operand_dims = (0,) + tuple(
            np.add(1, dimension_numbers.scatter_dims_to_operand_dims)
        )

        dnums = ScatterDimensionNumbers(
            update_window_dims=update_window_dims,
            inserted_window_dims=inserted_window_dims,
            scatter_dims_to_operand_dims=scatter_dims_to_operand_dims,
        )
        return (
            scatter_op(
                operand,
                indices,
                updates,
                dnums,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
            ),
            0,
        )

    @staticmethod
    def jvp(
        primals,
        tangents,
        *,
        update_jaxpr,
        update_consts,
        dimension_numbers,
        indices_are_sorted,
        unique_indices,
        mode,
    ):
        operand, indices, updates = primals
        g_operand, g_indices, g_updates = tangents
        del g_indices  # ignored
        val_out = scatter_add_p.bind(
            operand,
            indices,
            updates,
            update_jaxpr=update_jaxpr,
            update_consts=update_consts,
            dimension_numbers=dimension_numbers,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        )
        if type(g_operand) is ad_util.Zero and type(g_updates) is ad_util.Zero:
            tangent_out = ad_util.Zero.from_value(val_out)
        else:
            g_operand = ad.instantiate_zeros(g_operand)
            g_updates = ad.instantiate_zeros(g_updates)
            tangent_out = scatter_add_p.bind(
                g_operand,
                indices,
                g_updates,
                update_jaxpr=update_jaxpr,
                update_consts=update_consts,
                dimension_numbers=dimension_numbers,
                indices_are_sorted=indices_are_sorted,
                unique_indices=unique_indices,
                mode=mode,
            )
        return val_out, tangent_out


    @staticmethod
    def shape_eval(x: ArrayShape, idx, *, axis: Sequence[int]) -> List[ArrayShape]:
        shape = [x.shape[i] for i in axis]
        return [ArrayShape(shape, x.dtype)]

    @staticmethod
    def T(cts,operand,
        indices,
        updates, *,
        update_jaxpr,
        update_consts,
        dimension_numbers,
        indices_are_sorted,
        unique_indices,
        mode,
    ):
        assert not ad.is_undefined_primal(indices)
        if ad.is_undefined_primal(updates):
            updates_shape = updates.aval.shape
        else:
            updates_shape = updates.shape
        if type(t) is ad_util.Zero:
            operand_t = (
                ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
            )
            update_t = (
                ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
            )
        else:
            operand_t = update_t = None
            if ad.is_undefined_primal(operand):
                operand_t = t

            if ad.is_undefined_primal(updates):
                gather_dnums = GatherDimensionNumbers(
                    offset_dims=dimension_numbers.update_window_dims,
                    collapsed_slice_dims=dimension_numbers.inserted_window_dims,
                    start_index_map=dimension_numbers.scatter_dims_to_operand_dims,
                )
                slice_sizes = []
                pos = 0
                for i in range(len(t.shape)):
                    if i in dimension_numbers.inserted_window_dims:
                        slice_sizes.append(1)
                    else:
                        slice_sizes.append(
                            updates_shape[dimension_numbers.update_window_dims[pos]]
                        )
                        pos += 1
                update_t = gather(
                    t,
                    indices,
                    dimension_numbers=gather_dnums,
                    slice_sizes=slice_sizes,
                    mode=mode,
                    fill_value=0,
                )
        return [operand_t, None, update_t]