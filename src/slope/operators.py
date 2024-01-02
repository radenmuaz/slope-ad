import slope
import slope.core
from slope.core import (
    Operator,
    UnaryOperator,
    BinaryOperator,
    ReduceOperator,
    ShapeOperator,
    InitOperator,
    BinaryReduceOperator,
    OperatorSet,
    Tensor,
    VoidTensor,
    UndefPrimal,
    list_zip,
)

import math
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Sequence, Union, Iterator, NamedTuple
from collections import defaultdict
import iree.compiler
import iree.runtime
import os


# --------------
# Operator
# --------------

operator_set = OperatorSet()

# -----------------------
# Unary
# -----------------------


@operator_set.register("stop_gradient")
class StopGradient(UnaryOperator):
    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [x], [slope.zeros_like(x_dot)]

    def T(self, cotangents, x):
        return [None]


@operator_set.register("cast", aliases=["astype"])
class Cast(UnaryOperator):
    def typecheck(self, x: VoidTensor, *, dtype) -> List[VoidTensor]:
        return [VoidTensor(x.shape, dtype)]

    def jvp(self, primals, tangents, *, dtype):
        (x,), (x_dot,) = primals, tangents
        return [x.cast(dtype)], [x_dot.cast(dtype)]

    def T(self, cotangents, x):
        (grad_L_y,) = cotangents
        return [grad_L_y.cast(x.dtype)]


@operator_set.register("sqrt")
class Sqrt(UnaryOperator):
    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        y = x.sqrt()
        return [y], [x_dot / (y * 2)]

    def T(self, cotangents, x):
        (grad_L_y,) = cotangents
        return [grad_L_y / (x.sqrt() * 2)]


@operator_set.register("sin")
class Sin(UnaryOperator):
    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [x.sin()], [(x_dot * ((math.pi / 2) - x).sin())]

    def T(self, cotangents, x):
        (grad_L_y,) = cotangents
        return [(grad_L_y * ((math.pi / 2) - x).sin())]


@operator_set.register("exp")
class Exp(UnaryOperator):
    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        y = x.exp()
        return [y], [x_dot * y]

    def T(self, cotangents, x):
        (grad_L_y,) = cotangents
        return [1 / grad_L_y]


@operator_set.register("log")
class Log(UnaryOperator):
    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [x.log()], [x_dot / x]

    def T(self, cotangents, x):
        (grad_L_y,) = cotangents
        return [1 / grad_L_y]


@operator_set.register("invert")
class Invert(UnaryOperator):
    def typecheck(self, x, **params):
        return [VoidTensor(x.shape, slope.bool)]

    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [~x], [~x_dot]

    def T(self, cotangents, x):
        (grad_L_y,) = cotangents
        return [~grad_L_y]


# -----------------------
# Binary
# -----------------------


@operator_set.register("add")
class Add(BinaryOperator):
    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [x + w], [x_dot + w_dot]

    def T(self, cotangents, x, w):
        (grad_L_y,) = cotangents
        return [grad_L_y, grad_L_y]


@operator_set.register("sub")
class Sub(BinaryOperator):
    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [x - w], [x_dot - w_dot]

    def T(self, cotangents, x, w):
        (grad_L_y,) = cotangents
        return [grad_L_y, -grad_L_y]


@operator_set.register("mul")
class Mul(BinaryOperator):
    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [x * w], [(x_dot * w) + (w_dot * x)]

    def T(self, cotangents, x, w):
        (grad_L_y,) = cotangents
        assert (type(x) is UndefPrimal) ^ (type(w) is UndefPrimal)
        if type(x) is UndefPrimal:
            return [grad_L_y * w, None]
        elif type(w) is UndefPrimal:
            return [None, x * grad_L_y]


@operator_set.register("div")
class Div(BinaryOperator):
    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [x / w], [(x_dot / w) + (-w_dot * x * 1 / (w * w))]

    def T(self, cotangents, x, w):
        (grad_L_y,) = cotangents
        return [grad_L_y / w, None]


@operator_set.register("pow")
class Pow(BinaryOperator):
    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        y = x**w
        y_dot1 = x_dot * (w * (x ** (w - slope.ones_like(w))))
        y_dot2 = w_dot * (y * (x if x != 0.0 else slope.zeros_like(x)).log())
        return [y], [y_dot1 + y_dot2]

    def T(self, cotangents, x, w):
        (grad_L_y,) = cotangents
        assert (type(x) is UndefPrimal) ^ (type(w) is UndefPrimal)
        if type(x) is UndefPrimal:
            return [(grad_L_y * (w * (x ** (w - slope.ones_like(w))))), None]
        elif type(w) is UndefPrimal:
            return [None, grad_L_y * ((x**w) * (x.log() if x != 0.0 else slope.zeros_like(x)))]


@operator_set.register("maximum")
class Maximuum(BinaryOperator):
    def jvp(self, primals, tangents):
        def _balanced_eq(x, z, y):
            xz = (x == z).where(slope.ones_like(z), slope.zeros_like(z))
            yz = (y == z).where(slope.full_like(z, 2), slope.ones_like(z))
            return xz / yz

        (x, w), (x_dot, w_dot) = primals, tangents
        y = x.maximum(w)
        y_dot = x_dot * _balanced_eq(x, y, w) + w_dot * _balanced_eq(w, y, x)
        return [y], [y_dot]

    def T(self, cotangents, x, w):
        (grad_L_y,) = cotangents
        return [grad_L_y, None]


@operator_set.register("equal")
class Equal(BinaryOperator):
    def typecheck(self, x: VoidTensor, y: VoidTensor, **params) -> List[VoidTensor]:
        # difference with default binary typecheck: force dtype bool
        if not type(x) in (Tensor, VoidTensor) or not type(x) in (
            Tensor,
            VoidTensor,
        ):
            raise TypeError
        if x.dtype != y.dtype:
            raise TypeError
        void_x = VoidTensor.like(x)
        void_y = VoidTensor.like(y)
        if void_x == void_y:
            return [VoidTensor(void_x.shape, Tensor.bool)]
        shape_delta = len(void_x.shape) - len(void_y.shape)
        if shape_delta > 0:
            void_y = VoidTensor((1,) * shape_delta + void_y.shape, Tensor.bool)
        elif shape_delta < 0:
            x = x.reshape((1,) * -shape_delta + void_x.shape)
            void_x = VoidTensor((1,) * -shape_delta + void_x.shape, Tensor.bool)
        if void_x == void_y:
            return [void_x]
        else:
            shape_ret = tuple([max(x, w) for x, w in zip(void_x.shape, void_y.shape)])
            if void_x.shape != shape_ret:
                void_x = VoidTensor(shape_ret, Tensor.bool)
            if void_y.shape != shape_ret:
                void_y = VoidTensor(shape_ret, Tensor.bool)
            if void_x != void_y:
                raise TypeError
            return [void_x]

    def jvp(self, primals, tangents):
        (x, w), _ = primals, tangents
        out_primal = x.equal(w)
        return [out_primal], [slope.full(out_primal.shape, True, Tensor.bool)]

    def T(self, cotangents, x, w):
        (grad_L_y,) = cotangents
        grad_L_y = grad_L_y.cast(x.dtype)
        return [grad_L_y, None]


@operator_set.register("max")
class Max(ReduceOperator):
    def jvp(self, primals, tangents, *, dim, keepdim):
        (x,), (x_dot,) = primals, tangents
        y = x.max(dim, keepdim)
        y_ = y
        if not keepdim:
            dim = tuple([a if a >= 0 else len(y.shape) + a + 1 for a in dim])
            for a in reversed(sorted(dim)):
                y_ = y_.reshape(y.shape[:a] + (1,) + y.shape[a:])
        locs = x.equal(y_.expand(x.shape))
        locs = locs.cast(x_dot.dtype)
        counts = locs.sum(dim, keepdim)
        y_dot = (x_dot * locs).sum(dim, keepdim)
        y_dot = y_dot / counts.expand(y_dot.shape)

        return [y], [y_dot]

    def T(self, cotangents, x, *, dim, keepdim):
        # TODO: this is sum gradient, define max gradient
        (grad_L_y,) = cotangents
        grad_L_x = grad_L_y
        if not keepdim:
            dim = [a if a >= 0 else len(grad_L_x.shape) + a + 1 for a in dim]
            for a in reversed(sorted(dim)):
                grad_L_x = grad_L_x.reshape(grad_L_x.shape[:a] + (1,) + grad_L_x.shape[a:])
        grad_L_x = grad_L_x.expand(x.void_tensor.shape)


@operator_set.register("sum")
class Sum(ReduceOperator):
    def jvp(self, primals, tangents, *, dim, keepdim):
        (x,), (x_dot,) = primals, tangents
        y = x.sum(dim, keepdim)
        y_dot = x_dot.sum(dim, keepdim)
        return [y], [y_dot]

    def T(self, cotangents, x, *, dim, keepdim):
        (grad_L_y,) = cotangents
        grad_L_x = grad_L_y
        if not keepdim:
            dim = [a if a >= 0 else len(grad_L_x.shape) + a + 1 for a in dim]
            for a in reversed(sorted(dim)):
                grad_L_x = grad_L_x.reshape(grad_L_x.shape[:a] + (1,) + grad_L_x.shape[a:])
        grad_L_x = grad_L_x.expand(x.void_tensor.shape)

        return [grad_L_x]


# -----------------------
# Shape
# -----------------------


@operator_set.register("expand")
class Expand(ShapeOperator):
    def args_fixer(self, x, *args, **kwargs):
        if "shape" in kwargs.keys():
            shape = kwargs["shape"]
        elif isinstance(args[0], (tuple, list)):
            shape = args[0]
        else:
            shape = args
        shape = tuple(shape)
        if x.shape in ((), (1,)):
            x = x.reshape((1,) * len(shape))
        return (x,), dict(shape=shape)

    def typecheck(self, x: VoidTensor, *, shape: Sequence[int]) -> List[VoidTensor]:
        shape = tuple(shape)
        assert len(x.shape) == len(shape)
        assert all(a <= b for a, b in zip(x.shape, shape))
        return [VoidTensor(tuple(shape), x.dtype)]

    def vmap(self, dim_size, vals_in, dims_in, *, shape):
        (x,), (x_bdim,) = vals_in, dims_in
        shape = shape[:x_bdim] + (dim_size,) + shape[x_bdim:]
        return [self(x, shape)], [x_bdim]

    def jvp(self, primals, tangents, *, shape, dim=None):
        (x,), (x_dot,) = primals, tangents
        return (
            [self(x, shape=shape)],
            [self(x_dot, shape=shape)],
        )

    def T(self, cotangents, x, *, shape):
        (grad_L_y,) = cotangents
        grad_L_x = grad_L_y
        if x.void_tensor.shape == grad_L_x.shape:
            return [grad_L_x]
        else:
            b_dim = []
            assert len(x.void_tensor.shape) == len(grad_L_x.shape)
            for i, (xd, od) in enumerate(zip(x.void_tensor.shape, grad_L_x.shape)):
                if xd != od:
                    b_dim += [i]
            grad_L_x = grad_L_x.sum(dim=tuple(b_dim), keepdim=True)
        if grad_L_x.shape != x.void_tensor.shape:
            raise ValueError(f"not same {grad_L_x.shape=}, {x.void_tensor.shape=}")
        return [grad_L_x]


@operator_set.register("reshape", nary_inputs=True, aliases=["view"])
class Reshape(ShapeOperator):
    def args_fixer(self, x, *args, **kwargs):
        if "shape" in kwargs.keys():
            shape = kwargs["shape"]
        elif isinstance(args[0], (tuple, list)):
            shape = args[0]
        else:
            shape = args
        shape = tuple(shape)
        if -1 in shape:
            others = math.prod([d for d in shape if d != -1])
            numel = math.prod(x.shape)
            shape = tuple(d if d != -1 else (numel // others) for d in shape)
        return (x,), dict(shape=shape)

    def vmap(self, dim_size, vals_in, dims_in, *, shape):
        (x,), (x_bdim,) = vals_in, dims_in
        x = slope.core.VMapTrace.move_vmap_dim(x, dim_size, x_bdim, 0)
        y = self(x, tuple(x.shape[:1] + shape))
        return [y], [x_bdim]

    def typecheck(self, x: VoidTensor, *, shape: Sequence[int]) -> List[VoidTensor]:
        return [VoidTensor(tuple(shape), x.dtype)]

    def jvp(self, primals, tangents, *, shape):
        (x,), (x_dot,) = primals, tangents
        return [x.reshape(shape)], [x_dot.reshape(shape)]

    def T(self, cotangents, x, *, shape):
        (z,) = cotangents
        return [z.reshape(x.void_tensor.shape)]


@operator_set.register("permute")
class Permute(ShapeOperator):
    def typecheck(self, x: VoidTensor, *, perm: Sequence[int]) -> List[VoidTensor]:
        shape = [x.shape[i] for i in perm]
        return [VoidTensor(shape, x.dtype)]

    def vmap(self, dim_size, vals_in, dims_in, *, perm):
        (x,), (x_bdim,) = vals_in, dims_in
        assert x_bdim >= 0
        perm = perm[:x_bdim] + [x_bdim] + perm[x_bdim:]
        perm = tuple(d + int(d >= x_bdim) if i != x_bdim else d for i, d in enumerate(perm))
        assert len(set(perm)) == len(perm)
        return [x.permute(perm)], [x_bdim]

    def jvp(self, primals, tangents, *, perm):
        (x,), (x_dot,) = primals, tangents
        return [x.permute(perm)], [x_dot.permute(perm)]

    def T(self, cotangents, x, *, perm):
        (z,) = cotangents
        inv_perm = tuple(i[0] for i in sorted(enumerate(perm), key=lambda x: x[1]))
        return [z.permute(inv_perm)]


@operator_set.register("pad")
class Pad(ShapeOperator):
    def args_fixer(self, x, *, padding, mode="constant", value=0.0):
        if isinstance(padding, int):
            padding = (padding, padding) * x.ndim
        else:
            padding = tuple(padding)
        if (x.ndim * 2) != len(padding):
            assert len(padding) % 2 == 0
            padding += (0, 0) * (x.ndim - len(padding) // 2)
        assert (x.ndim * 2) % len(padding) == 0
        return (x,), dict(padding=padding, mode=mode, value=value)

    def typecheck(self, x: VoidTensor, *, padding, mode, value) -> List[VoidTensor]:
        padding = padding[::-1]
        lo, hi = padding[0::2], padding[1::2]
        interior = [0] * (len(padding) // 2)

        def _dilate_dim(d, dilation):
            return 0 if d == 0 else 1 + dilation * (d - 1)

        shape = tuple(sum([l, h, _dilate_dim(d, r + 1)]) for l, h, r, d in list_zip(lo, hi, interior, x.shape))
        if not all(d >= 0 for d in shape):
            raise ValueError(
                f"Dimension size after padding is not at least 0, "
                f"got result shape {res}, for {lo=} {hi=} {interior=} {value=}"
                f"{shape=}"
            )
        res = VoidTensor(shape, x.dtype)
        return [res]

    def vmap(self, dim_size, vals_in, dims_in, *, padding, mode, value):
        (x,), (x_bdim,) = vals_in, dims_in
        x = slope.core.VMapTrace.move_vmap_dim(x, dim_size, x_bdim, 0)
        y = self(x, padding + (0, 0), mode, value)
        return [y], [x_bdim]

    def jvp(self, primals, tangents, *, padding, mode, value):
        (x,), (x_dot,) = primals, tangents
        return [x.pad(padding, mode, value)], [x_dot.pad(padding, mode, value)]

    def T(self, cotangents, x, *, padding, mode, value):
        (z,) = cotangents
        lo, hi = padding[0::2], padding[1::2]
        interior = [0] * (len(padding) // 2)

        if isinstance(x, UndefPrimal):
            unpadded = z.slice(
                lo,
                tuple(s - h for s, h in list_zip(z.shape, hi)),
                tuple([1] * len(interior)),
            )
            res = unpadded.slice(tuple([0] * len(lo)), unpadded.shape, tuple(r + 1 for r in interior))
        else:
            res = None
        return [res]


@operator_set.register("slice")
class Slice(ShapeOperator):
    def args_fixer(self, x, *, starts, limits, strides=None):
        if strides is None:
            strides = (1,) * len(starts)
        return (x,), dict(starts=starts, limits=limits, strides=strides)

    def typecheck(self, x: VoidTensor, *, starts, limits, strides=None) -> List[VoidTensor]:
        if strides is None or tuple(strides) == (1,) * len(x.shape):
            shape = tuple(
                [
                    limit if type(start) is int and start == 0 else limit - start
                    for start, limit in list_zip(starts, limits)
                ]
            )
            return [VoidTensor(shape, x.dtype)]
        else:
            # TODO: compute strided shape without numpy
            x = np.zeros(x.shape)
            x = x[tuple(slice(s, l, r) for s, l, r in list_zip(starts, limits, strides))]
            return [VoidTensor(x.shape, x.dtype)]

    def vmap(self, dim_size, vals_in, dims_in, *, starts, limits, strides):
        (x,), (x_bdim,) = vals_in, dims_in
        x = slope.core.VMapTrace.move_vmap_dim(x, dim_size, x_bdim, 0)
        y = self((0,) + starts, (x.shape[0],) + limits, (1,) + strides)
        return [y], [x_bdim]

    def jvp(self, primals, tangents, *, starts, limits, strides=None):
        (x,), (x_dot,) = primals, tangents
        return [x.slice(starts, limits, strides)], [x_dot.slice(starts, limits, strides)]

    def T(self, cotangents, x, *, starts, limits, strides=None):
        # TODO: compute tuple arithmetic without numpy
        (z,) = cotangents
        x_shape = x.void_tensor.shape
        assert isinstance(x, UndefPrimal)
        if strides is None or np.all(np.equal(strides, 1)):
            lo, hi, interior = (
                starts,
                tuple(np.subtract(x.void_tensor.shape, limits)),
                (0,) * len(starts),
            )
        else:
            real_limits = np.add(
                starts,
                tuple(
                    np.where(
                        np.array(x.shape) == 0,
                        0,
                        np.add(1, np.multiply(np.subtract(t.shape, 1), strides)),
                    )
                ),
            )
            lo, hi, interior = list_zip(starts, np.subtract(x_shape, real_limits), np.subtract(strides, 1))
        padding = []
        for l, h in zip(reversed(lo), reversed(hi)):
            padding += [l, h]
        padding = tuple(padding)
        res = z.pad(padding)
        assert res.shape == x_shape, f"{res.shape=} {x_shape=}"
        return [res]


@operator_set.register("flip")
class Flip(ShapeOperator):
    def args_fixer(self, x, *, dim=None):
        if dim is None:
            dim = tuple(range((x.ndim)))
        elif type(dim) is int:
            dim = (dim,)
        elif type(dim) is list:
            dim = tuple(dim)
        return (x,), dict(dim=dim)

    def typecheck(self, x: VoidTensor, *, dim):
        return [VoidTensor(tuple(x.shape), x.dtype)]

    def vmap(self, dim_size, vals_in, dims_in, *, dim):
        (x,), (x_bdim,) = vals_in, dims_in
        x = slope.core.VMapTrace.move_vmap_dim(x, dim_size, x_bdim, 0)
        y = self(tuple(d + (x_bdim + 1) for d in dim))
        return [y], [x_bdim]

    def jvp(self, primals, tangents, *, dim):
        (x,), (x_dot,) = primals, tangents
        return [x.flip(dim)], [x_dot.flip(dim)]

    def T(self, cotangents, x, *, dim):
        (z,) = cotangents
        return [z.flip(dim)]


@operator_set.register("cat", nary_inputs=True, aliases=["concatenate"])
class Cat(ShapeOperator):
    def args_fixer(self, *xs, dim=None):
        if type(xs) in (tuple, list) and type(xs[0]) in (tuple, list):
            if len(xs) > 1:
                assert len(xs) == 2 and isinstance(xs[1], int) and dim is None
                dim = xs[1]
            xs = xs[0]
        xs = tuple(xs)
        if dim is None:
            dim = 0
        return xs, dict(dim=dim)

    def typecheck(self, *xs: VoidTensor, dim=0) -> List[VoidTensor]:
        if len(set(x.ndim for x in xs)) != 1:
            msg = "Cannot cat tensors with different numbers of dimensions: got {}."
            raise TypeError(msg.format(", ".join(str(o.shape) for o in xs)))
        if not 0 <= dim < xs[0].ndim:
            msg = "cat dimension out of bounds: dimension {} for shapes {}."
            raise TypeError(msg.format(dim, ", ".join([str(o.shape) for o in xs])))
        shapes = [x.shape[:dim] + x.shape[dim + 1 :] for x in xs]
        if not shapes[:-1] == shapes[1:]:
            msg = (
                "Cannot cat tensors with shapes that differ in dimensions "
                "other than the one being catd: concatenating along "
                "dimension {} for shapes {}."
            )
            shapes = [x.shape for x in xs]
            raise TypeError(msg.format(dim, ", ".join(map(str, shapes))))

        concat_size = sum(x.shape[dim] for x in xs)
        ex_shape = xs[0].shape
        return [VoidTensor(ex_shape[:dim] + (concat_size,) + ex_shape[dim + 1 :], xs[0].dtype)]

    def vmap(self, dim_size, vals_in, dims_in, *, dim):
        (*xs,), (*xs_bdim,) = vals_in, dims_in
        xs = tuple(slope.core.VMapTrace.move_vmap_dim(x, dim_size, x_bdim, 0) for x, x_bdim in zip(xs, xs_bdim))
        y = self(xs, dim=dim + 1)
        return [y], [0]

    def jvp(self, primals, tangents, *, dim=0):
        return [self(*primals, dim=dim)], [self(*tangents, dim=dim)]

    def T(self, cotangents, *xs, dim=0):
        (z,) = cotangents
        x_shapes = [o.void_tensor.shape if type(o) is UndefPrimal else o.shape for o in xs]
        if type(z) is None:
            return [None if type(o) is UndefPrimal else None for o in xs]
        else:  # TODO: replace numpy with pure Python
            limit_points = np.cumsum([shape[dim] for shape in x_shapes]).tolist()
            starts = np.zeros((len(xs), z.ndim), dtype=int).tolist()
            limits = np.tile(z.shape, (len(xs), 1)).tolist()

        for i, s in enumerate(starts[1:]):
            s[dim] = limit_points[:-1][i]
        for i, l in enumerate(limits):
            l[dim] = limit_points[i]

        return [
            z.slice(tuple(start), tuple(limit)) if type(o) is UndefPrimal else None
            for o, start, limit in zip(xs, starts, limits)
        ]


# -----------------------
# InitOps
# -----------------------


@operator_set.register("full")
class Full(InitOperator):
    def args_fixer(self, *, shape, fill_value, dtype=Tensor.float32):
        if isinstance(shape, int):
            shape = (shape,)
        elif shape is None:
            shape = ()
        if "float" in dtype.name:
            fill_value = float(fill_value)
        elif "int" in dtype.name:
            fill_value = int(fill_value)
        return (), dict(shape=shape, fill_value=fill_value, dtype=dtype)

    def typecheck(self, *, shape, fill_value, dtype) -> List[VoidTensor]:
        return [VoidTensor(tuple(shape), dtype)]

    # def jvp(self, primals, tangents, *, shape, fill_value, dtype):
    #     out = self(shape=shape, fill_value=fill_value, dtype=dtype)
    #     out_jvp = slope.ones_like(out)
    #     return [out], [out_jvp]


@operator_set.register("random_uniform", aliases=["rand"])
class RandomUniform(InitOperator):
    def args_fixer(self, *, shape=None, dtype=Tensor.float32):
        if isinstance(shape, int):
            shape = (shape,)
        elif shape is None:
            shape = ()
        return (), dict(shape=shape, dtype=dtype)

    def typecheck(self, *, shape, dtype) -> List[VoidTensor]:
        return [VoidTensor(tuple(shape), dtype)]


@operator_set.register("random_normal", aliases=["randn"])
class RandomNormal(InitOperator):
    def args_fixer(self, *, shape=None, dtype=Tensor.float32):
        if isinstance(shape, int):
            shape = (shape,)
        elif shape is None:
            shape = ()
        return (), dict(shape=shape, dtype=dtype)

    def typecheck(self, *, shape, dtype) -> List[VoidTensor]:
        return [VoidTensor(tuple(shape), dtype)]


@operator_set.register("arange", aliases=["iota"])
class Arange(InitOperator):
    def args_fixer(self, *, start, stop=None, stride=None, dtype=Tensor.int64):
        if stop is None:
            stop = start
            start = 0
        if stride is None:
            stride = 1
        return (), dict(start=start, stop=stop, stride=stride, dtype=dtype)

    def typecheck(self, *, start, stop, stride, dtype) -> List[VoidTensor]:
        return [VoidTensor((((stop - start) * stride),), dtype)]


# -------------------
# Other
# -------------------


@operator_set.register("matmul")
class Matmul(BinaryReduceOperator):
    def typecheck(self, x, w):
        assert x.dtype == w.dtype
        if x.ndim == w.ndim == 2:
            # Both arguments are 2-D, multiply like conventional matrices
            assert x.shape[-1] == w.shape[-2]
            shape = x.shape[:-1] + (w.shape[-1],)
        elif x.ndim > 2 and w.ndim > 2:
            # Treat as a stack of matrices and broadcast accordingly
            assert x.shape[-1] == w.shape[-2]
            shape = x.shape[:-2] + (x.shape[-2], y.shape[-1])
        elif x.ndim == 1 and w.ndim > 1:
            # Promote the 1-D argument to a matrix by prepending a 1
            assert x.shape[0] == w.shape[-2]
            shape = (1,) + (w.shape[-2], w.shape[-1])
        elif x.ndim > 1 and w.ndim == 1:
            # Promote the 1-D argument to a matrix by appending a 1
            assert x.shape[-1] == w.shape[0]
            shape = x.shape[:-1] + (w.shape[0],)
        else:
            raise ValueError("Invalid dimensions for matmul")

        return [VoidTensor(shape, x.dtype)]

    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [x @ w], [(x_dot @ w) + (x @ w_dot)]

    def T(self, cotangents, x, w):
        (grad_L_y,) = cotangents
        assert (type(x) is UndefPrimal) ^ (type(w) is UndefPrimal)
        if type(x) is UndefPrimal:
            return [grad_L_y @ w.transpose(-1, -2), None]
        elif type(w) is UndefPrimal:
            return [None, x.transpose(-1, -2) @ grad_L_y]


@operator_set.register("conv")
class Conv(BinaryReduceOperator):
    def args_fixer(self, x, w, *, groups=1, stride=1, dilation=1, padding=0):
        def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
            return (x,) * cnt if isinstance(x, int) else x

        (bs, cin_), (cout, cin), HW = x.shape[:2], w.shape[:2], w.shape[2:]
        assert groups * cin == cin_ and len(x.shape) == len(
            w.shape
        ), f"Input dim shape {x.shape} does not match the shape of the ws {w.shape}. ({groups*cin} vs. {cin_})"
        if isinstance(padding, (tuple, list)):
            assert len(padding) == 2 * len(HW) or len(padding) == len(
                HW
            ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {x.shape}"
        padding = (
            [padding] * 2 * len(HW)
            if isinstance(padding, int)
            else (padding if len(padding) == 2 * len(HW) else [p for p in padding for _ in range(2)])
        )
        padding = tuple(padding)
        if isinstance(stride, int):
            stride = make_pair(stride, len(HW))
        if isinstance(dilation, int):
            dilation = make_pair(dilation, len(HW))
        assert len(HW) == len(stride) and len(HW) == len(
            dilation
        ), f"stride/dilation mismatch kernel:{HW} stride:{stride} dilation:{dilation}"
        return (x, w), dict(groups=groups, stride=stride, dilation=dilation, padding=padding)

    def typecheck(self, x, w, *, groups, stride, dilation, padding):
        assert x.dtype == w.dtype
        x_shape = x.shape
        w_shape = w.shape
        s_dims = []
        padding_start, padding_end = padding[0::2], padding[1::2]
        for i, s in enumerate(x.shape[2:]):
            out_s = ((s + padding_start[i] + padding_end[i] - dilation[i] * (w_shape[i + 2] - 1) - 1) // stride[i]) + 1
            s_dims += [out_s]

        # Calculate output shape
        out_channels = w_shape[0]
        out_shape = (x_shape[0], out_channels, *s_dims)
        if out_shape[-2] != out_shape[-1]:
            breakpoint()

        return [VoidTensor(out_shape, x.dtype)]

    def jvp(self, primals, tangents, *, groups, stride, dilation, padding):
        (x, w), (x_dot, w_dot) = primals, tangents
        y = x.conv(w, groups=groups, stride=stride, dilation=dilation, padding=padding)
        y_dot1 = x_dot.conv(w, groups=groups, stride=stride, dilation=dilation, padding=padding)
        y_dot2 = x.conv(w_dot, groups=groups, stride=stride, dilation=dilation, padding=padding)

        return [y], [y_dot1 + y_dot2]

    # https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
    # x_grad = F.conv_transpose2d(y.grad, w, stride=stride, padding=padding, dilation=dilation, output_padding=stride-padding)
    # assert torch.allclose(x_grad, x.grad)
    # w_grad = F.conv2d(x.transpose(0,1), y.grad.transpose(0,1), stride=dilation, padding=padding, dilation=stride, groups=groups).transpose(0,1)
    # w_grad = w_grad[:,:,:w.size(2),:w.size(3)]
    # assert torch.allclose(w_grad, w.grad)

    def T(self, cotangents, x, w, *, groups, stride, dilation, padding):
        (grad_L_y,) = cotangents
        if type(x) is UndefPrimal:
            grad_L_x = grad_L_y.conv_transpose(
                w,
                groups=groups,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=stride[0] - dilation[0],
            )
            assert grad_L_x.shape == x.shape
            return [grad_L_x, None]
        elif type(w) is UndefPrimal:
            grad_L_w = (
                x.transpose(0, 1)
                .conv(grad_L_y.transpose(0, 1), groups=groups, stride=dilation, dilation=stride, padding=padding)
                .transpose(0, 1)
            )
            if grad_L_w.shape != w.shape:
                starts = (0,) * len(grad_L_w.shape)
                ends = (grad_L_w.shape[0], grad_L_w.shape[1]) + w.shape[2:]
                grad_L_w = grad_L_w.slice(starts, ends)
            assert grad_L_w.shape == w.shape
            return [None, grad_L_w]
