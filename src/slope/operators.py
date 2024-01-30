import slope
import slope.core
from slope.core import (
    Operator,
    UnaryOperator,
    BinaryOperator,
    ReduceOperator,
    ShapeOperator,
    InitOperator,
    GeneralReduceOperator,
    OperatorSet,
    Tensor,
    SymbolicTensor,
    UndefPrimal,
    list_zip,
    dtypes,
)

import math
import numpy as np
from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Optional,
    Sequence,
    Union,
    Iterator,
    NamedTuple,
)
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
    def typecheck(self, x: SymbolicTensor, *, dtype) -> List[SymbolicTensor]:
        return [SymbolicTensor(x.shape, dtype, x.device)]

    def jvp(self, primals, tangents, *, dtype):
        (x,), (x_dot,) = primals, tangents
        return [x.cast(dtype)], [x_dot.cast(dtype)]

    def T(self, cotangents, x, *, dtype):
        (gL_y,) = cotangents
        return [gL_y.cast(x.dtype)]


@operator_set.register("sqrt")
class Sqrt(UnaryOperator):
    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        y = x.sqrt()
        return [y], [x_dot / (y * 2)]

    def T(self, cotangents, x):
        (gL_y,) = cotangents
        return [gL_y / (x.sqrt() * 2)]


@operator_set.register("sin")
class Sin(UnaryOperator):
    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [x.sin()], [(x_dot * ((math.pi / 2) - x).sin())]

    def T(self, cotangents, x):
        (gL_y,) = cotangents
        return [(gL_y * ((math.pi / 2) - x).sin())]


@operator_set.register("exp")
class Exp(UnaryOperator):
    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        y = x.exp()
        return [y], [x_dot * y]

    def T(self, cotangents, x):
        (gL_y,) = cotangents
        return [1 / gL_y]


@operator_set.register("log")
class Log(UnaryOperator):
    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [x.log()], [x_dot / x]

    def T(self, cotangents, x):
        (gL_y,) = cotangents
        return [1 / gL_y]


@operator_set.register("invert")
class Invert(UnaryOperator):
    def typecheck(self, x, **params):
        return [SymbolicTensor(x.shape, dtypes.bool, x.device)]

    def jvp(self, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [~x], [~x_dot]

    def T(self, cotangents, x):
        (gL_y,) = cotangents
        return [~gL_y]


# -----------------------
# Binary
# -----------------------


@operator_set.register("add")
class Add(BinaryOperator):
    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [x + w], [x_dot + w_dot]

    def T(self, cotangents, x, w):
        (gL_y,) = cotangents
        return [gL_y, gL_y]


@operator_set.register("sub")
class Sub(BinaryOperator):
    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [x - w], [x_dot - w_dot]

    def T(self, cotangents, x, w):
        (gL_y,) = cotangents
        return [gL_y, -gL_y]


@operator_set.register("mul")
class Mul(BinaryOperator):
    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [x * w], [(x_dot * w) + (w_dot * x)]

    def T(self, cotangents, x, w):
        (gL_y,) = cotangents
        assert (type(x) is UndefPrimal) ^ (type(w) is UndefPrimal)
        if type(x) is UndefPrimal:
            return [gL_y * w, None]
        elif type(w) is UndefPrimal:
            return [None, x * gL_y]


@operator_set.register("div")
class Div(BinaryOperator):
    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [x / w], [(x_dot / w) + (-w_dot * x * 1 / (w * w))]

    def T(self, cotangents, x, w):
        (gL_y,) = cotangents
        return [gL_y / w, None]


@operator_set.register("pow")
class Pow(BinaryOperator):
    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        y = x**w
        y_dot1 = x_dot * (w * (x ** (w - slope.ones_like(w))))
        y_dot2 = w_dot * (y * (x if x != 0.0 else slope.zeros_like(x)).log())
        return [y], [y_dot1 + y_dot2]

    def T(self, cotangents, x, w):
        (gL_y,) = cotangents
        assert (type(x) is UndefPrimal) ^ (type(w) is UndefPrimal)
        if type(x) is UndefPrimal:
            return [(gL_y * (w * (x ** (w - slope.ones_like(w))))), None]
        elif type(w) is UndefPrimal:
            return [
                None,
                gL_y * ((x**w) * (x.log() if x != 0.0 else slope.zeros_like(x))),
            ]


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
        (gL_y,) = cotangents
        return [gL_y, None]


@operator_set.register("equal")
class Equal(BinaryOperator):
    boolean_output = True


@operator_set.register("less")
class Less(BinaryOperator):
    boolean_output = True


@operator_set.register("greater")
class Greater(BinaryOperator):
    boolean_output = True


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
        (gL_y,) = cotangents
        gL_x = gL_y
        if not keepdim:
            dim = [a if a >= 0 else len(gL_x.shape) + a + 1 for a in dim]
            for a in reversed(sorted(dim)):
                gL_x = gL_x.reshape(gL_x.shape[:a] + (1,) + gL_x.shape[a:])
        gL_x = gL_x.expand(x.symval.shape)


@operator_set.register("sum")
class Sum(ReduceOperator):
    def jvp(self, primals, tangents, *, dim, keepdim):
        (x,), (x_dot,) = primals, tangents
        y = x.sum(dim, keepdim)
        y_dot = x_dot.sum(dim, keepdim)
        return [y], [y_dot]

    def T(self, cotangents, x, *, dim, keepdim):
        (gL_y,) = cotangents
        gL_x = gL_y
        if not keepdim:
            dim = [a if a >= 0 else len(gL_x.shape) + a + 1 for a in dim]
            for a in reversed(sorted(dim)):
                gL_x = gL_x.reshape(gL_x.shape[:a] + (1,) + gL_x.shape[a:])
        gL_x = gL_x.expand(x.symval.shape)

        return [gL_x]


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

    def typecheck(self, x: SymbolicTensor, *, shape: Sequence[int]) -> List[SymbolicTensor]:
        shape = tuple(shape)
        assert len(x.shape) == len(shape)
        assert all(a <= b for a, b in zip(x.shape, shape))
        return [SymbolicTensor(tuple(shape), x.dtype, x.device)]

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
        (gL_y,) = cotangents
        gL_x = gL_y
        if x.symval.shape == gL_x.shape:
            return [gL_x]
        else:
            b_dim = []
            assert len(x.symval.shape) == len(gL_x.shape)
            for i, (xd, od) in enumerate(zip(x.symval.shape, gL_x.shape)):
                if xd != od:
                    b_dim += [i]
            gL_x = gL_x.sum(dim=tuple(b_dim), keepdim=True)
        if gL_x.shape != x.symval.shape:
            raise ValueError(f"not same {gL_x.shape=}, {x.symval.shape=}")
        return [gL_x]


@operator_set.register("reshape", variadic_inputs=True, aliases=["view"])
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

    def typecheck(self, x: SymbolicTensor, *, shape: Sequence[int]) -> List[SymbolicTensor]:
        return [SymbolicTensor(tuple(shape), x.dtype, x.device)]

    def jvp(self, primals, tangents, *, shape):
        (x,), (x_dot,) = primals, tangents
        return [x.reshape(shape)], [x_dot.reshape(shape)]

    def T(self, cotangents, x, *, shape):
        (z,) = cotangents
        return [z.reshape(x.symval.shape)]


@operator_set.register("permute")
class Permute(ShapeOperator):
    def typecheck(self, x: SymbolicTensor, *, perm: Sequence[int]) -> List[SymbolicTensor]:
        shape = [x.shape[i] for i in perm]
        return [SymbolicTensor(shape, x.dtype, x.device)]

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

    def typecheck(self, x: SymbolicTensor, *, padding, mode, value) -> List[SymbolicTensor]:
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
        res = SymbolicTensor(shape, x.dtype, x.device)
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
            res = unpadded.slice(
                tuple([0] * len(lo)),
                unpadded.shape,
                tuple(r + 1 for r in interior),
            )
        else:
            res = None
        return [res]


@operator_set.register("slice")
class Slice(ShapeOperator):
    def args_fixer(self, x, *, starts, limits, strides=None):
        if strides is None:
            strides = (1,) * len(starts)
        return (x,), dict(starts=starts, limits=limits, strides=strides)

    def typecheck(self, x: SymbolicTensor, *, starts, limits, strides=None) -> List[SymbolicTensor]:
        if strides is None or tuple(strides) == (1,) * len(x.shape):
            shape = tuple(
                [
                    limit if type(start) is int and start == 0 else limit - start
                    for start, limit in list_zip(starts, limits)
                ]
            )
            return [SymbolicTensor(shape, x.dtype, x.device)]
        else:
            # TODO: compute strided shape without numpy
            x = np.zeros(x.shape)
            x = x[tuple(slice(s, l, r) for s, l, r in list_zip(starts, limits, strides))]
            return [SymbolicTensor(x.shape, x.dtype, x.device)]

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
        x_shape = x.symval.shape
        assert isinstance(x, UndefPrimal)
        if strides is None or np.all(np.equal(strides, 1)):
            lo, hi, interior = (
                starts,
                tuple(np.subtract(x.symval.shape, limits)),
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
            lo, hi, interior = list_zip(
                starts,
                np.subtract(x_shape, real_limits),
                np.subtract(strides, 1),
            )
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

    def typecheck(self, x: SymbolicTensor, *, dim):
        return [SymbolicTensor(tuple(x.shape), x.dtype, x.device)]

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


@operator_set.register("cat", variadic_inputs=True, aliases=["concatenate"])
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

    def typecheck(self, *xs: Tuple[SymbolicTensor], dim=0) -> List[SymbolicTensor]:
        assert all(x.dtype == xs[0].dtype for x in xs[1:])
        assert all(x.device == xs[0].device for x in xs[1:])
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
        return [
            SymbolicTensor(
                ex_shape[:dim] + (concat_size,) + ex_shape[dim + 1 :],
                xs[0].dtype,
                xs[0].device,
            )
        ]

    def vmap(self, dim_size, vals_in, dims_in, *, dim):
        (*xs,), (*xs_bdim,) = vals_in, dims_in
        xs = tuple(slope.core.VMapTrace.move_vmap_dim(x, dim_size, x_bdim, 0) for x, x_bdim in zip(xs, xs_bdim))
        y = self(xs, dim=dim + 1)
        return [y], [0]

    def jvp(self, primals, tangents, *, dim=0):
        return [self(*primals, dim=dim)], [self(*tangents, dim=dim)]

    def T(self, cotangents, *xs, dim=0):
        (z,) = cotangents
        x_shapes = [o.symval.shape if type(o) is UndefPrimal else o.shape for o in xs]
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
    def args_fixer(self, *, shape, fill_value, dtype=None, device=None):
        if isinstance(shape, int):
            shape = (shape,)
        elif shape is None:
            shape = ()
        if dtype is None:
            dtype = slope.core.backend.DEFAULT_DTYPE
        if device is None:
            device = slope.core.backend.DEFAULT_DEVICE
        if "float" in dtype.name:
            fill_value = float(fill_value)
        elif "int" in dtype.name:
            fill_value = int(fill_value)
        return (), dict(shape=shape, fill_value=fill_value, dtype=dtype, device=device)

    def typecheck(self, *, shape, fill_value, dtype, device) -> List[SymbolicTensor]:
        return [SymbolicTensor(tuple(shape), dtype, device)]


@operator_set.register("random_uniform", variadic_inputs=True, aliases=["rand"])
class RandomUniform(InitOperator):
    def args_fixer(self, *args, **kwargs):
        if "shape" in kwargs.keys():
            shape = kwargs["shape"]
        elif isinstance(args[0], (tuple, list)):
            shape = args[0]
        else:
            shape = args
        shape = tuple(shape)
        dtype = kwargs.get("dtype", None)
        device = kwargs.get("device", None)
        if isinstance(shape, int):
            shape = (shape,)
        elif shape is None:
            shape = ()
        if dtype is None:
            dtype = slope.core.backend.DEFAULT_DTYPE
        if device is None:
            device = slope.core.backend.DEFAULT_DEVICE
        return (), dict(shape=shape, dtype=dtype, device=device)

    def typecheck(self, *, shape, dtype, device) -> List[SymbolicTensor]:
        return [SymbolicTensor(tuple(shape), dtype, device)]


@operator_set.register("random_normal", variadic_inputs=True, aliases=["randn"])
class RandomNormal(InitOperator):
    def args_fixer(self, *args, **kwargs):
        if "shape" in kwargs.keys():
            shape = kwargs["shape"]
        elif isinstance(args[0], (tuple, list)):
            shape = args[0]
        else:
            shape = args
        shape = tuple(shape)
        dtype = kwargs.get("dtype", None)
        device = kwargs.get("device", None)
        if isinstance(shape, int):
            shape = (shape,)
        elif shape is None:
            shape = ()
        if dtype is None:
            dtype = slope.core.backend.DEFAULT_DTYPE
        if device is None:
            device = slope.core.backend.DEFAULT_DEVICE
        return (), dict(shape=shape, dtype=dtype, device=device)

    def typecheck(self, *, shape, dtype, device) -> List[SymbolicTensor]:
        return [SymbolicTensor(tuple(shape), dtype, device)]


@operator_set.register("arange", aliases=["iota"])
class Arange(InitOperator):
    def args_fixer(self, *, start, stop=None, stride=None, dtype=None, device=None):
        if stop is None:
            stop = start
            start = 0
        if stride is None:
            stride = 1
        if dtype is None:
            dtype = dtypes.int64
        if device is None:
            device = slope.core.backend.DEFAULT_DEVICE
        if "f" in dtype.mlir:
            start, stop, stride = float(start), float(stop), float(stride)
        elif "i" in dtype.mlir:
            start, stop, stride = int(start), int(stop), int(stride)
        return (), dict(start=start, stop=stop, stride=stride, dtype=dtype, device=device)

    def typecheck(self, *, start, stop, stride, dtype, device) -> List[SymbolicTensor]:
        assert stride != 0
        if stride > 0:
            assert stop > start
        else:
            assert stop < start
        return [SymbolicTensor((int(math.ceil((abs(stop - start) / abs(stride)))),), dtype, device)]


# -------------------
# Other
# -------------------


@operator_set.register("matmul")
class Matmul(GeneralReduceOperator):
    def typecheck(self, x, w):
        shapes_str = f"{x.shape=}, {w.shape=}"
        assert x.dtype == w.dtype
        if x.ndim == w.ndim == 1:  # dot
            assert x.shape[0] == w.shape[0], f"{shapes_str}"
            shape = ()
        elif x.ndim == w.ndim == 2:  # mat@mat
            assert x.shape[1] == w.shape[0], f"{shapes_str}"
            shape = (x.shape[0], w.shape[1])
        elif x.ndim == 1 and w.ndim == 2:  # vec@mat
            assert x.shape[0] == w.shape[0], f"{shapes_str}"
            shape = (w.shape[1],)
        elif x.ndim == 2 and w.ndim == 1:  # mat@vec
            assert x.shape[1] == w.shape[0], f"{shapes_str}"
            shape = (x.shape[0],)
        elif x.ndim > 2 or w.ndim > 2:  # batched mat@mat
            if x.ndim == 1:
                assert x.shape[0] == w.shape[-2], f"{shapes_str}"
                shape = (*w.shape[:-2], w.shape[-1])
            elif w.ndim == 1:
                assert x.shape[-1] == w.shape[0], f"{shapes_str}"
                shape = x.shape[:-1]
            else:
                assert x.shape[-1] == w.shape[-2], f"{shapes_str}"
                assert len(x.shape) == len(w.shape), f"Different ndim broadcasting not supported, {shapes_str}"
                assert x.shape[:-2] == w.shape[:-2], f"dim -1 broadcasting not supported, {shapes_str}"
                shape = (*x.shape[:-2], x.shape[-2], w.shape[-1])
                # TODO: broadcasting support
                # x_bdims, w_bdims = x.shape[:-2], w.shape[:-2]
                # assert all((a == b) or (a==1) or (b==1) for a, b in zip(x_bdims, w_bdims))
                # bdim_shape = tuple([xd if xd >= wd else wd for (xd, wd) in zip(x_bdims, w_bdims)])
                # shape = (*bdim_shape, x.shape[-2], w.shape[-1])
        else:
            raise ValueError(f"Invalid dimensions for matmul, {shapes_str}")
        return [SymbolicTensor(shape, x.dtype, x.device)]

    def vmap(self, dim_size, vals_in, dims_in, **params):
        (x, w), (x_bdim, w_bdim) = vals_in, dims_in
        x = slope.core.VMapTrace.move_vmap_dim(x, dim_size, x_bdim, 0)
        w = slope.core.VMapTrace.move_vmap_dim(w, dim_size, w_bdim, 0)
        return [self(x, w, **params)], [x_bdim, w_bdim]

    def jvp(self, primals, tangents):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [x @ w], [(x_dot @ w) + (x @ w_dot)]

    def T(self, cotangents, x, w):
        (gL_y,) = cotangents
        assert (type(x) is UndefPrimal) ^ (type(w) is UndefPrimal)
        if type(x) is UndefPrimal:
            return [gL_y @ w.transpose(-1, -2), None]
        elif type(w) is UndefPrimal:
            return [None, x.transpose(-1, -2) @ gL_y]


@operator_set.register("conv")
class Conv(GeneralReduceOperator):
    def args_fixer(self, x, w, *, groups=1, stride=1, dilation=1, padding=0):
        assert x.ndim == w.ndim, "weight must be (N, C, *D), weight (O, I, *D) where D=(H, ...,  W, ...)"
        (bsz, cin_x), (cout, cin_w), D = x.shape[:2], w.shape[:2], w.shape[2:]
        assert groups * cin_x == cin_w, "input and weight input channel dim mismatch"
        if isinstance(padding, (tuple, list)):
            assert len(padding) == 2 * len(D) or len(padding) == len(
                D
            ), f"{2*len(D)=} or {len(D)=}, but {len(padding)=} for {x.shape=}"
        padding = tuple(
            [padding] * 2 * len(D)
            if isinstance(padding, int)
            else (padding if len(padding) == 2 * len(D) else [p for p in padding for _ in range(2)])
        )
        if isinstance(stride, int):
            stride = (stride,) * len(D)
        if isinstance(dilation, int):
            dilation = (dilation,) * len(D)
        assert len(D) == len(stride) and len(D) == len(dilation), f"{len(D)=} {len(stride)=} {len(D)=} {len(dilation)=}"
        return (x, w), dict(groups=groups, stride=stride, dilation=dilation, padding=padding)

    def typecheck(self, x, w, *, groups, stride, dilation, padding):
        assert x.dtype == w.dtype
        s_dims = []
        ps, pe = padding[0::2], padding[1::2]
        for i, s in enumerate(x.shape[2:]):
            out_s = ((s + ps[i] + pe[i] - dilation[i] * (w.shape[i + 2] - 1) - 1) // stride[i]) + 1
            s_dims += [out_s]
        bsz = x.shape[0]
        yc = w.shape[0]  # if x.ndim == w.ndim else 1]
        out_shape = (bsz, yc // groups, *tuple(s_dims))
        return [SymbolicTensor(out_shape, x.dtype, x.device)]

    def vmap(self, dim_size, vals_in, dims_in, **params):
        (x, w), (x_bdim, _) = vals_in, dims_in
        assert x.ndim == w.ndim + 1, "weight cannot be batched"
        N = x.shape[x_bdim]
        cin = w.shape[1]
        Dx = x.shape[-(w.ndim - 2) :]
        x = slope.core.VMapTrace.move_vmap_dim(x, dim_size, x_bdim, 0)
        op_bdims = x.shape[: -(w.ndim - 1)]
        x = x.reshape(math.prod(op_bdims), cin, *Dx)
        y = self(x, w, **params)
        cout = y.shape[1]
        Dy = y.shape[2:]
        y = y.reshape(*op_bdims, cout, *Dy)
        # w = slope.core.VMapTrace.move_vmap_dim(w, dim_size, w_bdim, 0)
        return [y], [x_bdim]

    def jvp(self, primals, tangents, *, groups, stride, dilation, padding):
        (x, w), (x_dot, w_dot) = primals, tangents
        y = x.conv(w, groups=groups, stride=stride, dilation=dilation, padding=padding)
        y_dot1 = x_dot.conv(w, groups=groups, stride=stride, dilation=dilation, padding=padding)
        y_dot2 = x.conv(
            w_dot,
            groups=groups,
            stride=stride,
            dilation=dilation,
            padding=padding,
        )

        return [y], [y_dot1 + y_dot2]

    # https://deeplearning.cs.cmu.edu/F21/document/recitation/Recitation5/CNN_Backprop_Recitation_5_F21.pdf
    # x_grad = F.conv_transpose2d(y.grad, w, stride=stride, padding=padding, dilation=dilation, output_padding=stride-padding)
    # assert torch.allclose(x_grad, x.grad)
    # w_grad = F.conv2d(x.transpose(0,1), y.grad.transpose(0,1), stride=dilation, padding=padding, dilation=stride, groups=groups).transpose(0,1)
    # w_grad = w_grad[:,:,:w.size(2),:w.size(3)]
    # assert torch.allclose(w_grad, w.grad)

    def T(self, cotangents, x, w, *, groups, stride, dilation, padding):
        (gL_y,) = cotangents
        if type(x) is UndefPrimal:
            gL_x = gL_y.conv_transpose(
                w,
                groups=groups,
                stride=stride,
                dilation=dilation,
                padding=padding,
                output_padding=stride[0] - dilation[0],
            )
            assert gL_x.shape == x.shape
            return [gL_x, None]
        elif type(w) is UndefPrimal:
            gL_w = (
                x.transpose(0, 1)
                .conv(
                    gL_y.transpose(0, 1),
                    groups=groups,
                    stride=dilation,
                    dilation=stride,
                    padding=padding,
                )
                .transpose(0, 1)
            )
            if gL_w.shape != w.shape:
                starts = (0,) * len(gL_w.shape)
                ends = (gL_w.shape[0], gL_w.shape[1]) + w.shape[2:]
                gL_w = gL_w.slice(starts, ends)
            assert gL_w.shape == w.shape
            return [None, gL_w]


# @operator_set.register("where")
# class Where(GeneralReduceOperator):
#     def args_fixer(self, x, w, u):
#         return (x, w, u), dict()

#     def typecheck(self, x, w, u):
#         return [w]

#     def vmap(self, dim_size, vals_in, dims_in, **params):
#         (x, w, u), (x_bdim, w_bdim, u_bdim) = vals_in, dims_in
#         x = slope.core.VMapTrace.move_vmap_dim(x, dim_size, x_bdim, 0)
#         w = slope.core.VMapTrace.move_vmap_dim(w, dim_size, w_bdim, 0)
#         u = slope.core.VMapTrace.move_vmap_dim(w, dim_size, w_bdim, 0)
#         return [self(x, w, u)], [x_bdim, w_bdim, u_bdim]

#     def jvp(self, primals, tangents):
#         (x, w, u), (x_dot, w_dot, u_dot) = primals, tangents
#         return [self(x, w, u)], [self(x_dot, w_dot, u_dot)]

#     def T(self, cotangents, x, w, u):
#         assert (type(x) is UndefPrimal) ^ (type(w) is UndefPrimal) ^ (type(u) is UndefPrimal)
#         (gL_y,) = cotangents
#         if type(x) is UndefPrimal:
#             return [None, None, None]
#         elif type(w) is UndefPrimal:
#             return [None, self(x, gL_y, gL_y.zeros_like()), None]
#         elif type(u) is UndefPrimal:
#             return [None, None, self(x, gL_y.zeros_like(), gL_y)]


@operator_set.register("gather_nd")
class GatherND(GeneralReduceOperator):
    def args_fixer(self, x, w, *, batch_dims: int = 0):
        return (x, w), dict(batch_dims=batch_dims)

    def typecheck(self, x, w, *, batch_dims: int):
        r = x.ndim
        q = w.ndim
        b = batch_dims
        assert r > 0 and q > 0
        assert 1 <= w.shape[-1] <= r
        assert w.shape[-1] <= r
        assert b < min(x.ndim, w.ndim)
        # shape = w.shape[: q - 1] + x.shape[w.shape[-1] :]
        bx = x.shape[b:]
        bw = w.shape[b:]
        shape = bx[:b] + bw[: len(bw) - 1] + bx[bw[-1] :]
        return [SymbolicTensor(shape, x.dtype, x.device)]

    def vmap(self, dim_size, vals_in, dims_in, **params):
        (x, w), (x_bdim, w_bdim) = vals_in, dims_in
        x = slope.core.VMapTrace.move_vmap_dim(x, dim_size, x_bdim, 0)
        w = slope.core.VMapTrace.move_vmap_dim(w, dim_size, w_bdim, 0)
        u = slope.core.VMapTrace.move_vmap_dim(u, dim_size, w_bdim, 0)
        return [self(x, w, **params)], [x_bdim, w_bdim]

    def jvp(self, primals, tangents, *, batch_dims):
        (x, w), (x_dot, w_dot) = primals, tangents
        return [self(x, w, batch_dims)], [self(x_dot, w_dot, batch_dims)]

    def T(self, cotangents, x, w):
        assert type(w) is UndefPrimal
        (gL_y,) = cotangents
        return [x.scatter_nd(w, gL_y), None]


@operator_set.register("scatter_nd")
class ScatterND(GeneralReduceOperator):
    def args_fixer(self, x, w, u):
        return (x, w, u), dict()

    def typecheck(self, x, w, u):
        assert w.ndim >= 2
        index_depth = w.shape[-1]
        batch_shape = w.shape[:-1]
        assert index_depth <= x.ndim
        inner_shape = x.shape[index_depth:]
        assert u.shape == batch_shape + inner_shape
        return [x]

    def vmap(self, dim_size, vals_in, dims_in, **params):
        (x, w), (x_bdim, w_bdim) = vals_in, dims_in
        x = slope.core.VMapTrace.move_vmap_dim(x, dim_size, x_bdim, 0)
        w = slope.core.VMapTrace.move_vmap_dim(w, dim_size, w_bdim, 0)
        return [self(x, w, **params)], [x_bdim, w_bdim]

    def jvp(self, primals, tangents):
        (x, w, u), (x_dot, w_dot, u_dot) = primals, tangents
        return [self(x, w, u)], [self(x_dot, w_dot, u_dot)]

    def T(self, cotangents, x, w, u):
        assert (type(x) is UndefPrimal) ^ (type(w) is UndefPrimal) ^ (type(u) is UndefPrimal)
        (gL_y,) = cotangents
        if type(u) is UndefPrimal:
            return [self(x.zeros_like(), w, gL_y), w.zeros_like(), None]
        elif type(w) is UndefPrimal:
            raise [x.zeros_like(), None, u.zeros_like()]
        else:
            raise [x.zeros_like(), w.zeros_like(), None]


# @operator_set.register("rng_bits")
# class RngBits(BinaryOperator):
#     def args_fixer(self, x, *, shape=None, dtype=None, device=None):
#         if isinstance(shape, int):
#             shape = (shape,)
#         elif shape is None:
#             shape = ()
#         if dtype is None:
#             dtype = slope.core.backend.DEFAULT_DTYPE
#         if device is None:
#             device = slope.core.backend.DEFAULT_DEVICE
#         return (x,), dict(shape=shape, dtype=dtype, device=device)

#     def typecheck(self, x, *, shape, dtype, device) -> List[SymbolicTensor]:
#         return [x.symval]
#         # return [SymbolicTensor(tuple(shape), dtype, device)]
