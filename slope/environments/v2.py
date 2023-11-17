import slope
from slope.core import (
    Backend,
    Environment,
    Operator,
    OperatorSet,
    ProcedureSet,
    Tensor,
    TensorBuffer,
    Typecheckor,
    PrimalProxy,
    list_zip,
    list_map,
)

import math
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Sequence, Union, Iterator, NamedTuple
from collections import defaultdict
import onnx
import onnxruntime
import os

sum_py = sum
slice_py = slice

# --------------
# Operator
# --------------

operator_set = OperatorSet()

# -----------------------
# Unary
# -----------------------

stop_gradient = Operator.unary("stop_gradient")
operator_set.register(stop_gradient)


@stop_gradient.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x], [slope.zeros_like(x_dot)]


@stop_gradient.set_method
def T(self, cts, x):
    (z,) = cts
    assert type(x) is PrimalProxy
    return [slope.zeros_like(z)]


cast = Operator.unary("cast")
astype = cast
operator_set.register(cast)
operator_set.alias(cast, "astype")


@cast.set_method
def typecheck(self, x: Typecheckor, *, dtype) -> List[Typecheckor]:
    return [Typecheckor(x.shape, dtype)]


@cast.set_method
def jvp(self, primals, tangents, *, dtype):
    (x,), (x_dot,) = primals, tangents
    return [x.cast(dtype)], [x_dot.cast(dtype)]


sqrt = Operator.unary("sqrt")
operator_set.register(sqrt)


@sqrt.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.sqrt()
    return [ans], [x_dot / (ans * 2)]


@sqrt.set_method
def T(self, cts, x):
    (z,) = cts
    return [z / (x.sqrt() * 2)]


sin = Operator.unary("sin")
operator_set.register(sin)


@sin.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x.sin()], [(x_dot * ((math.pi / 2) - x).sin())]


@sin.set_method
def T(self, cts, x):
    (z,) = cts
    return [(z * ((math.pi / 2) - x).sin())]


exp = Operator.unary("exp")
operator_set.register(exp)


@exp.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.exp()
    return [ans], [x_dot * ans]


@exp.set_method
def T(self, cts, x):
    (z,) = cts
    return [1 / z]


log = Operator.unary("log")
operator_set.register(log)


@log.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x.log()], [x_dot / x]


@log.set_method
def T(self, cts, x):
    (z,) = cts
    return [1 / z]


neg = Operator.unary("neg")
operator_set.register(neg)


@neg.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [-x], [-x_dot]


@neg.set_method
def T(self, cts, x):
    (z,) = cts
    return [-z]


invert = Operator.unary("invert")
operator_set.register(invert)


@invert.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [~x], [~x_dot]


@invert.set_method
def T(self, cts, x):
    (z,) = cts
    return [~z]


@invert.set_method
def typecheck(self, x, **params):
    return [Typecheckor(x.shape, slope.bool)]


# -----------------------
# Binary
# -----------------------


add = Operator.binary("add")
operator_set.register(add)


@add.set_method
def jvp(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x + y], [x_dot + y_dot]


@add.set_method
def T(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar, z_bar]


sub = Operator.binary("sub")
operator_set.register(sub)


@sub.set_method
def jvp(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x - y], [x_dot - y_dot]


@sub.set_method
def T(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar, -z_bar]


mul = Operator.binary("mul")
operator_set.register(mul)


@mul.set_method
def jvp(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x * y], [(x_dot * y) + (y_dot * x)]


@mul.set_method
def T(self, cts, x, y):
    (z_bar,) = cts
    assert (type(x) is PrimalProxy) ^ (type(y) is PrimalProxy)
    if type(x) is PrimalProxy:
        return [z_bar * y, None]
    elif type(y) is PrimalProxy:
        return [None, x * z_bar]


div = Operator.binary("div")
operator_set.register(div)


@div.set_method
def jvp(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x / y], [(x_dot / y) + (-y_dot * x * 1 / (y * y))]


@div.set_method
def T(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar / y, None]


maximum = Operator.binary("maximum")
operator_set.register(maximum)


@maximum.set_method
def jvp(self, primals, tangents):
    def _balanced_eq(x, z, y):
        xz = (x == z).where(slope.ones_like(z), slope.zeros_like(z))
        yz = (y == z).where(slope.full_like(z, 2.0 if "float" in z.dtype.name else 2), slope.ones_like(z))
        eps = slope.ones_like(z)
        return xz / (yz + eps)  # TODO: nan if no eps for onnxruntime

    (x, y), (x_dot, y_dot) = primals, tangents
    run_out = x.maximum(y)
    jvp_out = x_dot * _balanced_eq(x, run_out, y) + y_dot * _balanced_eq(y, run_out, x)
    return [run_out], [jvp_out]


@maximum.set_method
def T(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar, None]


equal = Operator.binary("equal")
operator_set.register(equal)


@equal.set_method
def jvp(self, primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = x.equal(y)
    return [out_primal], [slope.full(out_primal.shape, Tensor.bool)]


@equal.set_method
def T(self, cts, x, y):
    (z_bar,) = cts
    z_bar = z_bar.cast(x.dtype)
    return [z_bar, None]


@equal.set_method
def typecheck(self, x: Typecheckor, y: Typecheckor, **params) -> List[Typecheckor]:
    # difference with default binary typecheck: force dtype bool
    if not type(x) in (Tensor, Typecheckor) or not type(x) in (
        Tensor,
        Typecheckor,
    ):
        raise TypeError
    void_x = Typecheckor.like(x)
    void_y = Typecheckor.like(y)
    if void_x == void_y:
        return [Typecheckor(void_x.shape, Tensor.bool)]
    shape_delta = len(void_x.shape) - len(void_y.shape)
    if shape_delta > 0:
        void_y = Typecheckor((1,) * shape_delta + void_y.shape, Tensor.bool)
    elif shape_delta < 0:
        x = x.reshape((1,) * -shape_delta + void_x.shape)
        void_x = Typecheckor((1,) * -shape_delta + void_x.shape, Tensor.bool)
    if void_x == void_y:
        return [void_x]
    else:
        shape_ret = tuple([max(x, y) for x, y in zip(void_x.shape, void_y.shape)])
        if void_x.shape != shape_ret:
            void_x = Typecheckor(shape_ret, Tensor.bool)
        if void_y.shape != shape_ret:
            void_y = Typecheckor(shape_ret, Tensor.bool)
        if void_x != void_y:
            raise TypeError
        return [void_x]


max = Operator.reduce("max")
operator_set.register(max)


@max.set_method
def jvp(self, primals, tangents, *, axes, keepdims):
    (x,), (x_dot,) = primals, tangents
    out = x.max(axes, keepdims)
    _out = out
    if not keepdims:
        axes = tuple([a if a >= 0 else len(out.shape) + a + 1 for a in axes])
        for a in reversed(sorted(axes)):
            _out = _out.reshape(out.shape[:a] + (1,) + out.shape[a:])
    locs = x.equal(_out.broadcast_to(x.shape))
    locs = locs.cast(x_dot.dtype)
    counts = locs.sum(axes, keepdims)
    jvp_out = (x_dot * locs).sum(axes, keepdims)
    jvp_out = jvp_out / counts.broadcast_to(jvp_out.shape)

    return [out], [jvp_out]


@max.set_method
def T(self, cts, x, *, axes, keepdims):
    (z,) = cts
    out = z
    if not keepdims:
        axes = [a if a >= 0 else len(out.shape) + a + 1 for a in axes]
        for a in reversed(sorted(axes)):
            out = out.reshape(out.shape[:a] + (1,) + out.shape[a:])
    out = out.broadcast_to(x.aval.shape)


sum = Operator.reduce("sum")
operator_set.register(sum)


@sum.set_method
def jvp(self, primals, tangents, *, axes, keepdims):
    (x,), (x_dot,) = primals, tangents
    run_out = x.sum(axes, keepdims)
    jvp_out = x_dot.sum(axes, keepdims)
    return [run_out], [jvp_out]


@sum.set_method
def T(self, cts, x, *, axes, keepdims):
    (z,) = cts
    out = z
    if not keepdims:
        axes = [a if a >= 0 else len(out.shape) + a + 1 for a in axes]
        for a in reversed(sorted(axes)):
            out = out.reshape(out.shape[:a] + (1,) + out.shape[a:])
    out = out.broadcast_to(x.aval.shape)

    return [out]


# -----------------------
# Shape
# -----------------------


broadcast_to = Operator.shape("broadcast_to")
operator_set.register(broadcast_to)


@broadcast_to.set_method
def args_fixer(self, x, *, shape):
    return (x,), dict(shape=shape)


@broadcast_to.set_method
def vmap(self, axis_size, vals_in, dims_in, *, shape):
    (x,), (x_bdim,) = vals_in, dims_in
    shape = list(shape)

    shape = shape[:x_bdim] + [axis_size] + shape[x_bdim:]

    return [self(x, shape)], [x_bdim]


@broadcast_to.set_method
def jvp(self, primals, tangents, *, shape, axes=None):
    (x,), (x_dot,) = primals, tangents
    return (
        [self(x, shape=shape)],
        [self(x_dot, shape=shape)],
    )


@broadcast_to.set_method
def typecheck(self, x: Typecheckor, *, shape: Sequence[int]) -> List[Typecheckor]:
    e_shape = list(x.shape)
    assert len(e_shape) == len(shape)
    assert all(a <= b for a, b in zip(e_shape, shape))
    return [Typecheckor(tuple(shape), x.dtype)]


@broadcast_to.set_method
def T(self, cts, x, *, shape):
    (z,) = cts
    out = z
    if x.aval.shape == out.shape:
        return [out]
    else:
        b_axes = []
        assert len(x.aval.shape) == len(out.shape)
        for i, (xd, od) in enumerate(zip(x.aval.shape, out.shape)):
            if xd != od:
                b_axes += [i]
        out = out.sum(axes=tuple(b_axes), keepdims=True)
    if out.shape != x.aval.shape:
        raise ValueError(f"not same {out.shape=}, {x.aval.shape=}")
    return [out]


reshape = Operator.shape("reshape")
operator_set.register(reshape)


@reshape.set_method
def args_fixer(self, x, *, shape):
    if -1 in shape:
        others = math.prod([d for d in shape if d != -1])
        numel = math.prod(x.shape)
        shape = tuple(d if d != -1 else (numel // others) for d in shape)
    return (x,), dict(shape=shape)


@reshape.set_method
def jvp(self, primals, tangents, *, shape):
    (x,), (x_dot,) = primals, tangents
    return [x.reshape(shape)], [x_dot.reshape(shape)]


@reshape.set_method
def typecheck(self, x: Typecheckor, *, shape: Sequence[int]) -> List[Typecheckor]:
    return [Typecheckor(tuple(shape), x.dtype)]


@reshape.set_method
def T(self, cts, x, *, shape):
    (z,) = cts
    return [z.reshape(x.aval.shape)]


transpose = Operator.shape("transpose")
operator_set.register(transpose)


@transpose.set_method
def vmap(self, axis_size, vals_in, dims_in, *, perm):
    (x,), (x_bdim,) = vals_in, dims_in
    perm_ = list(perm)
    x_bdim_ = int(x_bdim)
    assert x_bdim >= 0
    perm = perm[:x_bdim] + [x_bdim] + perm[x_bdim:]
    perm = tuple(d + int(d >= x_bdim) if i != x_bdim else d for i, d in enumerate(perm))
    assert len(set(perm)) == len(perm)
    return [x.tranpose(perm)], [x_bdim]


@transpose.set_method
def jvp(self, primals, tangents, *, perm):
    (x,), (x_dot,) = primals, tangents
    return [x.transpose(perm)], [x_dot.transpose(perm)]


@transpose.set_method
def typecheck(self, x: Typecheckor, *, perm: Sequence[int]) -> List[Typecheckor]:
    shape = [x.shape[i] for i in perm]
    return [Typecheckor(shape, x.dtype)]


@transpose.set_method
def T(self, cts, x, *, perm):
    (z,) = cts
    return [z.transpose(perm)]


pad_hlo = Operator.shape("pad_hlo")
operator_set.register(pad_hlo)


@pad_hlo.set_method
def args_fixer(self, x, *, lo, hi, interior=None, value=0.0):
    if interior is None:
        interior = tuple([0] * len(lo))
    return (x,), dict(lo=lo, hi=hi, interior=interior, value=value)


@pad_hlo.set_method
def vmap(self, axis_size, vals_in, dims_in, *, pinterior=None, value=0.0):
    raise NotImplementedError
    Operand, padding_value = batched_args
    Operand_bdim, padding_value_bdim = batch_dims
    if Operand_bdim is None:
        Operand_bdim = 0
        Operand = broadcast_in_dim(operand, (padding_value.shape[padding_value_bdim],))

    padding_config = list(padding_config)
    padding_config.insert(operand_bdim, (0, 0, 0))
    if padding_value_bdim is None:
        return pad(operand, padding_value, padding_config), Operand_bdim

    assert padding_value_bdim == 0, padding_value_bdim

    x = pad(operand, _zero(operand), padding_config)
    mask = pad(full_like(operand, True, np.bool_), False, padding_config)
    broadcast_in_dimed_padding = broadcast_in_dim_in_dim(padding_value, x.shape, (operand_bdim,))
    return select(mask, x, broadcast_in_dimed_padding), Operand_bdim


@pad_hlo.set_method
def jvp(self, primals, tangents, *, lo, hi, interior=None, value=0.0):
    (x,), (x_dot,) = primals, tangents
    return [x.pad_hlo(lo, hi, interior, value)], [x_dot.pad_hlo(lo, hi, interior, value)]


@pad_hlo.set_method
def typecheck(self, x: Typecheckor, *, lo, hi, interior=None, value=0.0) -> List[Typecheckor]:
    def _dilate_dim(d, dilation):
        return 0 if d == 0 else 1 + dilation * (d - 1)

    shape = tuple(sum_py([l, h, _dilate_dim(d, r + 1)]) for l, h, r, d in list_zip(lo, hi, interior, x.shape))
    if not all(d >= 0 for d in shape):
        raise ValueError(
            f"Dimension size after padding is not at least 0, "
            f"got result shape {res}, for {lo=} {hi=} {interior=} {value=}"
            f"{shape=}"
        )
    res = Typecheckor(shape, x.dtype)
    return [res]


@pad_hlo.set_method
def T(self, cts, x, *, lo, hi, interior=None, value=0.0):
    (z,) = cts

    def t_op():
        unpadded = z.slice_hlo(
            lo,
            tuple(s - h for s, h in list_zip(z.shape, hi)),
            tuple([1] * len(interior)),
        )
        return unpadded.slice_hlo(tuple([0] * len(lo)), unpadded.shape, tuple(r + 1 for r in interior))

    res = t_op() if isinstance(x, PrimalProxy) else None
    return [res]


slice_hlo = Operator.shape("slice_hlo")
operator_set.register(slice_hlo)


@slice_hlo.set_method
def args_fixer(self, x, *, starts, limits, strides=None):
    if strides is None:
        strides = (1,) * len(starts)
    return (x,), dict(starts=starts, limits=limits, strides=strides)


@slice_hlo.set_method
def vmap(self, axis_size, vals_in, dims_in, *, starts, limits, strides=None):
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

    out = x.slice_hlo(new_start_indices, new_limit_indices, new_strides)
    return out, x_bdim


@slice_hlo.set_method
def jvp(self, primals, tangents, *, starts, limits, strides=None):
    (x,), (x_dot,) = primals, tangents
    return [x.slice_hlo(starts, limits, strides)], [x_dot.slice_hlo(starts, limits, strides)]


@slice_hlo.set_method
def typecheck(self, x: Typecheckor, *, starts, limits, strides=None) -> List[Typecheckor]:
    if strides is None or tuple(strides) == (1,) * len(x.shape):
        shape = tuple(
            [limit if type(start) is int and start == 0 else limit - start for start, limit in list_zip(starts, limits)]
        )
        return [Typecheckor(shape, x.dtype)]
    else:
        # TODO: compute strided shape without numpy
        x = np.zeros_like(x.shape)
        x = x[tuple(slice(s, l, r) for s, l, r in list_zip(starts, limits, strides))]
        return [Typecheckor(x.shape, x.dtype)]


@slice_hlo.set_method
def T(self, cts, x, *, starts, limits, strides=None):
    # TODO: compute tuple arithmetic without numpy
    (z,) = cts
    x_shape = x.aval.shape
    assert isinstance(x, PrimalProxy)
    if strides is None or np.all(np.equal(strides, 1)):
        lo, hi, interior = (
            starts,
            tuple(np.subtract(x.aval.shape, limits)),
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

    res = z.pad_hlo(lo, hi, interior)
    assert res.shape == x_shape, f"{res.shape=} {x_shape=}"
    return [res]


flip = Operator.shape("flip")
operator_set.register(flip)


@flip.set_method
def args_fixer(self, x, *, axes=None):
    if axes is None:
        axes = tuple(range((x.ndim)))
    elif type(axes) is int:
        axes = (axes,)
    elif type(axes) is list:
        axes = tuple(axes)
    return (x,), dict(axes=axes)


@flip.set_method
def vmap(self, axis_size, vals_in, dims_in, *, axes):
    raise NotImplementedError


@flip.set_method
def jvp(self, primals, tangents, *, axes):
    (x,), (x_dot,) = primals, tangents
    return [x.flip(axes)], [x_dot.flip(axes)]


@flip.set_method
def typecheck(self, x: Typecheckor, *, axes):
    return [Typecheckor(tuple(x.shape), x.dtype)]


@flip.set_method
def T(self, cts, x, *, axes):
    (z,) = cts
    return [z.flip(axes)]


concatenate = Operator.shape("concatenate", nary_inputs=True)
operator_set.register(concatenate)
operator_set.alias(concatenate, "cat")


@concatenate.set_method
def args_fixer(self, *xs, axis=0):
    if type(xs) in (tuple, list) and type(xs[0]) in (tuple, list):
        xs = xs[0]
    xs = tuple(xs)
    return xs, dict(axis=axis)


@concatenate.set_method
def vmap(self, axis_size, vals_in, dims_in, *, axis=0):
    raise NotImplementedError


@concatenate.set_method
def jvp(self, primals, tangents, *, axis=0):
    return [concatenate(*primals, axis=axis)], [concatenate(*tangents, axis=axis)]


@concatenate.set_method
def typecheck(self, *xs: Typecheckor, axis=0) -> List[Typecheckor]:
    if len(set(x.ndim for x in xs)) != 1:
        msg = "Cannot concatenate tensors with different numbers of dimensions: got {}."
        raise TypeError(msg.format(", ".join(str(o.shape) for o in xs)))
    if not 0 <= axis < xs[0].ndim:
        msg = "concatenate dimension out of bounds: dimension {} for shapes {}."
        raise TypeError(msg.format(axis, ", ".join([str(o.shape) for o in xs])))
    shapes = [x.shape[:axis] + x.shape[axis + 1 :] for x in xs]
    if not shapes[:-1] == shapes[1:]:
        msg = (
            "Cannot concatenate tensors with shapes that differ in dimensions "
            "other than the one being concatenated: concatenating along "
            "dimension {} for shapes {}."
        )
        shapes = [x.shape for x in xs]
        raise TypeError(msg.format(axis, ", ".join(map(str, shapes))))

    concat_size = sum_py(x.shape[axis] for x in xs)
    ex_shape = xs[0].shape
    return [Typecheckor(ex_shape[:axis] + (concat_size,) + ex_shape[axis + 1 :], xs[0].dtype)]


@concatenate.set_method
def T(self, cts, *xs, axis=0):
    (z,) = cts
    x_shapes = [o.aval.shape if type(o) is PrimalProxy else o.shape for o in xs]
    if type(z) is None:
        return [None if type(o) is PrimalProxy else None for o in xs]
    else:  # TODO: replace numpy with pure Python
        limit_points = np.cumsum([shape[axis] for shape in x_shapes]).tolist()
        starts = np.zeros((len(xs), z.ndim), dtype=int).tolist()
        limits = np.tile(z.shape, (len(xs), 1)).tolist()

    for i, s in enumerate(starts[1:]):
        s[axis] = limit_points[:-1][i]
    for i, l in enumerate(limits):
        l[axis] = limit_points[i]

    return [
        z.slice_hlo(start, limit) if type(o) is PrimalProxy else None for o, start, limit in zip(xs, starts, limits)
    ]


# -----------------------
# LoadOps
# -----------------------

full = Operator.load("full")
operator_set.register(full)


@full.set_method
def args_fixer(self, *, shape, fill_value, dtype=Tensor.float32):
    if isinstance(shape, int):
        shape = (shape,)
    elif shape is None:
        shape = ()
    return (), dict(shape=shape, fill_value=fill_value, dtype=dtype)


@full.set_method
def jvp(self, primals, tangents, *, shape, fill_value, dtype):
    out = self(shape=shape, fill_value=fill_value, dtype=dtype)
    out_jvp = slope.ones_like(out)
    return [out], [out_jvp]


@full.set_method
def T(self, cts, *, shape, fill_value, dtype):
    return [None]


@full.set_method
def typecheck(self, *, shape, fill_value, dtype) -> List[Typecheckor]:
    return [Typecheckor(tuple(shape), dtype)]


random_uniform = Operator.load("random_uniform")
rand = random_uniform
operator_set.register(random_uniform)
operator_set.alias(random_uniform, "rand")


@random_uniform.set_method
def args_fixer(self, *, shape=None, dtype=Tensor.float32):
    if isinstance(shape, int):
        shape = (shape,)
    elif shape is None:
        shape = ()
    return (), dict(shape=shape, dtype=dtype)


@random_uniform.set_method
def jvp(self, primals, tangents, *, shape, dtype):
    out = self(shape=shape, dtype=dtype)
    out_jvp = slope.ones_like(out)
    return [out], [out_jvp]


@random_uniform.set_method
def T(self, cts, *, shape, dtype):
    return [None]


@random_uniform.set_method
def typecheck(self, *, shape, dtype) -> List[Typecheckor]:
    return [Typecheckor(tuple(shape), dtype)]


random_normal = Operator.load("random_normal")
randn = random_normal
operator_set.register(random_normal)
operator_set.alias(random_normal, "randn")


@random_normal.set_method
def args_fixer(self, *, shape=None, dtype=Tensor.float32):
    if isinstance(shape, int):
        shape = (shape,)
    elif shape is None:
        shape = ()
    return (), dict(shape=shape, dtype=dtype)


@random_normal.set_method
def jvp(self, primals, tangents, *, shape, dtype=Tensor.float32):
    out = self(random_normal, shape, dtype)
    out_jvp = slope.ones_like(out)
    return [out], [out_jvp]


@random_normal.set_method
def T(self, cts, *, shape, dtype=Tensor.float32):
    return [None]


@random_normal.set_method
def typecheck(self, *, shape, dtype=Tensor.float32) -> List[Typecheckor]:
    return [Typecheckor(tuple(shape), dtype)]


arange = Operator.load("arange")
operator_set.register(arange)


@arange.set_method
def args_fixer(self, *, start, stop=None, stride=None, dtype=Tensor.int64):
    if stop is None:
        stop = start
        start = 0
    if stride is None:
        stride = 1
    return (), dict(start=start, stop=stop, stride=stride, dtype=dtype)


@arange.set_method
def jvp(self, primals, tangents, *, start, stop, stride, dtype):
    out = self(arange, start, stop, stride, dtype)
    out_jvp = slope.ones_like(out)
    return [out], [out_jvp]


@arange.set_method
def T(self, cts, *, start, stop, stride, dtype):
    return [None]


@arange.set_method
def typecheck(self, *, start, stop, stride, dtype) -> List[Typecheckor]:
    return [Typecheckor((((stop - start) * stride),), dtype)]


# -------------------
# BinaryReduce
# -------------------


# dot = Operator.binary_reduce("dot")


# @dot.set_method
# def typecheck(self, x, y):
#     # matrix-matrix
#     if x.ndim == y.ndim:
#         assert x.shape[-1] == y.shape[-2]
#         if x.ndim >= 3:
#             assert x.shape[0:-2] == y.shape[0:-2]
#         shape = (x.shape[0:-1]) + (y.shape[-2],)
#     # matrix-vector
#     elif x.ndim == y.ndim - 1:
#         assert x.shape[-1] == y.shape[-1]
#         if x.ndim >= 3:
#             assert x.shape[0:-1] == y.shape[0:-1]
#         shape = (x.shape[0:-1]) + (y.shape[-1],)
#     else:
#         raise ValueError
#     # matrix-matrix, vector-matrix, matrix-vector
#     return [Typecheckor(shape, x.dtype)]


# @dot.set_method
# def jvp(self, primals, tangents):
#     (x, y), (x_dot, y_dot) = primals, tangents
#     return [x.dot(y)], [(x_dot.dot(y)) + (x.dot(y_dot))]


# @dot.set_method
# def T(self, cts, x, y):
#     (z_bar,) = cts
#     assert (type(x) is PrimalProxy) ^ (type(y) is PrimalProxy)
#     if type(x) is PrimalProxy:
#         return [z_bar.dot(y), None]
#     elif type(y) is PrimalProxy:
#         return [None, x.dot(z_bar)]


# conv = Operator.binary_reduce("conv")


# def args_fixer(self, x, y, *, groups=1, stride=1, dilation=1, padding=0):
#     return (x, y), dict(groups=groups, stride=stride, dilation=dilation, padding=padding)


# @conv.set_method
# def jvp(self, primals, tangents, *, groups, stride, dilation, padding):
#     (x, y), (x_dot, y_dot) = primals, tangents
#     jvp1 = x_dot.conv(y, groups=groups, stride=stride, dilation=dilation, padding=padding)
#     jvp2 = x.conv(y_dot, groups=groups, stride=stride, dilation=dilation, padding=padding)

#     return [x.conv(y)], [jvp1 + jvp2]


# @conv.set_method
# def T(self, cts, x, y, *, groups, stride, dilation, padding):
#     (z_bar,) = cts
#     if type(x) is PrimalProxy:
#         gx = z_bar.conv_transpose(y, groups=groups, stride=stride, dilation=dilation, padding=padding)
#         return [gx, None]
#     elif type(y) is PrimalProxy:
#         x_T = x.swapaxes(0, 1)
#         z_bar_T = z_bar.swapaxes(0, 1)
#         gy = x_T.conv(z_bar_T, groups=groups, stride=stride, dilation=dilation, padding=padding).swapaxes(0, 1)
#         return [None, gy]


# --------------
# Backend
# --------------

#

compile_py = compile
onnxruntime_backend = Backend(name="onnxruntime", default_dtype=Tensor.float32, default_device=slope.DEFAULT_DEVICE)
onnxruntime_backend.set_dtype_map(
    {
        Tensor.float32: "float",
        Tensor.uint8: "uint8",
        Tensor.int8: "int8",
        Tensor.bool: "bool",
        Tensor.int32: "int32",
        Tensor.int64: "int64",
        Tensor.float16: "float16",
    }
)

# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto3
# used for impl args
onnx_dtype_enum_map = {
    Tensor.float32: 1,
    Tensor.uint8: 2,
    Tensor.int8: 3,
    Tensor.int32: 6,
    Tensor.int64: 7,
    Tensor.bool: 9,
    Tensor.float16: 10,
}


@onnxruntime_backend.set_method
def from_numpy(self, val, dtype=onnxruntime_backend.default_dtype_value, device=onnxruntime_backend.default_device):
    device_type, device_id = device.split(":") if ":" in device else (device, 0)
    np_val = np.array(val, dtype=dtype.numpy)
    val = onnxruntime.OrtValue.ortvalue_from_numpy(np_val, device_type=device_type, device_id=device_id)
    return Tensor(TensorBuffer(val))


@onnxruntime_backend.set_method
def numpy_of(self, tensor):
    return tensor.buf.val.numpy()


@onnxruntime_backend.set_method
def device_of(self, tensor):
    return tensor.buf.val.device_name()


@onnxruntime_backend.set_method
def shape_of(self, tensor):
    return tuple(tensor.buf.val.shape())


@onnxruntime_backend.set_method
def dtype_of(self, tensor):
    return self.dtype_map_inv[tensor.buf.val.data_type().replace("tensor(", "").replace(")", "")]


@onnxruntime_backend.set_method
def export(self, jit_object: slope.core.JitObject, output_path, *args, **kwargs):
    code = jit_object.code
    model = onnx.parser.parse_model(code)
    os.makedirs(output_path, exist_ok=True)
    in_binders = jit_object.codegen_out["in_binders"]
    outs = jit_object.codegen_out["outs"]
    num_consts = jit_object.program.num_consts
    for i in range(num_consts):
        const_array = in_binders[i]["type"].numpy()
        const_name = in_binders[i]["name"]
        const = onnx.numpy_helper.from_array(const_array, name=const_name)
        model.graph.initializer.append(const)
        # TODO: try if need these
        # const_tensor = next(t for t in model.graph.input if t.name == const_name)
        # const_tensor.type.tensor_type.shape.dim[0].dim_param = const_name
        # const_tensor.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

    onnx.save(model.SerializeToString(), os.path.join(output_path, "model.onnx"))
    input_arg_names = [ib["name"] for ib in in_binders[num_consts:]]
    input_arg_names_str = ", ".join(input_arg_names)
    outs_names = [out["name"] for out in outs]

    test_input_code = ""
    for i in range(num_consts, len(in_binders)):
        input_name = in_binders[i]["name"]
        input_shape = in_binders[i]["type"].shape
        dtype = in_binders[i]["type"].dtype
        input_dtype = ("np." + dtype.numpy.__name__) if dtype is not Tensor.bool else "bool"
        test_input_code += f"""    {input_name} = np.ones({input_shape}, dtype={input_dtype})\n"""

    module_path = os.path.join(output_path, "__init__.py")
    module_code = f"""import onnxruntime
import os
import numpy as np

root_path = os.path.dirname(__file__)
model_path = os.path.join(root_path, "model.onnx")
session = onnxruntime.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_arg_names = {input_arg_names}
out_names = {outs_names}

def run(*args, **kwargs):
    if len(args) > 0:
        for a_name, a in zip(input_arg_names, args):
            assert a_name not in kwargs.keys()
            kwargs[a_name] = a
    outputs = session.run(out_names, kwargs)
    return outputs
if __name__ == "__main__":
{test_input_code}
    print("inputs:")
    for inp_name, inp in zip(input_arg_names, ({input_arg_names_str})):
        print(f"{{inp_name}} = ")
        print(inp)
        print(f"dtype: {{inp.dtype}}")
        print(f"shape: {{inp.shape}}")
        print()

    outs = run({input_arg_names_str})

    print("outputs:")
    for out_name, out in zip(out_names, outs):
        print(f"{{out_name}} = ")
        print(out)
        print(f"dtype: {{out.dtype}}")
        print(f"shape: {{out.shape}}")
        print()
"""
    with open(module_path, "w") as f:
        f.write(module_code)
        slope.dblog(module_code, enable=slope.LOG_JIT)


@onnxruntime_backend.set_method
def compile(self, codegen_out):
    code_lines = codegen_out["code_lines"]
    code = "\n".join(code_lines)
    model = onnx.parser.parse_model(code)
    session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    def fn(*args):
        io_binding = session.io_binding()
        for a, in_binder in zip(args, codegen_out["in_binders"]):
            io_binding.bind_input(
                name=in_binder["name"],
                device_type=a.device_name(),
                device_id=0,
                element_type=self.dtype_map_inv[a.data_type().replace("tensor(", "").replace(")", "")].numpy,
                shape=a.shape(),
                buffer_ptr=a.data_ptr(),
            )
        for o in codegen_out["outs"]:
            io_binding.bind_output(o["name"], self.default_device)
        session.run_with_iobinding(io_binding)
        outputs = tuple(io_binding.get_outputs())
        return outputs

    return fn, code


@onnxruntime_backend.set_method
def codegen(self, program, args, *, fn_name: str = "main", fn_defs=dict()) -> List[Any]:
    if fn_name == "main":
        assert not hasattr(self, "fn_count")
        self.fn_count = 0

    def indent(code, amount):
        spaces = " " * (len(code) - len(code.lstrip()))
        spaces += " " * amount
        return "\n".join([spaces + line for line in code.strip().split("\n")])

    # codegen is recursive if jit-of-jit happens
    environment: Dict[slope.Var, Any] = {}
    il1 = 4  # indent length
    body_code_lines = []

    for inb in program.in_binders:
        prefix = "x" if type(inb.aval) is Typecheckor else "c"
        idx = sum_py([1 if v["name"][0] == prefix else 0 for v in environment.values()])
        environment[inb] = dict(name=f"{prefix}{idx}", type=inb.aval)

    for instruction in program.instructions:
        if len(instruction.out_binders) == 0:  # skip codegen for function returns nothing
            continue
        in_vals = list_map(lambda x: environment[x]["name"], instruction.inputs)
        for outb in instruction.out_binders:
            prefix = "y" if outb in program.outs else "z"
            idx = sum_py([1 if v["name"][0] == prefix else 0 for v in environment.values()])
            environment[outb] = dict(name=f"{prefix}{idx}", type=outb.aval)

        out_vals = list_map(lambda z: environment[z]["name"], instruction.out_binders)
        if instruction.op.op_type is slope.core.OperatorType.Meta:
            lhs = ", ".join(out_vals)
            rhs, fn_defs = self.impls[instruction.op](program, args, instruction, in_vals, fn_defs)
            impl_code = f"{lhs} = {rhs}"
        else:
            impl_code = self.impls[instruction.op](*in_vals, **instruction.params)
            if len(out_vals) == 1:
                impl_code = impl_code.replace("ret", out_vals[0])
            else:
                raise NotImplementedError
        for impl_code_line in impl_code.split("\n"):  # handle multi-line code
            body_code_lines += [indent(impl_code_line, il1)]

    # inb_consts = [v for v in environment.values() if "c" in v["name"]]
    # const_type_strs = [f"{self.dtype_map[c['type'].dtype]}[{repr(c['type'].shape)[1:-1]}] {c['name']}" for c in inb_consts]

    in_binders = list_map(lambda x: environment[x], program.in_binders)
    arg_type_strs = [
        f"{self.dtype_map[i['type'].dtype]}[{repr(list(i['type'].shape))[1:-1]}] {i['name']}" for i in in_binders
    ]
    fn_args_str = ", ".join(arg_type_strs)

    outs = list_map(lambda x: environment[x], program.outs)  # TODO: input that is output should has identity op
    out_type_strs = [
        f"{self.dtype_map[o['type'].dtype]}[{repr(list(o['type'].shape))[1:-1]}] {o['name']}" for o in outs
    ]
    out_type_str = ", ".join(out_type_strs)

    head_code_lines = []
    head_code_lines += ['<ir_version: 7, opset_import: ["" : 18, "slope":1]>']
    head_code_lines += [f"{fn_name} ({fn_args_str}) => ({out_type_str})"]
    model_code_lines = head_code_lines + ["{"] + body_code_lines + ["}"]

    functions_head_def = '<domain: "slope",  opset_import: ["" : 18, "slope":1]>'
    functions_code_lines = []
    for op, fn_def_code_lines in fn_defs.items():
        functions_code_lines += [functions_head_def] + fn_def_code_lines
    code_lines = model_code_lines + functions_code_lines
    slope.dblog(f"\n-- {program.name} codegen:\n\n" + "\n".join(code_lines) + "\n\n==\n", enable=slope.LOG_JIT)

    if fn_name == "main":
        del self.fn_count
    assert len(outs) == len(program.outs)
    return dict(code_lines=code_lines, fn_defs=fn_defs, in_binders=in_binders, outs=outs)


### Operator Impls


onnxruntime_backend.set_impl(operator_set.cast)(
    lambda self, x, *, dtype: f"ret = Cast<to={onnx_dtype_enum_map[dtype]}>({x})"
)
onnxruntime_backend.set_impl(operator_set.stop_gradient)(lambda self, x, *, dtype: f"ret = Identity({x})")
onnxruntime_backend.set_impl(operator_set.neg)(lambda self, x: f"ret =  Neg({x})")
onnxruntime_backend.set_impl(operator_set.sqrt)(lambda self, x: f"ret = Sqrt({x})")
onnxruntime_backend.set_impl(operator_set.exp)(lambda self, x: f"ret = Exp({x})")
onnxruntime_backend.set_impl(operator_set.log)(lambda self, x: f"ret = Log({x})")
onnxruntime_backend.set_impl(operator_set.sin)(lambda self, x: f"ret = Sin({x})")
onnxruntime_backend.set_impl(operator_set.add)(lambda self, x1, x2: f"ret = Add({x1}, {x2})")
onnxruntime_backend.set_impl(operator_set.sub)(lambda self, x1, x2: f"ret = Sub({x1}, {x2})")
onnxruntime_backend.set_impl(operator_set.mul)(lambda self, x1, x2: f"ret = Mul({x1}, {x2})")
onnxruntime_backend.set_impl(operator_set.div)(lambda self, x1, x2: f"ret = Div({x1}, {x2})")
onnxruntime_backend.set_impl(operator_set.invert)(lambda self, x: f"ret = Not({x})")
onnxruntime_backend.set_impl(operator_set.equal)(lambda self, x1, x2: f"ret = Equal({x1}, {x2})")
onnxruntime_backend.set_impl(operator_set.maximum)(lambda self, x1, x2: f"ret = Max({x1}, {x2})")


@onnxruntime_backend.set_impl(operator_set.sum)
def sum_impl(self, x, *, axes, keepdims):
    return f"""
ret_axes = Constant <value = int64[{len(axes)}]  {{ {repr(axes)[1:(-1 if len(axes) > 1 else -2)]} }} >()
ret = ReduceSum<keepdims={int(keepdims)}> ({x}, ret_axes)
"""


@onnxruntime_backend.set_impl(operator_set.max)
def max_impl(self, x, *, axes, keepdims):
    return f"""
ret_axes = Constant <value = int64[{len(axes)}]  {{ {repr(axes)[1:(-1 if len(axes) > 1 else -2)]} }} >()
ret = ReduceMax<keepdims={int(keepdims)}> ({x}, ret_axes)
"""


@onnxruntime_backend.set_impl(operator_set.arange)
def arange_impl(self, *, start, stop, stride, dtype):
    return f"""
ret_start = Constant <value_int = {start}> ()
ret_limit = Constant <value_int = {stop}> ()
ret_delta = Constant <value_int = {stride}> ()
{f'''
ret_range = Range(ret_start, ret_limit, ret_delta)
ret = Cast<to={onnx_dtype_enum_map[dtype]}>(ret_range)
''' if dtype is not Tensor.int64 else
f'''
ret = Range(ret_start, ret_limit, ret_delta)
'''
}
"""


# ret_range = Range(ret_start, ret_limit, ret_delta)
# {f'ret = Cast<to={onnx_dtype_enum_map[dtype]}>(ret_range)'}
@onnxruntime_backend.set_impl(operator_set.full)
def full_impl(self, *, shape, fill_value, dtype):
    if dtype is not Tensor.bool:
        if len(shape) > 0:
            return f"""
ret_fill_value = Constant < value = {self.dtype_map[dtype]}[1] {{ {fill_value} }}>()
ret_shape = Constant <value = int64[{len(shape)}] {{ {repr(list(shape))[1:-1]} }} >()
ret = Expand (ret_fill_value, ret_shape)
"""
        else:  # scalar case
            return f"""
ret_fill_value = Constant < value = {self.dtype_map[dtype]}[1] {{ {fill_value} }}>()
ret_squeeze_dim = Constant <value = int64[1] {{0}}> ()
ret = Squeeze (ret_fill_value, ret_squeeze_dim)
"""
    else:
        if len(shape) > 0:
            return f"""
ret_fill_value = Constant < value = int64[1] {{ {fill_value} }}>()
ret_shape = Constant <value = int64[{len(shape)}] {{ {repr(list(shape))[1:-1]} }} >()
ret_expand = Expand (ret_fill_value, ret_shape)
ret = Cast<to={onnx_dtype_enum_map[dtype]}>(ret_expand)
"""
        else:  # scalar case
            return f"""
ret_fill_value = Constant < value = {self.dtype_map[dtype]}[1] {{ {fill_value} }}>()
ret_squeeze_dim = Constant <value = int64[1] {{0}}> ()
ret_squeeze = Squeeze (ret_fill_value, ret_squeeze_dim)
ret = Cast<to={onnx_dtype_enum_map[dtype]}>(ret_squeeze)
"""


@onnxruntime_backend.set_impl(operator_set.random_uniform)
def random_uniform_impl(self, *, shape, dtype):
    if len(shape) > 0:
        return f"""
ret = RandomUniform<dtype={onnx_dtype_enum_map[dtype]},shape={repr(list(shape))}>()
"""
    else:  # scalar case
        return f"""
ret_rand = RandomUniform<dtype={onnx_dtype_enum_map[dtype]}, shape=[1]>()
ret_squeeze_dim = Constant <value = int64[1] {{0}}> ()
ret = Squeeze (ret_rand, ret_squeeze_dim)
"""


@onnxruntime_backend.set_impl(operator_set.random_normal)
def random_normal_impl(self, *, shape, dtype):
    if len(shape) > 0:
        return f"""
ret = RandomNormal<dtype={onnx_dtype_enum_map[dtype]}, shape={repr(list(shape))}>()
"""
    else:  # scalar case
        return f"""
ret_randn = RandomNormal<dtype={onnx_dtype_enum_map[dtype]}, shape=[1]>()
ret_squeeze_dim = Constant <value = int64[1] {{0}}> ()
ret = Squeeze (ret_randn, ret_squeeze_dim)
"""


@onnxruntime_backend.set_impl(operator_set.broadcast_to)
def broadcast_to_impl(self, x, *, shape):
    return f"""
ret_shape = Constant <value = int64[{len(shape)}] {{ {repr(list(shape))[1:-1]} }} >()
ret = Expand ({x}, ret_shape)
"""


@onnxruntime_backend.set_impl(operator_set.reshape)
def reshape_impl(self, x, *, shape):
    if len(shape) > 0:
        return f"""
ret_shape = Constant <value = int64[{len(shape)}] {{ {repr(list(shape))[1:-1]} }} >()
ret = Reshape({x}, ret_shape)
"""
    else:  # scalar case
        f"""
        ret_shape = Constant <value = int64[1] {1} >()
        ret_reshape = Reshape({x}, ret_shape)
        ret_squeeze_dim = Constant <value = int64[1] {{0}}> ()
        ret = Squeeze (ret_reshape, ret_squeeze_dim)"""


@onnxruntime_backend.set_impl(operator_set.pad_hlo)
def pad_hlo_impl(self, x, *, lo, hi, interior, value):  # TODO: interior not used
    pads = lo + hi
    return f"""
ret_pads = Constant <value = int64[{len(pads)}] {pads}>()
ret_constant_value =  Constant <value = {value} >()
ret = Pad({x} ret_pads, ret_constant_value)
"""


@onnxruntime_backend.set_impl(operator_set.slice_hlo)
def slice_hlo_impl(self, x, *, starts, limits, strides):
    return f"""
ret_starts = Constant <value = int64[{len(starts)}] {starts}>()
ret_ends = Constant <value = int64[{len(limits)}] {limits}>()
ret_steps = Constant <value = int64[{len(strides)}] {strides}>()
ret = Slice({x}, ret_starts, ret_ends, steps=ret_steps))])
"""


@onnxruntime_backend.set_impl(operator_set.concatenate)
def concatenate_impl(self, *xs, axis):
    return f"ret = Concat< axis={axis}>({repr(list(xs))[1:-1]})"


@onnxruntime_backend.set_impl(operator_set.transpose)
def transpose_impl(self, x, *, perm):
    return f"ret = Transpose<perm={repr(list(perm))}>({x})"


@onnxruntime_backend.set_impl(operator_set.flip)
def flip_impl(self, x, *, axes):
    return f"""
ret_starts = Constant <value = int64[{len(axes)}] {", ".join(["0"] * len(axes))}>()
ret_ends = Constant <value = int64[{len(axes)}] {", ".join(["-1"] * len(axes))}>()
ret_axes = Constant <value = int64[{len(axes)}] {repr(list(axes))[1:-1]}>()
ret_steps = Constant <value = int64[{len(axes)}] {", ".join(["-1"] * len(axes))}>()
ret = Slice({x}, ret_starts, ret_ends, ret_axes, steps)])
"""


@onnxruntime_backend.set_impl(slope.core.jit_op)
def jit_op_impl(self, program, args, instruction, in_vals, fn_defs):
    jit_program = instruction.params["program"]
    jit_name = f"{program.name}"
    jit_codegen_out = self.codegen(
        jit_program,
        args,
        fn_name=jit_name,
        fn_defs=fn_defs,
    )
    assert jit_name not in fn_defs.keys()
    fn_defs[jit_name] = jit_codegen_out["code_lines"]
    fn_defs = {**fn_defs, **jit_codegen_out["fn_defs"]}
    args_str = ", ".join(in_vals)
    rhs = f"slope.{jit_name}({args_str})"
    return rhs, fn_defs


@onnxruntime_backend.set_impl(slope.core.procedure_op)
def procedure_op_impl(self, program, args, instruction, in_vals, fn_defs):
    proc_program = instruction.params["program"]
    proc_name = f"{proc_program.name}_{self.fn_count}"
    self.fn_count += 1
    proc_codegen_out = self.codegen(
        proc_program,
        args,
        fn_name=proc_name,
        fn_defs=fn_defs,
    )
    fn_defs[proc_name] = proc_codegen_out["code_lines"]
    fn_defs = {**fn_defs, **proc_codegen_out["fn_defs"]}
    args_str = ", ".join(in_vals)
    rhs = f"slope.{proc_name}({args_str})"
    return rhs, fn_defs


# --------------
# Procedure
# --------------

procedure_set = ProcedureSet()


# Utils
def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
    return (x,) * cnt if isinstance(x, int) else x


def flatten_seq(l: Iterator):
    return [item for sublist in l for item in sublist]


@procedure_set.register(static_argnames="shape dtype")
def zeros(shape, dtype=Tensor.float32):
    return slope.full(shape, 0.0 if "float" in dtype.name else 0, dtype)


@procedure_set.register(static_argnames="shape dtype")
def ones(shape, dtype=Tensor.float32):
    return slope.full(shape, 1.0 if "float" in dtype.name else 1, dtype)


@procedure_set.register(static_argnames="fill_value")
def full_like(y, fill_value):
    return slope.full(shape=y.shape, fill_value=fill_value, dtype=y.dtype)


@procedure_set.register()
def zeros_like(y):
    return zeros(shape=y.shape, dtype=y.dtype)


@procedure_set.register()
def ones_like(y):
    return slope.ones(shape=y.shape, dtype=y.dtype)


@procedure_set.register()
def relu(x):
    return x.maximum(slope.zeros_like(x))


@procedure_set.register()
def where(x, trueval, falseval):
    assert x.dtype is Tensor.bool
    cond = x.cast(trueval.dtype)
    return cond * trueval + (ones_like(cond) - cond) * falseval


@procedure_set.register(static_argnames="axes keepdims")
def mean(x, axes=None, keepdims=False):
    out = x.sum(axes=axes, keepdim=keepdims)
    return out * (math.prod(out.shape) / math.prod(x.shape))


@procedure_set.register()
def rsqrt(x):
    return (1 / x).sqrt()


@procedure_set.register()
def cos(x):
    return ((math.pi / 2) - x).sin()


@procedure_set.register()
def tan(x):
    return x.sin() / x.cos()


@procedure_set.register()
def not_equal(x, y):
    return ~(x.equal(y))


@procedure_set.register()
def greater_equal(x, y):
    return x.maximum(y).equal(y)


@procedure_set.register()
def less_equal(x, y):
    return x.minimum(y).equal(y)


@procedure_set.register()
def greater(x, y):
    return 1.0 - (x <= y)


@procedure_set.register()
def less(x, y):
    return 1.0 - (x >= y)


@procedure_set.register()
def minimum(x, y):
    return -x.maximum(-x, -y)


@procedure_set.register(static_argnames="axes keepdims")
def min(x, axes=None, keepdims=False):
    return -((-x).max(x, axes, keepdims))


@procedure_set.register(static_argnames="axes keepdims")
def argmax(x, axes=None, keepdims=False):
    if axes is None:
        idx = (x == x.max(axes)) * slope.arange(
            math.prod(x.shape) - 1,
            -1,
            -1,
            dtype=slope.int32,
        ).reshape(x.shape)
        return math.prod(x.shape) - idx.max() - 1
    axis = axes + len(x.shape) if axes < 0 else axes
    m = x == x.max(axis=axis, keepdim=True)
    idx = m * slope.arange(x.shape[axis] - 1, -1, -1, dtype=slope.int32).reshape(
        x.shape[axis], *[1] * (x.ndim - axis - 1)
    )
    return x.shape[axis] - idx.max(axes=axes, keepdim=keepdims) - 1


@procedure_set.register(static_argnames="axes keepdims")
def argmin(x, axes=None, keepdims=False):
    return (-x).argmax(axes=axes, keepdims=keepdims)


def pow(self, x: Union[Tensor, float], reverse=False) -> Tensor:
    if x.__class__ is not Tensor and not reverse:
        # simple pow identities
        if x < 0:
            return (1 / self).pow(-x)
        if x == 3.0:
            return self * self * self
        if x == 2.0:
            return self * self
        if x == 1.0:
            return self
        if x == 0.5:
            return self.sqrt()
    if not isinstance(x, Tensor) and reverse and x > 0:
        return self.mul(math.log(x)).exp()
    ar = self.abs().log().mul(x).exp() if not reverse or isinstance(x, Tensor) else self.mul(math.log(abs(x))).exp()
    # correct sign of negative numbers raised to a power (cos has a period of 2pi so we use it here to get the oddness of the power)
    sign = (
        (x * math.pi).cos()
        if isinstance(x, Tensor)
        else math.cos(x * math.pi)
        if not reverse
        else (self * math.pi).cos()
    )
    # we only need to correct the sign if the base is negative
    base_sign = ((self.sign() if not reverse else x.sign() if isinstance(x, Tensor) else math.copysign(1, x)) - 1) / -2
    # we need 0 to be positive so we need to correct base_sign when the base is 0
    base_sign = base_sign - (
        1.5
        * (1 - (self.sign().abs() if not reverse else x.sign().abs() if isinstance(x, Tensor) else abs(int(bool(x)))))
    )
    # inject nan if the base is negative and the power is not an integer
    to_nan = (
        ((x - x.trunc()) * 1e10).abs().clip(0, 1)
        if isinstance(x, Tensor)
        else int(bool(x - int(x)))
        if not reverse
        else ((self - self.trunc()) * 1e10).abs().clip(0, 1)
    ) * base_sign
    inject_nan = (
        ((((-to_nan) * 2) + 1)).log().add(1) if isinstance(to_nan, Tensor) else 1 if not to_nan else float("nan")
    )
    return ar.mul(sign * base_sign + (1 - base_sign)).mul(inject_nan)


@procedure_set.register()
def log2(x):
    return x.log() / math.log(2)


@procedure_set.register()
def sigmoid(self):
    return 1 / (1 + (-self).exp())


@procedure_set.register()
@staticmethod
def _tri(r: int, c: int, k: int = 0, **kwargs) -> Tensor:
    return Tensor.arange(r, **kwargs).unsqueeze(1).expand(r, c) <= Tensor.arange(-k, c - k, **kwargs).unsqueeze(
        0
    ).expand(r, c)


@procedure_set.register()
def triu(self, k: int = 0) -> Tensor:
    return Tensor._tri(self.shape[-2], self.shape[-1], k=k, dtype=self.dtype, device=self.device).where(
        self, Tensor.zeros_like(self)
    )


@procedure_set.register()
def tril(self, k: int = 0) -> Tensor:
    return Tensor._tri(self.shape[-2], self.shape[-1], k=k + 1, dtype=self.dtype, device=self.device).where(
        Tensor.zeros_like(self), self
    )


@procedure_set.register()
def trunc(self: Tensor) -> Tensor:
    return self.cast(slope.int32).cast(self.dtype)


@procedure_set.register()
def ceil(self: Tensor) -> Tensor:
    return (self > (b := self.trunc())).where(b + 1, b)


@procedure_set.register()
def floor(self: Tensor) -> Tensor:
    return (self < (b := self.trunc())).where(b - 1, b)


@procedure_set.register()
def square(self):
    return self * self


@procedure_set.register()
def clip(self, min_, max_):
    return self.maximum(min_).minimum(max_)


@procedure_set.register()
def abs(self):
    return self.relu() + (-self).relu()


@procedure_set.register()
def sign(self):
    return self / (self.abs() + 1e-10)


@procedure_set.register()
def reciprocal(self):
    return 1.0 / self


# # ***** activation functions (unary) *****
@procedure_set.register()
def elu(self, alpha=1.0):
    return self.relu() - alpha * (1 - self.exp()).relu()


@procedure_set.register()
def celu(self, alpha=1.0):
    return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)


@procedure_set.register()
def swish(self):
    return self * self.sigmoid()


@procedure_set.register()
def silu(self):
    return self.swish()  # The SiLU function is also known as the swish function.


@procedure_set.register()
def relu6(self):
    return self.relu() - (self - 6).relu()


@procedure_set.register()
def hardswish(self):
    return self * (self + 3).relu6() * (1 / 6)


@procedure_set.register()
def tanh(self):
    return 2.0 * ((2.0 * self).sigmoid()) - 1.0


@procedure_set.register()
def hardtanh(self, min_val=-1, max_val=1):
    return self.clip(min_val, max_val)


@procedure_set.register()
def gelu(self):
    return 0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())


@procedure_set.register()
def quick_gelu(self):
    return self * (self * 1.702).sigmoid()


@procedure_set.register()
def leakyrelu(self, neg_slope=0.01):
    return self.relu() - (-neg_slope * self).relu()


@procedure_set.register()
def mish(self):
    return self * self.softplus().tanh()


@procedure_set.register()
def softplus(self, beta=1):
    return (1 / beta) * (1 + (self * beta).exp()).log()


@procedure_set.register()
def softsign(self):
    return self / (1 + self.abs())


@procedure_set.register()
def matmul(x, w):
    x = x.reshape((*x.shape[0:-1], 1, x.shape[-1]))
    w = w.reshape((*w.shape[0:-2], 1, w.shape[-2], w.shape[-1])).T()
    return (x * w).sum(-1).reshape((*x.shape[0:-2], -1))


procedure_set.alias(matmul, "dot")


@procedure_set.register()
def T(x):
    perm = list(range(x.ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return x.transpose(tuple(perm))


@procedure_set.register(static_argnames="axes")
def softmax(x, axes=-1):
    m = x - x.max(axes, keepdims=True)
    e = m.exp()
    ss = e.sum(axes, keepdims=True)
    return e / ss


@procedure_set.register(static_argnames="axes")
def log_softmax(x, axes=-1):
    m = x - x.max(axes, keepdims=True)
    e = m.exp()
    ss = e.sum(axes, keepdims=True)
    return m - ss.log()


# - Negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
# - A slice i:j returns the elements with indices in [i, j)
#    - If omitted, i and j will default to 0 and N, respectively, where N is the length of the sequence
#    - Negative values for i and j are taken relative to the end of the sequence
#    - Both i and j will be clamped to the range (-N, N], where N in the length of the sequence
# - Indexing with None on a given axis will add a new dimension of size one before that axis
# - Empty slices are not allowed (tensors with 0s in shape have to be supported first, for all backends).
# - For a slice [i:j:k] finding the correct indices is delegated to slice.indices(len).
# - Strides > 1 and < 0 are now allowed!:
#    - This works by applying Shrink -> [[Flip -> ] Pad -> Reshape -> Shrink] -> Reshape (ops in brackets are optional)
#    - Idea of stride < 0 support:
#        - Do the slice first, flip the axes were slice.step is negative, do slice.step -> -slice.step. Go to steps below.
#    - Idea of stride `s` > 1 support (Pad -> Reshape -> Shrink):
#        - Instead of doing [::s] on axis [dim_sz], do [:, 0] on axes [dim_sz_padded // s, s].
#        - So pad dim_sz with as many zeros as needed (dim_sz -> dim_sz_padded) so that reshape to [dim_sz_padded // s, s]
#          is possible.
#        - Apply Shrink to do the slice [:, 0] on axes of shapes [dim_sz_padded // s, s].
# - Fancy indexing and combined indexing is supported
#    - Combined indexing works by letting regular slicing finish first -> computing the resulting dims w.r.t to Tensors passed in -> fancy indexing
#    - Any Tensors passed in __getitem__ will perform (CMPEQ with arange -> MUL with self -> SUM_REDUCE) iteratively
#        - The first iteration will expand the dim of self while consecutive iterations will reduce the dim
#    - There's a special case where a permute is needed at the end:
#        - if first Tensor passed in (expand dims) is not at dim 0
#        - and following Tensors does not follow consecutively to the end of fancy indexing's dims
# val: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
@procedure_set.register(not_op=True)  # not_op because easier to support variadic dynamic and static args
def getitem(self, val):
    def normalize_int(e, i, dim_sz):
        if -dim_sz <= e < dim_sz:
            return e if e != -1 else dim_sz - 1
        raise IndexError(f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}")

    orig_slices = list(val) if isinstance(val, tuple) else [val]
    count = defaultdict(list)
    for i, v in enumerate(orig_slices):
        count[type(v) if not isinstance(v, slope.core.Tensor) else "tensor"] += [i]

    if (num_slices := len(count[int]) + len(count[slice_py]) + len(count["tensor"])) > len(self.shape):
        raise IndexError(f"too many indices for tensor of dimension {len(self.shape)}")
    if len(ellipsis_found := count[type(Ellipsis)]) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
    orig_slices[ellipsis_idx : ellipsis_idx + 1] = [slice_py(None)] * (len(self.shape) - num_slices)

    valid_slices = [v for v in orig_slices if v is not None]
    valid_slices = [
        v
        if isinstance(v, slice_py)
        else slice_py(y_ := normalize_int(v, i, dim_sz), y_ + 1)
        if isinstance(v, int)
        else slice_py(None)
        for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))
    ]

    start, stop, strides = (
        zip(*y) if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)]) else ((), (), ())
    )
    new_slice = tuple((s, e) if st > 0 else (e + 1, s + 1) for s, e, st in zip(start, stop, strides))
    sliced_tensor = self.padslice(new_slice).flip(axes=tuple([i for i, s in enumerate(strides) if s < 0]))
    new_shape = sliced_tensor.shape
    if any(abs(s) != 1 for s in strides):
        strides = tuple(abs(s) for s in strides)
        # Pad: add pad at the end: [dim_sz] -> [dim_sz_padded]
        padded_tensor = sliced_tensor.pad(
            tuple((0, s - (dim_sz % s) if dim_sz % s != 0 else 0) for s, dim_sz in zip(strides, sliced_tensor.shape))
        )
        # Reshape: [dim_sz_padded] -> [dim_sz_padded // s, s]
        reshaped_tensor = padded_tensor.reshape(flatten([sh // s, s] for sh, s in zip(padded_tensor.shape, strides)))
        new_shape = reshaped_tensor.shape[::2]
        # Shrink: do [:, 0]
        sliced_tensor = reshaped_tensor.padslice(tuple(flatten(((0, sh), (0, 1)) for sh in new_shape)))

    final_shape, it_shape, dim, tensors, dim_collapsed = [], iter(new_shape), [], [], 0
    for i, s in enumerate(orig_slices):
        if s is None:
            final_shape.append(1)
        else:  # s is int or slice or Tensor
            dim_shape = next(it_shape)
            if isinstance(s, int):
                dim_collapsed += 1
            else:
                final_shape.append(dim_shape)
                if isinstance(s, slope.core.Tensor):
                    tensors.append(s)
                    dim.append(i - dim_collapsed)
    sliced_tensor.reshape(tuple(final_shape))

    if tensors:  # Fancy/tensor indexing
        # normalize idx
        idx = [t.sign().neg().relu() * ret.shape[d] + t for d, t in zip(dim, tensors)]
        max_dim = max(i.ndim for i in idx)
        # compute sum_dim, arange, and idx
        sum_dim = [d if n == 0 else d + max_dim - n for n, d in enumerate(dim)]
        slice_arange = [
            slope.arange(ret.shape[d], dtype=slope.int32, requires_grad=False, device=self.device).reshape(
                *[1] * sd, ret.shape[d], *[1] * (ret.ndim + max_dim - n - sd - 1)
            )
            for n, (sd, d) in enumerate(zip(sum_dim, dim))
        ]
        first_idx = [
            idx[0].reshape(
                *[1] * dim[0],
                *[1] * (1 + max_dim - idx[0].ndim),
                *idx[0].shape,
                *[1] * (ret.ndim - dim[0] - 1),
            )
        ]
        rest_idx = [
            i.reshape(
                *[1] * dim[0],
                *[1] * (max_dim - i.ndim),
                *i.shape,
                *[1] * (ret.ndim - dim[0] - n),
            )
            for n, i in enumerate(idx[1:], 1)
        ]
        idx = first_idx + rest_idx
        ret.reshape(*ret.shape[: sum_dim[0] + 1], *[1] * max_dim, *ret.shape[sum_dim[0] + 1 :])
        # iteratively fancy index
        for a, i, sd in zip(slice_arange, idx, sum_dim):
            (a == i).mul(ret).sum(sd)
        # special permute case
        if dim[0] != 0 and len(dim) != 1 and dim != list(range(dim[0], dim[-1] + 1)):
            ret_dims = list(range(ret.ndim))
            ret.transpose(ret_dims[dim[0] : dim[0] + max_dim] + ret_dims[: dim[0]] + ret_dims[dim[0] + max_dim :])
    return ret


@procedure_set.register(static_argnames="pad_width mode constant_values")
def pad(x, pad_width, mode="constant", constant_values=0.0):
    assert mode == "constant", "Other modes not supported"
    if type(pad_width) is int:
        pad_width = (pad_width, pad_width, 0) * x.ndim
    elif all(type(pw) is int for pw in pad_width):
        assert len(pad_width) == x.ndim
        pad_width = tuple((pw, pw, 0) for pw in pad_width)
    elif len(pad_width) == 2 and all(type(item) is int for item in pad_width):
        pad_width = (*pad_width, 0) * x.ndim
    elif len(pad_width) == 3 and all(type(item) is int for item in pad_width):
        pad_width = (pad_width,) * x.ndim
    else:
        assert all(2 <= len(pw) <= 3 for pw in pad_width)
        pad_width = tuple((*pw, 0) if len(pw) == 2 else pw for pw in pad_width)
    lo, hi, interior = tuple(zip(*pad_width))
    return x.pad_hlo(lo, hi, interior, value=constant_values)


@procedure_set.register(static_argnames="arg")
def slice(x, arg):
    # assert all(2 <= len(a) <= 3 for a in arg)
    arg = tuple((*a, 1) if len(a) == 2 else a for a in arg)
    starts, limits, strides = tuple(zip(*arg))
    return x.slice_hlo(starts, limits, strides)


# @procedure_set.register(static_argnames=("arg", "value"))
@procedure_set.register(static_argnames=("arg", "value"))
def padslice(x, arg: Sequence[Optional[Tuple[int, int]]], value: float = 0):
    arg_ = tuple([a if a is not None else (0, s) for s, a in zip(x.shape, arg)])
    padding = tuple([(max(0, -p[0]), max(0, p[1] - x.shape[i])) for i, p in enumerate(arg_)])
    x = x.pad(padding, constant_values=value)
    slc = tuple([(p[0] + padding[i][0], p[1] + padding[i][0]) for i, p in enumerate(arg_)])
    x = x.slice(slc)
    return x


@procedure_set.register(static_argnames="dim")
def gather(x, idx, dim: int):
    assert idx.ndim == x.ndim, "x.ndim must equal idx.ndim"
    assert all(s >= i for s, i in zip(x.shape, idx.shape)), "all dim of idx.shape must be smaller than x.shape"
    if dim < 0:
        dim += x.ndim
    idx = idx.swapaxes(ax1=dim, ax2=0).expand_dims(-1)
    permarg = list(range(x.ndim))
    permarg = (
        permarg[1:dim] + [permarg[0]] + permarg[dim + 1 :] + [permarg[dim]] if dim != 0 else permarg[1:] + [permarg[0]]
    )
    return (
        (
            (
                idx
                == slope.arange(
                    x.shape[dim],
                    dtype=slope.int32,
                    requires_grad=False,
                    device=x.device,
                )
            )
            * x.transpose(*permarg)
            .padslice(tuple([*[(0, sh) for sh in idx.shape[1:-1]], (0, x.shape[dim])]))
            .expand_dims(0)
        )
        .sum(-1)
        .swapaxes(ax1=0, ax2=dim)
    )


@procedure_set.register(static_argnames="dim")
@staticmethod
def stack(tensors, dim=0):
    first = tensors[0].expand_dims(dim)
    expand_dimsd_tensors = [tensor.expand_dims(dim) for tensor in tensors[1:]]
    # checks for shapes and number of dimensions delegated to cat
    return first.concatenate(*expand_dimsd_tensors, dim=dim)


@procedure_set.register(static_argnames="repeats")
def repeat(x, repeats):
    base_shape = (1,) * (len(repeats) - x.ndim) + x.shape
    new_shape = [x for b in base_shape for x in [1, b]]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r * s for r, s in zip(repeats, base_shape)]
    return x.reshape(new_shape).broadcast(expand_shape).reshape(final_shape)


@procedure_set.register(static_argnames="dim")
def split(x, num: int, dim: int):
    dim, step = dim + x.ndim if dim < 0 else dim, math.ceil(x.shape[dim] / num)
    slice_params = [[slice(None)] * dim + [slice(k, k + step)] for k in range(0, x.shape[dim], step)]
    return [x[tuple(sl)] for sl in slice_params]


@procedure_set.register(static_argnames="dim")
def squeeze(x, dim=None):
    if dim is None:
        return x if 1 not in x.shape else x.reshape(*[size for size in x.shape if size != 1])
    if dim <= 0 and x.ndim == 0:
        return x  # This is to match PyTorch behavior
    if not -x.ndim <= dim < x.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-x.ndim if x.ndim > 0 else x.ndim-1}, {x.ndim-1 if x.ndim > 0 else x.ndim}], but got {dim})"
        )
    if dim < 0:
        dim += x.ndim
    return x if x.shape[dim] != 1 else x.reshape(*[size for idx, size in enumerate(x.shape) if idx != dim])


@procedure_set.register(static_argnames="dim")
def expand_dims(x, dim):
    if dim < 0:
        dim = len(x.shape) + dim + 1
    return x.reshape(x.shape[:dim] + (1,) + x.shape[dim:])


@procedure_set.register(static_argnames="ax1 ax2")
def swapaxes(x, ax1=1, ax2=0):
    order = list(range(len(x.shape)))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return x.transpose(order)


@procedure_set.register(static_argnames="start_dim")
def flatten(x, start_dim=0):
    return x.reshape(shape=x.shape[:start_dim] + (-1,))


@procedure_set.register(static_argnames="k_ stride dilation")
def _pool(
    x,
    k_: Tuple[int, ...],
    stride: Union[Tuple[int, ...], int] = 1,
    dilation: Union[Tuple[int, ...], int] = 1,
):
    assert len(x.shape) >= len(k_), f"can't pool {x.shape} with {k_}"
    s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
    assert len(k_) == len(s_) and len(k_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    slc_prefix, prefix, i_ = (
        [(0, x) for x in x.shape[0 : -len(k_)]],
        x.shape[0 : -len(k_)],
        x.shape[-len(k_) :],
    )
    if any(k > s for k, s in zip(k_, s_)) or any(d != 1 for d in d_):
        o_ = [(i - d * (k - 1) - 1) // s + 1 for i, d, k, s in zip(i_, d_, k_, s_)]
        e_ = [math.ceil(k * (i + d) / i) for k, i, d in zip(k_, i_, d_)]  # expands such that we don't need padding
        xup = x
        xup = xup.reshape((*prefix, *flatten_seq((1, i) for i in i_)))
        xup = xup.broadcast_to((*prefix, *flatten_seq((e, i) for e, i in zip(e_, i_))))
        xup = xup.reshape((*prefix, *[e * i for e, i in zip(e_, i_)]))
        # slide by dilation
        xup = xup.padslice(slc_prefix + [(0, k * (i + d)) for k, i, d in zip(k_, i_, d_)])
        xup = xup.reshape((*prefix, *flatten_seq((k, i + d) for k, i, d in zip(k_, i_, d_))))
        xup = xup.padslice(slc_prefix + flatten_seq(((0, k), (0, o * s)) for k, o, s in zip(k_, o_, s_)))
        # handle stride, and permute to move reduce to the end
        xup = xup.reshape((*prefix, *flatten_seq((k, o, s) for k, o, s in zip(k_, o_, s_))))
        xup = xup.padslice(slc_prefix + flatten_seq(((0, k), (0, o), (0, 1)) for k, o in zip(k_, o_)))
        xup = xup.reshape((*prefix, *flatten_seq((k, o) for k, o in zip(k_, o_))))
        return xup.transpose(
            (
                *range(len(prefix)),
                *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
                *[len(prefix) + i * 2 for i in range(len(k_))],
            )
        )
    # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
    o_ = [(i + (s - k)) // s for i, s, k in zip(i_, s_, k_)]
    xup = x.padslice(slc_prefix + [(0, o * s) for o, s in zip(o_, s_)])
    xup = xup.reshape((*prefix, *flatten_seq(((o, s) for o, s in zip(o_, s_)))))
    xup = xup.padslice((slc_prefix + flatten_seq(((0, o), (0, k)) for o, k in zip(o_, k_))))
    return xup.transpose(
        (
            *range(len(prefix)),
            *[len(prefix) + i * 2 for i in range(len(k_))],
            *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
        )
    )


# NOTE: these work for more than 2D
@procedure_set.register(static_argnames="kernel_size stride")
def avg_pool2d(x, kernel_size=(2, 2), stride=None):
    return x._pool(make_pair(kernel_size), stride if stride is not None else kernel_size).mean(
        axis=tuple(range(0 - len(make_pair(kernel_size)), 0))
    )


@procedure_set.register(static_argnames="kernel_size stride dilation")
def max_pool2d(x, kernel_size=(2, 2), stride=None, dilation=1):
    return x._pool(make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).max(
        axis=tuple(range(0 - len(make_pair(kernel_size)), 0))
    )

@procedure_set.register(static_argnames="axis")
def cumsum(x, axis: int = 0):
    return x.swapaxes(axis, -1).pad((x.shape[axis] - 1, 0))._pool((x.shape[axis],)).sum(-1).swapaxes(axis, -1)


v2_environment = Environment(operator_set, procedure_set, onnxruntime_backend)
