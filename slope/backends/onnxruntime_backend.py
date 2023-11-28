import slope
from slope.core import (
    Compiler,
    Backend,
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
def T(self, cotangents, x):
    (z,) = cotangents
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
def T(self, cotangents, x):
    (z,) = cotangents
    return [z / (x.sqrt() * 2)]


sin = Operator.unary("sin")
operator_set.register(sin)


@sin.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x.sin()], [(x_dot * ((math.pi / 2) - x).sin())]


@sin.set_method
def T(self, cotangents, x):
    (z,) = cotangents
    return [(z * ((math.pi / 2) - x).sin())]


exp = Operator.unary("exp")
operator_set.register(exp)


@exp.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.exp()
    return [ans], [x_dot * ans]


@exp.set_method
def T(self, cotangents, x):
    (z,) = cotangents
    return [1 / z]


log = Operator.unary("log")
operator_set.register(log)


@log.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x.log()], [x_dot / x]


@log.set_method
def T(self, cotangents, x):
    (z,) = cotangents
    return [1 / z]


neg = Operator.unary("neg")
operator_set.register(neg)


@neg.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [-x], [-x_dot]


@neg.set_method
def T(self, cotangents, x):
    (z,) = cotangents
    return [-z]


invert = Operator.unary("invert")
operator_set.register(invert)


@invert.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [~x], [~x_dot]


@invert.set_method
def T(self, cotangents, x):
    (z,) = cotangents
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
    (x, w), (x_dot, w_dot) = primals, tangents
    return [x + w], [x_dot + w_dot]


@add.set_method
def T(self, cotangents, x, w):
    (grad_L_y,) = cotangents
    return [grad_L_y, grad_L_y]


sub = Operator.binary("sub")
operator_set.register(sub)


@sub.set_method
def jvp(self, primals, tangents):
    (x, w), (x_dot, w_dot) = primals, tangents
    return [x - w], [x_dot - w_dot]


@sub.set_method
def T(self, cotangents, x, w):
    (grad_L_y,) = cotangents
    return [grad_L_y, -grad_L_y]


mul = Operator.binary("mul")
operator_set.register(mul)


@mul.set_method
def jvp(self, primals, tangents):
    (x, w), (x_dot, w_dot) = primals, tangents
    return [x * w], [(x_dot * w) + (w_dot * x)]


@mul.set_method
def T(self, cotangents, x, w):
    (grad_L_y,) = cotangents
    assert (type(x) is PrimalProxy) ^ (type(w) is PrimalProxy)
    if type(x) is PrimalProxy:
        return [grad_L_y * w, None]
    elif type(w) is PrimalProxy:
        return [None, x * grad_L_y]


div = Operator.binary("div")
operator_set.register(div)


@div.set_method
def jvp(self, primals, tangents):
    (x, w), (x_dot, w_dot) = primals, tangents
    return [x / w], [(x_dot / w) + (-w_dot * x * 1 / (w * w))]


@div.set_method
def T(self, cotangents, x, w):
    (grad_L_y,) = cotangents
    return [grad_L_y / w, None]


maximum = Operator.binary("maximum")
operator_set.register(maximum)


@maximum.set_method
def jvp(self, primals, tangents):
    def _balanced_eq(x, z, y):
        xz = (x == z).where(slope.ones_like(z), slope.zeros_like(z))
        yz = (y == z).where(slope.full_like(z, 2.0 if "float" in z.dtype.name else 2), slope.ones_like(z))
        eps = slope.ones_like(z)
        return xz / (yz + eps)  # TODO: nan if no eps for onnxruntime

    (x, w), (x_dot, w_dot) = primals, tangents
    y = x.maximum(w)
    y_dot = x_dot * _balanced_eq(x, y, w) + w_dot * _balanced_eq(w, y, x)
    return [y], [y_dot]


@maximum.set_method
def T(self, cotangents, x, w):
    (grad_L_y,) = cotangents
    return [grad_L_y, None]


equal = Operator.binary("equal")
operator_set.register(equal)


@equal.set_method
def jvp(self, primals, tangents):
    (x, w), _ = primals, tangents
    out_primal = x.equal(w)
    return [out_primal], [slope.full(out_primal.shape, Tensor.bool)]


@equal.set_method
def T(self, cotangents, x, w):
    (grad_L_y,) = cotangents
    grad_L_y = grad_L_y.cast(x.dtype)
    return [grad_L_y, None]


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
        shape_ret = tuple([max(x, w) for x, w in zip(void_x.shape, void_y.shape)])
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
    y = x.max(axes, keepdims)
    y_ = y
    if not keepdims:
        axes = tuple([a if a >= 0 else len(y.shape) + a + 1 for a in axes])
        for a in reversed(sorted(axes)):
            y_ = y_.reshape(y.shape[:a] + (1,) + y.shape[a:])
    locs = x.equal(y_.expand(x.shape))
    locs = locs.cast(x_dot.dtype)
    counts = locs.sum(axes, keepdims)
    y_dot = (x_dot * locs).sum(axes, keepdims)
    y_dot = y_dot / counts.expand(y_dot.shape)

    return [y], [y_dot]


@max.set_method
def T(self, cotangents, x, *, axes, keepdims):
    (z,) = cotangents
    out = z
    if not keepdims:
        axes = [a if a >= 0 else len(out.shape) + a + 1 for a in axes]
        for a in reversed(sorted(axes)):
            out = out.reshape(out.shape[:a] + (1,) + out.shape[a:])
    out = out.expand(x.aval.shape)


sum = Operator.reduce("sum")
operator_set.register(sum)


@sum.set_method
def jvp(self, primals, tangents, *, axes, keepdims):
    (x,), (x_dot,) = primals, tangents
    y = x.sum(axes, keepdims)
    y_dot = x_dot.sum(axes, keepdims)
    return [y], [y_dot]


@sum.set_method
def T(self, cotangents, x, *, axes, keepdims):
    (z,) = cotangents
    out = z
    if not keepdims:
        axes = [a if a >= 0 else len(out.shape) + a + 1 for a in axes]
        for a in reversed(sorted(axes)):
            out = out.reshape(out.shape[:a] + (1,) + out.shape[a:])
    out = out.expand(x.aval.shape)

    return [out]


# -----------------------
# Shape
# -----------------------


expand = Operator.other("expand")
operator_set.register(expand)


@expand.set_method
def args_fixer(self, x, *, shape):
    return (x,), dict(shape=shape)


@expand.set_method
def vmap(self, axis_size, vals_in, dims_in, *, shape):
    (x,), (x_bdim,) = vals_in, dims_in
    shape = list(shape)

    shape = shape[:x_bdim] + [axis_size] + shape[x_bdim:]

    return [self(x, shape)], [x_bdim]


@expand.set_method
def jvp(self, primals, tangents, *, shape, axes=None):
    (x,), (x_dot,) = primals, tangents
    return (
        [self(x, shape=shape)],
        [self(x_dot, shape=shape)],
    )


@expand.set_method
def typecheck(self, x: Typecheckor, *, shape: Sequence[int]) -> List[Typecheckor]:
    e_shape = list(x.shape)
    assert len(e_shape) == len(shape)
    assert all(a <= b for a, b in zip(e_shape, shape))
    return [Typecheckor(tuple(shape), x.dtype)]


@expand.set_method
def T(self, cotangents, x, *, shape):
    (z,) = cotangents
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


reshape = Operator.other("reshape")
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
def T(self, cotangents, x, *, shape):
    (z,) = cotangents
    return [z.reshape(x.aval.shape)]


permute = Operator.other("permute")
operator_set.register(permute)


@permute.set_method
def vmap(self, axis_size, vals_in, dims_in, *, perm):
    (x,), (x_bdim,) = vals_in, dims_in
    perm_ = list(perm)
    x_bdim_ = int(x_bdim)
    assert x_bdim >= 0
    perm = perm[:x_bdim] + [x_bdim] + perm[x_bdim:]
    perm = tuple(d + int(d >= x_bdim) if i != x_bdim else d for i, d in enumerate(perm))
    assert len(set(perm)) == len(perm)
    return [x.tranpose(perm)], [x_bdim]


@permute.set_method
def jvp(self, primals, tangents, *, perm):
    (x,), (x_dot,) = primals, tangents
    return [x.permute(perm)], [x_dot.permute(perm)]


@permute.set_method
def typecheck(self, x: Typecheckor, *, perm: Sequence[int]) -> List[Typecheckor]:
    shape = [x.shape[i] for i in perm]
    return [Typecheckor(shape, x.dtype)]


@permute.set_method
def T(self, cotangents, x, *, perm):
    (z,) = cotangents
    inv_perm = tuple(i[0] for i in sorted(enumerate(perm), key=lambda x: x[1]))
    return [z.permute(inv_perm)]


pad_lowlevel = Operator.other("pad_lowlevel")
operator_set.register(pad_lowlevel)


@pad_lowlevel.set_method
def args_fixer(self, x, *, lo, hi, interior=None, value=0.0):
    if interior is None:
        interior = tuple([0] * len(lo))
    return (x,), dict(lo=lo, hi=hi, interior=interior, value=value)


@pad_lowlevel.set_method
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


@pad_lowlevel.set_method
def jvp(self, primals, tangents, *, lo, hi, interior=None, value=0.0):
    (x,), (x_dot,) = primals, tangents
    return [x.pad_lowlevel(lo, hi, interior, value)], [x_dot.pad_lowlevel(lo, hi, interior, value)]


@pad_lowlevel.set_method
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


@pad_lowlevel.set_method
def T(self, cotangents, x, *, lo, hi, interior=None, value=0.0):
    (z,) = cotangents

    def t_op():
        unpadded = z.slice_lowlevel(
            lo,
            tuple(s - h for s, h in list_zip(z.shape, hi)),
            tuple([1] * len(interior)),
        )
        return unpadded.slice_lowlevel(tuple([0] * len(lo)), unpadded.shape, tuple(r + 1 for r in interior))

    res = t_op() if isinstance(x, PrimalProxy) else None
    return [res]


slice_lowlevel = Operator.other("slice_lowlevel")
operator_set.register(slice_lowlevel)


@slice_lowlevel.set_method
def args_fixer(self, x, *, starts, limits, strides=None):
    if strides is None:
        strides = (1,) * len(starts)
    return (x,), dict(starts=starts, limits=limits, strides=strides)


@slice_lowlevel.set_method
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

    out = x.slice_lowlevel(new_start_indices, new_limit_indices, new_strides)
    return out, x_bdim


@slice_lowlevel.set_method
def jvp(self, primals, tangents, *, starts, limits, strides=None):
    (x,), (x_dot,) = primals, tangents
    return [x.slice_lowlevel(starts, limits, strides)], [x_dot.slice_lowlevel(starts, limits, strides)]


@slice_lowlevel.set_method
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


@slice_lowlevel.set_method
def T(self, cotangents, x, *, starts, limits, strides=None):
    # TODO: compute tuple arithmetic without numpy
    (z,) = cotangents
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

    res = z.pad_lowlevel(lo, hi, interior)
    assert res.shape == x_shape, f"{res.shape=} {x_shape=}"
    return [res]


flip = Operator.other("flip")
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
def T(self, cotangents, x, *, axes):
    (z,) = cotangents
    return [z.flip(axes)]


cat = Operator.other("cat", nary_inputs=True)
operator_set.register(cat)
operator_set.alias(cat, "cat")


@cat.set_method
def args_fixer(self, *xs, axis=0):
    if type(xs) in (tuple, list) and type(xs[0]) in (tuple, list):
        xs = xs[0]
    xs = tuple(xs)
    return xs, dict(axis=axis)


@cat.set_method
def vmap(self, axis_size, vals_in, dims_in, *, axis=0):
    raise NotImplementedError


@cat.set_method
def jvp(self, primals, tangents, *, axis=0):
    return [cat(*primals, axis=axis)], [cat(*tangents, axis=axis)]


@cat.set_method
def typecheck(self, *xs: Typecheckor, axis=0) -> List[Typecheckor]:
    if len(set(x.ndim for x in xs)) != 1:
        msg = "Cannot cat tensors with different numbers of dimensions: got {}."
        raise TypeError(msg.format(", ".join(str(o.shape) for o in xs)))
    if not 0 <= axis < xs[0].ndim:
        msg = "cat dimension out of bounds: dimension {} for shapes {}."
        raise TypeError(msg.format(axis, ", ".join([str(o.shape) for o in xs])))
    shapes = [x.shape[:axis] + x.shape[axis + 1 :] for x in xs]
    if not shapes[:-1] == shapes[1:]:
        msg = (
            "Cannot cat tensors with shapes that differ in dimensions "
            "other than the one being catd: concatenating along "
            "dimension {} for shapes {}."
        )
        shapes = [x.shape for x in xs]
        raise TypeError(msg.format(axis, ", ".join(map(str, shapes))))

    concat_size = sum_py(x.shape[axis] for x in xs)
    ex_shape = xs[0].shape
    return [Typecheckor(ex_shape[:axis] + (concat_size,) + ex_shape[axis + 1 :], xs[0].dtype)]


@cat.set_method
def T(self, cotangents, *xs, axis=0):
    (z,) = cotangents
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
        z.slice_lowlevel(start, limit) if type(o) is PrimalProxy else None
        for o, start, limit in zip(xs, starts, limits)
    ]


# -----------------------
# InitOps
# -----------------------

full = Operator.init("full")
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
def T(self, cotangents, *, shape, fill_value, dtype):
    return [None]


@full.set_method
def typecheck(self, *, shape, fill_value, dtype) -> List[Typecheckor]:
    return [Typecheckor(tuple(shape), dtype)]


random_uniform = Operator.init("random_uniform")
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
def T(self, cotangents, *, shape, dtype):
    return [None]


@random_uniform.set_method
def typecheck(self, *, shape, dtype) -> List[Typecheckor]:
    return [Typecheckor(tuple(shape), dtype)]


random_normal = Operator.init("random_normal")
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
def T(self, cotangents, *, shape, dtype=Tensor.float32):
    return [None]


@random_normal.set_method
def typecheck(self, *, shape, dtype=Tensor.float32) -> List[Typecheckor]:
    return [Typecheckor(tuple(shape), dtype)]


arange = Operator.init("arange")
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
def T(self, cotangents, *, start, stop, stride, dtype):
    return [None]


@arange.set_method
def typecheck(self, *, start, stop, stride, dtype) -> List[Typecheckor]:
    return [Typecheckor((((stop - start) * stride),), dtype)]


# -------------------
# Other
# -------------------


matmul = Operator.other("matmul")
operator_set.register(matmul)


@matmul.set_method
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

    return [Typecheckor(shape, x.dtype)]


@matmul.set_method
def jvp(self, primals, tangents):
    (x, w), (x_dot, w_dot) = primals, tangents
    return [x @ w], [(x_dot @ w) + (x @ w_dot)]


@matmul.set_method
def T(self, cotangents, x, w):
    (grad_L_y,) = cotangents
    assert (type(x) is PrimalProxy) ^ (type(w) is PrimalProxy)
    if type(x) is PrimalProxy:
        return [grad_L_y @ w.transpose(-1, -2), None]
    elif type(w) is PrimalProxy:
        return [None, x.transpose(-1, -2) @ grad_L_y]


conv = Operator.other("conv")
operator_set.register(conv)


@conv.set_method
def args_fixer(self, x, w, *, groups=1, stride=1, dilation=1, padding=0):
    def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
        return (x,) * cnt if isinstance(x, int) else x
    def flatten_seq(l: Iterator):
        return [item for sublist in l for item in sublist]
    (bs, cin_), (cout, cin), HW = x.shape[:2], w.shape[:2], w.shape[2:]
    assert groups * cin == cin_ and len(x.shape) == len(
        w.shape
    ), f"Input axis shape {x.shape} does not match the shape of the ws {w.shape}. ({groups*cin} vs. {cin_})"
    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(HW) or len(padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {x.shape}"
    padding = (
        [padding] * 2 * len(HW)
        if isinstance(padding, int)
        else (padding if len(padding) == 2 * len(HW) else [p for p in padding for _ in range(2)][::-1])
    )
    padding = tuple(padding)
    if isinstance(stride, int):
        stride = make_pair(dilation, len(HW))
    if isinstance(dilation, int):
        dilation = make_pair(dilation, len(HW))
    assert len(HW) == len(stride) and len(HW) == len(
        dilation
    ), f"stride/dilation mismatch kernel:{HW} stride:{stride} dilation:{dilation}"
    return (x, w), dict(groups=groups, stride=stride, dilation=dilation, padding=padding)


@conv.set_method
def typecheck(self, x, w, *, groups, stride, dilation, padding):
    assert x.dtype == w.dtype
    x_shape = x.shape
    w_shape = w.shape
    (bs, cin_), (cout, cin), HW = x_shape[:2], w_shape[:2], w_shape[2:]
    assert (
        groups * cin == cin_
    ), f"Input axis shape {x_shape} does not match the shape of the weights {w_shape}. ({groups*cin} vs. {cin_})"

    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(HW) or len(padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {x_shape}"

    padding_ = (
        [padding] * 2 * len(HW)
        if isinstance(padding, int)
        else (padding if len(padding) == 2 * len(HW) else [p for p in padding for _ in range(2)][::-1])
    )
    padding_ = tuple(padding_)

    # Perform padding, TODO: N-D instead of 2D
    oy = ((x_shape[2] + 2 * padding_[0] - dilation[0] * (HW[0] - 1) - 1) // stride[0]) + 1
    ox = ((x_shape[3] + 2 * padding_[1] - dilation[1] * (HW[1] - 1) - 1) // stride[1]) + 1

    # Shape after pooling
    y_shape = (bs, groups * cin, oy, ox, *HW)

    rcout, oyx = cout // groups, y_shape[2 : -len(HW)]

    # Reshape and expand dimensions
    y_shape = (bs, groups, cin, rcout, *oyx, *HW)

    # Permute dimensions
    y_shape = (
        y_shape[0],
        y_shape[1],
        y_shape[3],
        *[4 + i for i in range(len(oyx))],
        y_shape[2],
        *[4 + len(oyx) + i for i in range(len(HW))],
    )

    # Shape after convolution
    result_shape = (bs, cout, *oyx)

    return [Typecheckor(result_shape, x.dtype)]


@conv.set_method
def jvp(self, primals, tangents, *, groups, stride, dilation, padding):
    (x, w), (x_dot, w_dot) = primals, tangents
    y = x.conv(w)
    y_dot1 = x_dot.conv(w, groups=groups, stride=stride, dilation=dilation, padding=padding)
    y_dot2 = x.conv(w_dot, groups=groups, stride=stride, dilation=dilation, padding=padding)

    return [y], [y_dot1 + y_dot2]


@conv.set_method
def T(self, cotangents, x, w, *, groups, stride, dilation, padding):
    (grad_L_y,) = cotangents
    if type(x) is PrimalProxy:
        grad_L_x = grad_L_y.conv_transpose(
            w, groups=groups, stride=stride, dilation=dilation, padding=padding, output_padding=0
        )
        return [grad_L_x, None]
    elif type(w) is PrimalProxy:
        x_T = x.transpose(0, 1)
        grad_L_y_T = grad_L_y.transpose(0, 1)
        grad_L_w = x_T.conv(grad_L_y_T, groups=groups, stride=stride, dilation=dilation, padding=padding).transpose(
            0, 1
        )
        return [None, grad_L_w]


conv_transpose = Operator.other("conv_transpose")
operator_set.register(conv_transpose)


@conv_transpose.set_method
def args_fixer(self, x, w, *, groups=1, stride=1, dilation=1, padding=0, output_padding=0):
    def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
        return (x,) * cnt if isinstance(x, int) else x
    def flatten_seq(l: Iterator):
        return [item for sublist in l for item in sublist]
    if isinstance(output_padding, int):
        if output_padding != 0:
            raise NotImplementedError
    elif isinstance(output_padding, tuple):
        if not all(o != 0 for o in output_padding):
            raise NotImplementedError
    (bs, cin_), (cin, cout), HW = x.shape[:2], w.shape[:2], w.shape[2:]
    assert groups * cin == cin_ and len(x.shape) == len(
        w.shape
    ), f"Input axis shape {x.shape} does not match the shape of the ws {w.shape}. ({groups*cin} vs. {cin_})"
    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(HW) or len(padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {x.shape}"

    if isinstance(output_padding, (tuple, list)):
        assert len(output_padding) == 2 * len(HW) or len(output_padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(output_padding)} for tensor of shape {x.shape}"
    padding = tuple(
        [padding] * 2 * len(HW)
        if isinstance(padding, int)
        else (padding if len(padding) == 2 * len(HW) else [p for p in padding for _ in range(2)][::-1])
    )
    output_padding = tuple(
        [output_padding] * 2 * len(HW)
        if isinstance(output_padding, int)
        else (
            output_padding
            if len(output_padding) == 2 * len(HW)
            else [p for p in output_padding for _ in range(2)][::-1]
        )
    )
    if isinstance(stride, int):
        stride = make_pair(dilation, len(HW))
    if isinstance(dilation, int):
        dilation = make_pair(dilation, len(HW))
    assert len(HW) == len(stride) and len(HW) == len(
        dilation
    ), f"stride/dilation mismatch kernel:{HW} stride:{stride} dilation:{dilation}"
    return (x, w), dict(groups=groups, stride=stride, dilation=dilation, padding=padding, output_padding=output_padding)


@conv_transpose.set_method
def typecheck(self, x, w, *, groups, stride, dilation, padding, output_padding):
    assert x.dtype == w.dtype
    x_shape = x.shape
    w_shape = w.shape
    (bs, cin_), (cin, cout), HW = x_shape[:2], w_shape[:2], w_shape[2:]
    assert (
        groups * cin == cin_
    ), f"Input axis shape {x_shape} does not match the shape of the ws {w_shape}. ({groups*cin} vs. {cin_})"

    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(HW) or len(padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {x_shape}"

    if isinstance(output_padding, (tuple, list)):
        assert len(output_padding) == 2 * len(HW) or len(output_padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(output_padding)} for tensor of shape {x_shape}"

    padding = tuple(
        [padding] * 2 * len(HW)
        if isinstance(padding, int)
        else (padding if len(padding) == 2 * len(HW) else [p for p in padding for _ in range(2)][::-1])
    )

    output_padding = tuple(
        [output_padding] * 2 * len(HW)
        if isinstance(output_padding, int)
        else (
            output_padding
            if len(output_padding) == 2 * len(HW)
            else [p for p in output_padding for _ in range(2)][::-1]
        )
    )

    if isinstance(stride, int):
        stride = [stride] * len(HW)

    if isinstance(dilation, int):
        dilation = [dilation] * len(HW)

    assert len(HW) == len(stride) and len(HW) == len(
        dilation
    ), f"stride/dilation mismatch kernel:{HW} stride:{stride} dilation:{dilation}"

    # Calculate output shape
    result_shape = tuple(
        [bs, cout]
        + [
            (s - 1) * stride[i] - 2 * padding[i] + dilation[i] * (HW[i] - 1) + output_padding[i] + 1
            for i, s in enumerate(x_shape[2:])
        ]
    )
    return [Typecheckor(result_shape, x.dtype)]


@conv_transpose.set_method
def jvp(self, primals, tangents, *, groups, stride, dilation, padding, output_padding):
    (x, w), (x_dot, w_dot) = primals, tangents
    y = x.conv_transpose(w)
    y_dot1 = x_dot.conv_transpose(
        w, groups=groups, stride=stride, dilation=dilation, padding=padding, output_padding=output_padding
    )
    y_dot2 = x.conv_transpose(
        w_dot, groups=groups, stride=stride, dilation=dilation, padding=padding, output_padding=output_padding
    )
    print(y.shape)

    return [y], [y_dot1 + y_dot2]


@conv_transpose.set_method
def T(self, cotangents, x, w, *, groups, stride, dilation, padding, output_padding):
    (grad_L_y,) = cotangents
    if type(x) is PrimalProxy:
        grad_L_x = grad_L_y.conv(w, groups=groups, stride=stride, dilation=dilation, padding=padding)
        return [grad_L_x, None]
    elif type(w) is PrimalProxy:
        x_T = x.transpose(0, 1)
        grad_L_y_T = grad_L_y.transpose(0, 1)
        grad_L_w = grad_L_y_T.conv(x_T, groups=groups, stride=stride, dilation=dilation, padding=padding)
        return [None, grad_L_w]


# --------------
# Compiler
# --------------

#

compile_py = compile
compiler = Compiler(name="onnxruntime", default_dtype=Tensor.float32, default_device=slope.SLOPE_DEVICE)
compiler.set_dtype_map(
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


@compiler.set_method
def from_numpy(self, val, dtype=compiler.default_dtype_value, device=compiler.default_device):
    device_type, device_id = device.split(":") if ":" in device else (device, 0)
    np_val = np.array(val, dtype=dtype.numpy)
    val = onnxruntime.OrtValue.ortvalue_from_numpy(np_val, device_type=device_type, device_id=device_id)
    return Tensor(TensorBuffer(val))


@compiler.set_method
def numpy_of(self, tensor):
    return tensor.buf.val.numpy()


@compiler.set_method
def device_of(self, tensor):
    return tensor.buf.val.device_name()


@compiler.set_method
def shape_of(self, tensor):
    return tuple(tensor.buf.val.shape())


@compiler.set_method
def dtype_of(self, tensor):
    return self.dtype_map_inv[tensor.buf.val.data_type().replace("tensor(", "").replace(")", "")]


@compiler.set_method
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


@compiler.set_method
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


@compiler.set_method
def codegen(self, program, args, *, fn_name: str = "main", fn_defs=dict()) -> List[Any]:
    if fn_name == "main":
        assert not hasattr(self, "fn_count")
        self.fn_count = 0

    def indent(code, amount):
        spaces = " " * (len(code) - len(code.lstrip()))
        spaces += " " * amount
        return "\n".join([spaces + line for line in code.strip().split("\n")])

    # codegen is recursive if jit-of-jit happens
    backend: Dict[slope.Var, Any] = {}
    il1 = 4  # indent length
    body_code_lines = []

    for inb in program.in_binders:
        prefix = "x" if type(inb.aval) is Typecheckor else "c"
        idx = sum_py([1 if v["name"][0] == prefix else 0 for v in backend.values()])
        backend[inb] = dict(name=f"{prefix}{idx}", type=inb.aval)

    for instruction in program.instructions:
        if len(instruction.out_binders) == 0:  # skip codegen for function returns nothing
            continue
        in_vals = list_map(lambda x: backend[x]["name"], instruction.inputs)
        for outb in instruction.out_binders:
            prefix = "y" if outb in program.outs else "z"
            idx = sum_py([1 if v["name"][0] == prefix else 0 for v in backend.values()])
            backend[outb] = dict(name=f"{prefix}{idx}", type=outb.aval)

        out_vals = list_map(lambda z: backend[z]["name"], instruction.out_binders)
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

    # inb_consts = [v for v in backend.values() if "c" in v["name"]]
    # const_type_strs = [f"{self.dtype_map[c['type'].dtype]}[{repr(c['type'].shape)[1:-1]}] {c['name']}" for c in inb_consts]

    in_binders = list_map(lambda x: backend[x], program.in_binders)
    arg_type_strs = [
        f"{self.dtype_map[i['type'].dtype]}[{repr(list(i['type'].shape))[1:-1]}] {i['name']}" for i in in_binders
    ]
    fn_args_str = ", ".join(arg_type_strs)

    outs = list_map(lambda x: backend[x], program.outs)  # TODO: input that is output should has identity op
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


compiler.set_impl(operator_set.cast)(lambda self, x, *, dtype: f"ret = Cast<to={onnx_dtype_enum_map[dtype]}>({x})")
compiler.set_impl(operator_set.stop_gradient)(lambda self, x, *, dtype: f"ret = Identity({x})")
compiler.set_impl(operator_set.neg)(lambda self, x: f"ret =  Neg({x})")
compiler.set_impl(operator_set.sqrt)(lambda self, x: f"ret = Sqrt({x})")
compiler.set_impl(operator_set.exp)(lambda self, x: f"ret = Exp({x})")
compiler.set_impl(operator_set.log)(lambda self, x: f"ret = Log({x})")
compiler.set_impl(operator_set.sin)(lambda self, x: f"ret = Sin({x})")
compiler.set_impl(operator_set.add)(lambda self, x, w: f"ret = Add({x}, {w})")
compiler.set_impl(operator_set.sub)(lambda self, x, w: f"ret = Sub({x}, {w})")
compiler.set_impl(operator_set.mul)(lambda self, x, w: f"ret = Mul({x}, {w})")
compiler.set_impl(operator_set.div)(lambda self, x, w: f"ret = Div({x}, {w})")
compiler.set_impl(operator_set.invert)(lambda self, x: f"ret = Not({x})")
compiler.set_impl(operator_set.equal)(lambda self, x, w: f"ret = Equal({x}, {w})")
compiler.set_impl(operator_set.maximum)(lambda self, x, w: f"ret = Max({x}, {w})")
compiler.set_impl(operator_set.matmul)(lambda self, x, w: f"ret = MatMul({x}, {w})")


@compiler.set_impl(operator_set.sum)
def sum_impl(self, x, *, axes, keepdims):
    return f"""
ret_axes = Constant <value = int64[{len(axes)}]  {{ {repr(axes)[1:(-1 if len(axes) > 1 else -2)]} }} >()
ret = ReduceSum<keepdims={int(keepdims)}> ({x}, ret_axes)
"""


@compiler.set_impl(operator_set.max)
def max_impl(self, x, *, axes, keepdims):
    return f"""
ret_axes = Constant <value = int64[{len(axes)}]  {{ {repr(axes)[1:(-1 if len(axes) > 1 else -2)]} }} >()
ret = ReduceMax<keepdims={int(keepdims)}> ({x}, ret_axes)
"""


@compiler.set_impl(operator_set.arange)
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
@compiler.set_impl(operator_set.full)
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


@compiler.set_impl(operator_set.random_uniform)
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


@compiler.set_impl(operator_set.random_normal)
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


@compiler.set_impl(operator_set.expand)
def expand_impl(self, x, *, shape):
    return f"""
ret_shape = Constant <value = int64[{len(shape)}] {{ {repr(list(shape))[1:-1]} }} >()
ret = Expand ({x}, ret_shape)
"""


@compiler.set_impl(operator_set.reshape)
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


@compiler.set_impl(operator_set.pad_lowlevel)
def pad_lowlevel_impl(self, x, *, lo, hi, interior, value):  # TODO: interior not used
    padding = lo + hi
    return f"""
ret_padding = Constant <value = int64[{len(padding)}] {padding}>()
ret_constant_value =  Constant <value = {value} >()
ret = Pad({x} ret_padding, ret_constant_value)
"""


@compiler.set_impl(operator_set.slice_lowlevel)
def slice_lowlevel_impl(self, x, *, starts, limits, strides):
    return f"""
ret_starts = Constant <value = int64[{len(starts)}] {starts}>()
ret_ends = Constant <value = int64[{len(limits)}] {limits}>()
ret_steps = Constant <value = int64[{len(strides)}] {strides}>()
ret = Slice({x}, ret_starts, ret_ends, steps=ret_steps))])
"""


@compiler.set_impl(operator_set.cat)
def cat_impl(self, *xs, axis):
    return f"ret = Concat< axis={axis}>({repr(list(xs))[1:-1]})"


@compiler.set_impl(operator_set.permute)
def permute_impl(self, x, *, perm):
    return f"ret = Transpose<perm={repr(list(perm))}>({x})"


@compiler.set_impl(operator_set.flip)
def flip_impl(self, x, *, axes):
    return f"""
ret_starts = Constant <value = int64[{len(axes)}] {", ".join(["0"] * len(axes))}>()
ret_ends = Constant <value = int64[{len(axes)}] {", ".join(["-1"] * len(axes))}>()
ret_axes = Constant <value = int64[{len(axes)}] {repr(list(axes))[1:-1]}>()
ret_steps = Constant <value = int64[{len(axes)}] {", ".join(["-1"] * len(axes))}>()
ret = Slice({x}, ret_starts, ret_ends, ret_axes, steps)])
"""


@compiler.set_impl(operator_set.conv)
def conv_impl(self, x, w, *, groups, stride, dilation, padding):
    dilations_attr = f"dilations=[{repr(list(dilation))[1:-1]}]"
    pads_attr = f"pads=[{repr(list(padding))[1:-1]}]"
    strides_attr = f"strides=[{repr(list(stride))[1:-1]}]"
    group_attr = f"group={groups}"
    return f"""ret = Conv<{dilations_attr}, {pads_attr}, {strides_attr}, {group_attr}>({x}, {w})"""


@compiler.set_impl(operator_set.conv_transpose)
def conv_transpose_impl(self, x, w, *, groups, stride, dilation, padding, output_padding):
    dilations_attr = f"dilations=[{repr(list(dilation))[1:-1]}]"
    pads_attr = f"pads=[{repr(list(padding))[1:-1]}]"
    pads_attr = f"pads=[{repr(list(padding))[1:-1]}]"
    output_padding_attr = f"pads=[{repr(list(output_padding))[1:-1]}]"
    strides_attr = f"strides=[{repr(list(stride))[1:-1]}]"
    group_attr = f"group={groups}"
    return f"""ret = ConvTranspose<{dilations_attr}, {group_attr}, {output_padding_attr}, {pads_attr}, {strides_attr}>({x}, {w})"""


@compiler.set_impl(slope.core.jit_op)
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


@compiler.set_impl(slope.core.procedure_op)
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


from slope.backends.common_procedures import procedure_set

onnxruntime_backend = Backend(operator_set, procedure_set, compiler)