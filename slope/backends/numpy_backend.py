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
from typing import Tuple, List, Dict, Any, Optional, Sequence, Union, Iterator, Callable
from collections import defaultdict
import importlib
import os
import functools

sum_py = sum
max_py = max
abs_py = abs
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
        return ((x == z).where(slope.ones_like(z), slope.zeros_like(z))) / (
            (y == z).where(slope.full_like(z, 2.0 if "float" in z.dtype.name else 2), slope.ones_like(z))
        )

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
    void_x = Typecheckor(x.shape, Tensor.bool)
    void_y = Typecheckor(y.shape, Tensor.bool)
    if void_x == void_y:
        return [void_x]
    shape_delta = len(void_x.shape) - len(void_y.shape)
    if shape_delta > 0:
        void_y = Typecheckor((1,) * shape_delta + void_y.shape, Tensor.bool)
    elif shape_delta < 0:
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
    out = x.max(axes, keepdims)
    _out = out
    if not keepdims:
        axes = [a if a >= 0 else len(out.shape) + a + 1 for a in axes]
        for a in reversed(sorted(axes)):
            _out = _out.reshape(out.shape[:a] + (1,) + out.shape[a:])
    locs = x.equal(_out.expand(x.shape))
    locs = locs.cast(x_dot.dtype)
    counts = locs.sum(axes, keepdims)
    y_dot = (x_dot * locs).sum(axes, keepdims)
    y_dot = y_dot / counts.expand(y_dot.shape)
    return [out], [y_dot]


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
    original_shape = list(x.shape)
    assert len(original_shape) == len(shape)
    assert all((od == d) or (od < d and od == 1) for od, d in zip(original_shape, shape))
    # assert all(a <= b for a, b in zip(e_shape, shape))
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
# def args_fixer(self, x, *, shape):
def args_fixer(self, x, *shape):
    if isinstance(shape[0], (tuple, list)):
        shape = shape[0]
    shape = tuple(shape)
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
    # inv_perm =  tuple(np.argsort(perm))
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
# LoadOps
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
def args_fixer(self, *, start, stop=None, stride=None, dtype=Tensor.int32):
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
    return [Typecheckor((stop - start,) * stride, dtype)]


# --------------
# Compiler
# --------------


compile_py = compile
compiler = Compiler(name="numpy", default_dtype=Tensor.dtype_names[slope.SLOPE_DTYPE])
compiler.set_dtype_map(
    {
        Tensor.float32: np.dtype("float32"),
        Tensor.int64: np.dtype("int64"),
        Tensor.int32: np.dtype("int32"),
        Tensor.int8: np.dtype("int8"),
        Tensor.bool: np.dtype("bool"),
    }
)


@compiler.set_method
def from_numpy(self, val, dtype=compiler.default_dtype_value):
    val = np.array(val, dtype=compiler.dtype_map[dtype])
    return Tensor(TensorBuffer(val))


@compiler.set_method
def numpy_of(self, tensor):
    return tensor.buf.val


@compiler.set_method
def device_of(self, tensor):
    return "cpu"


@compiler.set_method
def shape_of(self, tensor):
    return tensor.buf.val.shape


@compiler.set_method
def dtype_of(self, tensor):
    return self.dtype_map_inv[tensor.buf.val.dtype]


@compiler.set_method
def export(self, jit_object: slope.core.JitObject, output_path, *args, **kwargs):
    code = jit_object.code
    os.makedirs(output_path, exist_ok=True)
    consts_dir_path = os.path.join(output_path, "consts")
    os.makedirs(consts_dir_path, exist_ok=True)
    in_binders = jit_object.codegen_out["in_binders"]
    outs = jit_object.codegen_out["outs"]
    num_consts = jit_object.program.num_consts
    load_consts_code = ""
    for i in range(num_consts):
        const_name = in_binders[i]["name"]
        const_path = os.path.join(consts_dir_path, f"{const_name}.npy")
        load_consts_code += f"""{const_name} = np.load(os.path.join(consts_dir_path, "{const_name}.npy"))\n"""
        np.save(const_path, in_binders[i]["type"].numpy())
    input_args_code = ", ".join(ib["name"] for ib in in_binders[num_consts:])
    args_code = ", ".join(ib["name"] for ib in in_binders)
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
    module_code = f"""import numpy as np
import os
root_path = os.path.dirname(__file__)
consts_dir_path =  os.path.join(root_path, "consts")
{load_consts_code}
{code}

input_arg_names = {input_arg_names}
out_names = {outs_names}

def run({input_args_code}):
    return main({args_code})

if __name__ == "__main__":
{test_input_code}
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
    deps_dict = dict()
    deps_dict["numpy"] = importlib.import_module("numpy")
    deps_dict["np"] = deps_dict["numpy"]
    deps_dict["math"] = importlib.import_module("math")
    code_lines = codegen_out["code_lines"]
    exec_locals = dict()
    code = "\n".join(code_lines)
    exec(compile_py(code, "<string>", "exec"), deps_dict, exec_locals)
    fn = exec_locals["main"]
    return fn, code


@compiler.set_method
def codegen(self, program, args, *, fn_name: str = "main", fn_defs=dict()) -> List[Any]:
    if fn_name == "main":
        assert not hasattr(self, "fn_count")
        self.fn_count = 0
        assert not hasattr(self, "depth")
        self.depth = 0

    def indent(code, amount):
        spaces = " " * (len(code) - len(code.lstrip()))
        spaces += " " * amount
        return "\n".join([spaces + line for line in code.strip().split("\n")])

    # codegen is recursive if jit-of-jit happens
    backend: Dict[slope.Var, Any] = {}
    il1 = (self.depth + 1) * 4
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
            self.depth += 1
            rhs, fn_defs = self.impls[instruction.op](program, args, instruction, in_vals, fn_defs)
            self.depth -= 1
            impl_code = f"{lhs} = {rhs}"
        else:
            impl_code = self.impls[instruction.op](*in_vals, **instruction.params)
            if len(out_vals) == 1:
                impl_code = impl_code.replace("ret", out_vals[0])
            else:
                raise NotImplementedError
        for np_dtype in self.dtype_map.values():  # fix dtype kwargs not having 'np.' prefix
            impl_code = impl_code.replace(
                np_dtype.name, "bool" if np_dtype is np.dtype("bool") else f"np.{np_dtype.name}"
            )

        for impl_code_line in impl_code.split("\n"):  # handle multi-line code
            body_code_lines += [indent(impl_code_line, il1)]

    in_binders = list_map(lambda x: backend[x], program.in_binders)
    arg_type_strs = [i["name"] for i in in_binders]
    # arg_type_asserts = [
    #     f"{self.dtype_map[i['type'].dtype]}[{repr(list(i['type'].shape))[1:-1]}] {i['name']}" for i in in_binders
    # ]
    fn_args_str = ", ".join(arg_type_strs)

    outs = list_map(lambda x: backend[x], program.outs)  # TODO: input that is output should has identity op
    out_type_strs = [o["name"] for o in outs]
    # out_type_asserts = [
    #     f"{self.dtype_map[o['type'].dtype]}[{repr(list(o['type'].shape))[1:-1]}] {o['name']}" for o in outs
    # ]

    head_code_lines = []
    head_code_lines += [f"def {fn_name} ({fn_args_str}):"]
    out_type_str = ", ".join(out_type_strs) + ("," if len(outs) == 1 else "")
    return_line = [indent(f"return {out_type_str}", il1)]

    functions_code_lines = []
    for op, fn_def_code_lines in fn_defs.items():
        # functions_code_lines += fn_def_code_lines
        functions_code_lines += fn_def_code_lines

    code_lines = head_code_lines + [indent(l, il1) for l in functions_code_lines] + body_code_lines + return_line
    slope.dblog(f"\n-- {program.name} codegen:\n\n" + "\n".join(code_lines) + "\n\n==\n", enable=slope.LOG_JIT)

    if fn_name == "main":
        del self.fn_count
        del self.depth
    assert len(outs) == len(program.outs)
    return dict(code_lines=code_lines, fn_defs=fn_defs, in_binders=in_binders, outs=outs)


### Operator Impls

compiler.set_impl(operator_set.cast)(lambda self, x, *, dtype: f"ret = {x}.astype(dtype={dtype})")
compiler.set_impl(operator_set.stop_gradient)(lambda self, x, *, dtype: f"ret = {x}")
compiler.set_impl(operator_set.neg)(lambda self, x: f"ret = np.negative({x})")
compiler.set_impl(operator_set.sqrt)(lambda self, x: f"ret = np.sqrt({x})")
compiler.set_impl(operator_set.exp)(lambda self, x: f"ret = np.exp({x})")
compiler.set_impl(operator_set.log)(lambda self, x: f"ret = np.log({x})")
compiler.set_impl(operator_set.sin)(lambda self, x: f"ret = np.sin({x})")
compiler.set_impl(operator_set.add)(lambda self, x, w: f"ret = np.add({x}, {w})")
compiler.set_impl(operator_set.sub)(lambda self, x, w: f"ret = np.subtract({x}, {w})")
compiler.set_impl(operator_set.mul)(lambda self, x, w: f"ret = np.multiply({x}, {w})")
compiler.set_impl(operator_set.div)(lambda self, x, w: f"ret = np.divide({x}, {w})")
compiler.set_impl(operator_set.invert)(lambda self, x: f"ret = np.invert({x})")
compiler.set_impl(operator_set.equal)(lambda self, x, w: f"ret = np.equal({x}, {w})")
compiler.set_impl(operator_set.maximum)(lambda self, x, w: f"ret = np.maximum({x}, {w})")
compiler.set_impl(operator_set.sum)(
    lambda self, x, *, axes, keepdims: f"ret = np.sum({x}, axis={axes}, keepdims={keepdims})"
)
compiler.set_impl(operator_set.max)(
    lambda self, x, *, axes, keepdims: f"ret = np.max({x}, axis={axes}, keepdims={keepdims})"
)
compiler.set_impl(operator_set.arange)(
    lambda self, *, start, stop, stride, dtype: f"ret = np.arange(start={start}, stop={stop}, step={stride}, dtype={dtype})"
)
compiler.set_impl(operator_set.full)(
    lambda self, *, shape, fill_value, dtype: f"ret = np.full(shape={shape}, fill_value={fill_value}, dtype={dtype})"
)

compiler.set_impl(operator_set.random_uniform)(
    lambda self, *, shape, dtype: (
        f"ret = {'np.array(' if shape == () else ''}np.random.uniform(loc=np.zeros(shape={shape})){')' if shape == () else ''}.astype(dtype={dtype})"
    )
)
compiler.set_impl(operator_set.random_normal)(
    lambda self, *, shape, dtype: (
        f"ret = {'np.array(' if shape == () else ''}np.random.normal(loc=np.zeros(shape={shape})){')' if shape == () else ''}.astype(dtype={dtype})"
    )
)
compiler.set_impl(operator_set.expand)(lambda self, x, *, shape: f"ret = np.broadcast_to({x}, shape={shape})")

compiler.set_impl(operator_set.reshape)(lambda self, x, *, shape: f"ret = np.reshape({x}, newshape={shape})")
compiler.set_impl(operator_set.pad_lowlevel)(  # TODO: interior not used
    lambda self, x, *, lo, hi, interior, value: f"ret = np.pad({x}, list(zip({lo}, {hi})), constant_values={value})"
)


compiler.set_impl(operator_set.slice_lowlevel)(
    lambda self, x, *, starts, limits, strides: f"ret = {x}[tuple(slice(s, l, st) for s, l, st in zip({starts}, {limits}, {strides}))]"
)

compiler.set_impl(operator_set.cat)(lambda self, *xs, axis: f"ret = np.cat({xs}, axis={axis})")
compiler.set_impl(operator_set.permute)(lambda self, x, *, perm: f"ret = np.transpose({x}, axes={perm})")
compiler.set_impl(operator_set.flip)(lambda self, x, *, axes: f"ret = np.flip({x}, axis={axes})")


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
    rhs = f"{jit_name}({args_str})"
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
    rhs = f"{proc_name}({args_str})"
    return rhs, fn_defs

from slope.backends.common_procedures import procedure_set

numpy_backend = Backend(operator_set, procedure_set, compiler)
