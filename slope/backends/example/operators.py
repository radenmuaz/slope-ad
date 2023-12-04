import slope
from slope.core import OperatorSet, Operator, Tensor, Typecheckor, PrimalProxy

import math
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Sequence, Union, Iterator, Callable
from collections import defaultdict

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
    return [None]
    # (z,) = cotangents
    # assert type(x) is PrimalProxy
    # return [slope.zeros_like(x)]


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


@cast.set_method
def T(self, cotangents, x):
    (grad_L_y,) = cotangents
    return [grad_L_y.cast(x.dtype)]


sqrt = Operator.unary("sqrt")
operator_set.register(sqrt)


@sqrt.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    y = x.sqrt()
    return [y], [x_dot / (y * 2)]


@sqrt.set_method
def T(self, cotangents, x):
    (grad_L_y,) = cotangents
    return [grad_L_y / (x.sqrt() * 2)]


sin = Operator.unary("sin")
operator_set.register(sin)


@sin.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x.sin()], [(x_dot * ((math.pi / 2) - x).sin())]


@sin.set_method
def T(self, cotangents, x):
    (grad_L_y,) = cotangents
    return [(grad_L_y * ((math.pi / 2) - x).sin())]


exp = Operator.unary("exp")
operator_set.register(exp)


@exp.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    y = x.exp()
    return [y], [x_dot * y]


@exp.set_method
def T(self, cotangents, x):
    (grad_L_y,) = cotangents
    return [1 / grad_L_y]


log = Operator.unary("log")
operator_set.register(log)


@log.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x.log()], [x_dot / x]


@log.set_method
def T(self, cotangents, x):
    (grad_L_y,) = cotangents
    return [1 / grad_L_y]


neg = Operator.unary("neg")
operator_set.register(neg)


@neg.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [-x], [-x_dot]


@neg.set_method
def T(self, cotangents, x):
    (grad_L_y,) = cotangents
    return [-grad_L_y]


invert = Operator.unary("invert")
operator_set.register(invert)


@invert.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [~x], [~x_dot]


@invert.set_method
def T(self, cotangents, x):
    (grad_L_y,) = cotangents
    return [~grad_L_y]


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


pow = Operator.binary("pow")
operator_set.register(pow)


@pow.set_method
def jvp(self, primals, tangents):
    (x, w), (x_dot, w_dot) = primals, tangents
    y = x**w
    y_dot1 = x_dot * (w * (x ** (w - slope.ones_like(w))))
    y_dot2 = w_dot * (y * (x if x != 0.0 else slope.zeros_like(x)).log())
    return [y], [y_dot1 + y_dot2]


@pow.set_method
def T(self, cotangents, x, w):
    (grad_L_y,) = cotangents
    assert (type(x) is PrimalProxy) ^ (type(w) is PrimalProxy)
    if type(x) is PrimalProxy:
        return [(grad_L_y * (w * (x ** (w - slope.ones_like(w))))), None]
    elif type(w) is PrimalProxy:
        return [None, grad_L_y * ((x**w) * (x.log() if x != 0.0 else slope.zeros_like(x)))]


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
    if x.dtype != y.dtype:
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
    # TODO: this is sum gradient, define max gradient
    (grad_L_y,) = cotangents
    grad_L_x = grad_L_y
    if not keepdims:
        axes = [a if a >= 0 else len(grad_L_x.shape) + a + 1 for a in axes]
        for a in reversed(sorted(axes)):
            grad_L_x = grad_L_x.reshape(grad_L_x.shape[:a] + (1,) + grad_L_x.shape[a:])
    grad_L_x = grad_L_x.expand(x.aval.shape)


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
    (grad_L_y,) = cotangents
    grad_L_x = grad_L_y
    if not keepdims:
        axes = [a if a >= 0 else len(grad_L_x.shape) + a + 1 for a in axes]
        for a in reversed(sorted(axes)):
            grad_L_x = grad_L_x.reshape(grad_L_x.shape[:a] + (1,) + grad_L_x.shape[a:])
    grad_L_x = grad_L_x.expand(x.aval.shape)

    return [grad_L_x]


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
    (grad_L_y,) = cotangents
    grad_L_x = grad_L_y
    if x.aval.shape == grad_L_x.shape:
        return [grad_L_x]
    else:
        b_axes = []
        assert len(x.aval.shape) == len(grad_L_x.shape)
        for i, (xd, od) in enumerate(zip(x.aval.shape, grad_L_x.shape)):
            if xd != od:
                b_axes += [i]
        grad_L_x = grad_L_x.sum(axes=tuple(b_axes), keepdims=True)
    if grad_L_x.shape != x.aval.shape:
        raise ValueError(f"not same {grad_L_x.shape=}, {x.aval.shape=}")
    return [grad_L_x]


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


pad = Operator.other("pad")
operator_set.register(pad)


@pad.set_method
def args_fixer(self, x, *, padding, mode="constant", value=0.0):
    if isinstance(padding, int):
        padding = (padding, padding) * x.ndim
    elif all(isinstance(pw, int) for pw in padding):
        assert (x.ndim * 2) % len(padding) == 0
        padding = (0, 0) * (x.ndim - len(padding) // 2) + tuple(padding)
    return (x,), dict(padding=padding, mode=mode, value=value)


@pad.set_method
def vmap(self, axis_size, vals_in, dims_in, *, padding, mode, value):
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


@pad.set_method
def jvp(self, primals, tangents, *, padding, mode, value):
    (x,), (x_dot,) = primals, tangents
    return [x.pad(padding, mode, value)], [x_dot.pad(padding, mode, value)]


@pad.set_method
def typecheck(self, x: Typecheckor, *, padding, mode, value) -> List[Typecheckor]:
    lo = [padding[i] for i in range(0, len(padding), 2)]
    hi = [padding[i] for i in range(1, len(padding), 2)]
    interior = [0] * (len(padding) // 2)

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


@pad.set_method
def T(self, cotangents, x, *, lo, hi, interior=None, value=0.0):
    (z,) = cotangents

    def t_op():
        unpadded = z.slice(
            lo,
            tuple(s - h for s, h in list_zip(z.shape, hi)),
            tuple([1] * len(interior)),
        )
        return unpadded.slice(tuple([0] * len(lo)), unpadded.shape, tuple(r + 1 for r in interior))

    res = t_op() if isinstance(x, PrimalProxy) else None
    return [res]


slice = Operator.other("slice")
operator_set.register(slice)


@slice.set_method
def args_fixer(self, x, *, starts, limits, strides=None):
    if strides is None:
        strides = (1,) * len(starts)
    return (x,), dict(starts=starts, limits=limits, strides=strides)


@slice.set_method
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

    out = x.slice(new_start_indices, new_limit_indices, new_strides)
    return out, x_bdim


@slice.set_method
def jvp(self, primals, tangents, *, starts, limits, strides=None):
    (x,), (x_dot,) = primals, tangents
    return [x.slice(starts, limits, strides)], [x_dot.slice(starts, limits, strides)]


@slice.set_method
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


@slice.set_method
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

    res = z.pad(lo, hi, interior)
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

    return [z.slice(start, limit) if type(o) is PrimalProxy else None for o, start, limit in zip(xs, starts, limits)]


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
        assert grad_L_x.shape == x.shape
        return [grad_L_x, None]
    elif type(w) is PrimalProxy:
        x_T = x.transpose(0, 1)
        grad_L_y_T = grad_L_y.transpose(0, 1)
        grad_L_w = x_T.conv(grad_L_y_T, groups=groups, stride=stride, dilation=dilation, padding=padding).transpose(
            0, 1
        )
        assert grad_L_w.shape == w.shape
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
