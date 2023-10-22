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
from typing import (
    Tuple,
    List,
    Dict,
    Any,
    Optional,
    Sequence,
    Union,
    Iterator,
)
from collections import defaultdict
import importlib

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


convert = Operator.unary("convert")
astype = convert
operator_set.register(convert)
operator_set.alias(convert, "astype")


@convert.set_method
def typecheck(self, x: Typecheckor, *, dtype) -> List[Typecheckor]:
    return [Typecheckor(x.shape, dtype)]


@convert.set_method
def jvp(self, primals, tangents, *, dtype):
    (x,), (x_dot,) = primals, tangents
    return [x.convert(dtype)], [x_dot.convert(dtype)]


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
        return ((x == z).where(slope.ones_like(z), slope.zeros_like(z))) / (
            (y == z).where(slope.full_like(z, 2), slope.ones_like(z))
        )

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
    return [out_primal], [slope.zeros(out_primal.shape, out_primal.dtype)]


@equal.set_method
def T(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar, None]


max = Operator.reduce("max")
operator_set.register(max)


@max.set_method
def jvp(self, primals, tangents, *, axes=(), keepdims=False):
    (x,), (x_dot,) = primals, tangents
    out = x.max(axes, keepdims)
    _out = out
    if not keepdims:
        axes = [a if a >= 0 else len(out.shape) + a + 1 for a in axes]
        for a in reversed(sorted(axes)):
            _out = _out.reshape(out.shape[:a] + (1,) + out.shape[a:])
    locs = x.equal(_out.broadcast(x.shape))
    locs = locs.convert(x_dot.dtype)
    counts = locs.sum(axes)
    jvp_out = (x_dot * locs).sum(axes)
    jvp_out = jvp_out / counts.broadcast(jvp_out.shape)

    return [out], [jvp_out]


@max.set_method
def T(self, cts, x, *, axes=None, keepdims=False):
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
operator_set.alias(random_uniform, "randn")


@random_uniform.set_method
def args_fixer(self, *, shape, dtype=Tensor.float32):
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
def args_fixer(self, *, shape, dtype=Tensor.float32):
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
def args_fixer(self, *, start, stop=None, stride=None, dtype=Tensor.float32):
    if stop is None:
        stop = start
        start = 0
    if stride is None:
        stride = 1
    return (), dict(start=start, stop=stop, stride=stride, dtype=dtype)


@arange.set_method
def jvp(self, primals, tangents, *, start, stop, stride=None, dtype=Tensor.float32):
    out = self(arange, start, stop, stride, dtype)
    out_jvp = slope.ones_like(out)
    return [out], [out_jvp]


@arange.set_method
def T(self, cts, *, start, stop, stride=None, dtype=Tensor.float32):
    return [None]


@arange.set_method
def typecheck(self, *, start, stop, stride=None, dtype=Tensor.float32) -> List[Typecheckor]:
    return [Typecheckor(tuple((stop - start) * stride), dtype)]


# --------------
# Backend
# --------------


compile_py = compile
numpy_backend = Backend(name="numpy", default_dtype=Tensor.float32)
numpy_backend.set_dtype_map(
    {
        Tensor.float32: np.dtype("float32"),
        Tensor.int64: np.dtype("int64"),
        Tensor.int8: np.dtype("int8"),
        Tensor.bool: np.dtype("bool"),
    }
)


@numpy_backend.set_method
def tensor(self, val, dtype=numpy_backend.default_dtype_value):
    val = np.array(val, dtype=numpy_backend.dtype_map[dtype])
    return Tensor(TensorBuffer(val))


@numpy_backend.set_method
def numpy_of(self, tensor):
    return tensor.buf.val


@numpy_backend.set_method
def device_of(self, tensor):
    return "cpu"


@numpy_backend.set_method
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


@numpy_backend.set_method
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
    ncs = 0  # n constant
    nxs = 0  # n x inputs
    nzs = 0  # n z intermediate variables
    nys = 0  # n y outputs
    inb_args = []
    inb_consts = []

    for inb in program.in_binders:
        if type(inb.aval) is not Typecheckor:
            environment[inb] = f"c{ncs}"
            inb_consts += [environment[inb]]
            ncs += 1
        else:
            environment[inb] = f"x{nxs}"
            inb_args += [environment[inb]]
            nxs += 1

    code_lines = []
    fn_args_strs = f""
    if inb_consts:
        fn_args_strs += f"{', '.join(inb_consts)}, "
    fn_args_strs += f"{', '.join(inb_args)}"
    code_lines += [f"def {fn_name}({fn_args_strs}):"]
    for instruction in program.instructions:
        in_vals = list_map(lambda x: environment[x], instruction.inputs)
        for outb in instruction.out_binders:
            if outb in program.outs:
                environment[outb] = f"y{nys}"
                nys += 1
            else:
                environment[outb] = f"z{nzs}"
                nzs += 1

        out_vals = list_map(lambda z: environment[z], instruction.out_binders)
        if len(out_vals) == 0:  # skip codegen for function returns nothing
            continue

        if len(out_vals) == 1:
            lhs = f"{out_vals[0]}"
        else:
            lhs = ", ".join(out_vals)
        if instruction.op.op_type is slope.core.OperatorType.Meta:
            lhs += ", "
            rhs, fn_defs = self.impls[instruction.op](program, args, instruction, in_vals, fn_defs)
        else:
            rhs = self.impls[instruction.op](*in_vals, **instruction.params)
            if "\n" in rhs:  # multi-line impls
                impl_lines = rhs.strip().split("\n")
                lhs = "\n".join(impl_lines[:-1] + [lhs])
                rhs = impl_lines[-1]
            rhs = rhs.replace("return ", "")
        if "(, " in rhs:  # fix syntax error for function call has only keyword-only args
            rhs = rhs.replace("(, ", "(")
        for np_dtype in self.dtype_map.values():  # fix dtype kwargs not having 'np.' prefix
            rhs = rhs.replace(np_dtype.name, f"np.{np_dtype.name}")
        code_line = f"{lhs} = {rhs}"

        for code_line_line in code_line.split("\n"):  # handle multi-line code
            code_lines += [indent(code_line_line, il1)]

    outs = list_map(lambda x: environment[x], program.outs)
    ret_str = f"{', '.join(outs)}{',' if len(outs)==1 else ''}"
    code_lines += [indent(f"return {ret_str}", il1)]
    if fn_name == "main":
        if len(fn_defs) > 0:
            code_lines = (
                code_lines[0:1]
                + [indent(line, il1) for impl_lines in fn_defs.values() for line in impl_lines]
                + code_lines[1:]
            )

        # code_lines = code_lines[0:1] + [indent(f"float32 = np.float32", il1)] + code_lines[1:]

    slope.dblog(f"\n-- {program.name} codegen:\n\n" + "\n".join(code_lines) + "\n\n==\n", enable=slope.LOG_JIT)

    if fn_name == "main":
        del self.fn_count

    return dict(code_lines=code_lines, fn_defs=fn_defs)


### Operator Impls

numpy_backend.set_impl(operator_set.convert)(lambda self, x, *, dtype: f"return {x}.astype(dtype={dtype})")
numpy_backend.set_impl(operator_set.stop_gradient)(lambda self, x, *, dtype: f"return {x}")
numpy_backend.set_impl(operator_set.neg)(lambda self, x: f"return np.negative({x})")
numpy_backend.set_impl(operator_set.sqrt)(lambda self, x: f"return np.sqrt({x})")
numpy_backend.set_impl(operator_set.exp)(lambda self, x: f"return np.exp({x})")
numpy_backend.set_impl(operator_set.log)(lambda self, x: f"return np.log({x})")
numpy_backend.set_impl(operator_set.sin)(lambda self, x: f"return np.sin({x})")
numpy_backend.set_impl(operator_set.add)(lambda self, x1, x2: f"return np.add({x1}, {x2})")
numpy_backend.set_impl(operator_set.sub)(lambda self, x1, x2: f"return np.subtract({x1}, {x2})")
numpy_backend.set_impl(operator_set.mul)(lambda self, x1, x2: f"return np.multiply({x1}, {x2})")
numpy_backend.set_impl(operator_set.div)(lambda self, x1, x2: f"return np.divide({x1}, {x2})")
numpy_backend.set_impl(operator_set.invert)(lambda self, x: f"return np.invert({x})")
numpy_backend.set_impl(operator_set.equal)(lambda self, x1, x2: f"return np.equal({x1}, {x2})")
numpy_backend.set_impl(operator_set.maximum)(lambda self, x1, x2: f"return np.maximum({x1}, {x2})")
numpy_backend.set_impl(operator_set.sum)(
    lambda self, x, *, axes, keepdims: f"return np.sum({x}, axis={axes}, keepdims={keepdims})"
)
numpy_backend.set_impl(operator_set.max)(
    lambda self, x, *, axes, keepdims: f"return np.max({x}, axis={axes}, keepdims={keepdims})"
)
numpy_backend.set_impl(operator_set.arange)(
    lambda self, *, start, stop, stride, dtype: f"return np.arange(start={start}, stop={stop}, stride={stride}, dtype={dtype})"
)
numpy_backend.set_impl(operator_set.full)(
    lambda self, *, shape, fill_value, dtype: f"return np.full(shape={shape}, fill_value={fill_value}, dtype={dtype})"
)

numpy_backend.set_impl(operator_set.random_uniform)(
    lambda self, *, shape, dtype: f"return np.random.uniform(size={shape}).astype(dtype={dtype})"
)
numpy_backend.set_impl(operator_set.random_normal)(
    lambda self, *, shape, dtype: f"return np.random.normal(loc=np.zeros(shape={shape})).astype(dtype={dtype})"
)
numpy_backend.set_impl(operator_set.broadcast_to)(
    lambda self, x, *, shape: f"return np.broadcast_to({x}, shape={shape})"
)

numpy_backend.set_impl(operator_set.reshape)(lambda self, x, *, shape: f"return np.reshape({x}, newshape={shape})")
numpy_backend.set_impl(operator_set.pad_hlo)(  # TODO: interior not used
    lambda self, x, *, lo, hi, interior, value: f"return np.pad({x}, list(zip({lo}, {hi})), constant_values={value})"
)


numpy_backend.set_impl(operator_set.slice_hlo)(
    lambda self, x, *, starts, limits, strides: f"return {x}[tuple(slice(s, l, st) for s, l, st in zip({starts}, {limits}, {strides}))]"
)

numpy_backend.set_impl(operator_set.concatenate)(lambda self, *xs, axis: f"return np.concatenate({xs}, axis={axis})")
numpy_backend.set_impl(operator_set.transpose)(lambda self, x, *, perm: f"return np.transpose({x}, axes={perm})")
numpy_backend.set_impl(operator_set.flip)(lambda self, x, *, axes: f"return np.flip({x}, axis={axes})")


@numpy_backend.set_impl(slope.core.jit_op)
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


@numpy_backend.set_impl(slope.core.procedure_op)
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
    return slope.full(shape, 0.0, dtype)


@procedure_set.register(static_argnames="shape dtype")
def ones(shape, dtype=Tensor.float32):
    return slope.full(shape=shape, fill_value=1.0, dtype=dtype)


@procedure_set.register(static_argnames="fill_value")
def full_like(y, fill_value):
    return slope.full(shape=y.shape, fill_value=fill_value, dtype=y.dtype)


@procedure_set.register()
def zeros_like(y):
    return zeros(shape=y.shape, dtype=y.dtype)


@procedure_set.register()
def ones_like(y):
    return slope.full(shape=y.shape, fill_value=1.0, dtype=y.dtype)


@procedure_set.register()
def relu(x):
    return x.maximum(slope.zeros_like(x))


@procedure_set.register()
def where(x, trueval, falseval):
    cond = x != 0.0
    cond = cond.convert(trueval.dtype)
    return cond * trueval + (1.0 - cond) * falseval


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
    # return x.sin()


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
    return self.convert(slope.int32).convert(self.dtype)


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
def _softmax(x, axes):
    m = x  # - x.max(axes, keepdims=True) # BUG: enable this error in typecheck program
    e = m.exp()
    return m, e, e.sum(axes, keepdims=True)


@procedure_set.register(static_argnames="axes")
def softmax(x, axes=-1):
    _, e, ss = x._softmax(axes)
    return e.div(ss)


@procedure_set.register(static_argnames="axes")
def log_softmax(x, axes=-1):
    m, _, ss = x._softmax(axes)
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
    ret = sliced_tensor.reshape(tuple(final_shape))

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
        ret = ret.reshape(*ret.shape[: sum_dim[0] + 1], *[1] * max_dim, *ret.shape[sum_dim[0] + 1 :])
        # iteratively fancy index
        for a, i, sd in zip(slice_arange, idx, sum_dim):
            ret = (a == i).mul(ret).sum(sd)
        # special permute case
        if dim[0] != 0 and len(dim) != 1 and dim != list(range(dim[0], dim[-1] + 1)):
            ret_dims = list(range(ret.ndim))
            ret = ret.transpose(ret_dims[dim[0] : dim[0] + max_dim] + ret_dims[: dim[0]] + ret_dims[dim[0] + max_dim :])
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


@procedure_set.register(static_argnames="groups stride dilation padding output_padding")
def conv_transpose(x, weight, groups=1, stride=1, dilation=1, padding=0, output_padding=0):
    HW, trailing = weight.shape[2:], list(range(3, len(weight.shape) + 1))
    x, w = x, weight.reshape(groups, weight.shape[0] // groups, weight.shape[1], *weight.shape[2:]).transpose(
        0, 2, 1, *trailing
    ).flip(trailing)
    stride = make_pair(stride, len(HW))
    if any(s > 1 for s in stride):
        x = x.reshape(*x.shape[:2], *flatten_seq((k, 1) for k in x.shape[2:]))
        x = x.pad(((0, 0), (0, 0), *flatten_seq(((0, 0), (0, s - 1)) for s in stride)))
        x = x.reshape(*x.shape[:2], *[k * s for k, s in zip(x.shape[2::2], stride)])
        x = x.slice(
            (
                (0, x.shape[0]),
                (0, x.shape[1]),
                *[(0, k - (s - 1)) for k, s in zip(x.shape[2:], stride)],
            )
        )
    padding = flatten_seq(
        (
            ((k - 1) * d - p, (k - 1) * d - p + op)
            for k, d, p, op in reversed(
                list(
                    zip(
                        HW,
                        make_pair(dilation, len(HW)),
                        make_pair(padding, len(HW)),
                        make_pair(output_padding, len(HW)),
                    )
                )
            )
        )
    )
    return x.conv(
        w.reshape(w.shape[0] * w.shape[1], *w.shape[2:]),
        groups=groups,
        dilation=dilation,
        padding=padding,
    )


@procedure_set.register(static_argnames="groups stride dilation padding")
def conv(x, weight, groups=1, stride=1, dilation=1, padding=0):
    (bs, cin_), (cout, cin), HW = x.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups * cin == cin_ and len(x.shape) == len(
        weight.shape
    ), f"Input axis shape {x.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(HW) or len(padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {x.shape}"
    padding_ = (
        [padding] * 2 * len(HW)
        if isinstance(padding, int)
        else (padding if len(padding) == 2 * len(HW) else [p for p in padding for _ in range(2)][::-1])
    )
    padding_ = tuple(padding_)
    x = x.pad(padding_)
    x = x._pool(HW, stride, dilation)  # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout // groups, x.shape[2 : -len(HW)]
    x = x.reshape((bs, groups, cin, 1, *oyx, *HW))
    x = x.broadcast_to((bs, groups, cin, rcout, *oyx, *HW))
    x = x.transpose(
        (
            0,
            1,
            3,
            *[4 + i for i in range(len(oyx))],
            2,
            *[4 + len(oyx) + i for i in range(len(HW))],
        )
    )
    # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
    x = x * weight.reshape((1, groups, rcout, *[1] * len(oyx), cin, *HW))
    x = x.sum([-1 - i for i in range(1 + len(oyx))], keepdims=True)
    x = x.reshape((bs, cout, *oyx))
    ret = x
    return ret


@procedure_set.register(static_argnames="groups stride dilation padding")
def conv_wino(x, weight, groups=1, stride=1, dilation=1, padding=0):
    assert not all(x == 3 for x in HW) or stride != 1 or dilation != 1
    (bs, cin_), (cout, cin), HW = x.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups * cin == cin_ and len(x.shape) == len(
        weight.shape
    ), f"Input axis shape {x.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(HW) or len(padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {x.shape}"
    padding_ = (
        [padding] * 2 * len(HW)
        if isinstance(padding, int)
        else (padding if len(padding) == 2 * len(HW) else [p for p in padding for _ in range(2)][::-1])
    )

    x = x.padslice(padding_)._pool(HW, stride, dilation)  # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout // groups, x.shape[2 : -len(HW)]

    # winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
    def apply_matrix(mat, t, dim=0):
        return (
            t
            if dim == len(HW)
            else stack(
                [
                    apply_matrix(
                        mat,
                        sum(mat[i][j] * t[j] for j in range(len(mat[i])) if mat[i][j]),
                        dim=dim + 1,
                    )
                    for i in range(len(mat))
                ]
            )
        )

    HWI, HWO = (6,) * len(HW), (4,) * len(HW)  # F(4x4,3x3) winograd tiles
    winograd_Bt = [
        [4, 0, -5, 0, 1, 0],
        [0, -4, -4, 1, 1, 0],
        [0, 4, -4, -1, 1, 0],
        [0, -2, -1, 2, 1, 0],
        [0, 2, -1, -2, 1, 0],
        [0, 4, 0, -5, 0, 1],
    ]
    winograd_G = [
        [1 / 4, 0, 0],
        [-1 / 6, -1 / 6, -1 / 6],
        [-1 / 6, 1 / 6, -1 / 6],
        [1 / 24, 1 / 12, 1 / 6],
        [1 / 24, -1 / 12, 1 / 6],
        [0, 0, 1],
    ]
    winograd_At = [
        [1, 1, 1, 1, 1, 0],
        [0, 1, -1, 2, -2, 0],
        [0, 1, 1, 4, 4, 0],
        [0, 1, -1, 8, -8, 1],
    ]  # applying At in pre-order almost doubles compilation time

    # todo: stride == dilation
    # use padding to round up to 4x4 output tiles
    d = x.pad(
        sum(
            [
                [
                    padding_[i * 2],
                    padding_[i * 2 + 1] + (-(dim + sum(padding_[i * 2 : (i + 1) * 2]) - 2) % 4),
                ]
                for i, dim in enumerate(x.shape[-len(HW) :])
            ],
            [],
        )
    )._pool(
        HWI, HWO
    )  # (bs, cin_, tyx, HWI)
    d = d.transpose(
        *range(len(d.shape) - len(HW), len(d.shape)), *range(len(d.shape) - len(HW))
    ).contiguous_backward()  # move HW to the front: # (HWI, bs, cin_, tyx)
    tyx = d.shape[-len(HWI) :]  # dim of tiling

    g = weight.transpose(
        *range(len(weight.shape) - len(HW), len(weight.shape)),
        *range(len(weight.shape) - len(HW)),
    )  # move HW to the front

    # compute 6x6 winograd tiles: GgGt, BtdB
    gfactors = (
        apply_matrix(winograd_G, g).contiguous().reshape(*HWI, 1, groups, rcout, cin, *([1] * len(tyx)))
    )  # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    dfactors = (
        apply_matrix(winograd_Bt, d).contiguous().reshape(*HWI, bs, groups, 1, cin, *tyx)
    )  # (HWI, bs, cin_, tyx) -> (HWI, bs, groups, 1 ,cin, *tyx)

    ret = apply_matrix(
        winograd_At, (gfactors * dfactors).sum(axis=-1 - len(HW))
    )  # matmul; sum across cin: (HWI, bs, groups, rcout, *tyx); then HWI -> HWO: (HWO, bs, groups, rcout, *tyx)

    ret = ret.transpose(
        [
            *range(len(HW), len(ret.shape) - len(HW)),
            *[i + o for i in range(len(HW)) for o in [len(ret.shape) - len(HW), 0]],
        ]
    )  # interleave tyx and HWO: (bs, groups, rcout, oy, HO, ox, WO)
    ret = ret.reshape(bs, cout, *[c * HWO[i] for i, c in enumerate(tyx)]).slice(
        tuple((0, s) for s in [bs, cout, *oyx])
    )  # merge groups and rcout, tyx and HWO: (bs, groups, cout, *yx), shrink to final

    return ret


@procedure_set.register(static_argnames="axis")
def cumsum(x, axis: int = 0):
    return x.swapaxes(axis, -1).pad((x.shape[axis] - 1, 0))._pool((x.shape[axis],)).sum(-1).swapaxes(axis, -1)


v1_environment = Environment(operator_set, procedure_set, numpy_backend)
