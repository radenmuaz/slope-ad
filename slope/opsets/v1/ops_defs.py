import slope as sp
from slope.core import (
    Op,
    OpsDir,
    BaseArray,
    VoidArray,
    UndefPrimal,
    list_zip,
    list_map,
)

# from slope import Op, VoidArray, BaseArray
import math
import numpy as np
from typing import (
    Tuple,
    List,
    Any,
    Optional,
    Sequence,
)

ops = OpsDir()

# -----------------------
# UnaryOps
# -----------------------

# TODO: in eval_prog_transposed, try skip eval stop_gradient Op
stop_gradient = Op.unary("stop_gradient")
ops.register(stop_gradient)


@stop_gradient.set_eval
def f(self, x):
    return [x]


@stop_gradient.set_jvp
def f(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x], [sp.rt.zeros_like(x_dot)]


@stop_gradient.set_T
def f(self, cts, x):
    (z,) = cts
    assert type(x) is UndefPrimal
    return [sp.zeros_like(z)]


convert = Op.unary("convert")
astype = convert
ops.register(convert)
ops.alias(convert, "astype")


@convert.set_eval
def f(self, x, *, dtype):
    return [x.convert(dtype)]


@convert.set_jvp
def f(self, primals, tangents, *, dtype):
    (x,), (x_dot,) = primals, tangents
    return [x.convert(dtype)], [x_dot.convert(dtype)]


sqrt = Op.unary("sqrt")
ops.register(sqrt)


@sqrt.set_eval
def f(self, x):
    return [x.sqrt()]


@sqrt.set_jvp
def f(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.sqrt()
    # return [ans], [x_dot * (0.5 / ans)]
    return [ans], [x_dot / (ans * 2)]


@sqrt.set_T
def f(self, cts, x):
    (z,) = cts
    return [z / (x.sqrt() * 2)]


sin = Op.unary("sin")
ops.register(sin)


@sin.set_eval
def f(self, x):
    return [x.sin()]


@sin.set_jvp
def f(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.sin()
    return [ans], [(math.pi / 2) - (x_dot * ans)]


@sin.set_T
def f(self, cts, x):
    (z,) = cts
    return [(math.pi / 2) - (z * x.sin())]


exp = Op.unary("exp")
ops.register(exp)


@exp.set_eval
def f(self, x):
    return [x.exp()]


@exp.set_jvp
def f(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.exp()
    return [ans], [x_dot * ans]


@exp.set_T
def f(self, cts, x):
    (z,) = cts
    return [1 / z]


log = Op.unary("log")
ops.register(log)


@log.set_eval
def f(self, x):
    return [x.log()]


@log.set_jvp
def f(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.log()
    return [ans], [x_dot / x]


@log.set_T
def f(self, cts, x):
    (z,) = cts
    return [1 / z]


neg = Op.unary("neg")
ops.register(neg)


@neg.set_eval
def f(self, x):
    return [-x]


@neg.set_jvp
def f(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [-x], [-x_dot]


@neg.set_T
def f(self, cts, x):
    (z,) = cts
    return [-z]


relu = Op.unary("relu")
ops.register(relu)


@relu.set_eval
def f(self, x):
    return [x.maximum(0)]


@relu.set_jvp
def f(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x.maximum(0)], [-x_dot.maximum(0)]


@relu.set_T
def f(self, cts, x):
    (z,) = cts
    mask = 1 - (x.maximum(0) == 0)
    return [mask * z]


# -----------------------
# BinaryOps
# -----------------------


add = Op.binary("add")
ops.register(add)


@add.set_eval
def f(self, x, y):
    return [x + y]


@add.set_jvp
def f(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x + y], [x_dot + y_dot]


@add.set_T
def f(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar, z_bar]


sub = Op.binary("sub")
ops.register(sub)


@sub.set_eval
def f(self, x, y):
    return [x - y]


@sub.set_jvp
def f(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x - y], [x_dot - y_dot]


@sub.set_T
def f(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar, -z_bar]


mul = Op.binary("mul")
ops.register(mul)


@mul.set_eval
def f(self, x, y):
    return [x * y]


@mul.set_jvp
def f(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x * y], [(x_dot * y) + (y_dot * x)]
    # jvp_out = (y * x_dot) + (y_dot * x) # order problem, x*y_dot fails


@mul.set_T
def f(self, cts, x, y):
    (z_bar,) = cts
    assert (type(x) is UndefPrimal) ^ (type(y) is UndefPrimal)
    if type(x) is UndefPrimal:
        return [z_bar * y, None]
    elif type(y) is UndefPrimal:
        return [None, x * z_bar]


div = Op.binary("div")
ops.register(div)


@div.set_eval
def f(self, x, y):
    return [x / y]


@div.set_jvp
def f(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x / y], [
        (x_dot / y) + (-y_dot * x * (y**-2))
    ]  # bug: power returns float64


@div.set_T
def f(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar / y, None]


maximum = Op.binary("maximum")
ops.register(maximum)


@maximum.set_eval
def f(self, x, y):
    return [x.maximum(y)]


@maximum.set_jvp
def f(self, primals, tangents):
    def _balanced_eq(x, z, y):
        return (
            (x == z).where(self.rt.procs.ones_like(z), self.rt.procs.zeros_like(z))
        ) / ((y == z).where(self.rt.procs.full_like(z, 2), self.rt.procs.ones_like(z)))

    (x, y), (x_dot, y_dot) = primals, tangents
    eval_out = x.maximum(y)
    jvp_out = x_dot * _balanced_eq(x, eval_out, y) + y_dot * _balanced_eq(
        y, eval_out, x
    )

    return [eval_out], [jvp_out]


@maximum.set_T
def f(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar, None]


equal = Op.binary("equal")
ops.register(equal)


@equal.set_eval
def f(self, x, y):
    return [x.equal(y)]


@equal.set_jvp
def f(self, primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = x.equal(y)
    return [out_primal], [self.rt.zeros(out_primal.shape, out_primal.dtype)]


@equal.set_T
def f(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar, None]


not_equal = Op.binary("not_equal")
ops.register(not_equal)


@not_equal.set_eval
def f(self, x, y):
    return [x.not_equal(y)]


@not_equal.set_jvp
def f(self, primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = x.not_equal(y)
    return [out_primal], [self.rt.zeros(out_primal.shape, out_primal.dtype)]


@not_equal.set_T
def f(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar, None]


max = Op.reduce("max")
ops.register(max)


@max.set_args_fixer
def f(self, x, *, axes=None, keepdims=None):
    if isinstance(axes, int):
        axes = (axes,)
    elif axes is None:
        axes = tuple(range((x.ndim)))
    else:
        axes = tuple(a if a >= 0 else a + len(x.shape) for a in axes)
    return (x,), dict(axes=axes, keepdims=keepdims)


@max.set_eval
def f(self, x, *, axes=None, keepdims=False):
    return [x.max(axes=axes, keepdims=keepdims)]


@max.set_jvp
def f(self, primals, tangents, *, axes=None, keepdims=False):
    (x,), (x_dot,) = primals, tangents
    eval_out = x.max(axes, keepdims)
    locs = x.equal(eval_out.broadcast(x.shape, None if keepdims else axes))
    locs = locs.convert(x_dot.dtype)
    counts = locs.sum(axes)
    jvp_out = (x_dot * locs).sum(axes)
    jvp_out = jvp_out / counts.broadcast(jvp_out.shape)

    return [eval_out], [jvp_out]


@max.set_T
def f(self, cts, x, *, axes=None, keepdims=False):
    (z,) = cts
    return [z.broadcast(x.aval.shape, None if keepdims else axes)]


sum = Op.reduce("sum")
ops.register(sum)


@sum.set_args_fixer
def f(self, x, *, axes=None, keepdims=False):
    if isinstance(axes, int):
        axes = (axes,)
    elif axes is None:
        axes = tuple(range((x.ndim)))
    else:
        axes = tuple(a if a >= 0 else a + len(x.shape) for a in axes)
    return (x,), dict(axes=axes, keepdims=keepdims)


@sum.set_eval
def f(self, x, *, axes=None, keepdims=False):
    return [x.sum(axes, keepdims)]


@sum.set_jvp
def f(self, primals, tangents, *, axes=None, keepdims=False):
    (x,), (x_dot,) = primals, tangents
    eval_out = x.sum(axes, keepdims)
    jvp_out = x_dot.sum(axes, keepdims)
    return [eval_out], [jvp_out]


@sum.set_T
def f(self, cts, x, *, axes=None, keepdims=False):
    (z,) = cts
    out = z.broadcast(x.aval.shape, None if keepdims else axes)
    return [out]


# -----------------------
# ShapeOps
# -----------------------

broadcast = Op.shape("broadcast")
ops.register(broadcast)


@broadcast.set_args_fixer
def f(self, x, *, shape, axes=None):
    if isinstance(axes, int):
        axes = (axes,)
    elif axes is None:
        axes = ()
    else:
        axes = tuple(a if a >= 0 else a + len(x.shape) for a in axes)
    return (x,), dict(shape=shape, axes=axes)


@broadcast.set_eval
def f(self, x, *, shape, axes=None):
    out = x.broadcast(shape, axes=None)
    return [out]


@broadcast.set_vmap
def f(self, axis_size, vals_in, dims_in, *, shape, axes=None):
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


@broadcast.set_jvp
def f(self, primals, tangents, *, shape, axes=None):
    (x,), (x_dot,) = primals, tangents
    return (
        [x.broadcast(shape=shape, axes=axes)],
        [x_dot.broadcast(shape=shape, axes=axes)],
    )


@broadcast.set_shape_eval
def f(self, x: VoidArray, *, shape: Sequence[int], axes=None) -> List[VoidArray]:
    return [VoidArray(tuple(shape), x.dtype)]


@broadcast.set_T
def f(self, cts, x, *, shape, axes):
    (z,) = cts
    out = z
    if x.aval.shape == z.shape:
        return [out]

    x_ndim = len(x.aval.shape)
    if x_ndim < out.ndim:
        b_axes = []
        for i, dim in enumerate(out.shape):
            if dim not in x.aval.shape:
                b_axes += [i]
        out = out.sum(tuple(b_axes), keepdims=False)

    elif x.aval.shape != out.shape:
        b_axes = []
        for i, (dx, dz) in enumerate(list_zip(x.aval.shape, out.shape)):
            if dz > dx and i not in axes:
                b_axes += [i]
        out = out.sum(axes=tuple(b_axes), keepdims=True)
    if out.shape != x.aval.shape:
        print(f"not same {out.shape=}, {x.aval.shape=}")
        breakpoint()
    return [out]


reshape = Op.shape("reshape")
ops.register(reshape)


@reshape.set_args_fixer
def f(self, x, *, shape):
    if -1 in shape:
        others = math.prod([d for d in shape if d != -1])
        numel = math.prod(x.shape)
        shape = tuple(d if d != -1 else (numel // others) for d in shape)
    return (x,), dict(shape=shape)


@reshape.set_eval
def f(self, x, *, shape):
    return [x.reshape(shape)]


@reshape.set_jvp
def f(self, primals, tangents, *, shape):
    (x,), (x_dot,) = primals, tangents
    return [x.reshape(shape)], [x_dot.reshape(shape)]


@reshape.set_shape_eval
def f(self, x: VoidArray, *, shape: Sequence[int]) -> List[VoidArray]:
    return [VoidArray(tuple(shape), x.dtype)]


@reshape.set_T
def f(self, cts, x, *, shape):
    (z,) = cts
    return [z.reshape(x.aval.shape)]


transpose = Op.shape("transpose")
ops.register(transpose)


@transpose.set_eval
def f(self, x, *, perm):
    return [x.transpose(perm=perm)]


@transpose.set_vmap
def f(self, axis_size, vals_in, dims_in, *, perm):
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


@transpose.set_jvp
def f(self, primals, tangents, *, perm):
    (x,), (x_dot,) = primals, tangents
    return [x.transpose(perm)], [x_dot.transpose(perm)]


@transpose.set_shape_eval
def f(self, x: VoidArray, *, perm: Sequence[int]) -> List[VoidArray]:
    shape = [x.shape[i] for i in perm]
    return [VoidArray(shape, x.dtype)]


@transpose.set_T
def f(self, cts, x, *, perm):
    (z,) = cts
    return [z.transpose(perm)]


pad = Op.shape("pad")
ops.register(pad)


@pad.set_eval
def f(self, x, *, lo, hi, interior=None, value=0.0):
    return [x.pad(lo, hi, interior, value)]


@pad.set_args_fixer
def f(self, x, *, lo, hi, interior=None, value=0.0):
    if interior is None:
        interior = tuple([0] * len(lo))
    return (x,), dict(lo=lo, hi=hi, interior=interior, value=value)


@pad.set_vmap
def f(self, axis_size, vals_in, dims_in, *, pinterior=None, value=0.0):
    raise NotImplementedError
    Operand, padding_value = batched_args
    Operand_bdim, padding_value_bdim = batch_dims
    if Operand_bdim is None:
        Operand_bdim = 0
        Operand = broadcast(operand, (padding_value.shape[padding_value_bdim],))

    padding_config = list(padding_config)
    padding_config.insert(operand_bdim, (0, 0, 0))
    if padding_value_bdim is None:
        return pad(operand, padding_value, padding_config), Operand_bdim

    assert padding_value_bdim == 0, padding_value_bdim

    x = pad(operand, _zero(operand), padding_config)
    mask = pad(full_like(operand, True, np.bool_), False, padding_config)
    broadcasted_padding = broadcast_in_dim(padding_value, x.shape, (operand_bdim,))
    return select(mask, x, broadcasted_padding), Operand_bdim


@pad.set_jvp
def f(self, primals, tangents, *, lo, hi, interior=None, value=0.0):
    (x,), (x_dot,) = primals, tangents
    return [x.pad(lo, hi, interior, value)], [x_dot.pad(lo, hi, interior, value)]


@pad.set_shape_eval
def f(self, x: VoidArray, *, lo, hi, interior=None, value=0.0) -> List[VoidArray]:
    def _dilate_dim(d, dilation):
        return 0 if d == 0 else 1 + dilation * (d - 1)

    shape = (
        sum([l, h, _dilate_dim(d, r + 1)])
        for l, h, r, d in list_zip(lo, hi, interior, x.shape)
    )
    if not all(d >= 0 for d in shape):
        raise ValueError(
            f"Dimension size after padding is not at least 0, "
            f"got result shape {res}, for {lo=} {hi=} {interior=} {value=}"
            f"{shape=}"
        )
    res = VoidArray(shape, x.dtype)
    return [res]


@pad.set_T
def f(self, cts, x, *, lo, hi, interior=None, value=0.0):
    (z,) = cts

    def t_op():
        unpadded = z.slice(
            lo,
            tuple(s - h for s, h in list_zip(z.shape, hi)),
            tuple([1] * len(interior)),
        )
        return unpadded.slice(
            tuple([0] * len(lo)), unpadded.shape, tuple(r + 1 for r in interior)
        )

    res = t_op() if isinstance(x, UndefPrimal) else None
    return [res]


slice = Op.shape("slice")
ops.register(slice)


@slice.set_eval
def f(self, x, *, starts, limits, strides):
    return [x.slice(starts, limits, strides)]


@slice.set_vmap
def f(self, axis_size, vals_in, dims_in, *, starts, limits, strides):
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


@slice.set_jvp
def f(self, primals, tangents, *, starts, limits, strides):
    (x,), (x_dot,) = primals, tangents
    return [x.slice(starts, limits, strides)], [x_dot.slice(starts, limits, strides)]


@slice.set_shape_eval
def f(
    self, x: VoidArray, *, starts, limits, strides: Sequence[int]
) -> List[VoidArray]:
    if strides is None or tuple(strides) == (1,) * len(x.shape):
        shape = [
            limit if type(start) is int and start == 0 else limit - start
            for start, limit in list_zip(starts, limits)
        ]
        return [VoidArray(shape, x.dtype)]
    else:
        # TODO: compute strided shape without numpy
        x = np.zeros_like(x.shape)
        x = x[tuple(slice(s, l, r) for s, l, r in list_zip(starts, limits, strides))]
        return [VoidArray(x.shape, x.dtype)]


@slice.set_T
def T(cts, x, *, starts, limits, strides):
    # TODO: compute tuple arithmetic without numpy
    (z,) = cts
    x_shape = x.aval.shape
    assert isinstance(x, UndefPrimal)
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
        lo, hi, interior = list_zip(
            starts, np.subtract(x_shape, real_limits), np.subtract(strides, 1)
        )
    res = z.pad(lo, hi, interior)
    assert res.shape == x_shape, f"{res.shape=} {x_shape=}"
    return [res]


flip = Op.shape("flip")
ops.register(flip)


@flip.set_eval
def f(self, x, *, axes):
    return [x.flip(axes)]


@flip.set_vmap
def f(self, axis_size, vals_in, dims_in, *, axes):
    raise NotImplementedError


@flip.set_jvp
def f(self, primals, tangents, *, axes):
    (x,), (x_dot,) = primals, tangents
    return [x.flip(axes)], [x_dot.flip(axes)]


@flip.set_shape_eval
def f(self, x: VoidArray, *, axes):
    return [VoidArray(x.shape, x.dtype)]


@flip.set_T
def T(cts, *, axes):
    (z,) = cts
    return [z.flip(axes)]


concatenate = Op.shape("concatenate")
cat = concatenate
ops.register(concatenate)
ops.alias(concatenate, "cat")


@concatenate.set_eval
def f(self, xs: Sequence[Any], *, axis):
    return [backend.run_impl(concatenate, xs, axis=axis)]


@concatenate.set_vmap
def f(self, axis_size, vals_in, dims_in, *, axis):
    raise NotImplementedError


@concatenate.set_jvp
def jvp(primals, tangents, *, axis):
    (xs,), (xs_dot,) = primals, tangents
    return [concatenate(xs, axis=axis)], [concatenate(xs_dot, axis=axis)]


@concatenate.set_shape_eval
def f(self, xs: VoidArray, *, axis: Sequence[int]) -> List[VoidArray]:
    if not xs:
        msg = "concatenate expects at least one Operand, got 0."
        raise TypeError(msg)
    if len(set(operand.ndim for Operand in xs)) != 1:
        msg = "Cannot concatenate arrays with different numbers of dimensions: got {}."
        raise TypeError(msg.format(", ".join(str(o.shape) for o in xs)))
    if not 0 <= axis < xs[0].ndim:
        msg = "concatenate dimension out of bounds: dimension {} for shapes {}."
        raise TypeError(msg.format(axis, ", ".join([str(o.shape) for o in xs])))
    shapes = [x.shape[:axis] + x.shape[axis + 1 :] for x in xs]
    if not shapes[:-1] == shapes[1:]:
        msg = (
            "Cannot concatenate arrays with shapes that differ in dimensions "
            "other than the one being concatenated: concatenating along "
            "dimension {} for shapes {}."
        )
        shapes = [x.shape for x in xs]
        raise TypeError(msg.format(axis, ", ".join(map(str, shapes))))

    concat_size = sum(x.shape[axis] for x in xs)
    ex_shape = xs[0].shape
    return [
        VoidArray(ex_shape[:axis] + (concat_size,) + ex_shape[axis + 1 :], xs[0].dtype)
    ]


@concatenate.set_T
def T(cts, xs, *, axis):
    (z,) = cts
    x_shapes = [o.aval.shape if type(o) is UndefPrimal else o.shape for o in xs]
    if type(z) is None:
        return [None if type(o) is UndefPrimal else None for o in xs]
    else:  # TODO: replace numpy Ops with pure Python
        limit_points = np.cumsum([shape[axis] for shape in x_shapes]).tolist()
        starts = np.zeros((len(xs), z.ndim), dtype=int).tolist()
        limits = np.tile(z.shape, (len(xs), 1)).tolist()

    for i, s in enumerate(starts[1:]):
        s[axis] = limit_points[:-1][i]
    for i, l in enumerate(limits):
        l[axis] = limit_points[i]

    return [
        z.slice(start, limit) if type(o) is UndefPrimal else None
        for o, start, limit in zip(xs, starts, limits)
    ]


# -----------------------
# LoadOps
# -----------------------

constant = Op.load("constant")
ops.register(constant)


@constant.set_eval
def f(self, *, val, dtype=BaseArray.default_dtype):
    return [self.rt.array(val, dtype)]


@constant.set_jvp
def f(self, primals, tangents, *, val, dtype=BaseArray.default_dtype):
    out = sp.rt.array(val, dtype)
    out_jvp = sp.rt.ones_like(out)
    return [out], [out_jvp]


@constant.set_T
def f(self, cts, *, val, dtype=BaseArray.default_dtype):
    return [cts[0]]


@constant.set_shape_eval
def f(self, *, val, dtype=BaseArray.default_dtype):
    # TODO: not using numpy to extract shape
    return [VoidArray(np.array(val).shape, dtype)]


full = Op.load("full")
ops.register(full)


@full.set_eval
def f(self, *, shape, fill_value, dtype=BaseArray.default_dtype):
    return [
        self.rt.backend.run_impl(self, shape=shape, fill_value=fill_value, dtype=dtype)
    ]


@full.set_jvp
def f(self, primals, tangents, *, shape, fill_value, dtype=BaseArray.default_dtype):
    out = self.rt.backend.run_impl(
        self, shape=shape, fill_value=fill_value, dtype=dtype
    )
    out_jvp = self.rt.ones_like(out)
    return [out], [out_jvp]


@full.set_T
def f(self, cts, *, shape, fill_value, dtype=BaseArray.default_dtype):
    return [cts[0]]


@full.set_shape_eval
def f(self, *, shape, fill_value, dtype=BaseArray.default_dtype) -> List[VoidArray]:
    return [VoidArray(tuple(shape), dtype)]


random_uniform = Op.load("random_uniform")
rand = random_uniform
ops.register(random_uniform)
ops.alias(random_uniform, "randn")


@random_uniform.set_eval
def f(self, *, shape, dtype=BaseArray.default_dtype):
    return [self.rt.backend.run_impl(self, shape=shape, dtype=dtype)]


@random_uniform.set_jvp
def f(self, primals, tangents, *, shape, dtype=BaseArray.default_dtype):
    out = self.rt.backend.run_impl(self, shape=shape, dtype=dtype)
    out_jvp = sp.rt.ones_like(out)
    return [out], [out_jvp]


@random_uniform.set_T
def f(self, cts, *, shape, dtype=BaseArray.default_dtype):
    return [cts[0]]


@random_uniform.set_shape_eval
def f(self, *, shape, dtype=BaseArray.default_dtype) -> List[VoidArray]:
    return [VoidArray(tuple(shape), dtype)]


random_normal = Op.load("random_normal")
randn = random_normal
ops.register(random_normal)
ops.alias(random_normal, "randn")


@random_normal.set_eval
def f(self, *, shape, dtype=BaseArray.default_dtype):
    return [self.rt.backend.run_impl(random_normal, shape=shape, dtype=dtype)]


@random_normal.set_jvp
def f(self, primals, tangents, *, shape, dtype=BaseArray.default_dtype):
    out = self.rt.backend.run_impl(random_normal, shape, dtype)
    out_jvp = sp.rt.ones_like(out)
    return [out], [out_jvp]


@random_normal.set_T
def f(self, cts, *, shape, dtype=BaseArray.default_dtype):
    return [cts[0]]


@random_normal.set_shape_eval
def f(self, *, shape, dtype=BaseArray.default_dtype) -> List[VoidArray]:
    return [VoidArray(tuple(shape), dtype)]


arange = Op.load("arange")
ops.register(arange)


@arange.set_args_fixer
def f(self, *, start, stop=None, stride=None, dtype=BaseArray.default_dtype):
    if stop is None:
        stop = start
        start = 0
    if stride is None:
        stride = 1
    return (), dict(start=start, stop=stop, stride=stride, dtype=dtype)


@arange.set_eval
def f(self, *, start, stop, stride=None, dtype=BaseArray.default_dtype):
    return [self.rt.backend.run_impl(arange, start, stop, stride, dtype)]


@arange.set_jvp
def f(
    self, primals, tangents, *, start, stop, stride=None, dtype=BaseArray.default_dtype
):
    out = self.rt.backend.run_impl(arange, start, stop, stride, dtype)
    out_jvp = sp.rt.ones_like(out)
    return [out], [out_jvp]


@arange.set_T
def f(self, cts, *, start, stop, stride=None, dtype=BaseArray.default_dtype):
    return [cts[0]]


@arange.set_shape_eval
def f(
    self, *, start, stop, stride=None, dtype=BaseArray.default_dtype
) -> List[VoidArray]:
    return [VoidArray(tuple((stop - start) * stride), dtype)]
