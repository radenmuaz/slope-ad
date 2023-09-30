import slope
from slope.core import (
    Operator,
    OperatorSet,
    BaseArray,
    VoidArray,
    UndefPrimal,
    list_zip,
    list_map,
)

# from slope import Operator, VoidArray, BaseArray
import math
import numpy as np
from typing import (
    Tuple,
    List,
    Any,
    Optional,
    Sequence,
)
import inspect
sum_py = sum

operator_set = OperatorSet()

# -----------------------
# UnaryOps
# -----------------------

# TODO: in run_program_transposed, try skip run stop_gradient Operator
stop_gradient = Operator.unary("stop_gradient")
operator_set.register(stop_gradient)


@stop_gradient.set_method
def jvp(self, primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x], [slope.M().zeros_like(x_dot)]


@stop_gradient.set_method
def T(self, cts, x):
    (z,) = cts
    assert type(x) is UndefPrimal
    return [slope.environment.zeros_like(z)]


convert = Operator.unary("convert")
astype = convert
operator_set.register(convert)
operator_set.alias(convert, "astype")


@convert.set_method
def void_run(self, x: VoidArray, *, dtype) -> List[VoidArray]:
    return [VoidArray(x.shape, dtype)]


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
    # return [ans], [x_dot * (0.5 / ans)]
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
    ans = x.log()
    return [ans], [x_dot / x]


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


# relu = Operator.unary("relu")
# operator_set.register(relu)


# # @relu.set_run
# # def f(self, x):
# #     breakpoint()
# #     return [x.maximum(0)]


# # @relu.set_jvp
# # def f(self, primals, tangents, **params):
# #     (x,), (x_dot,) = primals, tangents
# #     return [x.maximum(0)], [x_dot.maximum(0)]


# @relu.set_run
# def f(self, x):
#     breakpoint()
#     return [x.maximum(0)]


# @relu.set_jvp
# def f(self, primals, tangents, **params):
#     (x,), (x_dot,) = primals, tangents
#     def sign(z):
#         def abs(z):
#             return z.maximum(0) + (-z).maximum(0)
#         breakpoint()
#         return z * (abs(z) + 1e-10)
#     return [x.maximum(0)], [x.maximum(0)]
#     # return [x.maximum(0)], [sign(x_dot)]
#     # return [x.maximum(0)], [ x_dot.maximum(0) + (-x_dot).maximum(0)]
#     # return [x.relu()], [x_dot.relu()]


# @relu.set_T
# def f(self, cts, x):
#     breakpoint()
#     (z,) = cts
#     return [-z]

# -----------------------
# BinaryOps
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
    # return [y * x], [(y * x_dot) + (y_dot * x)]
    # jvp_out = (y * x_dot) + (y_dot * x) # order problem, x*y_dot fails


@mul.set_method
def T(self, cts, x, y):
    (z_bar,) = cts
    assert (type(x) is UndefPrimal) ^ (type(y) is UndefPrimal)
    if type(x) is UndefPrimal:
        return [z_bar * y, None]
    elif type(y) is UndefPrimal:
        return [None, x * z_bar]


div = Operator.binary("div")
operator_set.register(div)


@div.set_method
def jvp(self, primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x / y], [(x_dot / y) + (-y_dot * x * 1/(y*y))]
    # return [x / y], [(x_dot / y) + (-y_dot * x * (y**-2))]


@div.set_method
def T(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar / y, None]


maximum = Operator.binary("maximum")
operator_set.register(maximum)


@maximum.set_method
def jvp(self, primals, tangents):
    def _balanced_eq(x, z, y):
        return ((x == z).where(slope.environment.ones_like(z), slope.environment.zeros_like(z))) / (
            (y == z).where(slope.environment.full_like(z, 2), slope.environment.ones_like(z))
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
    return [out_primal], [slope.environment.zeros(out_primal.shape, out_primal.dtype)]


@equal.set_method
def T(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar, None]


not_equal = Operator.binary("not_equal")
operator_set.register(not_equal)


@not_equal.set_method
def jvp(self, primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = x.not_equal(y)
    return [out_primal], [slope.environment.zeros(out_primal.shape, out_primal.dtype)]


@not_equal.set_method
def T(self, cts, x, y):
    (z_bar,) = cts
    return [z_bar, None]


max = Operator.reduce("max")
operator_set.register(max)



@max.set_method
def jvp(self, primals, tangents, *, axes=(), keepdims=False):
    (x,), (x_dot,) = primals, tangents
    run_out = x.max(axes, keepdims)
    locs = x.equal(run_out.broadcast_in_dim(x.shape, () if keepdims else axes))
    locs = locs.convert(x_dot.dtype)
    counts = locs.sum(axes)
    jvp_out = (x_dot * locs).sum(axes)
    jvp_out = jvp_out / counts.broadcast_in_dim(jvp_out.shape)

    return [run_out], [jvp_out]


@max.set_method
def T(self, cts, x, *, axes=None, keepdims=False):
    (z,) = cts
    return [z.broadcast_in_dim(x.aval.shape, () if keepdims else axes)]


sum = Operator.reduce("sum")
operator_set.register(sum)


@sum.set_method
def jvp(self, primals, tangents, *, axes=(), keepdims=False):
    (x,), (x_dot,) = primals, tangents
    run_out = x.sum(axes, keepdims)
    jvp_out = x_dot.sum(axes, keepdims)
    return [run_out], [jvp_out]


@sum.set_method
def T(self, cts, x, *, axes=None, keepdims=False):
    (z,) = cts
    out = z.broadcast_in_dim(x.aval.shape, () if keepdims else axes)
    return [out]


# -----------------------
# ShapeOps
# -----------------------

broadcast_in_dim = Operator.shape("broadcast_in_dim")
operator_set.register(broadcast_in_dim)


@broadcast_in_dim.set_method
def args_fixer(self, x, *, shape, axes=None):
    if isinstance(axes, int):
        axes = (axes,)
    elif axes is None:
        axes = ()
    else:
        axes = tuple(a if a >= 0 else a + len(x.shape) for a in axes)
    return (x,), dict(shape=shape, axes=axes)


@broadcast_in_dim.set_method
def vmap(self, axis_size, vals_in, dims_in, *, shape, axes=None):
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

    return [x.broadcast_in_dim(shape, axes)], [x_bdim]


@broadcast_in_dim.set_method
def jvp(self, primals, tangents, *, shape, axes=None):
    (x,), (x_dot,) = primals, tangents
    return (
        [x.broadcast_in_dim(shape=shape, axes=axes)],
        [x_dot.broadcast_in_dim(shape=shape, axes=axes)],
    )


@broadcast_in_dim.set_method
def void_run(self, x: VoidArray, *, shape: Sequence[int], axes=()) -> List[VoidArray]:
    e_shape = list(x.shape)
    for a in axes:
        e_shape.insert(a, 1)
    assert len(e_shape) == len(shape)
    assert all(a <= b for a, b in zip(e_shape, shape))
    return [VoidArray(tuple(shape), x.dtype)]


@broadcast_in_dim.set_method
def T(self, cts, x, *, shape, axes):
    (z,) = cts
    out = z
    if x.aval.shape == z.shape:
        return [out]

    if len(x.aval.shape) < out.ndim:
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
def void_run(self, x: VoidArray, *, shape: Sequence[int]) -> List[VoidArray]:
    return [VoidArray(tuple(shape), x.dtype)]


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
    # perm = [d - int(i >= x_bdim) for i, d in enumerate(perm)]
    perm = perm[:x_bdim] + [x_bdim] + perm[x_bdim:]
    perm = [d + int(d >= x_bdim) if i != x_bdim else d for i, d in enumerate(perm)]
    assert len(set(perm)) == len(perm)
    # perm[:x_bdim] = perm[:x_bdim][::-1]
    # breakpoint()
    return [x.tranpose(perm)], [x_bdim]


@transpose.set_method
def jvp(self, primals, tangents, *, perm):
    (x,), (x_dot,) = primals, tangents
    return [x.transpose(perm)], [x_dot.transpose(perm)]


@transpose.set_method
def void_run(self, x: VoidArray, *, perm: Sequence[int]) -> List[VoidArray]:
    shape = [x.shape[i] for i in perm]
    return [VoidArray(shape, x.dtype)]


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
def void_run(self, x: VoidArray, *, lo, hi, interior=None, value=0.0) -> List[VoidArray]:
    def _dilate_dim(d, dilation):
        return 0 if d == 0 else 1 + dilation * (d - 1)

    shape = tuple(sum_py([l, h, _dilate_dim(d, r + 1)]) for l, h, r, d in list_zip(lo, hi, interior, x.shape))
    if not all(d >= 0 for d in shape):
        raise ValueError(
            f"Dimension size after padding is not at least 0, "
            f"got result shape {res}, for {lo=} {hi=} {interior=} {value=}"
            f"{shape=}"
        )
    res = VoidArray(shape, x.dtype)
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

    res = t_op() if isinstance(x, UndefPrimal) else None
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
def void_run(self, x: VoidArray, *, starts, limits, strides=None) -> List[VoidArray]:
    if strides is None or tuple(strides) == (1,) * len(x.shape):
        shape = [
            limit if type(start) is int and start == 0 else limit - start for start, limit in list_zip(starts, limits)
        ]
        return [VoidArray(shape, x.dtype)]
    else:
        # TODO: compute strided shape without numpy
        x = np.zeros_like(x.shape)
        x = x[tuple(slice_hlo(s, l, r) for s, l, r in list_zip(starts, limits, strides))]
        return [VoidArray(x.shape, x.dtype)]


@slice_hlo.set_method
def T(cts, x, *, starts, limits, strides=None):
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
        lo, hi, interior = list_zip(starts, np.subtract(x_shape, real_limits), np.subtract(strides, 1))
    res = z.pad(lo, hi, interior)
    assert res.shape == x_shape, f"{res.shape=} {x_shape=}"
    return [res]


flip = Operator.shape("flip")
operator_set.register(flip)


@flip.set_method
def vmap(self, axis_size, vals_in, dims_in, *, axes):
    raise NotImplementedError


@flip.set_method
def jvp(self, primals, tangents, *, axes):
    (x,), (x_dot,) = primals, tangents
    return [x.flip(axes)], [x_dot.flip(axes)]


@flip.set_method
def void_run(self, x: VoidArray, *, axes):
    return [VoidArray(x.shape, x.dtype)]


@flip.set_method
def T(cts, *, axes):
    (z,) = cts
    return [z.flip(axes)]


concatenate = Operator.shape("concatenate", variadic_inputs=True)
operator_set.register(concatenate)
operator_set.alias(concatenate, "cat")

@concatenate.set_method
def args_fixer(self, xs, axis=0):
    return *xs, dict(axis=axis)
@concatenate.set_method

def reorg_args(self, args, params):
    args_, params_ = args, params
    sig = inspect.signature(self.void_run)
    args_strs = [
        k for k, v in sig.parameters.items() if v.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and k != "self"
    ]
    params_strs = [k for k, v in sig.parameters.items() if v.kind == inspect.Parameter.KEYWORD_ONLY and k != "self"]

    if args:
        if len(args) > len(args_strs):
            args, rest = args[: len(args_strs)], args[len(args_strs) :]
            if params_strs:
                new_params = {k: rest_arg for k, rest_arg in zip(params_strs, rest) if k not in params}
                params = {**new_params, **params}
        else:
            args = [params[k] if k in params else arg for k, arg in zip(args_strs, args)]
            assert len(args) == len(args_strs)
    return args, params



@concatenate.set_method
def vmap(self, axis_size, vals_in, dims_in, *, axis=0):
    raise NotImplementedError


@concatenate.set_method
def jvp(self, primals, tangents, *, axis=0):
    return [concatenate(primals, axis=axis)], [concatenate(tangents, axis=axis)]


@concatenate.set_method
def void_run(self, xs: VoidArray, *, axis=0) -> List[VoidArray]:
    if not xs:
        msg = "concatenate expects at least one Operand, got 0."
        raise TypeError(msg)
    if len(set(x.ndim for x in xs)) != 1:
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
    return [VoidArray(ex_shape[:axis] + (concat_size,) + ex_shape[axis + 1 :], xs[0].dtype)]


@concatenate.set_method
def T(cts, xs, *, axis=0):
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
        z.slice_hlo(start, limit) if type(o) is UndefPrimal else None for o, start, limit in zip(xs, starts, limits)
    ]


# -----------------------
# LoadOps
# -----------------------

constant = Operator.load("constant")
operator_set.register(constant)


@constant.set_method
def jvp(self, primals, tangents, *, val, dtype=BaseArray.float32):
    out = slope.environment.array(val, dtype)
    out_jvp = slope.environment.ones_like(out)
    return [out], [out_jvp]


@constant.set_method
def T(self, cts, *, val, dtype=BaseArray.float32):
    return [cts[0]]


@constant.set_method
def void_run(self, *, val, dtype=BaseArray.float32):
    # TODO: not using numpy to extract shape
    return [VoidArray(np.array(val).shape, dtype)]


full = Operator.load("full")
operator_set.register(full)


@full.set_method
def jvp(self, primals, tangents, *, shape, fill_value, dtype=BaseArray.float32):
    out = slope.M().backend.run_impl(self, shape=shape, fill_value=fill_value, dtype=dtype)
    out_jvp = slope.M().ones_like(out)
    return [out], [out_jvp]


@full.set_method
def T(self, cts, *, shape, fill_value, dtype=BaseArray.float32):
    return [cts[0]]


@full.set_method
def void_run(self, *, shape, fill_value, dtype=BaseArray.float32) -> List[VoidArray]:
    return [VoidArray(tuple(shape), dtype)]


random_uniform = Operator.load("random_uniform")
rand = random_uniform
operator_set.register(random_uniform)
operator_set.alias(random_uniform, "randn")


@random_uniform.set_method
def jvp(self, primals, tangents, *, shape, dtype=BaseArray.float32):
    out = slope.M().backend.run_impl(self, shape=shape, dtype=dtype)
    out_jvp = slope.M().ones_like(out)
    return [out], [out_jvp]


@random_uniform.set_method
def T(self, cts, *, shape, dtype=BaseArray.float32):
    return [cts[0]]


@random_uniform.set_method
def void_run(self, *, shape, dtype=BaseArray.float32) -> List[VoidArray]:
    return [VoidArray(tuple(shape), dtype)]


random_normal = Operator.load("random_normal")
randn = random_normal
operator_set.register(random_normal)
operator_set.alias(random_normal, "randn")


@random_normal.set_method
def jvp(self, primals, tangents, *, shape, dtype=BaseArray.float32):
    out = slope.M().backend.run_impl(random_normal, shape, dtype)
    out_jvp = slope.M().ones_like(out)
    return [out], [out_jvp]


@random_normal.set_method
def T(self, cts, *, shape, dtype=BaseArray.float32):
    return [cts[0]]


@random_normal.set_method
def void_run(self, *, shape, dtype=BaseArray.float32) -> List[VoidArray]:
    return [VoidArray(tuple(shape), dtype)]


arange = Operator.load("arange")
operator_set.register(arange)


@arange.set_method
def args_fixer(self, *, start, stop=None, stride=None, dtype=BaseArray.float32):
    if stop is None:
        stop = start
        start = 0
    if stride is None:
        stride = 1
    return (), dict(start=start, stop=stop, stride=stride, dtype=dtype)


@arange.set_method
def jvp(self, primals, tangents, *, start, stop, stride=None, dtype=BaseArray.float32):
    out = slope.M().backend.run_impl(arange, start, stop, stride, dtype)
    out_jvp = slope.M().ones_like(out)
    return [out], [out_jvp]


@arange.set_method
def T(self, cts, *, start, stop, stride=None, dtype=BaseArray.float32):
    return [cts[0]]


@arange.set_method
def void_run(self, *, start, stop, stride=None, dtype=BaseArray.float32) -> List[VoidArray]:
    return [VoidArray(tuple((stop - start) * stride), dtype)]
