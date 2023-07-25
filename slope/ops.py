import slope
from slope.base_ops import Op as Op
from slope.base_array import Array

stop_gradient = Op.unary("stop_gradient")


@stop_gradient.set_eval
def stop_gradient_eval(x):
    return [x]


@stop_gradient.set_jvp
def stop_gradient_jvp(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x], [Array.zeros_like(x_dot)]


@stop_gradient.set_T
def stop_gradient_T(cts, x):
    (z,) = cts
    assert type(x) is slope.ad.UndefPrimal
    return [array.Array.zeros_like(z)]


convert = Op.unary("convert")


@convert.set_eval
def convert_eval(x):
    return [x]


@convert.set_jvp
def convert_jvp(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [x], [Array.zeros_like(x_dot)]


@convert.set_T
def convert_T(cts, x):
    (z,) = cts
    assert type(x) is slope.ad.UndefPrimal
    return [Array.zeros_like(z)]


sqrt = Op.unary("sqrt")


@sqrt.set_eval
def sqrt_eval(x):
    return [x.sqrt()]


@sqrt.set_jvp
def sqrt_jvp(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.sqrt()
    return [ans], [x_dot * (0.5 / ans)]


@convert.set_T
def sqrt_T(cts, x):
    (z,) = cts
    assert type(x) is slope.ad.UndefPrimal
    return [Array.zeros_like(z)]


sin = Op.unary("sin")


@sin.set_eval
def sin_eval(x):
    return [x.sin()]


@sin.set_jvp
def sin_jvp(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.sin()
    return [ans], [x_dot * (1 - ans)]


@sin.set_T
def sin_T(cts, x):
    (z,) = cts
    return [-z * (1 - x.sin())]


exp = Op.unary("exp")


@exp.set_eval
def exp_eval(x):
    return [x.exp()]


@exp.set_jvp
def exp_jvp(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.exp()
    return [ans], [x_dot * ans]


@exp.set_T
def exp_T(cts, x):
    (z,) = cts
    return [1 / z]


log = Op.unary("log")


@log.set_eval
def log_eval(x):
    return [x.log()]


@log.set_jvp
def log_jvp(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    ans = x.log()
    return [ans], [x_dot / x]


@log.set_T
def log_T(cts, x):
    (z,) = cts
    return [1 / z]


neg = Op.unary("neg")


@neg.set_eval
def neg_eval(x):
    return [-x]


@neg.set_jvp
def neg_jvp(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [-x], [-x_dot]


@neg.set_T
def relu_T(cts, x):
    (z,) = cts
    return [-z]


relu = Op.unary("relu")


@relu.set_eval
def relu_eval(x):
    return [-x]


@relu.set_jvp
def relu_jvp(primals, tangents, **params):
    (x,), (x_dot,) = primals, tangents
    return [-x], [-x_dot]


@relu.set_T
def relu_T(cts, x):
    (z,) = cts
    return [-z]


# -----------------------
# BinaryOps
# -----------------------


add = Op.binary("add")


@add.set_eval
def add_eval(x, y):
    return [x + y]


@add.set_jvp
def add_jvp(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x + y], [x_dot + y_dot]


@add.set_T
def add_T(cts, x, y):
    (z_bar,) = cts
    return [z_bar, z_bar]


sub = Op.binary("sub")


@sub.set_eval
def sub_eval(x, y):
    return [x - y]


@sub.set_jvp
def sub_jvp(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x - y], [x_dot - y_dot]


@sub.set_T
def sub_T(cts, x, y):
    (z_bar,) = cts
    return [z_bar, -z_bar]


mul = Op.binary("mul")


@mul.set_eval
def mul_eval(x, y):
    return [x * y]


@mul.set_jvp
def mul_jvp(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x * y], [(x_dot * y) + (y_dot * x)]
    # jvp_out = (y * x_dot) + (y_dot * x) # order problem, x*y_dot fails


@mul.set_T
def mul_T(cts, x, y):
    (z_bar,) = cts
    assert (type(x) is slope.ad.UndefPrimal) ^ (type(y) is slope.ad.UndefPrimal)
    if type(x) is slope.ad.UndefPrimal:
        return [z_bar * y, None]
    elif type(y) is slope.ad.UndefPrimal:
        return [None, x * z_bar]


div = Op.binary("div")


@div.set_eval
def div_eval(x, y):
    return [x / y]


@div.set_jvp
def div_jvp(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x / y], [
        (x_dot / y) + (-y_dot * x * (y**-2))
    ]  # bug: power returns float64


@div.set_T
def div_T(cts, x, y):
    (z_bar,) = cts
    return [z_bar / y, None]


maximum = Op.binary("maximum")


@maximum.set_eval
def maximum_eval(x, y):
    return [x.maximum(y)]


@maximum.set_jvp
def maximum_jvp(primals, tangents):
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


@maximum.set_T
def maximum_T(cts, x, y):
    (z_bar,) = cts
    return [z_bar, None]


equal = Op.binary("equal")


@equal.set_eval
def equal_eval(x, y):
    return [x.equal(y)]


@equal.set_jvp
def equal_jvp(primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = x.equal(y)
    return [out_primal], [Array.zeros(out_primal.shape, out_primal.dtype)]


@equal.set_T
def equal_T(cts, x, y):
    (z_bar,) = cts
    return [z_bar, None]


max = Op.reduce("max")


@max.set_eval
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


"""


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
        """


# Functions
@classmethod
def zeros(cls, shape, dtype=default_dtype, **kwargs):
    return cls.full(shape, 0.0, dtype, **kwargs)


@classmethod
def ones(cls, shape, dtype=default_dtype, **kwargs):
    return cls.full(shape, 1.0, dtype, **kwargs)


@classmethod
def full_like(cls, other, fill_value, **kwargs):
    return cls.full(other.shape, fill_value, dtype=other.dtype, **kwargs)


@classmethod
def zeros_like(cls, other, **kwargs):
    return cls.zeros(other.shape, dtype=other.dtype, **kwargs)


@classmethod
def ones_like(cls, other, **kwargs):
    return cls.full(other.shape, fill_value=1.0, dtype=other.dtype, **kwargs)


def where(self, trueval, falseval):
    cond = self != 0.0
    cond = cond.convert(trueval.dtype)  # TODO: type promotion logic
    return cond * trueval + (1.0 - cond) * falseval


def pow(self, y):
    assert type(y) is int
    if y == 0:
        return self.ones_like(x)
    is_reciprocal = y < 0
    if is_reciprocal:
        y = -y
    acc = None
    while y > 0:
        if y & 1:
            acc = x if acc is None else acc * x
        y >>= 1
        if y > 0:
            x = x * x
    ret = acc
    if is_reciprocal:
        ret = self.ones_like(acc) / acc
    return ret


def cross_entropy(x, y):
    return x * y.log()


def mse(x, y):
    return pow((x - y), 2)


def mean(self, axes=None, keepdims=False):
    out = self.sum(axes=axes, keepdim=keepdims)
    return out * (math.prod(out.shape) / math.prod(self.shape))


def minimum(self, other):
    return -self.maximum(-self, -other)


def min(self, axes=None, keepdims=False):
    return -((-self).max(self, axes, keepdims))


def flatten(self, start_dim=0):
    return self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))


@classmethod
def glorot_uniform(cls, *shape, **kwargs):
    return cls.rand(*shape, **kwargs).mul(
        (6 / (shape[0] + math.prod(shape[1:]))) ** 0.5
    )


@property
def T(self):
    perm = list(range(self.ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return self.transpose(perm)


def _softmax(self, axes):
    m = self - self.max(axes, keepdims=True)
    e = m.exp()
    return m, e, e.sum(axes, keepdims=True)


def softmax(self, axes=-1):
    _, e, ss = self._softmax(axes)
    return e.div(ss)


def log_softmax(self, axes=-1):
    m, _, ss = self._softmax(axes)
    return m - ss.log()


def dot(self, w):
    x = self.reshape((*self.shape[0:-1], 1, self.shape[-1]))
    w = w.reshape((*w.shape[0:-2], 1, w.shape[-2], w.shape[-1])).T
    return (x * w).sum(-1).reshape((*x.shape[0:-2], -1))


def square(self):
    return self * self


def clip(self, min_, max_):
    return ((self - min_).relu() + min_) - (self - max_).relu()


def abs(self):
    return self.relu() + (-self).relu()


def sign(self):
    return self / (self.abs() + 1e-10)


def reciprocal(self):
    return 1.0 / self
