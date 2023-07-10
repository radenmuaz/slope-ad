import slope
import numpy as np
from slope.array_shape import ArrayShape
from typing import List, Tuple, Sequence, Any, Callable, Optional, Union
from abc import ABC, abstractmethod
import math
import functools
from slope.array import Array
from slope import utils


class Op(ABC):
    get_impl = lambda: None

    @classmethod
    def do(cls, *args, **params):
        return slope.RT.bind1(cls, *args, **params)

    @staticmethod
    @abstractmethod
    def eval(*args, **params):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def vmap(*args, **params):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def jvp(*args, **params):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def shape_eval(*args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def pprint(cls):
        return None

    @staticmethod
    @abstractmethod
    def jit(*args, **params):
        raise NotImplementedError


class UnaryOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in, **params):
        (x,), (x_bdim,) = vals_in, dims_in
        return [cls.do(x, **params)], [x_bdim]

    @staticmethod
    def shape_eval(x: ArrayShape, **params) -> List[ArrayShape]:
        return [ArrayShape(x.shape, x.dtype)]

    @classmethod
    def identity_jvp(cls, primals, tangents, **params):
        (x,), (x_dot,) = primals, tangents
        return [cls.do(x, **params)], [cls.do(x_dot, **params)]

    # @classmethod
    # def identity_T(cls, t, x):
    #     (z,) = t
    #     assert type(x) is slope.ad.UndefPrimal
    #     return [cls.do(z)]

    # @classmethod
    # def zero_T(cls, t, x):
    #     (z,) = t
    #     assert type(x) is slope.ad.UndefPrimal
    #     return [z.zeros_like()]


class BinaryOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in, **params):
        (x, y), (x_bdim, y_bdim) = vals_in, dims_in
        if x_bdim != y_bdim:
            if x_bdim is None:
                x = slope.ad.move_batch_axis(axis_size, x_bdim, y_bdim, x)
                x_bdim = y_bdim
            else:
                y = slope.ad.move_batch_axis(axis_size, y_bdim, x_bdim, y)
        return [cls.do(x, y, **params)], [x_bdim]

    @staticmethod
    def shape_eval(x: ArrayShape, y: ArrayShape, **params) -> List[ArrayShape]:
        # if not isinstance(x, ArrayShape) or not isinstance(y, ArrayShape):
        if not type(x) in (Array, ArrayShape) or not type(x) in (Array, ArrayShape):
            # breakpoint()
            raise TypeError
        if ArrayShape.like(x) != ArrayShape.like(y):
            raise TypeError(f"{x} != {y}")
        return [ArrayShape(x.shape, x.dtype)]


class ReduceOp(Op):
    @classmethod
    def vmap(cls, axis_size, vals_in, dims_in, **params):
        (x,), (x_bdim,) = vals_in, dims_in
        axes = list(params["axes"])
        axes = tuple(a + (x_bdim <= a) for a in axes)
        out_bdim = x_bdim - sum(a < x_bdim for a in axes)
        params["axes"] = tuple(axes)
        return [cls.do(x, **params)], [out_bdim]

    @staticmethod
    def shape_eval(x: ArrayShape, **params) -> List[ArrayShape]:
        axes = params["axes"]
        axes = [a + len(x.shape) if a < 0 else a for a in axes]
        axes_ = set(axes)
        new_shape = [d for i, d in enumerate(x.shape) if i not in axes_]
        return [ArrayShape(tuple(new_shape), x.dtype)]


class ShapeOp(Op):
    pass


class LoadOp(Op):
    pass


# -----------------------
# UnaryOps
# -----------------------


# class Identity(UnaryOp):
#     @staticmethod
#     def eval(x):
#         return [x]

#     @staticmethod
#     def jvp(cls, primals, tangents, **params):
#         (x,), (x_dot,) = primals, tangents
#         return [identity(x, **params)], [identity(x_dot, **params)]

#     @staticmethod
#     def T(t, x):
#         (z,) = t
#         assert type(x) is slope.ad.UndefPrimal
#         return [identity(z)]


# class StopGradient(UnaryOp):
#     @staticmethod
#     def eval(x):
#         return [identity(x)]

#     @staticmethod
#     def jvp(primals, tangents, **params):
#         (x,), (x_dot,) = primals, tangents
#         return [identity(x, **params)], [zeros_like(x)]

#     @staticmethod
#     def T(t, x):
#         (z,) = t
#         assert type(x) is slope.ad.UndefPrimal
#         return [zeros_like(z)]


class Convert(UnaryOp):
    get_impl = lambda: slope.RT.backend.ConvertImpl

    @staticmethod
    def eval(x, *, dtype):
        return [x.astype(dtype)]

    @staticmethod
    def jvp(primals, tangents, *, dtype):
        (x,), (x_dot,) = primals, tangents
        return [x.convert(dtype)], [x_dot.convert(dtype)]

    @staticmethod
    def T(t, x):
        (z,) = t
        assert type(x) is slope.ad.UndefPrimal
        return [z.convert(x.dtype)]


class Exp(UnaryOp):
    get_impl = lambda: slope.RT.backend.ExpImpl

    @staticmethod
    def eval(x):
        return [x.exp()]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [x.exp()], [x_dot * x.exp()]


class Log(UnaryOp):
    get_impl = lambda: slope.RT.backend.LogImpl

    @staticmethod
    def eval(x):
        return [x.log()]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [x.log()], [x_dot / x]


class Neg(UnaryOp):
    get_impl = lambda: slope.RT.backend.NegImpl

    @staticmethod
    def eval(x):
        return [-x]

    @staticmethod
    def jvp(primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return [-x], [-x_dot]

    @staticmethod
    def T(t, x):
        (z,) = t
        return [-z]


# -----------------------
# BinaryOps
# -----------------------


class Add(BinaryOp):
    get_impl = lambda: slope.RT.backend.AddImpl

    @staticmethod
    def eval(x, y):
        return [x + y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x + y], [x_dot + y_dot]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar, z_bar]


class Sub(BinaryOp):
    get_impl = lambda: slope.RT.backend.SubImpl

    @staticmethod
    def eval(x, y):
        return [x - y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x - y], [x_dot - y_dot]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar, -z_bar]


class Mul(BinaryOp):
    get_impl = lambda: slope.RT.backend.MulImpl

    @staticmethod
    def eval(x, y):
        return [x * y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        eval_out = x * y
        jvp_out = (x_dot * y) + (y_dot * x)
        # jvp_out = (y * x_dot) + (y_dot * x) # order problem, x*y_dot fails
        # check about __array_priority
        return [eval_out], [jvp_out]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        assert (type(x) is slope.ad.UndefPrimal) ^ (type(y) is slope.ad.UndefPrimal)
        if type(x) is slope.ad.UndefPrimal:
            return [z_bar * y, None]
        elif type(y) is slope.ad.UndefPrimal:
            return [None, x * z_bar]


class Div(BinaryOp):
    get_impl = lambda: slope.RT.backend.DivImpl

    @staticmethod
    def eval(x, y):
        return [x / y]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return [x / y], [
            (x_dot / y) + (-y_dot * x * (y**-2))
        ]  # bug: power returns float64

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar / y, None]


class Maximum(BinaryOp):
    get_impl = lambda: slope.RT.backend.MaximumImpl

    @staticmethod
    def eval(x, y):
        return [np.maximum(x, y)]

    @staticmethod
    def jvp(primals, tangents):
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

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar, None]


class Equal(BinaryOp):
    get_impl = lambda: slope.RT.backend.EqualImpl

    @staticmethod
    def eval(x, y):
        return [x.equal(y)]

    @staticmethod
    def jvp(primals, tangents):
        (x, y), _ = primals, tangents
        out_primal = x.equal(y)
        return [out_primal], [np.zeros(out_primal.shape, out_primal.dtype)]

    @staticmethod
    def T(cts, x, y):
        (z_bar,) = cts
        return [z_bar, None]


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


class Gather(ShapeOp):
    get_impl = lambda: slope.RT.backend.GatherImpl

    @staticmethod
    def eval(x, idx, *, axis):
        return [x.gather(idx)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, perm):
        raise NotImplementedError

    @staticmethod
    def jvp(primals, tangents, *, axis):
        (x, idx), (x_dot, _) = primals, tangents
        return [x.gather(idx)], [x_dot.gather(idx)]

    @staticmethod
    def shape_eval(x: ArrayShape, idx, *, axis: Sequence[int]) -> List[ArrayShape]:
        shape = [x.shape[i] for i in axis]
        return [ArrayShape(shape, x.dtype)]

    @staticmethod
    def T(cts, *, axis):
        (z, idx) = cts
        return [z.gather(axis)]


def _gather_jvp_rule(
    g,
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


def _gather_transpose_rule(
    t,
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


def _gather_batching_rule(
    batched_args,
    batch_dims,
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


class Scatter(ShapeOp):
    get_impl = lambda: slope.RT.backend.ScatterImpl

    @staticmethod
    def eval(x, idx, *, axis):
        return [x.gather(idx)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, perm):
        raise NotImplementedError

    @staticmethod
    def jvp(primals, tangents, *, axis):
        (x, idx), (x_dot, _) = primals, tangents
        return [x.gather(idx)], [x_dot.gather(idx)]

    @staticmethod
    def shape_eval(x: ArrayShape, idx, *, axis: Sequence[int]) -> List[ArrayShape]:
        shape = [x.shape[i] for i in axis]
        return [ArrayShape(shape, x.dtype)]

    @staticmethod
    def T(cts, *, axis):
        (z, idx) = cts
        return [z.gather(axis)]


def _scatter_batching_rule(
    scatter_op,
    batched_args,
    batch_dims,
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


def _scatter_add_jvp(
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


def _scatter_add_transpose_rule(
    t,
    operand,
    indices,
    updates,
    *,
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


class Pad(ShapeOp):
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
    @staticmethod
    def eval(x, *, padding):
        return [x.crop(padding)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, perm):
        raise NotImplementedError

    @staticmethod
    def jvp(primals, tangents, *, padding):
        (x,), (x_dot, _) = primals, tangents
        return [x.pad(padding)], [x_dot.pad(padding)]

    @staticmethod
    def shape_eval(x: ArrayShape, *, padding: Sequence[int]) -> List[ArrayShape]:
        shape = [x.shape[i] for i in padding]
        return [ArrayShape(shape, x.dtype)]

    @staticmethod
    def T(cts, *, padding):
        (z,) = cts
        return [z.crop(padding)]


class Concatenate(ShapeOp):
    @staticmethod
    def eval(xs: Sequence[Any], *, axis):
        return [Array.concatenate(xs, axis=axis)]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, perm):
        raise NotImplementedError

    @staticmethod
    def jvp(primals: Sequence[Any], tangents: Sequence[Any], *, axis):
        (xs,), (xs_dot,) = primals, tangents
        return [Array.concatenate(xs, axis=axis)], [
            Array.concatenate(xs_dot, axis=axis)
        ]

    @staticmethod
    def shape_eval(x: ArrayShape, idx, *, axis: Sequence[int]) -> List[ArrayShape]:
        shape = [x.shape[i] for i in axis]
        return [ArrayShape(shape, x.dtype)]

    @staticmethod
    def T(cts, *, axis):
        (zs,) = cts
        return [Array.concatenate(zs, axis=axis)]


#


class Full(LoadOp):
    @staticmethod
    def eval(*, fill_value, shape, dtype):
        out = Array.full(fill_value, shape, dtype)
        return [out]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, shape, axes):
        raise NotImplementedError

    @staticmethod
    def jvp(*, fill_value, shape, dtype):
        return (
            [Array.full(fill_value, shape, dtype)],
            [Array.full(0, shape)],
        )

    @staticmethod
    def shape_eval(fill_value, shape, dtype) -> List[ArrayShape]:
        return [ArrayShape(tuple(shape), dtype)]

    @staticmethod
    def T(cts, *, fill_value, shape, dtype):
        return [cts[0]]


class Arange(LoadOp):
    @staticmethod
    def eval(*, start, stop, stride, dtype):
        out = Array.arange(start, stop, stride, dtype)
        return [out]

    @staticmethod
    def vmap(
        axis_size,
        vals_in,
        dims_in,
        *,
        start,
        stop,
        stride,
    ):
        raise NotImplementedError

    @staticmethod
    def jvp(*, start, stop, stride, dtype):
        return (
            [Array.arange(start, stop, stride, dtype)],
            [Array.full(0, len(tuple(slice(start, stop, stride))))],
        )

    @staticmethod
    def shape_eval(start, stop, stride, dtype) -> List[ArrayShape]:
        return [ArrayShape(len(tuple(slice(start, stop, stride))), dtype)]

    @staticmethod
    def T(cts, *, start, stop, stride, dtype):
        return [cts[0]]


class Constant(LoadOp):
    @staticmethod
    def eval(*, x):
        out = Array(x)
        return [out]

    @staticmethod
    def vmap(axis_size, vals_in, dims_in, *, x):
        raise NotImplementedError

    @staticmethod
    def jvp(*, x):
        return (
            [Array(x)],
            [Array.full(0, x.shape, x.dtype)],
        )

    @staticmethod
    def shape_eval(*, x) -> List[ArrayShape]:
        return [ArrayShape(tuple(x.shape), x.dtype)]

    @staticmethod
    def T(cts, *, x):
        return [cts[0]]


class Jit(LoadOp):
    @staticmethod
    def eval(*args, hashable_prog, hashable_consts):
        jit_fn = slope.RT.backend.callable(hashable_prog, hashable_consts)
        return [jit_fn(*args), jit_fn]

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
