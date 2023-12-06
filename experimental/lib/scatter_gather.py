class Gather(ShapeOp):
    get_impl = lambda: slope.RT.compiler.GatherImpl

    @staticmethod
    def run(x, idx, *, dim):
        return [x.gather(idx)]

    @staticmethod
    def _gather_batching_rule(
        dim_size,
        vals_in,
        dims_in,
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
            operand = batching.movedim(operand, operand_bdim, 0)
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
            indices = batching.movedim(indices, indices_bdim, 0)
            offset_dims = tuple(1 + d for d in dimension_numbers.offset_dims)
            dnums = GatherDimensionNumbers(
                offset_dims=offset_dims,
                collapsed_slice_dims=dimension_numbers.collapsed_slice_dims,
                start_index_map=dimension_numbers.start_index_map,
            )
            # If batching indexed accesses into the same tensor, the batched gather may
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
            operand = batching.movedim(operand, operand_bdim, 0)
            indices = batching.movedim(indices, indices_bdim, 0)

            # This slightly awkward special case is needed because the shape rule for
            # gather does not allow size-1 slices out of a size-0 dimension, even if
            # the number of slices is zero. Likely the best fix would be to change the
            # definition of gather() so it can be batched without the construction of
            # an explicit iota of size-1 slices.
            if core.symbolic_equal_dim(operand.shape[0], 0):
                output_shape = _gather_shape_rule(
                    core.ShapedTensor(operand.shape[1:], operand.dtype),
                    core.ShapedTensor(indices.shape[1:], dtypes.canonicalize_dtype(indices.dtype)),
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
            # (7, 3, 4, 6) where we catd an iota that counts along our batch
            # dimension to the front of the ndindex.
            count_shape = list(indices.shape)
            count_shape[-1] = 1
            counts = lax.broadcasted_iota(indices.dtype, tuple(count_shape), 0)
            indices = lax.cat([counts, indices], len(count_shape) - 1)

            slice_sizes = (1,) + slice_sizes
            collapsed_slice_dims = (0,) + tuple(np.add(1, dimension_numbers.collapsed_slice_dims))
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
    def jvp(
        primals,
        tangents,
        indices,
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
    def typecheck(
        x: Typecheckor,
        idx,
        *,
        dimension_numbers,
        slice_sizes,
        unique_indices,
        indices_are_sorted,
        mode,
        fill_value,
    ) -> List[Typecheckor]:
        offset_dims = dimension_numbers.offset_dims
        collapsed_slice_dims = dimension_numbers.collapsed_slice_dims
        start_index_map = dimension_numbers.start_index_map

        # Note: in JAX, index_vector_dim is always computed as below, cf. the
        # documentation of the GatherDimensionNumbers class.
        index_vector_dim = _rank(indices) - 1

        # This case should never happen in JAX, due to the implicit construction of
        # index_vector_dim, but is included for completeness.
        if _rank(indices) < index_vector_dim or index_vector_dim < 0:
            raise TypeError(
                f"Gather index leaf dimension must be within [0, rank("
                f"indices) + 1). rank(indices) is {_rank(indices)} and "
                f"gather index leaf dimension is {index_vector_dim}."
            )

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
                raise TypeError(
                    f"Offset dimension {i} in gather op is out of bounds; "
                    f"got {offset_dim}, but should have been in "
                    f"[0, {output_shape_rank})"
                )

            if len(start_index_map) != indices.shape[index_vector_dim]:
                raise TypeError(
                    f"Gather op has {len(start_index_map)} elements in "
                    f"start_index_map and the bound of dimension "
                    f"{index_vector_dim=} of indices is "
                    f"{indices.shape[index_vector_dim]}. These two "
                    f"numbers must be equal."
                )

            for i in range(len(start_index_map)):
                operand_dim_for_start_index_i = start_index_map[i]
                if operand_dim_for_start_index_i < 0 or operand_dim_for_start_index_i >= _rank(operand):
                    raise TypeError(
                        f"Invalid start_index_map; domain is "
                        f"[0, {_rank(operand)}), got: "
                        f"{i}->{operand_dim_for_start_index_i}."
                    )

            _no_duplicate_dims(start_index_map, "gather", "start_index_map")

        # _is_sorted and _sorted_dims_in_range are checked in the opposite order
        # compared to the XLA implementation. In cases when the input is not sorted
        # AND there are problematic collapsed_slice_dims, the error message will thus
        # be different.
        _is_sorted(collapsed_slice_dims, "gather", "collapsed_slice_dims")
        _sorted_dims_in_range(collapsed_slice_dims, _rank(operand), "gather", "collapsed_slice_dims")
        _no_duplicate_dims(collapsed_slice_dims, "gather", "collapsed_slice_dims")
        # End ValidateGatherDimensions

        if _rank(operand) != len(slice_sizes):
            raise TypeExrror(
                f"Gather op must have one slice size for every input "
                f"dimension; got: len(slice_sizes)={len(slice_sizes)}, "
                f"input_shape.rank={_rank(operand)}"
            )

        if len(slice_sizes) != len(offset_dims) + len(collapsed_slice_dims):
            raise TypeError(
                f"All components of the offset index in a gather op must "
                f"either be a offset dimension or explicitly collapsed; "
                f"got len(slice_sizes)={len(slice_sizes)}, "
                f"output_slice_sizes={offset_dims}, collapsed_slice_dims="
                f"{collapsed_slice_dims}."
            )

        for i in range(len(slice_sizes)):
            slice_size = slice_sizes[i]
            corresponding_input_size = operand.shape[i]

            if not (
                core.greater_equal_dim(slice_size, 0) and core.greater_equal_dim(corresponding_input_size, slice_size)
            ):
                raise TypeError(
                    f"Slice size at index {i} in gather op is out of range, "
                    f"must be within [0, {corresponding_input_size} + 1), "
                    f"got {slice_size}."
                )

        for i in range(len(collapsed_slice_dims)):
            bound = slice_sizes[collapsed_slice_dims[i]]
            if bound != 1:
                raise TypeError(
                    f"Gather op can only collapse slice dims with bound 1, "
                    f"but bound is {bound} for index "
                    f"{collapsed_slice_dims[i]} at position {i}."
                )

        expanded_indices_shape.pop(index_vector_dim)
        indices_shape = iter(expanded_indices_shape)

        slice_sizes = (s for i, s in enumerate(slice_sizes) if i not in collapsed_slice_dims)
        return tuple(next(slice_sizes) if i in offset_dims else next(indices_shape) for i in range(output_shape_rank))

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
    get_impl = lambda: slope.RT.compiler.ScatterImpl

    @staticmethod
    def run(x, idx, *, dim):
        return [x.gatter(idx)]

    @staticmethod
    def vmap(
        dim_size,
        vals_in,
        dims_in,
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
        size = next(x.shape[ax] for x, ax in zip(batched_args, batch_dims) if ax is not None)
        operand = batching.bdim_at_front(operand, operand_bdim, size)
        operand_bdim = 0

        updates = batching.bdim_at_front(updates, updates_bdim, size)

        if indices_bdim is None:
            inserted_window_dims = tuple(np.add(1, dimension_numbers.inserted_window_dims))
            update_window_dims = (0,) + tuple(np.add(1, dimension_numbers.update_window_dims))
            scatter_dims_to_operand_dims = tuple(np.add(1, dimension_numbers.scatter_dims_to_operand_dims))
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
        indices = lax.cat([counts, indices], len(count_shape) - 1)

        update_window_dims = tuple(np.add(1, dimension_numbers.update_window_dims))
        inserted_window_dims = (0,) + tuple(np.add(1, dimension_numbers.inserted_window_dims))
        scatter_dims_to_operand_dims = (0,) + tuple(np.add(1, dimension_numbers.scatter_dims_to_operand_dims))

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
    def typecheck(x: Typecheckor, idx, *, dim: Sequence[int]) -> List[Typecheckor]:
        shape = [x.shape[i] for i in dim]
        return [Typecheckor(shape, x.dtype)]

    @staticmethod
    def T(
        cts,
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
            operand_t = ad_util.Zero(operand.aval) if ad.is_undefined_primal(operand) else None
            update_t = ad_util.Zero(updates.aval) if ad.is_undefined_primal(updates) else None
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
                        slice_sizes.append(updates_shape[dimension_numbers.update_window_dims[pos]])
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


@numpy_compiler.set_impl("scatter")
def fn(
    inputs,
    scatter_indices,
    updates,
    *,
    update_window_dims,
    inserted_window_dims,
    scatter_dims_to_operand_dims,
    index_vector_dim: int,
    slice_sizes,
    result_type,
    ret,
):
    return """
# SmallVector<Tensor> runScatterOp(
#     TensorRef<Tensor> inputs, const Tensor &scatterIndices,
#     TensorRef<Tensor> updates, const dim &updateWindowDims,
#     const dim &insertedWindowDims, const dim &scatterDimsToOperandDims,
#     dim indexVectorDim, Region &updateComputation, Scope &scope,
#     TensorRef<ShapedType> resultTypes) {
#   SmallVector<Tensor> results;
#   for (auto input : inputs) results.push_back(input);

#   dim updateScatterDims;
#   for (auto d : updates[0].getdim())
#     if (!llvm::is_contained(updateWindowDims, d))
#       updateScatterDims.push_back(d);

#   for (auto updateIndexIt = updates[0].index_begin();
#        updateIndexIt != updates[0].index_end(); ++updateIndexIt) {
#     auto updateIndex = *updateIndexIt;
#     Index updateScatterIndex;
#     for (auto d : updateScatterDims)
#       updateScatterIndex.push_back(updateIndex[d]);

#     auto startIndicesIndex = updateScatterIndex;
#     if (indexVectorDim < scatterIndices.getRank())
#       startIndicesIndex.insert(startIndicesIndex.begin() + indexVectorDim,
#                                kColon);
#     auto startIndex = runIndex(runSliceOp(scatterIndices, startIndicesIndex));

#     Index fullStartIndex(inputs[0].getRank(), 0);
#     for (auto dInput : inputs[0].getdim()) {
#       auto dStartIt = llvm::find(scatterDimsToOperandDims, dInput);
#       if (dStartIt == scatterDimsToOperandDims.end()) continue;
#       auto dStart = dStartIt - scatterDimsToOperandDims.begin();
#       fullStartIndex[dInput] = startIndex[dStart];
#     }

#     Index updateWindowIndex;
#     for (auto d : updateWindowDims) updateWindowIndex.push_back(updateIndex[d]);

#     Index fullWindowIndex(updateWindowIndex.size() + insertedWindowDims.size(),
#                           0);
#     for (size_t i = 0, wi = 0; i < fullWindowIndex.size(); ++i) {
#       if (llvm::is_contained(insertedWindowDims, i)) continue;
#       fullWindowIndex[i] = updateWindowIndex[wi++];
#     }

#     auto resultIndex = fullStartIndex + fullWindowIndex;
#     if (!resultIndex.inBounds(results[0].getShape())) continue;

#     SmallVector<InterpreterValue> updateComputationArgs;
#     for (auto result : results)
#       updateComputationArgs.push_back(
#           Tensor(RankedTensorType::get({}, result.getElementType()),
#                  result.get(resultIndex)));
#     for (auto update : updates)
#       updateComputationArgs.push_back(
#           Tensor(RankedTensorType::get({}, update.getElementType()),
#                  update.get(updateIndex)));

#     auto updatedValues = run(updateComputation, updateComputationArgs, &scope);
#     for (auto [result, updatedValue] : llvm::zip(results, updatedValues))
#       result.set(resultIndex, updatedValue.getTensor().get({}));
#   }

#   return results;
# }
"""


@numpy_compiler.set_impl("gather")
def fn(
    operand,
    start_indices,
    *,
    collapsed_slice_dims,
    start_index_map,
    offset_dims,
    index_vector_dim: int,
    slice_sizes,
):
    expanded_indices_shape = list(start_indices.shape)
    if len(expanded_indices_shape) == index_vector_dim:
        expanded_indices_shape.append(1)

    output_shape_rank = len(offset_dims) + start_indices.ndim - 1

    expanded_indices_shape.pop(index_vector_dim)
    indices_shape = iter(expanded_indices_shape)

    slice_sizes = (s for i, s in enumerate(slice_sizes) if i not in collapsed_slice_dims)
    res_size = tuple(next(slice_sizes) if i in offset_dims else next(indices_shape) for i in range(output_shape_rank))

    res = np.zeros(res_size)
    batch_dims = [d for d in list(range(res.ndim)) if d in offset_dims]

    for res_idx, _ in np.ndenumerate(res):
        batch_idx = [res_idx[d] for d in batch_dims]

    start_indices_idx = batch_idx[:]
    if index_vector_dim < start_indices.ndim:
        start_indices_idx.insert(index_vector_dim, -1)
    start_idx = start_indices[start_indices_idx]

    full_start_idx = [None] * operand.ndim
    for d in range(operand.ndim):
        dStartIt = start_index_map[d]
        if dStartIt == start_index_map[-1]:
            continue
        dStart = dStartIt - start_index_map[0]
        full_start_idx[d] = np.clip(start_idx[d], operand.shape[d] - slice_sizes[d])

    offset_idx = [res_idx[d] for d in offset_dims]
    full_offset_idx = [None] * (len(offset_dims) + len(collapsed_slice_dims))
    oi = 0
    for i in range(len(full_offset_idx)):
        if i in collapsed_slice_dims:
            continue
        full_offset_idx[i] = offset_idx[oi]
        oi += 1
    operand_idx = full_start_idx + full_offset_idx
    res[res_idx] = operand[operand_idx]
    return res
