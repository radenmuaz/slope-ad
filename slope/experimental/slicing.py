
def take(self, idx):
    treedef, static_idx, dynamic_idx = _split_index_for_jit(idx, self.shape)
    return _gather(self, treedef, static_idx, dynamic_idx)


def _gather(arr, treedef, static_idx, dynamic_idx):
    idx = _merge_static_and_dynamic_indices(treedef, static_idx, dynamic_idx)
    indexer = _index_to_gather(arr.shape, idx)  # shared with _scatter_update
    y = arr
    # handle cases like zeros(0)[array([], int32)].
    if indexer.slice_shape == ():
        return Array.zeros(shape=indexer.slice_shape, dtype=y.dtype)
    # We avoid generating a gather when indexer.gather_indices.size is empty.
    if indexer.gather_indices.shape != ():
        y = y.gather(indexer.gather_indices, indexer.dnums, indexer.gather_slice_shape)
    # Reverses axes with negative strides.
    if indexer.reversed_y_dims:
        raise NotImplementedError
    # This adds np.newaxis/None dimensions.
    return y.expand_dims(indexer.newaxis_dims)


class GatherDimensionNumbers(NamedTuple):
    offset_dims: Tuple[int, ...]
    collapsed_slice_dims: Tuple[int, ...]
    start_index_map: Tuple[int, ...]


class _Indexer(NamedTuple):
    slice_shape: Sequence[int]
    gather_slice_shape: Sequence[int]
    gather_indices: Array
    dnums: GatherDimensionNumbers
    unique_indices: bool
    indices_are_sorted: bool
    reversed_y_dims: Sequence[int]
    newaxis_dims: Sequence[int]


def _split_index_for_jit(idx, shape):
    """Splits indices into necessarily-static and dynamic parts.

    Used to pass indices into `jit`-ted function.
    """
    # Convert list indices to tuples in cases (deprecated by NumPy.)
    assert type(idx) in (int, tuple, Array)
    if type(idx) is int:
        idx = (idx,)
    # idx = _expand_bool_indices(idx, shape)

    leaves, treedef = slope.ad.tree_flatten(idx)
    dynamic = [None] * len(leaves)
    static = [None] * len(leaves)
    for i, x in enumerate(leaves):
        if x is Ellipsis:
            static[i] = x
        elif isinstance(x, slice):
            # slice objects aren't hashable.
            static[i] = (x.start, x.stop, x.step)
        else:
            dynamic[i] = x
    return treedef, tuple(static), dynamic


def _merge_static_and_dynamic_indices(treedef, static_idx, dynamic_idx):
    """Recombines indices that were split by _split_index_for_jit."""
    idx = []
    for s, d in zip(static_idx, dynamic_idx):
        if d is not None:
            idx.append(d)
        elif isinstance(s, tuple):
            idx.append(slice(s[0], s[1], s[2]))
        else:
            idx.append(s)
    return treedef.unflatten(idx)


# def _canonicalize_tuple_index(arr_ndim, idx, array_name="array"):
#     """Helper to remove Ellipsis and add in the implicit trailing slice(None)."""
#     len_without_none = sum(1 for e in idx if e is not None and e is not Ellipsis)
#     if len_without_none > arr_ndim:
#         raise IndexError(
#             f"Too many indices for {array_name}: {len_without_none} "
#             f"non-None/Ellipsis indices for dim {arr_ndim}."
#         )
#     ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
#     ellipsis_index = next(ellipses, None)
#     if ellipsis_index is not None:
#         if next(ellipses, None) is not None:
#             raise IndexError(
#                 f"Multiple ellipses (...) not supported: {list(map(type, idx))}."
#             )
#         colons = (slice(None),) * (arr_ndim - len_without_none)
#         idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1 :]
#     elif len_without_none < arr_ndim:
#         colons = (slice(None),) * (arr_ndim - len_without_none)
#         idx = tuple(idx) + colons
#     return idx


def _is_int_arraylike(x):
    """Returns True if x is array-like with integer dtype, False otherwise."""
    return (
        isinstance(x, int)
        and not isinstance(x, bool)
        or getattr(x, "dtype", None) == np.integer
        or isinstance(x, (list, tuple))
        and all(_is_int_arraylike(e) for e in x)
    )


def _normalize_index(index, axis_size):
    return (index < 0).select(index + axis_size, index)


def broadcast_arrays(*args) -> List[Array]:
    """Like Numpy's broadcast_arrays but doesn't return views."""
    shapes = [np.shape(arg) for arg in args]
    if not shapes or all(shapes[0] ==  s for s in shapes):
        return Array(arg)
    result_shape = _broadcast_shapes_uncached(*shapes)
    return [_broadcast_to(arg, result_shape) for arg in args]


def _broadcast_to(arr, shape) -> Array:
    if not isinstance(shape, tuple) and np.ndim(shape) == 0:
        shape = (shape,)
    if arr.shape == shape:
        return arr
    else:
        nlead = len(shape) - len(arr.shape)
        shape_tail = shape[nlead:]
        (diff,) = np.where(
            tuple(
                (len(arr_d) != len(shape_d))
                for arr_d, shape_d in zip(arr.shape, shape_tail)
            )
        )
        new_dims = tuple(range(nlead)) + tuple(nlead + diff)
        kept_dims = tuple(np.delete(np.arange(len(shape)), new_dims))
        return arr.squeeze(tuple(diff)).broadcast(shape, kept_dims)

def _broadcast_shapes_uncached(*shapes):
    def _broadcast_ranks(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        assert len(s1) <= len(s2)
        s1_ = s2[len(s2) - len(s1) :]
        if s1_.shape == s1.shape:
            return s2
        else:
            raise ValueError

    fst, *rst = shapes
    if not rst:
        return fst

    # First check if we need only rank promotion (and not singleton-broadcasting).
    try:
        return reduce(_broadcast_ranks, rst, fst)
    except ValueError:
        pass

    # Next try singleton-broadcasting, padding out ranks using singletons.
    ndim = max(len(shape) for shape in shapes)
    shape_list = [(1,) * (ndim - len(shape)) + shape for shape in shapes]
    result_shape = _try_broadcast_shapes(shape_list)
    if result_shape is None:
        raise ValueError(f"Incompatible shapes for broadcasting: shapes={list(shapes)}")
    return result_shape


def _try_broadcast_shapes(
    shapes: Sequence[Tuple[int, ...]]
) -> Optional[Tuple[int, ...]]:
    if len(shapes) == 1:
        return shapes[0]
    rank, *others = {len(shape) for shape in shapes}
    if others:
        return None  # must have consistent rank
    if not rank:
        return ()  # scalar case
    result_shape = []
    for ds in zip(*shapes):
        if all((d == ds[0]) for d in ds[1:]):
            result_shape.append(ds[0])
        else:
            non_1s = [d for d in ds if d != 1]
            if not non_1s:
                result_shape.append(1)
            elif all((non_1s[0] == d) for d in non_1s[1:]):
                result_shape.append(non_1s[0])
            else:
                return None
    return tuple(result_shape)


def _static_idx(idx: slice, size):
    """Helper function to compute the static slice start/limit/stride values."""
    if isinstance(size, int):
        start, stop, step = idx.indices(size)
    else:
        raise TypeError(size)

    if (step < 0 and stop >= start) or (step > 0 and start >= stop):
        return 0, 0, 1, False  # sliced to size zero

    if step > 0:
        return start, stop, step, False
    else:
        k = (start - stop - 1) % (-step)
        return stop + k + 1, start + 1, -step, True


def _index_to_gather(
    x_shape: Sequence[int], idx: Sequence[Any], normalize_indices: bool = True
) -> _Indexer:
    # Remove ellipses and add trailing slice(None)s.
    # idx = _canonicalize_tuple_index(len(x_shape), idx)
    arr_ndim = len(x_shape)
    len_without_none = sum(1 for e in idx if e is not None and e is not Ellipsis)
    if len_without_none > arr_ndim:
        raise IndexError(
            f"Too many indices for array: {len_without_none} "
            f"non-None/Ellipsis indices for dim {arr_ndim}."
        )
    ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
    ellipsis_index = next(ellipses, None)
    if ellipsis_index is not None:
        if next(ellipses, None) is not None:
            raise IndexError(
                f"Multiple ellipses (...) not supported: {list(map(type, idx))}."
            )
        colons = (slice(None),) * (arr_ndim - len_without_none)
        idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1 :]
    elif len_without_none < arr_ndim:
        colons = (slice(None),) * (arr_ndim - len_without_none)
        idx = tuple(idx) + colons
    advanced_axes_are_contiguous = False

    advanced_indexes: Optional[Sequence[Union[Array, np.ndarray]]] = None
    idx_advanced_axes: Sequence[int] = []
    x_advanced_axes: Optional[Sequence[int]] = None

    def _is_advanced_int_indexer(idx):
        assert isinstance(idx, tuple)
        if all(
            e is None
            or e is Ellipsis
            or isinstance(e, slice)
            or e.shape == ()
            and e.dtype in (int, np.int8, np.integer)
            for e in idx
        ):
            return False
        return all(
            e is None or e is Ellipsis or isinstance(e, slice) or _is_int_arraylike(e)
            for e in idx
        )

    if _is_advanced_int_indexer(idx):
        idx_no_nones = [(i, d) for i, d in enumerate(idx) if d is not None]
        advanced_pairs = (
            (Array(e), i, j) for j, (i, e) in enumerate(idx_no_nones) if e is not None
        )
        if normalize_indices:
            advanced_pairs = (
                (_normalize_index(e, x_shape[j]), i, j) for e, i, j in advanced_pairs
            )
        advanced_indexes, idx_advanced_axes, x_advanced_axes = zip(*advanced_pairs)
        advanced_axes_are_contiguous = bool(np.all(np.diff(idx_advanced_axes) == 1))

    x_axis = 0  # Current axis in x.
    y_axis = 0  # Current axis in y, before collapsing. See below.
    collapsed_y_axis = 0  # Current axis in y, after collapsing.

    # Scatter dimension numbers.
    offset_dims: Sequence[int] = []
    collapsed_slice_dims: Sequence[int] = []
    start_index_map: Sequence[int] = []

    index_dtype = np.int32
    gather_indices: List[Any] = []
    gather_indices_shape: List[int] = []
    slice_shape: Sequence[int] = []

    newaxis_dims: Sequence[int] = []

    reversed_y_dims: Sequence[int] = []

    gather_slice_shape: Sequence[int] = []

    for idx_pos, i in enumerate(idx):
        if advanced_indexes is not None and (
            advanced_axes_are_contiguous
            and idx_pos == idx_advanced_axes[0]
            or not advanced_axes_are_contiguous
            and idx_pos == 0
        ):
            advanced_indexes = broadcast_arrays(*advanced_indexes)
            shape = advanced_indexes[0].shape
            ndim = len(shape)

            start_dim = len(gather_indices_shape)
            gather_indices += (
                (a.astype(index_dtype), start_dim) for a in advanced_indexes
            )
            gather_indices_shape += shape

            start_index_map.extend(x_advanced_axes)
            collapsed_slice_dims.extend(x_advanced_axes)
            slice_shape.extend(shape)
            y_axis += ndim
            collapsed_y_axis += ndim

        # Per-index bookkeeping for advanced indexes.
        if idx_pos in idx_advanced_axes:
            x_axis += 1
            gather_slice_shape.append(1)
            continue

        try:
            abstract_i = Tracer.get_aval(i)
        except TypeError:
            abstract_i = None
        # Handle basic int indexes.
        if isinstance(abstract_i, (Array,)) and (
            not abstract_i.shape and abstract_i.dtype == np.integer
        ):
            if x_shape[x_axis] == 0:
                # XLA gives error when indexing into an axis of size 0
                raise IndexError(
                    f"index is out of bounds for axis {x_axis} with size 0"
                )
            i = _normalize_index(i, x_shape[x_axis]) if normalize_indices else i
            i = i.astype(index_dtype)
            gather_indices.append((i, len(gather_indices_shape)))
            collapsed_slice_dims.append(x_axis)
            gather_slice_shape.append(1)
            start_index_map.append(x_axis)
            x_axis += 1
        # Handle np.newaxis (None)
        elif i is None:
            slice_shape.append(1)
            newaxis_dims.append(y_axis)
            y_axis += 1

        elif isinstance(i, slice):
            # Normalize the slice to use None when possible
            start, stop, step = i.start, i.stop, i.step
            if step is None or step == 1:
                step = None
            if step is None:
                if start is None or start == 0:
                    start = None
                if stop is None or (
                    not isinstance(stop, Tracer) and (stop >= x_shape[x_axis])
                ):
                    stop = None
            elif step == -1:
                step = -1

            # Handle slice(None) and slice(None, None, -1)
            if (
                start is None
                and stop is None
                and (step is None or isinstance(step, int) and step == -1)
            ):
                if step == -1:
                    reversed_y_dims.append(collapsed_y_axis)
                slice_shape.append(x_shape[x_axis])
                gather_slice_shape.append(x_shape[x_axis])
                offset_dims.append(collapsed_y_axis)
                collapsed_y_axis += 1
                y_axis += 1
                x_axis += 1
            # Handle slice index (only static, otherwise an error is raised)
            else:
                if not all(
                    (elt == None or Tracer.get_aval(elt) is Array)
                    for elt in (start, stop, step)
                ):
                    msg = (
                        "Array slice indices must have static start/stop/step to be used "
                        "with NumPy indexing syntax. "
                        f"Found slice({start}, {stop}, {step}). "
                        "To index a statically sized "
                        "array at a dynamic position, try lax.dynamic_slice/"
                        "dynamic_update_slice (JAX does not support dynamically sized "
                        "arrays within JIT compiled functions)."
                    )
                    raise IndexError(msg)
                if not core.is_constant_dim(x_shape[x_axis]):
                    msg = (
                        "Cannot use NumPy slice indexing on an array dimension whose "
                        f"size is not statically known ({x_shape[x_axis]}). "
                        "Try using lax.dynamic_slice/dynamic_update_slice"
                    )
                    raise IndexError(msg)
                start, limit, stride, needs_rev = _static_idx(
                    slice(start, stop, step), x_shape[x_axis]
                )
                if needs_rev:
                    reversed_y_dims.append(collapsed_y_axis)
                if stride == 1:
                    i = start.astype(index_dtype)
                    gather_indices.append((i, len(gather_indices_shape)))
                    slice_shape.append(limit - start)
                    gather_slice_shape.append(limit - start)
                    offset_dims.append(collapsed_y_axis)
                    start_index_map.append(x_axis)
                else:
                    i = np.arange(start, limit, stride, dtype=index_dtype)
                    size = i.shape[0]
                    slice_shape.append(size)
                    gather_slice_shape.append(1)
                    gather_indices.append((i, len(gather_indices_shape)))
                    gather_indices_shape.append(size)

                    start_index_map.append(x_axis)
                    collapsed_slice_dims.append(x_axis)

                collapsed_y_axis += 1
                y_axis += 1
                x_axis += 1
        else:
            raise NotImplementedError

    if len(gather_indices) == 0:
        gather_indices_array = np.zeros((0,), dtype=index_dtype)
    elif len(gather_indices) == 1:
        g, _ = gather_indices[0]
        gather_indices_array = g.expand_dims((g.ndim,))
    else:
        last_dim = len(gather_indices_shape)
        gather_indices_shape.append(1)
        gather_indices_list = [
            g.broadcast(gather_indices_shape, tuple(range(i, i + g.ndim)))
            for g, i in gather_indices
        ]
        gather_indices_array = Tracer.concatenate(gather_indices_list, last_dim)

    return _Indexer(
        slice_shape=slice_shape,
        newaxis_dims=tuple(newaxis_dims),
        gather_slice_shape=gather_slice_shape,
        reversed_y_dims=reversed_y_dims,
        dnums=GatherDimensionNumbers(
            offset_dims=tuple(offset_dims),
            collapsed_slice_dims=tuple(sorted(collapsed_slice_dims)),
            start_index_map=tuple(start_index_map),
        ),
        gather_indices=gather_indices_array,
        unique_indices=advanced_indexes is None,
        indices_are_sorted=advanced_indexes is None,
    )


# def _is_boolean_index(i):
#   try:
#     abstract_i = core.get_aval(i)
#   except TypeError:
#     abstract_i = None
#   return (isinstance(abstract_i, ShapedArray) and issubdtype(abstract_i.dtype, bool_)
#           or isinstance(i, list) and i and all(_is_scalar(e)
#           and issubdtype(_dtype(e), np.bool_) for e in i))


# def _expand_bool_indices(idx, shape):
#     """Converts concrete bool indexes into advanced integer indexes."""
#     out = []
#     total_dims = len(shape)
#     num_ellipsis = sum(e is Ellipsis for e in idx)
#     if num_ellipsis > 1:
#         raise IndexError("an index can only have a single ellipsis ('...')")
#     elif num_ellipsis == 1:
#         total_dims = sum(e.ndim if _is_boolean_index(e) else 1 for e in idx
#                         if e is not None and e is not Ellipsis)
#     ellipsis_offset = 0
#     for dim_number, i in enumerate(idx):
#         try:
#             abstract_i = core.get_aval(i)
#         except TypeError:
#             abstract_i = None
#         if _is_boolean_index(i):
#             if isinstance(i, list):
#                 i = Array(i)
#             elif i.ndim == 0:
#                 raise TypeError("JAX arrays do not support boolean scalar indices")
#             else:
#                 i_shape = i.shape
#                 start = len(out) + ellipsis_offset
#                 expected_shape = shape[start: start + i.ndim]
#                 if i_shape != expected_shape:
#                     raise IndexError("boolean index did not match shape of indexed array in index "
#                                 f"{dim_number}: got {i_shape}, expected {expected_shape}")
#                 out.extend(np.where(i))
#         else:
#             out.append(i)
#         if i is Ellipsis:
#             ellipsis_offset = len(shape) - total_dims - 1
#     return tuple(out)