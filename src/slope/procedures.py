import slope
from slope.core import ProcedureSet, Tensor, dtypes
import math
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Sequence, Union, Iterator, NamedTuple, DefaultDict
from collections import defaultdict
import functools

max_ = max
abs_ = abs
min_ = min
sum_ = sum
procedure_set = ProcedureSet()


@procedure_set.register()
def zeros(*args, **kwargs):
    dtype = kwargs.get("dtype", slope.core.backend.DEFAULT_DTYPE)
    if kwargs.get("shape", None) is None:
        shape = args[0] if isinstance(args[0], (tuple, list)) else args
        assert all(i >= 0 for i in shape)
    return slope.full(shape, 0.0, dtype)


@procedure_set.register()
def ones(*args, **kwargs):
    dtype = kwargs.get("dtype", slope.core.backend.DEFAULT_DTYPE)
    if kwargs.get("shape", None) is None:
        shape = args[0] if isinstance(args[0], (tuple, list)) else args
        assert all(i >= 0 for i in shape)
    return slope.full(shape=shape, fill_value=1.0, dtype=dtype)


@procedure_set.register()
def full_like(y, fill_value):
    return slope.full(shape=y.shape, fill_value=fill_value, dtype=y.dtype)


@procedure_set.register()
def zeros_like(y):
    return full_like(y, 0.0)


@procedure_set.register()
def ones_like(y):
    return full_like(y, 1.0)


@procedure_set.register()
def eye(dim: int, **kwargs):
    return slope.ones((dim,1)).pad((0,dim,0,0)).flatten().padslice(((0,dim*dim),)).reshape(dim, dim)


@procedure_set.register()
def where(x, trueval, falseval):
    if not isinstance(trueval, Tensor):
        trueval = slope.full(trueval, device=x.device)
    if not isinstance(falseval, Tensor):
        falseval = slope.full(falseval, device=x.device)
    cond = x != x.zeros_like()
    if not any(val.dtype is dtypes.bool for val in (trueval, falseval)):
        cond = cond.cast(trueval.dtype)
        return cond * trueval + (cond.ones_like() - cond) * falseval
    else:
        cond = cond.cast(slope.float32)
        trueval = trueval.cast(slope.float32)
        falseval = falseval.cast(slope.float32)
        return (cond * trueval + (cond.ones_like() - cond) * falseval).cast(dtypes.bool)


@procedure_set.register()
def mean(x, dim=None, keepdim=False):
    out = x.sum(dim=dim, keepdim=keepdim)
    return out * (math.prod(out.shape) / math.prod(x.shape))


@procedure_set.register()
def rsqrt(x):
    return (x.ones_like() / x).sqrt()


@procedure_set.register()
def cos(x):
    return (x.full_like(math.pi / 2) - x).sin()


@procedure_set.register()
def tan(x):
    return x.sin() / x.cos()


@procedure_set.register()
def neg(x):
    return x.full_like(-1) * x


@procedure_set.register()
def not_equal(x, w):
    return ~(x == w)


@procedure_set.register()
def greater_equal(x, w):
    return ~(x < w)


@procedure_set.register()
def less_equal(x, w):
    return ~(x > w)


@procedure_set.register()
def minimum(x, w):
    return -x.maximum(-x, -w)


@procedure_set.register()
def min(x, dim=None, keepdim=False):
    return -((-x).max(x, dim, keepdim))


@procedure_set.register()
def argmax(x, dim=None, keepdim=False):
    if dim is None:
        ar = slope.arange(math.prod(x.shape) - 1, -1, -1, dtype=x.dtype)
        ar = ar.reshape(x.shape)
        idx = (x == x.max(dim)).cast(x.dtype) * ar
        return math.prod(x.shape) - idx.max().cast(slope.int32) - 1
    dim = dim + len(x.shape) if dim < 0 else dim
    m = (x == x.max(dim=dim, keepdim=True)).cast(x.dtype)
    idx = m * slope.arange(x.shape[dim] - 1, -1, -1, dtype=x.dtype)
    # idx = idx.reshape(x.shape[dim], *[1] * (x.ndim - dim - 1))
    ret = x.shape[dim] - idx.max(dim=dim, keepdim=keepdim) - 1
    return ret.cast(slope.int32)


@procedure_set.register()
def argmin(x, dim=None, keepdim=False):
    return (-x).argmax(dim=dim, keepdim=keepdim)


@procedure_set.register()
def log2(x):
    return x.log() / math.log(2)


@procedure_set.register()
@staticmethod
def _tri(r: int, c: int, k: int = 0, **kwargs) -> Tensor:
    return slope.arange(r, **kwargs).unsqueeze(1).expand(r, c) <= slope.arange(-k, c - k, **kwargs).unsqueeze(0).expand(
        r, c
    )


@procedure_set.register()
def triu(x, k: int = 0) -> Tensor:
    return _tri(x.shape[-2], x.shape[-1], k=k, dtype=x.dtype, device=x.device).where(x, slope.zeros_like(x))


@procedure_set.register()
def tril(x, k: int = 0) -> Tensor:
    return _tri(x.shape[-2], x.shape[-1], k=k + 1, dtype=x.dtype, device=x.device).where(slope.zeros_like(x), x)


@procedure_set.register()
def trunc(x: Tensor) -> Tensor:
    return x.cast(slope.int32).cast(x.dtype)


@procedure_set.register()
def ceil(x: Tensor) -> Tensor:
    return (x > (b := x.trunc())).where(b + 1, b)


@procedure_set.register()
def floor(x: Tensor) -> Tensor:
    return (x < (b := x.trunc())).where(b - 1, b)


@procedure_set.register()
def square(x):
    return x * x


@procedure_set.register()
def clip(x, min_, max_):
    return x.maximum(min_).minimum(max_)


@procedure_set.register()
def abs(x):
    return x.relu() + (-x).relu()


@procedure_set.register()
def sign(x):
    return x / (x.abs() + 1e-10)


@procedure_set.register()
def reciprocal(x):
    return x.ones_like() / x


@procedure_set.register()
def matmul(x, w):
    x = x.reshape((*x.shape[0:-1], 1, x.shape[-1]))
    w = w.reshape((*w.shape[0:-2], 1, w.shape[-2], w.shape[-1])).transpose(-1, -2)
    return (x * w).sum(-1).reshape((*x.shape[0:-2], -1))


@procedure_set.register()
def getitem(x, indices) -> Tensor:
    # 1. indices normalization and validation
    # treat internal tuples and lists as Tensors and standardize indices to list type
    if isinstance(indices, list) and all(isinstance(s, int) for s in indices):
        indices = [
            slope.tensor(
                indices,
                x.device,
            )
        ]
    elif isinstance(indices, (tuple, list)):
        indices = [slope.tensor(list(i), x.device) if isinstance(i, (tuple, list)) else i for i in indices]
    else:
        indices = [indices]

    # filter ellipsis and fill with slice(None) or fill rest of indices with slice(None)
    ellipsis_idx = [dim for dim, i in enumerate(indices) if i is Ellipsis]
    fill_idx = ellipsis_idx[0] if ellipsis_idx else len(indices)
    num_indices = len(indices) - len(ellipsis_idx) - sum(1 for i in indices if i is None)
    indices[fill_idx : fill_idx + 1] = [slice(None)] * (len(x.shape) - num_indices)

    # use Dict[type, List[dimension]] to track elements in indices
    type_dim: DefaultDict[Union[type, None], List[int]] = defaultdict(list)

    # record None for dimension injection later and filter None and record rest of indices
    type_dim[None] = [dim for dim, i in enumerate(indices) if i is None]
    indices_filtered = [v for v in indices if v is not None]
    for dim, i in enumerate(indices_filtered):
        type_dim[type(i)].append(dim)

    for index_type in type_dim:
        if index_type not in [None, int, slice, Tensor]:
            raise IndexError(f"{index_type=} not supported")
    if len(ellipsis_idx) > 1:
        raise IndexError("indices can only have a single ellipsis ('...')")
    if num_indices > x.ndim:
        raise IndexError(f"too many {num_indices=} for {x.ndim=}")

    # 2. basic indexing, uses only movement ops (no copy)
    # currently indices_filtered: Tuple[Union[slice, int, Tensor], ...]
    # turn indices in indices_filtered to Tuple[shrink_arg, strides]
    for dim in type_dim[int]:
        if (index := indices_filtered[dim]) >= (size := x.shape[dim]) or index < -size:
            raise IndexError(f"{index=} is out of bounds on {dim=} with {size=}")
        indices_filtered[dim] = ((index, index + 1), 1) if index >= 0 else ((size + index, size + index + 1), 1)
    for dim in type_dim[slice]:
        if (index := indices_filtered[dim]).step == 0:
            raise ValueError(f"{index=} on {dim=} cannot have 0 as step")
        s, e, st = index.indices(x.shape[dim])
        indices_filtered[dim] = ((0, 0) if (st * (e - s)) < 0 else (s, e) if st > 0 else (e + 1, s + 1), st)
    # record tensors and skip all Tensor dims for basic indexing
    tensor_index: List[Tensor] = []
    for dim in type_dim[Tensor]:
        tensor_index.append(index := indices_filtered[dim])
        if not dtypes.is_int(index.dtype):
            raise IndexError(f"{index.dtype=} on {dim=} is not supported, only int tensor indexing is supported")
        indices_filtered[dim] = ((0, x.shape[dim]), 1)

    new_slice, strides = ((), ()) if not indices_filtered else zip(*indices_filtered)
    ret = x.padslice(new_slice).flip(tuple(i for i, s in enumerate(strides) if s < 0))
    if any(abs_(s) != 1 for s in strides):
        strides = tuple(abs_(s) for s in strides)
        round_up = lambda num, amt: (num + amt - 1) // amt * amt
        ret = ret.pad(tuple((0, round_up(sh, s) - sh) for s, sh in zip(strides, ret.shape)))
        ret = ret.reshape(tuple(flatten((sh // s, s) for s, sh in zip(strides, ret.shape))))
        ret = ret.padslice(tuple(flatten(((0, sh), (0, 1)) for sh in ret.shape[::2]))).reshape(ret.shape[::2])

    # inject 1 for dim where it's None and collapse dim for int
    new_shape = list(ret.shape)
    for dim in type_dim[None]:
        new_shape.insert(dim, 1)
    for dim in (
        tuple(dim + sum_(1 for d in type_dim[None] if dim >= d) for dim in reversed(type_dim[int]))
        # dims_collapsed := tuple(dim + sum_(1 for d in type_dim[None] if dim >= d) for dim in reversed(type_dim[int]))
    ):
        new_shape.pop(dim)

    ret = ret.reshape(new_shape)

    # 3. advanced indexing (copy)
    if type_dim[Tensor]:
        for i in tensor_index:
            while i.ndim < ret.ndim:
                i = i[None]
            ret = ret.gather_nd(i)
    return ret
    ## impl like tinygrad tensor __getitem__:      
    # if type_dim[Tensor]:
        # # calculate dim of current ret by subtracting dims collapsed and adding dims injected up until tensor_dim
        # def calc_dim(tensor_dim: int) -> int:
        #     return (
        #         tensor_dim
        #         - sum_(1 for d in dims_collapsed if tensor_dim >= d)
        #         + sum_(1 for d in type_dim[None] if tensor_dim >= d)
        #     )

        # # track tensor_dim and tensor_index using a dict
        # # calc_dim to get dim and use that to normalize the negative tensor indices
        # idx: Dict[int, Tensor] = {
        #     (dim := calc_dim(td)): (t < 0).where(t.full_like(ret.shape[dim]), t.zeros_like()) + t
        #     for td, t in zip(type_dim[Tensor], tensor_index)
        # }
        # # compute sum_dim, arange, and idx
        # max_idx_dim, first_dim, last_dim = max_(i.ndim for i in idx.values()), min_(idx.keys()), max_(idx.keys())
        # sum_dim = tuple(d if n == 0 else d + max_idx_dim - n for n, d in enumerate(idx.keys()))
        # arange = [
        #     slope.arange(ret.shape[d], device=x.device).reshape(
        #         ret.shape[d : d + 1] + (1,) * (ret.ndim + max_idx_dim - n - sd - 1)
        #     )
        #     for n, (sd, d) in enumerate(zip(sum_dim, idx.keys()))
        # ]  # noqa: E501
        # reshaped_idx = [
        #     i.reshape(i.shape + (1,) * (ret.ndim - first_dim - (n or 1))) for n, i in enumerate(idx.values())
        # ]
        # ret_ = ret
        # ret = ret.reshape(ret.shape[: first_dim + 1] + (1,) * max_idx_dim + ret.shape[first_dim + 1 :])

        # for a, i, sd in zip(arange, reshaped_idx, sum_dim):
        #     ret = (a == i).cast(ret.dtype).mul(ret).sum(sd)

        # # special permute case
        # if first_dim != 0 and len(idx) != 1 and tuple(idx.keys()) != tuple(range(first_dim, last_dim + 1)):
        #     ret_dims = list(range(ret.ndim))
        #     ret = ret.permute(
        #         ret_dims[first_dim : first_dim + max_idx_dim]
        #         + ret_dims[:first_dim]
        #         + ret_dims[first_dim + max_idx_dim :]
        #     )


@procedure_set.register()
def padslice(x, arg: Sequence[Optional[Tuple[int, int]]], value: float = 0):
    def flatten_seq(l: Iterator):
        return tuple(item for sublist in l for item in sublist)

    # some dim are pad, some are sliced
    arg_ = tuple([a if a is not None else (0, s) for s, a in zip(x.shape, arg)])
    padding = tuple([(max(0, -p[0]), max(0, p[1] - x.shape[i])) for i, p in enumerate(arg_)])
    x = x.pad(flatten_seq(padding)[::-1], value=value)  # flatten
    starts, limits, strides = tuple(zip(*[(p[0] + padding[i][0], p[1] + padding[i][0], 1) for i, p in enumerate(arg_)]))
    x = x.slice(starts, limits, strides)
    return x


@procedure_set.register()
def pad2d(x, padding: Union[List[int], Tuple[int, ...]], value: float = 0):
    # (padding_left, padding_right, padding_top, padding_bottom)
    slc = [(-p0, s + p1) for p0, p1, s in zip(padding[::2], padding[1::2], x.shape[::-1])][::-1]
    return x.padslice([(0, s) for s in x.shape[: -(len(padding) // 2)]] + slc, value=value)


@procedure_set.register()
def gather(x, idx, dim: int):
    assert idx.ndim == x.ndim, "x.ndim must equal idx.ndim"
    assert all(s >= i for s, i in zip(x.shape, idx.shape)), "all dim of idx.shape must be smaller than x.shape"
    if dim < 0:
        dim += x.ndim
    idx = idx.transpose(ax=dim, aw=0).expand_dims(-1)
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
            * x.permute(*permarg)
            .padslice(tuple([*[(0, sh) for sh in idx.shape[1:-1]], (0, x.shape[dim])]))
            .expand_dims(0)
        )
        .sum(-1)
        .transpose(ax=0, aw=dim)
    )


@procedure_set.register()
@staticmethod
def stack(tensors, dim=0):
    first = tensors[0].expand_dims(dim)
    expand_dimsd_tensors = [tensor.expand_dims(dim) for tensor in tensors[1:]]
    return first.cat(*expand_dimsd_tensors, dim=dim)


@procedure_set.register()
def repeat(x, repeats):
    base_shape = (1,) * (len(repeats) - x.ndim) + x.shape
    new_shape = [x for b in base_shape for x in [1, b]]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r * s for r, s in zip(repeats, base_shape)]
    return x.reshape(new_shape).broadcast(expand_shape).reshape(final_shape)


@procedure_set.register()
def split(x, num: int, dim: int):
    dim, step = dim + x.ndim if dim < 0 else dim, math.ceil(x.shape[dim] / num)
    slice_params = [[slice(None)] * dim + [slice(k, k + step)] for k in range(0, x.shape[dim], step)]
    return tuple(x[tuple(sl)] for sl in slice_params)


@procedure_set.register()
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


@procedure_set.register(aliases=("expand_dims",))
def unsqueeze(x, dim) -> Tensor:
    if dim < 0:
        dim = len(x.shape) + dim + 1
    return x.reshape(x.shape[:dim] + (1,) + x.shape[dim:])


@procedure_set.register()
def transpose(x, ax=1, aw=0):
    order = list(range(len(x.shape)))
    order[ax], order[aw] = order[aw], order[ax]
    return x.permute(tuple(order))


@procedure_set.register()
def flatten(x, start_dim=0):
    return x.reshape(shape=x.shape[:start_dim] + (-1,))


@procedure_set.register()
def cumsum(x, dim: int = 0):
    return x.transpose(dim, -1).pad((x.shape[dim] - 1, 0)).pool((x.shape[dim],)).sum(-1).transpose(dim, -1)


@staticmethod
@procedure_set.register()
def arange_with_cumsum(start, stop=None, step=1):
    if stop is None:
        stop, start = start, 0
    return slope.full((math.ceil((stop - start) / step),), step).cumsum() + (start - step)


@procedure_set.register()
def one_hot(x, k, dtype=dtypes.int32):
    return (x[:, None].cast(dtype) == slope.arange(k, dtype=dtype)).cast(dtype)


@procedure_set.register()
def relu(x):
    return x.maximum(slope.zeros_like(x))


@procedure_set.register()
def leakyrelu(x, neg_slope=0.01):
    return x.relu() - (slope.full_like(x, -neg_slope) * x).relu()


@procedure_set.register()
def sigmoid(x):
    return 1 / (1 + (-x).exp())


@procedure_set.register()
def tanh(x):
    return 2.0 * ((2.0 * x).sigmoid()) - 1.0


@procedure_set.register()
def swish(x):
    return x * x.sigmoid()


@procedure_set.register()
def gelu(x):
    return 0.5 * x * (1 + (x * 0.7978845608 * (1 + 0.044715 * x * x)).tanh())


@procedure_set.register()
def linear(x, w, b=None):
    x = x @ w.transpose(-2, -1)
    return x + b[None, ...] if b is not None else x


@procedure_set.register()
def sequential(x, modules, *args, **kwargs):
    for module in modules:
        x = module(x, *args, **kwargs)
    return x


@procedure_set.register()
def pool(
    x,
    kernel_size: Tuple[int, ...],
    stride: Union[Tuple[int, ...], int] = 1,
    dilation: Union[Tuple[int, ...], int] = 1,
):
    def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
        return (x,) * cnt if isinstance(x, int) else x

    def flatten_seq(l):
        return [item for sublist in l for item in sublist]

    k_ = kernel_size
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
        xup = xup.expand((*prefix, *flatten_seq((e, i) for e, i in zip(e_, i_))))
        xup = xup.reshape((*prefix, *[e * i for e, i in zip(e_, i_)]))
        # slide by dilation
        xup = xup.padslice(slc_prefix + [(0, k * (i + d)) for k, i, d in zip(k_, i_, d_)])
        xup = xup.reshape((*prefix, *flatten_seq((k, i + d) for k, i, d in zip(k_, i_, d_))))
        xup = xup.padslice(slc_prefix + flatten_seq(((0, k), (0, o * s)) for k, o, s in zip(k_, o_, s_)))
        # handle stride, and permute to move reduce to the end
        xup = xup.reshape((*prefix, *flatten_seq((k, o, s) for k, o, s in zip(k_, o_, s_))))
        xup = xup.padslice(slc_prefix + flatten_seq(((0, k), (0, o), (0, 1)) for k, o in zip(k_, o_)))
        xup = xup.reshape((*prefix, *flatten_seq((k, o) for k, o in zip(k_, o_))))
        return xup.permute(
            (
                *range(len(prefix)),
                *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
                *[len(prefix) + i * 2 for i in range(len(k_))],
            )
        )
    o_ = [(i + (s - k)) // s for i, s, k in zip(i_, s_, k_)]
    xup = x.padslice(slc_prefix + [(0, o * s) for o, s in zip(o_, s_)])
    xup = xup.reshape((*prefix, *flatten_seq(((o, s) for o, s in zip(o_, s_)))))
    xup = xup.padslice((slc_prefix + flatten_seq(((0, o), (0, k)) for o, k in zip(o_, k_))))
    return xup.permute(
        (
            *range(len(prefix)),
            *[len(prefix) + i * 2 for i in range(len(k_))],
            *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
        )
    )


@procedure_set.register()
def avgpool2d(x, kernel_size=(2, 2), stride=None):
    def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
        return (x,) * cnt if isinstance(x, int) else x

    return x.pool(make_pair(kernel_size), stride if stride is not None else kernel_size).mean(
        dim=tuple(range(0 - len(make_pair(kernel_size)), 0))
    )


@procedure_set.register()
def maxpool2d(x, kernel_size=(2, 2), stride=None, dilation=1):
    def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
        return (x,) * cnt if isinstance(x, int) else x

    return x.pool(
        make_pair(kernel_size),
        stride if stride is not None else kernel_size,
        dilation,
    ).max(dim=tuple(range(0 - len(make_pair(kernel_size)), 0)))


@procedure_set.register()
def conv(x, w, groups=1, stride=1, dilation=1, padding=0):
    (bs, cin_), (cout, cin), D = x.shape[:2], w.shape[:2], w.shape[2:]
    assert groups * cin == cin_ and len(x.shape) == len(
        w.shape
    ), f"Input dim shape {x.shape} does not match the shape of the ws {w.shape}. ({groups*cin} vs. {cin_})"
    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(D) or len(padding) == len(
            D
        ), f"Expected padding of length {2*len(D)} or {len(D)}, but got {len(padding)} for tensor of shape {x.shape}"
    padding_ = (
        [padding] * 2 * len(D)
        if isinstance(padding, int)
        else (padding if len(padding) == 2 * len(D) else [p for p in padding for _ in range(2)][::-1])
    )
    padding_ = tuple(padding_)

    def pad2d(x, padding: Union[List[int], Tuple[int, ...]], value: float = 0):
        # (padding_left, padding_right, padding_top, padding_bottom)
        slc = [(-p0, s + p1) for p0, p1, s in zip(padding[::2], padding[1::2], x.shape[::-1])][::-1]
        return x.padslice([(0, s) for s in x.shape[: -(len(padding) // 2)]] + slc, value=value)

    x = pad2d(x, padding_)
    x = x.pool(D, stride, dilation)  # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout // groups, x.shape[2 : -len(D)]
    x = x.reshape((bs, groups, cin, 1, *oyx, *D))
    x = x.expand((bs, groups, cin, rcout, *oyx, *D))
    x = x.permute(
        (
            0,
            1,
            3,
            *[4 + i for i in range(len(oyx))],
            2,
            *[4 + len(oyx) + i for i in range(len(D))],
        )
    )
    # (bs, groups, rcout, *oyx, cin, *D)
    x = x * w.reshape((1, groups, rcout, *[1] * len(oyx), cin, *D))
    x = x.sum([-1 - i for i in range(1 + len(oyx))], keepdim=True)
    x = x.reshape((bs, cout, *oyx))
    ret = x
    return ret


@procedure_set.register()
def conv_transpose(x, w, groups=1, stride=1, dilation=1, padding=0, output_padding=0):
    make_pair = lambda x, cnt=2: (x,) * cnt if isinstance(x, int) else x
    flatten_seq = lambda l: [item for sublist in l for item in sublist]
    D, trailing = w.shape[2:], list(range(3, len(w.shape) + 1))
    w = w.reshape(((groups, w.shape[0] // groups, w.shape[1], *w.shape[2:])))
    w = w.permute((0, 2, 1, *trailing)).flip(trailing)
    stride = make_pair(stride, len(D))
    if any(s > 1 for s in stride):
        x = x.reshape((*x.shape[:2], *flatten_seq((k, 1) for k in x.shape[2:])))
        pads = (0, 0, 0, 0, *flatten_seq((0, 0, 0, s - 1) for s in stride))
        pads = pads[::-1]
        x = x.pad(pads)
        x = x.reshape(*x.shape[:2], *[k * s for k, s in zip(x.shape[2::2], stride)])
        x = x.slice(
            (0, 0) + (0,) * len(x.shape[2:]),
            (x.shape[0], x.shape[1]) + tuple([k - (s - 1) for k, s in zip(x.shape[2:], stride)])[::-1],
        )
    padding = flatten_seq(
        (
            ((k - 1) * d - p, (k - 1) * d - p + op)
            for k, d, p, op in reversed(
                list(
                    zip(
                        D,
                        make_pair(dilation, len(D)),
                        make_pair(padding, len(D)),
                        make_pair(output_padding, len(D)),
                    )
                )
            )
        )
    )
    w = w.reshape((w.shape[0] * w.shape[1], *w.shape[2:]))
    return x.conv(w, groups=groups, dilation=dilation, padding=padding)


@procedure_set.register()
def batchnorm(x, weight, bias, mean, invstd):
    broadcast_shape = (1, -1) + (1,) * len(x.shape[2:])
    x = (x - mean.reshape(broadcast_shape)) * invstd.reshape(broadcast_shape)
    if weight is not None and bias is not None:
        x = x * weight.reshape(broadcast_shape) + bias.reshape(broadcast_shape)
    return x


@procedure_set.register()
def layernorm(x, dim=-1, eps: float = 1e-5) -> Tensor:
    y = x - x.mean(dim, keepdim=True)
    return y.mul((y * y).mean(dim, keepdim=True).add(eps).rsqrt())


@procedure_set.register()
def dropout(x, p, training=False) -> Tensor:
    if not training or p == 0:
        return x
    mask = (slope.rand(*x.shape, requires_grad=False, device=x.device) >= p).cast(slope.bool)
    return x * mask * (1 / (1.0 - p))


@procedure_set.register()
def scaled_dot_product_attention(
    x,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> Tensor:
    if is_causal:
        attn_mask = (
            slope.ones(x.shape[-2], key.shape[-2], requires_grad=False, device=x.device).tril(0).cast(slope.bool)
        )
    if attn_mask is not None and attn_mask.dtype == slope.bool:
        attn_mask = (attn_mask == 0).where(-float("inf"), attn_mask)
    return (x @ key.transpose(-2, -1) / math.sqrt(x.shape[-1]) + attn_mask).softmax(-1).dropout(dropout_p) @ value


@procedure_set.register()
def binary_cross_entropy(x, y: Tensor) -> Tensor:
    return (-y * x.log() - (1 - y) * (1 - x).log()).mean()


@procedure_set.register()
def binary_cross_entropy_with_logits(x, y: Tensor) -> Tensor:
    return (x.maximum(0) - y * x + (1 + x.abs().__neg__().exp()).log()).mean()


@procedure_set.register()
def cross_entropy(x, y, ignore_index=-1) -> Tensor:
    # loss_mask = (y != ignore_index).reshape(-1, 1)
    y_counter = slope.arange(x.shape[-1], dtype=slope.int32)[None, ..., None]
    y_oh = (y_counter == y[..., None, None]).where(-1.0, 0.0).squeeze(-1)
    # y = y * loss_mask
    return (x.log_softmax(-1) * y_oh).sum()  # / loss_mask.sum()


@procedure_set.register()
def softmax(x, dim=-1):
    m = x - x.max(dim, keepdim=True)
    e = m.exp()
    ss = e.sum(dim, keepdim=True)
    return e / ss


@procedure_set.register()
def log_softmax(x, dim=-1):
    x = x - x.max(dim, keepdim=True)
    logsumexp_x = x.exp().sum(dim, keepdim=True).log()
    return x - logsumexp_x


# @procedure_set.register()
# def gather_nd(
#     params,
#     indices,
#     batch_dims=0):
#     def _gather_nd_single(params, indices):
#         idx = indices.moveaxis(-1, 0)
#         return params[idx]
#     assert batch_dims > 0, ('Negative `batch_dims` is currently unsupported.')
#     assert batch_dims == 0
#     gather_nd_ = functools.reduce(
#         lambda g, f: f(g), [slope.vmap] * int(batch_dims),
#         _gather_nd_single
#         ) if batch_dims > 0 else _gather_nd_single
#     return gather_nd_(params, indices)

# TODO:

# nograd_functions = [
#     np.all,
#     np.allclose,
#     np.any,
#     #np.argmax,
#     np.argmin,
#     np.argpartition,
#     np.argsort,
#     np.argwhere,
#     np.around,
#     np.array_equal,
#     np.array_equiv,
#     np.ceil,
#     np.count_nonzero,
#     #np.equal,
#     np.fix,
#     np.flatnonzero,
#     np.floor,
#     np.floor_divide,
#     np.greater,
#     np.greater_equal,
#     np.isclose,
#     np.isfinite,
#     np.isinf,
#     np.isnan,
#     np.isneginf,
#     np.isposinf,
#     np.isscalar,
#     np.less,
#     np.less_equal,
#     np.logical_and,
#     np.logical_not,
#     np.logical_or,
#     np.logical_xor,
#     np.ndim,
#     np.nonzero,
#     np.not_equal,
#     np.ones_like,
#     np.result_type,
#     np.rint,
#     np.round,
#     np.searchsorted,
#     np.shape,
#     np.sign,
#     np.size,
#     np.trunc,
#     np.zeros_like,
# ]
