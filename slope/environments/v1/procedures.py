import slope
from slope import environment as sev
from slope.core import ProcedureSet, BaseArray
from slope.environments.v1.operators import operator_set
import math
from typing import Tuple, Union, List, Iterator, Optional, Sequence
import itertools
import functools
import operator
import math
from collections import defaultdict

procedure_set = ProcedureSet()


# Utils
def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
    return (x,) * cnt if isinstance(x, int) else x


def flatten_seq(l: Iterator):
    return [item for sublist in l for item in sublist]


@procedure_set.register
def zeros(shape, dtype=BaseArray.float32):
    return sev.full(shape, 0.0, dtype)


@procedure_set.register
def ones(shape, dtype=BaseArray.float32):
    return sev.full(shape=shape, fill_value=1.0, dtype=dtype)


@procedure_set.register
def full_like(y, fill_value):
    return sev.full(shape=y.shape, fill_value=fill_value, dtype=y.dtype)


@procedure_set.register
def zeros_like(y):
    return zeros(shape=y.shape, dtype=y.dtype)


@procedure_set.register
def ones_like(y):
    return sev.full(shape=y.shape, fill_value=1.0, dtype=y.dtype)


@procedure_set.register
def where(x, trurun, falsrun):
    cond = x != 0.0
    cond = cond.convert(trurun.dtype)  # TODO: type promotion logic
    return cond * trurun + (1.0 - cond) * falsrun


@procedure_set.register
def mean(x, axes=None, keepdims=False):
    out = x.sum(axes=axes, keepdim=keepdims)
    return out * (math.prod(out.shape) / math.prod(x.shape))

@procedure_set.register
def rsqrt(self): return (1/self).sqrt()
@procedure_set.register
def cos(self): return ((math.pi/2)-self).sin()
@procedure_set.register
def tan(self): return self.sin() / self.cos()

@procedure_set.register
def minimum(x, y):
    return -x.maximum(-x, -y)


@procedure_set.register
def min(x, axes=None, keepdims=False):
    return -((-x).max(x, axes, keepdims))


@procedure_set.register
def argmax(self, axis=None, keepdim=False):
    if axis is None:
        idx = (self == self.max(axis)) * sev.arange(
            math.prod(self.shape) - 1,
            -1,
            -1,
            dtype=slope.int32,
        ).reshape(self.shape)
        return math.prod(self.shape) - idx.max() - 1
    axis = axis + len(self.shape) if axis < 0 else axis
    m = self == self.max(axis=axis, keepdim=True)
    idx = m * sev.arange(self.shape[axis] - 1, -1, -1, dtype=slope.int32).reshape(
        self.shape[axis], *[1] * (self.ndim - axis - 1)
    )
    return self.shape[axis] - idx.max(axis=axis, keepdim=keepdim) - 1


@procedure_set.register
def argmin(self, axis=None, keepdim=False):
    return (-self).argmax(axis=axis, keepdim=keepdim)


@procedure_set.register
def T(x):
    perm = list(range(x.ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return x.transpose(perm)


@procedure_set.register
def _softmax(x, axes):
    m = x - x.max(axes, keepdims=True)
    e = m.exp()
    return m, e, e.sum(axes, keepdims=True)


@procedure_set.register
def softmax(x, axes=-1):
    _, e, ss = x._softmax(axes)
    return e.div(ss)


@procedure_set.register
def log_softmax(x, axes=-1):
    m, _, ss = x._softmax(axes)
    return m - ss.log()


@procedure_set.register
def dot(x, w):
    x = x.reshape((*x.shape[0:-1], 1, x.shape[-1]))
    w = w.reshape((*w.shape[0:-2], 1, w.shape[-2], w.shape[-1])).T()
    return (x * w).sum(-1).reshape((*x.shape[0:-2], -1))


@procedure_set.register
def __getitem__(
    x, val
):  # val: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
    def normalize_int(e, i, dim_sz):
        if -dim_sz <= e < dim_sz:
            return e if e != -1 else dim_sz - 1
        raise IndexError(
            f"index {e} is out of bounds for dimension {i} with size {x.shape[i]}"
        )

    orig_slices = list(val) if isinstance(val, tuple) else [val]
    count = defaultdict(list)
    for i, v in enumerate(orig_slices):
        count[type(v)].append(i)

    if (num_slices := len(count[int]) + len(count[slice]) + len(count)) > len(x.shape):
        raise IndexError(f"too many indices for tensor of dimension {len(x.shape)}")
    if len(ellipsis_found := count[type(Ellipsis)]) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
    orig_slices[ellipsis_idx : ellipsis_idx + 1] = [slice(None)] * (
        len(x.shape) - num_slices
    )

    valid_slices = [v for v in orig_slices if v is not None]
    valid_slices = [
        v
        if isinstance(v, slice)
        else slice(y_ := normalize_int(v, i, dim_sz), y_ + 1)
        if isinstance(v, int)
        else slice(None)
        for i, (v, dim_sz) in enumerate(zip(valid_slices, x.shape))
    ]

    start, stop, strides = (
        zip(*y)
        if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, x.shape)])
        else ((), (), ())
    )
    new_slice = tuple(
        (s, e) if st > 0 else (e + 1, s + 1) for s, e, st in zip(start, stop, strides)
    )
    sliced_tensor = x.slice(new_slice).flip(
        axis=[i for i, s in enumerate(strides) if s < 0]
    )
    new_shape = sliced_tensor.shape
    if any(abs(s) != 1 for s in strides):
        strides = tuple(abs(s) for s in strides)
        # Pad: add pad at the end: [dim_sz] -> [dim_sz_padded]
        padded_tensor = sliced_tensor.pad(
            tuple(
                (0, s - (dim_sz % s) if dim_sz % s != 0 else 0)
                for s, dim_sz in zip(strides, sliced_tensor.shape)
            )
        )
        # Reshape: [dim_sz_padded] -> [dim_sz_padded // s, s]
        reshaped_tensor = padded_tensor.reshape(
            flatten([sh // s, s] for sh, s in zip(padded_tensor.shape, strides))
        )
        new_shape = reshaped_tensor.shape[::2]
        # Shrink: do [:, 0]
        sliced_tensor = reshaped_tensor.slice(
            tuple(flatten(((0, sh), (0, 1)) for sh in new_shape))
        )

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
                if isinstance(s, Tensor):
                    tensors.append(s)
                    dim.append(i - dim_collapsed)
    ret = sliced_tensor.reshape(tuple(final_shape))

    if tensors:  # Fancy/tensor indexing
        # normalize idx
        idx = [
            t.sign().contiguous().__neg__().contiguous().relu() * ret.shape[d] + t
            for d, t in zip(dim, tensors)
        ]  # TODO first contiguous fixes torch+cpu_only CI, but it causes llvm to fail. Second one fixes llvm
        max_dim = max(i.ndim for i in idx)
        # compute sum_dim, arange, and idx
        sum_dim = [d if n == 0 else d + max_dim - n for n, d in enumerate(dim)]
        arange = [
            sev.arange(
                ret.shape[d],
                dtype=slope.int32,
                requires_grad=False,
                device=x.device,
            ).reshape(*[1] * sd, ret.shape[d], *[1] * (ret.ndim + max_dim - n - sd - 1))
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
        ret = ret.reshape(
            *ret.shape[: sum_dim[0] + 1], *[1] * max_dim, *ret.shape[sum_dim[0] + 1 :]
        )
        # iteratively fancy index
        for a, i, sd in zip(arange, idx, sum_dim):
            ret = (a == i).mul(ret).sum(sd)
        # special permute case
        if dim[0] != 0 and len(dim) != 1 and dim != list(range(dim[0], dim[-1] + 1)):
            ret_dims = list(range(ret.ndim))
            ret = ret.transpose(
                ret_dims[dim[0] : dim[0] + max_dim]
                + ret_dims[: dim[0]]
                + ret_dims[dim[0] + max_dim :]
            )
    return ret


@procedure_set.register
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


@procedure_set.register
def slice(x, arg):
    assert all(2 <= len(a) <= 3 for a in arg)
    arg = tuple((*a, 1) if len(a) == 2 else a for a in arg)
    starts, limits, strides = tuple(zip(*arg))
    return x.slice_hlo(starts, limits, strides)


@procedure_set.register
def padslice(x, arg: Sequence[Optional[Tuple[int, int]]], value: float = 0):
    arg_ = tuple([a if a is not None else (0, s) for s, a in zip(x.shape, arg)])
    padding = tuple(
        [(max(0, -p[0]), max(0, p[1] - x.shape[i])) for i, p in enumerate(arg_)]
    )
    x = x.pad(padding, constant_values=value)
    slc = tuple(
        [(p[0] + padding[i][0], p[1] + padding[i][0]) for i, p in enumerate(arg_)]
    )
    x = x.slice(slc)
    return x


@procedure_set.register
def gather(x, idx, dim: int):
    assert idx.ndim == x.ndim, "x.ndim must equal idx.ndim"
    assert all(
        s >= i for s, i in zip(x.shape, idx.shape)
    ), "all dim of idx.shape must be smaller than x.shape"
    if dim < 0:
        dim += x.ndim
    idx = idx.swapaxes(ax1=dim, ax2=0).expand_dims(-1)
    permarg = list(range(x.ndim))
    permarg = (
        permarg[1:dim] + [permarg[0]] + permarg[dim + 1 :] + [permarg[dim]]
        if dim != 0
        else permarg[1:] + [permarg[0]]
    )
    return (
        (
            (
                idx
                == sev.arange(
                    x.shape[dim],
                    dtype=slope.int32,
                    requires_grad=False,
                    device=x.device,
                )
            )
            * x.transpose(*permarg)
            .slice(tuple([*[(0, sh) for sh in idx.shape[1:-1]], (0, x.shape[dim])]))
            .expand_dims(0)
        )
        .sum(-1)
        .swapaxes(ax1=0, ax2=dim)
    )


@procedure_set.register
@staticmethod
def stack(tensors, dim=0):
    first = tensors[0].expand_dims(dim)
    expand_dimsd_tensors = [tensor.expand_dims(dim) for tensor in tensors[1:]]
    # checks for shapes and number of dimensions delegated to cat
    return first.concatenate(*expand_dimsd_tensors, dim=dim)


@procedure_set.register
def repeat(x, repeats):
    base_shape = (1,) * (len(repeats) - x.ndim) + x.shape
    new_shape = [x for b in base_shape for x in [1, b]]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r * s for r, s in zip(repeats, base_shape)]
    return x.reshape(new_shape).broadcast(expand_shape).reshape(final_shape)


@procedure_set.register
def split(x, num: int, dim: int):
    dim, step = dim + x.ndim if dim < 0 else dim, math.ceil(x.shape[dim] / num)
    slice_params = [
        [slice(None)] * dim + [slice(k, k + step)] for k in range(0, x.shape[dim], step)
    ]
    return [x[tuple(sl)] for sl in slice_params]


@procedure_set.register
def squeeze(x, dim=None):
    if dim is None:
        return (
            x
            if 1 not in x.shape
            else x.reshape(*[size for size in x.shape if size != 1])
        )
    if dim <= 0 and x.ndim == 0:
        return x  # This is to match PyTorch behavior
    if not -x.ndim <= dim < x.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-x.ndim if x.ndim > 0 else x.ndim-1}, {x.ndim-1 if x.ndim > 0 else x.ndim}], but got {dim})"
        )
    if dim < 0:
        dim += x.ndim
    return (
        x
        if x.shape[dim] != 1
        else x.reshape(*[size for idx, size in enumerate(x.shape) if idx != dim])
    )


@procedure_set.register
def expand_dims(x, dim):
    if dim < 0:
        dim = len(x.shape) + dim + 1
    return x.reshape(x.shape[:dim] + (1,) + x.shape[dim:])


# (padding_left, padding_right, paddingp, padding_bottom)
# @procedure_set.register
# def pad2d(x, padding: Union[List[int], Tuple[int, ...]], value: float = 0):
#     slc = [
#         (-p0, s + p1)
#         for p0, p1, s in zip(padding[::2], padding[1::2], x.shape[::-1])
#     ][::-1]
#     slc_ = [(0, s) for s in x.shape[: -(len(padding) // 2)]] + slc
#     ret = x.padslice(slc_, value=value)
#     return ret


@procedure_set.register
def swapaxes(x, ax1=1, ax2=0):
    order = list(range(len(x.shape)))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return x.transpose(order)


@procedure_set.register
def flatten(x, start_dim=0):
    return x.reshape(shape=x.shape[:start_dim] + (-1,))


@procedure_set.register
def broadcast_to(x, shape):
    return x.broadcast_in_dim(shape=shape, axes=None)


@procedure_set.register
def _pool(
    x,
    k_: Tuple[int, ...],
    stride: Union[Tuple[int, ...], int] = 1,
    dilation: Union[Tuple[int, ...], int] = 1,
):
    assert len(x.shape) >= len(k_), f"can't pool {x.shape} with {k_}"
    s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
    assert len(k_) == len(s_) and len(k_) == len(
        d_
    ), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    slc_prefix, prefix, i_ = (
        [(0, x) for x in x.shape[0 : -len(k_)]],
        x.shape[0 : -len(k_)],
        x.shape[-len(k_) :],
    )
    if any(k > s for k, s in zip(k_, s_)) or any(d != 1 for d in d_):
        o_ = [(i - d * (k - 1) - 1) // s + 1 for i, d, k, s in zip(i_, d_, k_, s_)]
        e_ = [
            math.ceil(k * (i + d) / i) for k, i, d in zip(k_, i_, d_)
        ]  # expands such that we don't need padding
        xup = x
        xup = xup.reshape((*prefix, *flatten_seq((1, i) for i in i_)))
        xup = xup.broadcast((*prefix, *flatten_seq((e, i) for e, i in zip(e_, i_))))
        xup = xup.reshape((*prefix, *[e * i for e, i in zip(e_, i_)]))
        # slide by dilation
        xup = xup.padslice(
            slc_prefix + [(0, k * (i + d)) for k, i, d in zip(k_, i_, d_)]
        )
        xup = xup.reshape(
            (*prefix, *flatten_seq((k, i + d) for k, i, d in zip(k_, i_, d_)))
        )
        xup = xup.padslice(
            slc_prefix
            + flatten_seq(((0, k), (0, o * s)) for k, o, s in zip(k_, o_, s_))
        )
        # handle stride, and permute to move reduce to the end
        xup = xup.reshape(
            (*prefix, *flatten_seq((k, o, s) for k, o, s in zip(k_, o_, s_)))
        )
        xup = xup.padslice(
            slc_prefix + flatten_seq(((0, k), (0, o), (0, 1)) for k, o in zip(k_, o_))
        )
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
    xup = xup.padslice(
        (slc_prefix + flatten_seq(((0, o), (0, k)) for o, k in zip(o_, k_)))
    )
    return xup.transpose(
        (
            *range(len(prefix)),
            *[len(prefix) + i * 2 for i in range(len(k_))],
            *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
        )
    )


# NOTE: these work for more than 2D
@procedure_set.register
def avg_pool2d(x, kernel_size=(2, 2), stride=None):
    return x._pool(
        make_pair(kernel_size), stride if stride is not None else kernel_size
    ).mean(axis=tuple(range(0 - len(make_pair(kernel_size)), 0)))


@procedure_set.register
def max_pool2d(x, kernel_size=(2, 2), stride=None, dilation=1):
    return x._pool(
        make_pair(kernel_size), stride if stride is not None else kernel_size, dilation
    ).max(axis=tuple(range(0 - len(make_pair(kernel_size)), 0)))


@procedure_set.register
def conv_transpose2d(
    x, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0
):
    HW, trailing = weight.shape[2:], list(range(3, len(weight.shape) + 1))
    x, w = x, weight.reshape(
        groups, weight.shape[0] // groups, weight.shape[1], *weight.shape[2:]
    ).transpose(0, 2, 1, *trailing).flip(trailing)
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
        bias=bias,
        dilation=dilation,
        padding=padding,
    )


@procedure_set.register
def conv(x, weight, bias=None, groups=1, stride=1, dilation=1, padding=0):
    (bs, cin_), (cout, cin), HW = x.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups * cin == cin_ and len(x.shape) == len(
        weight.shape
    ), f"Input Tensor shape {x.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(HW) or len(padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {x.shape}"
    padding_ = (
        [padding] * 2 * len(HW)
        if isinstance(padding, int)
        else (
            padding
            if len(padding) == 2 * len(HW)
            else [p for p in padding for _ in range(2)][::-1]
        )
    )
    # x = x.pad2d(padding_)
    x = x.pad(padding_)
    x = x._pool(HW, stride, dilation)  # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout // groups, x.shape[2 : -len(HW)]
    x = x.reshape((bs, groups, cin, 1, *oyx, *HW))
    x = x.broadcast((bs, groups, cin, rcout, *oyx, *HW))
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
    return ret if bias is None else ret.add(bias.reshape((1, -1, *[1] * len(HW))))


def conv_wino(x, weight, bias=None, groups=1, stride=1, dilation=1, padding=0):
    assert not all(x == 3 for x in HW) or stride != 1 or dilation != 1
    (bs, cin_), (cout, cin), HW = x.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups * cin == cin_ and len(x.shape) == len(
        weight.shape
    ), f"Input Tensor shape {x.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(HW) or len(padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {x.shape}"
    padding_ = (
        [padding] * 2 * len(HW)
        if isinstance(padding, int)
        else (
            padding
            if len(padding) == 2 * len(HW)
            else [p for p in padding for _ in range(2)][::-1]
        )
    )

    # x = x.pad2d(padding_)._pool(
    x = x.pad(padding_)._pool(HW, stride, dilation)  # (bs, groups*cin, oy, ox, H, W)
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
                    padding_[i * 2 + 1]
                    + (-(dim + sum(padding_[i * 2 : (i + 1) * 2]) - 2) % 4),
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
        apply_matrix(winograd_G, g)
        .contiguous()
        .reshape(*HWI, 1, groups, rcout, cin, *([1] * len(tyx)))
    )  # (HWI, groups * rcout, cin) -> (HWI, bs=1, groups, rcout, cin, tyx=(1,1))
    dfactors = (
        apply_matrix(winograd_Bt, d)
        .contiguous()
        .reshape(*HWI, bs, groups, 1, cin, *tyx)
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

    return (
        (
            ret
            if bias is None
            else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))
        )
        .contiguous()
        .contiguous_backward()
    )


def cumsum(x, axis: int = 0):
    return (
        x.swapaxes(axis, -1)
        .pad((x.shape[axis] - 1, 0))
        ._pool((x.shape[axis],))
        .sum(-1)
        .swapaxes(axis, -1)
    )
