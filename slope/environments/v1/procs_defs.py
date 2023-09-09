import slope
from slope.core import ProcsDir, BaseArray
from slope.environments.v1.ops_defs import ops
import math
from typing import Tuple, Union, List, Iterator, Optional, Sequence
import itertools
import functools
import operator
import math
from collections import defaultdict

procs = ProcsDir()

# Functions


@procs.register
def zeros(shape, dtype=BaseArray.default_dtype):
    return ops.full(shape, 0.0, dtype)


@procs.register
def ones(shape, dtype=BaseArray.default_dtype):
    return ops.full(shape=shape, fill_value=1.0, dtype=dtype)


@procs.register
def full_like(y, fill_value):
    return ops.full(shape=y.shape, fill_value=fill_value, dtype=y.dtype)


@procs.register
def zeros_like(y):
    return zeros(shape=y.shape, dtype=y.dtype)


@procs.register
def ones_like(y):
    return ops.full(shape=y.shape, fill_value=1.0, dtype=y.dtype)


@procs.register
def where(x, trurun, falsrun):
    cond = x != 0.0
    cond = cond.convert(trurun.dtype)  # TODO: type promotion logic
    return cond * trurun + (1.0 - cond) * falsrun


@procs.register
def mean(x, axes=None, keepdims=False):
    out = x.sum(axes=axes, keepdim=keepdims)
    return out * (math.prod(out.shape) / math.prod(x.shape))


@procs.register
def minimum(x, y):
    return -x.maximum(-x, -y)


@procs.register
def min(x, axes=None, keepdims=False):
    return -((-x).max(x, axes, keepdims))


@procs.register
def T(x):
    perm = list(range(x.ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return x.transpose(perm)


@procs.register
def _softmax(x, axes):
    m = x - x.max(axes, keepdims=True)
    e = m.exp()
    return m, e, e.sum(axes, keepdims=True)


@procs.register
def softmax(x, axes=-1):
    _, e, ss = x._softmax(axes)
    return e.div(ss)


@procs.register
def log_softmax(x, axes=-1):
    m, _, ss = x._softmax(axes)
    return m - ss.log()


@procs.register
def dot(x, w):
    x = x.reshape((*x.shape[0:-1], 1, x.shape[-1]))
    w = w.reshape((*w.shape[0:-2], 1, w.shape[-2], w.shape[-1])).T()
    return (x * w).sum(-1).reshape((*x.shape[0:-2], -1))


@procs.register
def __getitem__(
    self, val
):  # val: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
    def normalize_int(e, i, dim_sz):
        if -dim_sz <= e < dim_sz:
            return e if e != -1 else dim_sz - 1
        raise IndexError(
            f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}"
        )

    orig_slices = list(val) if isinstance(val, tuple) else [val]
    count = defaultdict(list)
    for i, v in enumerate(orig_slices):
        count[type(v)].append(i)

    if (num_slices := len(count[int]) + len(count[slice]) + len(count)) > len(
        self.shape
    ):
        raise IndexError(f"too many indices for tensor of dimension {len(self.shape)}")
    if len(ellipsis_found := count[type(Ellipsis)]) > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")

    ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
    orig_slices[ellipsis_idx : ellipsis_idx + 1] = [slice(None)] * (
        len(self.shape) - num_slices
    )

    valid_slices = [v for v in orig_slices if v is not None]
    valid_slices = [
        v
        if isinstance(v, slice)
        else slice(y_ := normalize_int(v, i, dim_sz), y_ + 1)
        if isinstance(v, int)
        else slice(None)
        for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))
    ]

    start, stop, strides = (
        zip(*y)
        if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)])
        else ((), (), ())
    )
    new_slice = tuple(
        (s, e) if st > 0 else (e + 1, s + 1) for s, e, st in zip(start, stop, strides)
    )
    sliced_tensor = self.slice(new_slice).flip(
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
            Tensor.arange(
                ret.shape[d],
                dtype=dtypes.int32,
                requires_grad=False,
                device=self.device,
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
            ret = ret.permute(
                ret_dims[dim[0] : dim[0] + max_dim]
                + ret_dims[: dim[0]]
                + ret_dims[dim[0] + max_dim :]
            )
    return ret

@procs.register
def pad(self, pad_width, mode="constant", constant_values=0.0):
    assert mode == "constant", "Other modes not supported"
    if len(pad_width) == len(self.shape):
        assert all(len(pw) == len(pad_width[0]) for pw in pad_width)
        lo, hi, interior = [], [], []
        for pw in pad_width:
            lo += [pw[0]]; hi += [pw[1]]
            if len(pw) == 3:
                interior += [pw[2]]
    lo, hi = tuple(lo), tuple(hi)
    interior = None if len(interior) == 0 else tuple(interior)
    return self.pad_hlo(lo, hi, interior)

@procs.register
def slice(self, arg):
    starts, limits, strides = [], [], []
    assert all(len(a) == len(arg[0]) for a in arg)
    for a in arg:
        starts += [a[0]]; limits += [a[1]]
        if len(a) == 3:
            strides += [a[2]]
    starts, limits = tuple(starts), tuple(limits)
    strides = None if len(strides) == 0 else tuple(strides)
    return self.slice_hlo(starts, limits, strides)


@procs.register
def padslice(self, arg:Sequence[Optional[Tuple[int, int]]], value:float=0) :
    arg_ = tuple([a if a is not None else (0,s) for s,a in zip(self.shape, arg)])
    padding = tuple([(max(0, -p[0]), max(0, p[1]-self.shape[i])) for i,p in enumerate(arg_)])
    print(padding)
    ret = self
    ret = ret.pad(padding, constant_values=value)
    slc = tuple([(p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg_)])
    ret = ret.slice(slc)
    return ret

# def shrink(self, arg:Tuple[Tuple[int, int], ...]) -> Tensor: return mlops.Shrink.apply(self, arg=arg) if any(x != (0,s) for x,s in zip(arg, self.shape)) else self
#   def pad(self, arg: Tuple[Tuple[int, int], ...], value:float=0) -> Tensor:
#     ret = mlops.Pad.apply(self, arg=arg) if any(x != (0, 0) for x in arg) else self
#     return ret if 0 == value else ret + mlops.Pad.apply(Tensor.ones_like(self), arg=arg).where(0, value)

@procs.register
def gather(self, idx, dim: int):
    assert idx.ndim == self.ndim, "self.ndim must equal idx.ndim"
    assert all(
        s >= i for s, i in zip(self.shape, idx.shape)
    ), "all dim of idx.shape must be smaller than self.shape"
    if dim < 0:
        dim += self.ndim
    idx = idx.transpose(ax1=dim, ax2=0).unsqueeze(-1)
    permarg = list(range(self.ndim))
    permarg = (
        permarg[1:dim] + [permarg[0]] + permarg[dim + 1 :] + [permarg[dim]]
        if dim != 0
        else permarg[1:] + [permarg[0]]
    )
    return (
        (
            (
                idx
                == ops.arange(
                    self.shape[dim],
                    dtype=slope.int32,
                    requires_grad=False,
                    device=self.device,
                )
            )
            * self.permute(*permarg)
            .slice(tuple([*[(0, sh) for sh in idx.shape[1:-1]], (0, self.shape[dim])]))
            .unsqueeze(0)
        )
        .sum(-1)
        .transpose(ax1=0, ax2=dim)
    )


#   def cat(self, *args, dim=0):
#     dim = (dim + len(self.shape)) if dim < 0 else dim
#     assert all(len(y.shape) == len(self.shape) and all(y.shape[i] == s for i,s in enumerate(self.shape) if i != dim) for y in args)
#     catargs = [self, *args]
#     assert all(t.shape for t in catargs), "zero-dimensional tensor cannot be concatenated"
#     shapes = [s.shape[dim] for s in catargs]
#     shape_cumsum = [0, *accumulate(shapes)]
#     slc = [[(0, 0) for _ in self.shape] for _ in catargs]
#     for shp,k,s in zip(shapes, shape_cumsum[:-1], slc):
#       s[dim] = (k, shape_cumsum[-1] - k - shp)
#     return reduce(Tensor.__add__, [arg.pad(tuple(s)) for arg,s in zip(catargs, slc)])


@procs.register
@staticmethod
def stack(tensors, dim=0):
    first = tensors[0].unsqueeze(dim)
    unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors[1:]]
    # checks for shapes and number of dimensions delegated to cat
    return first.cat(*unsqueezed_tensors, dim=dim)


@procs.register
def repeat(self, repeats):
    base_shape = (1,) * (len(repeats) - self.ndim) + self.shape
    new_shape = [x for b in base_shape for x in [1, b]]
    expand_shape = [x for rs in zip(repeats, base_shape) for x in rs]
    final_shape = [r * s for r, s in zip(repeats, base_shape)]
    return self.reshape(new_shape).expand(expand_shape).reshape(final_shape)


@procs.register
def chunk(self, num: int, dim: int):
    dim, step = dim + self.ndim if dim < 0 else dim, math.ceil(self.shape[dim] / num)
    slice_params = [
        [slice(None)] * dim + [slice(k, k + step)]
        for k in range(0, self.shape[dim], step)
    ]
    return [self[tuple(sl)] for sl in slice_params]


@procs.register
def squeeze(self, dim=None):
    if dim is None:
        return (
            self
            if 1 not in self.shape
            else self.reshape(*[size for size in self.shape if size != 1])
        )
    if dim <= 0 and self.ndim == 0:
        return self  # This is to match PyTorch behavior
    if not -self.ndim <= dim < self.ndim:
        raise IndexError(
            f"Dimension out of range (expected to be in range of [{-self.ndim if self.ndim > 0 else self.ndim-1}, {self.ndim-1 if self.ndim > 0 else self.ndim}], but got {dim})"
        )
    if dim < 0:
        dim += self.ndim
    return (
        self
        if self.shape[dim] != 1
        else self.reshape(*[size for idx, size in enumerate(self.shape) if idx != dim])
    )


@procs.register
def unsqueeze(self, dim):
    if dim < 0:
        dim = len(self.shape) + dim + 1
    return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])


# (padding_left, padding_right, padding_top, padding_bottom)
@procs.register
def pad2d(self, padding: Union[List[int], Tuple[int, ...]], value: float = 0):
    slc = [
        (-p0, s + p1)
        for p0, p1, s in zip(padding[::2], padding[1::2], self.shape[::-1])
    ][::-1]
    slc_ =  [(0, s) for s in self.shape[: -(len(padding) // 2)]] + slc
    ret = self.padslice(slc_, value=value)
    return ret


@procs.register
def swapaxes(self, ax1=1, ax2=0):
    order = list(range(len(self.shape)))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return self.permute(order)


@procs.register
def flatten(self, start_dim=0):
    return self.reshape(shape=self.shape[:start_dim] + (-1,))


#


def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
    return (x,) * cnt if isinstance(x, int) else x


def flat(l: Iterator):
    return [item for sublist in l for item in sublist]


@procs.register
def _pool(
    self,
    k_: Tuple[int, ...],
    stride: Union[Tuple[int, ...], int] = 1,
    dilation: Union[Tuple[int, ...], int] = 1,
):
    assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
    s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
    assert len(k_) == len(s_) and len(k_) == len(
        d_
    ), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
    slc_prefix, prefix, i_ = (
        [(0, x) for x in self.shape[0 : -len(k_)]],
        self.shape[0 : -len(k_)],
        self.shape[-len(k_) :],
    )
    if any(k > s for k, s in zip(k_, s_)) or any(d != 1 for d in d_):
        o_ = [(i - d * (k - 1) - 1) // s + 1 for i, d, k, s in zip(i_, d_, k_, s_)]
        e_ = [
            math.ceil(k * (i + d) / i) for k, i, d in zip(k_, i_, d_)
        ]  # expands such that we don't need padding
        xup = (
            self.reshape(*prefix, *flatten((1, i) for i in i_))
            .expand(*prefix, *flatten((e, i) for e, i in zip(e_, i_)))
            .reshape(*prefix, *[e * i for e, i in zip(e_, i_)])
        )
        # slide by dilation
        xup = xup.padslice(slc_prefix + [(0, k * (i + d)) for k, i, d in zip(k_, i_, d_)])
        xup = xup.reshape(*prefix, *flatten((k, i + d) for k, i, d in zip(k_, i_, d_)))
        xup = xup.padslice(
            slc_prefix + flatten(((0, k), (0, o * s)) for k, o, s in zip(k_, o_, s_))
        )
        # handle stride, and permute to move reduce to the end
        xup = xup.reshape(*prefix, *flatten((k, o, s) for k, o, s in zip(k_, o_, s_)))
        xup = xup.padslice(
            slc_prefix + flatten(((0, k), (0, o), (0, 1)) for k, o in zip(k_, o_))
        )
        xup = xup.reshape(*prefix, *flatten((k, o) for k, o in zip(k_, o_)))
        return xup.permute(
            *range(len(prefix)),
            *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
            *[len(prefix) + i * 2 for i in range(len(k_))],
        )
    # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
    o_ = [(i + (s - k)) // s for i, s, k in zip(i_, s_, k_)]
    xup = self.padslice(slc_prefix + [(0, o * s) for o, s in zip(o_, s_)])
    xup = xup.reshape(*prefix, *flatten(((o, s) for o, s in zip(o_, s_))))
    xup = xup.padslice(slc_prefix + flatten(((0, o), (0, k)) for o, k in zip(o_, k_)))
    return xup.permute(
        *range(len(prefix)),
        *[len(prefix) + i * 2 for i in range(len(k_))],
        *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
    )


# NOTE: these work for more than 2D
@procs.register
def avg_pool2d(self, kernel_size=(2, 2), stride=None):
    return self._pool(
        make_pair(kernel_size), stride if stride is not None else kernel_size
    ).mean(axis=tuple(range(0 - len(make_pair(kernel_size)), 0)))


@procs.register
def max_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1):
    return self._pool(
        make_pair(kernel_size), stride if stride is not None else kernel_size, dilation
    ).max(axis=tuple(range(0 - len(make_pair(kernel_size)), 0)))


@procs.register
def conv_transpose2d(
    self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0
):
    HW, trailing = weight.shape[2:], list(range(3, len(weight.shape) + 1))
    x, w = self, weight.reshape(
        groups, weight.shape[0] // groups, weight.shape[1], *weight.shape[2:]
    ).permute(0, 2, 1, *trailing).flip(trailing)
    stride = make_pair(stride, len(HW))
    if any(s > 1 for s in stride):
        x = x.reshape(*x.shape[:2], *flatten((k, 1) for k in x.shape[2:]))
        x = x.pad(((0, 0), (0, 0), *flatten(((0, 0), (0, s - 1)) for s in stride)))
        x = x.reshape(*x.shape[:2], *[k * s for k, s in zip(x.shape[2::2], stride)])
        x = x.slice(
            (
                (0, x.shape[0]),
                (0, x.shape[1]),
                *[(0, k - (s - 1)) for k, s in zip(x.shape[2:], stride)],
            )
        )
    padding = flatten(
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
    return x.conv2d(
        w.reshape(w.shape[0] * w.shape[1], *w.shape[2:]),
        groups=groups,
        bias=bias,
        dilation=dilation,
        padding=padding,
    )


@procs.register
def conv2d(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0):
    (bs, cin_), (cout, cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups * cin == cin_ and len(self.shape) == len(
        weight.shape
    ), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(HW) or len(padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"
    padding_ = (
        [padding] * 2 * len(HW)
        if isinstance(padding, int)
        else (
            padding
            if len(padding) == 2 * len(HW)
            else [p for p in padding for _ in range(2)][::-1]
        )
    )
    x = self
    x = x.pad2d(padding_)
    breakpoint()
    x = x._pool(
        HW, stride, dilation
    )  # (bs, groups*cin, oy, ox, H, W)
    breakpoint()
    rcout, oyx = cout // groups, x.shape[2 : -len(HW)]
    x = (
        x.reshape(bs, groups, cin, 1, *oyx, *HW)
        .expand(bs, groups, cin, rcout, *oyx, *HW)
        .permute(
            0,
            1,
            3,
            *[4 + i for i in range(len(oyx))],
            2,
            *[4 + len(oyx) + i for i in range(len(HW))],
        )
    )
    # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
    ret = (
        (x * weight.reshape(1, groups, rcout, *[1] * len(oyx), cin, *HW))
        .sum([-1 - i for i in range(1 + len(oyx))], keepdim=True)
        .reshape(bs, cout, *oyx)
    )
    return ret if bias is None else ret.add(bias.reshape(1, -1, *[1] * len(HW)))


def conv2d_wino(self, weight, bias=None, groups=1, stride=1, dilation=1, padding=0):
    assert not all(x == 3 for x in HW) or stride != 1 or dilation != 1
    (bs, cin_), (cout, cin), HW = self.shape[:2], weight.shape[:2], weight.shape[2:]
    assert groups * cin == cin_ and len(self.shape) == len(
        weight.shape
    ), f"Input Tensor shape {self.shape} does not match the shape of the weights {weight.shape}. ({groups*cin} vs. {cin_})"
    if isinstance(padding, (tuple, list)):
        assert len(padding) == 2 * len(HW) or len(padding) == len(
            HW
        ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {self.shape}"
    padding_ = (
        [padding] * 2 * len(HW)
        if isinstance(padding, int)
        else (
            padding
            if len(padding) == 2 * len(HW)
            else [p for p in padding for _ in range(2)][::-1]
        )
    )

    x = self.pad2d(padding_)._pool(
        HW, stride, dilation
    )  # (bs, groups*cin, oy, ox, H, W)
    rcout, oyx = cout // groups, x.shape[2 : -len(HW)]

    # winograd conv 3 kernel f(4x4,3x3) see: http://arxiv.org/abs/1509.09308
    def apply_matrix(mat, t, dim=0):
        return (
            t
            if dim == len(HW)
            else Tensor.stack(
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
    d = self.pad2d(
        sum(
            [
                [
                    padding_[i * 2],
                    padding_[i * 2 + 1]
                    + (-(dim + sum(padding_[i * 2 : (i + 1) * 2]) - 2) % 4),
                ]
                for i, dim in enumerate(self.shape[-len(HW) :])
            ],
            [],
        )
    )._pool(
        HWI, HWO
    )  # (bs, cin_, tyx, HWI)
    d = d.permute(
        *range(len(d.shape) - len(HW), len(d.shape)), *range(len(d.shape) - len(HW))
    ).contiguous_backward()  # move HW to the front: # (HWI, bs, cin_, tyx)
    tyx = d.shape[-len(HWI) :]  # dim of tiling

    g = weight.permute(
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

    ret = ret.permute(
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


def cumsum(self, axis: int = 0):
    return (
        self.transpose(axis, -1)
        .pad2d((self.shape[axis] - 1, 0))
        ._pool((self.shape[axis],))
        .sum(-1)
        .transpose(axis, -1)
    )
