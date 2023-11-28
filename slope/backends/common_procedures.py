import slope
from slope.core import (
    ProcedureSet,
    Tensor,
)

import math
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Sequence, Union, Iterator, Callable
from collections import defaultdict

sum_py = sum
max_py = max
abs_py = abs
slice_py = slice
procedure_set = ProcedureSet()



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
def where(x, trueval, falseval):
    cond = x != 0.0
    cond = cond.cast(trueval.dtype)
    return cond * trueval + (1.0 - cond) * falseval


@procedure_set.register(static_argnames="axes keepdims")
def mean(x, axes=None, keepdims=False):
    out = x.sum(axes=axes, keepdims=keepdims)
    return out * (math.prod(out.shape) / math.prod(x.shape))


@procedure_set.register()
def rsqrt(x):
    return (slope.ones_like(x) / x).sqrt()


@procedure_set.register()
def cos(x):
    return ((math.pi / 2) - x).sin()


@procedure_set.register()
def tan(x):
    return x.sin() / x.cos()


@procedure_set.register()
def not_equal(x, w):
    return ~(x.equal(w))


@procedure_set.register()
def greater_equal(x, w):
    return x.maximum(w).equal(w)


@procedure_set.register()
def less_equal(x, w):
    return x.minimum(w).equal(w)


@procedure_set.register()
def greater(x, w):
    return 1.0 - (x <= w)


@procedure_set.register()
def less(x, w):
    return 1.0 - (x >= w)


@procedure_set.register()
def minimum(x, w):
    return -x.maximum(-x, -w)


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
    axes = axes + len(x.shape) if axes < 0 else axes
    m = (x == x.max(axes=axes, keepdims=True)).cast(slope.int32)
    idx = m * slope.arange(x.shape[axes] - 1, -1, -1, dtype=slope.int32).reshape(
        (x.shape[axes], *[1] * (x.ndim - axes - 1))
    )
    ret = x.shape[axes] - idx.max(axes=axes, keepdims=keepdims) - 1
    return ret


@procedure_set.register(static_argnames="axes keepdims")
def argmin(x, axes=None, keepdims=False):
    return (-x).argmax(axes=axes, keepdims=keepdims)


@procedure_set.register(inline=True)
def pow(x, w: Union[Tensor, float], reverse=False) -> Tensor:
    if w.__class__ is not Tensor and not reverse:
        # simple pow identities
        if w < 0:
            return (slope.ones_like(x) / x) ** (-w)
        if w == 3.0:
            return x * x * x
        if w == 2.0:
            return x * x
        if w == 1.0:
            return x
        if w == 0.5:
            return x.sqrt()
    if not isinstance(x, Tensor) and reverse and x > 0:
        return x.mul(math.log(x)).exp()
    ar = x.abs().log().mul(x).exp() if not reverse or isinstance(x, Tensor) else x.mul(math.log(abs_py(x))).exp()
    # correct sign of negative numbers raised to a power (cos has a period of 2pi so we use it here to get the oddness of the power)
    sign = (
        (x * math.pi).cos() if isinstance(x, Tensor) else math.cos(x * math.pi) if not reverse else (x * math.pi).cos()
    )
    # we only need to correct the sign if the base is negative
    base_sign = ((x.sign() if not reverse else x.sign() if isinstance(x, Tensor) else math.copysign(1, x)) - 1) / -2
    # we need 0 to be positive so we need to correct base_sign when the base is 0
    base_sign = base_sign - (
        1.5
        * (1 - (x.sign().abs() if not reverse else x.sign().abs() if isinstance(x, Tensor) else abs_py(int(bool(x)))))
    )
    # inject nan if the base is negative and the power is not an integer
    to_nan = (
        ((x - x.trunc()) * 1e10).abs().clip(0, 1)
        if isinstance(x, Tensor)
        else int(bool(x - int(x)))
        if not reverse
        else ((x - x.trunc()) * 1e10).abs().clip(0, 1)
    ) * base_sign
    inject_nan = (
        ((((-to_nan) * 2) + 1)).log().add(1) if isinstance(to_nan, Tensor) else 1 if not to_nan else float("nan")
    )
    return ar.mul(sign * base_sign + (1 - base_sign)).mul(inject_nan)


@procedure_set.register()
def log2(x):
    return x.log() / math.log(2)


@procedure_set.register()
@staticmethod
def _tri(r: int, c: int, k: int = 0, **kwargs) -> Tensor:
    return slope.arange(r, **kwargs).unsqueeze(1).expand(r, c) <= Tensor.arange(-k, c - k, **kwargs).unsqueeze(
        0
    ).expand(r, c)


@procedure_set.register()
def triu(self, k: int = 0) -> Tensor:
    return slope._tri(self.shape[-2], self.shape[-1], k=k, dtype=self.dtype, device=self.device).where(
        self, slope.zeros_like(self)
    )


@procedure_set.register()
def tril(self, k: int = 0) -> Tensor:
    return slope._tri(self.shape[-2], self.shape[-1], k=k + 1, dtype=self.dtype, device=self.device).where(
        slope.zeros_like(self), self
    )


@procedure_set.register()
def trunc(self: Tensor) -> Tensor:
    return self.cast(slope.int32).cast(self.dtype)


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


@procedure_set.register()
def matmul(x, w):
    x = x.reshape((*x.shape[0:-1], 1, x.shape[-1]))
    w = w.reshape((*w.shape[0:-2], 1, w.shape[-2], w.shape[-1])).T()
    return (x * w).sum(-1).reshape((*x.shape[0:-2], -1))


@procedure_set.register()
def T(x):
    perm = list(range(x.ndim))
    perm[-2], perm[-1] = perm[-1], perm[-2]
    return x.permute(tuple(perm))




@procedure_set.register(inline=True)
def getitem(self, val):
    # Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
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
    if any(abs_py(s) != 1 for s in strides):
        strides = tuple(abs_py(s) for s in strides)
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
            ret = ret.permute(ret_dims[dim[0] : dim[0] + max_dim] + ret_dims[: dim[0]] + ret_dims[dim[0] + max_dim :])
    return ret


@procedure_set.register(static_argnames="pad mode constant_values")
def pad(x, pad, mode="constant", constant_values=0.0):
    assert mode == "constant", "Other modes not supported"
    if type(pad) is int:
        pad = (pad, pad, 0) * x.ndim
    elif all(type(pw) is int for pw in pad):
        assert len(pad) == x.ndim
        pad = tuple((pw, pw, 0) for pw in pad)
    elif len(pad) == 2 and all(type(item) is int for item in pad):
        pad = (*pad, 0) * x.ndim
    elif len(pad) == 3 and all(type(item) is int for item in pad):
        pad = (pad,) * x.ndim
    else:
        assert all(2 <= len(pw) <= 3 for pw in pad)
        pad = tuple((*pw, 0) if len(pw) == 2 else pw for pw in pad)
    lo, hi, interior = tuple(zip(*pad))
    return x.pad_lowlevel(lo, hi, interior, value=constant_values)


@procedure_set.register(static_argnames="arg")
def slice(x, arg):
    arg = tuple((*a, 1) if len(a) == 2 else a for a in arg)
    starts, limits, strides = tuple(zip(*arg))
    return x.slice_lowlevel(starts, limits, strides)


@procedure_set.register(static_argnames=("arg", "value"))
def padslice(x, arg: Sequence[Optional[Tuple[int, int]]], value: float = 0):
    arg_ = tuple([a if a is not None else (0, s) for s, a in zip(x.shape, arg)])
    padding = tuple([(max_py(0, -p[0]), max_py(0, p[1] - x.shape[i])) for i, p in enumerate(arg_)])
    x = x.pad(padding, constant_values=value)
    slc = tuple([(p[0] + padding[i][0], p[1] + padding[i][0]) for i, p in enumerate(arg_)])
    x = x.slice(slc)
    return x


@procedure_set.register(static_argnames="padding value")
def pad2d(x, padding: Union[List[int], Tuple[int, ...]], value: float = 0):
    # (padding_left, padding_right, padding_top, padding_bottom)
    slc = [(-p0, s + p1) for p0, p1, s in zip(padding[::2], padding[1::2], x.shape[::-1])][::-1]
    return x.padslice([(0, s) for s in x.shape[: -(len(padding) // 2)]] + slc, value=value)


@procedure_set.register(static_argnames="dim")
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


@procedure_set.register(static_argnames="dim")
@staticmethod
def stack(tensors, dim=0):
    first = tensors[0].expand_dims(dim)
    expand_dimsd_tensors = [tensor.expand_dims(dim) for tensor in tensors[1:]]
    return first.cat(*expand_dimsd_tensors, dim=dim)


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
    return tuple(x[tuple(sl)] for sl in slice_params)


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


@procedure_set.register(static_argnames="ax aw")
def transpose(x, ax=1, aw=0):
    order = list(range(len(x.shape)))
    order[ax], order[aw] = order[aw], order[ax]
    return x.permute(tuple(order))


@procedure_set.register(static_argnames="start_dim")
def flatten(x, start_dim=0):
    return x.reshape(shape=x.shape[:start_dim] + (-1,))


@procedure_set.register(static_argnames="axis")
def cumsum(x, axis: int = 0):
    return x.transpose(axis, -1).pad((x.shape[axis] - 1, 0)).pool((x.shape[axis],)).sum(-1).transpose(axis, -1)


@staticmethod
@procedure_set.register(static_argnames="start stop step")
def arange_with_cumsum(start, stop=None, step=1):
    if stop is None:
        stop, start = start, 0
    return slope.full((math.ceil((stop - start) / step),), step).cumsum() + (start - step)


@procedure_set.register(static_argnames="dtype")
def one_hot(x, k, dtype=Tensor.int32):
    return (x[:, None] == slope.arange(k, dtype=dtype)).cast(dtype)


# ***** cast ops *****


def float(self) -> Tensor:
    return self.cast(slope.float32)


def half(self) -> Tensor:
    return self.cast(slope.float16)


