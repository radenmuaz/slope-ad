import slope as sp
from slope.opsets.v1.ops_defs import ops
import math

procs = sp.ProcsDir()

# Functions


@procs.register
def zeros(shape, dtype=sp.Array.default_dtype, **kwargs):
    return ops.full(shape, 0.0, dtype, **kwargs)


@procs.register
def ones(shape, dtype=sp.Array.default_dtype, **kwargs):
    return ops.full(shape, 1.0, dtype, **kwargs)


@procs.register
def full_like(y, fill_value, **kwargs):
    return ops.full(y.shape, fill_value, dtype=y.dtype, **kwargs)


@procs.register
def zeros_like(y, **kwargs):
    return zeros(y.shape, dtype=y.dtype, **kwargs)


@procs.register
def ones_like(y, **kwargs):
    return ops.full(y.shape, fill_value=1.0, dtype=y.dtype, **kwargs)


@procs.register
def where(x, trueval, falseval):
    cond = x != 0.0
    cond = cond.convert(trueval.dtype)  # TODO: type promotion logic
    return cond * trueval + (1.0 - cond) * falseval


@procs.register
def pow(x, y):
    assert type(y) is int
    if y == 0:
        return x.ones_like(x)
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
        ret = x.ones_like(acc) / acc
    return ret


@procs.register
def cross_entropy(x, y):
    return x * y.log()


@procs.register
def mse(x, y):
    return pow((x - y), 2)


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
def flatten(x, start_dim=0):
    return x.reshape(shape=tuple(list(x.shape[0:start_dim]) + [-1]))


@procs.register
@classmethod
def glorot_uniform(cls, *shape, **kwargs):
    return cls.rand(*shape, **kwargs).mul(
        (6 / (shape[0] + math.prod(shape[1:]))) ** 0.5
    )


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
def square(x):
    return x * x


@procs.register
def clip(x, min_, max_):
    return ((x - min_).relu() + min_) - (x - max_).relu()


@procs.register
def abs(x):
    return x.relu() + (-x).relu()


@procs.register
def sign(x):
    return x / (x.abs() + 1e-10)


@procs.register
def reciprocal(x):
    return 1.0 / x


# def reshape(self, shape, *args) -> Tensor:
#     new_shape = argfix(shape, *args)
#     assert 0 not in new_shape, f"zeros not allowed in shape {new_shape}"
#     return mlops.Reshape.apply(self, shape=tuple(-prod(self.shape) // prod(new_shape) if s == -1 else s for s in new_shape))
# def expand(self, shape, *args) -> Tensor: return mlops.Expand.apply(self, shape=tuple(x if x != -1 else s for s,x in zip(self.shape, argfix(shape, *args))))
# def permute(self, order, *args) -> Tensor: return mlops.Permute.apply(self, order=argfix(order, *args))
# def flip(self, axis, *args) -> Tensor: return mlops.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in argfix(axis, *args)])
# def pad(self, arg:Tuple[Tuple[int, int], ...]) -> Tensor: return mlops.Pad.apply(self, arg=arg) if any(x != (0,0) for x in arg) else self
# def shrink(self, arg:Tuple[Tuple[int, int], ...]) -> Tensor: return mlops.Shrink.apply(self, arg=arg) if any(x != (0,s) for x,s in zip(arg, self.shape)) else self


# - Negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
# - A slice i:j returns the elements with indices in [i, j)
#    - If omitted, i and j will default to 0 and N, respectively, where N is the length of the sequence
#    - Negative values for i and j are taken relative to the end of the sequence
#    - Both i and j will be clamped to the range (-N, N], where N in the length of the sequence
# - Indexing with np.newaxis or None on a given axis will add a new dimension of size one before that axis
# - Empty slices are not allowed (tensors with 0s in shape have to be supported first, for all backends).
# - For a slice [i:j:k] finding the correct indices is delegated to slice.indices(len).
# - Strides > 1 and < 0 are now allowed!:
#    - This works by applying Shrink -> [[Flip -> ] Pad -> Reshape -> Shrink] -> Reshape (ops in brackets are optional)
#    - Idea of stride < 0 support:
#        - Do the slice first, flip the axes were slice.step is negative, do slice.step -> -slice.step. Go to steps below.
#    - Idea of stride `s` > 1 support (Pad -> Reshape -> Shrink):
#        - Instead of doing [::s] on axis [dim_sz], do [:, 0] on axes [dim_sz_padded // s, s].
#        - So pad dim_sz with as many zeros as needed (dim_sz -> dim_sz_padded) so that reshape to [dim_sz_padded // s, s]
#          is possible.
#        - Apply Shrink to do the slice [:, 0] on axes of shapes [dim_sz_padded // s, s].
def __getitem__(self, val):
    def normalize_int(e, i, dim_sz):
        if -dim_sz <= e < dim_sz:
            return e if e != -1 else dim_sz - 1
        raise IndexError(
            f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}"
        )

    val = list(val) if isinstance(val, tuple) else [val]
    if (num_slices := sum(isinstance(v, (slice, int)) for v in val)) > len(self.shape):
        raise IndexError(f"too many indices for tensor of dimension {len(self.shape)}")
    orig_slices = list(val)
    ellipses_found = [i for i, v in enumerate(val) if v is Ellipsis]
    if len(ellipses_found) > 0:
        if len(ellipses_found) != 1:
            raise IndexError("an index can only have a single ellipsis ('...')")
        ellipsis_idx = ellipses_found[0]
        orig_slices[ellipsis_idx : ellipsis_idx + 1] = [slice(None)] * (
            len(self.shape) - num_slices
        )
    else:
        orig_slices += [slice(None)] * (len(self.shape) - num_slices)
    valid_slices = list(itertools.filterfalse(lambda x: x is None, orig_slices))
    valid_slices = [
        v if isinstance(v, slice) else slice(y := normalize_int(v, i, dim_sz), y + 1)
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
    new_shape = tuple(e - s for s, e in new_slice)
    # Shrink
    sliced_tensor = self.shrink(new_slice)
    # Flip
    if flip_axes := tuple(i for i, s in enumerate(strides) if s < 0):
        sliced_tensor = sliced_tensor.flip(axis=flip_axes)
    if any(s > 1 or s < 0 for s in strides):
        # normalize if negative strides
        strides = tuple(abs(s) for s in strides)

        def num_zeros(step, dim_sz):
            return 0 if step == 1 or (y := dim_sz % step) == 0 else (step - y)

        # Pad: add pad at the end: [dim_sz] -> [dim_sz_padded]
        paddings = tuple(
            (0, num_zeros(s, dim_sz)) for s, dim_sz in zip(strides, sliced_tensor.shape)
        )
        padded_tensor = sliced_tensor.pad(paddings)
        # Reshape: [dim_sz_padded] -> [dim_sz_padded // s, s]
        new_shape = functools.reduce(operator.add, [[sh // s, s] for sh, s in zip(padded_tensor.shape, strides)], [])  # type: ignore
        reshaped_tensor = padded_tensor.reshape(new_shape)
        # Shrink: do [:, 0]
        new_shape = new_shape[::2]
        final_slice = functools.reduce(
            operator.add, (((0, sh), (0, 1)) for sh in new_shape), ()
        )
        sliced_tensor = reshaped_tensor.shrink(final_slice)
    final_shape = []
    it_shape = iter(new_shape)
    for i in orig_slices:
        if isinstance(i, (int, slice)):
            dim_shape = next(it_shape)
            if isinstance(i, slice):
                final_shape.append(dim_shape)
        else:  # i is None
            final_shape.append(1)
    return sliced_tensor.reshape(tuple(final_shape))  # Reshape


def cat(self, *args, dim=0):
    dim = (dim + len(self.shape)) if dim < 0 else dim
    for y in args:
        assert len(y.shape) == len(self.shape) and all(
            y.shape[i] == s for i, s in enumerate(self.shape) if i != dim
        )
    catargs = [self] + list(args)
    assert all(
        len(t.shape) != 0 for t in catargs
    ), "zero-dimensional tensor cannot be concatenated"
    shape_cumsum = [0, *itertools.accumulate([y.shape[dim] for y in catargs])]
    slc = [[(0, s) for s in self.shape] for _ in catargs]
    for s, k in zip(slc, shape_cumsum):
        s[dim] = (-k, shape_cumsum[-1] - k)
    return functools.reduce(
        Tensor.__add__, [arg.slice(s) for arg, s in zip(catargs, slc)]
    )


@staticmethod
def stack(tensors, dim=0):
    first = tensors[0].unsqueeze(dim)
    unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors[1:]]
    # checks for shapes and number of dimensions delegated to cat
    return first.cat(*unsqueezed_tensors, dim=dim)


def repeat(self, repeats):
    base_shape = self.shape
    if len(repeats) > self.ndim:
        base_shape = (1,) * (len(repeats) - self.ndim) + base_shape
    new_shape = [x for i in range(len(base_shape)) for x in [1, base_shape[i]]]
    expand_shape = [x for r, s in zip(repeats, base_shape) for x in [r, s]]
    final_shape = [r * s for r, s in zip(repeats, base_shape)]
    return self.reshape(new_shape).expand(expand_shape).reshape(final_shape)


# TODO: make this nicer with syntactic sugar in slice
def chunk(self, num, dim):
    slice_params = [[(0, s) for s in self.shape] for _ in range(num)]
    for i, k in enumerate(range(0, self.shape[dim], self.shape[dim] // num)):
        slice_params[i][dim] = (k, min(self.shape[dim], k + self.shape[dim] // num))
    return [self.slice(p) for p in slice_params]


def unsqueeze(self, dim):
    if dim < 0:
        dim = len(self.shape) + dim + 1
    return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])


# (padding_left, padding_right, padding_top, padding_bottom)
def pad2d(self, padding: Union[List[int], Tuple[int, ...]]):
    slc = [
        (-p0, s + p1)
        for p0, p1, s in zip(padding[::2], padding[1::2], self.shape[::-1])
    ][::-1]
    return self.slice([(0, s) for s in self.shape[: -(len(padding) // 2)]] + slc)


@property
def T(self) -> Tensor:
    return self.transpose()


def transpose(self, ax1=1, ax2=0) -> Tensor:
    order = list(range(len(self.shape)))
    order[ax1], order[ax2] = order[ax2], order[ax1]
    return self.permute(order)


def flatten(self, start_dim=0):
    return self.reshape(shape=tuple(list(self.shape[0:start_dim]) + [-1]))


# ***** reduce ops *****


def _reduce(
    self,
    fxn: Type[Function],
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim=False,
):
    axis_: List[int] = (
        list(range(len(self.shape)))
        if axis is None
        else ([axis] if isinstance(axis, int) else list(axis))
    )
    axis_ = [x if x >= 0 else x + len(self.shape) for x in axis_]
    shape = [self.shape[i] for i in range(len(self.shape)) if i not in axis_]
    ret = fxn.apply(
        self,
        new_shape=tuple(
            1 if i in axis_ else self.shape[i] for i in range(len(self.shape))
        ),
    )
    return ret if keepdim else ret.reshape(shape=shape)


def sum(self, axis=None, keepdim=False):
    return self._reduce(mlops.Sum, axis, keepdim)


def max(self, axis=None, keepdim=False):
    return self._reduce(mlops.Max, axis, keepdim)


def min(self, axis=None, keepdim=False):
    return -((-self).max(axis=axis, keepdim=keepdim))


def mean(self, axis=None, keepdim=False):
    out = self.sum(axis=axis, keepdim=keepdim)
    return out * (prod(out.shape) / prod(self.shape))


def std(self, axis=None, keepdim=False, correction=1):
    square_sum = ((self - self.mean(axis=axis, keepdim=True)).square()).sum(
        axis=axis, keepdim=keepdim
    )
    return (
        square_sum / (prod(self.shape) / prod(square_sum.shape) - correction)
    ).sqrt()


def _pool(
    self,
    k_: Tuple[int, ...],
    stride: Union[Tuple[int, ...], int] = 1,
    dilation: Union[Tuple[int, ...], int] = 1,
    _insert_dims=tuple(),
) -> Tensor:
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
            self.reshape(
                *prefix, *([1] * len(_insert_dims)), *flatten((1, i) for i in i_)
            )
            .expand(*prefix, *_insert_dims, *flatten((e, i) for e, i in zip(e_, i_)))
            .reshape(*prefix, *_insert_dims, *[e * i for e, i in zip(e_, i_)])
        )
        # NOTE: _insert_dims is required because reduces can't be merged (yet)
        prefix += _insert_dims
        slc_prefix += [(0, x) for x in _insert_dims]
        # slide by dilation
        xup = xup.slice(slc_prefix + [(0, k * (i + d)) for k, i, d in zip(k_, i_, d_)])
        xup = xup.reshape(*prefix, *flatten((k, i + d) for k, i, d in zip(k_, i_, d_)))
        xup = xup.slice(
            slc_prefix + flatten(((0, k), (0, o * s)) for k, o, s in zip(k_, o_, s_))
        )
        # handle stride, and permute to move reduce to the end
        xup = xup.reshape(*prefix, *flatten((k, o, s) for k, o, s in zip(k_, o_, s_)))
        xup = xup.slice(
            slc_prefix + flatten(((0, k), (0, o), (0, 1)) for k, o in zip(k_, o_))
        )
        xup = xup.reshape(*prefix, *flatten((k, o) for k, o in zip(k_, o_)))
        return xup.permute(
            *range(len(prefix)),
            *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
            *[len(prefix) + i * 2 for i in range(len(k_))],
        )
    else:
        # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
        o_ = [(i + (s - k)) // s for i, s, k in zip(i_, s_, k_)]
        xup = self.slice(slc_prefix + [(0, o * s) for o, s in zip(o_, s_)])
        xup = xup.reshape(
            *prefix,
            *([1] * len(_insert_dims)),
            *flatten(((o, s) for o, s in zip(o_, s_))),
        )
        if len(_insert_dims):
            xup = xup.expand(
                *prefix, *_insert_dims, *flatten(((o, s) for o, s in zip(o_, s_)))
            )
            prefix += _insert_dims
            slc_prefix += [(0, x) for x in _insert_dims]
        xup = xup.slice(slc_prefix + flatten(((0, o), (0, k)) for o, k in zip(o_, k_)))
        return xup.permute(
            *range(len(prefix)),
            *[len(prefix) + i * 2 for i in range(len(k_))],
            *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
        )


# NOTE: these work for more than 2D
def avg_pool2d(self, kernel_size=(2, 2), stride=None):
    return self._pool(
        make_pair(kernel_size), stride if stride is not None else kernel_size
    ).mean(axis=tuple(range(0 - len(make_pair(kernel_size)), 0)))


def max_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1):
    return self._pool(
        make_pair(kernel_size), stride if stride is not None else kernel_size, dilation
    ).max(axis=tuple(range(0 - len(make_pair(kernel_size)), 0)))


def conv_transpose2d(
    self,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    groups=1,
    stride=1,
    dilation=1,
    padding=0,
    output_padding=0,
) -> Tensor:
    HW, trailing = weight.shape[2:], list(range(3, len(weight.shape) + 1))
    x, w = self, weight.reshape(
        groups, weight.shape[0] // groups, weight.shape[1], *weight.shape[2:]
    ).permute(0, 2, 1, *trailing).flip(trailing)
    stride = make_pair(stride, len(HW))
    if any(s > 1 for s in stride):
        x = x.reshape(*x.shape[:2], *flatten((k, 1) for k in x.shape[2:]))
        x = x.pad(((0, 0), (0, 0), *flatten(((0, 0), (0, s - 1)) for s in stride)))
        x = x.reshape(*x.shape[:2], *[k * s for k, s in zip(x.shape[2::2], stride)])
        x = x.shrink(
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


def conv2d(
    self,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    groups=1,
    stride=1,
    dilation=1,
    padding=0,
) -> Tensor:
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

    # conv2d is a pooling op (with padding)
    x = self.pad2d(padding_)._pool(
        HW, stride, dilation
    )  # (bs, groups*cin, oy, ox, H, W)
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

    # expand the channels with the pool
    # TODO: this reduces the number of kernels, but it's slower!
    # x = self.pad2d(padding_)._pool((H,W), stride, dilation, _insert_dims=(cout//groups,))   # (bs, groups*cin, rcout, oy, ox, H, W)
    # rcout, oy, ox = x.shape[2:5]
    # x = x.reshape(bs, groups, cin, rcout, oy, ox, H, W).permute(0,1,3,4,5,2,6,7)

    # conv! broadcasted to (bs, groups, rcout, *oyx, cin, *HW)
    ret = (
        (x * weight.reshape(1, groups, rcout, *[1 for _ in range(len(oyx))], cin, *HW))
        .sum([-1 - i for i in range(1 + len(oyx))], keepdim=True)
        .reshape(bs, cout, *oyx)
    )
    return (
        ret
        if bias is None
        else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))
    )


def dot(self, w: Tensor) -> Tensor:
    if (n1 := len(self.shape)) * (n2 := len(w.shape)) == 0:
        raise RuntimeError(
            f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
        )
    x = self.reshape(*self.shape[0:-1], 1, self.shape[-1])
    w = w.reshape(*w.shape[0:-2], 1, w.shape[-2], w.shape[-1]).transpose(-1, -2)
    r = (x * w).sum(-1)
    return r.reshape((*r.shape[:-2], r.shape[-1])) if len(self.shape) == 1 else r


def cumsum(self, axis=0):
    x = self.permute(*(i for i in range(self.ndim) if i != axis), axis)
    return (
        x.reshape(1, 1, -1, self.shape[axis])
        .conv2d(
            Tensor.ones(
                1, 1, 1, self.shape[axis], dtype=self.dtype, device=self.device
            ),
            padding=(self.shape[axis] - 1, 0, 0, 0),
        )
        .reshape(*x.shape)
        .permute(*range(axis), self.ndim - 1, *range(axis, self.ndim - 1))
    )


# ***** mlops (unary) *****


def contiguous(self):
    return mlops.Contiguous.apply(self)


def log(self):
    return mlops.Log.apply(self)


def exp(self):
    return mlops.Exp.apply(self)


def relu(self):
    return mlops.Relu.apply(self)


def sin(self):
    return mlops.Sin.apply(self)


def cos(self):
    return ((math.pi / 2) - self).sin()


def tan(self):
    return self.sin() / self.cos()


@staticmethod
def _tri(r: int, c: int, k: int = 0) -> Tensor:
    return Tensor.arange(r).unsqueeze(1).expand(r, c) <= Tensor.arange(
        c - k, start=-k
    ).unsqueeze(0).expand(r, c)


def triu(self, k: int = 0) -> Tensor:
    return Tensor._tri(self.shape[-2], self.shape[-1], k=k).where(
        self, Tensor.zeros_like(self)
    )


def tril(self, k: int = 0) -> Tensor:
    return Tensor._tri(self.shape[-2], self.shape[-1], k=k + 1).where(
        Tensor.zeros_like(self), self
    )


# ***** math functions (unary) *****


def __neg__(self):
    return 0.0 - self


def sqrt(self):
    return self.pow(0.5)


def rsqrt(self):
    return self.pow(-0.5)


def square(self):
    return self * self


def clip(self, min_, max_):
    return self.maximum(min_).minimum(max_)


def abs(self):
    return self.relu() + (-self).relu()


def sign(self):
    return self / (self.abs() + 1e-10)


def reciprocal(self):
    return 1.0 / self


# ***** activation functions (unary) *****


def sigmoid(self):
    return (1.0 + (-self).exp()).reciprocal()


def elu(self, alpha=1.0):
    return self.relu() - alpha * (1 - self.exp()).relu()


def celu(self, alpha=1.0):
    return self.maximum(0) + (alpha * ((self / alpha).exp() - 1)).minimum(0)


def swish(self):
    return self * self.sigmoid()


def silu(self):
    return self.swish()  # The SiLU function is also known as the swish function.


def relu6(self):
    return self.relu() - (self - 6).relu()


def hardswish(self):
    return self * (self + 3).relu6() * (1 / 6)


def tanh(self):
    return 2.0 * ((2.0 * self).sigmoid()) - 1.0


def hardtanh(self, min_val=-1, max_val=1):
    return self.clip(min_val, max_val)


def gelu(self):
    return (
        0.5 * self * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
    )


def quick_gelu(self):
    return self * (self * 1.702).sigmoid()


def leakyrelu(self, neg_slope=0.01):
    return self.relu() - (-neg_slope * self).relu()


def mish(self):
    return self * self.softplus().tanh()


def softplus(self, beta=1):
    return (1 / beta) * (1 + (self * beta).exp()).log()


def softsign(self):
    return self / (1 + self.abs())


# ***** broadcasted binary mlops *****
def _broadcasted(
    self, fxn: Type[Function], other: Union[Tensor, float], reverse: bool = False
) -> Tensor:
    dtype = (
        self.dtype
        if self.dtype != dtypes.bool and not isinstance(self.dtype, ImageDType)
        else dtypes.float32
    )
    x, y = [
        Tensor(t, device=self.device, requires_grad=False, dtype=dtype)
        if not isinstance(t, Tensor)
        else t
        for t in ([other, self] if reverse else [self, other])
    ]
    x, y = [
        t.reshape(
            [1] * (max(len(x.shape), len(y.shape)) - len(t.shape)) + list(t.shape)
        )
        for t in [x, y]
    ]
    shape_ret = tuple(max(sx, sy) for sx, sy in zip(x.shape, y.shape))
    return fxn.apply(x.expand(shape_ret), y.expand(shape_ret))


def add(self, x: Union[Tensor, float], reverse=False) -> Tensor:
    return (
        self._broadcasted(mlops.Add, x, reverse)
        if isinstance(x, Tensor) or x != 0.0
        else self
    )


def sub(self, x: Union[Tensor, float], reverse=False) -> Tensor:
    return (
        self._broadcasted(mlops.Sub, x, reverse)
        if isinstance(x, Tensor) or x != 0.0 or reverse
        else self
    )


def mul(self, x: Union[Tensor, float], reverse=False) -> Tensor:
    return (
        self._broadcasted(mlops.Mul, x, reverse)
        if isinstance(x, Tensor) or x != 1.0
        else self
    )


def pow(self, x: Union[Tensor, float], reverse=False) -> Tensor:
    return (
        self._broadcasted(mlops.Pow, x, reverse)
        if isinstance(x, Tensor) or x != 1.0 or reverse
        else self
    )


def div(self, x: Union[Tensor, float], reverse=False) -> Tensor:
    return (
        self._broadcasted(mlops.Div, x, reverse)
        if isinstance(x, Tensor) or reverse or x == 0.0
        else self.mul(1 / x)
    )


def matmul(self, x: Tensor, reverse=False) -> Tensor:
    return x.dot(self) if reverse else self.dot(x)


def maximum(self, x: Union[Tensor, float]) -> Tensor:
    return self._broadcasted(mlops.Maximum, x)


def minimum(self, x: Union[Tensor, float]) -> Tensor:
    return -((-self).maximum(-x))


def eq(self, x) -> Tensor:
    return self._broadcasted(mlops.Equal, x, False)
