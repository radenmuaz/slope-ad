import math
from contextlib import contextmanager
from typing import (
    Callable,
    Tuple,
    List,
    Any,
    List,
    Tuple,
    Optional,
    Any,
    Union,
    Callable,
    NamedTuple,
    Final,
)
import functools
import slope
from slope import utils

from abc import ABC, abstractmethod

class BaseArray:
    def notimplemented(self, *args, **kwargs):
        raise NotImplementedError

    constant = notimplemented
    convert = notimplemented
    astype = convert
    neg = notimplemented
    exp = notimplemented
    log = notimplemented
    add = notimplemented
    sub = notimplemented
    mul = notimplemented
    div = notimplemented
    equal = notimplemented
    not_equal = notimplemented
    maximum = notimplemented
    max = notimplemented
    sum = notimplemented
    choose = notimplemented
    where = notimplemented

    __neg__ = lambda self: self.neg()
    __add__ = lambda self, other: self.add(other)
    __radd__ = lambda self, other: self.__class__.add(other, self)
    __sub__ = lambda self, other: self.sub(other)
    __rsub__ = lambda self, other: self.__class__.sub(other, self)
    __mul__ = __rmul__ = mul
    __div__ = div
    __rdiv__ = lambda self, other: self.__class__.div(other, self)
    __truediv__ = __div__
    __truerdiv__ = __rdiv__
    __eq__ = lambda self, other: self.equal(other)
    __ne__ = lambda self, other: self.not_equal(other)
    __ge__ = lambda self, other: self.maximum(other).equal(self)
    __le__ = lambda self, other: self.minimum(other).equal(self)
    __gt__ = lambda self, other: 1.0 - (self <= other)
    __lt__ = lambda self, other: 1.0 - (self >= other)

    def random_normal(self, x, dtype):
        # Box-Muller transform
        nbits = dtype.np.itemsize
        u1 = 0
        while u1 == 0:
            u1 = x / (2**nbits)  # Convert the 64-bit integer to a float between 0 and 1
        u2 = self.rng_bit(x) / (2**nbits)
        z0 = np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return z0

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

    # # NOTE: using slice is discouraged and things should migrate to pad and shrink
    # def slice(self, arg: Sequence[Optional[Tuple[int, int]]]):
    #     arg_ = tuple(a if a is not None else (0, s) for s, a in zip(self.shape, arg))
    #     padding = tuple(
    #         (max(0, -p[0]), max(0, p[1] - self.shape[i])) for i, p in enumerate(arg_)
    #     )
    #     return self.pad(padding).shrink(
    #         tuple(
    #             (p[0] + padding[i][0], p[1] + padding[i][0]) for i, p in enumerate(arg_)
    #         )
    #     )

    # def __getitem__(self, val):
    #     def slcfix(i, sz, default):
    #         return (
    #             default if i is None else max(0, min(sz, sz + i if i < 0 else i))
    #         )  # Fix negative idxs, clamp to [0,N]

    #     new_slice, new_shape = [], []
    #     val = [val] if not isinstance(val, (list, tuple)) else val
    #     assert sum(s is not None for s in val) <= len(self.shape)
    #     assert all(s.step is None or s.step == 1 for s in val if isinstance(s, slice))
    #     for i, (sz, s) in enumerate(
    #         zip(self.shape, [v for v in val if v is not None])
    #     ):  # Slicing only depends on ints + slices
    #         if isinstance(s, int) and not (-sz <= s < sz):
    #             raise IndexError(
    #                 f"index {s} is out of bounds for dimension {i} with size {sz}"
    #             )
    #     new_slice.append(
    #         (s % sz, s % sz + 1)
    #         if isinstance(s, int)
    #         else (slcfix(s.start, sz, 0), slcfix(s.stop, sz, sz))
    #     )
    #     for s, sz in zip(
    #         val,
    #         [
    #             self.shape[i - 1]
    #             for i in itertools.accumulate([int(s is not None) for s in val])
    #         ],
    #     ):  # Shape depends on slices + positions of Nones
    #         if not isinstance(s, int):
    #             new_shape.append(
    #                 1 if s is None else slcfix(s.stop, sz, sz) - slcfix(s.start, sz, 0)
    #             )
    #     new_shape += [self.shape[i] for i in range(len(new_slice), len(self.shape))]
    #     new_slice += [
    #         (0, self.shape[i]) for i in range(len(new_slice), len(self.shape))
    #     ]
    #     return self.slice(new_slice).reshape(new_shape if len(new_shape) else (1,))

    # def concatenate(self, *args, dim=0):
    #     dim = (dim + len(self.shape)) if dim < 0 else dim
    #     for y in args:
    #         assert len(y.shape) == len(self.shape) and all(
    #             y.shape[i] == s for i, s in enumerate(self.shape) if i != dim
    #         )
    #     catargs = [self] + list(args)
    #     shape_cumsum = [0, *itertools.accumulate([y.shape[dim] for y in catargs])]
    #     slc = [[(0, s) for s in self.shape] for _ in catargs]
    #     for s, k in zip(slc, shape_cumsum):
    #         s[dim] = (-k, shape_cumsum[-1] - k)
    #     return functools.reduce(
    #         self.__class__.__add__, [arg.slice(s) for arg, s in zip(catargs, slc)]
    #     )

    # # TODO: make this nicer with syntactic sugar in slice
    # def split(self, num, dim):
    #     slice_params = [[(0, s) for s in self.shape] for _ in range(num)]
    #     for i, k in enumerate(range(0, self.shape[dim], self.shape[dim] // num)):
    #         slice_params[i][dim] = (k, min(self.shape[dim], k + self.shape[dim] // num))
    #     return [self.slice(p) for p in slice_params]

    # # (padding_left, padding_right, padding_top, padding_bottom)
    # def pad2d(self, padding: Union[List[int], Tuple[int, ...]]):
    #     return self.slice(
    #         (
    #             (0, self.shape[0]),
    #             (0, self.shape[1]),
    #             (-padding[2], self.shape[2] + padding[3]),
    #             (-padding[0], self.shape[3] + padding[1]),
    #         )
    #     )

    # @staticmethod
    # def stack(tensors, dim=0):
    #     first = tensors[0].unsqueeze(dim)
    #     unsqueezed_tensors = [tensor.unsqueeze(dim) for tensor in tensors[1:]]
    #     # checks for shapes and number of dimensions delegated to cat
    #     return first.cat(*unsqueezed_tensors, dim=dim)

    # def repeat(self, repeats):
    #     base_shape = self.shape
    #     if len(repeats) > self.ndim:
    #         base_shape = (1,) * (len(repeats) - self.ndim) + base_shape
    #     new_shape = [x for i in range(len(base_shape)) for x in [1, base_shape[i]]]
    #     expand_shape = [x for r, s in zip(repeats, base_shape) for x in [r, s]]
    #     final_shape = [r * s for r, s in zip(repeats, base_shape)]
    #     return self.reshape(new_shape).broadcast(expand_shape).reshape(final_shape)

    # # TODO: make this nicer with syntactic sugar in slice
    # def chunk(self, num, dim):
    #     slice_params = [[(0, s) for s in self.shape] for _ in range(num)]
    #     for i, k in enumerate(range(0, self.shape[dim], self.shape[dim] // num)):
    #         slice_params[i][dim] = (k, min(self.shape[dim], k + self.shape[dim] // num))
    #     return [self.slice(p) for p in slice_params]

    # def unsqueeze(self, dim):
    #     if dim < 0:
    #         dim = len(self.shape) + dim + 1
    #     return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

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

    def sqrt(self):
        return self.pow(0.5)

    def rsqrt(self):
        return self.pow(-0.5)

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

    # # ***** activation functions (unary) *****

    def relu(self):
        return ops.ReLU.do(self)

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
            0.5
            * self
            * (1 + (self * 0.7978845608 * (1 + 0.044715 * self * self)).tanh())
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

    # ***** functional nn ops *****

    def linear(self, weight, bias=None):
        x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
        return x.add(bias) if bias is not None else x

    def serial(self, ll: List[Callable]):
        return functools.reduce(lambda x, f: f(x), ll, self)

    def layernorm(self, axis=-1, eps: float = 1e-5):
        y = self - self.mean(axis, keepdim=True)
        return y.mul((y * y).mean(axis, keepdim=True).add(eps).rsqrt())

    def batchnorm(
        self,
        weight,
        bias,
        mean,
        invstd,
    ):
        x = self - mean.reshape(shape=[1, -1, 1, 1])
        if weight:
            x = x * weight.reshape(shape=[1, -1, 1, 1])
        ret = x.mul(
            invstd.reshape(shape=[1, -1, 1, 1]) if len(invstd.shape) == 1 else invstd
        )
        return (ret + bias.reshape(shape=[1, -1, 1, 1])) if bias else ret

    #
    def _pool(
        self,
        k_: Tuple[int, ...],
        stride: Union[Tuple[int, ...], int] = 1,
        dilation: Union[Tuple[int, ...], int] = 1,
        _insert_dims=tuple(),
    ):
        assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
        s_, d_ = utils.make_pair(stride, len(k_)), utils.make_pair(dilation, len(k_))
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
                    *prefix,
                    *([1] * len(_insert_dims)),
                    *utils.flatten((1, i) for i in i_),
                )
                .broadcast(
                    *prefix,
                    *_insert_dims,
                    *utils.flatten((e, i) for e, i in zip(e_, i_)),
                )
                .reshape(*prefix, *_insert_dims, *[e * i for e, i in zip(e_, i_)])
            )
            # NOTE: _insert_dims is required because reduces can't be merged (yet)
            prefix += _insert_dims
            slc_prefix += [(0, x) for x in _insert_dims]
            # slide by dilation
            xup = xup.slice(
                slc_prefix + [(0, k * (i + d)) for k, i, d in zip(k_, i_, d_)]
            )
            xup = xup.reshape(
                *prefix, *utils.flatten((k, i + d) for k, i, d in zip(k_, i_, d_))
            )
            xup = xup.slice(
                slc_prefix
                + utils.flatten(((0, k), (0, o * s)) for k, o, s in zip(k_, o_, s_))
            )
            # handle stride, and permute to move reduce to the end
            xup = xup.reshape(
                *prefix, *utils.flatten((k, o, s) for k, o, s in zip(k_, o_, s_))
            )
            xup = xup.slice(
                slc_prefix
                + utils.flatten(((0, k), (0, o), (0, 1)) for k, o in zip(k_, o_))
            )
            xup = xup.reshape(*prefix, *utils.flatten((k, o) for k, o in zip(k_, o_)))
            return xup.transpose(
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
                *utils.flatten(((o, s) for o, s in zip(o_, s_))),
            )
            if len(_insert_dims):
                xup = xup.broadcast(
                    *prefix,
                    *_insert_dims,
                    *utils.flatten(((o, s) for o, s in zip(o_, s_))),
                )
                prefix += _insert_dims
                slc_prefix += [(0, x) for x in _insert_dims]
            xup = xup.slice(
                slc_prefix + utils.flatten(((0, o), (0, k)) for o, k in zip(o_, k_))
            )
            return xup.transpose(
                *range(len(prefix)),
                *[len(prefix) + i * 2 for i in range(len(k_))],
                *[len(prefix) + i * 2 + 1 for i in range(len(k_))],
            )

    # NOTE: these work for more than 2D
    def avg_pool2d(self, kernel_size=(2, 2), stride=None):
        return self._pool(
            utils.make_pair(kernel_size), stride if stride is not None else kernel_size
        ).mean(axis=tuple(range(0 - len(utils.make_pair(kernel_size)), 0)))

    def max_pool2d(self, kernel_size=(2, 2), stride=None, dilation=1):
        return self._pool(
            utils.make_pair(kernel_size),
            stride if stride is not None else kernel_size,
            dilation,
        ).max(axis=tuple(range(0 - len(utils.make_pair(kernel_size)), 0)))

    def conv_transpose2d(
        self,
        weight,
        bias: Optional[Any] = None,
        groups=1,
        stride=1,
        dilation=1,
        padding=0,
        output_padding=0,
    ):
        HW, trailing = weight.shape[2:], list(range(3, len(weight.shape) + 1))
        x, w = self, weight.reshape(
            groups, weight.shape[0] // groups, weight.shape[1], *weight.shape[2:]
        ).transpose(0, 2, 1, *trailing).flip(trailing)
        stride = utils.make_pair(stride, len(HW))
        if any(s > 1 for s in stride):
            x = x.reshape(*x.shape[:2], *utils.flatten((k, 1) for k in x.shape[2:]))
            x = x.pad(
                ((0, 0), (0, 0), *utils.flatten(((0, 0), (0, s - 1)) for s in stride))
            )
            x = x.reshape(*x.shape[:2], *[k * s for k, s in zip(x.shape[2::2], stride)])
            x = x.shrink(
                (
                    (0, x.shape[0]),
                    (0, x.shape[1]),
                    *[(0, k - (s - 1)) for k, s in zip(x.shape[2:], stride)],
                )
            )
        padding = utils.flatten(
            (
                ((k - 1) * d - p, (k - 1) * d - p + op)
                for k, d, p, op in reversed(
                    list(
                        zip(
                            HW,
                            utils.make_pair(dilation, len(HW)),
                            utils.make_pair(padding, len(HW)),
                            utils.make_pair(output_padding, len(HW)),
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
        weight,
        bias: Optional[Any] = None,
        groups=1,
        stride=1,
        dilation=1,
        padding=0,
    ):
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
            .broadcast(bs, groups, cin, rcout, *oyx, *HW)
            .transpose(
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
            (
                x
                * weight.reshape(
                    1, groups, rcout, *[1 for _ in range(len(oyx))], cin, *HW
                )
            )
            .sum([-1 - i for i in range(1 + len(oyx))], keepdim=True)
            .reshape(bs, cout, *oyx)
        )
        return (
            ret
            if bias is None
            else ret.add(bias.reshape(1, -1, *[1 for _ in range(len(HW))]))
        )
