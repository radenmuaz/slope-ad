import slope
from slope.core import Tensor, Typecheckor, PyTreeDef
from typing import Tuple, List, Optional

import operator as operator_py

from typing import Sequence, Callable, Union, Callable, NamedTuple
import math
import numpy as np

# ====================
# Module
# ====================


class Module:
    def __hash__(self):
        self_flat, tree = slope.tree_flatten(self)
        # TODO: also use tree to compute hash
        return hash(self_flat)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return hash(self) == hash(other)

    def get_metadata(self):
        tensor_attrs = dict()  # dict as ordered set
        module_attrs = dict()

        for k, v in self.__dict__.items():
            if isinstance(v, (Tensor, Typecheckor)):
                tensor_attrs[k] = None
            elif isinstance(v, (list, tuple)) and not isinstance(v, PyTreeDef):
                v_flat, v_treedef = slope.tree_flatten(v)
                if all(isinstance(vi, (Tensor, Typecheckor)) for vi in v_flat):
                    tensor_attrs[k] = None
            elif isinstance(v, Module):
                module_attrs[k] = None

        static_dict = {k: v for k, v in self.__dict__.items() if k not in tuple(tensor_attrs) + tuple(module_attrs)}
        return dict(
            cls=self.__class__,
            tensor_attrs=tuple(tensor_attrs.keys()),
            module_attrs=tuple(module_attrs.keys()),
            static_dict=static_dict,
        )

    def get_attrs(self, attr_types, with_name=False):
        attrs = dict()
        for k, v in self.__dict__.items():
            if isinstance(v, attr_types):
                attrs[k] = v
        return attrs if with_name else tuple(attrs.values())

    def get_tensors(self, with_name=False):
        return self.get_attrs((Tensor, Typecheckor), with_name)

    def get_modules(self, with_name=False):
        return self.get_attrs(Module, with_name)

    def flatten(self):
        metadata = self.get_metadata()
        tensors = tuple(getattr(self, attr) for attr in metadata["tensor_attrs"])
        modules = tuple(getattr(self, attr) for attr in metadata["module_attrs"])
        return metadata, (tensors, modules)

    @staticmethod
    def unflatten(metadata, tensors_modules):
        mod = metadata["cls"].__new__(metadata["cls"])
        mod.__dict__.update(metadata["static_dict"])
        tensors, modules = tensors_modules
        for k, v in zip(metadata["tensor_attrs"], tensors):
            setattr(mod, k, v)
        for k, v in zip(metadata["module_attrs"], modules):
            setattr(mod, k, v)
        return mod

    def leaf_get_metadata(self):
        tensor_attrs = set()
        module_attrs = set()

        def find(obj, prefix):
            nonlocal tensor_attrs, module_attrs
            if isinstance(obj, (Tensor, Typecheckor)):
                tensor_attrs.add(prefix.strip("."))
                return
            if isinstance(obj, Module):
                if obj is not self:
                    module_attrs.add(prefix.strip("."))
                for k, v in obj.__dict__.items():
                    find(v, f"{prefix}{str(k)}.")

        find(self, "")
        static_dict = {k: v for k, v in self.__dict__.items() if k not in tuple(tensor_attrs) + tuple(module_attrs)}
        return dict(
            cls=self.__class__,
            tensor_attrs=tuple(tensor_attrs),
            module_attrs=tuple(module_attrs),
            static_dict=static_dict,
        )

    def leaf_flatten(self):
        metadata = self.get_metadata()
        tensors = tuple(operator_py.attrgetter(attr)(self) for attr in metadata["tensor_attrs"])
        rest = dict()
        for mod_attr in metadata["module_attrs"]:
            mod = operator_py.attrgetter(mod_attr)(self)
            mod_rest, _ = mod.flatten()
            rest[mod_attr] = mod_rest

        return (metadata, rest), tensors

    @staticmethod
    def leaf_unflatten(metadata_rest, tensors):
        def reassamble(metadata, rest):
            cls = metadata["cls"]
            mod = cls.__new__(cls)
            mod.__dict__.update(metadata["static_dict"])
            for mod_attr, (metadata_, rest_) in rest.items():
                setattr(mod, mod_attr, reassamble(metadata_, rest_))
            return mod

        metadata, rest = metadata_rest
        mod = reassamble(metadata, rest)

        def set_nested_attr(obj, attr, value):
            nested_attrs = attr.split(".")
            target_obj = obj
            for a in nested_attrs[:-1]:
                target_obj = getattr(target_obj, a)
            setattr(target_obj, nested_attrs[-1], value)

        for tensor, tensor_attr in tuple(zip(tuple(tensors), metadata["tensor_attrs"])):
            set_nested_attr(mod, tensor_attr, tensor)
        return mod


slope.M().register_node(Module, Module.flatten, Module.unflatten, "Module")
# slope.M().register_node(Module, Module.leaf_flatten, Module.leaf_unflatten, "Module")


# ====================
# Init
# ====================


def compute_fans(shape: Sequence, in_dim=-2, out_dim=-1, batch_dim=()):
    if isinstance(in_dim, int):
        in_size = shape[in_dim]
    else:
        in_size = int(np.prod([shape[i] for i in in_dim]))
    if isinstance(out_dim, int):
        out_size = shape[out_dim]
    else:
        out_size = int(np.prod([shape[i] for i in out_dim]))
    if isinstance(batch_dim, int):
        batch_size = shape[batch_dim]
    else:
        batch_size = int(np.prod([shape[i] for i in batch_dim]))
    receptive_field_size = math.prod(shape) / in_size / out_size / batch_size
    fan_in = in_size * receptive_field_size
    fan_out = out_size * receptive_field_size
    return fan_in, fan_out


def normal(dtype=slope.float32) -> Callable:
    def init(shape, dtype=dtype):
        return slope.randn(shape)

    return init


def variance_scaling(
    scale,
    mode: str,
    distribution: str,
    in_dim: Union[int, Sequence[int]] = 1,
    out_dim: Union[int, Sequence[int]] = 0,
    batch_dim: Sequence[int] = (),
    dtype=slope.float32,
) -> Callable:
    def init(shape, dtype=dtype):
        fan_in, fan_out = compute_fans(shape, in_dim, out_dim, batch_dim)
        if mode == "fan_in":
            denominator = fan_in
        elif mode == "fan_out":
            denominator = fan_out
        elif mode == "fan_avg":
            denominator = (fan_in + fan_out) / 2
        else:
            raise ValueError(f"invalid mode for variance scaling initializer: {mode}")
        variance = slope.tensor(scale / denominator, dtype=dtype)
        if distribution == "normal":
            return slope.randn(shape) * variance.sqrt()
        elif distribution == "uniform":
            return slope.rand(size=shape.astype(dtype)) * (3 * variance).sqrt()

        else:
            raise ValueError(f"invalid distribution for variance scaling initializer: {distribution}")

    return init


def glorot_normal(
    in_dim: Union[int, Sequence[int]] = 1,
    out_dim: Union[int, Sequence[int]] = 0,
    batch_dim: Sequence[int] = (),
    dtype=slope.float32,
) -> Callable:
    return variance_scaling(
        1.0,
        "fan_avg",
        "normal",
        in_dim=in_dim,
        out_dim=out_dim,
        batch_dim=batch_dim,
        dtype=dtype,
    )


def glorot_uniform(
    in_dim: Union[int, Sequence[int]] = 1,
    out_dim: Union[int, Sequence[int]] = 0,
    batch_dim: Sequence[int] = (),
    dtype=slope.float32,
) -> Callable:
    return variance_scaling(
        1.0,
        "fan_avg",
        "uniform",
        in_dim=in_dim,
        out_dim=out_dim,
        batch_dim=batch_dim,
        dtype=dtype,
    )


# ====================
# Layers
# ====================


class Fn(Module):
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, x):
        return self.fn(x)


class ReLU(Module):
    def __call__(self, x):
        return x.relu()

    @slope.M().backend.procedure_set.register(inline=True)
    @staticmethod
    def relu(x):
        return x.maximum(slope.zeros_like(x))


class LeakyReLU(Module):
    def __call__(self, x):
        return x.relu()

    @slope.M().backend.procedure_set.register(inline=True)
    @staticmethod
    def leakyrelu(x, neg_slope=0.01):
        return x.relu() - (slope.full_like(x, -neg_slope) * x).relu()


class Sigmoid(Module):
    def __call__(self, x):
        return x.sigmoid()

    @slope.M().backend.procedure_set.register(inline=True)
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + (-x).exp())


class Tanh(Module):
    def __call__(self, x):
        return x.tanh()

    @slope.M().backend.procedure_set.register(inline=True)
    @staticmethod
    def tanh(x):
        return 2.0 * ((2.0 * x).sigmoid()) - 1.0


class Swish(Module):
    def __call__(self, x):
        return x.swish()

    @slope.M().backend.procedure_set.register(inline=True)
    @staticmethod
    def swish(x):
        return x * x.sigmoid()


SiLU = Swish


class GELU(Module):
    def __call__(self, x):
        return x.gelu()

    @slope.M().backend.procedure_set.register(inline=True)
    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + (x * 0.7978845608 * (1 + 0.044715 * x * x)).tanh())


class Embedding(Module):
    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_size = vocab_size
        self.weight = Tensor.glorot_uniform(vocab_size, embed_size)

    def __call__(self, idx: Tensor) -> Tensor:
        if not hasattr(self, "vocab_counter"):
            self.vocab_counter = Tensor.arange(self.vocab_size, requires_grad=False).reshape(1, 1, self.vocab_size)
        return (self.vocab_counter == idx.unsqueeze(2)).expand(*idx.shape, self.vocab_size) @ self.weight


class Linear(Module):
    def __init__(self, in_features, out_features, bias=False, W_init=glorot_normal()):
        self.weight = W_init((out_features, in_features))
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x):
        return self.linear(x, self.weight, self.bias)

    @slope.M().backend.procedure_set.register(inline=True)
    @staticmethod
    def linear(x, w, b=None):
        x = x @ w.transpose(-2, -1)
        return x + b[None, ...] if b is not None else x


class Sequential(Module):
    def __init__(self, *modules):
        if isinstance(modules[0], (tuple, list)):
            assert len(modules) == 1
            modules = modules[0]
        self.num_modules = len(modules)
        for i in range(len(modules)):
            setattr(self, f"m{i}", modules[i])

    def __call__(self, x, *args, **kwargs):
        modules = [getattr(self, f"m{i}") for i in range(self.num_modules)]
        return x.serial(modules, *args, **kwargs)

    @slope.M().backend.procedure_set.register(inline=True)
    @staticmethod
    def serial(x, modules, *args, **kwargs):
        for module in modules:
            x = module(x, *args, **kwargs)
        return x


class Pool(Module):
    def __call__(x, kernel_size, stride=1, dilation=1):
        return x.pool(kernel_size, stride, dilation)

    @slope.M().backend.procedure_set.register(static_argnames="k_ stride dilation")
    @staticmethod
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


class AvgPool2d(Module):
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x):
        return x.avgpool2d(self.kernel_size, self.stride)

    @slope.M().backend.procedure_set.register(static_argnames="kernel_size stride")
    @staticmethod
    def avgpool2d(x, kernel_size=(2, 2), stride=None):
        def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
            return (x,) * cnt if isinstance(x, int) else x

        return x.pool(make_pair(kernel_size), stride if stride is not None else kernel_size).mean(
            dim=tuple(range(0 - len(make_pair(kernel_size)), 0))
        )


class MaxPool2d(Module):
    def __call__(x, kernel_size, stride=None):
        return x.maxpool2d(kernel_size, stride)

    @slope.M().backend.procedure_set.register(static_argnames="kernel_size stride dilation")
    @staticmethod
    def maxpool2d(x, kernel_size=(2, 2), stride=None, dilation=1):
        def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
            return (x,) * cnt if isinstance(x, int) else x

        return x.pool(make_pair(kernel_size), stride if stride is not None else kernel_size, dilation).max(
            dim=tuple(range(0 - len(make_pair(kernel_size)), 0))
        )


class ConvNd(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        W_init=glorot_normal(),
        dims=2,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dims = dims
        self.weight = W_init((out_channels, in_channels, *(kernel_size,) * dims))
        self.bias = slope.zeros(out_channels) if bias else None
        # m.weight.data.normal_(0, math.sqrt(2.0 / kernel_size * kernel_size * out_channels))

    def __call__(self, x):
        return x.conv(self.weight, groups=self.groups, stride=self.stride, dilation=self.dilation, padding=self.padding)

    @slope.M().backend.procedure_set.register(static_argnames="groups stride dilation padding")
    @staticmethod
    def conv(x, w, groups=1, stride=1, dilation=1, padding=0):
        (bs, cin_), (cout, cin), HW = x.shape[:2], w.shape[:2], w.shape[2:]
        assert groups * cin == cin_ and len(x.shape) == len(
            w.shape
        ), f"Input dim shape {x.shape} does not match the shape of the ws {w.shape}. ({groups*cin} vs. {cin_})"
        if isinstance(padding, (tuple, list)):
            assert len(padding) == 2 * len(HW) or len(padding) == len(
                HW
            ), f"Expected padding of length {2*len(HW)} or {len(HW)}, but got {len(padding)} for tensor of shape {x.shape}"
        padding_ = (
            [padding] * 2 * len(HW)
            if isinstance(padding, int)
            else (padding if len(padding) == 2 * len(HW) else [p for p in padding for _ in range(2)][::-1])
        )
        padding_ = tuple(padding_)

        def pad2d(x, padding: Union[List[int], Tuple[int, ...]], value: float = 0):
            # (padding_left, padding_right, padding_top, padding_bottom)
            slc = [(-p0, s + p1) for p0, p1, s in zip(padding[::2], padding[1::2], x.shape[::-1])][::-1]
            return x.padslice([(0, s) for s in x.shape[: -(len(padding) // 2)]] + slc, value=value)

        x = pad2d(x, padding_)
        x = x.pool(HW, stride, dilation)  # (bs, groups*cin, oy, ox, H, W)
        rcout, oyx = cout // groups, x.shape[2 : -len(HW)]
        x = x.reshape((bs, groups, cin, 1, *oyx, *HW))
        x = x.expand((bs, groups, cin, rcout, *oyx, *HW))
        x = x.permute(
            (
                0,
                1,
                3,
                *[4 + i for i in range(len(oyx))],
                2,
                *[4 + len(oyx) + i for i in range(len(HW))],
            )
        )
        # (bs, groups, rcout, *oyx, cin, *HW)
        x = x * w.reshape((1, groups, rcout, *[1] * len(oyx), cin, *HW))
        x = x.sum([-1 - i for i in range(1 + len(oyx))], keepdim=True)
        x = x.reshape((bs, cout, *oyx))
        ret = x
        return ret


class Conv1d(ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        W_init=glorot_normal(),
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, W_init, dims=1
        )


class Conv2d(ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        W_init=glorot_normal(),
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, W_init, dims=2
        )


class ConvNdTranspose(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        dims=2,
        output_padding=0,
        W_init=glorot_normal(),
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.dims = dims
        if output_padding != 0:
            raise NotImplementedError
        self.output_padding = output_padding
        self.weight = W_init((out_channels, in_channels, *(kernel_size,) * dims))
        self.bias = Tensor.zeros(out_channels) if bias else None

    def __call__(self, x):
        return x.conv_transpose(
            self.weight,
            groups=self.groups,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
            output_padding=self.output_padding,
        )

    @slope.M().backend.procedure_set.register(static_argnames="groups stride dilation padding output_padding")
    def conv_transpose(x, w, groups=1, stride=1, dilation=1, padding=0, output_padding=0):
        def make_pair(x: Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]:
            return (x,) * cnt if isinstance(x, int) else x

        def flatten_seq(l):
            return [item for sublist in l for item in sublist]

        HW, trailing = w.shape[2:], list(range(3, len(w.shape) + 1))
        w = w.reshape(((groups, w.shape[0] // groups, w.shape[1], *w.shape[2:])))
        w = w.permute((0, 2, 1, *trailing)).flip(trailing)
        stride = make_pair(stride, len(HW))
        if any(s > 1 for s in stride):
            x = x.reshape((*x.shape[:2], *flatten_seq((k, 1) for k in x.shape[2:])))
            x = x.pad(((0, 0), (0, 0), *flatten_seq(((0, 0), (0, s - 1)) for s in stride)))
            x = x.reshape((*x.shape[:2], *[k * s for k, s in zip(x.shape[2::2], stride)]))
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
        w = w.reshape((w.shape[0] * w.shape[1], *w.shape[2:]))
        return x.conv(w, groups=groups, dilation=dilation, padding=padding)


# ==============
# Regularization
# =============


class BatchNorm(Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
        self.eps = eps
        self.track_running_stats = track_running_stats
        self.momentum = momentum

        self.weight = slope.ones(num_features) if affine else None
        self.bias = slope.zeros(num_features) if affine else None

        self.running_mean = slope.zeros(num_features)
        self.running_var = slope.ones(num_features)
        self.num_batches_tracked = slope.zeros(1)

    def __call__(self, x, training=False):
        if training:
            broadcast_shape = (1, -1) + (1,) * len(x.shape[2:])
            reduce_dim = (0,) + tuple(2 + i for i in range(len(x.shape[2:])))
            mean = x.stop_gradient().mean(reduce_dim)
            z = x.stop_gradient() - mean.reshape(broadcast_shape)
            var = (z * z).mean(reduce_dim)
            invstd = 1.0 / (var + self.eps).sqrt()
            if self.track_running_stats:
                z_numel = math.prod(z.shape)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * z_numel / (
                    z_numel - z.shape[1]
                ) * var
                self.num_batches_tracked = self.num_batches_tracked + 1
        else:
            mean = self.running_mean
            invstd = (self.running_var.reshape((1, -1, 1, 1)).expand(x.shape) + self.eps).rsqrt()
        return x.batchnorm(self.weight, self.bias, mean, invstd)

    @slope.M().backend.procedure_set.register(inline=True)
    @staticmethod
    def batchnorm(x, weight, bias, mean, invstd):
        broadcast_shape = (1, -1) + (1,) * len(x.shape[2:])
        x = x - mean.reshape(broadcast_shape)
        if weight is not None:
            x = x * weight.reshape(broadcast_shape)
        ret = x * invstd.reshape(broadcast_shape) if len(invstd.shape) == 1 else invstd
        return (ret + bias.reshape(broadcast_shape)) if bias is not None else ret


BatchNorm1d = BatchNorm2d = BatchNorm


class LayerNorm(Module):
    def __init__(
        self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5, elementwise_affine: bool = True
    ):
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.dim, self.eps, self.elementwise_affine = (
            tuple(-1 - i for i in range(len(self.normalized_shape))),
            eps,
            elementwise_affine,
        )
        self.weight, self.bias = (
            (Tensor.ones(*self.normalized_shape), Tensor.zeros(*self.normalized_shape))
            if elementwise_affine
            else (None, None)
        )

    def __call__(self, x: Tensor):
        assert (
            self.normalized_shape == x.shape[-len(self.normalized_shape) :]
        ), f"last dimensions of {x.shape} must match {self.normalized_shape}"
        x = x.layernorm(eps=self.eps, dim=self.dim)
        if not self.elementwise_affine:
            return x
        return x * self.weight + self.bias

    @slope.M().backend.procedure_set.register(static_argnames="dim eps")
    @staticmethod
    def layernorm(self, dim=-1, eps: float = 1e-5) -> Tensor:
        y = self - self.mean(dim, keepdim=True)
        return y.mul((y * y).mean(dim, keepdim=True).add(eps).rsqrt())


class LayerNorm1d(LayerNorm):
    def __call__(self, x):
        return super().__call__(x.transpose(1, 2)).transpose(1, 2)


class LayerNorm2d(LayerNorm):
    def __call__(self, x):
        return super().__call__(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Dropout(Module):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, x, training=False):
        return x.dropout(self.p, training)

    @slope.M().backend.procedure_set.register(static_argnames="p")
    def dropout(x, p, training=False) -> Tensor:
        if not Tensor.training or p == 0:
            return x
        mask = (Tensor.rand(*x.shape, requires_grad=False, device=x.device) >= p).cast(slope.bool)
        return x * mask * (1 / (1.0 - p))


class ScaledDotProductAttention(Module):
    def __call__(
        x,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ):
        return x.scaled_dot_product_attention(key, value, attn_mask, dropout_p, is_causal)

    @slope.M().backend.procedure_set.register(static_argnames="dropout_p, is_causal")
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
                Tensor.ones(x.shape[-2], key.shape[-2], requires_grad=False, device=x.device).tril(0).cast(slope.bool)
            )
        if attn_mask is not None and attn_mask.dtype == slope.bool:
            attn_mask = (attn_mask == 0).where(-float("inf"), attn_mask)
        return (x @ key.transpose(-2, -1) / math.sqrt(x.shape[-1]) + attn_mask).softmax(-1).dropout(dropout_p) @ value


class BCELoss(Module):
    def __call__(x, y):
        return x.binary_cross_entropy(y)

    @slope.M().backend.procedure_set.register()
    @staticmethod
    def binary_cross_entropy(x, y: Tensor) -> Tensor:
        return (-y * x.log() - (1 - y) * (1 - x).log()).mean()


class BCEWithLogitsLoss(Module):
    def __call__(x, y):
        return x.binary_cross_entropy_with_logits(y)

    @slope.M().backend.procedure_set.register()
    @staticmethod
    def binary_cross_entropy_with_logits(x, y: Tensor) -> Tensor:
        return (x.maximum(0) - y * x + (1 + x.abs().__neg__().exp()).log()).mean()


class CrossEntropyLoss(Module):
    def __call__(x, y):
        return x.cross_entropy(y)

    @slope.M().backend.procedure_set.register(static_argnames="ignore_index")
    @staticmethod
    def cross_entropy(x, y, ignore_index=-1) -> Tensor:
        # NOTE: self is a logits input
        loss_mask = y != ignore_index
        y_counter = (
            slope.arange(x.shape[-1], dtype=slope.int32, device=x.device).unsqueeze(0).expand(y.numel(), x.shape[-1])
        )
        y = ((y_counter == y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(
            *y.shape, x.shape[-1]
        )
        return (x.log_softmax(-1) * y).sum() / loss_mask.sum()


class Softmax(Module):
    def __call__(x, dim=-1):
        return x.softmax(dim)

    @slope.M().backend.procedure_set.register(static_argnames="dim")
    @staticmethod
    def softmax(x, dim=-1):
        m = x - x.max(dim, keepdim=True)
        e = m.exp()
        ss = e.sum(dim, keepdim=True)
        return e / ss


class LogSoftmax(Module):
    def __call__(x, dim=-1):
        return x.log_softmax(dim)

    @slope.M().backend.procedure_set.register(static_argnames="dim")
    @staticmethod
    def log_softmax(x, dim=-1):
        m = x - x.max(dim, keepdim=True)
        e = m.exp()
        ss = e.sum(dim, keepdim=True)
        return m - ss.log()


# ====================
# Optimizers
# ====================


class Optimizer(Module):
    def __init__(self, params, lr: float):
        self.params = params
        self.params_treedef = slope.tree_flatten(params)[1]
        self.state = Module()
        self.hp = Module()
        self.hp.lr = slope.full((), lr)
        self.iters = slope.zeros(())
        # self.iters = slope.ones(())

    def step(self, p, g, *state_attrs):
        return p, state_attrs

    def __call__(self, params, g_params):
        state_names, state_attrs = zip(*self.state.get_modules(with_name=True).items())
        step_out, (leaf0, leaf0_treedef) = slope.tree_map(self.step, params, *(g_params, *state_attrs), out_leaf=True)
        step_out_T = slope.tree_transpose(self.params_treedef, leaf0_treedef, step_out)
        params_out, state = step_out_T
        self.state = state
        self.iters = self.iters + 1
        return (params_out, self)


class GD(Optimizer):
    def __init__(self, params, lr=0.001):
        super().__init__(params, lr)

    def step(self, p, g, *state_attrs):
        lr = self.hp.lr
        p = p - lr * g
        return p, state_attrs


class SGD(Optimizer):
    def __init__(self, params, lr=0.001, momentum: float = 0.9, weight_decay=0.0, nesterov=False):
        super().__init__(params, lr)
        self.hp.momentum = momentum
        self.hp.weight_decay = weight_decay
        self.hp.nesterov = nesterov
        self.state.b = slope.tree_map(lambda x: x.zeros_like(), self.params)

    def step(self, p, g, b):
        lr, m, wd = self.hp.lr, self.hp.momentum, self.hp.weight_decay
        g = g + wd * p
        b = m * b + g
        g = (g + m * b) if self.hp.nesterov else b
        p = p - lr * g
        state = Module()
        state.b = b
        return (p, state)


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-5, weight_decay=0.0):
        super().__init__(params, lr)
        self.hp.b1 = b1
        self.hp.b2 = b2
        self.hp.eps = eps
        self.hp.wd = weight_decay
        self.state.m = slope.tree_map(lambda x: x.ones_like(), self.params)
        self.state.v = slope.tree_map(lambda x: x.ones_like(), self.params)

    def step(self, p, g, m, v):
        lr, wd = self.hp.lr, self.hp.wd
        b1, b2, eps = self.hp.b1, self.hp.b2, self.hp.eps
        i = self.iters + slope.ones_like(self.iters)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * (g * g)
        m_hat = m / (1.0 - b1**i)
        v_hat = v / (1.0 - b2**i)
        up = m_hat / ((v_hat).sqrt() + eps)
        if wd > 0:
            up = up + wd * p.stop_gradient()
        p = p - lr * up
        state = Module()
        state.m = m
        state.v = v
        return (p, state)


AdamW = Adam
