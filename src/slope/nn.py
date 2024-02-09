import slope
from slope.core import Tensor, SymbolicTensor, TreeDef
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
        self_flat, treedef = slope.tree_flatten(self)
        return hash(self_flat)

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        return hash(self) == hash(other)

    def get_metadata(self):
        tensor_attrs = dict()  # dict as ordered set
        module_attrs = dict()

        for k, v in self.__dict__.items():
            if isinstance(v, (Tensor, SymbolicTensor)):
                tensor_attrs[k] = None
            elif isinstance(v, (list, tuple)) and not isinstance(v, TreeDef):
                v_flat, v_treedef = slope.tree_flatten(v)
                if all(isinstance(vi, (Tensor, SymbolicTensor)) for vi in v_flat):
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
        return self.get_attrs((Tensor, SymbolicTensor), with_name)

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
            if isinstance(obj, (Tensor, SymbolicTensor)):
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

    def export(self, *args, **kwargs):
        f_jitobj = slope.jit(self.__call__).jit_program(*args, **kwargs)
        return


slope.core.backend.register_node(Module, Module.flatten, Module.unflatten, "Module")

# ====================
# Optimizers
# ====================


class Optimizer(Module):
    def __init__(self, params, lr: float):
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
        # g_params = slope.tree_map(lambda x: (x==slope.tensor([float('nan')])).where(0.0, x), g_params)
        step_out, (leaf0, leaf0_treedef) = slope.tree_map(self.step, params, *(g_params, *state_attrs), out_leaf=True)
        step_out_T = slope.tree_transpose(self.params_treedef, leaf0_treedef, step_out)
        params_out, state = step_out_T
        # params_out = slope.tree_map(lambda x: (x==slope.tensor([float('nan')])).where(0., x), params_out)
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
    def __init__(
        self,
        params,
        lr=0.001,
        momentum: float = 0,
        weight_decay=0.0,
        nesterov=False,
    ):
        super().__init__(params, lr)
        self.hp.momentum = momentum
        self.hp.weight_decay = weight_decay
        self.hp.nesterov = nesterov
        self.state.b = slope.tree_map(lambda x: x.zeros_like(), params)

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
        self.state.m = slope.tree_map(lambda x: x.ones_like(), params)
        self.state.v = slope.tree_map(lambda x: x.ones_like(), params)

    def step(self, p, g, m, v):
        lr, wd = self.hp.lr, self.hp.wd
        b1, b2, eps = self.hp.b1, self.hp.b2, self.hp.eps
        i = self.iters + 1  # slope.ones_like(self.iters)
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


def normal(dtype=slope.core.backend.DEFAULT_DTYPE) -> Callable:
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
    dtype=slope.core.backend.DEFAULT_DTYPE,
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
    dtype=slope.core.backend.DEFAULT_DTYPE,
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
    dtype=slope.core.backend.DEFAULT_DTYPE,
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


class LeakyReLU(Module):
    def __call__(self, x):
        return x.relu()


class Sigmoid(Module):
    def __call__(self, x):
        return x.sigmoid()


class Tanh(Module):
    def __call__(self, x):
        return x.tanh()


class Swish(Module):
    def __call__(self, x):
        return x.swish()


SiLU = Swish


class GELU(Module):
    def __call__(self, x):
        return x.gelu()


class Embedding(Module):
    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_size = vocab_size
        self.weight = Tensor.glorot_uniform(vocab_size, embed_size)

    def __call__(self, idx: Tensor) -> Tensor:
        if not hasattr(self, "vocab_counter"):
            self.vocab_counter = Tensor.arange(self.vocab_size, requires_grad=False).reshape(1, 1, self.vocab_size)
        return (self.vocab_counter == idx.unsqueeze(2)).expand(*idx.shape, self.vocab_size) @ self.weight


class Linear(Module):
    def __init__(self, in_features, out_features, bias=False):
        self.weight = slope.zeros(out_features, in_features)
        self.bias = slope.zeros(out_features) if bias else None
        self.reset_parameters()

    def __call__(self, x):
        return x.linear(self.weight, self.bias)
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight = slope.rand_like(self.weight)*2*stdv - stdv
        if self.bias is not None:
            self.bias = slope.rand_like(self.bias)*2*stdv - stdv



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
        return x.sequential(modules, *args, **kwargs)


class Pool(Module):
    def __call__(x, kernel_size, stride=1, dilation=1):
        return x.pool(kernel_size, stride, dilation)


class AvgPool2d(Module):
    def __init__(self, kernel_size=2, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride

    def __call__(self, x):
        return x.avgpool2d(self.kernel_size, self.stride)


class MaxPool2d(Module):
    def __call__(x, kernel_size, stride=None):
        return x.maxpool2d(kernel_size, stride)


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
        self.weight =  slope.zeros(out_channels, in_channels, *(kernel_size,) * dims)
        self.bias = slope.zeros(out_channels) if bias else None
        self.reset_parameters()

    def __call__(self, x):
        return x.conv(
            self.weight,
            groups=self.groups,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
        )
    
    def reset_parameters(self):
        # n = self.in_channels
        # for k in self.kernel_size:
        #     n *= k
        n = self.in_channels *  (self.kernel_size ** self.dims)
        stdv = 1. / math.sqrt(n)
        self.weight = slope.rand_like(self.weight)*2*stdv - stdv
        if self.bias is not None:
            self.bias = slope.rand_like(self.bias)*2*stdv - stdv
        


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
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            dims=1,
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
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            dims=2,
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


# ==============
# Regularization
# =============


class BatchNorm(Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, momentum=0.1):
        self.eps = eps
        self.momentum = momentum

        self.weight = slope.ones(num_features) if affine else None
        self.bias = slope.zeros(num_features) if affine else None

        self.running_mean = slope.zeros(num_features)
        self.running_var = slope.ones(num_features)
        self.num_batches_tracked = slope.zeros(1)

    def __call__(self, x, training=True, track_running_stats=True):
        # training, track_running_stats = True, False
        if training:
            D = len(x.shape[2:])
            broadcast_shape = (1, -1) + (1,) * D
            reduce_dim = (0,) + tuple(2 + i for i in range(D))
            xsg = x.stop_gradient()
            mean = xsg.mean(reduce_dim)
            z = xsg - mean.reshape(broadcast_shape)
            var = (z * z).mean(reduce_dim)
            invstd = (var + self.eps).rsqrt()
            if track_running_stats:
                z_numel = math.prod(z.shape)
                z_ratio = z_numel / (z_numel - z.shape[1]) if z_numel != z.shape[1] else 1
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * z_ratio * var
                self.num_batches_tracked = self.num_batches_tracked + 1
                # mean = self.running_mean
                # invstd = (self.running_var + self.eps).rsqrt()
        else:
            mean = self.running_mean
            invstd = (self.running_var + self.eps).rsqrt()

        return x.batchnorm(self.weight, self.bias, mean, invstd)


BatchNorm1d = BatchNorm2d = BatchNorm


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.dim, self.eps, self.elementwise_affine = (
            tuple(-1 - i for i in range(len(self.normalized_shape))),
            eps,
            elementwise_affine,
        )
        self.weight, self.bias = (
            (
                Tensor.ones(*self.normalized_shape),
                Tensor.zeros(*self.normalized_shape),
            )
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


class BCELoss(Module):
    def __call__(x, y):
        return x.binary_cross_entropy(y)


class BCEWithLogitsLoss(Module):
    def __call__(x, y):
        return x.binary_cross_entropy_with_logits(y)


class CrossEntropyLoss(Module):
    def __call__(x, y):
        return x.cross_entropy(y)


class Softmax(Module):
    def __call__(x, dim=-1):
        return x.softmax(dim)


class LogSoftmax(Module):
    def __call__(x, dim=-1):
        return x.log_softmax(dim)
