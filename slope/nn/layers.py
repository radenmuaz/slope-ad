import numpy as np

from slope.nn.init import glorot_normal, normal
from slope import ops


def Dense(out_dim, W_init=glorot_normal(), b_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer."""

    def init_fun(input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        W, b = W_init((input_shape[-1], out_dim)), b_init((out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        x = inputs
        x = x.dot(W)
        x = x + b.broadcast((1, *b.shape), (0,))
        # out = ops.mm(x, W)
        # out = out + ops.broadcast(b, out.shape, (0,))
        return x

    return init_fun, apply_fun


def Fn(fun, **fun_kwargs):
    """Layer that applies a scalar function act on its inputs."""
    init_fun = lambda input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
    return init_fun, apply_fun

def Identity():
    """Layer construction function for an identity layer."""
    init_fun = lambda input_shape: (input_shape, ())
    apply_fun = lambda params, inputs, **kwargs: inputs
    return init_fun, apply_fun

def serial(*layers):
    init_funs, apply_funs = zip(*layers)

    def init_fun(input_shape):
        params = []
        for init_fun in init_funs:
            input_shape, param = init_fun(input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, inputs, **kwargs):
        for fun, param in zip(apply_funs, params):
            inputs = fun(param, inputs, **kwargs)
        return inputs

    return init_fun, apply_fun


def parallel(*layers):
    init_funs, apply_funs = zip(*layers)

    def init_fun(input_shape):
        return zip(*[init(shape) for init, shape in zip(init_funs, input_shape)])

    def apply_fun(params, inputs, **kwargs):
        return [f(p, x, **kwargs) for f, p, x in zip(apply_funs, params, inputs)]

    return init_fun, apply_fun
