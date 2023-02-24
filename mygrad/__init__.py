from mygrad import utils
from mygrad import registry as reg
from mygrad import tracing as trc
from mygrad import primitives as pm
from mygrad import arrays as arr
from mygrad import forward as fwd
from mygrad import pytrees as pt
import numpy as np

# import batch
# import flow
# import jaxpr
# import jvp
# import linearize
# import pprint
# import pytree

# import vjp
# import xla_jit


# startup
reg.trace_stack += [trc.MainTrace(0, trc.EvalTrace, None)]

reg.impl_rules[pm.add_p] = lambda x, y: [np.add(x, y)]
reg.impl_rules[pm.mul_p] = lambda x, y: [np.multiply(x, y)]
reg.impl_rules[pm.neg_p] = lambda x: [np.negative(x)]
reg.impl_rules[pm.sin_p] = lambda x: [np.sin(x)]
reg.impl_rules[pm.cos_p] = lambda x: [np.cos(x)]
reg.impl_rules[pm.reduce_sum_p] = lambda x, *, axis: [np.sum(x, axis)]
reg.impl_rules[pm.greater_p] = lambda x, y: [np.greater(x, y)]
reg.impl_rules[pm.less_p] = lambda x, y: [np.less(x, y)]
reg.impl_rules[pm.transpose_p] = lambda x, *, perm: [np.transpose(x, perm)]


def broadcast_impl(x, *, shape, axes):
    for axis in sorted(axes):
        x = np.expand_dims(x, axis)
    return [np.broadcast_to(x, shape)]


reg.impl_rules[pm.broadcast_p] = broadcast_impl


def add_jvp(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x + y], [x_dot + y_dot]


reg.jvp_rules[pm.add_p] = add_jvp


def mul_jvp(primals, tangents):
    (x, y), (x_dot, y_dot) = primals, tangents
    return [x * y], [x_dot * y + x * y_dot]


reg.jvp_rules[pm.mul_p] = mul_jvp


def sin_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return [pm.sin(x)], [pm.cos(x) * x_dot]


reg.jvp_rules[pm.sin_p] = sin_jvp


def cos_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return [pm.cos(x)], [-pm.sin(x) * x_dot]


reg.jvp_rules[pm.cos_p] = cos_jvp


def neg_jvp(primals, tangents):
    (x,), (x_dot,) = primals, tangents
    return [pm.neg(x)], [pm.neg(x_dot)]


reg.jvp_rules[pm.neg_p] = neg_jvp


def reduce_sum_jvp(primals, tangents, *, axis):
    (x,), (x_dot,) = primals, tangents
    return [pm.reduce_sum(x, axis)], [pm.reduce_sum(x_dot, axis)]


reg.jvp_rules[pm.reduce_sum_p] = reduce_sum_jvp


def greater_jvp(primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = pm.greater(x, y)
    return [out_primal], [trc.zeros_like(out_primal)]


reg.jvp_rules[pm.greater_p] = greater_jvp


def less_jvp(primals, tangents):
    (x, y), _ = primals, tangents
    out_primal = pm.less(x, y)
    return [out_primal], [trc.zeros_like(out_primal)]


reg.jvp_rules[pm.less_p] = less_jvp

reg.node_types[tuple] = pt.NodeType(
    str(tuple), lambda x: (None, x, lambda _, xs: tuple(xs))
)
reg.node_types[list] = pt.NodeType(
    str(list), lambda x: (None, x), lambda _, xs: list(xs)
)
reg.node_types[dict] = pt.NodeType(
    str(dict),
    lambda d: map(tuple, utils.unzip2(sorted(d.items()))),
    lambda keys, vals: dict(zip(keys, vals)),
)
