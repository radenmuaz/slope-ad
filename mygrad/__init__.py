from mygrad import utils

# from mygrad import registry as reg
from mygrad import runtime as rt
from mygrad import llops as llops
from mygrad import arrays as arr

# from mygrad import forward as fwd
from mygrad import pytrees as pt
import numpy as np

RT = rt.Runtime()
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

# reg.impl_rules[llops.add_p] = lambda x, y: [np.add(x, y)]
# reg.impl_rules[llops.mul_p] = lambda x, y: [np.multiply(x, y)]
# reg.impl_rules[llops.neg_p] = lambda x: [np.negative(x)]
# reg.impl_rules[llops.sin_p] = lambda x: [np.sin(x)]
# reg.impl_rules[llops.cos_p] = lambda x: [np.cos(x)]
# reg.impl_rules[llops.reduce_sum_p] = lambda x, *, axis: [np.sum(x, axis)]
# reg.impl_rules[llops.greater_p] = lambda x, y: [np.greater(x, y)]
# reg.impl_rules[llops.less_p] = lambda x, y: [np.less(x, y)]
# reg.impl_rules[llops.transpose_p] = lambda x, *, perm: [np.transpose(x, perm)]


# def broadcast_impl(x, *, shape, axes):
#     for axis in sorted(axes):
#         x = np.expand_dims(x, axis)
#     return [np.broadcast_to(x, shape)]

# reg.impl_rules[llops.broadcast_p] = broadcast_impl


# def add_jvp(primals, tangents):
#     (x, y), (x_dot, y_dot) = primals, tangents
#     return [x + y], [x_dot + y_dot]


# reg.jvp_rules[llops.add_p] = add_jvp


# def mul_jvp(primals, tangents):
#     (x, y), (x_dot, y_dot) = primals, tangents
#     return [x * y], [x_dot * y + x * y_dot]


# reg.jvp_rules[llops.mul_p] = mul_jvp


# def sin_jvp(primals, tangents):
#     (x,), (x_dot,) = primals, tangents
#     return [llops.sin(x)], [llops.cos(x) * x_dot]


# reg.jvp_rules[llops.sin_p] = sin_jvp


# def cos_jvp(primals, tangents):
#     (x,), (x_dot,) = primals, tangents
#     return [llops.cos(x)], [-llops.sin(x) * x_dot]


# reg.jvp_rules[llops.cos_p] = cos_jvp


# def neg_jvp(primals, tangents):
#     (x,), (x_dot,) = primals, tangents
#     return [llops.neg(x)], [llops.neg(x_dot)]


# reg.jvp_rules[llops.neg_p] = neg_jvp


# def reduce_sum_jvp(primals, tangents, *, axis):
#     (x,), (x_dot,) = primals, tangents
#     return [llops.reduce_sum(x, axis)], [llops.reduce_sum(x_dot, axis)]


# reg.jvp_rules[llops.reduce_sum_p] = reduce_sum_jvp


# def greater_jvp(primals, tangents):
#     (x, y), _ = primals, tangents
#     out_primal = llops.greater(x, y)
#     return [out_primal], [trc.zeros_like(out_primal)]


# reg.jvp_rules[llops.greater_p] = greater_jvp


# def less_jvp(primals, tangents):
#     (x, y), _ = primals, tangents
#     out_primal = llops.less(x, y)
#     return [out_primal], [trc.zeros_like(out_primal)]


# reg.jvp_rules[llops.less_p] = less_jvp
